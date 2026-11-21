"""Conditioned fit-row capture and loss forwards for AxBench encdec baselines."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
import torch.nn as nn

import nanochat.flash_attention as _fa_mod
import steering.segmentation as segmentation
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
    build_segmented_feature_encoder_inputs,
)
from steering.eval.axbench_upstream import methods as methods_module
from steering.eval.axbench_upstream.encdec_baselines.capture import (
    FitRowConditioning,
    build_fit_row_conditioning,
    capture_post_prefix_activations,
    conditioned_forward_inputs,
    conditioned_loss_forward,
)
from steering.eval.axbench_upstream.methods import (
    PROMPT_CONTEXT_WINDOW,
    PROMPT_SEGMENT_SIZE,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SingleEntryContextProvider,
    SingleTargetSegment,
)
from steering.segmentation import token_to_char_offsets
from steering.sft.turn_spans import rendered_text_and_turn_char_starts
from tests.conftest import MockTokenizer, build_tiny_encdec
from tests.encdec_test_helpers import method_setup_double as _fixed_setup
from tests.steering.test_context_cache_in_memory_conversion import _ZeroWidthTokenizer

_LAYER = 1
_PREFIX_LENGTH = 2


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Use the repository's CPU-backward pattern for tiny encdec tests."""

    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _block(model: nn.Module, layer: int) -> nn.Module:
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    return blocks[layer]


def _fit_row() -> FitRowConditioning:
    return FitRowConditioning(
        token_ids=(10, 11, 12, 13, 14, 15),
        segments=(
            ContextFeatureSegment(
                feature_ids=(2, 5),
                buckets=(1, 2),
                char_start=0,
                char_end=2,
            ),
            ContextFeatureSegment(
                feature_ids=(7,),
                buckets=(3,),
                char_start=2,
                char_end=4,
            ),
            ContextFeatureSegment(
                feature_ids=(11, 13),
                buckets=(0, 1),
                char_start=4,
                char_end=6,
            ),
        ),
        segment_assignments=(0, 0, 1, 1, 2, 2),
    )


def _manual_capture(model: nn.Module, row: FitRowConditioning, *, layer: int) -> t.Tensor:
    """Independent oracle: explicit encoder bags and hand-derived char routing."""

    enc_feature_ids, enc_magnitudes, enc_mask = build_segmented_feature_encoder_inputs(
        [{2: 1, 5: 2}, {7: 3}, {11: 0, 13: 1}],
        "cpu",
    )
    # MockTokenizer decodes every non-special token to exactly one character.
    seg_map = t.tensor([[0, 0, 1, 1, 2, 2]], dtype=t.int64)
    idx = t.tensor([row.token_ids], dtype=t.int64)
    captured: list[t.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: t.Tensor) -> None:
        captured.append(output)

    handle = _block(model, layer).register_forward_hook(hook)
    try:
        with t.no_grad():
            model(
                idx,
                None,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
                encoder_strict=True,
            )
    finally:
        handle.remove()
    return captured[0][0, _PREFIX_LENGTH:, :].detach().to(t.float32)


def _manual_post_cross_attention_capture(
    model: nn.Module,
    row: FitRowConditioning,
    *,
    layer: int,
) -> t.Tensor:
    """Independent oracle for the explicit post-cross-attention capture site."""

    enc_feature_ids, enc_magnitudes, enc_mask = build_segmented_feature_encoder_inputs(
        [{2: 1, 5: 2}, {7: 3}, {11: 0, 13: 1}],
        "cpu",
    )
    seg_map = t.tensor([[0, 0, 1, 1, 2, 2]], dtype=t.int64)
    idx = t.tensor([row.token_ids], dtype=t.int64)
    captured: list[t.Tensor] = []

    def callback(layer_index: int, output: t.Tensor) -> t.Tensor:
        if layer_index == layer:
            captured.append(output)
        return output

    callback_model = cast(Any, model)
    assert callback_model._post_cross_attention_eval_callback is None
    callback_model._post_cross_attention_eval_callback = callback
    try:
        with t.no_grad():
            model(
                idx,
                None,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
                encoder_strict=True,
            )
    finally:
        callback_model._post_cross_attention_eval_callback = None
    assert len(captured) == 1
    return captured[0][0, _PREFIX_LENGTH:, :].detach().to(t.float32)


def test_capture_equals_independent_conditioned_forward_exactly() -> None:
    t.manual_seed(0)
    model = build_tiny_encdec()
    row = _fit_row()

    expected = _manual_capture(model, row, layer=_LAYER)
    actual = capture_post_prefix_activations(
        model,
        row,
        layer=_LAYER,
        prefix_length=_PREFIX_LENGTH,
    )

    assert t.equal(actual, expected)
    assert actual.dtype == t.float32
    assert actual.shape == (len(row.token_ids) - _PREFIX_LENGTH, 64)


def test_explicit_post_cross_attention_capture_matches_independent_site() -> None:
    t.manual_seed(0)
    model = build_tiny_encdec()
    row = _fit_row()

    expected = _manual_post_cross_attention_capture(model, row, layer=_LAYER)
    actual = capture_post_prefix_activations(
        model,
        row,
        layer=_LAYER,
        prefix_length=_PREFIX_LENGTH,
        capture_site="post_cross_attention",
    )
    historical_default = capture_post_prefix_activations(
        model,
        row,
        layer=_LAYER,
        prefix_length=_PREFIX_LENGTH,
    )

    assert t.equal(actual, expected)
    assert not t.equal(actual, historical_default)
    assert model._post_cross_attention_eval_callback is None


def test_capture_differential_detects_the_wrong_layer() -> None:
    t.manual_seed(0)
    model = build_tiny_encdec()
    row = _fit_row()

    actual = capture_post_prefix_activations(
        model,
        row,
        layer=_LAYER,
        prefix_length=_PREFIX_LENGTH,
    )
    wrong_layer = _manual_capture(model, row, layer=_LAYER - 1)

    assert not t.equal(actual, wrong_layer)
    # Production must HONOR its layer argument, not merely capture a layer
    # that happens to differ from a manual other-layer capture: asked for
    # layer 0, it must reproduce the manual layer-0 tensor exactly.
    other_actual = capture_post_prefix_activations(
        model,
        row,
        layer=_LAYER - 1,
        prefix_length=_PREFIX_LENGTH,
    )
    assert t.equal(other_actual, wrong_layer)


def test_conditioned_forward_seg_map_contract() -> None:
    row = _fit_row()

    _enc_ids, _enc_mags, _enc_mask, seg_map = conditioned_forward_inputs(row, "cpu")

    assert seg_map.shape == (1, len(row.token_ids))
    assert seg_map.dtype == t.int64
    assert bool(t.all(seg_map[:, 1:] >= seg_map[:, :-1]))
    assert set(seg_map[0].tolist()) == set(range(len(row.segments)))
    assert int(seg_map.min()) >= 0
    assert int(seg_map.max()) < len(row.segments)


def test_build_fit_row_conditioning_rejects_empty_token_tuple() -> None:
    with pytest.raises(
        ValueError,
        match=_exact("empty fit row: fit row: token tuple must be nonempty"),
    ):
        build_fit_row_conditioning(
            cast(Any, object()),
            object(),
            object(),
            object(),
            token_ids=(),
            surface="empty fit row",
        )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            FitRowConditioning(token_ids=(), segments=(), segment_assignments=()),
            "AxBench encdec-baseline fit row token tuple must be nonempty",
        ),
        (
            FitRowConditioning(token_ids=(10,), segments=(), segment_assignments=(0,)),
            "AxBench encdec-baseline fit row has no real feature segments",
        ),
        (
            FitRowConditioning(
                token_ids=(10, 11),
                segments=(ContextFeatureSegment((1,), (0,), 0, 2),),
                segment_assignments=(0,),
            ),
            "AxBench encdec-baseline fit row assignment count must match token count: 1 != 2",
        ),
        (
            FitRowConditioning(
                token_ids=(10,),
                segments=(ContextFeatureSegment((1, 2), (0,), 0, 1),),
                segment_assignments=(0,),
            ),
            (
                "AxBench encdec-baseline fit row segment feature/bucket length "
                "mismatch at segment 0: 2 != 1"
            ),
        ),
    ],
    ids=("empty-tokens", "empty-segments", "assignment-count", "feature-bucket-length"),
)
def test_conditioned_forward_inputs_rejects_invalid_hand_built_rows(
    row: FitRowConditioning,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=_exact(expected)):
        conditioned_forward_inputs(row, "cpu")


def test_conditioned_forward_inputs_rejects_out_of_range_assignment() -> None:
    row = FitRowConditioning(
        token_ids=(10,),
        segments=(ContextFeatureSegment((1,), (0,), 0, 1),),
        segment_assignments=(1,),
    )

    with pytest.raises(
        ValueError,
        match=_exact(
            "AxBench encdec-baseline fit row segment assignment is outside the "
            "real segment range [0, 1)"
        ),
    ):
        conditioned_forward_inputs(row, "cpu")


def test_capture_removes_hook_when_model_forward_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_encdec()
    block = _block(model, _LAYER)

    def fail_forward(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("intentional fit-row forward failure")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="intentional fit-row forward failure"):
        capture_post_prefix_activations(
            model,
            _fit_row(),
            layer=_LAYER,
            prefix_length=_PREFIX_LENGTH,
        )

    assert not block._forward_hooks


def test_capture_rejects_prefix_length_outside_token_sequence() -> None:
    with pytest.raises(
        ValueError,
        match=_exact(
            "AxBench encdec-baseline fit row prefix length must be within the "
            "token sequence: 7 not in [0, 6]"
        ),
    ):
        capture_post_prefix_activations(
            build_tiny_encdec(),
            _fit_row(),
            layer=_LAYER,
            prefix_length=7,
        )


def test_conditioned_loss_matches_wrapper_on_context_span() -> None:
    """Parity is exact before the wrapper's nonempty final target span.

    ``SingleTargetSegment(features={})`` appends one empty encoder slot and routes
    the final decoder position to it. The direct fit-row path deliberately keeps
    that position under its real context segment. Ignoring the final loss position
    therefore compares the exact shared CONTEXT-SPAN boundary without pretending
    the two target-slot semantics are identical.
    """

    t.manual_seed(0)
    model = build_tiny_encdec()
    tokenizer = MockTokenizer()
    row = _fit_row()
    idx = t.tensor([row.token_ids], dtype=t.int64)
    targets = t.tensor(
        [[*row.token_ids[1:], -1]],
        dtype=t.int64,
    )
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SingleTargetSegment(features={}),
        context_provider=SingleEntryContextProvider(row.token_ids, row.segments),
    )

    direct_loss = conditioned_loss_forward(
        model,
        row,
        targets=targets,
        loss_reduction="sum",
    )
    wrapper_loss = wrapper(idx, targets, loss_reduction="sum")

    assert t.equal(direct_loss, wrapper_loss)


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (
            t.zeros((1, 5), dtype=t.int64),
            ("AxBench encdec-baseline fit row targets must have shape (1, 6), got (1, 5)"),
        ),
        (
            t.zeros((1, 6), dtype=t.float32),
            ("AxBench encdec-baseline fit row targets must have dtype int64, got torch.float32"),
        ),
    ],
    ids=("shape", "dtype"),
)
def test_conditioned_loss_rejects_invalid_targets(
    targets: t.Tensor,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=_exact(expected)):
        conditioned_loss_forward(
            build_tiny_encdec(),
            _fit_row(),
            targets=targets,
            loss_reduction="sum",
        )


def test_conditioned_loss_rejects_non_tensor_model_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_encdec()
    monkeypatch.setattr(model, "forward", lambda *_args, **_kwargs: "not-a-tensor")

    with pytest.raises(
        TypeError,
        match=_exact("AxBench encdec-baseline fit row loss forward must return a tensor, got str"),
    ):
        conditioned_loss_forward(
            model,
            _fit_row(),
            targets=t.zeros((1, 6), dtype=t.int64),
            loss_reduction="sum",
        )


def test_alignment_extends_last_segment_and_emits_fit_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters: list[tuple[str, float, dict[str, Any]]] = []

    def record_counter(name: str, value: float = 1, **attrs: Any) -> None:
        counters.append((name, float(value), attrs))

    monkeypatch.setattr(segmentation, "record_counter", record_counter)
    tokenizer = MockTokenizer()
    token_ids = (10, 11, 12, 13, 14, 15)
    unaligned_segments = (
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=0, char_end=2),
        ContextFeatureSegment(feature_ids=(7,), buckets=(2,), char_start=2, char_end=4),
    )
    setup = _fixed_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="hook"),
        prompt_pipeline=object(),
        prompt_consolidation=None,
        n_sae_features=64,
    )

    class _GemmaTokenizer:
        bos_token_id = 1

        def decode(self, _ids: list[int]) -> str:
            return "<bos>"

    extraction_model = SimpleNamespace(
        tokenizer=_GemmaTokenizer(),
        cfg=SimpleNamespace(n_ctx=8192),
    )
    monkeypatch.setattr(
        methods_module,
        "truncate_to_training_window",
        lambda *_args: ([1, 2], 2),
    )
    monkeypatch.setattr(
        methods_module,
        "extract_document_features",
        lambda **_kwargs: ([t.tensor([2])], [t.tensor([1.0])], [0], [0], [4], 2),
    )
    monkeypatch.setattr(
        methods_module,
        "build_context_entry_from_extracted_segments",
        lambda **_kwargs: SimpleNamespace(
            token_ids=token_ids,
            segments=unaligned_segments,
            token_assignments=None,
        ),
    )

    row = build_fit_row_conditioning(
        setup,
        extraction_model,
        object(),
        tokenizer,
        token_ids=token_ids,
        surface="alignment fit row",
    )

    _enc_ids, _enc_mags, _enc_mask, seg_map = conditioned_forward_inputs(row, "cpu")

    assert row.segments[-1].char_end == 6
    assert seg_map.tolist() == [[0, 0, 1, 1, 1, 1]]
    matching = [
        event
        for event in counters
        if event[0] == "segmentation.char_end_extended.encdec_baseline_fit"
    ]
    assert matching == [
        (
            "segmentation.char_end_extended.encdec_baseline_fit",
            1.0,
            {"delta": 2, "direction": "forward"},
        )
    ]


class _TrainableCProjAdapter(nn.Module):
    """Minimal test adapter proving the loss forward preserves autograd."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.delta = nn.Parameter(t.zeros(()))

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return self.base(inputs) + self.delta * inputs


def test_capture_is_no_grad_but_conditioned_loss_preserves_adapter_grad() -> None:
    t.manual_seed(0)
    model = build_tiny_encdec()
    row = _fit_row()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block = _block(model, 0)
    original_c_proj = block.attn.c_proj
    adapter = _TrainableCProjAdapter(original_c_proj)
    block.attn.c_proj = adapter

    captured = capture_post_prefix_activations(
        model,
        row,
        layer=_LAYER,
        prefix_length=_PREFIX_LENGTH,
    )
    targets = t.tensor([[*row.token_ids[1:], -1]], dtype=t.int64)
    loss = conditioned_loss_forward(
        model,
        row,
        targets=targets,
        loss_reduction="mean",
    )
    loss.backward()

    assert captured.requires_grad is False
    assert loss.requires_grad is True
    assert adapter.delta.grad is not None


def test_build_fit_row_conditioning_mirrors_prompt_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = MockTokenizer()
    token_ids = (0, 10, 11, 12)
    expected_segments = (
        ContextFeatureSegment(feature_ids=(4,), buckets=(1,), char_start=0, char_end=3),
    )
    pipeline = object()
    consolidation = object()
    setup = _fixed_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
        prompt_pipeline=pipeline,
        prompt_consolidation=consolidation,
        n_sae_features=64,
    )

    class _GemmaTokenizer:
        bos_token_id = 1

        def decode(self, ids: list[int]) -> str:
            assert ids == [1]
            return "<bos>"

    extraction_model = SimpleNamespace(
        tokenizer=_GemmaTokenizer(),
        cfg=SimpleNamespace(n_ctx=8192),
    )
    sae = object()
    calls: dict[str, dict[str, Any]] = {}

    def fake_truncate(
        text: str,
        gemma_tokenizer: Any,
        context_window: int,
        sae_max_context: int,
    ) -> tuple[list[int], int]:
        assert text == tokenizer.decode(list(token_ids[1:]))
        assert gemma_tokenizer is extraction_model.tokenizer
        assert context_window == PROMPT_CONTEXT_WINDOW
        assert sae_max_context == 8192
        return [1, 2, 3], 3

    def fake_extract(**kwargs: Any) -> tuple[Any, ...]:
        calls["extract"] = kwargs
        return [t.tensor([4])], [t.tensor([0.75])], [0], [0], [3], 3

    def fake_build(**kwargs: Any) -> Any:
        calls["build"] = kwargs
        return SimpleNamespace(token_ids=token_ids, segments=expected_segments, token_assignments=None)

    monkeypatch.setattr(methods_module, "truncate_to_training_window", fake_truncate)
    monkeypatch.setattr(methods_module, "extract_document_features", fake_extract)
    monkeypatch.setattr(
        methods_module,
        "build_context_entry_from_extracted_segments",
        fake_build,
    )

    result = build_fit_row_conditioning(
        setup,
        extraction_model,
        sae,
        tokenizer,
        token_ids=token_ids,
        surface="concept 2 positive row 1",
    )

    assert result == FitRowConditioning(
        token_ids=token_ids,
        segments=expected_segments,
        segment_assignments=(0, 0, 0, 0),
    )
    assert calls["extract"] == {
        "model": extraction_model,
        "sae": sae,
        "text": tokenizer.decode(list(token_ids[1:])),
        "layer": 22,
        "context_window": PROMPT_CONTEXT_WINDOW,
        "pooling": methods_module.PoolingMode.max_pool,
        "segment_mode": methods_module.SegmentMode.fixed_tokens,
        "segment_size": PROMPT_SEGMENT_SIZE,
        "hook_name": "blocks.22.hook_resid_post",
        "context_independent_features": True,
        "ignore_bos_features": True,
        "turn_char_starts": None,
    }
    build_call = calls["build"]
    assert build_call["token_ids"] == token_ids
    assert build_call["pipeline"] is pipeline
    assert build_call["feature_clusters"] is consolidation
    assert int(build_call["n_sae_features"]) == 64
    assert int(build_call["num_buckets"]) == 2
    assert int(build_call["source_bos_char_len"]) == len("<bos>")
    assert build_call["source"] == "concept 2 positive row 1: fit row"


def test_build_fit_row_conditioning_rejects_gemma_window_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = MockTokenizer()
    setup = _fixed_setup()
    extraction_model = SimpleNamespace(
        tokenizer=object(),
        cfg=SimpleNamespace(n_ctx=8192),
    )
    monkeypatch.setattr(
        methods_module,
        "truncate_to_training_window",
        lambda *_args: ([], PROMPT_CONTEXT_WINDOW + 1),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"concept 3 negative row 4: fit row has 2049 Gemma tokens, "
            r"exceeding the extraction window 2048"
        ),
    ):
        build_fit_row_conditioning(
            setup,
            extraction_model,
            object(),
            tokenizer,
            token_ids=(10, 11),
            surface="concept 3 negative row 4",
        )


def test_build_fit_row_conditioning_routes_every_token_under_turn_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn-mode fit row must still route every token to a real segment.

    Only the Gemma extractor is stubbed; the turn dispatch,
    ``build_context_entry_from_selected_segments``,
    ``align_segments_to_decoded_prefix`` and
    the shared context dispatcher all run for real, using persisted ownership.
    """

    # kills: dropping the turn dispatch (turn_char_starts) in
    # build_extracted_context_entry -- the fixed path would tile the row on
    # segment_size boundaries and the per-turn char starts below would no
    # longer appear as segment boundaries.
    tokenizer = _ZeroWidthTokenizer()
    rendered_ids, _mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "known"},
            ]
        }
    )
    # Two zero-width byte pieces on the user/assistant boundary: training
    # routes them with the FOLLOWING window; the fixed-cache rule would keep
    # them with the preceding one.
    boundary = rendered_ids.index(tokenizer.encode_special("<assistant_start>"))
    token_ids = (
        *rendered_ids[:boundary],
        tokenizer.ZERO_WIDTH,
        tokenizer.ZERO_WIDTH,
        *rendered_ids[boundary:],
    )
    rendered_text, turn_starts = rendered_text_and_turn_char_starts(token_ids, tokenizer)
    bos_text = "<bos>"
    bos_len = len(bos_text)
    source_starts = [0, *(bos_len + start for start in turn_starts[1:])]
    source_ends = [*source_starts[1:], bos_len + len(rendered_text)]
    n_segments = len(source_starts)
    assert n_segments >= 2

    class _GemmaTokenizer:
        bos_token_id = 1

        def decode(self, ids: list[int]) -> str:
            assert ids == [1]
            return bos_text

    setup = _fixed_setup(
        sae_layer=22,
        resolved_sae=SimpleNamespace(hook_name="blocks.22.hook_resid_post"),
        prompt_pipeline=object(),
        prompt_consolidation=None,
        n_sae_features=64,
        segment_mode=methods_module.SegmentMode.turn_delimited_fixed_tokens,
        prompt_segment_size=17,
        prompt_context_window=333,
        prompt_selection_context=object(),
    )
    monkeypatch.setattr(
        methods_module,
        "truncate_to_training_window",
        lambda *_args: ([1], 3),
    )
    extract_calls: dict[str, Any] = {}

    def fake_extract(**kwargs: Any) -> tuple[Any, ...]:
        extract_calls.update(kwargs)
        return (
            [t.tensor([index]) for index in range(n_segments)],
            [t.tensor([0.5]) for _ in range(n_segments)],
            list(range(n_segments)),
            source_starts,
            source_ends,
            3,
        )

    monkeypatch.setattr(methods_module, "extract_document_features", fake_extract)
    monkeypatch.setattr(
        methods_module,
        "select_conversation_segments",
        lambda raw_segments, **_kwargs: [
            SimpleNamespace(
                feature_ids=segment.feature_ids,
                buckets=t.zeros_like(segment.feature_ids),
                char_start=segment.char_start,
                char_end=segment.char_end,
            )
            for segment in raw_segments
        ],
    )

    row = build_fit_row_conditioning(
        cast(methods_module.MethodSetup, setup),
        SimpleNamespace(tokenizer=_GemmaTokenizer(), cfg=SimpleNamespace(n_ctx=8192)),
        object(),
        tokenizer,
        token_ids=token_ids,
        surface="concept 0 positive row 0",
    )

    assert extract_calls["segment_mode"] is methods_module.SegmentMode.turn_delimited_fixed_tokens
    assert extract_calls["turn_char_starts"] == turn_starts
    assert extract_calls["text"] == rendered_text
    assert len(row.segment_assignments) == len(token_ids)
    assert list(row.segment_assignments) == sorted(row.segment_assignments)
    assert min(row.segment_assignments) == 0
    assert max(row.segment_assignments) == len(row.segments) - 1
    # kills: reverting the fit-row dispatch to compute_context_token_segment_assignments.
    following = row.segment_assignments[boundary + 2]
    assert row.segment_assignments[boundary : boundary + 2] == (following, following)
    assert row.segment_assignments[boundary - 1] != following
    # ``rendered_text_and_turn_char_starts`` reports BOS-free coordinates; the
    # cache entry is in with-BOS nanochat coordinates, so shift by the width of
    # the target tokenizer's own BOS token.
    nanochat_offsets = token_to_char_offsets(tokenizer, list(token_ids))
    nanochat_bos_char_len = nanochat_offsets[1] - nanochat_offsets[0]
    expected_boundary_chars = {nanochat_offsets[0]} | {
        nanochat_bos_char_len + start for start in turn_starts[1:]
    }
    assert expected_boundary_chars <= {segment.char_start for segment in row.segments}
    assert len(expected_boundary_chars) == n_segments
