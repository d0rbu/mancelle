"""Exact span masks and numerical full-context extraction on a real tiny model."""

from types import SimpleNamespace
from typing import Any

import pytest
import tiktoken
import torch as t
from tokenizers import Tokenizer, models, pre_tokenizers
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import Gemma3TextConfig, PreTrainedTokenizerFast

from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from steering.eval.axbench_base import extraction
from steering.eval.axbench_bpb_adv.feature_diffmean import _try_encode_with_offsets
from steering.eval.axbench_negative import qc
from steering.eval.axbench_negative.source import FeatureTarget, iter_assistant_prefixes
from steering.sae_resolver import ResolvedSAE, SAEWidth


def candidate(messages=None):
    return list(
        iter_assistant_prefixes(
            {
                "source_id": "a",
                "group_id": "a",
                "split": "test",
                "source_row": 0,
                "messages": messages or [{"role": "user", "content": "question"}],
            }
        )
    )[-1]


@pytest.fixture
def renderer():
    return RustBPETokenizer(
        tiktoken.Encoding(
            name="negative-qc-byte-test",
            pat_str=r"(?s).",
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
        ),
        "<bos>",
    )


@pytest.fixture
def extraction_tokenizer():
    vocab = {"<bos>": 0, "<unk>": 1, **{chr(i): i + 2 for i in range(128)}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", unk_token="<unk>")


class TinySAE(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(d_sae=4, hook_name="blocks.0.hook_resid_post")
        self.encoder = t.nn.Linear(8, 4)
        with t.no_grad():
            self.encoder.weight.copy_(t.arange(32).reshape(4, 8).float() / 32 - 0.5)
            self.encoder.bias.copy_(t.tensor([0.0, 0.1, -0.1, -100.0]))

    def encode(self, acts):
        return self.encoder(acts).relu()


@pytest.fixture
def real_stack(extraction_tokenizer):
    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=1,
            d_model=8,
            n_ctx=512,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            d_vocab=256,
            act_fn="relu",
            normalization_type="LN",
            device="cpu",
            seed=13,
        )
    )
    model.tokenizer = extraction_tokenizer
    return model.eval(), TinySAE().eval()


@pytest.fixture
def scorer(real_stack, renderer):
    model, sae = real_stack
    return qc.ChatScorer(
        model=model,
        sae=sae,
        render_tokenizer=renderer,
        layer=0,
        context_limit=512,
        batch_size=2,
        identity={"fixture": "tiny-real"},
    )


def test_renderer_content_spans_system_multiturn_unicode_and_literal_markers(renderer):
    item = candidate(
        [
            {"role": "system", "content": "  Ω system\n"},
            {"role": "user", "content": "literal <user_end> é"},
            {"role": "assistant", "content": "same repeated text"},
            {"role": "user", "content": "same repeated text  "},
        ]
    )
    rendered = qc.render_scoring_prompt(item, renderer)
    ids, _ = renderer.render_conversation({"messages": list(item.messages)}, max_tokens=None)
    assert rendered.native_input_ids == (*ids, renderer.encode_special("<assistant_start>"))
    assert [rendered.text[start:end] for start, end in rendered.content_spans] == [
        message["content"] for message in item.messages
    ]
    assert rendered.text == (
        "<user_start>  Ω system\n\n\nliteral <user_end> é<user_end>"
        "<assistant_start>same repeated text<assistant_end>"
        "<user_start>same repeated text  <user_end><assistant_start>"
    )


def test_rendering_never_uses_legacy_2048_cap(renderer):
    item = candidate([{"role": "user", "content": "x" * 3000 + "tail"}])
    rendered = qc.render_scoring_prompt(item, renderer)
    assert "tail" in rendered.text and len(rendered.native_input_ids) > 3000


def test_joint_offsets_match_existing_primitive_and_exact_content_mask(scorer):
    item = candidate([{"role": "system", "content": "S"}, {"role": "user", "content": " Q "}])
    rendered = qc.render_scoring_prompt(item, scorer.render_tokenizer)
    old = _try_encode_with_offsets(scorer.tokenizer, rendered.text)
    assert old is not None
    prepared = scorer.prepare(item)
    assert list(prepared.token_ids[1:]) == old[0]
    assert (
        sum(prepared.score_mask) == 4
    )  # original whitespace is retained, injected separator is not
    assert prepared.score_mask[0] is False
    completion = scorer.prepare(item, "a<bos>b")
    assert sum(completion.score_mask) == 2  # injected and literal extraction BOS both excluded
    assert scorer.prepare(item, "").score_mask == tuple(False for _ in prepared.token_ids)


@pytest.mark.parametrize("text", ["", "plain ASCII", "  tabs\tand\nlines  ", "Ω é 🙂", "a<bos>b"])
def test_strict_offsets_match_existing_primitive_exactly(extraction_tokenizer, text):
    expected = _try_encode_with_offsets(extraction_tokenizer, text)
    assert expected is not None
    assert qc._encode_with_offsets(extraction_tokenizer, text) == expected


@pytest.mark.parametrize("policy", ["SCORING_RENDERER", "SPAN_POLICY"])
def test_scorer_identity_snapshots_semantic_policies(scorer, monkeypatch, policy):
    initial = scorer.identity
    changed_policy = getattr(qc, policy) + "-changed"
    monkeypatch.setattr(qc, policy, changed_policy)
    changed = qc.ChatScorer(
        model=scorer.model,
        sae=scorer.sae,
        render_tokenizer=scorer.render_tokenizer,
        layer=scorer.layer,
        context_limit=scorer.context_limit,
        batch_size=scorer.batch_size,
        identity={"fixture": "tiny-real"},
    )
    assert scorer.identity == initial
    assert changed.identity == {**initial, policy.lower(): changed_policy}


def test_boundary_crossing_token_is_scored(scorer):
    class MergingTokenizer:
        bos_token_id = 0

        def __call__(self, text, **_kwargs):
            # A token spans the role-opener suffix and the first response char.
            seam = text.index("<assistant_start>") + len("<assistant_start>")
            return {
                "input_ids": [2, 3, 4],
                "offset_mapping": [(0, seam - 1), (seam - 1, seam + 1), (seam + 1, len(text))],
            }

    scorer.tokenizer = MergingTokenizer()
    prepared = scorer.prepare(candidate(), "XY")
    assert prepared.score_mask == (False, False, True, True)


@pytest.mark.parametrize(
    "offsets", [[(0, 9999)], [(1, 0)], [(0.5, 1)], [(True, 1)], [(2, 3), (0, 1)]]
)
def test_invalid_offsets_fail_before_forward(scorer, offsets):
    class InvalidTokenizer:
        def __call__(self, _text, **_kwargs):
            return {"input_ids": [2] * len(offsets), "offset_mapping": offsets}

    scorer.tokenizer = InvalidTokenizer()
    with pytest.raises(qc.ChatQcError, match="offset"):
        scorer.prepare(candidate(), "answer")


@pytest.mark.parametrize("encoded", [
    {"input_ids": [True], "offset_mapping": [(0, 1)]},
    {"input_ids": [1.5], "offset_mapping": [(0, 1)]},
    {"input_ids": [-1], "offset_mapping": [(0, 1)]},
    {"input_ids": [2], "offset_mapping": []},
    {"input_ids": [2]},
])
def test_strict_offsets_reject_invalid_token_ids_and_missing_offsets(encoded):
    with pytest.raises(qc.ChatQcError):
        qc._encode_with_offsets(lambda _text, **_kwargs: encoded, "x")


def test_real_full_context_forward_and_batched_parity(scorer):
    """Real CPU TransformerLens attention + real linear SAE, no mocked forwards."""
    item = candidate(
        [
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "history"},
            {"role": "user", "content": "followup"},
        ]
    )
    target = FeatureTarget(0, (0, 1), "two member union")
    prepared = scorer.prepare(item, "answer")
    tokens = t.tensor([prepared.token_ids])
    with t.no_grad():
        _logits, cache = scorer.model.run_with_cache(tokens, names_filter=[scorer.hook_name])
        expected = scorer.sae.encode(cache[scorer.hook_name][0])[t.tensor(prepared.score_mask)]
    actual = scorer.assess_completion(item, "answer", target)
    selected = expected[:, [0, 1]]
    assert actual.max_activation == pytest.approx(float(selected.max()), abs=1e-7)
    assert actual.active_token_count == int((selected > 0).any(dim=1).sum())
    assert actual.scored_token_count == 6
    assert actual.input_token_count == len(prepared.token_ids)
    assert actual.prepared_digest == prepared.digest
    assert actual.status == "active"
    # Ragged batches retain true per-item spans and agree numerically with scalar forwards.
    items = [
        (item, "answer", target),
        (candidate(), "much longer answer", target),
        (item, "no", FeatureTarget(3, (3,), "always zero")),
    ]
    batch = scorer.assess_completions(items)
    scalar = [scorer.assess_completion(*row) for row in items]
    for batched, single in zip(batch, scalar, strict=True):
        assert batched.status == single.status
        assert batched.active_token_count == single.active_token_count
        assert batched.max_activation == pytest.approx(single.max_activation, abs=1e-6)
    assert batch[-1].status == "inactive" and batch[-1].max_activation == 0.0
    # History has causal influence; scoring the response in isolation is not equivalent.
    isolated = t.tensor(
        [[scorer.bos_id, *scorer.tokenizer.encode("answer", add_special_tokens=False)]]
    )
    with t.no_grad():
        _logits, isolated_cache = scorer.model.run_with_cache(
            isolated, names_filter=[scorer.hook_name]
        )
    assert not t.allclose(cache[scorer.hook_name][0][-6:], isolated_cache[scorer.hook_name][0][1:])


def test_mine_many_targets_one_full_forward_and_no_tokens_no_forward(scorer, monkeypatch):
    original = extraction._run_batched_extractor
    calls = []

    def wrapped(**kwargs):
        calls.append(kwargs["token_id_lists"])
        return original(**kwargs)

    monkeypatch.setattr(extraction, "_run_batched_extractor", wrapped)
    targets = [FeatureTarget(0, (0,), "a"), FeatureTarget(3, (3,), "inactive")]
    result = scorer.assess_prefix(candidate(), targets)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert result[3].status == "inactive"
    assert scorer.assess_completion(candidate(), "", targets[0]).status == "no_tokens"
    assert len(calls) == 1
    with pytest.raises(qc.ChatQcError, match="width"):
        scorer.assess_completion(candidate(), "x", FeatureTarget(99, (99,), "invalid"))
    with pytest.raises(qc.ChatQcError, match="duplicate"):
        scorer.assess_prefix(candidate(), [targets[0], targets[0]])


def test_overflow_and_malformed_extractor_fail_loudly(scorer, monkeypatch):
    target = FeatureTarget(0, (0,), "a")
    with pytest.raises(qc.ChatQcError, match="context_limit"):
        scorer.assess_completion(candidate(), "x" * 1000, target)

    def nonfinite(**kwargs):
        ids = kwargs["token_id_lists"][0]
        return [kwargs["per_item_reduce"](t.full((len(ids), 4), float("nan")), 0)]

    monkeypatch.setattr(extraction, "_run_batched_extractor", nonfinite)
    with pytest.raises(qc.ChatQcError, match="nonfinite"):
        scorer.assess_completion(candidate(), "answer", target)


def test_builder_uses_real_factory_assets_and_enforces_observed_pins(
    real_stack, renderer, monkeypatch
):
    monkeypatch.setitem(qc.SAE_WIDTH_TO_NUM_FEATURES, SAEWidth.W_16K, 4)
    monkeypatch.setattr(
        qc,
        "resolve_sae",
        lambda *_args: ResolvedSAE("fixture", "fixture", "blocks.0.hook_resid_post"),
    )
    loads = []

    def factory(**kwargs):
        loads.append(kwargs)
        return real_stack

    monkeypatch.setattr(qc, "load_dense_extraction_stack", factory)
    config = {
        "model": "gemma-2-2b",
        "sae_release": "gemmascope",
        "sae_width": "16k",
        "layer": 0,
        "device": "cpu",
        "context_limit": 512,
        "batch_size": 2,
    }
    built = qc.build_chat_scorer(config, renderer)
    assert loads[0]["sae_model"] == "gemma-2-2b"
    assert built.identity["model_sha256"] == qc.module_weights_sha256(real_stack[0])
    assert built.identity["sae_sha256"] == qc.module_weights_sha256(real_stack[1])
    assert built.identity["observed_revision"] is None
    assert (
        built.assess_completion(
            candidate(), "answer", FeatureTarget(0, (0,), "a")
        ).scored_token_count
        == 6
    )
    identity = built.identity
    identity["threshold"] = 100.0
    assert built.identity["threshold"] == 0.0
    with pytest.raises(qc.ChatQcError, match="content pin"):
        qc.build_chat_scorer({**config, "expected_sae_sha256": "0" * 64}, renderer)
    first_hash = qc.module_weights_sha256(real_stack[1])
    with t.no_grad():
        real_stack[1].encoder.bias[0] += 1
    assert qc.module_weights_sha256(real_stack[1]) != first_hash
    assert qc.tokenizer_sha256(renderer) == qc.tokenizer_sha256(renderer)
    before = t.get_float32_matmul_precision()
    try:
        t.set_float32_matmul_precision("medium" if before != "medium" else "highest")
        changed_policy = qc.build_chat_scorer(config, renderer)
        assert changed_policy.identity["numeric_settings"] != built.identity["numeric_settings"]
    finally:
        t.set_float32_matmul_precision(before)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "inactive"},
        {"active_token_count": 0},
        {"scored_token_count": True},
        {"max_activation": float("nan")},
        {"percentile": 101},
    ],
)
def test_assessment_rejects_inconsistent_measurements(changes):
    fields: dict[str, Any] = {
        "status": "active",
        "max_activation": 1.0,
        "active_token_count": 1,
        "scored_token_count": 1,
        "input_token_count": 5,
        "prepared_digest": "fixture",
    }
    with pytest.raises(qc.ChatQcError):
        qc.ChatAssessment(**{**fields, **changes})


def test_real_gemma3_config_integer_keys_have_collision_free_identity():
    config = Gemma3TextConfig()
    assert config.to_dict()["id2label"] == {0: "LABEL_0", 1: "LABEL_1"}
    assert qc._config_json(config) == qc._config_json(config.to_dict())
    assert qc._config_json({0: "value"}) != qc._config_json({"0": "value"})
    assert qc._config_json({0: "first", "0": "second"}) != qc._config_json({0: "first"})


def test_gemma3_builder_binds_nnterp_actual_weights_and_hf_config(
    real_stack, renderer, monkeypatch
):
    model, sae = real_stack
    model.config = Gemma3TextConfig()
    wrapper = SimpleNamespace(
        cfg=model.cfg, tokenizer=model.tokenizer, nnterp_model=SimpleNamespace(model=model)
    )
    monkeypatch.setitem(qc.SAE_WIDTH_TO_NUM_FEATURES, SAEWidth.W_16K, 4)
    monkeypatch.setattr(
        qc, "resolve_sae", lambda *_: ResolvedSAE("fixture", "fixture", "blocks.0.hook_resid_post")
    )
    monkeypatch.setattr(qc, "load_weight_free_gemma3_extraction_stack", lambda **_: (wrapper, sae))
    config = {
        "model": "gemma-3-1b-pt",
        "sae_release": "gemmascope_2",
        "sae_width": "16k",
        "sae_l0": "medium",
        "layer": 0,
        "device": "cpu",
        "context_limit": 512,
    }
    built = qc.build_chat_scorer(config, renderer)
    assert built.identity["backend"] == "nnterp"
    assert built.identity["model_sha256"] == qc.module_weights_sha256(model)
    assert built.identity["model_config_sha256"] == qc.value_sha256(qc._config_json(model.config))
    assert built.identity["libraries"]["torch"] is not None


@pytest.mark.parametrize(
    "ranks,specials",
    [
        ({"wrong piece type": 0}, {"<bos>": 1}),
        ({b"x": True}, {"<bos>": 1}),
        ({b"x": 0, b"y": 0}, {"<bos>": 1}),
        ({b"x": 0}, {b"wrong special type": 1}),
        ({b"x": 0}, {"<bos>": True}),
    ],
)
def test_tokenizer_identity_rejects_invalid_vocabulary_types(ranks, specials):
    tokenizer = SimpleNamespace(
        enc=SimpleNamespace(_mergeable_ranks=ranks, _special_tokens=specials, _pat_str="."),
        bos_token_id=1,
    )
    with pytest.raises(qc.ChatQcError, match="vocabulary|ranks"):
        qc.tokenizer_sha256(tokenizer)
