"""Unit tests for synth/qc.py (Task 13)."""

from __future__ import annotations

import random
import unittest.mock as mock
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from steering.conditional.percentile_table import percentile_to_uint8
from steering.eval.axbench_base import (
    _stage0c_internals as _si,
)
from steering.eval.axbench_base import (
    extraction,
)
from steering.eval.axbench_base._stage0c_internals import (
    _PrefixCandidate,
    _verify_candidate,
)
from steering.eval.axbench_base.synth import qc
from steering.eval.axbench_base.synth.qc import (
    AvoidContinuationOutcome,
    ContinuationQcInput,
    PrefixPercentileCache,
    _max_percentile,
    _max_percentile_for_target,
    classify_avoid_continuation,
    classify_avoid_continuations,
    compute_fc_percentiles,
    fc_percentile_on_prefix,
    make_cluster_prefix_percentile_cache_key,
    make_prefix_percentile_cache_key,
    output_fc_percentile_for_continuation,
    output_fc_percentiles_for_continuations,
)
from steering.eval.axbench_base.synth.response_parser import (
    DEFAULT_EXTRACTION_TOKENIZER_ID,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cluster_feature_contract import ClusterFeatureTarget
from tests.axbench_base._loguru_capture import capture_loguru_info
from tests.axbench_base.unit.synth_qc_reference import fc_active_on_continuation
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_sae(
    hook_name: str = "blocks.8.hook_resid_post",
    *,
    release: str = "rel",
    d_sae: int = 16,
) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(hook_name=hook_name, release=release, d_sae=d_sae),
    )


def _fake_gemma(
    *, n_ctx: int = 2048, model_name: str = "gemma-test",
) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(n_ctx=n_ctx, model_name=model_name))


class _WordTokenizer:
    """Whitespace-level tokenizer.

    ``encode("a b c") -> [1, 2, 3]``. Test data uses a leading space in
    continuations (e.g. ``prefix="one two"``, ``continuation=" three"``)
    so that ``prefix + continuation`` yields ``"one two three"`` and the
    continuation is distinguishable as an extra token at the seam. This
    mirrors the real Gemma tokenizer's behavior, where a continuation
    typically begins with a word-boundary token.

    Deterministic per-word id assignment is not required -- only counts
    matter for the QC tests, since the fake extractor produces activation
    tensors shaped by token count.

    ``decode`` is a trivial inverse used by Stage 0c verification path
    tests that build continuations from stubbed token streams.

    ``bos_token_id`` is a sentinel far outside the tiny word-count id
    range this tokenizer assigns (1, 2, 3, ...), so it never collides
    with a real content token in test data.
    """

    bos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.split()
        return [i + 1 for i, _ in enumerate(words)]

    def decode(self, token_ids: list[int]) -> str:
        # Join integer ids back into a whitespace-separated token stream.
        # The actual text doesn't matter for the QC tests — only that
        # re-encoding the result yields the expected token count.
        return " ".join(str(t) for t in token_ids)


class _ContentTokenizer:
    """Content-stable tokenizer for cache-key tests.

    ``bos_token_id`` is a sentinel far outside the small sequential
    vocab ids this tokenizer assigns per distinct word.
    """

    bos_token_id = 999

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
            ids.append(self.vocab[word])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)


class _SeamMergeTokenizer:
    """Tokenizer simulating a prefix/continuation boundary retokenization.

    The prefix's own tokens remain a strict PREFIX of the combined
    tokenization (``full_ids[:n_prefix] == prefix_tokens``), but the
    continuation retokenizes, as part of the combined string, into a
    DIFFERENT token count/values than a naive standalone encoding would
    give -- so tests using this must read the tail of the COMBINED
    encoding, not a separate encode of the continuation text alone.
    """

    bos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "alpha":
            return [1]
        if text == "alpha beta gamma":
            return [1, 30, 31]
        raise AssertionError(f"unexpected text {text!r}")

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(tid) for tid in token_ids)


class _OffsetTokenizer:
    """Tokenizer whose ids depend on which tokenizer id was requested.

    ``bos_token_id`` is a constant sentinel (999) regardless of
    ``offset`` -- larger than any id either offset instance (0 or 80)
    produces for the tiny word counts used in tests, so it never
    collides with a real content token.
    """

    bos_token_id = 999

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.split()
        return [self.offset + i + 1 for i, _ in enumerate(words)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(tid) for tid in token_ids)


@pytest.fixture
def stub_qc_env(monkeypatch):
    state: dict[str, Any] = {
        "tokenizer": _WordTokenizer(),
        "extractor_calls": [],
        "extractor_return": None,  # per-test override
    }

    def _fake_tokenizer(*args, **kwargs):
        del args, kwargs
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        state["extractor_calls"].append(kwargs)
        if state["extractor_return"] is None:
            n_tokens = len(kwargs["token_id_lists"][0])
            return [t.zeros(n_tokens, 16)]
        return state["extractor_return"]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    yield state


def _identity_percentile_table(n_features: int = 16) -> t.Tensor:
    """A percentile table where p_k = k/100 for every feature.

    With this table, magnitude_to_percentile returns percentiles that
    roughly equal the magnitude * 100. This makes test assertions
    mechanical: "max activation was 0.5" -> "seed_percentile ~= 50".
    """
    table = t.zeros(n_features, 99, dtype=t.float64)
    for i in range(n_features):
        for j in range(99):
            table[i, j] = (j + 1) / 100.0
    return table


def _cluster_target(
    representative_id: int,
    members: tuple[int, ...],
    *,
    fingerprint: str = "a" * 64,
) -> ClusterFeatureTarget:
    return ClusterFeatureTarget(
        representative_id=representative_id,
        allowed_member_ids=members,
        projecting_source_ids=members,
        feature_contract_fingerprint=fingerprint,
    )


class _IndexOnlyMetadata:
    """Mapping-like object with __getitem__ but no .get."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


# ---------------------------------------------------------------------------
# Internal cache/QC helpers
# ---------------------------------------------------------------------------


def test_cluster_target_percentile_uses_each_member_own_statistics_then_amax():
    table = _identity_percentile_table()
    # Feature 2 has a much larger activation than representative 1, but a
    # ten-times-larger percentile scale. Raw-magnitude max would choose feature
    # 2; the required own-statistic percentile oracle chooses representative 1.
    table[2] *= 10.0
    activations = t.zeros((3, 16), dtype=t.float32)
    activations[0, 1] = 0.8
    activations[1, 2] = 0.9
    target = _cluster_target(1, (1, 2))

    actual = _max_percentile_for_target(activations, target, table)

    member_percentiles = [
        _max_percentile(activations[:, member], member, table)
        for member in target.allowed_member_ids
    ]
    assert actual == max(member_percentiles)
    assert member_percentiles[0] > member_percentiles[1]
    assert float(activations[:, 1].max()) < float(activations[:, 2].max())


def test_cluster_target_silent_members_return_zero():
    assert (
        _max_percentile_for_target(
            t.zeros((4, 16)),
            _cluster_target(1, (1, 2)),
            _identity_percentile_table(),
        )
        == 0.0
    )


def test_singleton_cluster_percentile_is_exact_legacy_scalar_oracle() -> None:
    """Identity-after-mask targets preserve the scalar QC computation exactly."""

    generator = t.Generator().manual_seed(737)
    for dtype in (t.float16, t.float32, t.float64):
        for n_tokens in (0, 1, 2, 7, 31):
            for feature_id in (0, 1, 7, 15):
                table = t.sort(
                    t.rand((16, 99), generator=generator, dtype=t.float64),
                    dim=1,
                ).values
                activations = t.rand(
                    (n_tokens, 16),
                    generator=generator,
                    dtype=t.float32,
                ).to(dtype)
                if n_tokens and (n_tokens + feature_id) % 3 == 0:
                    activations[:, feature_id] = 0
                if n_tokens and (n_tokens + feature_id) % 5 == 0:
                    activations[:, feature_id] = -activations[:, feature_id]

                clustered = _max_percentile_for_target(
                    activations,
                    _cluster_target(feature_id, (feature_id,)),
                    table,
                )
                legacy = _max_percentile(
                    activations[:, feature_id],
                    feature_id,
                    table,
                )

                assert clustered == legacy


def test_cluster_prefix_cache_identity_binds_members_table_and_contract():
    kwargs = {
        "token_ids": [1, 2, 3],
        "base_model": _fake_gemma(),
        "base_sae": _fake_sae(),
        "sae_layer": 8,
        "tokenizer_id": "tok",
        "percentile_namespace": "stats",
    }
    baseline = make_cluster_prefix_percentile_cache_key(
        target=_cluster_target(1, (1, 2)),
        percentile_table_checksum="c" * 64,
        **kwargs,
    )
    assert baseline != make_cluster_prefix_percentile_cache_key(
        target=_cluster_target(1, (1, 2, 3)),
        percentile_table_checksum="c" * 64,
        **kwargs,
    )
    assert baseline != make_cluster_prefix_percentile_cache_key(
        target=_cluster_target(1, (1, 2)),
        percentile_table_checksum="d" * 64,
        **kwargs,
    )
    assert baseline != make_cluster_prefix_percentile_cache_key(
        target=_cluster_target(1, (1, 2), fingerprint="b" * 64),
        percentile_table_checksum="c" * 64,
        **kwargs,
    )


def test_qc_percentile_currency_is_unquantized_unlike_training_uint8_roundtrip():
    """Pin: QC's percentile currency is the RAW ``[0.5, 99.5]`` float, never
    the lossy uint8-quantized round-trip
    (``steering.conditional.percentile_table.percentile_to_uint8`` /
    ``uint8_to_percentile``) that training/packed-rows data uses on disk. See
    the module docstring's "Percentile currency (deliberate asymmetry)"
    section. Both the legacy scalar path (:func:`_max_percentile`) and the
    clustered member-set path (:func:`_max_percentile_for_target`) share this
    unquantized currency -- this test pins BOTH against a magnitude chosen to
    land off the uint8 quantization grid (256 bins over ``[0, 100]``), so a
    future change that accidentally routes either path through
    ``percentile_to_uint8``/``uint8_to_percentile`` is caught here.
    """
    table = _identity_percentile_table()  # p_k = k/100 for every feature
    magnitude = 0.4555  # interpolates to exactly 45.55 on this table

    scalar_acts = t.zeros((1, 16), dtype=t.float32)
    scalar_acts[0, 3] = magnitude
    raw_scalar = _max_percentile(scalar_acts[:, 3], f_c=3, percentile_table=table)
    assert raw_scalar == pytest.approx(45.55)

    cluster_acts = t.zeros((1, 16), dtype=t.float32)
    cluster_acts[0, 3] = magnitude
    raw_cluster = _max_percentile_for_target(
        cluster_acts, _cluster_target(3, (3,)), table,
    )
    assert raw_cluster == pytest.approx(45.55)

    # What the training/packed-rows uint8 currency would have stored instead
    # (encode -> decode, rescaled back to percent units per uint8_to_
    # percentile's own docstring: "multiply by 100.0 to recover an
    # approximation of the encoded percentile").
    quantized_and_rescaled = (
        float(percentile_to_uint8(t.tensor([raw_scalar])).item()) / 255.0 * 100.0
    )
    assert quantized_and_rescaled != pytest.approx(45.55)
    # Both QC currencies must diverge from the lossy uint8 round-trip by more
    # than float noise -- proving QC never collapsed onto that grid.
    assert abs(raw_scalar - quantized_and_rescaled) > 0.01
    assert abs(raw_cluster - quantized_and_rescaled) > 0.01


def test_metadata_get_returns_none_when_metadata_missing():
    assert qc._metadata_get(object(), "anything") is None


def test_metadata_get_uses_dotget_when_available():
    obj = SimpleNamespace(metadata={"hook_name": "x"})
    assert qc._metadata_get(obj, "hook_name") == "x"
    assert qc._metadata_get(obj, "missing") is None


def test_metadata_get_falls_back_to_bracket_indexing():
    obj = SimpleNamespace(metadata=_IndexOnlyMetadata({"hook_name": "bracket"}))
    assert qc._metadata_get(obj, "hook_name") == "bracket"


def test_metadata_get_returns_none_on_non_mapping_metadata():
    obj = SimpleNamespace(metadata=object())
    assert qc._metadata_get(obj, "hook_name") is None


def test_model_cache_id_falls_back_to_type_name():
    class _AnonymousModel:
        cfg = SimpleNamespace()

    assert qc._model_cache_id(_AnonymousModel()).endswith("._AnonymousModel")


def test_sae_cache_id_reads_metadata_fallbacks(monkeypatch):
    monkeypatch.setattr(qc, "_sae_hook_name", lambda sae: "blocks.8.hook_resid_post")
    sae = SimpleNamespace(
        cfg=SimpleNamespace(
            metadata=_IndexOnlyMetadata({
                "release": "meta-release",
                "sae_id": "meta-id",
                "d_sae": 321,
            }),
        ),
    )
    assert qc._sae_cache_id(sae) == (
        "meta-release|meta-id|blocks.8.hook_resid_post|321"
    )


def test_base_context_window_missing_n_ctx_raises():
    with pytest.raises(AxBenchBaseError, match="no \\.cfg\\.n_ctx"):
        qc._base_context_window(SimpleNamespace(cfg=SimpleNamespace()))


def test_run_sae_on_token_lists_empty_input_returns_empty(stub_qc_env):
    del stub_qc_env
    assert qc._run_sae_on_token_lists(
        [], _fake_gemma(), _fake_sae(), sae_layer=8,
    ) == []


def test_run_sae_on_token_lists_rejects_zero_token_sequences(stub_qc_env):
    del stub_qc_env
    with pytest.raises(AxBenchBaseError, match="zero tokens"):
        qc._run_sae_on_token_lists(
            [[]], _fake_gemma(), _fake_sae(), sae_layer=8,
        )


def test_run_sae_on_token_lists_clamps_before_extractor(stub_qc_env):
    qc._run_sae_on_token_lists(
        [[1, 2, 3, 4, 5, 6]], _fake_gemma(n_ctx=4), _fake_sae(), sae_layer=8,
    )
    assert stub_qc_env["extractor_calls"][-1]["token_id_lists"] == [[1, 2, 3, 4]]


def test_max_percentiles_for_feature_empty_list_returns_empty():
    assert qc._max_percentiles_for_feature([], 5, _identity_percentile_table()) == []


def test_max_percentiles_for_feature_empty_tensor_row_returns_zero():
    acts = [
        t.zeros((0, 16), dtype=t.float32),
        t.tensor([[0.0] * 5 + [0.5] + [0.0] * 10], dtype=t.float32),
    ]
    out = qc._max_percentiles_for_feature(acts, 5, _identity_percentile_table())
    assert out[0] == 0.0
    assert 45.0 <= out[1] <= 55.0


def test_max_percentiles_for_feature_matches_scalar_reference():
    table = _identity_percentile_table()
    acts_list = [
        t.tensor(
            [
                [0.0] * 5 + [0.0] + [0.0] * 10,
                [0.0] * 5 + [0.2] + [0.0] * 10,
                [0.0] * 5 + [0.8] + [0.0] * 10,
            ],
            dtype=t.float32,
        ),
        t.tensor(
            [
                [0.0] * 5 + [0.1] + [0.0] * 10,
                [0.0] * 5 + [0.4] + [0.0] * 10,
            ],
            dtype=t.float32,
        ),
        t.zeros((0, 16), dtype=t.float32),
    ]
    expected = [
        _max_percentile(acts[:, 5], f_c=5, percentile_table=table)
        for acts in acts_list
    ]
    actual = qc._max_percentiles_for_feature(acts_list, 5, table)
    assert actual == pytest.approx(expected)


def test_tokenize_prefix_and_full_default_uses_default_tokenizer(monkeypatch):
    seen: list[str] = []

    def _fake_tokenizer(tokenizer_id):
        seen.append(tokenizer_id)
        return _WordTokenizer()

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    prefix_tokens, full_tokens = qc._tokenize_prefix_and_full("one", " two")

    assert seen == [DEFAULT_EXTRACTION_TOKENIZER_ID]
    assert prefix_tokens == [1]
    # full_tokens carries the injected BOS at position 0; prefix_tokens does not.
    assert full_tokens == [_WordTokenizer.bos_token_id, 1, 2]


def test_tokenize_prefix_and_full_injects_one_bos_on_full_only(monkeypatch):
    """full_tokens[0] is always the injected BOS; prefix_tokens never carries one."""
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *a, **k: _WordTokenizer())
    prefix_tokens, full_tokens = qc._tokenize_prefix_and_full("one two", " three")

    assert prefix_tokens == [1, 2]
    assert full_tokens[0] == _WordTokenizer.bos_token_id
    assert full_tokens == [_WordTokenizer.bos_token_id, 1, 2, 3]
    assert _WordTokenizer.bos_token_id not in prefix_tokens


def test_tokenize_prefix_and_full_tolerates_seam_merge(monkeypatch):
    """A seam merge is a LEGITIMATE QC outcome, NOT an error.

    When a continuation subword glues onto the prefix's trailing token so the
    combined tokenization does not start with the prefix's own tokens, QC must
    NOT raise (unlike the derive path). ``classify_avoid_continuation`` routes
    ``1 + n_prefix >= len(full_tokens)`` to NO_TOKENS; a hard raise here would
    crash the QC stage on a normal continuation (regression caught by the
    Stage-1b/1c golden fixtures). The function returns normally, BOS-prepended.
    """

    class _MismatchTokenizer:
        bos_token_id = 999

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if text == "alpha":
                return [1]
            if text == "alphabeta":  # seam merge: full doesn't start with [1]
                return [7, 8]
            raise AssertionError(f"unexpected text {text!r}")

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *a, **k: _MismatchTokenizer())
    prefix_tokens, full_tokens = qc._tokenize_prefix_and_full("alpha", "beta")
    assert prefix_tokens == [1]  # length is the boundary; content approximate on merge
    assert full_tokens == [999, 7, 8]  # BOS prepended; no raise
    # The downstream NO_TOKENS guard (1 + n_prefix >= len(full_tokens)) fires:
    # 1 + 1 == 2 < 3, so this particular merge is scored; a fuller merge
    # (len(full) <= 1 + n_prefix) routes to NO_TOKENS in classify_avoid_continuation.


def test_tokenize_prefix_and_full_raises_on_missing_bos(monkeypatch):
    class _NoBosWordTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            words = text.split()
            return [i + 1 for i, _ in enumerate(words)]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *a, **k: _NoBosWordTokenizer())
    with pytest.raises(AxBenchBaseError, match="no BOS id"):
        qc._tokenize_prefix_and_full("one", " two")


class _BosLiteralWordTokenizer:
    """Whitespace tokenizer where the literal word '<bos>' re-encodes to the
    real BOS id even with add_special_tokens=False -- the double-BOS the guard
    now WARNS on but PRESERVES (mirrors real Gemma behavior)."""

    bos_token_id = 999
    all_special_ids = [999]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [999 if w == "<bos>" else i + 1 for i, w in enumerate(text.split())]


def test_tokenize_prefix_and_full_preserves_in_data_special(monkeypatch):
    """A literal '<bos>' in prefix/continuation re-encodes to the real BOS id;
    it is natural corpus/synth text, so we PRESERVE it and prepend our own BOS
    anyway (an intentional double-BOS) rather than dropping the row.

    The returned ``full_tokens`` carries our injected BOS at position 0 (dropped
    by the ``acts[1:...]`` slices) and the in-data BOS preserved as data. Teeth:
    with prefix boundary ``n_prefix == len(prefix_tokens) == 1``, the
    continuation span is ``full_tokens[1 + n_prefix:] == full_tokens[2:]``, which
    must still contain the in-data BOS id -- if preservation dropped/deduped it,
    ``full_tokens`` would shrink to ``[999, 1]`` and the continuation span would
    be empty, REDing this test.
    """
    monkeypatch.setattr(
        qc, "get_extraction_tokenizer", lambda *a, **k: _BosLiteralWordTokenizer()
    )
    prefix_tokens, full_tokens = qc._tokenize_prefix_and_full("one", " <bos>")
    assert prefix_tokens == [1]  # length is the prefix/continuation boundary
    # injected BOS at 0, content 'one'->1 at 1, in-data '<bos>'->999 preserved at 2
    assert full_tokens == [999, 1, 999]
    n_prefix = len(prefix_tokens)
    continuation_span = full_tokens[1 + n_prefix:]
    assert continuation_span == [999], (
        "the in-data special must survive in the continuation span "
        f"(acts[1 + n_prefix:]); got {continuation_span!r}"
    )


def test_tokenize_prefix_and_full_warns_on_special_id_pollution(monkeypatch):
    """The in-data special fires a loguru WARNING naming the polluting id, but
    the call proceeds and returns the preserved (double-BOS) tokens."""
    monkeypatch.setattr(
        qc, "get_extraction_tokenizer", lambda *a, **k: _BosLiteralWordTokenizer()
    )
    with capture_loguru_info() as captured:
        _prefix_tokens, full_tokens = qc._tokenize_prefix_and_full("one", " <bos>")
    assert full_tokens == [999, 1, 999]  # proceeded, in-data special preserved
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected exactly one pollution warning: {captured!r}"
    assert "[999]" in warnings[0], (
        f"warning must name the polluting id 999: {warnings[0]!r}"
    )


def test_get_many_raises_on_empty_corpus_seed():
    """An empty dirty seed (a corpus fragment that should never be empty) RAISES,
    rather than emitting a silent 0.0 zero-percentile that would mis-classify the
    seed as clean. An all-empty batch never loads a tokenizer, so the raise fires
    at the first empty seed."""
    cache = qc.PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=128),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="dummy",
        percentile_namespace="test",
    )
    with pytest.raises(
        AxBenchBaseError, match="a corpus seed should never be empty"
    ):
        cache.get_many([""], 5)


def test_get_many_preserves_in_data_special_and_warns(monkeypatch):
    """A Stage-1b seed prefix containing a literal '<bos>' encodes to the real
    BOS id; get_many PRESERVES it (the seed is natural corpus text) and forwards
    the double-BOS rather than dropping the seed from the percentile scoring --
    warning for observability.

    Teeth: the exact token list forwarded to the SAE must keep the in-data BOS as
    data (``[999, 1, 999, 3]`` -- our injected BOS at 0, then the seed's own
    ``clean``->1, ``<bos>``->999, ``prefix``->3), and a percentile is still
    returned. If preservation dropped the in-data 999, the forwarded list would
    change, REDing this test.
    """
    monkeypatch.setattr(
        qc, "get_extraction_tokenizer", lambda *a, **k: _BosLiteralWordTokenizer()
    )
    forwarded: list[list[list[int]]] = []

    def _capture_forward(token_lists, base_model, base_sae, *, sae_layer):
        del base_model, base_sae, sae_layer
        forwarded.append([list(toks) for toks in token_lists])
        return [t.zeros(len(toks), 16) for toks in token_lists]

    monkeypatch.setattr(qc, "_run_sae_on_token_lists", _capture_forward)
    cache = qc.PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=128),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="dummy",
        percentile_namespace="test",
    )
    with capture_loguru_info() as captured:
        results = cache.get_many(["clean <bos> prefix"], 5)
    assert len(results) == 1
    assert isinstance(results[0], float)  # scored, not dropped
    assert forwarded == [[[999, 1, 999, 3]]], (
        "get_many must forward the seed's original tokens with the in-data BOS "
        f"preserved (plus our injected BOS at 0); got {forwarded!r}"
    )
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected exactly one pollution warning: {captured!r}"
    assert "[999]" in warnings[0], (
        f"warning must name the polluting id 999: {warnings[0]!r}"
    )


def test_qc_helpers_forward_explicit_tokenizer_id(monkeypatch):
    seen: list[str] = []

    def _fake_tokenizer(tokenizer_id):
        seen.append(tokenizer_id)
        return _WordTokenizer()

    def _fake_extractor(**kwargs):
        return [
            t.zeros((len(tokens), 16), dtype=t.float32)
            for tokens in kwargs["token_id_lists"]
        ]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    table = _identity_percentile_table()

    fc_active_on_continuation(
        "one", " two", 5, _fake_gemma(), _fake_sae(),
        sae_layer=8,
        tokenizer_id="custom/qc-tokenizer",
    )
    compute_fc_percentiles(
        "one", " two", 5, _fake_gemma(), _fake_sae(), table,
        sae_layer=8,
        tokenizer_id="custom/qc-tokenizer",
    )
    fc_percentile_on_prefix(
        "one", 5, _fake_gemma(), _fake_sae(), table,
        sae_layer=8,
        tokenizer_id="custom/qc-tokenizer",
    )

    assert seen == ["custom/qc-tokenizer"] * 3


def test_qc_helpers_are_tokenizer_id_sensitive(monkeypatch):
    def _fake_tokenizer(tokenizer_id):
        if tokenizer_id == DEFAULT_EXTRACTION_TOKENIZER_ID:
            return _OffsetTokenizer(0)
        if tokenizer_id == "custom/high-offset":
            return _OffsetTokenizer(80)
        raise AssertionError(f"unexpected tokenizer_id={tokenizer_id!r}")

    def _fake_extractor(**kwargs):
        out = []
        for token_ids in kwargs["token_id_lists"]:
            acts = t.zeros((len(token_ids), 16), dtype=t.float32)
            for row_idx, token_id in enumerate(token_ids):
                acts[row_idx, 5] = float(token_id) / 100.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    table = _identity_percentile_table()

    default_prefix_pct = fc_percentile_on_prefix(
        "one two", 5, _fake_gemma(), _fake_sae(), table, sae_layer=8,
    )
    custom_prefix_pct = fc_percentile_on_prefix(
        "one two", 5, _fake_gemma(), _fake_sae(), table,
        sae_layer=8,
        tokenizer_id="custom/high-offset",
    )
    assert custom_prefix_pct > default_prefix_pct

    default_seed_pct, default_out_pct = compute_fc_percentiles(
        "one", " two", 5, _fake_gemma(), _fake_sae(), table, sae_layer=8,
    )
    custom_seed_pct, custom_out_pct = compute_fc_percentiles(
        "one", " two", 5, _fake_gemma(), _fake_sae(), table,
        sae_layer=8,
        tokenizer_id="custom/high-offset",
    )
    assert custom_seed_pct > default_seed_pct
    assert custom_out_pct > default_out_pct

    assert fc_active_on_continuation(
        "one", " two", 5, _fake_gemma(), _fake_sae(),
        sae_layer=8,
        extraction_threshold=0.5,
    ) is False
    assert fc_active_on_continuation(
        "one", " two", 5, _fake_gemma(), _fake_sae(),
        sae_layer=8,
        extraction_threshold=0.5,
        tokenizer_id="custom/high-offset",
    ) is True


# ---------------------------------------------------------------------------
# fc_active_on_continuation  (reference — tests/axbench_base/unit/synth_qc_reference.py)
# ---------------------------------------------------------------------------


def test_fc_active_returns_true_above_threshold(stub_qc_env):
    """Activation > threshold in the continuation slice -> True."""
    # prefix "one two" (2 tokens), continuation " three" (1 token); +1 for the
    # injected BOS at row 0 -> 4 rows total. Continuation is row 3.
    acts = t.zeros(4, 16)
    acts[3, 5] = 0.5  # continuation position, feature 5
    stub_qc_env["extractor_return"] = [acts]
    result = fc_active_on_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.0,
    )
    assert result is True
    assert stub_qc_env["extractor_calls"][-1]["token_id_lists"] == [
        [_WordTokenizer.bos_token_id, 1, 2, 3]
    ]


def test_fc_active_forwards_sae_layer(stub_qc_env):
    """The sae_layer kwarg is forwarded to the batched extractor."""
    acts = t.zeros(4, 16)
    acts[3, 5] = 0.5
    stub_qc_env["extractor_return"] = [acts]
    fc_active_on_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=12,
    )
    assert stub_qc_env["extractor_calls"][-1]["layer"] == 12


def test_fc_active_returns_false_when_zero(stub_qc_env):
    """All-zero continuation activations -> False."""
    acts = t.zeros(4, 16)
    stub_qc_env["extractor_return"] = [acts]
    result = fc_active_on_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.0,
    )
    assert result is False


def test_fc_active_threshold_strict(stub_qc_env):
    """Activation equal to threshold is NOT above -> False."""
    acts = t.zeros(4, 16)
    acts[3, 5] = 0.5
    stub_qc_env["extractor_return"] = [acts]
    result = fc_active_on_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.5,
    )
    assert result is False


def test_fc_active_above_threshold_but_not_strict(stub_qc_env):
    """Activation barely above threshold -> True."""
    acts = t.zeros(4, 16)
    acts[3, 5] = 0.51
    stub_qc_env["extractor_return"] = [acts]
    result = fc_active_on_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.5,
    )
    assert result is True


def test_fc_active_only_prefix_activation_returns_false(stub_qc_env):
    """Activation only in the BOS/prefix rows, not in the continuation -> False."""
    acts = t.zeros(4, 16)
    acts[0, 5] = 3.0  # injected BOS row -- must be excluded
    acts[1, 5] = 2.0  # prefix token
    acts[2, 5] = 1.0  # prefix token
    stub_qc_env["extractor_return"] = [acts]
    result = fc_active_on_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.0,
    )
    assert result is False


def test_fc_active_empty_continuation_returns_false(stub_qc_env):
    """Empty continuation -> no tokens to check -> False (short-circuit, no extractor call)."""
    result = fc_active_on_continuation(
        "one two", "", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.0,
    )
    assert result is False
    # No extractor call because the short-circuit fires before running Gemma.
    assert stub_qc_env["extractor_calls"] == []


def test_fc_active_raises_on_empty_both(stub_qc_env):
    """Both prefix and continuation empty -> tokenizer returns [] -> error."""
    with pytest.raises(AxBenchBaseError, match="zero tokens"):
        fc_active_on_continuation(
            "", "", f_c=5,
            base_model=_fake_gemma(), base_sae=_fake_sae(),
            sae_layer=8,
        )


def test_fc_active_raises_on_empty_extractor_return(stub_qc_env):
    stub_qc_env["extractor_return"] = []
    with pytest.raises(AxBenchBaseError, match="no activation tensors"):
        fc_active_on_continuation(
            "one two", " three", f_c=5,
            base_model=_fake_gemma(), base_sae=_fake_sae(),
            sae_layer=8,
        )


def test_fc_active_respects_seam_retok_tail_slice(monkeypatch):
    """Strict-zero QC checks only the combined-token tail after seam merges,
    excluding the injected BOS (row 0) and the prefix (row 1)."""
    acts = t.zeros(4, 16)
    acts[0, 5] = 1.0  # injected BOS row -- must be excluded
    acts[1, 5] = 1.0  # prefix row ("alpha") -- must be excluded
    acts[3, 5] = 0.9  # continuation row -- fires ACTIVE

    captured: dict[str, Any] = {}

    def _extractor(**kwargs):
        captured["token_id_lists"] = kwargs["token_id_lists"]
        return [acts]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *args, **kwargs: _SeamMergeTokenizer())
    monkeypatch.setattr(qc, "_run_batched_extractor", _extractor)
    result = fc_active_on_continuation(
        "alpha", " beta gamma", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8,
        extraction_threshold=0.5,
    )
    assert captured["token_id_lists"] == [[_SeamMergeTokenizer.bos_token_id, 1, 30, 31]]
    assert result is True


# ---------------------------------------------------------------------------
# classify_avoid_continuation  (issues #513, #518 — unified tokenize-once avoid classifier)
# ---------------------------------------------------------------------------


def test_classify_avoid_continuation_checks_every_allowed_cluster_member(
    stub_qc_env,
):
    acts = t.zeros((3, 16), dtype=t.float32)
    # BOS, one prefix token, one continuation token. Only sibling 5 fires.
    acts[2, 5] = 0.25
    stub_qc_env["extractor_return"] = [acts]

    active = classify_avoid_continuation(
        "one",
        " two",
        2,
        _fake_gemma(),
        _fake_sae(),
        sae_layer=8,
        cluster_target=_cluster_target(2, (2, 5)),
    )
    masked_sibling_excluded = classify_avoid_continuation(
        "one",
        " two",
        2,
        _fake_gemma(),
        _fake_sae(),
        sae_layer=8,
        cluster_target=_cluster_target(2, (2,)),
    )

    assert active is AvoidContinuationOutcome.ACTIVE
    assert masked_sibling_excluded is AvoidContinuationOutcome.INACTIVE


def test_classify_active_above_threshold(stub_qc_env):
    # prefix "one two" (2 tokens), continuation " three" (1 token); +1 for the
    # injected BOS at row 0 -> 4 rows total. Continuation is row 3.
    acts = t.zeros(4, 16)
    acts[3, 5] = 0.5  # continuation position
    stub_qc_env["extractor_return"] = [acts]
    out = classify_avoid_continuation(
        "one two", " three", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8, extraction_threshold=0.0,
    )
    assert out is AvoidContinuationOutcome.ACTIVE


def test_classify_inactive_when_zero(stub_qc_env):
    stub_qc_env["extractor_return"] = [t.zeros(4, 16)]
    out = classify_avoid_continuation(
        "one two", " three", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8, extraction_threshold=0.0,
    )
    assert out is AvoidContinuationOutcome.INACTIVE


def test_classify_threshold_is_strict(stub_qc_env):
    acts = t.zeros(4, 16)
    acts[3, 5] = 0.5
    stub_qc_env["extractor_return"] = [acts]
    out = classify_avoid_continuation(
        "one two", " three", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8, extraction_threshold=0.5,
    )
    assert out is AvoidContinuationOutcome.INACTIVE


def test_classify_only_prefix_activation_inactive(stub_qc_env):
    acts = t.zeros(4, 16)
    acts[0, 5] = 3.0  # injected BOS row -- must be excluded
    acts[1, 5] = 2.0
    acts[2, 5] = 1.0
    stub_qc_env["extractor_return"] = [acts]
    out = classify_avoid_continuation(
        "one two", " three", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8, extraction_threshold=0.0,
    )
    assert out is AvoidContinuationOutcome.INACTIVE


def test_classify_no_tokens_when_continuation_empty(stub_qc_env):
    out = classify_avoid_continuation(
        "one two", "", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8, extraction_threshold=0.0,
    )
    assert out is AvoidContinuationOutcome.NO_TOKENS
    # No SAE forward pass when the continuation contributes no tokens.
    assert stub_qc_env["extractor_calls"] == []


def test_classify_no_tokens_when_within_prefix(monkeypatch):
    # Seam-merge tokenizer: prefix "alpha" -> [1]; "alpha beta" -> [3,4] (adds);
    # but a continuation that re-tokenizes to the same length as the prefix
    # yields NO_TOKENS. Use the dedicated zero-add fake.
    class _NoAddTokenizer:
        bos_token_id = 999

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            # prefix and prefix+continuation both map to the SAME 2 tokens.
            return [1, 2]
        def decode(self, token_ids):
            return " ".join(str(i) for i in token_ids)
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *a, **k: _NoAddTokenizer())
    calls: list[Any] = []
    monkeypatch.setattr(qc, "_run_batched_extractor", lambda **kw: calls.append(kw) or [])
    out = classify_avoid_continuation(
        "ab", "c", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(), sae_layer=8,
    )
    assert out is AvoidContinuationOutcome.NO_TOKENS
    assert calls == []  # no extractor call


def test_classify_respects_seam_retok_tail_slice(monkeypatch):
    """AvoidContinuationOutcome checks only the combined-token tail after seam
    merges, excluding the injected BOS (row 0) and the prefix (row 1)."""
    acts = t.zeros(4, 16)
    acts[0, 5] = 1.0  # injected BOS row -- must be excluded
    acts[1, 5] = 1.0  # prefix row ("alpha") -- must be excluded
    acts[3, 5] = 0.9  # continuation row -- fires ACTIVE

    captured: dict[str, Any] = {}

    def _extractor(**kwargs):
        captured["token_id_lists"] = kwargs["token_id_lists"]
        return [acts]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *a, **k: _SeamMergeTokenizer())
    monkeypatch.setattr(qc, "_run_batched_extractor", _extractor)
    out = classify_avoid_continuation(
        "alpha", " beta gamma", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(),
        sae_layer=8, extraction_threshold=0.5,
    )
    assert captured["token_id_lists"] == [[_SeamMergeTokenizer.bos_token_id, 1, 30, 31]]
    assert out is AvoidContinuationOutcome.ACTIVE


def test_classify_raises_on_empty_both(stub_qc_env):
    with pytest.raises(AxBenchBaseError, match="zero tokens"):
        classify_avoid_continuation(
            "", "", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(), sae_layer=8,
        )


def test_classify_tokenizes_once_and_runs_sae_once(monkeypatch):
    tok_calls: list[tuple[str, str]] = []
    ext_calls: list[Any] = []

    def _fake_tok(tokenizer_id=None):
        return _WordTokenizer()

    def _counting_tokenize(prefix, cont, *, tokenizer_id=qc.DEFAULT_EXTRACTION_TOKENIZER_ID):
        tok_calls.append((prefix, cont))
        tok = _WordTokenizer()
        ptok = tok.encode(prefix)
        ftok = tok.encode(prefix + cont)
        # Mirror the real _tokenize_prefix_and_full contract: full_tokens
        # carries exactly one leading BOS at position 0.
        return ptok, [_WordTokenizer.bos_token_id, *ftok]

    monkeypatch.setattr(qc, "_tokenize_prefix_and_full", _counting_tokenize)
    monkeypatch.setattr(
        qc, "_run_batched_extractor",
        lambda **kw: ext_calls.append(kw) or [t.zeros(len(kw["token_id_lists"][0]), 16)],
    )
    classify_avoid_continuation(
        "one two", " three", f_c=5, base_model=_fake_gemma(), base_sae=_fake_sae(), sae_layer=8,
    )
    assert len(tok_calls) == 1   # tokenized exactly once (the issue-513 fix)
    assert len(ext_calls) == 1   # one SAE forward pass when tokens are added


def test_classify_releases_cuda_after_sae_forward(stub_qc_env, monkeypatch):
    releases: list[None] = []
    stub_qc_env["extractor_return"] = [t.zeros(4, 16)]
    monkeypatch.setattr(
        qc, "release_cuda_iteration_memory", lambda: releases.append(None),
    )

    classify_avoid_continuation(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(), sae_layer=8,
    )

    assert len(releases) == 1


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    prefix_words=st.lists(st.sampled_from(["a", "b", "FIRE"]), min_size=0, max_size=4),
    cont_words=st.lists(st.sampled_from(["a", "b", "FIRE", "MERGE"]), min_size=0, max_size=4),
    threshold=st.sampled_from([0.0, 0.5, 0.9]),
)
def test_classify_equivalent_to_old_two_function_combo(
    prefix_words, cont_words, threshold, monkeypatch,
):
    FIRE_ID = 9_000_001
    BOS_ID = 999

    class _FireMergeTokenizer:
        bos_token_id = BOS_ID

        def __init__(self):
            self._vocab: dict[str, int] = {}
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            ids = []
            for w in text.split():
                if w == "MERGE":
                    continue
                ids.append(FIRE_ID if w == "FIRE" else self._vocab.setdefault(w, len(self._vocab) + 2))
            return ids
        def decode(self, token_ids):
            return " ".join(str(i) for i in token_ids)

    def _extractor(**kwargs):
        out = []
        for tokens in kwargs["token_id_lists"]:
            acts = t.zeros(len(tokens), 64, dtype=t.float32)
            for i, tok in enumerate(tokens):
                if tok == FIRE_ID:
                    acts[i, :] = 1.0
            out.append(acts)
        return out

    tok = _FireMergeTokenizer()
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *a, **k: tok)
    monkeypatch.setattr(qc, "_run_batched_extractor", _extractor)

    prefix = " ".join(prefix_words)
    # A leading space before the continuation's own words (when non-empty)
    # guarantees a word boundary at the prefix/continuation seam -- mirroring
    # the file's real-Gemma-like convention (see _WordTokenizer's docstring):
    # the combined tokenization starts with the prefix's own tokens. Without
    # the leading space, a word-level tokenizer like this one would fuse the
    # prefix's last word and the continuation's first word into a single unseen
    # "word" (a seam merge) -- which QC now TOLERATES (routes to NO_TOKENS, no
    # raise), but which would make this property test's clean prefix/output
    # boundary ambiguous, so the space keeps the boundary well-defined here.
    cont = (" " + " ".join(cont_words)) if cont_words else ""
    ptok = tok.encode(prefix)
    ftok = tok.encode(prefix + cont)
    assume(len(ftok) > 0)  # empty-both raises; covered separately

    f_c = 5
    model, sae = _fake_gemma(), _fake_sae()
    adds = len(ftok) > len(ptok)
    outcome = classify_avoid_continuation(
        prefix, cont, f_c, model, sae, sae_layer=8, extraction_threshold=threshold,
    )
    if not adds:
        assert outcome is AvoidContinuationOutcome.NO_TOKENS
    else:
        fc_active_ref = fc_active_on_continuation(
            prefix, cont, f_c, model, sae, sae_layer=8, extraction_threshold=threshold,
        )
        expected = (
            AvoidContinuationOutcome.ACTIVE if fc_active_ref
            else AvoidContinuationOutcome.INACTIVE
        )
        assert outcome is expected


# ---------------------------------------------------------------------------
# compute_fc_percentiles
# ---------------------------------------------------------------------------


def test_percentile_delta_returns_pair(stub_qc_env):
    """Returns a (seed_pct, out_pct) tuple of floats."""
    # prefix 2 tokens with fc=5 activation 0.3 max (row 1, after the +1 BOS
    # offset); continuation 1 token with fc=5 activation 0.8 (row 3).
    acts = t.zeros(4, 16)
    acts[1, 5] = 0.3
    acts[3, 5] = 0.8
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    seed_pct, out_pct = compute_fc_percentiles(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
    )
    assert isinstance(seed_pct, float)
    assert isinstance(out_pct, float)
    # With identity percentile table, a magnitude of 0.3 maps to ~30,
    # magnitude 0.8 maps to ~80 (interpolated).
    assert 25.0 <= seed_pct <= 35.0
    assert 75.0 <= out_pct <= 85.0
    # Delta is positive, as expected for a concept-eliciting continuation.
    assert out_pct > seed_pct


def test_output_only_percentile_is_exactly_pair_api_output(stub_qc_env):
    """The new Stage 1b seam must preserve the established output oracle."""

    acts = t.zeros(4, 16)
    acts[0, 5] = 0.99  # BOS decoy
    acts[1, 5] = 0.3
    acts[2, 5] = 0.1
    acts[3, 5] = 0.8
    stub_qc_env["extractor_return"] = [acts]
    base_model = _fake_gemma()
    base_sae = _fake_sae()
    percentile_table = _identity_percentile_table()
    pair_seed, pair_output = compute_fc_percentiles(
        "one two",
        " three",
        5,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=8,
    )
    output_only = output_fc_percentile_for_continuation(
        "one two",
        " three",
        5,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=8,
    )

    assert pair_seed != pair_output
    assert output_only == pair_output
    assert len(stub_qc_env["extractor_calls"]) == 2


def test_output_only_percentile_does_not_reduce_prefix(
    stub_qc_env,
    monkeypatch,
):
    """Stage 1b must not silently recompute or replace its cached baseline."""

    acts = t.zeros(4, 16)
    acts[1, 5] = 0.3
    acts[3, 5] = 0.8
    stub_qc_env["extractor_return"] = [acts]
    reduced_vectors: list[t.Tensor] = []
    original = qc._max_percentile

    def _recording_max(
        activations: t.Tensor,
        f_c: int,
        percentile_table: t.Tensor,
    ) -> float:
        reduced_vectors.append(activations.detach().clone())
        return original(activations, f_c, percentile_table)

    monkeypatch.setattr(qc, "_max_percentile", _recording_max)

    output = output_fc_percentile_for_continuation(
        "one two",
        " three",
        f_c=5,
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
    )

    assert 75.0 <= output <= 85.0
    assert len(reduced_vectors) == 1
    assert t.equal(reduced_vectors[0], acts[3:, 5])


def test_output_only_percentile_uses_nonrepresentative_cluster_member(
    stub_qc_env,
):
    """Clustered Stage 1b still aggregates each allowed member's percentile."""

    acts = t.zeros(4, 16)
    acts[1, 5] = 0.4
    acts[3, 7] = 0.8
    stub_qc_env["extractor_return"] = [acts]

    output = output_fc_percentile_for_continuation(
        "one two",
        " three",
        f_c=5,
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        cluster_target=_cluster_target(5, (5, 7)),
    )

    assert 75.0 <= output <= 85.0


def test_output_only_percentile_forwards_release_policy(
    stub_qc_env,
    monkeypatch,
):
    releases: list[None] = []
    stub_qc_env["extractor_return"] = [t.zeros(4, 16)]
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )

    output_fc_percentile_for_continuation(
        "one two",
        " three",
        f_c=5,
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        release_memory=False,
    )
    assert releases == []

    output_fc_percentile_for_continuation(
        "one two",
        " three",
        f_c=5,
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        release_memory=True,
    )
    assert releases == [None]


def test_percentile_delta_forwards_sae_layer(stub_qc_env):
    """sae_layer kwarg is forwarded to the batched extractor."""
    acts = t.zeros(4, 16)
    acts[1, 5] = 0.3
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    compute_fc_percentiles(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=12,
    )
    assert stub_qc_env["extractor_calls"][-1]["layer"] == 12


def test_percentile_delta_uses_cluster_target_across_members(stub_qc_env):
    """compute_fc_percentiles wires ``cluster_target`` through to the
    per-member own-percentile amax path (not just the standalone helper unit
    tests) -- a signal on a NON-representative member is still picked up,
    proving the legacy ``f_c``-only indexing (``bpa.prefix()[:, f_c]``) is
    not silently used once ``cluster_target`` is supplied.
    """
    # Same token layout as test_percentile_delta_returns_pair, but the
    # activation lives on member 7, not the representative f_c=5.
    acts = t.zeros(4, 16)
    acts[1, 7] = 0.3  # prefix row (member 7)
    acts[3, 7] = 0.8  # continuation row (member 7)
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()

    seed_pct, out_pct = compute_fc_percentiles(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
        cluster_target=_cluster_target(5, (5, 7)),
    )

    assert 25.0 <= seed_pct <= 35.0
    assert 75.0 <= out_pct <= 85.0


def test_percentile_delta_handles_empty_prefix(stub_qc_env):
    """Empty prefix -> seed_pct = 0.0, out_pct from continuation only."""
    # n_prefix=0; full_tokens = [BOS, <one content token>] -> 2 rows. The
    # continuation occupies row 1 (acts[1 + 0:]), row 0 (BOS) is excluded.
    acts = t.zeros(2, 16)
    acts[1, 5] = 0.5
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    seed_pct, out_pct = compute_fc_percentiles(
        "", "one", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
    )
    assert seed_pct == 0.0
    assert 45.0 <= out_pct <= 55.0


def test_compute_fc_percentiles_raises_on_input_longer_than_n_ctx(stub_qc_env):
    """#602 finding 2 (LOUD, not graceful): an input longer than n_ctx would be
    silently front-clamped by the SAE forward, scoring a TRUNCATED continuation.
    QC must fail loudly rather than score a partial view. (Never fires on real
    synth data; the loud raise surfaces the anomaly if a misconfig ever trips it.)
    """
    # n_ctx=4: prefix "a b c" (3 tokens) + continuation " d e" (2) -> full_tokens
    # = [BOS,1..5] = 6 tokens > n_ctx=4.
    with pytest.raises(AxBenchBaseError, match="> n_ctx=4"):
        compute_fc_percentiles(
            "a b c", " d e", f_c=5,
            base_model=_fake_gemma(n_ctx=4), base_sae=_fake_sae(),
            percentile_table=_identity_percentile_table(),
            sae_layer=8,
        )


def test_compute_fc_percentiles_raises_on_strict_seam_over_merge(monkeypatch):
    """#602 finding 4 (LOUD): a STRICT seam over-merge (combined tokenization
    SHORTER than the standalone prefix, n_prefix > len(full_ids)) is a nonsensical
    boundary. from_forward's range assert RAISES rather than silently scoring the
    empty completion as 0.0. (Near-impossible for an appended non-empty
    continuation, but surfaced loudly if it ever occurs.)
    """

    class _OverMergeTokenizer:
        """Standalone prefix is long; the combined string (carries 'CONT')
        collapses to a single token, shorter than the prefix."""

        bos_token_id = 999

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if "CONT" in text:
                return [7]
            return [i + 1 for i, _ in enumerate(text.split())]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    def _extractor(**kwargs):
        return [t.zeros(len(kwargs["token_id_lists"][0]), 16)]

    monkeypatch.setattr(
        qc, "get_extraction_tokenizer", lambda *a, **k: _OverMergeTokenizer(),
    )
    monkeypatch.setattr(qc, "_run_batched_extractor", _extractor)

    # n_prefix=3, full_ids=[7] (len 1), full_tokens=[BOS,7] (len 2); n_ctx=128 so
    # no clamp -> from_forward's n_prefix range assert (3 > 2-1) fires.
    with pytest.raises(AxBenchBaseError, match="n_prefix=3 out of range"):
        compute_fc_percentiles(
            "p1 p2 p3", "CONT", f_c=5,
            base_model=_fake_gemma(n_ctx=128), base_sae=_fake_sae(),
            percentile_table=_identity_percentile_table(),
            sae_layer=8,
        )


def test_classify_avoid_continuation_raises_on_input_longer_than_n_ctx(stub_qc_env):
    """#602 finding 2 (LOUD): classify must fail loudly on an input the SAE
    forward would clamp, not classify a truncated continuation.
    """
    # prefix "a b c" (3) + " d e" (2) -> full_tokens = 6 > n_ctx=4. The NO_TOKENS
    # guard (1+n_prefix >= len -> 4 >= 6) is False, so the n_ctx guard is reached.
    with pytest.raises(AxBenchBaseError, match="> n_ctx=4"):
        classify_avoid_continuation(
            "a b c", " d e", f_c=5,
            base_model=_fake_gemma(n_ctx=4), base_sae=_fake_sae(),
            sae_layer=8, extraction_threshold=0.0,
        )


def test_percentile_delta_respects_seam_retok_tail_slice(monkeypatch):
    """Continuation percentiles come from the combined-token tail (excluding
    the injected BOS at row 0), not naive concat of separate encodings."""
    acts = t.zeros(4, 16)
    acts[0, 5] = 0.99  # injected BOS row -- extreme decoy, must be excluded
    acts[1, 5] = 0.2   # prefix row ("alpha")
    acts[2, 5] = 0.4   # continuation row 1
    acts[3, 5] = 0.8   # continuation row 2 (max)

    captured: dict[str, Any] = {}

    def _extractor(**kwargs):
        captured["token_id_lists"] = kwargs["token_id_lists"]
        return [acts]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *args, **kwargs: _SeamMergeTokenizer())
    monkeypatch.setattr(qc, "_run_batched_extractor", _extractor)
    table = _identity_percentile_table()
    seed_pct, out_pct = compute_fc_percentiles(
        "alpha", " beta gamma", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
    )
    assert captured["token_id_lists"] == [[_SeamMergeTokenizer.bos_token_id, 1, 30, 31]]
    # If the BOS decoy (0.99 -> ~99th pct) leaked into either span, these
    # tight ranges would fail -- the exact off-by-one regression guard.
    assert 15.0 <= seed_pct <= 25.0
    assert 75.0 <= out_pct <= 85.0



def test_percentile_delta_handles_empty_continuation(stub_qc_env):
    """Empty continuation -> out_pct = 0.0, seed_pct from prefix only."""
    # n_prefix=2; full_tokens = [BOS, tok0, tok1] -> 3 rows. Prefix occupies
    # rows 1:3 (acts[1:1+2]); the continuation slice acts[3:] is empty.
    acts = t.zeros(3, 16)
    acts[1, 5] = 0.4
    acts[2, 5] = 0.7
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    seed_pct, out_pct = compute_fc_percentiles(
        "one two", "", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
    )
    # With empty continuation, the full-tokens list is just the prefix,
    # so n_prefix == len(full_tokens) -> continuation slice is empty.
    assert out_pct == 0.0
    assert 65.0 <= seed_pct <= 75.0


def test_percentile_delta_raises_on_empty_both(stub_qc_env):
    table = _identity_percentile_table()
    with pytest.raises(AxBenchBaseError, match="zero tokens"):
        compute_fc_percentiles(
            "", "", f_c=5,
            base_model=_fake_gemma(), base_sae=_fake_sae(),
            percentile_table=table,
            sae_layer=8,
        )


def test_percentile_delta_raises_on_empty_extractor_return(stub_qc_env):
    stub_qc_env["extractor_return"] = []
    table = _identity_percentile_table()
    with pytest.raises(AxBenchBaseError, match="no activation tensors"):
        compute_fc_percentiles(
            "one", " two", f_c=5,
            base_model=_fake_gemma(), base_sae=_fake_sae(),
            percentile_table=table,
            sae_layer=8,
        )


def test_percentile_delta_releases_cuda_after_sae_forward(
    stub_qc_env, monkeypatch,
):
    releases: list[None] = []
    stub_qc_env["extractor_return"] = [t.zeros(4, 16)]
    monkeypatch.setattr(
        qc, "release_cuda_iteration_memory", lambda: releases.append(None),
    )

    compute_fc_percentiles(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
    )

    assert len(releases) == 1


def test_percentile_delta_can_defer_cuda_release(
    stub_qc_env, monkeypatch,
):
    releases: list[None] = []
    stub_qc_env["extractor_return"] = [t.zeros(4, 16)]
    monkeypatch.setattr(
        qc, "release_cuda_iteration_memory", lambda: releases.append(None),
    )

    compute_fc_percentiles(
        "one two", " three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        release_memory=False,
    )

    assert releases == []


# ---------------------------------------------------------------------------
# fc_percentile_on_prefix
# ---------------------------------------------------------------------------


def test_fc_percentile_on_prefix_returns_max_pct(stub_qc_env):
    # "one two three" (3 tokens) + injected BOS at row 0 -> 4 rows.
    acts = t.zeros(4, 16)
    acts[1, 5] = 0.3
    acts[3, 5] = 0.6
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    pct = fc_percentile_on_prefix(
        "one two three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
    )
    assert 55.0 <= pct <= 65.0


def test_fc_percentile_on_prefix_forwards_sae_layer(stub_qc_env):
    """sae_layer kwarg is forwarded to the batched extractor."""
    acts = t.zeros(4, 16)
    acts[1, 5] = 0.3
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    fc_percentile_on_prefix(
        "one two three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=12,
    )
    assert stub_qc_env["extractor_calls"][-1]["layer"] == 12


def test_fc_percentile_on_prefix_excludes_bos(stub_qc_env):
    """The forwarded model_input carries exactly one leading BOS, and the
    percentile pool excludes that row (acts[1:]) -- an extreme BOS-row decoy
    must not influence the result."""
    acts = t.zeros(4, 16)
    acts[0, 5] = 0.99  # injected BOS row -- extreme decoy, must be excluded
    acts[1, 5] = 0.1
    acts[2, 5] = 0.3
    acts[3, 5] = 0.6
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    pct = fc_percentile_on_prefix(
        "one two three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
    )
    assert stub_qc_env["extractor_calls"][-1]["token_id_lists"] == [
        [_WordTokenizer.bos_token_id, 1, 2, 3]
    ]
    # max over rows 1..3 is 0.6 -> ~60th pct; if the BOS decoy (0.99) leaked
    # in, this would jump to ~99, failing the tight range.
    assert 55.0 <= pct <= 65.0


def test_fc_percentile_on_prefix_uses_cluster_target_across_members(stub_qc_env):
    """fc_percentile_on_prefix wires ``cluster_target`` through to
    :func:`_max_percentile_for_target` -- a signal on a NON-representative
    member (with a BOS-row decoy still correctly excluded) is picked up,
    proving the legacy ``f_c``-only indexing (``bpa.content()[:, f_c]``) is
    not silently used once ``cluster_target`` is supplied.
    """
    acts = t.zeros(4, 16)
    acts[0, 7] = 0.99  # injected BOS row decoy on the member column
    acts[1, 7] = 0.1
    acts[2, 7] = 0.3
    acts[3, 7] = 0.6  # member 7, not representative f_c=5
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()

    pct = fc_percentile_on_prefix(
        "one two three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
        cluster_target=_cluster_target(5, (5, 7)),
    )

    # max over non-BOS rows is 0.6 -> ~60th pct; if either the BOS decoy
    # leaked in, or cluster_target were silently ignored (f_c=5's own column
    # never fires), this would fail the tight range (jump to ~99 or drop to 0).
    assert 55.0 <= pct <= 65.0


def test_fc_percentile_on_prefix_empty_text_raises(stub_qc_env):
    """An empty corpus-seed prefix RAISES (consistent with get_many) without
    ever calling the extractor."""
    table = _identity_percentile_table()
    with pytest.raises(
        AxBenchBaseError, match="a corpus seed should never be empty"
    ):
        fc_percentile_on_prefix(
            "", f_c=5,
            base_model=_fake_gemma(), base_sae=_fake_sae(),
            percentile_table=table,
            sae_layer=8,
        )
    assert stub_qc_env["extractor_calls"] == []


def test_fc_percentile_on_prefix_empty_tokens_raises(stub_qc_env):
    """A non-empty string the tokenizer maps to [] RAISES (a degenerate corpus
    seed) without ever calling the extractor."""
    class _EmptyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return []
    stub_qc_env["tokenizer"] = _EmptyTokenizer()
    table = _identity_percentile_table()
    with pytest.raises(AxBenchBaseError, match="tokenized to zero tokens"):
        fc_percentile_on_prefix(
            "whitespace only", f_c=5,
            base_model=_fake_gemma(), base_sae=_fake_sae(),
            percentile_table=table,
            sae_layer=8,
        )
    assert stub_qc_env["extractor_calls"] == []


def test_fc_percentile_on_prefix_releases_cuda_after_sae_forward(
    stub_qc_env, monkeypatch,
):
    releases: list[None] = []
    stub_qc_env["extractor_return"] = [t.zeros(4, 16)]
    monkeypatch.setattr(
        qc, "release_cuda_iteration_memory", lambda: releases.append(None),
    )

    fc_percentile_on_prefix(
        "one two three", f_c=5,
        base_model=_fake_gemma(), base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
    )

    assert len(releases) == 1


# ---------------------------------------------------------------------------
# PrefixPercentileCache
# ---------------------------------------------------------------------------


def test_prefix_percentile_cache_key_invalidation_inputs():
    """Key construction includes tokens, feature, model, SAE, layer, tokenizer, stats."""
    base = make_prefix_percentile_cache_key(
        token_ids=[1, 2, 3],
        f_c=5,
        base_model=_fake_gemma(model_name="model-a", n_ctx=128),
        base_sae=_fake_sae(release="release-a"),
        sae_layer=8,
        tokenizer_id="tok-a",
        percentile_namespace="stats-a",
    )
    same = make_prefix_percentile_cache_key(
        token_ids=[1, 2, 3],
        f_c=5,
        base_model=_fake_gemma(model_name="model-a", n_ctx=128),
        base_sae=_fake_sae(release="release-a"),
        sae_layer=8,
        tokenizer_id="tok-a",
        percentile_namespace="stats-a",
    )
    assert base == same

    variants = [
        {"token_ids": [1, 2, 4]},
        {"f_c": 6},
        {"base_model": _fake_gemma(model_name="model-b", n_ctx=128)},
        {"base_model": _fake_gemma(model_name="model-a", n_ctx=256)},
        {"base_sae": _fake_sae(release="release-b")},
        {"base_sae": _fake_sae(hook_name="blocks.9.hook_resid_post")},
        {"base_sae": SimpleNamespace(
            cfg=SimpleNamespace(
                hook_name="blocks.8.hook_resid_post",
                release="release-a",
                sae_id="explicit-sae-id",
                d_sae=16,
            ),
        )},
        {"base_sae": SimpleNamespace(
            cfg=SimpleNamespace(
                hook_name="blocks.8.hook_resid_post",
                metadata=_IndexOnlyMetadata({"release": "release-a", "sae_id": "meta-sae-id", "d_sae": 16}),
            ),
        )},
        {"base_sae": SimpleNamespace(
            cfg=SimpleNamespace(
                hook_name="blocks.8.hook_resid_post",
                release="release-a",
                n_features=32,
            ),
        )},
        {"sae_layer": 9},
        {"tokenizer_id": "tok-b"},
        {"percentile_namespace": "stats-b"},
    ]
    for override in variants:
        kwargs = {
            "token_ids": [1, 2, 3],
            "f_c": 5,
            "base_model": _fake_gemma(model_name="model-a", n_ctx=128),
            "base_sae": _fake_sae(release="release-a"),
            "sae_layer": 8,
            "tokenizer_id": "tok-a",
            "percentile_namespace": "stats-a",
        }
        kwargs.update(override)
        assert make_prefix_percentile_cache_key(**kwargs) != base


def test_get_many_injects_bos_and_excludes_it(monkeypatch):
    """get_many prepends exactly one BOS per prefix and excludes that row
    from the percentile pool -- an extreme BOS-row decoy must not leak in."""
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer()}
    captured: dict[str, Any] = {}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        captured["token_id_lists"] = [
            list(tokens) for tokens in kwargs["token_id_lists"]
        ]
        out = []
        for tokens in kwargs["token_id_lists"]:
            acts = t.zeros(len(tokens), 16)
            acts[0, 5] = 0.99  # injected BOS row decoy -- must be excluded
            for row_idx in range(1, len(tokens)):
                acts[row_idx, 5] = 0.4
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    cache = PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=16),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )

    [pct] = cache.get_many(["alpha beta"], 5)
    assert captured["token_id_lists"] == [[_ContentTokenizer.bos_token_id, 1, 2]]
    # max over the non-BOS rows is 0.4 -> ~40th pct; if the BOS decoy (0.99)
    # leaked in, this would jump to ~99, failing the tight range.
    assert 35.0 <= pct <= 45.0


def test_get_many_uses_cluster_target_across_members(monkeypatch):
    """get_many wires ``cluster_target`` through to
    :func:`_max_percentiles_for_target` (not just the standalone helper unit
    tests) -- a signal on a NON-representative member is still picked up,
    proving the legacy ``f_c``-only column indexing is not silently used when
    ``cluster_target`` is supplied. Also re-confirms the BOS-row exclusion
    (``acts[1:]``) is shared by both branches: it is sliced BEFORE the
    ``cluster_target is None`` dispatch.
    """
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer()}
    captured: dict[str, Any] = {}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        captured["token_id_lists"] = [
            list(tokens) for tokens in kwargs["token_id_lists"]
        ]
        out = []
        for tokens in kwargs["token_id_lists"]:
            acts = t.zeros(len(tokens), 16)
            # Decoy on the MEMBER column at the BOS row -- must be excluded.
            acts[0, 7] = 0.99
            for row_idx in range(1, len(tokens)):
                acts[row_idx, 7] = 0.4  # signal on a non-representative member
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    cache = PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=16),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )

    [pct] = cache.get_many(
        ["alpha beta"], 5, cluster_target=_cluster_target(5, (5, 7)),
    )
    assert captured["token_id_lists"] == [[_ContentTokenizer.bos_token_id, 1, 2]]
    # f_c=5's own column never fires here; only member 7 does. If get_many
    # silently ignored cluster_target and scored column f_c=5 alone, this
    # would be 0.0 instead.
    assert 35.0 <= pct <= 45.0


def test_get_many_raises_on_missing_bos(monkeypatch):
    class _NoBosContentTokenizer:
        def __init__(self) -> None:
            self.vocab: dict[str, int] = {}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            ids: list[int] = []
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab) + 1
                ids.append(self.vocab[word])
            return ids

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _NoBosContentTokenizer())
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(AxBenchBaseError, match="no BOS id"):
        cache.get_many(["alpha beta"], 5)


def test_prefix_percentile_cache_hits_dedupes_and_preserves_order(monkeypatch):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer(), "calls": []}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        token_lists = kwargs["token_id_lists"]
        state["calls"].append([list(tokens) for tokens in token_lists])
        out = []
        for tokens in token_lists:
            acts = t.zeros(len(tokens), 16)
            # A deterministic positive magnitude per text, from the
            # non-BOS (row 0 is always the injected BOS) tokens only --
            # the large BOS id would otherwise saturate every text to the
            # identity table's "above p99" ceiling, collapsing this
            # test's distinctness assertion below.
            acts[:, 5] = float(sum(tokens[1:])) / 100.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    cache = PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=4),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
        batch_size=8,
    )

    first = cache.get_many(["alpha beta", "alpha beta", "gamma delta"], 5)
    assert len(first) == 3
    assert first[0] == first[1]
    assert first[0] != first[2]
    # Duplicate text in the same call is one cache miss/fill, not two.
    assert len(cache._values) == 2
    assert len(state["calls"]) == 1
    assert len(state["calls"][0]) == 2

    second = cache.get_many(["gamma delta", "alpha beta"], 5)
    assert second == [first[2], first[0]]
    # Second call was fully served from cache.
    assert len(state["calls"]) == 1


def test_prefix_percentile_cache_releases_cuda_after_each_fill_chunk(monkeypatch):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer(), "calls": []}
    releases: list[None] = []

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        token_lists = kwargs["token_id_lists"]
        state["calls"].append([list(tokens) for tokens in token_lists])
        return [
            t.zeros((len(tokens), 16), dtype=t.float32)
            for tokens in token_lists
        ]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    monkeypatch.setattr(
        qc, "release_cuda_iteration_memory", lambda: releases.append(None),
    )

    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
        batch_size=2,
    )

    assert cache.get_many(["alpha", "beta", "gamma"], 5) == [0.0, 0.0, 0.0]
    assert [len(call) for call in state["calls"]] == [2, 1]
    assert len(releases) == 2


def test_prefix_percentile_cache_loads_configured_tokenizer_id(monkeypatch):
    seen: list[str] = []

    def _fake_tokenizer(tokenizer_id):
        seen.append(tokenizer_id)
        return _ContentTokenizer()

    def _fake_extractor(**kwargs):
        return [
            t.zeros((len(tokens), 16), dtype=t.float32)
            for tokens in kwargs["token_id_lists"]
        ]

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="custom/cache-tokenizer",
        percentile_namespace="stats-cache",
    )

    assert cache.get_many(["alpha beta"], 5) == [0.0]
    assert seen == ["custom/cache-tokenizer"]


def test_prefix_percentile_cache_invalidates_by_feature(monkeypatch):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer(), "calls": 0}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        state["calls"] += 1
        out = []
        for tokens in kwargs["token_id_lists"]:
            acts = t.zeros(len(tokens), 16)
            acts[:, 5] = 0.2
            acts[:, 6] = 0.8
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    pct5 = cache.get_many(["same prefix"], 5)[0]
    pct6 = cache.get_many(["same prefix"], 6)[0]
    assert pct6 > pct5
    assert state["calls"] == 2
    assert len(cache._values) == 2


def test_prefix_percentile_cache_matches_individual_prefix_helper(monkeypatch):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer()}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        out = []
        for tokens in kwargs["token_id_lists"]:
            acts = t.zeros(len(tokens), 16)
            acts[:, 5] = float(sum(tokens)) / 100.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    table = _identity_percentile_table()
    texts = ["alpha beta", "gamma delta epsilon", "zeta"]

    individual = [
        fc_percentile_on_prefix(
            text, 5,
            base_model=_fake_gemma(n_ctx=8),
            base_sae=_fake_sae(),
            percentile_table=table,
            sae_layer=8,
            tokenizer_id="tok-cache",
        )
        for text in texts
    ]
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=8),
        base_sae=_fake_sae(),
        percentile_table=table,
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
        batch_size=2,
    )
    assert cache.get_many(texts, 5) == pytest.approx(individual)


def test_prefix_percentile_cache_empty_texts_raise_without_touching_model(monkeypatch):
    """An empty corpus seed RAISES — and does so WITHOUT loading the tokenizer or
    touching the model (an all-empty batch never loads a tokenizer, so the raise
    fires at the first empty seed; if the tokenizer WERE loaded, _raise_tokenizer
    would surface an AssertionError instead of the expected AxBenchBaseError)."""
    def _raise_tokenizer(*args, **kwargs):
        raise AssertionError("tokenizer should not be loaded for empty prefixes")

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _raise_tokenizer)
    cache = PrefixPercentileCache(
        base_model=object(),
        base_sae=object(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(
        AxBenchBaseError, match="a corpus seed should never be empty"
    ):
        cache.get_many(["", ""], 5)


def test_prefix_percentile_cache_empty_batch_returns_empty(monkeypatch):
    def _raise_tokenizer(*args, **kwargs):
        raise AssertionError("tokenizer should not be loaded for empty batches")

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _raise_tokenizer)
    cache = PrefixPercentileCache(
        base_model=object(),
        base_sae=object(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    assert cache.get_many([], 5) == []


def test_prefix_percentile_cache_empty_tokenization_raises(monkeypatch):
    """A non-empty seed that tokenizes to ZERO tokens (a degenerate corpus seed,
    e.g. whitespace-only) RAISES rather than short-circuiting to 0.0."""
    class _EmptyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return []

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _EmptyTokenizer())
    cache = PrefixPercentileCache(
        base_model=object(),
        base_sae=object(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(
        AxBenchBaseError, match="tokenized to zero tokens"
    ):
        cache.get_many(["nonempty but empty tokens"], 5)


def test_prefix_percentile_cache_extractor_empty_return_raises(monkeypatch):
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _ContentTokenizer())
    monkeypatch.setattr(qc, "_run_batched_extractor", lambda **kwargs: [])
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(AxBenchBaseError, match="no activation tensors"):
        cache.get_many(["will fail"], 5)


def test_prefix_percentile_cache_cardinality_mismatch_short_return_raises(monkeypatch):
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _ContentTokenizer())
    monkeypatch.setattr(
        qc, "_run_batched_extractor", lambda **kwargs: [t.zeros((2, 16))],
    )
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
        batch_size=8,
    )
    with pytest.raises(AxBenchBaseError, match="activation tensors for 2 token lists"):
        cache.get_many(["alpha beta", "gamma delta"], 5)


def test_prefix_percentile_cache_cardinality_mismatch_long_return_raises(monkeypatch):
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _ContentTokenizer())
    monkeypatch.setattr(
        qc,
        "_run_batched_extractor",
        lambda **kwargs: [t.zeros((2, 16)), t.zeros((2, 16)), t.zeros((2, 16))],
    )
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(AxBenchBaseError, match="activation tensors for 2 token lists"):
        cache.get_many(["alpha beta", "gamma delta"], 5)


def test_prefix_percentile_cache_raises_if_fill_leaves_missing_result(monkeypatch):
    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _ContentTokenizer())

    def _fake_extractor(**kwargs):
        token_lists = kwargs["token_id_lists"]
        return [t.zeros((len(tokens), 16), dtype=t.float32) for tokens in token_lists]

    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    monkeypatch.setattr(qc, "_max_percentiles_for_feature", lambda *args, **kwargs: [None])
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(AxBenchBaseError, match="failed to fill result indices"):
        cache.get_many(["alpha beta"], 5)


def test_prefix_percentile_cache_zero_value_is_cached_hit(monkeypatch):
    state: dict[str, Any] = {"calls": 0}

    monkeypatch.setattr(qc, "get_extraction_tokenizer", lambda *_: _ContentTokenizer())

    def _fake_extractor(**kwargs):
        state["calls"] += 1
        token_lists = kwargs["token_id_lists"]
        return [t.zeros((len(tokens), 16), dtype=t.float32) for tokens in token_lists]

    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    assert cache.get_many(["alpha beta"], 5) == [0.0]
    assert len(cache._values) == 1
    assert cache.get_many(["alpha beta"], 5) == [0.0]
    assert len(cache._values) == 1
    assert state["calls"] == 1


def test_prefix_percentile_cache_and_scalar_refuse_context_truncation(
    monkeypatch,
):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer(), "calls": []}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        token_lists = kwargs["token_id_lists"]
        state["calls"].append([list(tokens) for tokens in token_lists])
        out = []
        for tokens in token_lists:
            acts = t.zeros(len(tokens), 16)
            acts[:, 5] = float(sum(tokens)) / 100.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    base_model = _fake_gemma(n_ctx=4)
    base_sae = _fake_sae()
    table = _identity_percentile_table()
    long_a = "alpha beta gamma delta epsilon"
    long_b = "alpha beta gamma delta zeta"

    for text in (long_a, long_b):
        with pytest.raises(
            AxBenchBaseError,
            match=r"prepared prefix is 6 tokens including BOS > n_ctx=4",
        ):
            fc_percentile_on_prefix(
                text,
                5,
                base_model=base_model,
                base_sae=base_sae,
                percentile_table=table,
                sae_layer=8,
                tokenizer_id="tok-cache",
            )

    cache = PrefixPercentileCache(
        base_model=base_model,
        base_sae=base_sae,
        percentile_table=table,
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )
    with pytest.raises(
        AxBenchBaseError,
        match=r"prepared prefix is 6 tokens including BOS > n_ctx=4",
    ):
        cache.get_many([long_a, long_b], 5)

    assert cache._values == {}
    assert state["calls"] == []


def test_prefix_percentile_cache_accepts_exact_context_boundary(
    monkeypatch,
):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer(), "calls": []}
    monkeypatch.setattr(
        qc,
        "get_extraction_tokenizer",
        lambda tokenizer_id="tok-cache": state["tokenizer"],
    )

    def _fake_extractor(**kwargs):
        state["calls"].append(
            [list(tokens) for tokens in kwargs["token_id_lists"]]
        )
        return [
            t.zeros(len(tokens), 16)
            for tokens in kwargs["token_id_lists"]
        ]

    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)
    cache = PrefixPercentileCache(
        base_model=_fake_gemma(n_ctx=4),
        base_sae=_fake_sae(),
        percentile_table=_identity_percentile_table(),
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
    )

    assert cache.get_many(["alpha beta gamma"], 5) == [0.0]
    assert state["calls"] == [[[_ContentTokenizer.bos_token_id, 1, 2, 3]]]


def test_prefix_percentile_cache_matches_reference_across_randomized_calls(
    monkeypatch,
):
    state: dict[str, Any] = {"tokenizer": _ContentTokenizer()}

    def _fake_tokenizer(tokenizer_id="tok-cache"):
        del tokenizer_id
        return state["tokenizer"]

    def _fake_extractor(**kwargs):
        out = []
        for tokens in kwargs["token_id_lists"]:
            acts = t.zeros(len(tokens), 16)
            for feature_id in (3, 5, 7):
                acts[:, feature_id] = (
                    float(sum(tokens)) + float(feature_id)
                ) / 100.0
            out.append(acts)
        return out

    monkeypatch.setattr(qc, "get_extraction_tokenizer", _fake_tokenizer)
    monkeypatch.setattr(qc, "_run_batched_extractor", _fake_extractor)

    base_model = _fake_gemma(n_ctx=8)
    base_sae = _fake_sae()
    table = _identity_percentile_table()
    cache = PrefixPercentileCache(
        base_model=base_model,
        base_sae=base_sae,
        percentile_table=table,
        sae_layer=8,
        tokenizer_id="tok-cache",
        percentile_namespace="stats-cache",
        batch_size=3,
    )

    rng = random.Random(0)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for _ in range(24):
        feature_id = rng.choice([3, 5, 7])
        batch: list[str] = []
        for _ in range(rng.randint(1, 8)):
            # Only NON-empty seeds: an empty corpus seed now RAISES in get_many
            # (a data bug), covered by the dedicated empty-seed raise tests. This
            # equivalence oracle compares the batched vs single percentile path on
            # valid seeds.
            length = rng.randint(1, 6)
            batch.append(" ".join(rng.choice(vocab) for _ in range(length)))
        expected = [
            fc_percentile_on_prefix(
                text, feature_id,
                base_model=base_model,
                base_sae=base_sae,
                percentile_table=table,
                sae_layer=8,
                tokenizer_id="tok-cache",
            )
            for text in batch
        ]
        actual = cache.get_many(batch, feature_id)
        assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# build_percentile_table_from_stats (extraction.py addition)
# ---------------------------------------------------------------------------


def _clear_stats_derived_caches() -> None:
    extraction._stats_file_sha256_cached.cache_clear()
    cache_clear = getattr(extraction._load_per_feature_stats_cached, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    extraction._build_percentile_table_for_stats_cached.cache_clear()


def test_build_percentile_table_from_stats_caches(monkeypatch, tmp_path):
    """Repeated calls with the same (path, n_features) use the cached result."""
    del tmp_path
    call_count = {"n": 0}

    def _fake_stats_key(stats_path: str) -> tuple[str, int, int, str]:
        return stats_path, 0, 0, stats_path

    def _fake_stats_loader(stats_path, mtime_ns, size, sha256):
        del stats_path, mtime_ns, size, sha256
        call_count["n"] += 1
        return {
            str(i): SimpleNamespace(
                count=100,
                percentiles={f"p{k}": (k / 100.0) for k in range(1, 100)},
            )
            for i in range(16)
        }

    _clear_stats_derived_caches()
    monkeypatch.setattr(extraction, "_stats_file_cache_key", _fake_stats_key)
    monkeypatch.setattr(
        extraction, "_load_per_feature_stats_cached", _fake_stats_loader,
    )
    table1 = extraction.build_percentile_table_from_stats("stub_path", 16)
    table2 = extraction.build_percentile_table_from_stats("stub_path", 16)
    # Same tensor instance because lru_cache returns the cached object.
    assert table1 is table2
    # The outer lru_cache hits on the second call, so the stub loader is
    # only invoked once.
    assert call_count["n"] == 1
    assert table1.shape == (16, 99)
    # Clear cache state built from the stub loader before later tests.
    _clear_stats_derived_caches()


def test_build_percentile_table_from_stats_different_key_rebuilds(monkeypatch):
    """Different (path, n_features) keys produce distinct cached tables."""
    def _fake_stats_key(stats_path: str) -> tuple[str, int, int, str]:
        return stats_path, 0, 0, stats_path

    def _fake_loader(stats_path, mtime_ns, size, sha256):
        del stats_path, mtime_ns, size, sha256
        return {
            str(i): SimpleNamespace(
                count=100,
                percentiles={f"p{k}": (k / 100.0) for k in range(1, 100)},
            )
            for i in range(16)
        }

    _clear_stats_derived_caches()
    monkeypatch.setattr(extraction, "_stats_file_cache_key", _fake_stats_key)
    monkeypatch.setattr(
        extraction, "_load_per_feature_stats_cached", _fake_loader,
    )
    table_a = extraction.build_percentile_table_from_stats("path_a", 16)
    table_b = extraction.build_percentile_table_from_stats("path_b", 16)
    assert table_a is not table_b  # different paths -> different cache entries
    _clear_stats_derived_caches()


def test_feature_statistics_sha256_reuses_warmed_content_hash(
    tmp_path,
    monkeypatch,
):
    stats_path = tmp_path / "feature_statistics.safetensors"
    stats_path.write_bytes(b"synthetic statistics identity")

    def _fake_stats_loader(stats_path, mtime_ns, size, sha256):
        del stats_path, mtime_ns, size, sha256
        return {
            "0": SimpleNamespace(
                count=100,
                percentiles={f"p{k}": (k / 100.0) for k in range(1, 100)},
            )
        }

    _clear_stats_derived_caches()
    monkeypatch.setattr(
        extraction,
        "_load_per_feature_stats_cached",
        _fake_stats_loader,
    )

    extraction.build_percentile_table_from_stats(str(stats_path), 1)
    first = extraction.feature_statistics_file_sha256(str(stats_path))

    assert first
    cache_info = extraction._stats_file_sha256_cached.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1
    _clear_stats_derived_caches()


# ---------------------------------------------------------------------------
# _max_percentile: "feature never fires" must return 0.0, not floor to 0.5
#
# Background: magnitude_to_percentile floors any sub-p1 magnitude to 0.5
# (its "below p1" sentinel). If _max_percentile blindly forwards a
# max_mag=0 tensor to magnitude_to_percentile, it returns 0.5 — making
# a truly-silent feature indistinguishable from "fired at the very
# bottom of the observed distribution". Downstream threshold-based
# callers near 0.0 would then admit inactive continuations.
#
# These tests pin the new contract: _max_percentile returns 0.0 when
# the feature literally never fired (all-zero activations OR empty
# input), and routes to the percentile table otherwise.
# ---------------------------------------------------------------------------


def test_max_percentile_empty_input_returns_zero():
    """Empty activation tensor -> 0.0 (pre-existing behavior, regression guard)."""
    table = _identity_percentile_table()
    got = _max_percentile(t.tensor([], dtype=t.float32), f_c=3, percentile_table=table)
    assert got == 0.0


def test_max_percentile_all_zero_activations_returns_zero():
    """All-zero activations (feature didn't fire) -> 0.0.

    Without this check the call would forward max_mag=0 to
    magnitude_to_percentile and get back 0.5 — the "below p1" floor
    — indistinguishable from "fired at the very bottom of observed".
    """
    table = _identity_percentile_table()
    got = _max_percentile(
        t.zeros(64, dtype=t.float32), f_c=5, percentile_table=table,
    )
    assert got == 0.0


def test_max_percentile_mixed_zero_and_positive_uses_max():
    """A tensor of all zeros except one positive value still routes
    through the percentile table. The ``max_mag <= 0`` short-circuit
    only fires when EVERY position is zero."""
    table = _identity_percentile_table()
    # With the identity table, p_k = k/100, so magnitude 0.5 maps to
    # roughly the 50th percentile.
    acts = t.zeros(64, dtype=t.float32)
    acts[17] = 0.5
    got = _max_percentile(acts, f_c=0, percentile_table=table)
    # _identity_percentile_table has p1=0.01 .. p99=0.99 with mag 0.5
    # landing between p49 (0.49) and p50 (0.50). magnitude_to_percentile
    # uses searchsorted with right=True and linear interpolation; the
    # exact value is within [49, 51]. The key assertion is: NOT 0.0
    # (short-circuit) and NOT 0.5 (the "below p1" floor).
    assert got > 1.0  # above the below-p1 sentinel
    assert got < 99.0  # below the above-p99 sentinel


def test_max_percentile_tiny_positive_floors_to_below_p1():
    """A tiny positive max magnitude (below p1) still maps to 0.5 via
    the existing floor in magnitude_to_percentile. The zero-short-
    circuit does NOT generalize to "anything tiny" — it only fires on
    a true zero. This preserves the meaningful distinction between
    "fires at the bottom of observed" and "never fires".
    """
    table = _identity_percentile_table()
    # 1e-8 is positive but far below p1 (0.01). Fall through to the
    # magnitude_to_percentile floor.
    acts = t.zeros(64, dtype=t.float32)
    acts[0] = 1e-8
    got = _max_percentile(acts, f_c=0, percentile_table=table)
    assert got == 0.5


def test_max_percentile_negative_defensive_short_circuits():
    """SAE activations are ReLU'd in production, so a negative max
    shouldn't happen. But if one leaks in (e.g., from a bug upstream),
    return 0.0 rather than forward a nonsense magnitude to
    magnitude_to_percentile. This is a belt-and-braces check.
    """
    table = _identity_percentile_table()
    acts = t.full((64,), -0.5, dtype=t.float32)  # unambiguously negative
    got = _max_percentile(acts, f_c=0, percentile_table=table)
    assert got == 0.0


# ---------------------------------------------------------------------------
# Integration: Stage 0c verify_candidate, threshold=0.0 semantics
#
# Exercises the full stack for the issue the zero-fix resolves: with
# threshold=0.0 (raw 0.0) and strict-gt (p > threshold), a continuation
# that never activates the feature must be rejected. Before the fix
# it would have been admitted via the 0.5 floor.
# ---------------------------------------------------------------------------


def test_integration_threshold_zero_rejects_silent_continuation(tmp_path):
    """End-to-end: feed _verify_candidate a continuation where the feature
    never fires. With threshold=0.0 and the fix in place, the candidate
    must be rejected.
    """

    # Build an ACTUAL qc path (no monkey-patch of compute_fc_percentiles
    # this time) with a stub extractor that returns all-zero activations.
    # This exercises _max_percentile → magnitude_to_percentile chain.
    state: dict[str, Any] = {}

    def _fake_tokenizer(*args, **kwargs):
        del args, kwargs
        return _WordTokenizer()

    def _fake_extractor(**kwargs):
        state.setdefault("calls", 0)
        state["calls"] += 1
        n = len(kwargs["token_id_lists"][0])
        # All-zero activations across the full (prefix + continuation).
        return [t.zeros(n, 16)]

    cfg = make_valid_config(
        run_tag="test_zero_threshold",
        output_root=str(tmp_path / "eval_out"),
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        validation_shards=[0],
        n_verification_seeds=5,
        min_consistency_passes=3,
        verification_percentile_threshold=0.0,  # raw_threshold = 0.0
        verification_length=4,
        verification_prefill_mode="repeated",
    )
    candidate = _PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="one two",
        prefix_gemma_tokens=[1, 2],
    )
    # Use the identity percentile table (shape/values don't actually
    # matter here because _max_percentile short-circuits on max_mag=0
    # before consulting the table).
    table = _identity_percentile_table()
    def _fake_batched_generate(base_model, token_ids, *, seeds, **kwargs):
        assert isinstance(seeds, list), (
            f"seeds must be list, got {type(seeds).__name__}"
        )
        return [[1, 2, 3, 4] for _ in seeds]

    with (
        mock.patch.object(qc, "get_extraction_tokenizer", _fake_tokenizer),
        mock.patch.object(qc, "_run_batched_extractor", _fake_extractor),
        mock.patch.object(_si, "get_extraction_tokenizer", _fake_tokenizer),
        mock.patch.object(_si, "batched_generate_seeds", _fake_batched_generate),
    ):
        result = _verify_candidate(
            cfg, candidate, f_c=0, concept_id=0,
            base_model=_fake_gemma(),
            base_sae=_fake_sae(),
            percentile_table=table,
            sae_layer=8,
        )
    assert result is None, (
        "silent feature (all-zero activations) must be rejected at "
        "threshold=0.0 after the zero-short-circuit fix"
    )


# ---------------------------------------------------------------------------
# Batched post-provider continuation QC (issue #1191)
# ---------------------------------------------------------------------------


def test_batched_continuation_qc_empty_inputs_do_no_work(monkeypatch):
    monkeypatch.setattr(
        qc,
        "get_extraction_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty batch loaded tokenizer")
        ),
    )
    releases: list[None] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )

    assert output_fc_percentiles_for_continuations(
        [],
        object(),
        object(),
        _identity_percentile_table(),
        sae_layer=8,
    ) == []
    assert classify_avoid_continuations(
        [],
        object(),
        object(),
        sae_layer=8,
    ) == []
    assert releases == []


def test_one_item_batched_continuation_qc_matches_scalar_exactly(stub_qc_env):
    acts = t.zeros((4, 16), dtype=t.float32)
    acts[0, 5] = 0.99  # BOS decoy
    acts[1, 5] = 0.70  # prefix decoy
    acts[3, 5] = 0.80
    stub_qc_env["extractor_return"] = [acts]
    table = _identity_percentile_table()
    base_model = _fake_gemma()
    base_sae = _fake_sae()
    item = ContinuationQcInput("one two", " three", 5)

    scalar_percentile = output_fc_percentile_for_continuation(
        item.prefix_text,
        item.continuation_text,
        item.feature_id,
        base_model,
        base_sae,
        table,
        sae_layer=8,
    )
    batched_percentile = output_fc_percentiles_for_continuations(
        [item],
        base_model,
        base_sae,
        table,
        sae_layer=8,
    )[0]
    scalar_outcome = classify_avoid_continuation(
        item.prefix_text,
        item.continuation_text,
        item.feature_id,
        base_model,
        base_sae,
        sae_layer=8,
        extraction_threshold=0.8,
    )
    batched_outcome = classify_avoid_continuations(
        [item],
        base_model,
        base_sae,
        sae_layer=8,
        extraction_threshold=0.8,
    )[0]

    assert batched_percentile == scalar_percentile
    assert batched_outcome is scalar_outcome is AvoidContinuationOutcome.INACTIVE
    assert len(stub_qc_env["extractor_calls"]) == 4


def test_batched_output_percentiles_preserve_ragged_mixed_target_order(
    stub_qc_env,
    monkeypatch,
):
    releases: list[None] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )
    first = t.zeros((3, 16), dtype=t.float32)
    first[0, 2] = 0.99
    first[1, 2] = 0.90
    first[2, 2] = 0.30
    second = t.zeros((5, 16), dtype=t.float32)
    second[0, 5] = 0.99
    second[1:3, 7] = 0.95
    second[4, 7] = 0.80  # only a non-representative member fires
    third = t.zeros((6, 16), dtype=t.float32)
    third[0, 10] = 0.99
    third[1:4, 10] = 0.95
    third[5, 10] = 0.60  # third member of a differently-sized cluster
    stub_qc_env["extractor_return"] = [first, second, third]
    inputs = [
        ContinuationQcInput("one", " two", 2),
        ContinuationQcInput(
            "one two",
            " three four",
            5,
            _cluster_target(5, (5, 7)),
        ),
        ContinuationQcInput(
            "one two three",
            " four five",
            8,
            _cluster_target(8, (8, 9, 10)),
        ),
    ]

    output = output_fc_percentiles_for_continuations(
        inputs,
        _fake_gemma(),
        _fake_sae(),
        _identity_percentile_table(),
        sae_layer=12,
    )

    assert len(stub_qc_env["extractor_calls"]) == 1
    call = stub_qc_env["extractor_calls"][0]
    assert [len(tokens) for tokens in call["token_id_lists"]] == [3, 5, 6]
    assert call["layer"] == 12
    assert 25.0 <= output[0] <= 35.0
    assert 75.0 <= output[1] <= 85.0
    assert 55.0 <= output[2] <= 65.0
    assert bool(t.isfinite(t.tensor(output)).all())
    assert all(0.0 <= percentile <= 99.5 for percentile in output)
    assert releases == [None]


def test_batched_output_percentile_empty_completion_forwards_and_returns_zero(
    stub_qc_env,
):
    acts = t.zeros((3, 16), dtype=t.float32)
    acts[0, 5] = 0.99
    acts[1:, 5] = 0.80
    stub_qc_env["extractor_return"] = [acts]

    output = output_fc_percentiles_for_continuations(
        [ContinuationQcInput("one two", "", 5)],
        _fake_gemma(),
        _fake_sae(),
        _identity_percentile_table(),
        sae_layer=8,
    )

    assert output == [0.0]
    assert len(stub_qc_env["extractor_calls"]) == 1


def test_batched_strict_zero_excludes_no_token_rows_and_scatters(
    stub_qc_env,
    monkeypatch,
):
    releases: list[None] = []
    counters: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )
    monkeypatch.setattr(
        qc,
        "record_counter",
        lambda name, _value=1, **attrs: counters.append(
            (name, bool(attrs["active"]))
        ),
    )
    active_acts = t.zeros((3, 16), dtype=t.float32)
    active_acts[0, 2] = 1.0
    active_acts[1, 2] = 1.0
    active_acts[2, 2] = 0.5
    inactive_cluster_acts = t.zeros((4, 16), dtype=t.float32)
    inactive_cluster_acts[0, 7] = 1.0
    inactive_cluster_acts[1:3, 7] = 1.0
    stub_qc_env["extractor_return"] = [active_acts, inactive_cluster_acts]
    inputs = [
        ContinuationQcInput("one two", "", 5),
        ContinuationQcInput("one", " two", 2),
        ContinuationQcInput(
            "one two",
            " three",
            5,
            _cluster_target(5, (5, 7)),
        ),
    ]

    outcomes = classify_avoid_continuations(
        inputs,
        _fake_gemma(),
        _fake_sae(),
        sae_layer=8,
        extraction_threshold=0.0,
    )

    assert outcomes == [
        AvoidContinuationOutcome.NO_TOKENS,
        AvoidContinuationOutcome.ACTIVE,
        AvoidContinuationOutcome.INACTIVE,
    ]
    assert len(stub_qc_env["extractor_calls"]) == 1
    assert [
        len(tokens)
        for tokens in stub_qc_env["extractor_calls"][0]["token_id_lists"]
    ] == [3, 4]
    assert counters == [
        ("synth_qc.fc_active_checks", True),
        ("synth_qc.fc_active_checks", False),
    ]
    assert releases == [None]


def test_batched_strict_zero_stage_region_counts_only_forwarded_rows(
    stub_qc_env,
    monkeypatch,
):
    stub_qc_env["extractor_return"] = [
        t.zeros((3, 16), dtype=t.float32)
    ]
    regions: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        qc,
        "profile_region",
        lambda name, **attrs: (
            regions.append((name, attrs)) or nullcontext()
        ),
    )

    outcomes = classify_avoid_continuations(
        [
            ContinuationQcInput("one two", "", 5),
            ContinuationQcInput("one", " two", 2),
        ],
        _fake_gemma(),
        _fake_sae(),
        sae_layer=8,
        profile_region_name="stage1b.avoid_strict_zero_qc_batch",
    )

    assert outcomes == [
        AvoidContinuationOutcome.NO_TOKENS,
        AvoidContinuationOutcome.INACTIVE,
    ]
    assert [
        region for region in regions
        if region[0] == "stage1b.avoid_strict_zero_qc_batch"
    ] == [
        (
            "stage1b.avoid_strict_zero_qc_batch",
            {"rows": 1, "sae_layer": 8},
        )
    ]


def test_batched_strict_zero_all_no_tokens_skips_forward_and_release(
    stub_qc_env,
    monkeypatch,
):
    releases: list[None] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )

    outcomes = classify_avoid_continuations(
        [
            ContinuationQcInput("one", "", 2),
            ContinuationQcInput("one two", "", 5),
        ],
        _fake_gemma(n_ctx=1),
        _fake_sae(),
        sae_layer=8,
    )

    assert outcomes == [
        AvoidContinuationOutcome.NO_TOKENS,
        AvoidContinuationOutcome.NO_TOKENS,
    ]
    assert stub_qc_env["extractor_calls"] == []
    assert releases == []


def test_batched_strict_zero_threshold_is_strict_per_item(stub_qc_env):
    equal = t.zeros((3, 16), dtype=t.float32)
    equal[2, 2] = 0.5
    greater = t.zeros((3, 16), dtype=t.float32)
    greater[2, 7] = 0.5001
    stub_qc_env["extractor_return"] = [equal, greater]

    outcomes = classify_avoid_continuations(
        [
            ContinuationQcInput("one", " two", 2),
            ContinuationQcInput("one", " two", 7),
        ],
        _fake_gemma(),
        _fake_sae(),
        sae_layer=8,
        extraction_threshold=0.5,
    )

    assert outcomes == [
        AvoidContinuationOutcome.INACTIVE,
        AvoidContinuationOutcome.ACTIVE,
    ]


def test_batched_continuation_qc_validates_all_contexts_before_forward(
    stub_qc_env,
):
    inputs = [
        ContinuationQcInput("one", " two", 2),  # BOS + 2 == n_ctx
        ContinuationQcInput("one two", " three", 5),  # BOS + 3 > n_ctx
    ]
    with pytest.raises(AxBenchBaseError, match=r"input is 4 tokens > n_ctx=3"):
        output_fc_percentiles_for_continuations(
            inputs,
            _fake_gemma(n_ctx=3),
            _fake_sae(),
            _identity_percentile_table(),
            sae_layer=8,
        )
    assert stub_qc_env["extractor_calls"] == []

    with pytest.raises(AxBenchBaseError, match=r"input is 4 tokens > n_ctx=3"):
        classify_avoid_continuations(
            inputs,
            _fake_gemma(n_ctx=3),
            _fake_sae(),
            sae_layer=8,
        )
    assert stub_qc_env["extractor_calls"] == []


def test_batched_output_context_exact_limit_passes(stub_qc_env):
    stub_qc_env["extractor_return"] = [t.zeros((3, 16), dtype=t.float32)]
    assert output_fc_percentiles_for_continuations(
        [ContinuationQcInput("one", " two", 2)],
        _fake_gemma(n_ctx=3),
        _fake_sae(),
        _identity_percentile_table(),
        sae_layer=8,
    ) == [0.0]
    assert len(stub_qc_env["extractor_calls"]) == 1

    stub_qc_env["extractor_calls"].clear()
    stub_qc_env["extractor_return"] = [t.zeros((3, 16), dtype=t.float32)]
    assert classify_avoid_continuations(
        [ContinuationQcInput("one", " two", 2)],
        _fake_gemma(n_ctx=3),
        _fake_sae(),
        sae_layer=8,
    ) == [AvoidContinuationOutcome.INACTIVE]
    assert len(stub_qc_env["extractor_calls"]) == 1


def test_batched_output_cleanup_once_on_cardinality_error(
    stub_qc_env,
    monkeypatch,
):
    stub_qc_env["extractor_return"] = [t.zeros((3, 16), dtype=t.float32)]
    releases: list[None] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )
    with pytest.raises(AxBenchBaseError, match="activation tensors for 2 token lists"):
        output_fc_percentiles_for_continuations(
            [
                ContinuationQcInput("one", " two", 2),
                ContinuationQcInput("one", " two", 5),
            ],
            _fake_gemma(),
            _fake_sae(),
            _identity_percentile_table(),
            sae_layer=8,
        )
    assert releases == [None]


def test_batched_strict_zero_cleanup_once_on_invalid_cluster(
    stub_qc_env,
    monkeypatch,
):
    stub_qc_env["extractor_return"] = [
        t.zeros((3, 16), dtype=t.float32),
        t.zeros((3, 16), dtype=t.float32),
    ]
    releases: list[None] = []
    counters: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )
    monkeypatch.setattr(
        qc,
        "record_counter",
        lambda name, _value=1, **attrs: counters.append(
            (name, bool(attrs["active"]))
        ),
    )
    with pytest.raises(AxBenchBaseError, match="representative does not match"):
        classify_avoid_continuations(
            [
                ContinuationQcInput("one", " two", 2),
                ContinuationQcInput(
                    "one",
                    " two",
                    2,
                    _cluster_target(3, (3,)),
                )
            ],
            _fake_gemma(),
            _fake_sae(),
            sae_layer=8,
        )
    assert counters == [("synth_qc.fc_active_checks", False)]
    assert releases == [None]


def test_batched_output_release_memory_false_suppresses_cleanup(
    stub_qc_env,
    monkeypatch,
):
    stub_qc_env["extractor_return"] = [t.zeros((3, 16), dtype=t.float32)]
    releases: list[None] = []
    monkeypatch.setattr(
        qc,
        "release_cuda_iteration_memory",
        lambda: releases.append(None),
    )
    output = output_fc_percentiles_for_continuations(
        [ContinuationQcInput("one", " two", 2)],
        _fake_gemma(),
        _fake_sae(),
        _identity_percentile_table(),
        sae_layer=8,
        release_memory=False,
    )
    assert output == [0.0]
    assert releases == []


def test_batched_continuation_qc_missing_bos_and_zero_tokens_fail_before_forward(
    monkeypatch,
):
    class _NoBosTokenizer:
        bos_token_id = None

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [1]

    calls: list[Any] = []
    monkeypatch.setattr(
        qc,
        "get_extraction_tokenizer",
        lambda *_args, **_kwargs: _NoBosTokenizer(),
    )
    monkeypatch.setattr(
        qc,
        "_run_batched_extractor",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    with pytest.raises(AxBenchBaseError, match="exposes no BOS id"):
        output_fc_percentiles_for_continuations(
            [ContinuationQcInput("one", " two", 2)],
            _fake_gemma(),
            _fake_sae(),
            _identity_percentile_table(),
            sae_layer=8,
        )
    assert calls == []

    class _EmptyTokenizer:
        bos_token_id = 999

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return []

    monkeypatch.setattr(
        qc,
        "get_extraction_tokenizer",
        lambda *_args, **_kwargs: _EmptyTokenizer(),
    )
    with pytest.raises(AxBenchBaseError, match="tokenized to zero tokens"):
        classify_avoid_continuations(
            [ContinuationQcInput("", "", 2)],
            _fake_gemma(),
            _fake_sae(),
            sae_layer=8,
        )
    assert calls == []
