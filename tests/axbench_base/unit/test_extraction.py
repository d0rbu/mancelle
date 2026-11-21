"""Unit tests for steering.eval.axbench_base.extraction (Task 12)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

from steering.eval.axbench_base import extraction
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base._loguru_capture import capture_loguru_info

# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


def _fake_sae(
    *, hook_name: str = "blocks.8.hook_resid_post", release: str = "gemmascope-res-16k",
) -> SimpleNamespace:
    """Return a stand-in for a SAE with just the cfg attributes we need."""
    return SimpleNamespace(cfg=SimpleNamespace(hook_name=hook_name, release=release))


def _fake_gemma(n_ctx: int = 2048) -> SimpleNamespace:
    """Return a stand-in for HookedTransformer with a ``cfg.n_ctx``."""
    return SimpleNamespace(cfg=SimpleNamespace(n_ctx=n_ctx))


# ---------------------------------------------------------------------------
# Fail-fast cfg accessor (_sae_hook_name)
# ---------------------------------------------------------------------------
# A missing SAE cfg or hook_name field indicates an upstream
# bug (wrong SAE type, failed load). These helpers used to return a
# "default" fallback, which would silently collide in the extraction
# cache across distinct SAEs; they now raise AxBenchBaseError instead.


def test_sae_hook_name_raises_when_cfg_missing():
    """A SAE without a ``cfg`` attribute raises instead of falling back."""
    sae_no_cfg = SimpleNamespace()
    with pytest.raises(AxBenchBaseError, match="no .cfg attribute"):
        extraction._sae_hook_name(sae_no_cfg)


def test_sae_hook_name_raises_when_hook_name_missing():
    """A SAE cfg with neither direct hook_name nor metadata['hook_name']
    raises instead of falling back."""
    sae_empty_cfg = SimpleNamespace(cfg=SimpleNamespace())
    with pytest.raises(AxBenchBaseError, match="hook_name"):
        extraction._sae_hook_name(sae_empty_cfg)


def test_sae_hook_name_reads_from_metadata_when_direct_attr_missing():
    """SAELens 6.37+ stores hook_name on ``cfg.metadata`` (a dict-like
    ``SAEMetadata``) instead of as a direct ``cfg.hook_name`` attribute.
    The helper must fall back to the metadata path so axbench_base can
    run against the current SAELens release — without this we crash at
    the first ``_run_sae_on_full`` call in Stage 0c.
    """
    metadata = {"hook_name": "blocks.8.hook_resid_post"}
    sae = SimpleNamespace(cfg=SimpleNamespace(metadata=metadata))
    assert extraction._sae_hook_name(sae) == "blocks.8.hook_resid_post"


def test_sae_hook_name_prefers_direct_attr_over_metadata():
    """If both direct ``cfg.hook_name`` and ``cfg.metadata['hook_name']``
    are set, the direct attribute wins. Matches how older SAELens
    releases behaved and keeps the pre-metadata path intact."""
    metadata = {"hook_name": "metadata_hook"}
    sae = SimpleNamespace(
        cfg=SimpleNamespace(hook_name="direct_hook", metadata=metadata),
    )
    assert extraction._sae_hook_name(sae) == "direct_hook"


# ---------------------------------------------------------------------------
# _cfg_metadata_get — direct tests for the mapping-extraction helper
# ---------------------------------------------------------------------------


def test_cfg_metadata_get_returns_none_when_metadata_attribute_missing():
    """No ``metadata`` attribute at all → return ``None`` so the caller's
    ``getattr(cfg, 'hook_name', None) or _cfg_metadata_get(...)`` chain
    falls through to the terminal error branch."""
    cfg = SimpleNamespace()
    assert extraction._cfg_metadata_get(cfg, "anything") is None


def test_cfg_metadata_get_returns_none_when_metadata_is_none():
    """Explicit ``metadata=None`` (possible if the upstream SAE loader
    nulls the field instead of omitting it) must not raise; the helper
    returns ``None`` just like the missing-attribute case."""
    cfg = SimpleNamespace(metadata=None)
    assert extraction._cfg_metadata_get(cfg, "hook_name") is None


def test_cfg_metadata_get_uses_dotget_when_available():
    """A ``dict``-like ``metadata`` with a callable ``.get`` method takes
    the fast path at line 178-179."""
    cfg = SimpleNamespace(metadata={"hook_name": "x"})
    assert extraction._cfg_metadata_get(cfg, "hook_name") == "x"
    assert extraction._cfg_metadata_get(cfg, "missing") is None


def test_cfg_metadata_get_falls_back_to_bracket_indexing():
    """If ``metadata`` lacks ``.get`` but supports ``__getitem__`` (e.g.,
    a dataclass-style object that defines only bracket access), the
    helper still works via the try/except at lines 180-183. Pins the
    bracket-indexing contract so the fallback path has coverage."""
    class _BracketOnly:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
    cfg = SimpleNamespace(metadata=_BracketOnly({"hook_name": "bracket-value"}))
    assert extraction._cfg_metadata_get(cfg, "hook_name") == "bracket-value"


def test_cfg_metadata_get_returns_none_when_bracket_raises_keyerror():
    """Bracket-only metadata with a missing key raises ``KeyError`` at
    line 181; the helper catches it and returns ``None`` so the caller
    can treat missing-key the same as missing-attribute."""
    class _BracketOnly:
        def __getitem__(self, key):
            raise KeyError(key)
    cfg = SimpleNamespace(metadata=_BracketOnly())
    assert extraction._cfg_metadata_get(cfg, "hook_name") is None


def test_cfg_metadata_get_returns_none_when_metadata_is_non_mapping_string():
    """If a future SAELens release stores metadata as some other type
    (e.g., a string), neither ``.get`` nor ``[key]`` behaves the way we
    want. ``metadata[key]`` on a string raises ``TypeError`` when the
    key is a non-int; the helper catches it and returns ``None``.
    """
    cfg = SimpleNamespace(metadata="not a mapping")
    # ``"not a mapping".get`` is not callable, so we skip the .get path
    # and try metadata["hook_name"], which raises TypeError.
    assert extraction._cfg_metadata_get(cfg, "hook_name") is None


# ---------------------------------------------------------------------------
# _infer_n_sae_features (Fix 5: raise instead of defensive fallback)
# ---------------------------------------------------------------------------


def test_infer_n_sae_features_from_d_sae():
    """cfg.d_sae -> return as int."""
    sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=32768))
    assert extraction._infer_n_sae_features(sae) == 32768


def test_infer_n_sae_features_from_n_features():
    """cfg.n_features -> return as int (fallback path when d_sae absent)."""
    sae = SimpleNamespace(cfg=SimpleNamespace(n_features=65536))
    assert extraction._infer_n_sae_features(sae) == 65536


def test_infer_n_sae_features_prefers_d_sae_over_n_features():
    """If both attrs are present, d_sae wins (SAELens newer API)."""
    sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=32768, n_features=65536))
    assert extraction._infer_n_sae_features(sae) == 32768


def test_infer_n_sae_features_raises_when_no_cfg():
    """Missing cfg attribute raises instead of silently returning a default."""
    sae = SimpleNamespace()  # no cfg
    with pytest.raises(AxBenchBaseError, match="no .cfg attribute"):
        extraction._infer_n_sae_features(sae)


def test_infer_n_sae_features_raises_when_cfg_has_neither():
    """cfg present but missing both d_sae and n_features raises."""
    sae = SimpleNamespace(cfg=SimpleNamespace())  # cfg with neither attr
    with pytest.raises(AxBenchBaseError, match="neither d_sae nor n_features"):
        extraction._infer_n_sae_features(sae)


# ---------------------------------------------------------------------------
# _run_batched_extractor (unpatched delegation)
# ---------------------------------------------------------------------------


def test_run_batched_extractor_delegates_to_extract_features(monkeypatch):
    """Unpatched wrapper must forward its kwargs to ``_get_sae_activations_batched``.

    This test covers the wrapper's actual body (the return statement),
    which the ``stub_extraction_env`` fixture otherwise monkey-patches out.
    Patches the imported reference in the ``extraction`` module itself
    (not the source module) since ``from ... import`` binds a new name.
    """
    captured: dict[str, Any] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return [t.zeros(2, 4)]

    monkeypatch.setattr(extraction, "_get_sae_activations_batched", _capture)

    result = extraction._run_batched_extractor(
        model=_fake_gemma(),
        sae=_fake_sae(),
        token_id_lists=[[1, 2]],
        layer=8,
        hook_name="blocks.8.hook_resid_post",
    )
    assert len(result) == 1
    assert result[0].shape == (2, 4)
    assert captured["layer"] == 8
    assert captured["hook_name"] == "blocks.8.hook_resid_post"
    assert captured["token_id_lists"] == [[1, 2]]
    # No nnterp_model attribute on the stub → pass through as None so
    # the underlying extractor falls to the TL ``run_with_cache`` path.
    assert captured["nnterp_model"] is None


def test_run_batched_extractor_passes_through_nnterp_side_channel(monkeypatch):
    """If the ``model`` argument carries a ``nnterp_model`` attribute
    (as populated by
    :func:`steering.eval.axbench_base.runtime_state._load_feature_extraction_pair`),
    the wrapper forwards it so the underlying extractor tries the
    nnterp path first. Without this passthrough every call falls back
    to ``HookedTransformer.run_with_cache`` — which is slower and fails
    outright when ``model`` has been replaced by the nnterp wrapper.
    """
    captured: dict[str, Any] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return [t.zeros(3, 4)]

    monkeypatch.setattr(extraction, "_get_sae_activations_batched", _capture)

    fake_nnterp = SimpleNamespace(_name="fake-nnterp")
    tl_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=2048),
        nnterp_model=fake_nnterp,
    )

    extraction._run_batched_extractor(
        model=tl_model,
        sae=_fake_sae(),
        token_id_lists=[[1, 2, 3]],
        layer=8,
        hook_name="blocks.8.hook_resid_post",
    )
    assert captured["nnterp_model"] is fake_nnterp
    assert captured["model"] is tl_model


# ---------------------------------------------------------------------------
# BOS-prepended model input helper
# ---------------------------------------------------------------------------
# Bug 1 (#602): every SAE-forward site must inject exactly one leading BOS and
# exclude it from pooling, consistent with steering/extract_features.py. These
# tests pin the shared helper's BOS-prepend, front-clamp, and in-data-special
# warn+preserve contracts.


class _BosAwareTokenizer:
    """Whitespace tokenizer that exposes a BOS id (like a real Gemma tokenizer)."""

    bos_token_id = 999
    all_special_ids = [999]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        # No specials added; a literal '<bos>' word re-encodes to the BOS id.
        return [self.bos_token_id if w == "<bos>" else int(w) for w in text.split()]


class _NoBosTok:
    """Tokenizer stub exposing no discoverable BOS id (neither attr nor method)."""

    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(w) for w in text.split()]


def test_bos_prepended_model_input_prepends_exactly_one_bos():
    """Plain content tokens get exactly one BOS prepended at position 0."""
    mi = extraction._bos_prepended_model_input(
        "1 2 3", tokenizer=_BosAwareTokenizer(), max_tokens=64, n_ctx=64,
    )
    assert mi == [999, 1, 2, 3]


def test_bos_prepended_model_input_front_clamps_including_bos():
    """The BOS occupies one of the max_tokens slots when front-clamping."""
    mi = extraction._bos_prepended_model_input(
        "1 2 3 4 5", tokenizer=_BosAwareTokenizer(), max_tokens=3, n_ctx=64,
    )
    assert mi == [999, 1, 2]  # BOS + first 2 content tokens == 3 total


def test_bos_prepended_model_input_raises_on_missing_bos():
    """A tokenizer exposing no discoverable BOS id is a real config bug."""
    with pytest.raises(AxBenchBaseError, match="no BOS"):
        extraction._bos_prepended_model_input(
            "1 2", tokenizer=_NoBosTok(), max_tokens=64, n_ctx=64,
        )


def test_bos_prepended_model_input_preserves_in_data_bos_as_double_bos():
    """A literal '<bos>' in the TEXT re-encodes to the real BOS id; we PRESERVE
    it as data and prepend our own BOS anyway -> an intentional double-BOS.

    The in-data '<bos>' is genuine document data, not corruption: our injected
    BOS is position 0 (dropped by the acts[1:] pool), and the in-data BOS stays
    at position 1 as data. So a '<bos>'-starting input keeps its original tokens
    and double-BOSes rather than raising. Teeth: if the preservation logic
    dropped/deduped the in-data BOS, result[1] would be a content token (1), not
    the BOS id, and the length would shrink -- this asserts both.
    """
    result = extraction._bos_prepended_model_input(
        "<bos> 1 2", tokenizer=_BosAwareTokenizer(), max_tokens=64, n_ctx=64,
    )
    # our injected BOS at 0, the in-data BOS preserved at 1, then content [1, 2].
    assert result == [999, 999, 1, 2]
    assert result[0] == result[1] == 999  # double-BOS: injected + in-data


def test_bos_prepended_model_input_warns_on_special_id_pollution():
    """An in-data special id fires a loguru WARNING naming the polluting id
    (observability), but the call proceeds and returns the preserved tokens."""
    with capture_loguru_info() as captured:
        result = extraction._bos_prepended_model_input(
            "<bos> 1 2", tokenizer=_BosAwareTokenizer(), max_tokens=64, n_ctx=64,
        )
    assert result == [999, 999, 1, 2]  # proceeded, tokens preserved
    warnings = [line for line in captured if "in-data special token id" in line]
    assert len(warnings) == 1, f"expected exactly one pollution warning: {captured!r}"
    assert "[999]" in warnings[0], (
        f"warning must name the polluting id 999: {warnings[0]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BosPrependedActivations — the +1 BOS-exclusion offset object (#602 Q3)
# ─────────────────────────────────────────────────────────────────────────────


def _distinct_acts(n_rows: int, n_features: int = 4) -> t.Tensor:
    """Distinct per-row values so span slices are verifiable by value."""
    return t.arange(n_rows * n_features, dtype=t.float32).reshape(n_rows, n_features)


def test_bos_prepended_activations_content_excludes_bos_row():
    acts = _distinct_acts(4)
    bpa = extraction.BosPrependedActivations.from_forward(acts, [10, 11, 12, 13])
    assert t.equal(bpa.content(), acts[1:])  # BOS row 0 excluded


def test_bos_prepended_activations_prefix_and_completion_spans():
    acts = _distinct_acts(6)  # BOS + 2 prefix + 3 completion
    bpa = extraction.BosPrependedActivations.from_forward(
        acts, [0, 1, 2, 3, 4, 5], n_prefix=2,
    )
    assert t.equal(bpa.prefix(), acts[1:3])  # acts[1 : 1 + n_prefix]
    assert t.equal(bpa.completion(), acts[3:])  # acts[1 + n_prefix :]
    assert t.equal(bpa.content(), acts[1:])  # prefix + completion (all but BOS)


def test_bos_prepended_activations_prefix_empty_when_n_prefix_zero():
    acts = _distinct_acts(4)
    bpa = extraction.BosPrependedActivations.from_forward(acts, [0, 1, 2, 3], n_prefix=0)
    assert bpa.prefix().shape[0] == 0  # empty prefix span
    assert t.equal(bpa.completion(), acts[1:])  # completion == all content


def test_bos_prepended_activations_raises_on_row_count_mismatch():
    """acts not 1:1 with model_input (extractor injected its own BOS) → raise."""
    acts = _distinct_acts(3)  # 3 rows for a 4-token input
    with pytest.raises(AxBenchBaseError, match="acts must be 1:1"):
        extraction.BosPrependedActivations.from_forward(acts, [0, 1, 2, 3])


def test_bos_prepended_activations_raises_on_n_prefix_out_of_range():
    acts = _distinct_acts(4)  # 4 rows -> n_prefix must be in [0, 3]
    with pytest.raises(AxBenchBaseError, match="n_prefix=4 out of range"):
        extraction.BosPrependedActivations.from_forward(acts, [0, 1, 2, 3], n_prefix=4)
    with pytest.raises(AxBenchBaseError, match="n_prefix=-1 out of range"):
        extraction.BosPrependedActivations.from_forward(acts, [0, 1, 2, 3], n_prefix=-1)


def test_bos_prepended_activations_raises_on_non_2d_acts():
    with pytest.raises(AxBenchBaseError, match="acts must be 2-D"):
        extraction.BosPrependedActivations.from_forward(t.zeros(4), [0, 1, 2, 3])
