"""Producer-freeze tests for ``ignore_bos_features``.

``ignore_bos_features`` is no longer a user-facing config option: the extraction
producer (:func:`steering.extract_features.main`) is frozen ALWAYS-ON via the
:func:`steering.extract_features._resolve_ignore_bos_features` helper. Because
``main`` is ``@hydra.main``-decorated (it cannot be called directly with a
synthetic cfg), the freeze + stale-key logic lives in that small pure helper so
it is unit-testable without a GPU or Hydra entrypoint — mirroring the existing
``_warn_if_path_flag_mismatch`` / ``_coerce_config_l0`` helper pattern.

These tests pin the NEW behavior only:
  * the helper returns ``True`` for every config shape (key absent / stale
    ``false`` / stale ``true``);
  * a key still present in the resolved config is IGNORED and logged loudly via
    ``logger.error`` (non-fatal), so a user notices the dead key;
  * the presence check is OmegaConf-struct-safe (mirrors the Hydra-composed cfg);
  * EXACT-EQUALITY: feeding the resolved (frozen) value through the real
    :func:`process_shard` compute path yields byte-identical output
    (decompressed tensors) to an explicit ``ignore_bos_features=True`` run, for
    every config shape — proving the option removal introduces ZERO drift versus
    the prior ``flag=True`` behavior (the "existing implementation");
  * the frozen value still raises ``BosOnlySegmentError`` on an all-``<bos>``
    segment (``segment_size=1``) and leaves NO partial shard (resume-safe).

The compute/format/consumer behavior of ``ignore_bos_features=True`` itself
(seg-0 drop, ``segment_size=1`` raise, no-BOS ``ValueError``, ``/no_bos`` leaf,
``verify_feature_dir``, ``find_optimal`` forwarding) is already covered by
``test_ignore_bos_integration.py`` and is intentionally NOT duplicated here.
"""

from __future__ import annotations

import glob
import logging

import pytest
import torch as t
from omegaconf import OmegaConf

from errors import BosOnlySegmentError
from steering.extract_features import _resolve_ignore_bos_features, process_shard
from steering.segmentation import SegmentMode

# Reuse the EXACT existing tiny stub harness + shared process_shard driver.
from tests.steering.test_context_independent_extraction import (
    _make_parquet,
    _tiny_model_sae,
)
from tests.steering.test_ignore_bos_integration import (
    _CONTEXT_WINDOW,
    _LAYER,
    _POOLING,
    _run,
)


def _error_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# Helper: freeze to True + loud-but-non-fatal stale-key handling
# ---------------------------------------------------------------------------


def test_resolve_absent_returns_true_and_logs_nothing(caplog):
    """Key absent (the new default) ⇒ always-on, no error logged."""
    cfg = OmegaConf.create({"model": "gemma-2-2b"})  # no ignore_bos_features key
    with caplog.at_level(logging.ERROR):
        result = _resolve_ignore_bos_features(cfg)
    assert result is True
    assert "ignore_bos_features" not in caplog.text
    assert _error_records(caplog) == []


def test_resolve_stale_false_returns_true_and_logs_error(caplog):
    """A stale ``ignore_bos_features: false`` is IGNORED (still always-on) and
    logged via ``logger.error`` — the user's attempt to turn it OFF must not
    silently take effect."""
    cfg = OmegaConf.create({"ignore_bos_features": False})
    with caplog.at_level(logging.ERROR):
        result = _resolve_ignore_bos_features(cfg)
    assert result is True, "stale false must NOT disable always-on"
    errs = _error_records(caplog)
    assert len(errs) == 1, f"expected exactly one error log, got {len(errs)}"
    assert "ignore_bos_features" in caplog.text
    assert "IGNORED" in caplog.text


def test_resolve_stale_true_returns_true_and_logs_error(caplog):
    """A stale ``ignore_bos_features: true`` is also flagged (the key is gone,
    even a matching value is reported so users remove it)."""
    cfg = OmegaConf.create({"ignore_bos_features": True})
    with caplog.at_level(logging.ERROR):
        result = _resolve_ignore_bos_features(cfg)
    assert result is True
    assert len(_error_records(caplog)) == 1
    assert "ignore_bos_features" in caplog.text


def test_resolve_is_struct_mode_safe(caplog):
    """The presence check must not raise under OmegaConf struct mode (which is
    how the Hydra-composed cfg arrives in ``main``)."""
    absent = OmegaConf.create({"model": "gemma-2-2b"})
    OmegaConf.set_struct(absent, True)
    present = OmegaConf.create({"ignore_bos_features": False})
    OmegaConf.set_struct(present, True)
    with caplog.at_level(logging.ERROR):
        assert _resolve_ignore_bos_features(absent) is True  # must not raise
        assert _resolve_ignore_bos_features(present) is True  # must not raise
    # Only the present-key cfg logs.
    assert len(_error_records(caplog)) == 1


# ---------------------------------------------------------------------------
# Crown-jewel: resolved (frozen) value == explicit ignore_bos_features=True
# ---------------------------------------------------------------------------


def test_resolved_value_matches_explicit_flag_true_output(tmp_path):
    """EXACT-EQUALITY against the existing implementation.

    The only producer change is that ``main`` resolves ``ignore_bos_features``
    to ``True`` (instead of reading it from config). Feeding that resolved value
    through the unchanged :func:`process_shard` compute path must yield output
    byte-identical (decompressed tensors) to an explicit
    ``ignore_bos_features=True`` run — for EVERY config shape (absent / stale
    false / stale true). This proves the option removal causes zero drift versus
    the behavior we are freezing to.
    """
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(
        parquet_path,
        [
            "hello world this is a longer pretraining document for pooling",
            "second document with several tokens here too",
        ],
    )
    # ONE model/SAE pair shared across all runs (weights are random per
    # _tiny_model_sae() call) so the only variable is the resolved flag.
    model, sae = _tiny_model_sae()

    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    golden = _run(model, sae, parquet_path, golden_dir, ignore_bos=True)
    assert golden["ignore_bos_features"] is True

    cfg_variants = [
        OmegaConf.create({}),  # key absent — the new default
        OmegaConf.create({"ignore_bos_features": False}),  # stale false, ignored
        OmegaConf.create({"ignore_bos_features": True}),  # stale true
    ]
    for i, cfg in enumerate(cfg_variants):
        resolved = _resolve_ignore_bos_features(cfg)
        assert resolved is True
        out_dir = tmp_path / f"frozen_{i}"
        out_dir.mkdir()
        out = _run(model, sae, parquet_path, out_dir, ignore_bos=resolved)

        assert out["ignore_bos_features"] is True
        # Feature content: exact tensor equality (same code path, same value).
        assert t.equal(out["feature_ids"], golden["feature_ids"]), (
            f"feature_ids differ from explicit flag=True for cfg variant {i}"
        )
        assert t.equal(out["feature_mags"], golden["feature_mags"]), (
            f"feature_mags differ from explicit flag=True for cfg variant {i}"
        )
        assert t.equal(out["offsets"], golden["offsets"]), (
            f"offsets differ from explicit flag=True for cfg variant {i}"
        )
        # Segmentation structure + scalars: exact equality too.
        assert out["n_segments"] == golden["n_segments"]
        assert out["n_docs"] == golden["n_docs"]
        for k in (
            "doc_ids",
            "segment_indices",
            "segment_positions",
            "segment_char_starts",
            "segment_char_ends",
            "doc_token_counts",
        ):
            assert t.equal(out[k], golden[k]), (
                f"segmentation array {k!r} differs from explicit flag=True for "
                f"cfg variant {i}"
            )


# ---------------------------------------------------------------------------
# Frozen value is fail-loud + resume-safe on an all-<bos> segment
# ---------------------------------------------------------------------------


def test_frozen_value_segment_size_1_raises_and_writes_no_partial_shard(tmp_path):
    """Under the frozen value, ``segment_size=1`` makes segment 0 exactly
    ``[<bos>]`` (an all-``False`` keep-mask) ⇒ ``BosOnlySegmentError``. The shard
    is written atomically only after every doc is pooled, so the raise must leave
    NO ``.safetensors.zst`` behind (otherwise resume would wrongly skip it)."""
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(parquet_path, ["hello world", "second doc here"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    resolved = _resolve_ignore_bos_features(OmegaConf.create({}))  # frozen True
    model, sae = _tiny_model_sae()
    with pytest.raises(BosOnlySegmentError):
        process_shard(
            parquet_path,
            model,
            sae,
            layer=_LAYER,
            output_dir=str(out_dir),
            context_window=_CONTEXT_WINDOW,
            pooling=_POOLING,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=1,
            gpu_batch_size=2,
            cpu_batch_size=2,
            ignore_bos_features=resolved,
        )

    assert glob.glob(str(out_dir / "*.safetensors.zst")) == [], (
        "a partial shard was written despite BosOnlySegmentError — resume would "
        "wrongly treat this shard as complete"
    )
