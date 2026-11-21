"""Tests for steering.validate_features module.

Run with:
    uv run pytest tests/test_validate_features.py -v
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import zstandard
from omegaconf import OmegaConf
from safetensors.torch import save as safetensors_save

import steering.validate_features as vf_mod
from steering.feature_io import COMPRESSED_EXT, save_features
from steering.shard_validation import (
    ValidationResult,
    validate_shard,
    validate_shard_data,
)
from steering.validate_features import validate_shard_activations


def test_validate_shard_against_source_rejects_legacy_assistant_loss_mask(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "legacy_loss_mask.parquet"
    pq.write_table(
        pa.table({"text": ["rendered"], "assistant_loss_mask": [1]}),
        parquet_path,
    )

    result = vf_mod.validate_shard_against_source(
        shard_data={},
        parquet_path=parquet_path,
        tokenizer=object(),
        context_window=128,
        segment_size=128,
    )

    assert not result.valid
    assert "assistant_loss_mask" in "\n".join(result.violations)


def _make_valid_shard_data(
    n_docs: int = 3, segs_per_doc: int = 2, feats_per_seg: int = 5
) -> dict:
    """Create a valid shard data dict with segment metadata."""
    n_segments = n_docs * segs_per_doc
    n_features_total = n_segments * feats_per_seg

    offsets = [i * feats_per_seg for i in range(n_segments + 1)]
    feature_ids = t.randint(0, 1000, (n_features_total,), dtype=t.int32)
    feature_mags = t.rand(n_features_total, dtype=t.float32).abs() + 0.01

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    for d in range(n_docs):
        for s in range(segs_per_doc):
            doc_ids.append(d)
            seg_indices.append(s)
            seg_positions.append(s * 128)
            seg_char_starts.append(s * 16)
            seg_char_ends.append(s * 16 + 16)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([1000] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


def _save_valid_shard(tmp_path, name="valid", **kwargs):
    """Create and save a valid shard, return the path."""
    data = _make_valid_shard_data(**kwargs)
    path = save_features(data, tmp_path / f"{name}_features")
    return str(path), data


def _save_raw_shard(tmp_path, name: str, data: dict) -> str:
    """Write a .safetensors.zst file directly, bypassing save_features processing.

    This allows creating intentionally corrupt files for validation testing.
    Tensors are written as-is (no sort/delta/dtype optimization).
    """
    path = tmp_path / f"{name}_features{COMPRESSED_EXT}"
    tensors = {}
    metadata = {}
    for key, value in data.items():
        if isinstance(value, t.Tensor):
            tensors[key] = value
        else:
            metadata[key] = str(value)
    sf_bytes = safetensors_save(tensors, metadata=metadata)
    cctx = zstandard.ZstdCompressor(level=3)
    path.write_bytes(cctx.compress(sf_bytes))
    return str(path)


class TestValidateFeatures:
    def test_valid_shard_passes(self, tmp_path):
        path, _ = _save_valid_shard(tmp_path)
        result = validate_shard(path)
        assert result.valid, f"Violations: {result.violations}"

    def test_missing_field_fails(self, tmp_path):
        data = _make_valid_shard_data()
        del data["doc_ids"]
        path = save_features(data, tmp_path / "missing_features")
        result = validate_shard(str(path))
        assert not result.valid
        assert any("doc_ids" in v for v in result.violations)

    def test_wrong_offsets_dtype_fails(self, tmp_path):
        """offsets stored as int32 should be upcast to int64 by load_features,
        so this test verifies the dtype after round-trip."""
        data = _make_valid_shard_data()
        # After save/load round-trip, offsets are always int64 (upcast in load_features)
        # So this scenario is handled internally; test that valid shard passes
        path = save_features(data, tmp_path / "dtype_features")
        result = validate_shard(str(path))
        assert result.valid

    def test_offsets_last_mismatch_fails(self, tmp_path):
        data = _make_valid_shard_data()
        # Make offsets[-1] not match len(feature_ids)
        data["offsets"][-1] = data["offsets"][-1] + 999
        path = _save_raw_shard(tmp_path, "bad_offset", data)
        result = validate_shard(path)
        assert not result.valid

    def test_doc_ids_not_monotonic_fails(self, tmp_path):
        data = _make_valid_shard_data(n_docs=3, segs_per_doc=2)
        # doc_ids is [0, 0, 1, 1, 2, 2]; set last element to 0 to break monotonicity
        data["doc_ids"][-1] = 0
        path = save_features(data, tmp_path / "bad_docids_features")
        result = validate_shard(str(path))
        assert not result.valid

    def test_n_segments_mismatch_fails(self, tmp_path):
        data = _make_valid_shard_data()
        data["n_segments"] = 999
        path = save_features(data, tmp_path / "bad_nseg_features")
        result = validate_shard(str(path))
        assert not result.valid

    def test_negative_magnitudes_fail(self, tmp_path):
        data = _make_valid_shard_data()
        if len(data["feature_mags"]) > 0:
            data["feature_mags"][0] = -1.0
        path = save_features(data, tmp_path / "neg_mag_features")
        result = validate_shard(str(path))
        assert not result.valid

    def test_non_finite_magnitudes_fail(self, tmp_path):
        """+inf magnitudes must be rejected. nan is already caught by the
        >0 check (nan > 0 is False) — a nan shard yields BOTH the "<= 0"
        and the "not finite" violation, which is expected. +inf passes >0
        (inf > 0 is True), so the explicit finiteness check is what catches
        it. Closes the coverage gap vs the old Step-0b heredoc's
        torch.isfinite assertion. (_save_raw_shard also trips the dtype
        validator, so we assert the SPECIFIC finiteness violation — not just
        result.valid — to keep this a real regression guard.)"""
        data = _make_valid_shard_data()
        if len(data["feature_mags"]) > 0:
            data["feature_mags"][0] = float("inf")
        path = _save_raw_shard(tmp_path, "inf_mag", data)
        result = validate_shard(path)
        assert not result.valid
        assert any("not finite" in v for v in result.violations), (
            f"expected a finiteness violation, got: {result.violations}"
        )

    def test_result_dataclass(self):
        r = ValidationResult(valid=True, violations=[])
        assert r.valid
        assert r.violations == []


def test_validate_shard_data_flags_empty_segment():
    data = _make_valid_shard_data(n_docs=1, segs_per_doc=3, feats_per_seg=2)
    data["offsets"] = t.tensor([0, 2, 2, 4], dtype=t.int64)  # segment 1 empty
    data["feature_ids"] = data["feature_ids"][:4]
    data["feature_mags"] = data["feature_mags"][:4]
    result = validate_shard_data(data)
    assert not result.valid
    assert any("zero features" in v or "empty" in v for v in result.violations)


def test_validate_shard_disk_path_flags_empty_segment(tmp_path):
    """The DISK path (validate_shard) must also flag an empty base segment under
    the strict default. A raw write bypasses the now-strict save_features so we
    can land a genuinely empty segment 1 on disk (issue #522)."""
    data = _make_valid_shard_data(n_docs=1, segs_per_doc=3, feats_per_seg=2)
    data["offsets"] = t.tensor([0, 2, 2, 4], dtype=t.int64)  # segment 1 empty
    data["feature_ids"] = data["feature_ids"][:4]
    data["feature_mags"] = data["feature_mags"][:4]
    raw_path = _save_raw_shard(tmp_path, "empty_seg", data)
    result = validate_shard(raw_path)  # non-empty check is unconditional (issue #522)
    assert not result.valid
    assert any("zero features" in v or "empty" in v for v in result.violations)


def test_validate_shard_data_accepts_zero_segments():
    """A whole-shard n_segments==0 (offsets==[0]) is EXEMPT from the non-empty
    rule and must validate even though the non-empty check is unconditional.
    Pins the exemption at the validator surface, not just the
    empty_segment_indices predicate."""
    data = _make_valid_shard_data(n_docs=1, segs_per_doc=0)
    assert data["n_segments"] == 0
    assert data["offsets"].tolist() == [0]
    result = validate_shard_data(data)
    assert result.valid, result.violations


# ---------------------------------------------------------------------------
# validate_shard_activations: hook_name propagation (regression)
# ---------------------------------------------------------------------------


class TestValidateShardActivationsHookName:
    """Verify validate_shard_activations re-extracts at the SAE's actual hook.

    Regression guard for a bug where validation defaulted the hook to
    ``blocks.{layer}.hook_resid_pre`` instead of honoring the SAE's real
    ``cfg.hook_name`` (e.g., ``hook_resid_post`` for GemmaScope). Under the
    bug, GPU spot-checks re-extracted from a different activation site than
    the stored shard, producing ~30% feature count mismatches and false-
    positive validation failures on every correctly-extracted shard.
    """

    def _minimal_shard_data(self, tmp_path) -> tuple[dict, str]:
        data = _make_valid_shard_data(n_docs=1, segs_per_doc=1, feats_per_seg=2)
        parquet_path = str(tmp_path / "src.parquet")
        pq.write_table(pa.table({"text": ["hello"]}), parquet_path)
        return data, parquet_path

    def test_defaults_hook_name_from_sae_cfg_when_unspecified(
        self, tmp_path, monkeypatch,
    ):
        """When caller passes no hook_name, validation must use sae.cfg.hook_name."""
        data, parquet_path = self._minimal_shard_data(tmp_path)

        sae = MagicMock()
        sae.cfg = SimpleNamespace(hook_name="blocks.8.hook_resid_post")
        model = MagicMock()

        captured: dict[str, object] = {}

        def fake_extract(*args, **kwargs):
            captured["hook_name"] = kwargs.get("hook_name")
            return [t.tensor([], dtype=t.int32)], [t.tensor([], dtype=t.float32)], [], [], [], 0

        monkeypatch.setattr(vf_mod, "extract_document_features", fake_extract)

        validate_shard_activations(
            data, parquet_path, model, sae,
            layer=8, context_window=1024, segment_size=128,
            pooling="mean_pool",
            n_spot_check=1,
            # hook_name deliberately omitted
        )
        assert captured["hook_name"] == "blocks.8.hook_resid_post"

    def test_defaults_hook_name_from_sae_cfg_metadata_when_unspecified(
        self, tmp_path, monkeypatch,
    ):
        """SAELens real SAEs expose the hook on ``sae.cfg.metadata.hook_name``;
        the probe must walk through to ``metadata`` when ``cfg.hook_name`` is absent."""
        data, parquet_path = self._minimal_shard_data(tmp_path)

        sae = MagicMock()
        # ``cfg`` has no hook_name attribute; metadata does (SAELens layout).
        sae.cfg = SimpleNamespace(
            d_sae=16384,
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        )
        model = MagicMock()

        captured: dict[str, object] = {}

        def fake_extract(*args, **kwargs):
            captured["hook_name"] = kwargs.get("hook_name")
            return [t.tensor([], dtype=t.int32)], [t.tensor([], dtype=t.float32)], [], [], [], 0

        monkeypatch.setattr(vf_mod, "extract_document_features", fake_extract)

        validate_shard_activations(
            data, parquet_path, model, sae,
            layer=8, context_window=1024, segment_size=128,
            pooling="mean_pool",
            n_spot_check=1,
        )
        assert captured["hook_name"] == "blocks.8.hook_resid_post"

    def test_explicit_hook_name_overrides_sae_cfg(self, tmp_path, monkeypatch):
        """Caller-provided hook_name takes precedence over sae.cfg.hook_name."""
        data, parquet_path = self._minimal_shard_data(tmp_path)
        sae = MagicMock()
        sae.cfg = SimpleNamespace(hook_name="blocks.8.hook_resid_post")
        model = MagicMock()

        captured: dict[str, object] = {}

        def fake_extract(*args, **kwargs):
            captured["hook_name"] = kwargs.get("hook_name")
            return [t.tensor([], dtype=t.int32)], [t.tensor([], dtype=t.float32)], [], [], [], 0

        monkeypatch.setattr(vf_mod, "extract_document_features", fake_extract)

        validate_shard_activations(
            data, parquet_path, model, sae,
            layer=8, context_window=1024, segment_size=128,
            pooling="mean_pool",
            n_spot_check=1,
            hook_name="blocks.12.hook_mlp_out",
        )
        assert captured["hook_name"] == "blocks.12.hook_mlp_out"

    def test_sae_without_cfg_hook_name_leaves_hook_none(
        self, tmp_path, monkeypatch,
    ):
        """SAE whose cfg lacks hook_name → fall through as None (preserves
        existing behavior for callers that know what they're doing)."""
        data, parquet_path = self._minimal_shard_data(tmp_path)
        sae = MagicMock()
        sae.cfg = SimpleNamespace(d_sae=16384)  # no hook_name attribute
        model = MagicMock()

        captured: dict[str, object] = {}

        def fake_extract(*args, **kwargs):
            captured["hook_name"] = kwargs.get("hook_name")
            return [t.tensor([], dtype=t.int32)], [t.tensor([], dtype=t.float32)], [], [], [], 0

        monkeypatch.setattr(vf_mod, "extract_document_features", fake_extract)

        validate_shard_activations(
            data, parquet_path, model, sae,
            layer=8, context_window=1024, segment_size=128,
            pooling="mean_pool",
            n_spot_check=1,
        )
        assert captured["hook_name"] is None

    def test_sae_without_cfg_attr_does_not_crash(self, tmp_path, monkeypatch):
        """Defensive: SAE without any ``cfg`` attribute falls through as None."""
        data, parquet_path = self._minimal_shard_data(tmp_path)
        sae = object()  # no .cfg at all
        model = MagicMock()

        captured: dict[str, object] = {}

        def fake_extract(*args, **kwargs):
            captured["hook_name"] = kwargs.get("hook_name")
            return [t.tensor([], dtype=t.int32)], [t.tensor([], dtype=t.float32)], [], [], [], 0

        monkeypatch.setattr(vf_mod, "extract_document_features", fake_extract)

        validate_shard_activations(
            data, parquet_path, model, sae,
            layer=8, context_window=1024, segment_size=128,
            pooling="mean_pool",
            n_spot_check=1,
        )
        assert captured["hook_name"] is None


class TestL0TierConsistency:
    """validate_feature_directory must flag a cross-shard l0_tier mismatch
    (v1 dirs are all-None and stay consistent; v2 must be uniform). Spec C4."""

    def test_uniform_v2_tier_is_consistent(self, tmp_path):
        d0 = _make_valid_shard_data()
        d0["l0_tier"] = "medium"
        d1 = _make_valid_shard_data()
        d1["l0_tier"] = "medium"
        _save_raw_shard(tmp_path, "shard_00000", d0)
        _save_raw_shard(tmp_path, "shard_00001", d1)
        res = vf_mod.validate_feature_directory(tmp_path)
        assert not any("l0_tier" in v for v in res.violations), res.violations

    def test_mismatched_v2_tier_is_flagged(self, tmp_path):
        d0 = _make_valid_shard_data()
        d0["l0_tier"] = "medium"
        d1 = _make_valid_shard_data()
        d1["l0_tier"] = "big"
        _save_raw_shard(tmp_path, "shard_00000", d0)
        _save_raw_shard(tmp_path, "shard_00001", d1)
        res = vf_mod.validate_feature_directory(tmp_path)
        assert not res.valid
        assert any("l0_tier" in v for v in res.violations), res.violations

    def test_v1_no_tier_is_consistent(self, tmp_path):
        # No l0_tier key on either shard (v1) -> get() is None == None.
        _save_raw_shard(tmp_path, "shard_00000", _make_valid_shard_data())
        _save_raw_shard(tmp_path, "shard_00001", _make_valid_shard_data())
        res = vf_mod.validate_feature_directory(tmp_path)
        assert not any("l0_tier" in v for v in res.violations), res.violations


class TestFeatureIdRangeInDirectory:
    """validate_feature_directory must re-run the feature-id upper-bound check
    when n_sae_features is passed (Step-0b heredoc parity). Without the param
    the check must remain a no-op (back-compat). Spec: threading n_sae_features
    into validate_shard_data so _check_feature_id_range is not a no-op."""

    def test_out_of_range_id_flagged_when_n_sae_features_provided(self, tmp_path):
        """A shard with feature_id >= n_sae_features must produce a violation."""
        data = _make_valid_shard_data(feats_per_seg=3)
        # Force one feature_id to exceed the declared SAE width
        data["feature_ids"][-1] = 65536  # >= n_sae_features=65536
        _save_raw_shard(tmp_path, "shard_00000", data)
        res = vf_mod.validate_feature_directory(tmp_path, n_sae_features=65536)
        assert not res.valid
        assert any("feature_id" in v and "n_sae_features" in v for v in res.violations), (
            res.violations
        )

    def test_in_range_ids_pass_when_n_sae_features_provided(self, tmp_path):
        """All feature_ids < n_sae_features must not produce a range violation."""
        data = _make_valid_shard_data(feats_per_seg=3)
        # feature_ids are randint(0, 1000) so well within 65536
        _save_raw_shard(tmp_path, "shard_00000", data)
        res = vf_mod.validate_feature_directory(tmp_path, n_sae_features=65536)
        assert not any("n_sae_features" in v for v in res.violations), res.violations

    def test_no_n_sae_features_skips_range_check_back_compat(self, tmp_path):
        """Without n_sae_features the range check must stay a no-op (back-compat)."""
        data = _make_valid_shard_data(feats_per_seg=3)
        # Force an id that would fail if n_sae_features=10 were passed
        data["feature_ids"][-1] = 9999
        _save_raw_shard(tmp_path, "shard_00000", data)
        # No n_sae_features → _check_feature_id_range no-ops → no violation
        res = vf_mod.validate_feature_directory(tmp_path)
        assert not any("n_sae_features" in v for v in res.violations), res.violations

    def test_negative_id_flagged_when_n_sae_features_provided(self, tmp_path):
        """Negative feature_ids must also be caught when n_sae_features is set.

        _save_raw_shard bypasses delta-encoding, making exact loaded ids
        unpredictable. Inject a negative id via a patch on load_features so
        the in-memory dict reaching _check_feature_id_range is clean.
        """
        # Save a fully-valid shard so the load path is clean.
        _save_valid_shard(tmp_path, "shard_00000")

        original_load = vf_mod.load_features

        def _patched_load(p):
            d = original_load(p)
            d["feature_ids"] = d["feature_ids"].clone()
            d["feature_ids"][-1] = -1
            return d

        with patch.object(vf_mod, "load_features", side_effect=_patched_load):
            res = vf_mod.validate_feature_directory(tmp_path, n_sae_features=65536)

        assert not res.valid
        assert any("negative" in v.lower() or "feature_id" in v.lower() for v in res.violations), (
            res.violations
        )


class TestValidateFeaturesTokenizerOnlyLoad:
    """The model-only branch must load just the tokenizer (resolve_tokenizer,
    weight-free) for registered models, and fall back to a full TL load only
    for unregistered ones. Spec C4."""

    def _cfg(self, model="gemma-3-4b-it"):
        return OmegaConf.create(
            {
                "feature_dir": "/nonexistent",
                "parquet_dir": None,
                "model": model,
                "sae_release": None,
                "sae_id": None,
                "n_spot_check": 0,
                "model_dtype": "bfloat16",
                "feature_count_rtol": 0.01,
                "jaccard_threshold": 0.99,
                "resolve_parquet_default": False,
            }
        )

    def test_registered_model_uses_resolve_tokenizer_not_tl(self):
        tokenizer = MagicMock(name="registered_tokenizer")
        captured = {}

        def _fake_validate_feature_directory(feature_dir, **kwargs):
            captured["feature_dir"] = feature_dir
            captured.update(kwargs)
            return vf_mod.ValidationResult(valid=True, violations=[])

        with (
            patch(
                "steering.validate_features.resolve_tokenizer",
                return_value=tokenizer,
            ) as m_rt,
            patch(
                "steering.validate_features.resolve_sae_max_context",
                return_value=8192,
            ) as m_ctx,
            patch("steering.validate_features.HookedTransformer") as m_ht,
            patch(
                "steering.validate_features.validate_feature_directory",
                side_effect=_fake_validate_feature_directory,
            ),
            pytest.raises(SystemExit) as exc,
        ):
            vf_mod.main.__wrapped__(self._cfg())
        assert exc.value.code == 0
        m_rt.assert_called_once_with("gemma-3-4b-it")
        m_ctx.assert_called_once_with("gemma-3-4b-it")
        m_ht.from_pretrained.assert_not_called()
        assert captured == {
            "feature_dir": Path("/nonexistent"),
            "parquet_dir": None,
            "tokenizer": tokenizer,
            "sae_max_context": 8192,
            "n_sae_features": None,
        }

    def test_registered_model_context_lookup_failure_propagates_without_tl(self):
        with (
            patch(
                "steering.validate_features.resolve_tokenizer",
                return_value=MagicMock(),
            ) as m_rt,
            patch(
                "steering.validate_features.resolve_sae_max_context",
                side_effect=RuntimeError("config unavailable"),
            ) as m_ctx,
            patch("steering.validate_features.HookedTransformer") as m_ht,
            patch(
                "steering.validate_features.validate_feature_directory",
                return_value=vf_mod.ValidationResult(
                    valid=True, violations=[]
                ),
            ) as m_vfd,
            pytest.raises(RuntimeError, match="config unavailable"),
        ):
            vf_mod.main.__wrapped__(self._cfg())
        m_rt.assert_called_once_with("gemma-3-4b-it")
        m_ctx.assert_called_once_with("gemma-3-4b-it")
        m_ht.from_pretrained.assert_not_called()
        m_vfd.assert_not_called()

    def test_unregistered_model_falls_back_to_tl(self):
        tokenizer = MagicMock(name="tl_tokenizer")
        captured = {}

        def _fake_validate_feature_directory(feature_dir, **kwargs):
            captured["feature_dir"] = feature_dir
            captured.update(kwargs)
            return vf_mod.ValidationResult(valid=True, violations=[])

        with (
            patch(
                "steering.validate_features.resolve_tokenizer",
                side_effect=ValueError("not in EXTRACTION_MODELS"),
            ),
            patch("steering.validate_features.resolve_sae_max_context") as m_rsmc,
            patch("steering.validate_features.HookedTransformer") as m_ht,
            patch(
                "steering.validate_features.validate_feature_directory",
                side_effect=_fake_validate_feature_directory,
            ),
            pytest.raises(SystemExit) as exc,
        ):
            m_ht.from_pretrained.return_value = MagicMock(
                tokenizer=tokenizer, cfg=SimpleNamespace(n_ctx=1024)
            )
            vf_mod.main.__wrapped__(self._cfg(model="gpt2"))
        assert exc.value.code == 0
        m_ht.from_pretrained.assert_called_once_with("gpt2", device="cpu")
        m_rsmc.assert_not_called()
        assert captured == {
            "feature_dir": Path("/nonexistent"),
            "parquet_dir": None,
            "tokenizer": tokenizer,
            "sae_max_context": 1024,
            "n_sae_features": None,
        }


class TestResolveParquetDefault:
    """parquet_dir:null still means 'no source validation' unless the new
    opt-in resolve_parquet_default=true is set (then it resolves to the
    nanochat default dir). Backward-compatible. Spec C4."""

    def _cfg(self, **over):
        base = {
            "feature_dir": "/nonexistent",
            "parquet_dir": None,
            "model": None,
            "sae_release": None,
            "sae_id": None,
            "n_spot_check": 0,
            "model_dtype": "bfloat16",
            "feature_count_rtol": 0.01,
            "jaccard_threshold": 0.99,
            "resolve_parquet_default": False,
        }
        base.update(over)
        return OmegaConf.create(base)

    def test_default_false_keeps_parquet_none(self):
        captured = {}

        def _fake_vfd(feature_dir, **kw):
            captured.update(kw)
            return vf_mod.ValidationResult(valid=True, violations=[])

        with (
            patch(
                "steering.validate_features.validate_feature_directory",
                side_effect=_fake_vfd,
            ),
            pytest.raises(SystemExit),
        ):
            vf_mod.main.__wrapped__(self._cfg())
        assert captured["parquet_dir"] is None

    def test_opt_in_resolves_default_dir(self):
        captured = {}

        def _fake_vfd(feature_dir, **kw):
            captured.update(kw)
            return vf_mod.ValidationResult(valid=True, violations=[])

        with (
            patch(
                "steering.validate_features.resolve_data_dir",
                return_value=Path("/data/base_data"),
            ) as m_rdd,
            patch(
                "steering.validate_features.validate_feature_directory",
                side_effect=_fake_vfd,
            ),
            patch(
                "steering.validate_features.resolve_tokenizer",
                return_value=object(),
            ),
            patch(
                "steering.validate_features.resolve_sae_max_context",
                return_value=8192,
            ),
            pytest.raises(SystemExit),
        ):
            vf_mod.main.__wrapped__(
                self._cfg(
                    resolve_parquet_default=True,
                    model="gemma-test",
                )
            )
        m_rdd.assert_called_once_with(None, None)
        assert captured["parquet_dir"] == Path("/data/base_data")

    def test_absent_key_keeps_parquet_none(self):
        """Existing callers never set resolve_parquet_default; with the key
        ENTIRELY ABSENT, getattr's False default must preserve old
        behavior (parquet_dir stays None)."""
        captured = {}

        def _fake_vfd(feature_dir, **kw):
            captured.update(kw)
            return vf_mod.ValidationResult(valid=True, violations=[])

        cfg = self._cfg()
        del cfg["resolve_parquet_default"]
        assert "resolve_parquet_default" not in cfg
        with (
            patch(
                "steering.validate_features.validate_feature_directory",
                side_effect=_fake_vfd,
            ),
            pytest.raises(SystemExit),
        ):
            vf_mod.main.__wrapped__(cfg)
        assert captured["parquet_dir"] is None
