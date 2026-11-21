"""Unit tests for steering/feature_paths.py — slice 01 Phase 1 audit.

TDD context: These tests pin the audit fixes for items A-1, A-2, A-3, A-4
in plan.md. They should FAIL against the unfixed production code and PASS
after the fixes land.

Test seed: deterministic; no randomness needed for these unit tests.
"""

from pathlib import Path

import pytest

from steering.feature_paths import (
    DatasetPhase,
    FeatureDirectoryComponents,
    FeatureDirectoryMismatchError,
    PoolingMode,
    _parse_strict_bool_metadata,
    build_feature_dir,
    parse_feature_dir,
    resolve_output_dir,
    sanitize_model_name,
    unsanitize_model_name,
)
from steering.sae_resolver import ActivationType, SAEWidth

# ---------------------------------------------------------------------------
# A-1: sanitize_model_name promotion (manifest §1.5)
# ---------------------------------------------------------------------------


class TestSanitizeModelNamePublicAPI:
    """Pin the public sanitize_model_name / unsanitize_model_name behavior."""

    def test_public_sanitize_exists(self):
        """The public name must be exported."""
        assert callable(sanitize_model_name)

    def test_public_unsanitize_exists(self):
        """The public inverse must be exported."""
        assert callable(unsanitize_model_name)

    def test_public_round_trip(self):
        """unsanitize(sanitize(x)) == x."""
        for name in ["gemma-2-2b", "google/gemma-2-2b", "a/b/c", "x_y_z"]:
            assert unsanitize_model_name(sanitize_model_name(name)) == name


# ---------------------------------------------------------------------------
# A-2: resolve_output_dir validates dataset_phase
# ---------------------------------------------------------------------------


class TestResolveOutputDirInvalidPhase:
    """resolve_output_dir must raise a clear error for invalid dataset_phase."""

    def test_invalid_phase_raises_with_phase_list(self, tmp_path: Path):
        """An unknown phase string must produce a typed ValueError listing valid phases."""
        with pytest.raises(ValueError, match="dataset_phase"):
            resolve_output_dir(
                output_dir=None,
                feature_root=str(tmp_path),
                segment_size=128,
                layer=8,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="gemma-2-2b",
                dataset_phase="totally_invalid_phase",
            )

    def test_typo_phase_lists_valid_options(self, tmp_path: Path):
        """The error message must list valid phases."""
        with pytest.raises(ValueError) as exc_info:
            resolve_output_dir(
                output_dir=None,
                feature_root=str(tmp_path),
                segment_size=128,
                layer=8,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="gemma-2-2b",
                dataset_phase="safetysft",  # missing underscore
            )
        msg = str(exc_info.value)
        assert "base" in msg or "standard_sft" in msg or "valid" in msg.lower()


# ---------------------------------------------------------------------------
# A-3: parse_feature_dir error chaining preserves __cause__
# ---------------------------------------------------------------------------


class TestParseFeatureDirErrorChaining:
    """parse_feature_dir must use `from exc` for typed reraises."""

    def test_pooling_error_preserves_cause(self, tmp_path: Path):
        """Bad pooling label must chain the original ValueError via __cause__."""
        bad_path = (
            tmp_path
            / "s128"
            / "l8"
            / "a_res"
            / "p_NOT_A_POOLING_MODE"
            / "f16k"
            / "m_gemma-2-2b"
            / "base"
        )
        with pytest.raises(ValueError) as exc_info:
            parse_feature_dir(bad_path)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_width_error_preserves_cause(self, tmp_path: Path):
        """Bad width label must chain via __cause__."""
        bad_path = (
            tmp_path
            / "s128"
            / "l8"
            / "a_res"
            / "p_last_token"
            / "fNOT_A_WIDTH"
            / "m_gemma-2-2b"
            / "base"
        )
        with pytest.raises(ValueError) as exc_info:
            parse_feature_dir(bad_path)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_phase_error_preserves_cause(self, tmp_path: Path):
        """Bad phase label must chain via __cause__."""
        bad_path = (
            tmp_path
            / "s128"
            / "l8"
            / "a_res"
            / "p_last_token"
            / "f16k"
            / "m_gemma-2-2b"
            / "NOT_A_PHASE"
        )
        with pytest.raises(ValueError) as exc_info:
            parse_feature_dir(bad_path)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# A-4: verify_feature_dir context_independent metadata strict parsing
# ---------------------------------------------------------------------------


class TestVerifyFeatureDirContextIndependentMetaParsing:
    """meta_context_independent_features must be strict 'true'/'false', not silent fallback."""

    def test_unknown_string_raises(self, tmp_path: Path):
        """Garbage value like 'yes' or 'tru' must raise (not silently default to False).

        Pins the strict-parser behavior directly. The real parser is also
        invoked deep inside ``verify_feature_dir``; this test targets the
        strict helper so the contract is locked at the unit level.
        """
        with pytest.raises((ValueError, FeatureDirectoryMismatchError)):
            _parse_strict_bool_metadata("yes", field="context_independent_features")

        with pytest.raises((ValueError, FeatureDirectoryMismatchError)):
            _parse_strict_bool_metadata("True", field="context_independent_features")

        with pytest.raises((ValueError, FeatureDirectoryMismatchError)):
            _parse_strict_bool_metadata("tru", field="context_independent_features")

    def test_valid_lowercase_true_false(self):
        """Lowercase 'true'/'false' must parse cleanly."""
        assert _parse_strict_bool_metadata("true", field="x") is True
        assert _parse_strict_bool_metadata("false", field="x") is False


# ---------------------------------------------------------------------------
# A-9 (basic round-trip): build_feature_dir + parse_feature_dir
# ---------------------------------------------------------------------------


class TestBuildParseRoundTrip:
    """Pin canonical bidirectional path encoding."""

    def test_simple_roundtrip(self, tmp_path: Path):
        """Build then parse returns matching components."""
        path = build_feature_dir(
            root=tmp_path,
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )
        parsed = parse_feature_dir(path)
        assert parsed == FeatureDirectoryComponents(
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )

    def test_hf_style_model_roundtrip(self, tmp_path: Path):
        """HuggingFace-style 'org/model' round-trips via _S/_underscore encoding."""
        path = build_feature_dir(
            root=tmp_path,
            segment_size=64,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_131K,
            model="google/gemma-2-2b",
            dataset_phase=DatasetPhase.STANDARD_SFT,
            context_independent_features=True,
        )
        parsed = parse_feature_dir(path)
        assert parsed.model == "google/gemma-2-2b"
        assert parsed.context_independent_features is True
