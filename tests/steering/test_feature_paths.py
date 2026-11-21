"""Unit tests for steering.feature_paths.

Run with:
    uv run pytest tests/test_feature_paths.py -v
"""

import json
from pathlib import Path

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.feature_paths import (
    _PHASE_TO_SFT_DIR,
    _SFT_PHASE_INFO,
    STEERED_JSONL_PLACEHOLDER,
    DatasetPhase,
    FeatureDirectoryComponents,
    PoolingMode,
    build_feature_dir,
    check_sft_data_ready,
    parse_feature_dir,
    resolve_data_dir,
    resolve_output_dir,
    sft_phase_for,
    sft_phase_split,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.sft_schema import SFTSplit
from tasks.mixtures import SFTMixture, sft_val_family

# ---------------------------------------------------------------------------
# Task 1: DatasetPhase enum
# ---------------------------------------------------------------------------


class TestDatasetPhase:
    def test_enum_values(self):
        assert DatasetPhase.BASE == "base"
        assert DatasetPhase.STANDARD_SFT == "standard_sft"
        assert DatasetPhase.STANDARD_SFT_VAL == "standard_sft_val"
        assert DatasetPhase.SAFETY_SFT == "safety_sft"
        assert DatasetPhase.SAFETY_SFT_VAL == "safety_sft_val"
        assert DatasetPhase.CAPABILITIES_SFT == "capabilities_sft"
        assert DatasetPhase.CAPABILITIES_SFT_VAL == "capabilities_sft_val"
        assert DatasetPhase.STANDARD_STEERED_SFT == "standard_steered_sft"
        assert DatasetPhase.STANDARD_STEERED_BALANCED_SFT == "standard_steered_balanced_sft"

    def test_from_string(self):
        assert DatasetPhase("base") is DatasetPhase.BASE
        assert DatasetPhase("standard_sft") is DatasetPhase.STANDARD_SFT
        assert DatasetPhase("standard_sft_val") is DatasetPhase.STANDARD_SFT_VAL
        assert DatasetPhase("safety_sft") is DatasetPhase.SAFETY_SFT
        assert DatasetPhase("safety_sft_val") is DatasetPhase.SAFETY_SFT_VAL
        assert DatasetPhase("capabilities_sft") is DatasetPhase.CAPABILITIES_SFT
        assert DatasetPhase("capabilities_sft_val") is DatasetPhase.CAPABILITIES_SFT_VAL
        assert DatasetPhase("standard_steered_sft") is DatasetPhase.STANDARD_STEERED_SFT
        assert (
            DatasetPhase("standard_steered_balanced_sft")
            is DatasetPhase.STANDARD_STEERED_BALANCED_SFT
        )

    def test_round_trip(self):
        for member in DatasetPhase:
            assert DatasetPhase(member.value) is member

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            DatasetPhase("nonexistent_phase")

    def test_is_str(self):
        """StrEnum members should be usable as plain strings."""
        assert isinstance(DatasetPhase.BASE, str)
        assert f"phase={DatasetPhase.BASE}" == "phase=base"


# ---------------------------------------------------------------------------
# Task 2: build_feature_dir
# ---------------------------------------------------------------------------


class TestBuildFeatureDir:
    """Tests for build_feature_dir path construction."""

    def test_basic_residual_path(self):
        result = build_feature_dir(
            root="/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        expected = Path("/data/features/s128/l12/a_res/p_last_token/f16k/m_gemma-2-2b/base")
        assert result == expected

    def test_returns_path_object(self):
        result = build_feature_dir(
            root="/tmp/feat",
            segment_size=64,
            layer=0,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert isinstance(result, Path)

    # -- All activation types --

    def test_residual_activation_label(self):
        result = build_feature_dir(
            root="/r",
            segment_size=1,
            layer=0,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="m",
            dataset_phase=DatasetPhase.BASE,
        )
        assert "a_res" in str(result)

    def test_mlp_activation_label(self):
        result = build_feature_dir(
            root="/r",
            segment_size=1,
            layer=0,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="m",
            dataset_phase=DatasetPhase.BASE,
        )
        assert "a_mlp" in str(result)

    def test_attention_activation_label(self):
        result = build_feature_dir(
            root="/r",
            segment_size=1,
            layer=0,
            activation_type=ActivationType.attention,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="m",
            dataset_phase=DatasetPhase.BASE,
        )
        assert "a_att" in str(result)

    # -- All widths --

    @pytest.mark.parametrize(
        "width",
        list(SAEWidth),
        ids=[w.name for w in SAEWidth],
    )
    def test_all_widths(self, width: SAEWidth):
        result = build_feature_dir(
            root="/r",
            segment_size=1,
            layer=0,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=width,
            model="m",
            dataset_phase=DatasetPhase.BASE,
        )
        assert f"f{width.value}" in str(result)

    # -- All phases --

    @pytest.mark.parametrize(
        "phase",
        list(DatasetPhase),
        ids=[p.name for p in DatasetPhase],
    )
    def test_all_phases(self, phase: DatasetPhase):
        result = build_feature_dir(
            root="/r",
            segment_size=1,
            layer=0,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="m",
            dataset_phase=phase,
        )
        # Phase is the last component
        assert result.name == phase.value

    # -- All pooling modes --

    @pytest.mark.parametrize(
        "pooling",
        list(PoolingMode),
        ids=[p.name for p in PoolingMode],
    )
    def test_all_pooling_modes(self, pooling: PoolingMode):
        result = build_feature_dir(
            root="/r",
            segment_size=1,
            layer=0,
            activation_type=ActivationType.residual,
            pooling=pooling,
            width=SAEWidth.W_16K,
            model="m",
            dataset_phase=DatasetPhase.BASE,
        )
        assert f"p_{pooling.value}" in str(result)

    # -- Validation errors --

    def test_segment_size_zero_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_size"):
            build_feature_dir(
                root="/r",
                segment_size=0,
                layer=0,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="m",
                dataset_phase=DatasetPhase.BASE,
            )

    def test_segment_size_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_size"):
            build_feature_dir(
                root="/r",
                segment_size=-5,
                layer=0,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="m",
                dataset_phase=DatasetPhase.BASE,
            )

    def test_layer_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="layer"):
            build_feature_dir(
                root="/r",
                segment_size=1,
                layer=-1,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="m",
                dataset_phase=DatasetPhase.BASE,
            )

    def test_model_empty_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="model"):
            build_feature_dir(
                root="/r",
                segment_size=1,
                layer=0,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="",
                dataset_phase=DatasetPhase.BASE,
            )

    def test_model_whitespace_only_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="model"):
            build_feature_dir(
                root="/r",
                segment_size=1,
                layer=0,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model="   ",
                dataset_phase=DatasetPhase.BASE,
            )


# ---------------------------------------------------------------------------
# Task 3: parse_feature_dir
# ---------------------------------------------------------------------------


class TestParseFeatureDir:
    """Tests for parse_feature_dir path parsing."""

    # -- Round-trip tests --

    @pytest.mark.parametrize(
        "phase",
        list(DatasetPhase),
        ids=[p.name for p in DatasetPhase],
    )
    def test_round_trip_all_phases(self, phase: DatasetPhase):
        path = build_feature_dir(
            root="/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=phase,
        )
        parsed = parse_feature_dir(path)
        assert parsed.segment_size == 128
        assert parsed.layer == 12
        assert parsed.activation_type is ActivationType.residual
        assert parsed.pooling is PoolingMode.last_token
        assert parsed.width is SAEWidth.W_16K
        assert parsed.model == "gemma-2-2b"
        assert parsed.dataset_phase is phase

    @pytest.mark.parametrize(
        "act",
        list(ActivationType),
        ids=[a.name for a in ActivationType],
    )
    def test_round_trip_all_activations(self, act: ActivationType):
        path = build_feature_dir(
            root="/root",
            segment_size=64,
            layer=5,
            activation_type=act,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_65K,
            model="gemma-2-9b",
            dataset_phase=DatasetPhase.STANDARD_SFT,
        )
        parsed = parse_feature_dir(path)
        assert parsed.activation_type is act

    @pytest.mark.parametrize(
        "width",
        list(SAEWidth),
        ids=[w.name for w in SAEWidth],
    )
    def test_round_trip_all_widths(self, width: SAEWidth):
        path = build_feature_dir(
            root="/root",
            segment_size=32,
            layer=0,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.mean_pool,
            width=width,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.SAFETY_SFT,
        )
        parsed = parse_feature_dir(path)
        assert parsed.width is width

    @pytest.mark.parametrize(
        "pooling",
        list(PoolingMode),
        ids=[p.name for p in PoolingMode],
    )
    def test_round_trip_all_pooling(self, pooling: PoolingMode):
        path = build_feature_dir(
            root="/root",
            segment_size=256,
            layer=25,
            activation_type=ActivationType.attention,
            pooling=pooling,
            width=SAEWidth.W_131K,
            model="gemma-2-27b",
            dataset_phase=DatasetPhase.CAPABILITIES_SFT,
        )
        parsed = parse_feature_dir(path)
        assert parsed.pooling is pooling

    # -- String input --

    def test_string_input_works(self):
        path_str = "/data/features/s128/l12/a_res/p_last_token/f16k/m_gemma-2-2b/base"
        parsed = parse_feature_dir(path_str)
        assert parsed.segment_size == 128
        assert parsed.layer == 12
        assert parsed.activation_type is ActivationType.residual
        assert parsed.pooling is PoolingMode.last_token
        assert parsed.width is SAEWidth.W_16K
        assert parsed.model == "gemma-2-2b"
        assert parsed.dataset_phase is DatasetPhase.BASE

    # -- Malformed paths --

    def test_too_few_components_raises(self):
        with pytest.raises(ValueError, match="7"):
            parse_feature_dir("/data/s128/l12/a_res")

    def test_bad_segment_prefix_raises(self):
        with pytest.raises(ValueError, match="segment"):
            parse_feature_dir("/r/x128/l12/a_res/p_last_token/f16k/m_model/base")

    def test_bad_layer_prefix_raises(self):
        with pytest.raises(ValueError, match="layer"):
            parse_feature_dir("/r/s128/x12/a_res/p_last_token/f16k/m_model/base")

    def test_bad_activation_prefix_raises(self):
        with pytest.raises(ValueError, match="activation"):
            parse_feature_dir("/r/s128/l12/x_res/p_last_token/f16k/m_model/base")

    def test_bad_pooling_prefix_raises(self):
        with pytest.raises(ValueError, match="pooling"):
            parse_feature_dir("/r/s128/l12/a_res/x_last_token/f16k/m_model/base")

    def test_bad_width_prefix_raises(self):
        with pytest.raises(ValueError, match="width"):
            parse_feature_dir("/r/s128/l12/a_res/p_last_token/x16k/m_model/base")

    def test_bad_model_prefix_raises(self):
        with pytest.raises(ValueError, match="model"):
            parse_feature_dir("/r/s128/l12/a_res/p_last_token/f16k/x_model/base")

    def test_unknown_activation_label_raises(self):
        with pytest.raises(ValueError, match="activation"):
            parse_feature_dir("/r/s128/l12/a_zzz/p_last_token/f16k/m_model/base")

    def test_unknown_phase_raises(self):
        with pytest.raises(ValueError, match="phase"):
            parse_feature_dir("/r/s128/l12/a_res/p_last_token/f16k/m_model/bogus")


# ---------------------------------------------------------------------------
# Phase 3 Wave A / Task 5: regression tests — P0 model-name sanitization
#
# `build_feature_dir` must preserve `parse(build(x)) == x` for ANY non-empty
# model name string — including HuggingFace-style IDs with `/` (e.g.
# "google/gemma-2-2b") and names that incidentally contain the reserved escape
# sequence (e.g. "foo__bar", "bar_S"). A single `sanitize_model_name` helper
# in `steering/feature_paths.py` must encode on build and the inverse must be
# applied on parse.
# ---------------------------------------------------------------------------


class TestModelNameSanitization:
    """Round-trip preservation for model names with `/` and escape collisions."""

    def test_parse_build_roundtrip_with_slash_in_model_name(self):
        """HuggingFace-style `org/model` must survive build -> parse."""
        path = build_feature_dir(
            root="/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="google/gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        parsed = parse_feature_dir(path)
        assert parsed == FeatureDirectoryComponents(
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="google/gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
            context_independent_features=False,
        )

    def test_parse_build_does_not_create_extra_path_components_for_slash(self):
        """The encoded `m_*` component must remain a single filesystem segment."""
        path = build_feature_dir(
            root="/data",
            segment_size=64,
            layer=0,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="google/gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        # Must still be exactly 8 parts ("/", "data", s*, l*, a_*, p_*, f*, m_*, phase)
        assert len(path.parts) == 9, (
            f"Expected 9 path parts (including root), got {len(path.parts)}: {path.parts}"
        )
        # The m_* component should be a single path segment, not two.
        m_components = [p for p in path.parts if p.startswith("m_")]
        assert len(m_components) == 1, (
            f"Expected exactly one m_* component, got {m_components} in {path.parts}"
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "google/gemma-2-2b",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-0.5B",
            "org/nested/model",  # multiple slashes
            "gemma-2-2b",  # baseline: no slash, must still round-trip
            "no_slash_with_underscore",  # underscore but no slash
            "foo__bar",  # incidental `__` pair (escape-sentinel collision)
            "foo___bar",  # odd-count underscores
            "____",  # only underscores
            "a/b__c/d__e",  # mixed `/` and `__`
        ],
    )
    def test_parse_build_roundtrip_all_hf_style_names(self, model_name: str):
        """Round-trip must hold for any non-empty model string."""
        path = build_feature_dir(
            root="/data",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.mean_pool,
            width=SAEWidth.W_16K,
            model=model_name,
            dataset_phase=DatasetPhase.STANDARD_SFT,
        )
        parsed = parse_feature_dir(path)
        assert parsed.model == model_name, (
            f"Round-trip failed for {model_name!r}: got {parsed.model!r} (path was {path})"
        )

    def test_sanitize_escape_scheme_is_bijective_for_underscore_collision(self):
        """A name containing the sentinel escape literal must still round-trip.

        The escape scheme must handle the case where the input already contains
        the escape sequence — e.g. `foo__bar` (contains `__`) or `bar_Sbaz`
        (contains `_S`). The encode-then-decode must return the original string
        unchanged, with no information loss and no collisions with the escape
        of `/`.
        """
        tricky_names = [
            "foo__bar",
            "bar_Sbaz",  # looks like an encoded slash
            "__leading",
            "trailing__",
            "a_Sb_Sc",  # multiple potential-collisions
            "/slash/start",
            "end/slash/",
        ]
        for name in tricky_names:
            path = build_feature_dir(
                root="/data",
                segment_size=128,
                layer=0,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.last_token,
                width=SAEWidth.W_16K,
                model=name,
                dataset_phase=DatasetPhase.BASE,
            )
            parsed = parse_feature_dir(path)
            assert parsed.model == name, f"Bijection broken for {name!r}: got {parsed.model!r}"

    def test_sanitize_preserves_legacy_model_names_unchanged(self):
        """Backwards compat: existing dirs with plain names like `gemma-2-2b`
        must produce identical paths (no hidden encoding of '-' or digits)."""
        path = build_feature_dir(
            root="/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        # The literal path on disk should be unchanged for legacy callers.
        assert str(path) == "/data/features/s128/l12/a_res/p_last_token/f16k/m_gemma-2-2b/base"


# ---------------------------------------------------------------------------
# Task 8: resolve_output_dir
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    def test_auto_construct_base(self):
        result = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        assert result == Path("out/data/features/s128/l12/a_res/p_max_pool/f16k/m_gemma-2-2b/base")

    def test_auto_construct_sft(self):
        result = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase="standard_sft",
        )
        assert result == Path(
            "out/data/features/s128/l12/a_res/p_max_pool/f16k/m_gemma-2-2b/standard_sft"
        )

    def test_explicit_override(self):
        result = resolve_output_dir(
            output_dir="/custom/path",
            feature_root="out/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        assert result == Path("/custom/path")


# ---------------------------------------------------------------------------
# Task 8: resolve_data_dir
# ---------------------------------------------------------------------------


class TestResolveDataDir:
    def test_base_with_explicit_data_dir(self):
        result = resolve_data_dir(data_dir="/some/data", dataset_phase=None)
        assert result == Path("/some/data")

    def test_base_with_default(self):
        result = resolve_data_dir(data_dir=None, dataset_phase=None)
        # Should end with "base_data"
        assert result.name == "base_data"

    def test_sft_phase(self):
        result = resolve_data_dir(data_dir=None, dataset_phase="standard_sft")
        assert result.name == "standard"
        assert "sft_data" in str(result)

    def test_sft_phase_with_explicit_data_dir(self):
        result = resolve_data_dir(
            data_dir="/some/full_nanochat/data",
            dataset_phase="capabilities_sft",
        )
        assert result == Path("/some/full_nanochat/data")

    @pytest.mark.parametrize("mixture", ["standard_steered", "standard_steered_balanced"])
    def test_steered_train_phase_resolves_its_own_data_dir(self, mixture: str):
        result = resolve_data_dir(data_dir=None, dataset_phase=f"{mixture}_sft")
        assert result.name == mixture
        assert result.parent.name == "sft_data"

    @pytest.mark.parametrize("mixture", ["standard_steered", "standard_steered_balanced"])
    def test_steered_train_phase_readiness_names_its_own_train_conversion(
        self, tmp_path, mixture: str
    ):
        ok, msg = check_sft_data_ready(tmp_path / mixture, f"{mixture}_sft")
        assert not ok
        assert f"mixture={mixture} split=train" in msg
        # The steered TRAIN conversion is its own; only VAL borrows standard.
        assert "split=val" not in msg
        # A steered conversion cannot run without its JSONL, so pasting the
        # command as printed must not fail on a missing required override.
        assert f"steered_jsonl={STEERED_JSONL_PLACEHOLDER}" in msg

    @pytest.mark.parametrize(
        "phase",
        [
            DatasetPhase.STANDARD_SFT,
            DatasetPhase.STANDARD_SFT_VAL,
            DatasetPhase.SAFETY_SFT,
            DatasetPhase.CAPABILITIES_SFT,
        ],
    )
    def test_non_steered_readiness_never_mentions_steered_jsonl(self, tmp_path, phase):
        ok, msg = check_sft_data_ready(tmp_path / phase.value, phase.value)
        assert not ok
        assert "steered_jsonl" not in msg

    def test_phase_mapping(self):
        assert _PHASE_TO_SFT_DIR[DatasetPhase.STANDARD_SFT] == "standard"
        assert _PHASE_TO_SFT_DIR[DatasetPhase.STANDARD_SFT_VAL] == "standard_val"
        assert _PHASE_TO_SFT_DIR[DatasetPhase.SAFETY_SFT] == "safety"
        assert _PHASE_TO_SFT_DIR[DatasetPhase.SAFETY_SFT_VAL] == "safety_val"
        assert _PHASE_TO_SFT_DIR[DatasetPhase.CAPABILITIES_SFT] == "capabilities"
        assert _PHASE_TO_SFT_DIR[DatasetPhase.CAPABILITIES_SFT_VAL] == "capabilities_val"
        assert _PHASE_TO_SFT_DIR[DatasetPhase.STANDARD_STEERED_SFT] == "standard_steered"
        assert (
            _PHASE_TO_SFT_DIR[DatasetPhase.STANDARD_STEERED_BALANCED_SFT]
            == "standard_steered_balanced"
        )

    def test_phase_table_accessors_round_trip_every_sft_phase(self):
        assert set(_SFT_PHASE_INFO) == set(DatasetPhase) - {DatasetPhase.BASE}
        assert sft_phase_split(DatasetPhase.BASE) is None
        for phase, info in _SFT_PHASE_INFO.items():
            assert sft_phase_split(phase) is info.split
            assert sft_phase_for(info.mixture_dir, info.split) is phase

        for mixture in ("standard", "safety", "capabilities"):
            assert sft_phase_split(sft_phase_for(mixture, SFTSplit.TRAIN)) is SFTSplit.TRAIN
            assert sft_phase_split(sft_phase_for(mixture, SFTSplit.VAL)) is SFTSplit.VAL

    @pytest.mark.parametrize(
        "mixture", ["standard_steered", "standard_steered_balanced"]
    )
    def test_steered_mixtures_have_a_train_phase_but_no_val_phase(self, mixture: str):
        """Steered mixtures validate on standard's phase, so they mint no val one."""
        train_phase = sft_phase_for(mixture, SFTSplit.TRAIN)
        assert train_phase.value == f"{mixture}_sft"
        assert sft_phase_split(train_phase) is SFTSplit.TRAIN
        assert _PHASE_TO_SFT_DIR[train_phase] == mixture
        with pytest.raises(
            ValueError, match=rf"Unknown SFT phase identity \('{mixture}', 'val'\)"
        ):
            sft_phase_for(mixture, SFTSplit.VAL)
        assert sft_val_family(SFTMixture(mixture)) is SFTMixture.STANDARD


# ---------------------------------------------------------------------------
# Task 8: check_sft_data_ready
# ---------------------------------------------------------------------------


class TestCheckSFTDataReady:
    @pytest.mark.parametrize(
        ("phase", "mixture"),
        [
            (DatasetPhase.STANDARD_SFT_VAL, "standard"),
            (DatasetPhase.SAFETY_SFT_VAL, "safety"),
            (DatasetPhase.CAPABILITIES_SFT_VAL, "capabilities"),
        ],
    )
    def test_val_recovery_command_uses_real_mixture_and_val_split(
        self,
        tmp_path,
        phase: DatasetPhase,
        mixture: str,
    ) -> None:
        assert SFTMixture(mixture).value == mixture
        ok, msg = check_sft_data_ready(tmp_path / phase.value, phase.value)
        assert not ok
        assert f"mixture={mixture} split=val" in msg
        assert f"mixture={mixture}_val" not in msg

    def test_missing_dir(self, tmp_path):
        ok, msg = check_sft_data_ready(tmp_path / "nonexistent", "standard_sft")
        assert not ok
        assert "not found" in msg
        assert "convert_sft_to_parquet" in msg

    def test_incomplete(self, tmp_path):
        meta = {"mixture": "standard", "completed_shards": 3, "total_shards": 10}
        (tmp_path / "conversion_meta.json").write_text(json.dumps(meta))
        ok, msg = check_sft_data_ready(tmp_path, "standard_sft")
        assert not ok
        assert "incomplete" in msg.lower()
        assert "3" in msg and "10" in msg

    def test_complete(self, tmp_path):
        meta = {"mixture": "standard", "completed_shards": 10, "total_shards": 10}
        (tmp_path / "conversion_meta.json").write_text(json.dumps(meta))
        (tmp_path / "shard_00000.parquet").touch()
        ok, msg = check_sft_data_ready(tmp_path, "standard_sft")
        assert ok

    def test_empty_dir_no_parquets(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        ok, msg = check_sft_data_ready(tmp_path, "standard_sft")
        assert not ok
        assert "not found" in msg


# ---------------------------------------------------------------------------
# Task 11: End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline path verification."""

    def test_config_to_path_round_trip(self):
        """Simulate config -> build path -> parse path -> verify."""
        config = {
            "segment_size": 128,
            "layer": 12,
            "activation_type": "residual",
            "pooling": "max_pool",
            "model": "gemma-2-2b",
            "sae_num_features": "16k",
            "dataset_phase": None,
            "feature_root": "out/data/features",
            "output_dir": None,
        }

        output_dir = resolve_output_dir(
            output_dir=config["output_dir"],
            feature_root=config["feature_root"],
            segment_size=config["segment_size"],
            layer=config["layer"],
            activation_type=ActivationType(config["activation_type"]),
            pooling=PoolingMode(config["pooling"]),
            width=SAEWidth(config["sae_num_features"]),
            model=config["model"],
            dataset_phase=config["dataset_phase"],
        )

        parsed = parse_feature_dir(output_dir)
        assert parsed.segment_size == 128
        assert parsed.layer == 12
        assert parsed.activation_type == ActivationType.residual
        assert parsed.pooling == PoolingMode.max_pool
        assert parsed.width == SAEWidth.W_16K
        assert parsed.model == "gemma-2-2b"
        assert parsed.dataset_phase == DatasetPhase.BASE

    def test_sft_config_round_trip(self):
        """SFT dataset_phase flows through correctly."""
        output_dir = resolve_output_dir(
            output_dir=None,
            feature_root="features",
            segment_size=256,
            layer=20,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.mean_pool,
            width=SAEWidth.W_65K,
            model="gemma-2-9b",
            dataset_phase="safety_sft",
        )

        parsed = parse_feature_dir(output_dir)
        assert parsed.dataset_phase == DatasetPhase.SAFETY_SFT
        assert parsed.activation_type == ActivationType.mlp
        assert parsed.width == SAEWidth.W_65K


# ---------------------------------------------------------------------------
# Task 4: build_feature_dir appends context_independent leaf
# ---------------------------------------------------------------------------


class TestBuildFeatureDirContextIndependent:
    def _kwargs(self):
        return {
            "root": "out/data/features",
            "segment_size": 128,
            "layer": 8,
            "activation_type": ActivationType.residual,
            "pooling": PoolingMode.max_pool,
            "width": SAEWidth.W_16K,
            "model": "gemma-2-2b",
            "dataset_phase": DatasetPhase.BASE,
        }

    def test_default_no_leaf(self):
        path = build_feature_dir(**self._kwargs())
        # Legacy 7-component suffix after the root.
        parts = path.parts
        assert parts[-1] == "base"
        assert "context_independent" not in parts

    def test_ci_flag_appends_leaf(self):
        path = build_feature_dir(**self._kwargs(), context_independent_features=True)
        assert path.parts[-1] == "context_independent"
        assert path.parts[-2] == "base"

    def test_ci_flag_false_identical_to_default(self):
        path_default = build_feature_dir(**self._kwargs())
        path_flag = build_feature_dir(**self._kwargs(), context_independent_features=False)
        assert path_default == path_flag


# ---------------------------------------------------------------------------
# Task 5: parse_feature_dir recognizes context_independent leaf
# ---------------------------------------------------------------------------


class TestParseFeatureDirContextIndependent:
    def test_legacy_7_components_flag_false(self):
        path = "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        parsed = parse_feature_dir(path)
        assert parsed.context_independent_features is False
        assert parsed.dataset_phase.value == "base"

    def test_8_components_ci_leaf_sets_flag(self):
        path = (
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/context_independent"
        )
        parsed = parse_feature_dir(path)
        assert parsed.context_independent_features is True
        assert parsed.dataset_phase.value == "base"
        assert parsed.segment_size == 128
        assert parsed.layer == 8

    def test_8_components_wrong_leaf_parses_preceding_7(self):
        """Arbitrary trailing component is NOT a ci leaf; it becomes an extra
        path segment. parse_feature_dir only strips the literal "context_independent"
        leaf, so the last 7 components here yield structural parse failures."""
        path = "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/nonsense"
        with pytest.raises(ValueError):
            parse_feature_dir(path)

    def test_build_parse_round_trip_ci_true(self):
        kwargs = {
            "root": "out/data/features",
            "segment_size": 64,
            "layer": 12,
            "activation_type": ActivationType.mlp,
            "pooling": PoolingMode.mean_pool,
            "width": SAEWidth.W_32K,
            "model": "gemma-2-2b",
            "dataset_phase": DatasetPhase.BASE,
        }
        path = build_feature_dir(**kwargs, context_independent_features=True)  # ty: ignore[invalid-argument-type]
        parsed = parse_feature_dir(path)
        assert parsed.context_independent_features is True
        assert parsed.segment_size == 64
        assert parsed.layer == 12
        assert parsed.activation_type is ActivationType.mlp
        assert parsed.pooling is PoolingMode.mean_pool

    def test_build_parse_round_trip_ci_false(self):
        kwargs = {
            "root": "out/data/features",
            "segment_size": 64,
            "layer": 12,
            "activation_type": ActivationType.mlp,
            "pooling": PoolingMode.mean_pool,
            "width": SAEWidth.W_32K,
            "model": "gemma-2-2b",
            "dataset_phase": DatasetPhase.BASE,
        }
        path = build_feature_dir(**kwargs, context_independent_features=False)  # ty: ignore[invalid-argument-type]
        parsed = parse_feature_dir(path)
        assert parsed.context_independent_features is False


# ---------------------------------------------------------------------------
# Task 6: resolve_output_dir threads context_independent_features
# ---------------------------------------------------------------------------


class TestResolveOutputDirContextIndependent:
    def test_explicit_override_ignores_flag(self):
        # When output_dir is set explicitly, the flag does NOT change it.
        out = resolve_output_dir(
            output_dir="/custom/dir",
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
            context_independent_features=True,
        )
        assert str(out) == "/custom/dir"

    def test_auto_resolve_with_ci_flag_true(self):
        out = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
            context_independent_features=True,
        )
        assert out.parts[-1] == "context_independent"
        assert out.parts[-2] == "base"

    def test_auto_resolve_without_flag_unchanged(self):
        out = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        assert out.parts[-1] == "base"
        assert "context_independent" not in out.parts
