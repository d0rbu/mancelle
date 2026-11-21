import dataclasses
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

from steering.eval.axbench_base.config import (
    MAX_RAW_PERCENTILE,
    AxBenchBaseConfig,
    JudgeConfig,
    LayerHooksConfig,
    LLMClientConfig,
    PositiveQcThresholds,
    SynthConfig,
    bare_model_id,
    derive_positive_qc_thresholds,
    effective_extraction_model_id,
    validate_config,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
)
from steering.eval.cluster_config import (
    CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
    CLUSTERED_AXBENCH_BASE_STAGE4_METHODS,
)
from steering.feature_paths import sanitize_model_name
from tests.runtime_attribute_helpers import set_runtime_attribute


def make_valid_config(**overrides):
    """Factory for a valid config; override any field with kwargs.

    Only the 6 mandatory fields (+ ``run_tag`` for test clarity) are passed
    explicitly; everything else inherits from the ``AxBenchBaseConfig``
    defaults. Override any field with kwargs.
    """
    base = AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root=None,
        run_tag="test",
    )
    if overrides:
        return base.replace(**overrides)
    return base


# Canonical gemma-3 feature_dirs. The ``m_{model}`` segment drives the
# model <-> feature_dir consistency check in validate_config (issue #657), so a
# gemma-3 extraction model must be paired with a gemma-3 feature_dir to form a
# coherent config; the default make_valid_config feature_dir is gemma-2-2b.
_GEMMA3_PT_FEATURE_DIR = (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base"
)
_GEMMA3_IT_FEATURE_DIR = (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-it/base"
)
# The gemma-scope-2 Neuronpedia source matching the l22 / f65k gemma-3
# feature_dirs above (the source encodes layer + width, not the -pt/-it model
# suffix, so both feature_dirs pair with the same source). validate_config
# cross-checks source layer/width against feature_dir (issue #657).
_GEMMA3_SOURCE = "22-gemmascope-2-res-65k"


def test_valid_config_passes():
    validate_config(make_valid_config())


@pytest.mark.parametrize("output_root", ["", "   ", 42])
def test_validate_config_rejects_invalid_output_root(output_root):
    with pytest.raises(BeartypeCallHintParamViolation, match="output_root"):
        make_valid_config(output_root=output_root)


@pytest.mark.parametrize(
    "output_root", [None, "relative/custom-root", "/tmp/custom-root"]
)
def test_validate_config_accepts_null_or_nonempty_output_root(output_root):
    validate_config(make_valid_config(output_root=output_root))


def test_invalid_concept_feature_mode():
    with pytest.raises(BeartypeCallHintParamViolation, match="concept_feature_mode"):
        make_valid_config(concept_feature_mode="bogus")


def test_invalid_qc_fallback():
    with pytest.raises(BeartypeCallHintParamViolation, match="qc_fallback_on_exhaust"):
        make_valid_config(qc_fallback_on_exhaust="launch_missiles")


def test_empty_validation_shards():
    with pytest.raises(BeartypeCallHintParamViolation, match="validation_shards"):
        make_valid_config(validation_shards=[])


def test_n_positives_zero_invalid():
    with pytest.raises(BeartypeCallHintParamViolation, match="n_positives_per_concept"):
        make_valid_config(n_positives_per_concept=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_prefix_chars": -1}, "non-negative integer"),
        ({"min_prefix_chars": True}, "non-negative integer"),
        ({"min_prefix_chars": 1.5}, "non-negative integer"),
        ({"max_prefix_chars": 0}, "positive integer"),
        ({"max_prefix_chars": False}, "positive integer"),
        ({"max_prefix_chars": 2048.0}, "positive integer"),
        (
            {"min_prefix_chars": 2049, "max_prefix_chars": 2048},
            "must be <= max_prefix_chars",
        ),
    ],
)
def test_validate_config_enforces_shared_prefix_character_bounds(
    overrides: dict[str, object],
    message: str,
) -> None:
    if overrides == {"min_prefix_chars": 2049, "max_prefix_chars": 2048}:
        cfg = make_valid_config(**overrides)
        with pytest.raises(AxBenchBaseError, match=message):
            validate_config(cfg)
    else:
        with pytest.raises(BeartypeCallHintParamViolation):
            make_valid_config(**overrides)


@pytest.mark.parametrize(
    ("min_prefix_chars", "max_prefix_chars"),
    [(0, 1), (1, 1), (100, 2048), (2048, 2048)],
)
def test_validate_config_accepts_prefix_character_boundaries(
    min_prefix_chars: int,
    max_prefix_chars: int,
) -> None:
    validate_config(
        make_valid_config(
            min_prefix_chars=min_prefix_chars,
            max_prefix_chars=max_prefix_chars,
        )
    )


@pytest.mark.parametrize("value", [True, False, 1.0, 2.5, "3", None])
def test_n_positives_requires_a_real_positive_integer(value):
    with pytest.raises(BeartypeCallHintParamViolation, match="n_positives_per_concept"):
        make_valid_config(n_positives_per_concept=value)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "8", None])
def test_positive_max_attempts_requires_a_real_positive_integer(value):
    with pytest.raises(
        BeartypeCallHintParamViolation, match="positive_max_attempts_per_seed"
    ):
        make_valid_config(positive_max_attempts_per_seed=value)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "2000"])
def test_positive_zero_hit_bail_requires_null_or_positive_integer(value):
    with pytest.raises(
        BeartypeCallHintParamViolation, match="positive_zero_hit_bail_after"
    ):
        make_valid_config(positive_zero_hit_bail_after=value)


@pytest.mark.parametrize("value", [None, 1, 2000])
def test_positive_zero_hit_bail_accepts_boundaries(value):
    validate_config(make_valid_config(positive_zero_hit_bail_after=value))


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "16", None])
def test_qc_batch_size_requires_a_real_positive_integer(value):
    with pytest.raises(BeartypeCallHintParamViolation, match="qc_batch_size"):
        make_valid_config(qc_batch_size=value)


def test_positive_qc_threshold_defaults_are_exactly_30_90_60():
    cfg = make_valid_config()

    thresholds = derive_positive_qc_thresholds(
        cfg.accept_threshold,
        cfg.positive_prefix_headroom_ceiling,
    )

    assert thresholds == PositiveQcThresholds(
        accept_fraction=0.3,
        headroom_ceiling_fraction=0.9,
        required_delta=30.0,
        completion_ceiling=90.0,
        maximum_seed_percentile=60.0,
    )
    assert MAX_RAW_PERCENTILE == 99.5


def test_positive_qc_ceiling_one_uses_representable_99_point_5_maximum():
    thresholds = derive_positive_qc_thresholds(0.3, 1.0)

    assert thresholds.required_delta == 30.0
    assert thresholds.completion_ceiling == 99.5
    assert thresholds.maximum_seed_percentile == 69.5


def test_positive_qc_zero_cap_is_valid_for_silent_prefixes():
    thresholds = derive_positive_qc_thresholds(0.995, 1.0)

    assert thresholds.required_delta == 99.5
    assert thresholds.completion_ceiling == 99.5
    assert thresholds.maximum_seed_percentile == 0.0
    validate_config(
        make_valid_config(
            accept_threshold=0.995,
            positive_prefix_headroom_ceiling=1.0,
        )
    )


@pytest.mark.parametrize(
    ("accept_threshold", "headroom_ceiling"),
    [
        (0.0, 0.9),
        (-0.1, 0.9),
        (0.9, 0.9),
        (0.91, 0.9),
        (0.3, 0.0),
        (0.3, -0.1),
        (0.3, 1.0000001),
        (1.0, 1.0),
    ],
)
def test_positive_qc_rejects_invalid_fraction_ordering(
    accept_threshold,
    headroom_ceiling,
):
    with pytest.raises(
        AxBenchBaseError,
        match=(
            "finite real percentile fraction|"
            "0 < accept_threshold < positive_prefix_headroom_ceiling <= 1"
        ),
    ):
        derive_positive_qc_thresholds(accept_threshold, headroom_ceiling)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("accept_threshold", float("nan")),
        ("accept_threshold", float("inf")),
        ("accept_threshold", float("-inf")),
        ("positive_prefix_headroom_ceiling", float("nan")),
        ("positive_prefix_headroom_ceiling", float("inf")),
        ("positive_prefix_headroom_ceiling", float("-inf")),
    ],
)
def test_positive_qc_rejects_nonfinite_values(field_name, value):
    values = {
        "accept_threshold": 0.3,
        "positive_prefix_headroom_ceiling": 0.9,
    }
    values[field_name] = value

    with pytest.raises(
        AxBenchBaseError,
        match=rf"{field_name} must be a finite real percentile fraction",
    ):
        derive_positive_qc_thresholds(
            values["accept_threshold"],
            values["positive_prefix_headroom_ceiling"],
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("accept_threshold", True),
        ("accept_threshold", False),
        ("accept_threshold", "0.3"),
        ("accept_threshold", None),
        ("positive_prefix_headroom_ceiling", True),
        ("positive_prefix_headroom_ceiling", False),
        ("positive_prefix_headroom_ceiling", "0.9"),
        ("positive_prefix_headroom_ceiling", None),
    ],
)
def test_positive_qc_rejects_bool_and_nonnumeric_values(field_name, value):
    values = {
        "accept_threshold": 0.3,
        "positive_prefix_headroom_ceiling": 0.9,
    }
    values[field_name] = value

    with pytest.raises(
        AxBenchBaseError,
        match=rf"{field_name} must be a finite real percentile fraction",
    ):
        derive_positive_qc_thresholds(
            values["accept_threshold"],
            values["positive_prefix_headroom_ceiling"],
        )


def test_positive_qc_rejects_fraction_order_that_is_still_unrepresentable():
    with pytest.raises(
        AxBenchBaseError,
        match="no representable prefix-headroom band.*99.5",
    ):
        derive_positive_qc_thresholds(0.996, 0.999)


def test_seed_pool_train_ratio_out_of_range():
    with pytest.raises(BeartypeCallHintParamViolation, match="seed_pool_train_ratio"):
        make_valid_config(seed_pool_train_ratio=1.5)


def test_max_not_published_description_rate_out_of_range():
    for bad in (-0.1, 1.5):
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="max_not_published_description_rate",
        ):
            make_valid_config(max_not_published_description_rate=bad)


def test_max_not_published_description_rate_boundaries_ok():
    for ok in (0.0, 0.5, 1.0):
        validate_config(make_valid_config(max_not_published_description_rate=ok))


def test_max_not_published_description_rate_none_gives_clean_error():
    """The required rate rejects a natural null-to-disable guess at construction."""
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="max_not_published_description_rate",
    ):
        make_valid_config(max_not_published_description_rate=None)


def test_max_missing_description_rate_out_of_range():
    for bad in (-0.1, 1.5, None):
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="max_missing_description_rate",
        ):
            make_valid_config(max_missing_description_rate=bad)


def test_max_failed_description_rate_out_of_range():
    for bad in (-0.1, 1.5, None):
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="max_failed_description_rate",
        ):
            make_valid_config(max_failed_description_rate=bad)


def test_max_failed_description_rate_boundaries_ok():
    for ok in (0.0, 0.5, 1.0):
        validate_config(make_valid_config(max_failed_description_rate=ok))


def test_min_source_mismatch_features_negative_invalid():
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="min_source_mismatch_features",
    ):
        make_valid_config(min_source_mismatch_features=-1)


def test_min_source_mismatch_features_none_gives_clean_error():
    """The required floor rejects a natural null-to-disable guess at construction."""
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="min_source_mismatch_features",
    ):
        make_valid_config(min_source_mismatch_features=None)


def test_empty_extraction_model_id_gives_clean_error():
    """Nulling the required model ID fails at the owning constructor."""
    with pytest.raises(BeartypeCallHintParamViolation, match="neuronpedia_model_id"):
        make_valid_config(
            neuronpedia_model_id=None,
            feature_extraction_model_id=None,
        )


def test_extraction_tokenizer_accepts_bare_and_path_spellings():
    """A bare or path-prefixed tokenizer that normalizes to the model's
    tokenizer basename is accepted (not a false mismatch)."""
    for tok in (
        "gemma-3-4b-pt",
        "/workspace/models/gemma-3-4b-pt",
        "google/gemma-3-4b-pt",
    ):
        validate_config(
            make_valid_config(
                feature_dir=_GEMMA3_PT_FEATURE_DIR,
                feature_extraction_model_id="gemma-3-4b-pt",
                neuronpedia_model_id="gemma-3-4b",
                neuronpedia_source=_GEMMA3_SOURCE,
                extraction_tokenizer_id=tok,
            )
        )


def test_extraction_tokenizer_must_match_gemmascope2_model():
    """A gemma-scope-2 extraction model with a mismatched extraction tokenizer
    is rejected -- the issue #657 misconfig where a gemma-3 run keeps the
    gemma-2-2b default tokenizer."""
    cfg = make_valid_config(
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    with pytest.raises(AxBenchBaseError, match="extraction_tokenizer_id"):
        validate_config(cfg)


def test_extraction_tokenizer_mismatch_within_gemma3_rejected():
    """Even a sibling gemma-3 tokenizer (-it for a -pt model) is rejected: the
    tokenizer must be the extraction model's own."""
    cfg = make_valid_config(
        feature_extraction_model_id="gemma-3-4b-pt",
        extraction_tokenizer_id="google/gemma-3-4b-it",
    )
    with pytest.raises(AxBenchBaseError, match="extraction_tokenizer_id"):
        validate_config(cfg)


def test_extraction_tokenizer_matches_gemmascope2_model_ok():
    validate_config(
        make_valid_config(
            feature_dir=_GEMMA3_PT_FEATURE_DIR,
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_model_id="gemma-3-4b",
            neuronpedia_source=_GEMMA3_SOURCE,
            extraction_tokenizer_id="google/gemma-3-4b-pt",
        )
    )


def test_extraction_tokenizer_resolves_via_neuronpedia_model_when_no_override():
    """With feature_extraction_model_id=None the check uses neuronpedia_model_id
    (mirrors runtime_state's `feature_extraction_model_id or neuronpedia_model_id`)."""
    validate_config(
        make_valid_config(
            feature_dir=_GEMMA3_IT_FEATURE_DIR,
            feature_extraction_model_id=None,
            neuronpedia_model_id="gemma-3-4b-it",
            neuronpedia_source=_GEMMA3_SOURCE,
            extraction_tokenizer_id="google/gemma-3-4b-it",
        )
    )
    with pytest.raises(AxBenchBaseError, match="extraction_tokenizer_id"):
        validate_config(
            make_valid_config(
                feature_dir=_GEMMA3_IT_FEATURE_DIR,
                feature_extraction_model_id=None,
                neuronpedia_model_id="gemma-3-4b-it",
                neuronpedia_source=_GEMMA3_SOURCE,
                extraction_tokenizer_id="google/gemma-2-2b",
            )
        )


def test_non_gemmascope2_model_skips_tokenizer_check():
    """v1 models (gemma-2-2b) are not in the gemma-scope-2 map, so this check
    does not apply -- the default gemma-2 config is unaffected."""
    validate_config(
        make_valid_config(
            neuronpedia_model_id="gemma-2-2b",
            extraction_tokenizer_id="google/gemma-2-2b",
        )
    )


def test_effective_extraction_model_id_prefers_feature_extraction_model():
    """The single #667 API/local split resolution: feature_extraction_model_id
    when set, else neuronpedia_model_id (shared by config, concept_list,
    runtime_state, and the Stage 0c fingerprint)."""
    assert (
        effective_extraction_model_id(
            make_valid_config(
                feature_extraction_model_id="gemma-3-4b-pt",
                neuronpedia_model_id="gemma-3-4b",
            )
        )
        == "gemma-3-4b-pt"
    )
    assert (
        effective_extraction_model_id(
            make_valid_config(
                feature_extraction_model_id=None,
                neuronpedia_model_id="gemma-2-2b",
            )
        )
        == "gemma-2-2b"
    )


def test_org_prefixed_feature_extraction_model_id_rejected():
    """An org-/path-prefixed feature_extraction_model_id is rejected at config
    time: the SAE resolver and Neuronpedia API are keyed by the bare name, so an
    org-prefixed id would pass the identity checks (compared bare) but crash at
    Stage 0c SAE load. (extraction_tokenizer_id may stay org-prefixed -- it is a
    full HF tokenizer id, exercised by other tests.)"""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,
        feature_extraction_model_id="google/gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        neuronpedia_source=_GEMMA3_SOURCE,
        extraction_tokenizer_id="google/gemma-3-4b-pt",
    )
    with pytest.raises(AxBenchBaseError, match="must be a BARE model id"):
        validate_config(cfg)


def test_feature_dir_model_mismatch_rejected():
    """validate_config fails loud when the SAE model encoded in feature_dir
    (the ``m_{model}`` segment) does not match the configured extraction model.

    This is the all-defaults issue #657 misconfig: a gemma-3 SAE evaluated with
    the gemma-2-2b defaults. It is the config-time backstop the tokenizer check
    CANNOT provide -- the default extraction model resolves to gemma-2-2b, a v1
    model absent from the gemma-scope-2 tokenizer map, so the tokenizer check is
    skipped and only this feature_dir check catches the mismatch."""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,
        neuronpedia_model_id="gemma-2-2b",
        feature_extraction_model_id=None,
    )
    with pytest.raises(AxBenchBaseError, match="Model mismatch") as exc_info:
        validate_config(cfg)
    message = str(exc_info.value)
    assert "gemma-3-4b-pt" in message  # the SAE model from feature_dir
    assert "gemma-2-2b" in message  # the configured extraction model
    assert "#657" in message


def test_feature_dir_model_match_accepted():
    """A gemma-3 feature_dir paired with the matching gemma-3-4b-pt extraction
    model (and its tokenizer) passes. The neuronpedia API id may still differ
    (gemma-3-4b), per the #667 API/local split."""
    validate_config(
        make_valid_config(
            feature_dir=_GEMMA3_PT_FEATURE_DIR,
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_model_id="gemma-3-4b",
            neuronpedia_source="22-gemmascope-2-res-65k",
            extraction_tokenizer_id="google/gemma-3-4b-pt",
        )
    )


def test_org_prefixed_neuronpedia_model_id_rejected():
    """An org-/path-prefixed neuronpedia_model_id is rejected: it is the
    Neuronpedia API namespace (used verbatim in the URL) and must be bare, or a
    correct-looking bare-comparison would pass while the runtime query hits the
    wrong URL."""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="google/gemma-3-4b",
        neuronpedia_source=_GEMMA3_SOURCE,
        extraction_tokenizer_id="google/gemma-3-4b-pt",
    )
    with pytest.raises(AxBenchBaseError, match="must be a BARE model id"):
        validate_config(cfg)


def test_non_canonical_feature_dir_rejected():
    """A non-canonical feature_dir that ``parse_feature_dir`` cannot read is a
    config error and fails loud in validate_config -- Stage 0a parses the layer,
    segment size, and SAE model from feature_dir, so a malformed path would
    break there anyway; failing at config-load time is earlier and clearer than
    deferring it. (The tokenizer is set consistently so this isolates the
    feature_dir failure, not the tokenizer check.)"""
    cfg = make_valid_config(
        feature_dir="out/data/features/test",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        extraction_tokenizer_id="google/gemma-3-4b-pt",
    )
    with pytest.raises(AxBenchBaseError, match="canonical feature path") as exc_info:
        validate_config(cfg)
    assert "out/data/features/test" in str(exc_info.value)


def test_source_layer_mismatch_rejected():
    """validate_config fails loud when neuronpedia_source's layer does not match
    feature_dir's layer. A same-model, wrong-layer source returns valid HTTP
    200s (never 404/empty), so it evades all three runtime rate guards and would
    silently mislabel every concept with another layer's descriptions (#657)."""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,  # layer 22
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        extraction_tokenizer_id="google/gemma-3-4b-pt",
        neuronpedia_source="9-gemmascope-2-res-65k",  # layer 9 != 22
    )
    with pytest.raises(AxBenchBaseError, match="layer") as exc_info:
        validate_config(cfg)
    assert "9-gemmascope-2-res-65k" in str(exc_info.value)


def test_source_width_mismatch_rejected():
    """...and when the source's SAE width does not match feature_dir's width
    (the canonical 16k-source-against-a-65k-SAE #657 case; indices < 16384
    return wrong-width 200s that trip no rate guard)."""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,  # f65k
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        extraction_tokenizer_id="google/gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-16k",  # 16k != 65k
    )
    with pytest.raises(AxBenchBaseError, match="width"):
        validate_config(cfg)


def test_org_prefixed_feature_dir_model_resolves_via_bare():
    """A feature_dir whose ``m_{model}`` segment decodes to an org-prefixed id
    (google/gemma-3-4b-pt) is passed BARE to resolve_neuronpedia_ids, like every
    sibling check, so it resolves instead of failing with a misleading
    'SAE unpublished' error. (feature_dir model ids are not subject to the bare
    org-prefix *rejection*; that applies to the resolver/API model fields.)"""
    san = sanitize_model_name("google/gemma-3-4b-pt")
    feature_dir = f"out/data/features/s128/l22/a_res/p_max_pool/f65k/m_{san}/base"
    validate_config(
        make_valid_config(
            feature_dir=feature_dir,
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_model_id="gemma-3-4b",
            neuronpedia_source=_GEMMA3_SOURCE,
            extraction_tokenizer_id="google/gemma-3-4b-pt",
        )
    )


def test_source_matching_layer_width_accepted():
    """A source whose layer and width both match feature_dir passes."""
    validate_config(
        make_valid_config(
            feature_dir=_GEMMA3_PT_FEATURE_DIR,
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_model_id="gemma-3-4b",
            extraction_tokenizer_id="google/gemma-3-4b-pt",
            neuronpedia_source=_GEMMA3_SOURCE,
        )
    )


def test_neuronpedia_model_id_wrong_sibling_rejected():
    """validate_config fails loud when neuronpedia_model_id (the API namespace
    actually queried) is a wrong-but-existing sibling of the SAE's model. A
    gemma-3-4b-pt SAE is published on Neuronpedia under model 'gemma-3-4b'; a
    run that queries the '-it' sibling namespace returns valid HTTP 200s for a
    DIFFERENT model's features, tripping no runtime guard (#657). The SAE's
    canonical Neuronpedia model is DERIVED from feature_dir via
    resolve_neuronpedia_ids, so neuronpedia_model_id can be validated even when
    feature_extraction_model_id shadows it in the model<->feature_dir check."""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b-it",  # wrong sibling namespace
        extraction_tokenizer_id="google/gemma-3-4b-pt",
        neuronpedia_source=_GEMMA3_SOURCE,
    )
    with pytest.raises(AxBenchBaseError, match="Neuronpedia model mismatch"):
        validate_config(cfg)


def test_neuronpedia_source_activation_mismatch_rejected():
    """A source whose activation type differs from feature_dir's a_{act} segment
    is rejected (a res feature_dir against an mlp source)."""
    cfg = make_valid_config(
        feature_dir=_GEMMA3_PT_FEATURE_DIR,  # a_res
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        extraction_tokenizer_id="google/gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-mlp-65k",  # mlp != res
    )
    with pytest.raises(AxBenchBaseError, match="Activation mismatch"):
        validate_config(cfg)


def test_neuronpedia_source_none_gives_clean_error():
    """The required source string rejects null at the owning constructor."""
    with pytest.raises(BeartypeCallHintParamViolation, match="neuronpedia_source"):
        make_valid_config(neuronpedia_source=None)


def test_neuronpedia_model_id_none_gives_clean_error():
    """The required Neuronpedia model string rejects null at construction."""
    with pytest.raises(BeartypeCallHintParamViolation, match="neuronpedia_model_id"):
        make_valid_config(
            feature_dir=_GEMMA3_PT_FEATURE_DIR,
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_model_id=None,
            neuronpedia_source=_GEMMA3_SOURCE,
            extraction_tokenizer_id="google/gemma-3-4b-pt",
        )


def test_default_gemma2_source_matches_default_feature_dir():
    """The default config (gemma-2-2b l8 f16k + 8-gemmascope-res-16k v1 source)
    is self-consistent -- the source-layer/width guard does not false-trip it."""
    validate_config(make_valid_config())


def test_unparseable_neuronpedia_source_rejected():
    """A neuronpedia_source that is not a recognized gemma-scope source fails
    loud: the AxBench-Base Neuronpedia path is gemma-scope-based, so an
    unparseable source cannot be cross-checked against the SAE and is itself a
    misconfiguration (issue #657)."""
    cfg = make_valid_config(neuronpedia_source="not-a-real-source")
    with pytest.raises(AxBenchBaseError, match="neuronpedia_source"):
        validate_config(cfg)


def test_feature_dir_none_gives_clean_error():
    """The required feature path rejects null at the owning constructor."""
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_dir"):
        make_valid_config(feature_dir=None)


def test_bare_model_id_strips_trailing_slash():
    """A local model/tokenizer path spelled with a trailing separator must
    resolve to its basename, not an empty string (which would spuriously
    fail-loud the tokenizer check or fail-open the model<->feature_dir check)."""
    assert bare_model_id("/models/gemma-3-4b-pt/") == "gemma-3-4b-pt"
    assert bare_model_id("google/gemma-2-2b") == "gemma-2-2b"
    assert bare_model_id("gemma-3-4b-pt") == "gemma-3-4b-pt"


def test_trailing_slash_tokenizer_path_not_falsely_rejected():
    """A trailing-slash tokenizer path that normalizes to the model's tokenizer
    basename is accepted, not a false mismatch (regression: bare_model_id used
    to return '' for a trailing slash)."""
    validate_config(
        make_valid_config(
            feature_dir=_GEMMA3_PT_FEATURE_DIR,
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_model_id="gemma-3-4b",
            neuronpedia_source=_GEMMA3_SOURCE,
            extraction_tokenizer_id="/models/gemma-3-4b-pt/",
        )
    )


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "7", None])
def test_max_retries_requires_a_real_nonnegative_integer(value):
    with pytest.raises(BeartypeCallHintParamViolation, match="max_retries"):
        make_valid_config(max_retries=value)


def test_max_retries_allows_zero():
    validate_config(make_valid_config(max_retries=0))


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "3", None])
def test_synth_transport_max_retries_requires_real_nonnegative_integer(value):
    with pytest.raises(BeartypeCallHintParamViolation, match="transport_max_retries"):
        SynthConfig(transport_max_retries=value)


def test_synth_transport_max_retries_allows_zero():
    base = make_valid_config()
    validate_config(
        base.replace(
            synth=dataclasses.replace(base.synth, transport_max_retries=0),
        )
    )


@pytest.mark.parametrize("value", ["300", None, True, False])
def test_synth_transport_retry_budget_requires_a_number(value: object) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="transport_retry_budget_seconds",
    ):
        SynthConfig(transport_retry_budget_seconds=cast(Any, value))


@pytest.mark.parametrize("value", [-1, -0.5, float("nan")])
def test_synth_transport_retry_budget_rejects_negative_and_nan(value: float) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="transport_retry_budget_seconds",
    ):
        SynthConfig(transport_retry_budget_seconds=value)


@pytest.mark.parametrize("value", [0, 0.0, 7.5, 300.0])
def test_synth_transport_retry_budget_accepts_nonnegative(value: float) -> None:
    base = make_valid_config()
    validate_config(
        base.replace(
            synth=dataclasses.replace(base.synth, transport_retry_budget_seconds=value),
        )
    )


def test_synth_transport_retry_budget_defaults_to_five_minutes() -> None:
    """The single-provider synth model needs to outlast upstream 429 bursts.

    A 7-second attempt-count tolerance loses a whole stage to one burst; if
    this default is ever dropped back to 0, that regression returns silently.
    """
    assert SynthConfig().transport_retry_budget_seconds == 300.0
    # The shared base stays opt-in: only synth pays the longer wait.
    assert LLMClientConfig().transport_retry_budget_seconds == 0.0


@pytest.mark.parametrize("value", [0, 0.0, 0.3, 0.7, 1, 1.0])
def test_verification_percentile_threshold_accepts_closed_unit_interval(
    value: object,
) -> None:
    validate_config(make_valid_config(verification_percentile_threshold=value))


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "0.3",
        [],
        {},
        -math.inf,
        math.inf,
        math.nan,
        -1.0,
        -1e-12,
        1.0 + 1e-12,
        2.0,
    ],
)
def test_verification_percentile_threshold_rejects_invalid_values(
    value: object,
) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="verification_percentile_threshold",
    ):
        make_valid_config(verification_percentile_threshold=value)


def test_judge_parse_max_retries_negative_rejected():
    """A negative ``judge.parse_max_retries`` (from a Hydra config typo)
    would silently turn into ``range(0)`` inside the Stage 5 judge and
    Stage 0c examples-QC, producing a 100% reject rate that looks
    identical to the original production bug. Validator rejects it
    early so operators see a clear error before any API calls."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parse_max_retries"):
        JudgeConfig(parse_max_retries=-1)


def test_judge_parse_max_retries_zero_valid():
    """``0`` is explicitly supported (one attempt, no retries). Not a
    typo: operators who want to minimize API cost and accept the
    parse-failure rate can set this deliberately."""
    cfg = make_valid_config(judge=JudgeConfig(parse_max_retries=0))
    validate_config(cfg)


def test_diffmean_layer_negative_invalid():
    with pytest.raises(
        BeartypeCallHintParamViolation, match="diffmean_and_probe_layer"
    ):
        LayerHooksConfig(diffmean_and_probe_layer=-1)


def test_concept_description_fetch_limit_negative_invalid():
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="concept_description_fetch_limit",
    ):
        make_valid_config(concept_description_fetch_limit=-1)


def test_concept_examples_qc_target_surviving_negative_invalid():
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="concept_examples_qc_target_surviving",
    ):
        make_valid_config(concept_examples_qc_target_surviving=-1)


@pytest.mark.parametrize("value", [0, 1, 5])
def test_concept_examples_qc_target_surviving_non_negative_valid(value):
    validate_config(make_valid_config(concept_examples_qc_target_surviving=value))


def test_verification_candidate_caps_negative_invalid():
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="verification_candidate_scan_limit",
    ):
        make_valid_config(verification_candidate_scan_limit=-1)
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="verification_candidate_attempt_limit",
    ):
        make_valid_config(verification_candidate_attempt_limit=-1)


@pytest.mark.parametrize("value", [None, 1, 2000])
def test_verification_zero_hit_bail_after_valid_values(value):
    """``None`` disables the bailout; any int >= 1 enables it."""
    validate_config(make_valid_config(verification_zero_hit_bail_after=value))


@pytest.mark.parametrize("value", [0, -1, -100])
def test_verification_zero_hit_bail_after_rejects_zero_and_negative(value):
    """0 is degenerate (would bail before any real attempt) and is rejected
    deliberately. Negative values are obviously invalid. The validator must
    name the field and the constraint so a Hydra typo surfaces clearly."""
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="verification_zero_hit_bail_after",
    ):
        make_valid_config(verification_zero_hit_bail_after=value)


def test_verification_api_model_default_is_ling_flash() -> None:
    cfg = make_valid_config()
    validate_config(cfg)
    assert cfg.verification_api_model == "inclusionai/ling-2.6-flash"


def test_synth_model_default_is_ling_flash() -> None:
    cfg = make_valid_config()
    validate_config(cfg)
    assert cfg.synth.model == "inclusionai/ling-2.6-flash"


def test_verification_api_model_null_preserves_valid_local_mode() -> None:
    cfg = make_valid_config(verification_api_model=None)
    validate_config(cfg)
    assert cfg.verification_api_model is None


def test_verification_api_model_accepts_openrouter_model() -> None:
    cfg = make_valid_config(
        verification_api_model="openai/gpt-5-nano",
        synth=SynthConfig(
            provider="openrouter",
            requests_per_minute=0,
            max_concurrent=8,
        ),
    )
    validate_config(cfg)


@pytest.mark.parametrize(
    "bad_model",
    ["", "   ", " openai/gpt-5-nano", "openai/gpt-5-nano ", 123, True],
)
def test_verification_api_model_rejects_invalid_model_id(bad_model: object) -> None:
    if isinstance(bad_model, str):
        cfg = make_valid_config(verification_api_model=bad_model)
        with pytest.raises(AxBenchBaseError, match="verification_api_model"):
            validate_config(cfg)
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation, match="verification_api_model"
        ):
            make_valid_config(verification_api_model=bad_model)


@pytest.mark.parametrize(
    ("synth", "match"),
    [
        (SynthConfig(provider="other"), "synth.provider='openrouter'"),
        (SynthConfig(max_concurrent=0), "synth.max_concurrent >= 1"),
        (SynthConfig(requests_per_minute=-1), "synth.requests_per_minute >= 0"),
    ],
)
def test_verification_api_model_rejects_invalid_synth_config(
    synth: SynthConfig,
    match: str,
) -> None:
    cfg = make_valid_config(
        verification_api_model="openai/gpt-5-nano",
        synth=synth,
    )
    with pytest.raises(AxBenchBaseError, match=match):
        validate_config(cfg)


def test_verification_api_model_requires_activation_evidence() -> None:
    cfg = make_valid_config(
        verification_api_model="openai/gpt-5-nano",
        n_activation_examples=0,
    )
    with pytest.raises(AxBenchBaseError, match="n_activation_examples >= 1"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [1, 20, 100, 1000])
def test_verification_api_prefix_wave_size_valid_values(value: int) -> None:
    validate_config(make_valid_config(verification_api_prefix_wave_size=value))


@pytest.mark.parametrize("value", [0, -1, -100])
def test_verification_api_prefix_wave_size_rejects_non_positive_active_values(
    value: int,
) -> None:
    cfg = make_valid_config(verification_api_prefix_wave_size=value)
    with pytest.raises(
        AxBenchBaseError,
        match=r"verification_api_prefix_wave_size must be >= 1",
    ):
        validate_config(cfg)


def test_verification_api_prefix_wave_size_is_dormant_in_local_mode() -> None:
    cfg = make_valid_config(
        verification_api_model=None,
        verification_api_prefix_wave_size=0,
    )
    validate_config(cfg)


@pytest.mark.parametrize("value", ["shared", "repeated"])
def test_verification_prefill_mode_valid_values(value):
    validate_config(make_valid_config(verification_prefill_mode=value))


@pytest.mark.parametrize(
    "value",
    ["share", "Shared", "SHARED", "native", "compat", "", "auto"],
)
def test_verification_prefill_mode_invalid_values(value):
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="verification_prefill_mode",
    ):
        make_valid_config(verification_prefill_mode=value)


@pytest.mark.parametrize("value", [1, 4, 16, 64, 1024])
def test_stage0c_multi_gpu_block_size_valid_values(value):
    validate_config(make_valid_config(stage0c_multi_gpu_block_size=value))


@pytest.mark.parametrize("value", [1, 2, 4, 8, 64])
def test_verification_cross_prefix_batch_size_valid_values(value):
    validate_config(make_valid_config(verification_cross_prefix_batch_size=value))


@pytest.mark.parametrize("value", [0, -1, -100])
def test_stage0c_multi_gpu_block_size_rejects_zero_and_negative(value):
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="stage0c_multi_gpu_block_size",
    ):
        make_valid_config(stage0c_multi_gpu_block_size=value)


@pytest.mark.parametrize("value", [0, -1, -100])
def test_verification_cross_prefix_batch_size_rejects_zero_and_negative(value):
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="verification_cross_prefix_batch_size",
    ):
        make_valid_config(verification_cross_prefix_batch_size=value)


def test_validate_config_rejects_cuda_with_index_under_multi_gpu(monkeypatch):
    """Bug I-3: ``cfg.device='cuda:0'`` under torchrun (WORLD_SIZE>1)
    silently breaks multi-GPU because every rank loads onto GPU 0.

    The validator must reject the explicit GPU index and tell the
    operator to use ``cfg.device='cuda'`` (no index) so that
    ``compute_init``'s ``torch.cuda.set_device`` can assign each rank
    its own GPU.
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    cfg = make_valid_config(device="cuda:0")
    with pytest.raises(
        AxBenchBaseError,
        match="incompatible with multi-GPU launch",
    ):
        validate_config(cfg)


def test_validate_config_accepts_bare_cuda_under_multi_gpu(monkeypatch):
    """Counterpart: ``cfg.device='cuda'`` (no index) is the correct
    multi-GPU value and must pass validation."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = make_valid_config(device="cuda")
    validate_config(cfg)  # no raise


def test_validate_config_accepts_cuda_index_when_world_size_is_one(monkeypatch):
    """Single-GPU runs (``WORLD_SIZE=1`` or unset) must still accept
    ``cfg.device='cuda:0'`` for backwards compatibility — historically
    the dev workflow used ``cuda:0`` to pin a specific GPU."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = make_valid_config(device="cuda:0")
    validate_config(cfg)
    # Also valid with WORLD_SIZE explicitly set to 1.
    monkeypatch.setenv("WORLD_SIZE", "1")
    validate_config(cfg)


def test_validate_config_rejects_cuda_with_higher_index_under_multi_gpu(monkeypatch):
    """``cfg.device='cuda:7'`` should also be rejected — the rule is
    'no explicit index' regardless of which index."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    cfg = make_valid_config(device="cuda:7")
    with pytest.raises(
        AxBenchBaseError,
        match="incompatible with multi-GPU launch",
    ):
        validate_config(cfg)


def test_seed_prefix_max_below_continuation_rejected():
    """validate_config enforces seed_prefix_max_nanochat_tokens >=
    seed_continuation_nanochat_tokens, so every SeedRow has at least
    as many prefix tokens as continuation tokens (review-fix E).
    """
    cfg = make_valid_config(
        seed_prefix_max_nanochat_tokens=100,
        seed_continuation_nanochat_tokens=200,
    )
    with pytest.raises(AxBenchBaseError, match="seed_prefix_max_nanochat_tokens"):
        validate_config(cfg)


def test_seed_prefix_max_equal_continuation_accepted():
    """Equal values satisfy the `>=` check — prefix always at least as
    long as continuation, never shorter."""
    cfg = make_valid_config(
        seed_prefix_max_nanochat_tokens=64,
        seed_continuation_nanochat_tokens=64,
    )
    validate_config(cfg)  # no raise


def test_judge_config_inherits_llm_client_config():
    assert issubclass(JudgeConfig, LLMClientConfig)
    cfg = JudgeConfig()
    # Inherited fields present with defaults
    assert cfg.provider == "openrouter"
    assert cfg.model == "openai/gpt-5-nano"
    assert cfg.requests_per_minute == 60
    assert cfg.max_concurrent == 10
    # Own fields present with defaults
    assert cfg.parse_max_retries == 3
    assert cfg.invalid_warning_threshold == 0.01
    assert cfg.invalid_drop_threshold == 0.05


def test_synth_config_inherits_llm_client_config():
    assert issubclass(SynthConfig, LLMClientConfig)
    cfg = SynthConfig()
    assert cfg.provider == "openrouter"
    assert cfg.model == "inclusionai/ling-2.6-flash"
    assert cfg.requests_per_minute == 6_000
    assert cfg.max_concurrent == 100
    assert cfg.transport_max_retries == 3


def test_llm_client_config_is_frozen():
    cfg = LLMClientConfig()
    # Slice 16 audit (X2 §4.3): narrow from broad-Exception. Frozen
    # dataclass mutation raises ``FrozenInstanceError``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        set_runtime_attribute(cfg, "provider", "changed")


def test_judge_config_override_inherited_field():
    cfg = JudgeConfig(model="openai/gpt-4o", requests_per_minute=120)
    assert cfg.model == "openai/gpt-4o"
    assert cfg.requests_per_minute == 120
    assert cfg.parse_max_retries == 3  # default still present


def test_validate_config_rejects_max_concepts_below_one():
    """D2: max_concepts <= 0 must be rejected. It was previously
    unbounded; combined with adaptive _block_size it is a footgun and
    is wrong for the pipeline (Stage 0c would accept 0 concepts)."""
    base = make_valid_config()  # use this file's existing valid-config factory
    for bad in (0, -1):
        with pytest.raises(BeartypeCallHintParamViolation, match="max_concepts"):
            base.replace(max_concepts=bad)
    validate_config(base.replace(max_concepts=1))


def test_axbench_base_config_has_sensible_defaults():
    """Construct AxBenchBaseConfig with only mandatory fields and verify
    that sensible defaults kick in for optional ones.
    """
    cfg = AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out/eval/axbench_base",
    )
    # Locked-in decisions
    assert cfg.concept_feature_mode == "canonical"
    assert cfg.accept_threshold == 0.3
    assert cfg.positive_prefix_headroom_ceiling == 0.9
    assert cfg.positive_max_attempts_per_seed == 8
    assert cfg.positive_zero_hit_bail_after == 2000
    assert cfg.output_length == 64
    assert cfg.max_retries == 100
    assert cfg.synth.transport_max_retries == 3
    assert cfg.qc_fallback_on_exhaust == "error"
    assert cfg.validation_shards == [369]
    # Spec defaults
    assert cfg.n_rank_store == 256
    assert cfg.max_concepts == 500
    assert cfg.seed_pool_train_ratio == 0.75
    # Reviewer override: 64 -> 256 -> 512 for richer natural-continuation
    # context. Review-fix E renamed the field to
    # seed_prefix_max_nanochat_tokens (variable-length prefix, not a fixed
    # length).
    assert cfg.seed_prefix_max_nanochat_tokens == 512
    # Review-fix 4: the seed pool also stores up to this many nanochat
    # tokens from the document tail, for Stage 1a's document-continuation
    # negatives to use in place of live base-model generation.
    assert cfg.seed_continuation_nanochat_tokens == 256
    assert cfg.run_negative_mode is True
    assert cfg.extraction_tokenizer_id == "google/gemma-2-2b"
    # Config-only target-model context bound (Stage 2 / Stage 4 extraction).
    assert cfg.extraction_max_tokens == 1024
    # Auto-generated run_tag
    assert cfg.run_tag is None
    # Nested configs still have their defaults
    assert isinstance(cfg.synth, SynthConfig)
    assert isinstance(cfg.judge, JudgeConfig)
    # Review-fix G: concept examples QC on by default (was off under F)
    assert cfg.enable_concept_examples_qc is True
    assert cfg.n_activation_examples == 5
    assert cfg.activation_example_judge_min_score == 1
    assert isinstance(cfg.concept_examples_qc_judge, LLMClientConfig)
    assert cfg.concept_examples_qc_target_surviving is None
    assert cfg.concept_description_fetch_limit is None
    assert cfg.verification_candidate_scan_limit is None
    assert cfg.verification_candidate_attempt_limit is None
    assert cfg.verification_zero_hit_bail_after == 2000


def test_axbench_base_config_extraction_tokenizer_id_default():
    """The extraction_tokenizer_id default matches the default SAE vocab
    (gemmascope-res-16k was trained against google/gemma-2-2b)."""
    cfg = make_valid_config()
    assert cfg.extraction_tokenizer_id == "google/gemma-2-2b"


def test_axbench_base_config_extraction_tokenizer_id_override():
    """The extraction_tokenizer_id can be overridden for non-default SAEs."""
    cfg = make_valid_config(extraction_tokenizer_id="meta-llama/Llama-3.1-8B")
    assert cfg.extraction_tokenizer_id == "meta-llama/Llama-3.1-8B"


def test_axbench_base_config_mandatory_fields_are_required():
    """Omitting any of the 6 mandatory fields must raise TypeError at construction."""
    # ty narrows a simple ``ctor: Any = AxBenchBaseConfig`` assignment back
    # to the class type, so we hide the constructor inside an untyped
    # container to keep the runtime assertion while staying ty-clean.
    ctors: list[Any] = [AxBenchBaseConfig]
    with pytest.raises(TypeError, match="missing.*argument"):
        ctors[0]()


def test_neuronpedia_config_default_attached():
    """:attr:`AxBenchBaseConfig.neuronpedia` exposes a default
    :class:`NeuronpediaConfig` so callers don't have to construct one."""
    from steering.eval.neuronpedia_config import NeuronpediaConfig

    cfg = make_valid_config()
    assert isinstance(cfg.neuronpedia, NeuronpediaConfig)
    assert cfg.neuronpedia.max_concurrent == 10


def test_neuronpedia_config_overridable():
    """Operators can tune the sub-config via ``cfg.replace(...)``."""
    from steering.eval.neuronpedia_config import NeuronpediaConfig

    cfg = make_valid_config().replace(
        neuronpedia=NeuronpediaConfig(max_concurrent=4, max_retries=5),
    )
    assert cfg.neuronpedia.max_concurrent == 4
    assert cfg.neuronpedia.max_retries == 5


def test_concept_feature_whitelist_config_is_file_only():
    field_names = {field.name for field in dataclasses.fields(AxBenchBaseConfig)}
    assert "concept_feature_whitelist" not in field_names
    assert "concept_feature_whitelist_path" in field_names

    validate_config(make_valid_config(concept_feature_whitelist_path="x.txt"))
    validate_config(make_valid_config())

    repo_root = Path(__file__).resolve().parents[3]
    root_yaml = OmegaConf.load(
        repo_root / "configs/eval/axbench_base/axbench_base.yaml"
    )
    assert "concept_feature_whitelist" not in root_yaml
    assert root_yaml.concept_feature_whitelist_path is None


@pytest.mark.parametrize("bad_path", [123, True, [1, 2], Path("ids.txt")])
def test_concept_feature_whitelist_path_requires_string_or_null(bad_path):
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="concept_feature_whitelist_path",
    ):
        make_valid_config(concept_feature_whitelist_path=bad_path)


# ============================================================================
# Clustered conditioning gate (PR #739 Track C2, config.py:validate_config's
# new block). The oracle has ZERO tests for this block -- authored from
# scratch, flat test_validate_config_{rejects,accepts}_* naming per the plan.
# ============================================================================

_CLUSTERED_CONDITIONING = {
    "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
    "feature_clusters": {"path": "/clusters.yaml"},
}


@pytest.mark.parametrize("bad_conditioning", [123, "a string", [1, 2], True])
def test_validate_config_rejects_non_dict_conditioning(bad_conditioning):
    """None of these 4 values are ``Mapping`` instances (post review
    finding 4 fix, the gate checks ``isinstance(Mapping)``, not
    ``isinstance(dict)`` -- see test_validate_config_accepts_non_dict_
    mapping_conditioning_without_feature_clusters below for the case this
    test does NOT cover: a genuine non-dict Mapping)."""
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="conditioning",
    ):
        make_valid_config(conditioning=bad_conditioning)


def test_validate_config_accepts_null_conditioning():
    """conditioning=None is the pre-C2 default path (and the
    ``AxBenchBaseConfig.conditioning`` field default itself): the new
    clustered gate must be a complete no-op -- byte-identical
    validate_config behavior to every pre-C2 config -- whenever
    conditioning is null."""
    cfg = make_valid_config(conditioning=None)
    validate_config(cfg)


def test_validate_config_rejects_forced_revalidation_without_clusters():
    cfg = make_valid_config(
        conditioning=None,
        force_cluster_shard_revalidation=True,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="requires clustered conditioning",
    ):
        validate_config(cfg)


def test_validate_config_rejects_forced_revalidation_bool_lookalike():
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="force_cluster_shard_revalidation",
    ):
        make_valid_config(force_cluster_shard_revalidation=1)


def test_validate_config_accepts_non_dict_mapping_conditioning_without_feature_clusters():
    """U-invariant fix (review finding 4, MEDIUM): a non-dict ``Mapping``
    conditioning (here ``types.MappingProxyType``, a real stdlib non-dict
    Mapping) with NO ``feature_clusters`` must pass ``validate_config``
    exactly as it did pre-C2, which never checked ``conditioning``'s type
    at all -- this is the UNCLUSTERED path, and the new clustered gate
    must not change its behavior. Matches ``clustered_source_enabled``'s
    own acceptance criterion (cluster_artifacts.py:226:
    ``isinstance(conditioning, Mapping)``), not the block's previous,
    stricter ``isinstance(dict)`` (a named BUG-FIX deviation from the
    oracle, whose own block is ``isinstance(dict)`` too)."""
    cfg = make_valid_config(
        conditioning=MappingProxyType(
            {"context_cache_dir": None, "n_sae_features": 16384, "feature_mask": None}
        ),
    )
    validate_config(cfg)  # must not raise


def test_validate_config_rejects_malformed_clustered_conditioning_even_as_non_dict_mapping():
    """The clustered gate must still reject a genuinely-malformed
    clustered conditioning even when delivered as a non-dict Mapping --
    widening the top-level type check from ``dict`` to ``Mapping`` must
    not weaken the CLUSTERED-path validation itself. Mirrors
    test_validate_config_rejects_feature_clusters_without_feature_mask
    above, but through a non-dict Mapping this time."""
    cfg = make_valid_config(
        conditioning=MappingProxyType(
            {"feature_clusters": {"path": "/clusters.yaml"}},
        ),
    )
    with pytest.raises(AxBenchBaseError, match="feature_mask is null"):
        validate_config(cfg)


def test_validate_config_rejects_feature_clusters_without_feature_mask():
    cfg = make_valid_config(
        conditioning={"feature_clusters": {"path": "/clusters.yaml"}},
    )
    with pytest.raises(AxBenchBaseError, match="feature_mask is null"):
        validate_config(cfg)


def test_validate_config_ignores_feature_mask_without_feature_clusters():
    """NOT symmetric at this layer -- verified finding, not an assumption:
    ``validate_config``'s own block gates entry on ``feature_clusters``
    alone (``if feature_clusters is not None:``), never on
    ``feature_mask``. ``feature_mask`` is the pre-existing #714
    conditioning field (predates clustering; still live with
    ``feature_clusters`` absent in configs/eval/axbench_base/axbench_
    base.yaml:106-115) -- a mask-only conditioning dict is simply "not
    clustered" from this block's point of view and falls straight
    through untouched. ``parse_cluster_eval_paths``'s own mask-without-
    clusters symmetric rejection IS real (see the paired
    feature_clusters-without-feature_mask test above, and
    tests/steering/test_cluster_eval_config.py for the direct unit
    coverage of that function), but it is only reached from THIS call
    site when ``feature_clusters`` is already non-null; the reverse
    direction is enforced by callers that always pass both keys from a
    single source of truth (e.g. ``resolve_cluster_feature_contract``'s
    own ``ClusterFeatureContractRequest``, out of C2's scope). Ported
    code confirmed byte-identical to the oracle (see the file-level
    diff); this asymmetry is inherited, not introduced by this port."""
    cfg = make_valid_config(
        conditioning={"feature_mask": {"mode": "whitelist", "path": "/mask.txt"}},
    )
    validate_config(cfg)


def test_validate_config_rejects_non_canonical_concept_feature_mode_when_clustered():
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        concept_feature_mode="delta_ranked",
        enabled_methods=["diffmean_decoder_only"],
        stage="0a",
    )
    with pytest.raises(
        AxBenchBaseError,
        match="clustered AxBench-Base requires concept_feature_mode='canonical'",
    ):
        validate_config(cfg)


def test_validate_config_rejects_unsupported_stage_when_clustered():
    """``3c`` (TRAIN_PROMPTING) is a real ``Stage`` member but explicitly
    outside the clustered-supported set -- the block's own error message
    names exactly this stage class ("Prompting/LoRA training and later
    judge/aggregation stages remain unsupported")."""
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        enabled_methods=["diffmean_decoder_only"],
        stage="3c",
    )
    with pytest.raises(
        AxBenchBaseError,
        match="clustered AxBench-Base supports source construction",
    ):
        validate_config(cfg)


def test_validate_config_rejects_null_enabled_methods_when_clustered():
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        enabled_methods=None,
    )
    with pytest.raises(
        AxBenchBaseError,
        match="axbench_base requires an explicit method allowlist for clustered runs",
    ):
        validate_config(cfg)


def test_validate_config_rejects_stage4_only_method_at_source_stage():
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        enabled_methods=[
            "control_encdec"
        ],  # Stage-4-only, not one of the 6 source methods
        stage="0a",
    )
    with pytest.raises(
        AxBenchBaseError,
        match=r"clustered AxBench-Base stage 0a does not support",
    ):
        validate_config(cfg)


def test_validate_config_rejects_stage4_only_method_at_a_different_source_stage():
    """Reverse of the ``stage=0a`` case above: a DIFFERENT non-{3e,4}
    clustered-supported stage must be equally rejected for a Stage-4-only
    method -- pins that the stage-conditional allowlist branches on
    membership in ``{"3e", "4"}``, not a hardcoded ``"0a"`` special case."""
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        enabled_methods=["neg_control_encdec"],
        stage="1c",
    )
    with pytest.raises(
        AxBenchBaseError,
        match=r"clustered AxBench-Base stage 1c does not support",
    ):
        validate_config(cfg)


def test_validate_config_accepts_clustered_source_stage_with_source_method_subset():
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        enabled_methods=list(CLUSTERED_AXBENCH_BASE_SOURCE_METHODS),
        stage="0a",
    )
    validate_config(cfg)


def test_validate_config_accepts_clustered_stage4_with_full_method_set():
    """Reverse/positive mirror of
    test_validate_config_rejects_stage4_only_method_at_source_stage: the
    same encdec-only methods rejected at stage=0a are accepted once
    ``stage`` is in the Stage-4 group."""
    cfg = make_valid_config(
        conditioning=_CLUSTERED_CONDITIONING,
        enabled_methods=list(CLUSTERED_AXBENCH_BASE_STAGE4_METHODS),
        stage="4",
    )
    validate_config(cfg)


def test_validate_config_accepts_real_production_conditioning_shape_unchanged():
    """Regression pin vs the REAL production shape (configs/eval/axbench_
    base/axbench_base.yaml:106-115): context_cache_dir/identity/
    n_sae_features/num_buckets/feature_mask, with NO feature_clusters key
    at all. The new clustered gate must be a complete no-op on this exact
    shape -- mirrors test_concept_feature_whitelist_config_is_file_only's
    own root-yaml cross-check above."""
    repo_root = Path(__file__).resolve().parents[3]
    root_yaml = OmegaConf.load(
        repo_root / "configs/eval/axbench_base/axbench_base.yaml"
    )
    conditioning = OmegaConf.to_container(root_yaml.conditioning, resolve=True)
    assert isinstance(conditioning, dict)
    assert "feature_clusters" not in conditioning
    cfg = make_valid_config(conditioning=conditioning)
    validate_config(cfg)
