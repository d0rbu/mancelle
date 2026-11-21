"""Dataclasses in feature_filter.types: construction, frozen-ness, schema."""
from dataclasses import FrozenInstanceError

import pytest

import steering.eval.feature_filter.types as feature_filter_types
from steering.eval.feature_filter.types import (
    SCHEMA_VERSION,
    ActivatingExample,
    BinSample,
    BinSampleRecord,
    FeatureFilterReport,
    FeatureInterpScore,
)


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == 3


def test_activating_example_frozen() -> None:
    ex = ActivatingExample(tokens=["a"], text="a", max_value=1.0, max_value_token_index=0, peak_char_offset=0)
    field_name = "max_value"
    with pytest.raises(FrozenInstanceError):
        setattr(ex, field_name, 2.0)


def test_activating_example_text_windowed() -> None:
    tokens = ["hello ", "world"]
    ex = ActivatingExample(tokens=tokens, text="hello world", max_value=5.0, max_value_token_index=1, peak_char_offset=6)
    assert ex.example_text(window_chars=0) == "hello world"
    assert ex.example_text(window_chars=3) == "...lo wor..."


def test_bin_sample_is_frozen() -> None:
    ex = ActivatingExample(
        tokens=["a"], text="a", max_value=1.0,
        max_value_token_index=0, peak_char_offset=0,
    )
    s = BinSample(
        bin_index=4, band_label="band 5/5 (peak firing)",
        is_absent=False, max_value=1.0, example=ex,
    )
    field_name = "bin_index"
    with pytest.raises(FrozenInstanceError):
        setattr(s, field_name, 0)


def test_bin_sample_record_is_frozen() -> None:
    r = BinSampleRecord(
        bin_index=4, band_label="band 5/5 (peak firing)",
        is_absent=False, max_value=1.0, snippet="x",
    )
    field_name = "bin_index"
    with pytest.raises(FrozenInstanceError):
        setattr(r, field_name, 0)


def test_bin_sample_invariant_is_absent_requires_zero_max_value() -> None:
    ex = ActivatingExample(
        tokens=["a"], text="a", max_value=1.0,
        max_value_token_index=0, peak_char_offset=0,
    )
    with pytest.raises(ValueError, match="max_value"):
        BinSample(
            bin_index=0,
            band_label="band 1/5 (absent — feature did not fire)",
            is_absent=True, max_value=0.5, example=ex,
        )


def test_bin_sample_invariant_is_absent_requires_bin_zero() -> None:
    ex = ActivatingExample(
        tokens=["a"], text="a", max_value=0.0,
        max_value_token_index=0, peak_char_offset=0,
    )
    with pytest.raises(ValueError, match="bin_index"):
        BinSample(
            bin_index=4,
            band_label="band 1/5 (absent — feature did not fire)",
            is_absent=True, max_value=0.0, example=ex,
        )


def test_bin_sample_invariant_label_must_match_is_absent() -> None:
    ex = ActivatingExample(
        tokens=["a"], text="a", max_value=1.0,
        max_value_token_index=0, peak_char_offset=0,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        BinSample(
            bin_index=0,
            band_label="band 1/5 (absent — feature did not fire)",
            is_absent=False, max_value=1.0, example=ex,
        )


def test_percentile_sample_not_exported() -> None:
    assert not hasattr(feature_filter_types, "PercentileSample")


def test_percentile_sample_record_not_exported() -> None:
    assert not hasattr(feature_filter_types, "PercentileSampleRecord")


def test_feature_interp_score_allows_none_scores() -> None:
    score = FeatureInterpScore(
        feature_id=42,
        description=None,
        outcome="fetch_error",
        interpretability=None,
        consistency=None,
        samples=[],
        error_message="network",
        schema_version=SCHEMA_VERSION,
    )
    assert score.interpretability is None
    assert score.consistency is None


def test_feature_filter_report_populates() -> None:
    rep = FeatureFilterReport(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        samples_per_bin=5,
        thresholds={"interpretability": 1, "consistency": 3},
        n_total=2,
        n_accepted=1,
        n_rejected=1,
        n_fetch_errors=0,
        n_judge_errors=0,
        n_no_nonzero=0,
        n_prompt_errors=0,
        n_not_published=0,
        interpretability_histogram={3: 2},
        consistency_histogram={4: 1, 2: 1},
        per_feature_path="out/feature_scores.jsonl",
        accepted_feature_ids_path="out/accepted_feature_ids.txt",
        schema_version=SCHEMA_VERSION,
    )
    assert rep.n_accepted == 1


class TestNotPublishedSchema:
    def test_schema_version_is_3(self) -> None:
        from steering.eval.feature_filter.types import SCHEMA_VERSION

        assert SCHEMA_VERSION == 3

    def test_outcome_type_includes_not_published(self) -> None:
        import typing

        from steering.eval.feature_filter.types import OutcomeType

        assert "not_published" in typing.get_args(OutcomeType)

    def test_report_has_n_not_published_field(self) -> None:
        import dataclasses

        from steering.eval.feature_filter.types import FeatureFilterReport

        names = {f.name for f in dataclasses.fields(FeatureFilterReport)}
        assert "n_not_published" in names
