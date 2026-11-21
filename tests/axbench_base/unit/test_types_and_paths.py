import dataclasses
from pathlib import Path

import pytest

from steering.eval.axbench_base import trainable_concepts
from steering.eval.axbench_base.baselines import _capture
from steering.eval.axbench_base.paths import PROMPT_DIR, AxBenchBasePaths
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
    SeedRow,
    SteeringMode,
    TargetModelMetadata,
)
from tests.runtime_attribute_helpers import set_runtime_attribute


def test_error_is_exception_subclass():
    assert issubclass(AxBenchBaseError, Exception)
    with pytest.raises(AxBenchBaseError, match="boom"):
        raise AxBenchBaseError("boom")


def test_seed_row_is_frozen_dataclass():
    r = SeedRow(
        shard_idx=0,
        doc_idx=1,
        source_nanochat_token_count=4,
        prefix_source_token_start=0,
        prefix_source_token_end=2,
        prefix_text="hi",
        prefix_content_nanochat_tokens=[1, 2],
        sae_feature_ids=[3, 4],
        sae_feature_mags=[0.5, 0.8],
        continuation_text="tail",
        continuation_content_nanochat_tokens=[5, 6],
    )
    assert r.prefix_text == "hi"
    assert r.continuation_text == "tail"
    assert r.continuation_content_nanochat_tokens == [5, 6]
    # Slice 16 audit (X2 §4.3): narrow from broad-Exception. Frozen
    # dataclass mutation raises ``FrozenInstanceError``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        set_runtime_attribute(r, "prefix_text", "changed")


def test_seed_row_rejects_empty_continuation_tokens():
    """``SeedRow.__post_init__`` must reject empty
    ``continuation_content_nanochat_tokens`` as a defensive guard — the pipeline
    (review-fix E) guarantees every row has exactly M continuation tokens.
    """
    with pytest.raises(AxBenchBaseError, match="empty continuation"):
        SeedRow(
            shard_idx=0,
            doc_idx=1,
            source_nanochat_token_count=3,
            prefix_source_token_start=0,
            prefix_source_token_end=3,
            prefix_text="hi",
            prefix_content_nanochat_tokens=[1, 2, 3],
            sae_feature_ids=[],
            sae_feature_mags=[],
            continuation_text="",
            continuation_content_nanochat_tokens=[],  # empty -> must raise
        )


def test_target_model_metadata_is_frozen_dataclass():
    m = TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cuda",
        sae_layer=8,
        segment_size=128,
    )
    assert m.num_buckets == 8
    assert m.n_sae_features == 16384
    assert m.device == "cuda"
    assert m.sae_layer == 8
    assert m.segment_size == 128
    # Slice 16 audit (X2 §4.3): narrow from broad-Exception. Frozen
    # dataclass mutation raises ``FrozenInstanceError``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        set_runtime_attribute(m, "num_buckets", 10)


def test_run_output_root_sanitizes_run_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root_good = AxBenchBasePaths(
        tmp_path / "eval" / "axbench_base", "safe_tag_42"
    ).run_output_root()
    assert root_good.name == "safe_tag_42"
    # Path traversal attempt
    root_bad = AxBenchBasePaths(
        tmp_path / "eval" / "axbench_base", "bad/../../etc/passwd"
    ).run_output_root()
    assert ".." not in root_bad.parts
    assert "passwd" not in root_bad.name


def test_run_output_root_strips_unsafe_chars(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root = AxBenchBasePaths(
        tmp_path / "eval" / "axbench_base", "tag with spaces!"
    ).run_output_root()
    assert " " not in root.name
    assert "!" not in root.name


def test_run_output_root_empty_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root = AxBenchBasePaths(tmp_path / "eval" / "axbench_base", "").run_output_root()
    # Must produce a non-empty safe name — fallback to something like underscore
    assert root.name != ""


def test_run_output_root_dot_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root = AxBenchBasePaths(tmp_path / "eval" / "axbench_base", ".").run_output_root()
    assert root.name != "."
    assert root != tmp_path / "eval" / "axbench_base"
    assert root.parent == tmp_path / "eval" / "axbench_base"


def test_all_path_resolvers_under_run_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tag = "test-run-1"
    output_paths = AxBenchBasePaths(tmp_path / "eval" / "axbench_base", tag)
    root = output_paths.run_output_root()
    assert root == tmp_path / "eval" / "axbench_base" / tag
    for resolver in (
        output_paths.candidate_concepts_json,
        output_paths.concepts_json,
        output_paths.rejected_concepts_json,
        output_paths.concept_examples_qc_report,
        output_paths.seeds_train,
        output_paths.seeds_eval,
        output_paths.negative_mode_prefixes,
        output_paths.synth_training_positive,
        output_paths.synth_training_negative,
        output_paths.concept_features,
        output_paths.concept_features_neg,
        output_paths.generations_dir,
        output_paths.judge_cache,
        output_paths.judge_scores_dir,
        output_paths.report_yaml,
        output_paths.leaderboard_csv,
        output_paths.aggregation_report,
        output_paths.synth_response_cache,
    ):
        p = resolver()
        assert str(p).startswith(str(root)), f"{resolver.__name__} → {p} not under {root}"


def test_baseline_path_resolvers(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    output_paths = AxBenchBasePaths(tmp_path / "eval" / "axbench_base", "tag")
    p = output_paths.diffmean_direction(
        target_model="decoder_only",
        mode=SteeringMode.POSITIVE,
        concept_id=42,
    )
    assert "diffmean" in str(p)
    assert "decoder_only" in str(p)
    assert "positive" in str(p)
    assert "concept_0042" in str(p)

    p = output_paths.linear_probe_direction(
        target_model="encdec",
        mode=SteeringMode.NEGATIVE,
        concept_id=7,
    )
    assert "linear_probe" in str(p)
    assert "encdec" in str(p)
    assert "negative" in str(p)
    assert "concept_0007" in str(p)

    p = output_paths.lora_adapter(mode=SteeringMode.POSITIVE, concept_id=3)
    assert "lora" in str(p)
    assert str(p).endswith("concept_0003.pt")

    p = output_paths.prompting_descriptions(mode=SteeringMode.POSITIVE)
    assert "prompting" in str(p)
    assert "positive_descriptions.json" in str(p)


def test_prompt_dir_points_inside_package():
    assert PROMPT_DIR.name == "prompts"
    assert PROMPT_DIR.parent.name == "axbench_base"


def test_content_class_label_values():
    assert CONCEPT_RICH_LABEL == "concept_rich"
    assert CONCEPT_AVOIDING_LABEL == "concept_avoiding"


def test_label_constants_single_source():
    """trainable_concepts + _capture must reference the SAME canonical objects."""
    assert trainable_concepts.CONCEPT_RICH_LABEL is CONCEPT_RICH_LABEL
    assert trainable_concepts.CONCEPT_AVOIDING_LABEL is CONCEPT_AVOIDING_LABEL
    assert _capture._POS_LABEL is CONCEPT_RICH_LABEL
    assert _capture._NEG_LABEL is CONCEPT_AVOIDING_LABEL


def test_synth_training_dispatch_positive():
    resolved = AxBenchBasePaths(output_root=Path("/tmp"), run_tag="rt")
    assert resolved.synth_training(SteeringMode.POSITIVE) == resolved.synth_training_positive()


def test_synth_training_dispatch_negative():
    resolved = AxBenchBasePaths(output_root=Path("/tmp"), run_tag="rt")
    assert resolved.synth_training(SteeringMode.NEGATIVE) == resolved.synth_training_negative()
