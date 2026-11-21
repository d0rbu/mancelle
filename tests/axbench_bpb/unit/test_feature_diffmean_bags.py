from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.config import FeatureDiffMeanConfig
from steering.eval.axbench_bpb.feature_diffmean_bags import (
    build_feature_diffmean_bags_plain,
    validate_feature_diffmean_concept_modes,
)
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, SteeringMode
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag


def _row(
    *,
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "content_class": content_class,
        "prefix_text": prefix_text,
        "output_text": output_text,
        "qc_accepted": True,
    }


def _write_positive_rows(path: Any, rows: list[dict[str, Any]]) -> None:
    base_schema = pa.Table.from_pylist(rows).schema
    schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            base_schema,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_plain_feature_diffmean_derives_and_writes_bag(tmp_path, monkeypatch) -> None:
    import steering.eval.axbench_bpb.feature_diffmean_bags as module

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_positive_rows(
        source_dir / "synth_training_positive.parquet",
        [
            _row(
                concept_id=7,
                content_class="concept_rich",
                prefix_text="rich prefix",
                output_text="rich completion",
            ),
            _row(
                concept_id=7,
                content_class="concept_avoiding",
                prefix_text="avoid prefix",
                output_text="avoid completion",
            ),
            _row(
                concept_id=8,
                content_class="concept_rich",
                prefix_text="filtered prefix",
                output_text="filtered completion",
            ),
        ],
    )

    calls: list[dict[str, Any]] = []

    def _fake_feature_diffmean_bag(rich_items, avoid_items, **kwargs):
        calls.append(
            {
                "rich": rich_items,
                "avoid": avoid_items,
                "mode": kwargs["mode"],
                "k_pos": kwargs["k_pos"],
                "k_neg": kwargs["k_neg"],
            }
        )
        return FeatureConditioning(((11, 0), (22, 1)))

    monkeypatch.setattr(module, "assert_sae_matches_encdec", lambda cfg, model: None)
    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda model, cfg: (1, 1))
    monkeypatch.setattr(
        module,
        "parse_neuronpedia_source",
        lambda source: (None, None, 8, None, None),
    )
    monkeypatch.setattr(
        module,
        "build_percentile_table_from_stats",
        lambda path, n_sae: t.zeros((1, 1)),
    )
    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_feature_diffmean_bag)

    cfg = SimpleNamespace(
        feature_statistics_path="/unused/stats.safetensors",
        filter_unaccepted_samples=True,
        feature_diffmean=FeatureDiffMeanConfig(top_k_pos=1, top_k_neg=1),
        neuronpedia_source="8-gemmascope-res-16k",
    )
    output_paths = OutputPaths(output_root=tmp_path / "out", run_tag="run")
    model = SimpleNamespace(
        encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=1)),
        feature_vocab=None,
    )

    build_feature_diffmean_bags_plain(
        cfg=cfg,
        output_paths=output_paths,
        source_paths=SourcePaths(source_dir),
        methods=[SimpleNamespace(name="feature_diffmean_encdec")],
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        device="cpu",
        concept_ids={7},
    )

    assert len(calls) == 1
    assert calls[0]["mode"] is SteeringMode.POSITIVE
    assert calls[0]["k_pos"] == 1
    assert calls[0]["k_neg"] == 1
    assert calls[0]["rich"][0].prefix_text == "rich prefix"
    assert calls[0]["rich"][0].completion_text == "rich completion"
    assert calls[0]["avoid"][0].prefix_text == "avoid prefix"
    assert calls[0]["avoid"][0].completion_text == "avoid completion"

    bag, metadata = read_ours_gcg_bag(
        output_paths.ours_gcg_bag(
            7,
            SteeringMode.POSITIVE,
            0.0,
            "feature_diffmean",
        )
    )
    assert bag == {11: 0, 22: 1}
    assert metadata["mode_tag"] == "feature_diffmean"
    assert metadata["coefficient_str"] == "None"
    assert not output_paths.ours_gcg_bag(
        8,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean",
    ).exists()


def test_plain_feature_diffmean_raises_on_one_sided_concepts(tmp_path, monkeypatch) -> None:
    import steering.eval.axbench_bpb.feature_diffmean_bags as module

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = [
        _row(
            concept_id=7,
            content_class="concept_rich",
            prefix_text="rich prefix",
            output_text="rich completion",
        ),
        _row(
            concept_id=7,
            content_class="concept_avoiding",
            prefix_text="avoid prefix",
            output_text="avoid completion",
        ),
        _row(
            concept_id=8,
            content_class="concept_rich",
            prefix_text="one sided prefix",
            output_text="one sided completion",
        ),
    ]
    _write_positive_rows(
        source_dir / "synth_training_positive.parquet",
        rows,
    )

    def _fake_feature_diffmean_bag(rich_items, avoid_items, **_kwargs):
        raise AssertionError("one-sided validation should run before bag building")
        return FeatureConditioning(((11, 0),))

    monkeypatch.setattr(module, "assert_sae_matches_encdec", lambda cfg, model: None)
    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda model, cfg: (1, 1))
    monkeypatch.setattr(
        module,
        "parse_neuronpedia_source",
        lambda source: (None, None, 8, None, None),
    )
    monkeypatch.setattr(
        module,
        "build_percentile_table_from_stats",
        lambda path, n_sae: t.zeros((1, 1)),
    )
    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_feature_diffmean_bag)

    cfg = SimpleNamespace(
        feature_statistics_path="/unused/stats.safetensors",
        filter_unaccepted_samples=True,
        feature_diffmean=FeatureDiffMeanConfig(top_k_pos=1, top_k_neg=1),
        neuronpedia_source="8-gemmascope-res-16k",
    )
    source_paths = SourcePaths(source_dir)
    output_paths = OutputPaths(output_root=tmp_path / "out", run_tag="run")
    methods = [SimpleNamespace(name="feature_diffmean_encdec")]
    model = SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=1)))

    with pytest.raises(
        AxBenchBpbError,
        match="requires both content classes.*concept_id=8",
    ):
        validate_feature_diffmean_concept_modes(
            cfg=cfg,
            source_paths=source_paths,
            methods=methods,
        )

    with pytest.raises(
        AxBenchBpbError,
        match="requires both content classes.*concept_id=8",
    ):
        build_feature_diffmean_bags_plain(
            cfg=cfg,
            output_paths=output_paths,
            source_paths=source_paths,
            methods=methods,
            model=model,
            base_model=SimpleNamespace(),
            sae=SimpleNamespace(),
            device="cpu",
        )


def test_plain_feature_diffmean_no_complete_groups_raises_before_heavy_setup(
    tmp_path, monkeypatch
) -> None:
    import steering.eval.axbench_bpb.feature_diffmean_bags as module

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = [
        _row(
            concept_id=8,
            content_class="concept_rich",
            prefix_text="one sided prefix",
            output_text="one sided completion",
        )
    ]
    _write_positive_rows(
        source_dir / "synth_training_positive.parquet",
        rows,
    )

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("heavy feature-DiffMean setup should not run")

    monkeypatch.setattr(module, "assert_sae_matches_encdec", _should_not_run)
    monkeypatch.setattr(module, "feature_diffmean_bag", _should_not_run)

    cfg = SimpleNamespace(
        feature_statistics_path="/unused/stats.safetensors",
        filter_unaccepted_samples=True,
        feature_diffmean=FeatureDiffMeanConfig(top_k_pos=1, top_k_neg=1),
        neuronpedia_source="8-gemmascope-res-16k",
    )
    source_paths = SourcePaths(source_dir)
    methods = [SimpleNamespace(name="feature_diffmean_encdec")]
    with pytest.raises(
        AxBenchBpbError,
        match="requires both content classes.*concept_id=8",
    ):
        build_feature_diffmean_bags_plain(
            cfg=cfg,
            output_paths=OutputPaths(output_root=tmp_path / "out", run_tag="run"),
            source_paths=source_paths,
            methods=methods,
            model=SimpleNamespace(),
            base_model=SimpleNamespace(),
            sae=SimpleNamespace(),
            device="cpu",
        )
