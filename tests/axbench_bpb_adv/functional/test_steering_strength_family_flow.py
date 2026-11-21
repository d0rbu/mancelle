from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.eval.axbench_bpb_adv_eval import _stage_plot
from scripts.eval.plot_steering_strength_families import main as replot_main
from steering.eval.axbench_bpb_adv.aggregation import (
    STEERING_STRENGTH_FAMILY_PLOT_NAME,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
)


def _write_contract(paths: AxBenchBpbAdvPaths, contract_hash: str) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.source_contract().write_text(json.dumps({"contract_hash": contract_hash}))


def _roundtrip_row(
    *,
    coefficient_str: str = "1.0",
    general_val_corpus: str | None = "nd=1;nb=8;h=deadbeefdeadbeef",
) -> dict[str, Any]:
    return {
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": coefficient_str,
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 3.0,
        "mean_roundtrip_pre_act": 0.5,
        "mean_roundtrip_post_act": 2.0,
        "mean_roundtrip_percentile": 90.0,
        "primary_roundtrip_y": 1.5,
        "mean_control_roundtrip_pre_act": 0.5,
        "mean_control_roundtrip_post_act": 0.5,
        "mean_control_roundtrip_percentile": 50.0,
        "control_primary_roundtrip_y": 0.0,
        "general_val_bpb": 2.4,
        "general_val_corpus": general_val_corpus,
        "n_concepts": 1,
        "n_samples": 1,
        "n_generated_tokens": 8,
        "n_control_generated_tokens": 8,
        "max_tokens_rate": 0.0,
        "control_max_tokens_rate": 0.0,
        "is_control": True,
    }


def _bpb_row(
    *,
    coefficient_str: str = "1.0",
    general_val_corpus: str | None = "nd=1;nb=8;h=deadbeefdeadbeef",
) -> dict[str, Any]:
    return {
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": coefficient_str,
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 2.0,
        "baseline_primary_margin": 2.0,
        "general_val_bpb": 2.4,
        "general_val_corpus": general_val_corpus,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": True,
    }


def _bpb_feature_diffmean_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    mode = str(first["mode"])
    method_name = "neg_feature_diffmean_encdec" if mode == "negative" else "feature_diffmean_encdec"
    row = _bpb_row(general_val_corpus=first.get("general_val_corpus"))
    row["mode"] = mode
    row["method_name"] = method_name
    row["target_model"] = "encdec"
    row["coefficient_str"] = "None"
    row["adversarial_method"] = str(first["adversarial_method"])
    row["strength_unit"] = str(first["strength_unit"])
    row["context_signal_mode"] = str(first["context_signal_mode"])
    return row


def _with_bpb_feature_diffmean_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*rows, _bpb_feature_diffmean_target(rows)]


def _roundtrip_feature_diffmean_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    mode = str(first["mode"])
    method_name = "neg_feature_diffmean_encdec" if mode == "negative" else "feature_diffmean_encdec"
    row = _roundtrip_row(general_val_corpus=first.get("general_val_corpus"))
    row["mode"] = mode
    row["method_name"] = method_name
    row["target_model"] = "encdec"
    row["coefficient_str"] = "None"
    row["adversarial_method"] = str(first["adversarial_method"])
    row["strength_unit"] = str(first["strength_unit"])
    row["context_signal_mode"] = str(first["context_signal_mode"])
    return row


def _with_roundtrip_feature_diffmean_target(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [*rows, _roundtrip_feature_diffmean_target(rows)]


def _cfg(tmp_path: Path, *, run_tag: str, plot_strength: bool) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag=run_tag,
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        target_models=["decoder_only"],
        steering_metric="roundtrip",
        plot_steering_strength_families=plot_strength,
    )


def test_stage_plot_flag_emits_strength_family_plots(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, run_tag="flag-on", plot_strength=True)
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag=cfg.run_tag or "")
    paths.roundtrip_aggregates_dir().mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            _with_roundtrip_feature_diffmean_target(
                [_roundtrip_row(coefficient_str="1.0"), _roundtrip_row(coefficient_str="2.0")]
            ),
            schema=ROUNDTRIP_MACRO_AGG_SCHEMA,
        ),
        paths.roundtrip_aggregates_dir() / "macro.parquet",
    )

    _stage_plot(cfg)

    assert (
        paths.steering_strength_plots_dir()
        / "positive"
        / STEERING_STRENGTH_FAMILY_PLOT_NAME
    ).exists()
    assert (paths.roundtrip_plots_dir() / "positive" / "primary_roundtrip_y_vs_strength.png").exists()


def test_stage_plot_flag_off_does_not_emit_strength_family_plots(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, run_tag="flag-off", plot_strength=False)
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag=cfg.run_tag or "")
    paths.roundtrip_aggregates_dir().mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            _with_roundtrip_feature_diffmean_target([_roundtrip_row()]),
            schema=ROUNDTRIP_MACRO_AGG_SCHEMA,
        ),
        paths.roundtrip_aggregates_dir() / "macro.parquet",
    )

    _stage_plot(cfg)

    assert not paths.steering_strength_plots_dir().exists()
    assert (paths.roundtrip_plots_dir() / "positive" / "primary_roundtrip_y_vs_strength.png").exists()


def test_standalone_replot_cli_routes_bpb_macro(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")
    paths.aggregates_dir().mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            _with_bpb_feature_diffmean_target(
                [_bpb_row(coefficient_str="1.0"), _bpb_row(coefficient_str="2.0")]
            ),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    _write_contract(paths, "solo-hash")

    rc = replot_main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--run-tag",
            "run",
            "--metric",
            "bpb_class_margin",
        ]
    )

    assert rc == 0
    assert (
        paths.steering_strength_plots_dir()
        / "positive"
        / STEERING_STRENGTH_FAMILY_PLOT_NAME
    ).exists()


def test_old_schema_macro_is_rejected_by_replot(tmp_path: Path) -> None:
    # A pre-635 macro.parquet (schema v1, no general_val_corpus column) must be
    # rejected by _read_macro's strict schema check before any contract-hash or
    # plotting logic runs -- no source_contract.json is written here on purpose,
    # since the schema raise inside the run-tag loop fires first.
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="old")
    paths.aggregates_dir().mkdir(parents=True)
    old_schema = pa.schema([f for f in MACRO_AGG_SCHEMA if f.name != "general_val_corpus"])
    pq.write_table(
        pa.Table.from_pylist([], schema=old_schema),
        paths.aggregates_dir() / "macro.parquet",
    )
    with pytest.raises(AxBenchBpbAdvError, match="schema mismatch"):
        replot_main(
            [
                "--output-root",
                str(tmp_path / "out"),
                "--run-tag",
                "old",
                "--metric",
                "bpb_class_margin",
            ]
        )


def test_standalone_replot_cli_requires_output_dir_for_multiple_runs(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="--plots-dir is required"):
        replot_main(
            [
                "--output-root",
                str(tmp_path / "out"),
                "--run-tag",
                "run-a",
                "--run-tag",
                "run-b",
            ]
        )


def test_combiner_requires_equal_contract_hash(tmp_path: Path) -> None:
    for tag, ch in [("run-a", "aaaa"), ("run-b", "bbbb")]:
        paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag=tag)
        paths.aggregates_dir().mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                _with_bpb_feature_diffmean_target([_bpb_row()]),
                schema=MACRO_AGG_SCHEMA,
            ),
            paths.aggregates_dir() / "macro.parquet",
        )
        _write_contract(paths, ch)
    with pytest.raises(AxBenchBpbAdvError, match="contract_hash"):
        replot_main(
            [
                "--output-root",
                str(tmp_path / "out"),
                "--run-tag",
                "run-a",
                "--run-tag",
                "run-b",
                "--metric",
                "bpb_class_margin",
                "--plots-dir",
                str(tmp_path / "combined"),
            ]
        )


def test_combiner_writes_combined_macro_and_provenance(tmp_path: Path) -> None:
    for tag in ["run-a", "run-b"]:
        paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag=tag)
        paths.aggregates_dir().mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                _with_bpb_feature_diffmean_target(
                    [_bpb_row(coefficient_str="1.0"), _bpb_row(coefficient_str="2.0")]
                ),
                schema=MACRO_AGG_SCHEMA,
            ),
            paths.aggregates_dir() / "macro.parquet",
        )
        _write_contract(paths, "same-hash")
    rc = replot_main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--run-tag",
            "run-a",
            "--run-tag",
            "run-b",
            "--metric",
            "bpb_class_margin",
            "--plots-dir",
            str(tmp_path / "combined"),
        ]
    )
    assert rc == 0
    combined = pq.read_table(tmp_path / "combined" / "combined_macro.parquet")
    assert combined.num_rows == 6
    prov = yaml.safe_load((tmp_path / "combined" / "provenance.yaml").read_text())
    assert [r["run_tag"] for r in prov["runs"]] == ["run-a", "run-b"]
    assert prov["runs"][0]["contract_hash"] == "same-hash"
    assert prov["general_val_corpus"] == _bpb_row()["general_val_corpus"]


def test_combiner_provenance_corpus_ignores_whitelisted_out_rows(tmp_path: Path) -> None:
    # Two runs sharing one contract hash but carrying DIFFERENT corpora on
    # different method families. With --plot-method selecting only run-a's
    # family, provenance must record the DRAWN figure's corpus (a_m's), never
    # the whitelisted-out b_m corpus. Pre-fix this read the full concatenated
    # macro and picked arbitrarily from a 2-element set.
    a_corpus = "nd=1;nb=8;h=deadbeefdeadbeef"
    b_corpus = "nd=9;nb=99;h=bbbbbbbbbbbbbbbb"
    for tag, method, corpus in [("run-a", "a_m", a_corpus), ("run-b", "b_m", b_corpus)]:
        paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag=tag)
        paths.aggregates_dir().mkdir(parents=True)
        pq.write_table(
            # This intentionally omits fair target rows: --plot-method whitelists
            # custom method names and this test is about provenance corpus
            # filtering, not representative-coefficient selection.
            pa.Table.from_pylist(
                [
                    {**_bpb_row(coefficient_str="1.0", general_val_corpus=corpus), "method_name": method},
                    {**_bpb_row(coefficient_str="2.0", general_val_corpus=corpus), "method_name": method},
                ],
                schema=MACRO_AGG_SCHEMA,
            ),
            paths.aggregates_dir() / "macro.parquet",
        )
        _write_contract(paths, "same-hash")
    rc = replot_main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--run-tag",
            "run-a",
            "--run-tag",
            "run-b",
            "--metric",
            "bpb_class_margin",
            "--plot-method",
            "a_m",
            "--plots-dir",
            str(tmp_path / "combined"),
        ]
    )
    assert rc == 0
    prov = yaml.safe_load((tmp_path / "combined" / "provenance.yaml").read_text())
    assert prov["general_val_corpus"] == a_corpus


def test_single_run_replot_requires_contract(tmp_path: Path) -> None:
    # Every replotted run must carry its validated source_contract.json -- a
    # single run too (issue #635: the pipeline always writes one, so a missing
    # contract is an incomplete/hand-copied artifact, rejected fail-loud).
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")
    paths.aggregates_dir().mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            _with_bpb_feature_diffmean_target(
                [_bpb_row(coefficient_str="1.0"), _bpb_row(coefficient_str="2.0")]
            ),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    # deliberately NO _write_contract
    with pytest.raises(AxBenchBpbAdvError, match="no source_contract.json"):
        replot_main(
            ["--output-root", str(tmp_path / "out"), "--run-tag", "run",
             "--metric", "bpb_class_margin"]
        )


def test_combiner_missing_contract_still_raises(tmp_path: Path) -> None:
    # Combining (>1 run) requires each run's contract (D3), same as single-run.
    for tag in ["run-a", "run-b"]:
        paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag=tag)
        paths.aggregates_dir().mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                _with_bpb_feature_diffmean_target([_bpb_row()]),
                schema=MACRO_AGG_SCHEMA,
            ),
            paths.aggregates_dir() / "macro.parquet",
        )
    _write_contract(AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run-a"), "h")
    # run-b has no contract
    with pytest.raises(AxBenchBpbAdvError, match="no source_contract.json"):
        replot_main(
            ["--output-root", str(tmp_path / "out"), "--run-tag", "run-a",
             "--run-tag", "run-b", "--metric", "bpb_class_margin",
             "--plots-dir", str(tmp_path / "combined")]
        )


def test_replot_malformed_contract_raises_named(tmp_path: Path) -> None:
    # A present-but-malformed contract (missing contract_hash key) is always a
    # loud, named AxBenchBpbAdvError -- not a bare KeyError (issue #635 review [3]).
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")
    paths.aggregates_dir().mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            _with_bpb_feature_diffmean_target([_bpb_row()]),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.source_contract().write_text(json.dumps({"not_the_hash": "x"}))
    with pytest.raises(AxBenchBpbAdvError, match="malformed source_contract"):
        replot_main(
            ["--output-root", str(tmp_path / "out"), "--run-tag", "run",
             "--metric", "bpb_class_margin"]
        )


def test_single_run_writes_provenance_only(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")
    paths.aggregates_dir().mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            _with_bpb_feature_diffmean_target(
                [_bpb_row(coefficient_str="1.0"), _bpb_row(coefficient_str="2.0")]
            ),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    _write_contract(paths, "solo-hash")

    rc = replot_main(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--run-tag",
            "run",
            "--metric",
            "bpb_class_margin",
        ]
    )

    assert rc == 0
    provenance_path = paths.steering_strength_plots_dir() / "provenance.yaml"
    assert provenance_path.exists()
    assert not (paths.steering_strength_plots_dir() / "combined_macro.parquet").exists()
    prov = yaml.safe_load(provenance_path.read_text())
    assert [r["run_tag"] for r in prov["runs"]] == ["run"]
    assert prov["runs"][0]["contract_hash"] == "solo-hash"
    assert prov["runs"][0]["n_macro_rows"] == 3
