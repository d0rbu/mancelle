from __future__ import annotations

import csv
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.profile_progression.report import render_macro_markdown
from scripts.profile_progression.runner import (
    _profile_path,
    _resolve_run_dir,
    _validate_path_c_conditioned_configs,
    _write_macro_csv,
    run_hotspot,
)
from steering.profiling.markers import TimingRecorder

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs/eval/profile_progression"


@pytest.mark.parametrize(
    ("selector", "model_family", "expected_path"),
    [("A", "decoder", "A"), ("C", "encoder_decoder", "C")],
)
def test_shipped_path_configs_use_current_routing_schema(
    selector, model_family, expected_path, tmp_path
):
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="profile_progression",
            overrides=[f"path={selector}"],
        )

    assert cfg.progression.model_family == model_family
    assert _profile_path(cfg) == expected_path
    assert "path" not in cfg
    assert "source" not in cfg.progression
    assert "profile_scale" not in cfg

    cfg.out_root = str(tmp_path)
    dirs = _resolve_run_dir(cfg)
    assert dirs.run_dir.name.endswith(f"_{expected_path}_macro")


def test_macro_csv_has_append_only_variant_columns(tmp_path):
    recorder = TimingRecorder()
    recorder.append(
        "run_core_eval",
        1_000_000_000,
        {
            "path": "C",
            "suite": "core",
            "eval_variant": "conditioned",
            "conditioned_config": "safe",
            "seed": 1234,
            "iter": 0,
            "warmup": False,
        },
    )
    out = tmp_path / "macro.csv"
    _write_macro_csv(recorder, out, path="C")

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0].keys() >= {
        "path",
        "suite",
        "stage",
        "iter",
        "warmup",
        "wall_seconds",
        "variant",
        "conditioned_config",
        "seed",
    }
    assert rows[0]["variant"] == "conditioned"
    assert rows[0]["conditioned_config"] == "safe"
    assert rows[0]["seed"] == "1234"


def test_macro_csv_accounts_for_context_coverage_preflight(tmp_path):
    recorder = TimingRecorder()
    recorder.append(
        "validate_context_coverage",
        250_000_000,
        {"path": "C", "iter": 0, "warmup": False},
    )
    out = tmp_path / "macro.csv"
    _write_macro_csv(recorder, out, path="C")

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["stage"] == "context_coverage"
    assert rows[0]["wall_seconds"] == "0.250000"


def test_macro_report_shows_conditioned_path_c_variant(tmp_path):
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "path,suite,stage,iter,warmup,wall_seconds,variant,conditioned_config,seed\n"
        "C,core,suite_total,0,true,20.0,conditioned,safe,1234\n"
        "C,core,suite_total,1,false,18.0,conditioned,safe,1234\n"
    )

    rendered = render_macro_markdown(csv_path)
    assert "| C | core | conditioned | safe" in rendered


def test_macro_report_aggregates_conditioned_rows_by_seed_and_warmup(tmp_path):
    csv_path = tmp_path / "macro.csv"
    csv_path.write_text(
        "path,suite,stage,iter,warmup,wall_seconds,variant,conditioned_config,seed\n"
        "C,core,suite_total,0,true,20.0,conditioned,safe,1234\n"
        "C,core,suite_total,1,false,18.0,conditioned,safe,1234\n"
        "C,core,suite_total,1,false,19.0,conditioned,safe,2234\n"
    )

    rendered = render_macro_markdown(csv_path)
    conditioned = next(
        line
        for line in rendered.splitlines()
        if line.startswith("| C | core | conditioned | safe |")
    )

    assert "| 18.5 |" in conditioned
    assert "| 2 |" in conditioned
    assert conditioned.endswith("| 20.0 |")


def test_path_c_profile_requires_conditioned_configs():
    progression_cfg = OmegaConf.create(
        {
            "model_family": "encoder_decoder",
            "phase": "pretrain",
            "conditioning": {
                "conditioned_configs": [],
            },
        }
    )

    with pytest.raises(ValueError, match="path=C profiling requires"):
        _validate_path_c_conditioned_configs(progression_cfg)


@pytest.mark.parametrize(
    ("model_family", "expected"),
    [("decoder", "A"), ("encoder_decoder", "C")],
)
def test_profile_path_is_derived_from_model_family(model_family, expected):
    cfg = OmegaConf.create({"progression": {"model_family": model_family}})
    assert _profile_path(cfg) == expected


def test_profile_path_rejects_automatic_model_detection():
    cfg = OmegaConf.create({"progression": {"model_family": "auto"}})
    with pytest.raises(ValueError, match="must be decoder or encoder_decoder"):
        _profile_path(cfg)


def test_hotspot_null_fixture_uses_default_path(monkeypatch, tmp_path):
    seen = {}

    def fake_get_microbench(name: str):
        assert name == "encdec_build_conditioning"

        def fake_run(fixture_path, device_str: str, n_iters: int):
            seen["fixture_path"] = fixture_path
            seen["device_str"] = device_str
            seen["n_iters"] = n_iters
            return [{"name": name, "ok": True}]

        return fake_run

    monkeypatch.setattr(
        "scripts.profile_progression.runner.get_microbench",
        fake_get_microbench,
    )
    cfg = OmegaConf.create(
        {
            "hotspot_name": "encdec_build_conditioning",
            "hotspot_fixture": None,
            "hotspot_n_iters": 3,
            "out_root": str(tmp_path / "out"),
        }
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_hotspot(cfg, run_dir)

    assert str(seen["fixture_path"]) == (
        "tests/fixtures/profile_progression/"
        "encdec_build_conditioning.pt"
    )
    assert seen["n_iters"] == 3
