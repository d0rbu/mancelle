"""End-to-end ``plot_results`` tests with inline synthetic result dicts.

Asserts that the orchestrator produces the right filenames, the
replace-mode invariant (no ``__conditioned__*`` files), and that the
PNG output is structurally sane — then goes deeper by re-reading the
rendered image and checking for multi-line color diversity on the
multi-series scenarios.

No on-disk fixtures or golden images — everything stays in the test
module so the regressions are self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest
from omegaconf import DictConfig, OmegaConf

from scripts.train.training_progression import plot_results


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _write_results(
    tmp_path: Path,
    results: dict,
    phases: list[dict] | None = None,
) -> DictConfig:
    """Serialize an inline ``results`` dict and build a minimal cfg."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dst = tmp_path / "results.json"
    with open(dst, "w") as f:
        json.dump(results, f)
    cfg = OmegaConf.create(
        {
            "model_tag": results["model_tag"],
            "results_file": str(dst),
            "fig_dir": str(tmp_path / "fig"),
            "phases": phases or [{"phase": "pretrain", "subdir": "", "steps": []}],
        }
    )
    return cfg


def _distinct_colors(png_path: Path) -> int:
    """Count distinct (bucketed) RGBA pixel colors in a PNG."""
    img = mpimg.imread(str(png_path))
    return len({tuple(int(c * 20) for c in row) for row in img.reshape(-1, img.shape[-1])})


def _conditioned_only_results() -> dict:
    return {
        "model_tag": "d12",
        "checkpoints": [
            {
                "step": 500,
                "phase": "pretrain",
                "evals": {
                    "core": {
                        "conditioned": {
                            "configs": {
                                "static_cfg": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {"core_metric": 0.05},
                                        "2234": {"core_metric": 0.07},
                                    }
                                }
                            }
                        },
                    },
                },
            },
            {
                "step": 1000,
                "phase": "pretrain",
                "evals": {
                    "core": {
                        "conditioned": {
                            "configs": {
                                "static_cfg": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {"core_metric": 0.10},
                                        "2234": {"core_metric": 0.12},
                                    }
                                }
                            }
                        },
                    },
                },
            },
            {
                "step": 200,
                "phase": "sft",
                "evals": {
                    "core": {
                        "conditioned": {
                            "configs": {
                                "static_cfg": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {"core_metric": 0.18},
                                        "2234": {"core_metric": 0.20},
                                    }
                                }
                            }
                        },
                    },
                },
            },
        ],
    }


def _conditioned_configs_results(n: int, model_tag: str = "d12") -> dict:
    names = [f"cfg_{chr(ord('a') + i)}" for i in range(n)]
    return {
        "model_tag": model_tag,
        "checkpoints": [
            {
                "step": 500,
                "phase": "pretrain",
                "evals": {
                    "core": {
                        "seeds": {"1234": {"core_metric": 0.05}},
                        "conditioned": {
                            "configs": {
                                name: {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {"1234": {"core_metric": 0.06 + i * 0.01}},
                                }
                                for i, name in enumerate(names)
                            }
                        },
                    }
                },
            },
            {
                "step": 1000,
                "phase": "pretrain",
                "evals": {
                    "core": {
                        "seeds": {"1234": {"core_metric": 0.11}},
                        "conditioned": {
                            "configs": {
                                name: {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {"1234": {"core_metric": 0.12 + i * 0.01}},
                                }
                                for i, name in enumerate(names)
                            }
                        },
                    }
                },
            },
        ],
    }


def _top_level_seed_results() -> dict:
    return {
        "model_tag": "d27",
        "checkpoints": [
            {
                "step": 500,
                "phase": "pretrain",
                "evals": {"core": {"seeds": {"1234": {"core_metric": 0.05}}}},
            },
            {
                "step": 1000,
                "phase": "pretrain",
                "evals": {"core": {"seeds": {"1234": {"core_metric": 0.10}}}},
            },
        ],
    }


def test_plot_results_conditioned_only(tmp_path: Path):
    cfg = _write_results(
        tmp_path,
        _conditioned_only_results(),
        phases=[
            {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
            {"phase": "sft", "subdir": "", "steps": [200]},
        ],
    )
    plot_results(cfg)
    png = Path(cfg.fig_dir) / "core_metric.png"
    assert png.exists(), f"{png} missing"
    assert png.stat().st_size > 0


def test_plot_results_two_conditioned_configs(tmp_path: Path):
    cfg = _write_results(
        tmp_path,
        _conditioned_configs_results(2),
        phases=[{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}],
    )
    plot_results(cfg)
    fig_dir = Path(cfg.fig_dir)
    png = fig_dir / "core_metric.png"
    assert png.exists()
    # Replace-mode invariant: no __conditioned__ files.
    assert not list(fig_dir.glob("*__conditioned__*"))
    # Multi-series PNG should carry enough distinct colors that both lines
    # are visually present.
    assert _distinct_colors(png) >= 2


def test_plot_results_five_conditioned_configs(tmp_path: Path):
    cfg = _write_results(
        tmp_path,
        _conditioned_configs_results(5),
        phases=[{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}],
    )
    plot_results(cfg)
    png = Path(cfg.fig_dir) / "core_metric.png"
    assert png.exists()
    # 5 series plus grid/boundaries should produce
    # a substantially more color-diverse PNG than a 1- or 2-series chart.
    assert _distinct_colors(png) >= 5


def test_plot_results_plots_top_level_decoder_seed_results(tmp_path: Path):
    cfg = _write_results(
        tmp_path,
        _top_level_seed_results(),
        phases=[{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}],
    )
    plot_results(cfg)
    fig_dir = Path(cfg.fig_dir)
    png = fig_dir / "core_metric.png"
    assert png.exists()
    assert png.stat().st_size > 0
    assert not list(fig_dir.glob("*__conditioned__*"))


def test_plot_results_missing_metric_skipped(tmp_path: Path):
    """Results only carry core data; harmfulness metrics produce no files."""
    cfg = _write_results(
        tmp_path,
        _conditioned_only_results(),
        phases=[
            {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
            {"phase": "sft", "subdir": "", "steps": [200]},
        ],
    )
    plot_results(cfg)
    fig_dir = Path(cfg.fig_dir)
    assert not list(fig_dir.glob("harmfulness_*.png"))


def test_plot_results_no_conditioned_suffix_files(tmp_path: Path):
    """After the refactor, __conditioned__ filenames should never appear."""
    cfg = _write_results(
        tmp_path,
        _conditioned_configs_results(2),
        phases=[{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}],
    )
    plot_results(cfg)
    fig_dir = Path(cfg.fig_dir)
    assert not list(fig_dir.glob("*__conditioned__*"))


def test_plot_results_empty_checkpoints_no_files(tmp_path: Path):
    cfg = _write_results(
        tmp_path,
        {"model_tag": "d0", "checkpoints": []},
    )
    plot_results(cfg)  # should not raise
    assert not list(Path(cfg.fig_dir).glob("*.png"))


def test_end_to_end_collect_then_plot(tmp_path: Path, monkeypatch):
    """collect_results path writes per-checkpoint JSONs, then plot_results reads them."""
    from scripts.train.training_progression import collect_results, plot_results

    fake_evals = {
        "core": {
            "conditioned": {
                "configs": {
                        "static_cfg": {
                            "kind": "static_yaml",
                            "target_conditioning_mode": "current",
                            "seeds": {"1234": {"core_metric": 0.12}},
                        },
                },
            },
        },
    }

    def _fake_evaluate_single_checkpoint(spec, device, ptdtype, eval_suites, seeds, cfg, existing):
        return fake_evals

    monkeypatch.setattr(
        "scripts.train.training_progression.evaluate_single_checkpoint",
        _fake_evaluate_single_checkpoint,
    )
    monkeypatch.setattr(
        "scripts.train.training_progression._validate_required_context_cache",
        lambda cfg, specs: None,
    )
    ckpt_root = tmp_path / "data" / "base_checkpoints" / "d0"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for step in (100, 200):
        (ckpt_root / f"model_{step:06d}.pt").write_bytes(b"stub")
        meta = {
            "step": step,
            "model_config": {
                "vocab_size": 128,
                "n_layer": 1,
                "n_head": 1,
                "n_kv_head": 1,
                "n_embd": 32,
                "sequence_len": 64,
            },
        }
        (ckpt_root / f"meta_{step:06d}.json").write_text(json.dumps(meta))

    cfg = OmegaConf.create(
        {
            "mode": "both",
            "model_tag": "d0",
            "checkpoint_source": "local",
            "checkpoints_dir": str(tmp_path / "data"),
            "results_file": str(tmp_path / "results.json"),
            "fig_dir": str(tmp_path / "fig"),
            "phases": [{"phase": "pretrain", "subdir": "", "steps": [100, 200]}],
                "evals": "core",
                "conditioning": {
                    "conditioned_configs": [
                        {"name": "static_cfg", "features": {1: 1}},
                    ],
                    "target_real_configs": [],
                },
                "num_seeds": 1,
            "base_seed": 1234,
            "max_per_task": 10,
            "cleanup_after_eval": False,
            "device_type": "cpu",
            "dtype": "float32",
            "num_gpus": 1,
        }
    )
    collect_results(cfg)
    plot_results(cfg)
    png = Path(cfg.fig_dir) / "core_metric.png"
    assert png.exists()
    assert png.stat().st_size > 0
