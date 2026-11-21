"""Regression tests for progression plot orchestration.

These tests exercise the full ``plot_results`` orchestrator against
conditioned-result fixtures and assert structural invariants of the emitted
PNGs without depending on on-disk golden images.
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

from scripts.train.training_progression import (
    _preflight_harmfulness_plot_schemas,
    plot_results,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _full_results(tmp_path: Path) -> DictConfig:
    """Multi-phase, multi-metric conditioned results."""
    src = {
        "model_tag": "d27",
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
                                    "seeds": {"1234": {"core_metric": 0.05}}
                                }
                            }
                        }
                    },
                    "chat": {
                        "conditioned": {
                            "configs": {
                                "static_cfg": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {
                                            "ARC-Easy": 0.25,
                                            "MMLU": 0.26,
                                            "chatcore_metric": 0.25,
                                        }
                                    }
                                }
                            }
                        }
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
                                    "seeds": {"1234": {"core_metric": 0.10}}
                                }
                            }
                        }
                    },
                    "chat": {
                        "conditioned": {
                            "configs": {
                                "static_cfg": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {
                                            "ARC-Easy": 0.30,
                                            "MMLU": 0.31,
                                            "chatcore_metric": 0.30,
                                        }
                                    }
                                }
                            }
                        }
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
                                    "seeds": {"1234": {"core_metric": 0.15}}
                                }
                            }
                        }
                    },
                    "chat": {
                        "conditioned": {
                            "configs": {
                                "static_cfg": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {
                                            "ARC-Easy": 0.40,
                                            "MMLU": 0.42,
                                            "chatcore_metric": 0.41,
                                        }
                                    }
                                }
                            }
                        }
                    },
                },
            },
        ],
    }
    path = tmp_path / "results.json"
    with open(path, "w") as f:
        json.dump(src, f)
    return OmegaConf.create(
        {
            "model_tag": "d27",
            "results_file": str(path),
            "fig_dir": str(tmp_path / "fig"),
            "phases": [
                {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
                {"phase": "sft", "subdir": "", "steps": [200]},
            ],
        }
    )


def test_core_metric_chart_produced(tmp_path: Path):
    cfg = _full_results(tmp_path)
    plot_results(cfg)
    png = Path(cfg.fig_dir) / "core_metric.png"
    assert png.exists()
    img = mpimg.imread(str(png))
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_per_task_chat_charts_produced(tmp_path: Path):
    cfg = _full_results(tmp_path)
    plot_results(cfg)
    fig_dir = Path(cfg.fig_dir)
    # The fixture carries ARC-Easy, MMLU, and chatcore_metric data.
    assert (fig_dir / "chat_arc_easy.png").exists()
    assert (fig_dir / "chat_mmlu.png").exists()
    assert (fig_dir / "chatcore_metric.png").exists()


def test_no_conditioned_suffix_files(tmp_path: Path):
    cfg = _full_results(tmp_path)
    plot_results(cfg)
    assert not list(Path(cfg.fig_dir).glob("*__conditioned__*"))


def test_missing_harmfulness_skipped(tmp_path: Path):
    """Fixture has no harmfulness data; those charts must not be written."""
    cfg = _full_results(tmp_path)
    plot_results(cfg)
    fig_dir = Path(cfg.fig_dir)
    assert not list(fig_dir.glob("harmfulness_*.png"))


def _harmfulness_payload(schema_version: int = 4) -> dict:
    return {
        "harmfulness_eval_schema_version": schema_version,
        "harmful_prompts": {"asr": 0.2, "safe_response_rate": 0.8},
        "benign_prompts": {"compliance_rate": 0.9},
        "overall": {"mean_score": 1.5},
    }


def test_harmfulness_plot_preflight_accepts_current_decoder_and_conditioned_seeds() -> None:
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "harmfulness": {
                    "seeds": {"1234": _harmfulness_payload()},
                    "conditioned": {
                        "configs": {
                            "static": {
                                "seeds": {"1234": _harmfulness_payload()},
                            }
                        }
                    },
                }
            },
        }
    ]

    _preflight_harmfulness_plot_schemas(checkpoints)


@pytest.mark.parametrize("schema_version", [None, 2])
def test_harmfulness_plot_preflight_rejects_missing_or_stale_schema(
    schema_version: int | None,
) -> None:
    payload = _harmfulness_payload()
    if schema_version is None:
        del payload["harmfulness_eval_schema_version"]
    else:
        payload["harmfulness_eval_schema_version"] = schema_version
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {"harmfulness": {"seeds": {"1234": payload}}},
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"harmfulness\.seeds\['1234'\].*must equal 4",
    ):
        _preflight_harmfulness_plot_schemas(checkpoints)


def test_harmfulness_plot_preflight_rejects_mixed_conditioned_schemas() -> None:
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "harmfulness": {
                    "seeds": {"1234": _harmfulness_payload()},
                    "conditioned": {
                        "configs": {
                            "current": {"seeds": {"1234": _harmfulness_payload()}},
                            "stale": {"seeds": {"1234": _harmfulness_payload(2)}},
                        }
                    },
                }
            },
        }
    ]

    with pytest.raises(
        ValueError,
        match=r"conditioned\.configs\['stale'\]\.seeds\['1234'\].*must equal 4",
    ):
        _preflight_harmfulness_plot_schemas(checkpoints)


def test_harmfulness_plot_preflight_does_not_traverse_target_real() -> None:
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "harmfulness": {
                    "target_real": {
                        "configs": {
                            "unused": {"seeds": {"1234": _harmfulness_payload(2)}}
                        }
                    }
                }
            },
        }
    ]

    _preflight_harmfulness_plot_schemas(checkpoints)


@pytest.mark.parametrize(
    ("seeds", "match"),
    [
        ([], r"harmfulness\.seeds must be a mapping"),
        ({"1234": []}, r"harmfulness\.seeds\['1234'\] must be a mapping"),
    ],
)
def test_harmfulness_plot_preflight_rejects_invalid_seed_shapes(
    seeds: object,
    match: str,
) -> None:
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {"harmfulness": {"seeds": seeds}},
        }
    ]

    with pytest.raises(ValueError, match=match):
        _preflight_harmfulness_plot_schemas(checkpoints)
