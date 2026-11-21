"""load_progression_data: strict per-seed schema, mean-across-seeds correctness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.eval.harmfulness_progression import (
    load_progression_data,
)


def _write_results(
    tmp_path: Path,
    checkpoints: list[dict],
    name: str = "results.json",
) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"checkpoints": checkpoints}))
    return p


def _ckpt(
    *,
    step: int,
    phase: str,
    seeds: dict | None = None,
    raw_harmfulness: dict | None = None,
) -> dict:
    if raw_harmfulness is not None:
        h = raw_harmfulness
    elif seeds is not None:
        h = {"seeds": seeds}
    else:
        h = {}
    return {"step": step, "phase": phase, "evals": {"harmfulness": h}}


def _seed(
    *,
    asr: float,
    safe_rate: float,
    mean_score: float,
    compliance_rate: float = 0.75,
) -> dict:
    return {
        "harmfulness_eval_schema_version": 4,
        "harmful_prompts": {
            "asr": asr,
            "safe_response_rate": safe_rate,
        },
        "benign_prompts": {"compliance_rate": compliance_rate},
        "overall": {"mean_score": mean_score},
    }


def test_mean_across_seeds_is_arithmetic_mean(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": _seed(asr=0.2, safe_rate=0.8, mean_score=2.0),
                    "2234": _seed(asr=0.4, safe_rate=0.6, mean_score=3.0),
                    "3234": _seed(asr=0.6, safe_rate=0.4, mean_score=4.0),
                },
            ),
        ],
    )
    pts = load_progression_data(p)
    assert len(pts) == 1
    pt = pts[0]
    assert pt.asr == pytest.approx(0.4)
    assert pt.safe_rate == pytest.approx(0.6)
    assert pt.compliance_rate == pytest.approx(0.75)
    assert pt.mean_score == pytest.approx(3.0)


def test_filter_min_step_excludes_pretrain_below_threshold(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=10000,
                phase="pretrain",
                seeds={
                    "1234": _seed(asr=0.0, safe_rate=1.0, mean_score=1.0),
                },
            ),
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": _seed(asr=0.5, safe_rate=0.5, mean_score=2.5),
                },
            ),
        ],
    )
    pts = load_progression_data(p, min_step=53000)
    assert [(pt.step, pt.phase) for pt in pts] == [(53000, "pretrain")]


def test_filter_min_step_does_not_exclude_sft_phases(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=100,
                phase="safety_sft",
                seeds={
                    "1234": _seed(asr=0.1, safe_rate=0.9, mean_score=1.5),
                },
            ),
        ],
    )
    pts = load_progression_data(p, min_step=53000)
    assert [(pt.step, pt.phase) for pt in pts] == [(100, "safety_sft")]


def test_sort_order_pretrain_safety_capabilities_then_step(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=1000,
                phase="capabilities_sft",
                seeds={"1": _seed(asr=0.0, safe_rate=1.0, mean_score=1.0)},
            ),
            _ckpt(
                step=54000,
                phase="pretrain",
                seeds={"1": _seed(asr=0.0, safe_rate=1.0, mean_score=1.0)},
            ),
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={"1": _seed(asr=0.0, safe_rate=1.0, mean_score=1.0)},
            ),
            _ckpt(
                step=900,
                phase="safety_sft",
                seeds={"1": _seed(asr=0.0, safe_rate=1.0, mean_score=1.0)},
            ),
        ],
    )
    pts = load_progression_data(p)
    assert [(pt.phase, pt.step) for pt in pts] == [
        ("pretrain", 53000),
        ("pretrain", 54000),
        ("safety_sft", 900),
        ("capabilities_sft", 1000),
    ]


def test_parses_seeds_shape(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": _seed(asr=0.5, safe_rate=0.5, mean_score=2.5),
                },
            ),
        ],
    )
    pts = load_progression_data(p)
    assert len(pts) == 1


def test_raises_when_seed_has_asr_but_missing_safe_response_rate(
    tmp_path: Path,
) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": {
                        "harmfulness_eval_schema_version": 4,
                        "harmful_prompts": {"asr": 0.5},
                        "benign_prompts": {"compliance_rate": 0.5},
                        "overall": {"mean_score": 2.0},
                    },
                },
            ),
        ],
    )
    with pytest.raises(ValueError, match="invalid nested harmfulness schema"):
        load_progression_data(p)


def test_raises_when_seed_has_asr_but_missing_mean_score(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": {
                        "harmfulness_eval_schema_version": 4,
                        "harmful_prompts": {
                            "asr": 0.5,
                            "safe_response_rate": 0.5,
                        },
                        "benign_prompts": {"compliance_rate": 0.5},
                        "overall": {},
                    },
                },
            ),
        ],
    )
    with pytest.raises(ValueError, match="invalid nested harmfulness schema"):
        load_progression_data(p)


@pytest.mark.parametrize("mean_score", [0.0, 5.01, float("nan"), float("inf"), True])
def test_rejects_invalid_nested_mean_score(tmp_path: Path, mean_score: object) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": _seed(
                        asr=0.5,
                        safe_rate=0.5,
                        mean_score=mean_score,  # ty: ignore[invalid-argument-type]
                    )
                },
            )
        ],
    )
    with pytest.raises(ValueError, match=r"mean_score.*(must be numeric|\[1\.0, 5\.0\])"):
        load_progression_data(p)


def test_raises_when_harmfulness_has_no_seeds(
    tmp_path: Path,
) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(step=53000, phase="pretrain", raw_harmfulness={"some_other_key": {"asr": 0.5}}),
        ],
    )
    with pytest.raises(ValueError, match="no 'seeds' mapping"):
        load_progression_data(p)


def test_empty_harmfulness_skipped_silently(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(step=53000, phase="pretrain", raw_harmfulness={}),
            _ckpt(
                step=54000,
                phase="pretrain",
                seeds={
                    "1234": _seed(asr=0.5, safe_rate=0.5, mean_score=2.5),
                },
            ),
        ],
    )
    pts = load_progression_data(p)
    assert [pt.step for pt in pts] == [54000]


def test_h_is_none_silently_skipped(tmp_path: Path) -> None:
    """Covers harmfulness_progression.py:410-411 — checkpoint with
    `evals: {harmfulness: null}` produces no point and does not raise."""
    # Build manually because _ckpt always wraps in a dict.
    raw = {
        "checkpoints": [
            {"step": 53000, "phase": "pretrain", "evals": {"harmfulness": None}},
            {
                "step": 54000,
                "phase": "pretrain",
                "evals": {
                    "harmfulness": {
                        "seeds": {
                            "1234": _seed(asr=0.5, safe_rate=0.5, mean_score=2.5),
                        }
                    }
                },
            },
        ]
    }
    p = tmp_path / "h_is_none.json"
    p.write_text(json.dumps(raw))
    pts = load_progression_data(p)
    assert [pt.step for pt in pts] == [54000]


def test_h_is_non_dict_raises(tmp_path: Path) -> None:
    """Covers harmfulness_progression.py:413-417 — checkpoint with
    `evals: {harmfulness: 42}` raises ValueError."""
    raw = {
        "checkpoints": [
            {"step": 53000, "phase": "pretrain", "evals": {"harmfulness": 42}},
        ]
    }
    p = tmp_path / "h_is_int.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="non-dict harmfulness evals"):
        load_progression_data(p)


def test_seeds_is_not_dict_silently_skipped(tmp_path: Path) -> None:
    """Covers harmfulness_progression.py:434-435 — `seeds = []` (a list,
    not a dict) produces no point and does not raise."""
    raw = {
        "checkpoints": [
            {"step": 53000, "phase": "pretrain", "evals": {"harmfulness": {"seeds": []}}},
        ]
    }
    p = tmp_path / "seeds_list.json"
    p.write_text(json.dumps(raw))
    pts = load_progression_data(p)
    assert pts == []


def test_seeds_is_empty_dict_silently_skipped(tmp_path: Path) -> None:
    """Covers harmfulness_progression.py:434-435 — empty `seeds = {}`
    produces no point and does not raise."""
    p = _write_results(
        tmp_path,
        [
            _ckpt(step=53000, phase="pretrain", seeds={}),
        ],
    )
    pts = load_progression_data(p)
    assert pts == []


def test_seed_value_not_a_dict_raises(tmp_path: Path) -> None:
    """Covers harmfulness_progression.py:440-444 — a seed entry that is
    not a dict (e.g. ``42``) raises ValueError mentioning the type."""
    raw = {
        "checkpoints": [
            {"step": 53000, "phase": "pretrain", "evals": {"harmfulness": {"seeds": {"1234": 42}}}},
        ]
    }
    p = tmp_path / "seed_int.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"expected dict, got int"):
        load_progression_data(p)


def test_legacy_empty_seed_dict_fails_loudly(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": {"harmfulness_eval_schema_version": 4},
                    "2234": {"harmfulness_eval_schema_version": 4},
                    "3234": {"harmfulness_eval_schema_version": 4},
                },
            ),
            # A valid checkpoint after to confirm only the empty one is filtered.
            _ckpt(
                step=54000,
                phase="pretrain",
                seeds={
                    "1234": _seed(asr=0.4, safe_rate=0.6, mean_score=2.0),
                },
            ),
        ],
    )
    with pytest.raises(ValueError, match="invalid nested harmfulness schema"):
        load_progression_data(p)


def test_legacy_flat_seed_fails_with_migration_error(tmp_path: Path) -> None:
    p = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={
                    "1234": {
                        "harmfulness_eval_schema_version": 4,
                        "asr": 0.4,
                        "safe_response_rate": 0.6,
                        "mean_score": 2.0,
                    }
                },
            )
        ],
    )
    with pytest.raises(ValueError, match="legacy flat harmfulness seed"):
        load_progression_data(p)


@pytest.mark.parametrize("schema_version", [None, 2])
def test_rejects_missing_or_stale_seed_schema(
    tmp_path: Path,
    schema_version: int | None,
) -> None:
    seed = _seed(asr=0.4, safe_rate=0.6, mean_score=2.0)
    if schema_version is None:
        del seed["harmfulness_eval_schema_version"]
    else:
        seed["harmfulness_eval_schema_version"] = schema_version
    path = _write_results(
        tmp_path,
        [_ckpt(step=53000, phase="pretrain", seeds={"1234": seed})],
    )

    with pytest.raises(
        ValueError,
        match=r"harmfulness_eval_schema_version must equal 4",
    ):
        load_progression_data(path)


def test_rejects_mixed_current_and_stale_seed_schemas(tmp_path: Path) -> None:
    current = _seed(asr=0.4, safe_rate=0.6, mean_score=2.0)
    stale = _seed(asr=0.2, safe_rate=0.8, mean_score=1.5)
    stale["harmfulness_eval_schema_version"] = 2
    path = _write_results(
        tmp_path,
        [
            _ckpt(
                step=53000,
                phase="pretrain",
                seeds={"1234": current, "2234": stale},
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"seed '2234'.*harmfulness_eval_schema_version must equal 4",
    ):
        load_progression_data(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("harmful_prompts", "asr", True),
        ("harmful_prompts", "safe_response_rate", "0.5"),
        ("benign_prompts", "compliance_rate", None),
        ("overall", "mean_score", float("nan")),
        ("overall", "mean_score", 5.1),
    ],
)
def test_nested_seed_rejects_invalid_metric_values(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    seed = _seed(asr=0.4, safe_rate=0.6, mean_score=2.0)
    seed[section][field] = value
    path = _write_results(
        tmp_path,
        [_ckpt(step=53000, phase="pretrain", seeds={"1234": seed})],
    )
    with pytest.raises(ValueError, match=f"invalid nested harmfulness schema.*{field}"):
        load_progression_data(path)
