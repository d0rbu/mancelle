"""Direction-fitting contracts for AxBench encoder-decoder baselines."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from sklearn.linear_model import LogisticRegression

from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_upstream.encdec_baselines import directions
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    PROBE_C,
    PROBE_MAX_ITER,
    PROBE_SEED_DOMAIN,
    PROBE_SOLVER,
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
    load_direction,
    save_direction,
    unit_norm_vendored,
)
from steering.safetensors_io import canonicalize_safetensors_bytes

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "axbench_encdec"


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def _fixture(name: str) -> dict[str, t.Tensor]:
    return load_file(FIXTURE_DIR / f"{name}_fixture.safetensors", device="cpu")


def _diffmean_rows() -> tuple[list[t.Tensor], list[t.Tensor]]:
    fixture = _fixture("diffmean")
    positive = [fixture[f"positive_row_{index}"] for index in range(8)]
    negative = [fixture[f"negative_row_{index}"] for index in range(8)]
    return positive, negative


def _probe_rows() -> tuple[list[t.Tensor], list[t.Tensor]]:
    positive = [
        t.tensor([[2.0, 1.0, 0.0], [4.0, 3.0, 1.0]]),
        t.tensor([[1.5, 2.0, 0.5], [3.0, 2.5, 1.0], [2.0, 3.0, 0.0]]),
        t.tensor([[2.5, 0.5, 1.5], [3.5, 1.5, 0.5]]),
    ]
    negative = [
        t.tensor([[-2.0, -1.0, 0.0], [-4.0, -3.0, -1.0]]),
        t.tensor([[-1.5, -2.0, -0.5], [-3.0, -2.5, -1.0]]),
        t.tensor([[-2.5, -0.5, -1.5], [-3.5, -1.5, -0.5], [-2.0, -3.0, 0.0]]),
    ]
    return positive, negative


def _direct_probe_direction(
    positive: list[t.Tensor],
    negative: list[t.Tensor],
    *,
    seed: int,
    concept_id: int,
) -> t.Tensor:
    rows = [row.detach().to(dtype=t.float32, device="cpu") for row in [*positive, *negative]]
    row_means = t.stack([row.mean(dim=0) for row in rows])
    labels = t.tensor([1] * len(positive) + [0] * len(negative), dtype=t.int64)
    estimator = LogisticRegression(
        l1_ratio=1.0,
        C=1.0,
        max_iter=500,
        solver="liblinear",
        random_state=directions._derive_probe_seed(seed, concept_id),
        class_weight=None,
    ).fit(row_means.numpy(), labels.numpy())
    return F.normalize(t.as_tensor(estimator.coef_[0], dtype=t.float32), dim=0)


@pytest.mark.parametrize("fit_name", ["fit_diffmean_direction", "fit_probe_direction"])
@pytest.mark.parametrize("calibration_only", [False, True])
def test_skipping_calibration_preserves_fit_and_never_computes_statistic(
    monkeypatch: pytest.MonkeyPatch, fit_name: str, calibration_only: bool,
) -> None:
    positive, negative = _probe_rows()
    fit = getattr(directions, fit_name)
    kwargs = {"seed": 42} if fit_name == "fit_probe_direction" else {}
    expected = fit(positive, negative, intervention_layer=1, surface="skip", **kwargs)

    def unexpected_calibration(*_args, **_kwargs):
        pytest.fail("calibrate=False must not compute an unused statistic")

    monkeypatch.setattr(directions, "compute_direction_max_activation", unexpected_calibration)
    actual = fit(
        positive, negative, intervention_layer=1, surface="skip",
        calibrate=False, calibration_only=calibration_only, **kwargs,
    )
    assert t.equal(actual.direction, expected.direction)
    assert actual.metadata == expected.metadata
    assert actual.max_activation == 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_probe_seed_derivation_matches_canonical_sha256_golden() -> None:
    fields = (PROBE_SEED_DOMAIN, "42", "0")
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).digest()
    expected = int.from_bytes(digest[:8], "big") % (2**31 - 1)

    assert expected == 1802994348
    assert directions._derive_probe_seed(42, 0) == expected


#: Independent literals pinned in TEST code (house rule: committed blob +
#: sha256). Reading expectations only from the adjacent MANIFEST would let a
#: coordinated fixture+manifest regeneration silently replace the oracle.
_PINNED_FIXTURE_SHA256 = {
    "MANIFEST.json": "acd27e7aae19a8eb863b1ddaed24845ed736b5b9bf74fef8731377b74d958441",
    "diffmean_fixture.safetensors": (
        "8cdf17527d5f39f4c08b10fb0a50276515bf2ccd05eccac7a98157d8da0e2335"
    ),
    "lora_fixture.safetensors": (
        "a3b83336a2a93a0da01111d072375365634f3d38bf1014ea1759575afe5ca371"
    ),
}


def test_fixture_sha256s_match_manifest_and_pinned_literals() -> None:
    # The directory holds EXACTLY the pinned files — an undeclared committed
    # fixture is a failure, not an unnoticed addition.
    assert {path.name for path in FIXTURE_DIR.iterdir()} == set(_PINNED_FIXTURE_SHA256)
    for filename, pinned in _PINNED_FIXTURE_SHA256.items():
        assert _file_sha256(FIXTURE_DIR / filename) == pinned

    manifest = json.loads((FIXTURE_DIR / "MANIFEST.json").read_text())
    assert set(manifest["files"]) == set(_PINNED_FIXTURE_SHA256) - {"MANIFEST.json"}
    for filename, record in manifest["files"].items():
        assert record["sha256"] == _PINNED_FIXTURE_SHA256[filename]


def test_diffmean_reproduces_vendored_fixture_and_falsifies_wrong_math() -> None:
    fixture = _fixture("diffmean")
    positive, negative = _diffmean_rows()

    fitted = fit_diffmean_direction(
        positive, negative, intervention_layer=9, surface="fixture diffmean"
    )

    assert t.equal(fitted.direction, fixture["direction"])
    raw = t.cat(positive).mean(dim=0) - t.cat(negative).mean(dim=0)
    assert t.equal(raw, fixture["raw"])

    normalize_alternative = F.normalize(raw, dim=0)
    assert not t.equal(normalize_alternative, fixture["direction"])

    row_weighted_raw = t.stack([row.mean(dim=0) for row in positive]).mean(dim=0)
    row_weighted_raw -= t.stack([row.mean(dim=0) for row in negative]).mean(dim=0)
    row_weighted_direction = unit_norm_vendored(row_weighted_raw)
    assert not t.equal(row_weighted_direction, fixture["direction"])


def test_probe_matches_direct_sklearn_and_normalize_differential() -> None:
    positive, negative = _probe_rows()
    seed = 314159
    concept_id = 7

    fitted = fit_probe_direction(
        positive,
        negative,
        seed=seed,
        concept_id=concept_id,
        intervention_layer=9,
        surface="probe differential",
    )
    expected = _direct_probe_direction(
        positive,
        negative,
        seed=seed,
        concept_id=concept_id,
    )

    assert t.equal(fitted.direction, expected)
    assert fitted.metadata["estimator"] == {
        "name": "sklearn.linear_model.LogisticRegression",
        "pooling": "per_row_mean_post_prefix_activations",
        "solver": PROBE_SOLVER,
        "l1_ratio": "1.0 (L1)",
        "C": PROBE_C,
        "max_iter": PROBE_MAX_ITER,
        "random_state": directions._derive_probe_seed(seed, concept_id),
        "class_weight": None,
        "direction": "F.normalize(coef_[0])",
    }


def test_probe_differential_falsifies_dense_l2_variant() -> None:
    row_means = t.tensor(
        [
            [2.0, 2.1, 0.1],
            [1.8, 1.9, -0.1],
            [2.2, 2.0, 0.05],
            [1.5, 1.6, 0.0],
            [-2.0, -2.1, -0.1],
            [-1.8, -1.9, 0.1],
            [-2.2, -2.0, -0.05],
            [-1.5, -1.6, 0.0],
        ],
        dtype=t.float32,
    )
    positive = [row[None, :] for row in row_means[:4]]
    negative = [row[None, :] for row in row_means[4:]]
    fitted = fit_probe_direction(
        positive,
        negative,
        seed=1,
        intervention_layer=9,
        surface="L1 versus L2 falsification",
    )
    labels = t.tensor([1] * 4 + [0] * 4, dtype=t.int64)
    l2_estimator = LogisticRegression(
        C=1.0,
        max_iter=500,
        solver="liblinear",
        random_state=directions._derive_probe_seed(1, 0),
        class_weight=None,
    ).fit(row_means.numpy(), labels.numpy())
    l2_direction = F.normalize(
        t.as_tensor(l2_estimator.coef_[0], dtype=t.float32),
        dim=0,
    )

    assert not t.equal(fitted.direction, l2_direction)


def test_probe_is_byte_deterministic_and_data_sensitive() -> None:
    positive, negative = _probe_rows()
    first = fit_probe_direction(
        positive,
        negative,
        seed=23,
        concept_id=4,
        intervention_layer=9,
        surface="probe determinism",
    )
    second = fit_probe_direction(
        positive,
        negative,
        seed=23,
        concept_id=4,
        intervention_layer=9,
        surface="probe determinism",
    )
    changed_positive = [row.clone() for row in positive]
    changed_positive[0][0] += t.tensor([-10.0, 20.0, 0.0])
    changed = fit_probe_direction(
        changed_positive,
        negative,
        seed=23,
        concept_id=4,
        intervention_layer=9,
        surface="probe changed data",
    )

    assert first.direction.numpy().tobytes() == second.direction.numpy().tobytes()
    assert not t.equal(first.direction, changed.direction)


def test_probe_uses_one_mean_per_row_not_token_weighted_examples() -> None:
    positive = [
        t.tensor([[8.0, 0.0]]),
        t.tensor([[0.0, 2.0]]).repeat(9, 1),
    ]
    negative = [
        t.tensor([[-8.0, 0.0]]),
        t.tensor([[0.0, -2.0]]).repeat(9, 1),
    ]
    fitted = fit_probe_direction(
        positive,
        negative,
        seed=42,
        concept_id=0,
        intervention_layer=9,
        surface="row pooling falsification",
    )
    expected = _direct_probe_direction(positive, negative, seed=42, concept_id=0)

    token_activations = t.cat([*positive, *negative])
    token_labels = t.tensor(
        [1] * sum(row.shape[0] for row in positive) + [0] * sum(row.shape[0] for row in negative),
        dtype=t.int64,
    )
    token_estimator = LogisticRegression(
        l1_ratio=1.0,
        C=1.0,
        max_iter=500,
        solver="liblinear",
        random_state=directions._derive_probe_seed(42, 0),
        class_weight=None,
    ).fit(token_activations.numpy(), token_labels.numpy())
    token_direction = F.normalize(
        t.as_tensor(token_estimator.coef_[0], dtype=t.float32),
        dim=0,
    )

    assert t.equal(fitted.direction, expected)
    assert not t.equal(fitted.direction, token_direction)


def test_probe_direction_has_fnormalize_unit_norm_and_fp32() -> None:
    positive, negative = _probe_rows()

    fitted = fit_probe_direction(
        positive,
        negative,
        seed=42,
        concept_id=0,
        intervention_layer=9,
        surface="probe normalization",
    )

    assert fitted.direction.dtype == t.float32
    assert fitted.direction.device.type == "cpu"
    assert abs(float(fitted.direction.norm()) - 1.0) <= 1.0e-6
    assert t.equal(fitted.direction, F.normalize(fitted.direction, dim=0))


def test_diffmean_zero_direction_delegates_zero_max_activation() -> None:
    rows = [t.tensor([[1.0, -2.0], [3.0, 4.0]])]

    fitted = fit_diffmean_direction(
        rows, [rows[0].clone()], intervention_layer=9, surface="zero direction"
    )

    assert t.equal(fitted.direction, t.zeros(2))
    assert fitted.max_activation == 0.0


@pytest.mark.parametrize("fit_name", ["diffmean", "probe"])
def test_nonpositive_max_activation_error_is_not_caught(fit_name: str) -> None:
    positive = [t.tensor([[-1.0, 0.0]])]
    negative = [t.tensor([[-2.0, 0.0]])]
    if fit_name == "probe":
        positive *= 4
        negative *= 4

    with pytest.raises(AxBenchBaseError, match="non-positive max_activation"):
        if fit_name == "diffmean":
            fit_diffmean_direction(
                positive, negative, intervention_layer=9, surface="negative projection"
            )
        else:
            fit_probe_direction(
                positive,
                negative,
                seed=42,
                concept_id=0,
                intervention_layer=9,
                surface="negative projection",
            )


def test_save_direction_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torn serialization leaves no final-path artifact and no stale temp."""

    fitted = fit_diffmean_direction(
        [t.ones(2, 8)],
        [t.zeros(2, 8)],
        intervention_layer=9,
        surface="atomic save",
    )
    path = tmp_path / "concept_00000_diffmean.safetensors"

    def torn_save(_tensors: object, *, metadata: object) -> bytes:
        del _tensors, metadata
        raise OSError("disk full mid-serialize")

    monkeypatch.setattr(directions, "deterministic_safetensors_bytes", torn_save)
    with pytest.raises(OSError, match="disk full mid-serialize"):
        save_direction(path, fitted)
    assert not path.exists()

    monkeypatch.undo()
    save_direction(path, fitted)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_direction_save_load_round_trip_is_byte_exact_and_pins_metadata_keys(
    tmp_path: Path,
) -> None:
    fitted = FittedDirection(
        direction=t.tensor([0.25, -0.5, 0.75], dtype=t.float32),
        max_activation=3.125,
        metadata={"method": "probe", "seed": 17, "nested": {"steps": 48}},
    )
    first_path = tmp_path / "first.safetensors"

    save_direction(first_path, fitted)
    loaded = load_direction(first_path)

    assert loaded.direction.numpy().tobytes() == fitted.direction.numpy().tobytes()
    assert loaded.max_activation == fitted.max_activation
    assert loaded.metadata == fitted.metadata
    with safe_open(first_path, framework="pt", device="cpu") as handle:
        assert set(handle.metadata() or {}) == {
            "format",
            "max_activation",
            "metadata_json",
        }
        assert list(handle.keys()) == ["direction"]
    assert first_path.read_bytes() == canonicalize_safetensors_bytes(first_path.read_bytes())

    second_path = tmp_path / "second.safetensors"
    save_direction(
        second_path,
        FittedDirection(
            direction=fitted.direction.clone(),
            max_activation=fitted.max_activation,
            metadata=dict(reversed(list(fitted.metadata.items()))),
        ),
    )
    assert second_path.read_bytes() == first_path.read_bytes()


def test_direction_bytes_are_stable_across_fresh_processes(tmp_path: Path) -> None:
    script = """
from pathlib import Path
import torch as t
from steering.eval.axbench_upstream.encdec_baselines.directions import FittedDirection, save_direction

save_direction(
    Path(__import__('sys').argv[1]),
    FittedDirection(
        direction=t.tensor([0.25, -0.5, 0.75], dtype=t.float32),
        max_activation=3.125,
        metadata={'method': 'probe', 'seed': 17, 'nested': {'steps': 48}},
    ),
)
"""
    first = tmp_path / "process-1.safetensors"
    second = tmp_path / "process-2.safetensors"

    subprocess.run([sys.executable, "-c", script, str(first)], check=True)
    subprocess.run([sys.executable, "-c", script, str(second)], check=True)

    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    ("metadata", "tensors", "expected_suffix"),
    [
        (
            {
                "format": "wrong-format",
                "max_activation": "1.0",
                "metadata_json": "{}",
            },
            {"direction": t.ones(2)},
            "invalid or missing direction artifact format",
        ),
        (
            {"format": "axbench-encdec-direction-v1", "metadata_json": "{}"},
            {"direction": t.ones(2)},
            "missing max_activation metadata",
        ),
        (
            {"format": "axbench-encdec-direction-v1", "max_activation": "1.0"},
            {"direction": t.ones(2)},
            "missing metadata_json metadata",
        ),
        (
            {
                "format": "axbench-encdec-direction-v1",
                "max_activation": "1.0",
                "metadata_json": "{}",
            },
            {"direction": t.ones(2), "extra": t.zeros(1)},
            "direction artifact must contain only 'direction'",
        ),
    ],
    ids=("wrong-format", "missing-max-activation", "missing-metadata-json", "extra-tensor"),
)
def test_load_direction_rejects_corrupted_artifact_contract(
    tmp_path: Path,
    metadata: dict[str, str],
    tensors: dict[str, t.Tensor],
    expected_suffix: str,
) -> None:
    path = tmp_path / f"{expected_suffix.replace(' ', '_')}.safetensors"
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(
        ValueError,
        match=_exact(f"{path}: {expected_suffix}"),
    ):
        load_direction(path)


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (t.ones(2, 2, dtype=t.float32), "finite, nonempty, contiguous CPU float32 vector"),
        (t.ones(2, dtype=t.float64), "finite, nonempty, contiguous CPU float32 vector"),
        (t.tensor([1.0, float("nan")]), "finite, nonempty, contiguous CPU float32 vector"),
        (t.empty(0, dtype=t.float32), "finite, nonempty, contiguous CPU float32 vector"),
    ],
)
def test_load_direction_rejects_invalid_tensor_semantics(
    tmp_path: Path,
    tensor: t.Tensor,
    message: str,
) -> None:
    path = tmp_path / "direction.safetensors"
    save_file(
        {"direction": tensor},
        path,
        metadata={
            "format": "axbench-encdec-direction-v1",
            "max_activation": "1.0",
            "metadata_json": "{}",
        },
    )

    with pytest.raises(ValueError, match=message):
        load_direction(path)


@pytest.mark.parametrize("maximum", ["nan", "inf", "-1.0", "not-a-number"])
def test_load_direction_rejects_invalid_max_activation(
    tmp_path: Path,
    maximum: str,
) -> None:
    path = tmp_path / "direction.safetensors"
    save_file(
        {"direction": t.ones(2)},
        path,
        metadata={
            "format": "axbench-encdec-direction-v1",
            "max_activation": maximum,
            "metadata_json": "{}",
        },
    )

    with pytest.raises(ValueError, match="max_activation"):
        load_direction(path)


@pytest.mark.parametrize(
    "metadata_json",
    ["[]", "NaN", '{"duplicate":1,"duplicate":2}', '{"value":NaN}'],
)
def test_load_direction_rejects_nonobject_or_nonstrict_fit_metadata(
    tmp_path: Path,
    metadata_json: str,
) -> None:
    path = tmp_path / "direction.safetensors"
    save_file(
        {"direction": t.ones(2)},
        path,
        metadata={
            "format": "axbench-encdec-direction-v1",
            "max_activation": "1.0",
            "metadata_json": metadata_json,
        },
    )

    with pytest.raises(ValueError, match="metadata_json"):
        load_direction(path)


def test_load_direction_rejects_unexpected_metadata_keys(tmp_path: Path) -> None:
    path = tmp_path / "direction.safetensors"
    save_file(
        {"direction": t.ones(2)},
        path,
        metadata={
            "format": "axbench-encdec-direction-v1",
            "max_activation": "1.0",
            "metadata_json": "{}",
            "unexpected": "value",
        },
    )

    with pytest.raises(ValueError, match="unexpected direction metadata keys"):
        load_direction(path)


@pytest.mark.parametrize(
    ("positive", "negative", "expected"),
    [
        ([], [t.ones(1, 2)], "validation fixture: positive activations must not be empty"),
        ([t.ones(1, 2)], [], "validation fixture: negative activations must not be empty"),
        (
            [t.ones(1, 2)],
            [t.ones(1, 3)],
            "validation fixture: negative row 0 width 3 does not match width 2",
        ),
        (
            [t.tensor([[1.0, float("nan")]])],
            [t.ones(1, 2)],
            "validation fixture: positive row 0 contains non-finite activations",
        ),
    ],
)
@pytest.mark.parametrize("fit_name", ["diffmean", "probe"])
def test_fit_input_validation_names_surface(
    positive: list[t.Tensor],
    negative: list[t.Tensor],
    expected: str,
    fit_name: str,
) -> None:
    with pytest.raises(ValueError, match=rf"^{expected}$"):
        if fit_name == "diffmean":
            fit_diffmean_direction(
                positive, negative, intervention_layer=9, surface="validation fixture"
            )
        else:
            fit_probe_direction(
                positive,
                negative,
                seed=1,
                concept_id=2,
                intervention_layer=9,
                surface="validation fixture",
            )


def test_fit_input_validation_rejects_non_matrix_and_empty_token_rows() -> None:
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        fit_diffmean_direction(
            [t.ones(2)],
            [t.ones(1, 2)],
            intervention_layer=9,
            surface="rank validation",
        )
    with pytest.raises(ValueError, match="has no token activations"):
        fit_probe_direction(
            [t.empty(0, 2)],
            [t.ones(1, 2)],
            seed=1,
            concept_id=2,
            intervention_layer=9,
            surface="empty-row validation",
        )


def test_fit_input_validation_rejects_zero_width_row() -> None:
    with pytest.raises(
        ValueError,
        match=r"^zero-width validation: positive row 0 has zero width$",
    ):
        fit_diffmean_direction(
            [t.empty(1, 0)],
            [t.empty(1, 0)],
            intervention_layer=9,
            surface="zero-width validation",
        )
