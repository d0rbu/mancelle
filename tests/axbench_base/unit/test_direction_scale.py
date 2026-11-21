"""Unit tests for the per-direction max-activation contract (#1228)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from safetensors.torch import save, save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    STEERING_SCALE_MARKER_NAME,
    DirectionScaleRequest,
    compute_direction_max_activation,
    compute_signed_direction_max_activation,
    direction_scale_identity,
    encode_max_activation,
    enforce_steering_scale_identity,
    normalize_direction,
    parse_max_activation,
    preflight_direction_scales,
    require_direction_max_activation,
)
from steering.eval.axbench_base.types import AxBenchBaseError

_SURFACE = "test concept 0 mode=positive"


def _rows_and_unit_direction():
    # direction along axis 0; rows (T,C) with a known peak token.
    d = t.tensor([1.0, 0.0, 0.0, 0.0])
    pos = t.tensor([[3.0, 9.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    neg = t.tensor([[-2.0, 0.0, 5.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    return [pos, neg], d


class TestComputeExactness:
    def test_signed_token_level_max_over_pos_and_neg_rows(self):
        rows, d = _rows_and_unit_direction()
        assert compute_direction_max_activation(rows, d, surface=_SURFACE) == 3.0

    def test_max_attained_on_a_negative_row_is_honored(self):
        # peak lives in the second (neg) row — pos+neg population is real.
        d = t.tensor([1.0, 0.0])
        rows = [t.tensor([[1.0, 0.0]]), t.tensor([[7.0, 0.0]])]
        assert compute_direction_max_activation(rows, d, surface=_SURFACE) == 7.0

    def test_token_level_not_row_mean(self):
        # row mean projection is 0.5*(10-8)=1.0 but the token peak is 10.
        d = t.tensor([1.0, 0.0])
        rows = [t.tensor([[10.0, 0.0], [-8.0, 0.0]])]
        assert compute_direction_max_activation(rows, d, surface=_SURFACE) == 10.0

    def test_signed_no_abs(self):
        # all projections negative except one small positive: max is +0.25,
        # never |-9|.
        d = t.tensor([1.0, 0.0])
        rows = [t.tensor([[-9.0, 0.0], [0.25, 0.0]])]
        assert compute_direction_max_activation(rows, d, surface=_SURFACE) == 0.25


class TestEdgeRules:
    def test_zero_direction_returns_zero(self):
        rows = [t.tensor([[3.0, 4.0]])]
        assert compute_direction_max_activation(rows, t.zeros(2), surface=_SURFACE) == 0.0

    def test_nonzero_direction_nonpositive_max_raises(self):
        d = t.tensor([1.0, 0.0])
        rows = [t.tensor([[-1.0, 0.0], [-0.5, 0.0]])]
        with pytest.raises(AxBenchBaseError, match="non-positive max_activation"):
            compute_direction_max_activation(rows, d, surface=_SURFACE)

    @pytest.mark.parametrize("bad_value", [float("nan"), float("-inf"), float("inf")])
    @pytest.mark.parametrize("bad_row_position", [0, 1, 2])
    def test_non_finite_projection_raises_regardless_of_row_order(
        self, bad_value: float, bad_row_position: int
    ):
        # Python's max(finite, nan) keeps the finite value, so a non-finite
        # projection in a LATER row was silently swallowed pre-review-fix.
        # Every (value, position) combination must raise identically.
        d = t.tensor([1.0, 0.0])
        rows = [t.tensor([[1.0, 0.0]]) for _ in range(3)]
        rows[bad_row_position] = t.tensor([[bad_value, 0.0]])
        with pytest.raises(AxBenchBaseError, match="non-finite projection"):
            compute_direction_max_activation(rows, d, surface=_SURFACE)

    def test_direction_wrong_ndim_raises(self):
        with pytest.raises(AxBenchBaseError, match="non-empty 1-D"):
            compute_direction_max_activation([t.ones(2, 2)], t.ones(2, 2), surface=_SURFACE)

    def test_direction_empty_raises(self):
        with pytest.raises(AxBenchBaseError, match="non-empty 1-D"):
            compute_direction_max_activation([t.ones(2, 0)], t.zeros(0), surface=_SURFACE)

    def test_direction_non_finite_raises(self):
        with pytest.raises(AxBenchBaseError, match="contains non-finite"):
            compute_direction_max_activation(
                [t.ones(2, 2)], t.tensor([float("nan"), 1.0]), surface=_SURFACE
            )

    def test_empty_rows_raises(self):
        with pytest.raises(AxBenchBaseError, match="no fit rows"):
            compute_direction_max_activation([], t.tensor([1.0]), surface=_SURFACE)

    def test_row_width_mismatch_raises(self):
        with pytest.raises(AxBenchBaseError, match="does not match direction width"):
            compute_direction_max_activation([t.ones(2, 3)], t.tensor([1.0, 0.0]), surface=_SURFACE)

    def test_zero_token_row_raises(self):
        with pytest.raises(AxBenchBaseError, match="no continuation tokens"):
            compute_direction_max_activation([t.ones(0, 2)], t.tensor([1.0, 0.0]), surface=_SURFACE)


class TestEncode:
    @pytest.mark.parametrize("value", [3048.0, 1e-8, 0.0, 1.0000001, 2.5e-45])
    def test_repr_round_trip_exact(self, value):
        assert float(encode_max_activation(value)) == value

    def test_float32_item_round_trips(self):
        v = float(t.tensor([3.14159], dtype=t.float32).max().item())
        assert float(encode_max_activation(v)) == v

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(AxBenchBaseError, match="non-finite"):
            encode_max_activation(bad)


def test_metadata_key_is_stable():
    assert MAX_ACTIVATION_METADATA_KEY == "max_activation"


def _metadata(
    *,
    concept_id: int = 7,
    mode: str = "positive",
    target_model: str = "decoder_only",
    layer: int = 6,
    value: str = "4.5",
) -> dict[str, str]:
    return {
        "concept_id": str(concept_id),
        "mode": mode,
        "target_model": target_model,
        "layer": str(layer),
        MAX_ACTIVATION_METADATA_KEY: value,
    }


def _require(
    metadata: dict[str, str] | None,
    *,
    direction: t.Tensor | None = None,
) -> float:
    return require_direction_max_activation(
        metadata,
        direction=t.tensor([1.0, 0.0]) if direction is None else direction,
        expected_concept_id=7,
        expected_mode="positive",
        expected_target_model="decoder_only",
        expected_layer=6,
        surface="test direction artifact",
    )


class TestParseAndRequire:
    def test_parse_round_trips_encoded_value(self):
        metadata = _metadata(value=encode_max_activation(1.234567890123))
        assert parse_max_activation(metadata, surface="test artifact") == float(
            metadata[MAX_ACTIVATION_METADATA_KEY]
        )

    @pytest.mark.parametrize("metadata", [None, {}, {"other": "1"}])
    def test_e3_missing_field_names_issue_and_remedy(self, metadata):
        with pytest.raises(
            AxBenchBaseError,
            match=r"missing 'max_activation'.*issue #1228.*re-run stage 3a/3b",
        ):
            parse_max_activation(metadata, surface="test artifact")

    @pytest.mark.parametrize("value", ["wat", "nan", "inf", "-inf"])
    def test_e4_invalid_value_raises(self, value):
        with pytest.raises(AxBenchBaseError, match="invalid max_activation"):
            parse_max_activation(_metadata(value=value), surface="test artifact")

    def test_positive_value_for_nonzero_direction_passes(self):
        assert _require(_metadata(value="4.5")) == 4.5

    def test_zero_value_for_zero_direction_passes(self):
        assert _require(_metadata(value="0.0"), direction=t.zeros(2)) == 0.0

    def test_e5_negative_value_raises(self):
        with pytest.raises(AxBenchBaseError, match="negative max_activation"):
            _require(_metadata(value="-0.5"))

    @pytest.mark.parametrize(
        ("direction", "value"),
        [(t.tensor([1.0, 0.0]), "0.0"), (t.zeros(2), "1.0")],
    )
    def test_e6_zero_direction_mismatch_raises(self, direction, value):
        with pytest.raises(AxBenchBaseError, match="zero-direction max_activation mismatch"):
            _require(_metadata(value=value), direction=direction)

    @pytest.mark.parametrize(
        ("field", "override"),
        [
            ("concept_id", 8),
            ("mode", "negative"),
            ("target_model", "encdec"),
            ("layer", 9),
        ],
    )
    def test_e7_binding_mismatch_names_field(self, field, override):
        kwargs = {field: override}
        with pytest.raises(AxBenchBaseError, match=rf"direction metadata mismatch.*{field}"):
            _require(_metadata(**kwargs))

    def test_error_class_is_site_local(self):
        class LocalError(Exception):
            pass

        with pytest.raises(LocalError, match="negative max_activation"):
            require_direction_max_activation(
                _metadata(value="-1.0"),
                direction=t.tensor([1.0, 0.0]),
                expected_concept_id=7,
                expected_mode="positive",
                expected_target_model="decoder_only",
                expected_layer=6,
                surface="local surface",
                error_cls=LocalError,
            )


class TestNormalizeDirection:
    def test_normalizes_nonzero_direction(self):
        result = normalize_direction(t.tensor([3.0, 4.0]))
        assert t.equal(result, t.tensor([0.6, 0.8]))

    def test_zero_direction_is_preserved(self):
        direction = t.zeros(3)
        assert normalize_direction(direction) is direction

    def test_non_finite_direction_uses_requested_error_class(self):
        class LocalError(Exception):
            pass

        with pytest.raises(LocalError, match="non-finite"):
            normalize_direction(t.tensor([float("nan")]), error_cls=LocalError)


class TestDirectionScaleIdentity:
    def test_deterministic_and_order_insensitive(self):
        left = {
            ("diffmean", "decoder_only", "positive", 2): 4.0,
            ("linear_probe", "encdec", "negative", 1): 3.0,
        }
        right = dict(reversed(tuple(left.items())))
        assert direction_scale_identity(left) == direction_scale_identity(right)
        assert direction_scale_identity(left).startswith("dirmaxact:")

    def test_sensitive_to_value_change(self):
        key = ("diffmean", "decoder_only", "positive", 2)
        assert direction_scale_identity({key: 4.0}) != direction_scale_identity({key: 4.5})

    def test_sensitive_to_consumed_set_change(self):
        key = ("diffmean", "decoder_only", "positive", 2)
        extra = ("diffmean", "decoder_only", "negative", 2)
        assert direction_scale_identity({key: 4.0}) != direction_scale_identity(
            {key: 4.0, extra: 4.0}
        )


def _write_direction(
    path: Path,
    *,
    metadata: dict[str, str],
    direction: t.Tensor | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.tensor([1.0, 0.0]) if direction is None else direction},
        str(path),
        metadata=metadata,
    )
    return path


class TestPreflight:
    def test_loads_path_and_payload_and_deduplicates_keys(self, tmp_path):
        path = _write_direction(tmp_path / "path.safetensors", metadata=_metadata())
        payload_metadata = _metadata(
            concept_id=8, mode="negative", value=encode_max_activation(9.0)
        )
        payload = save({"direction": t.tensor([0.0, 1.0])}, payload_metadata)
        result = preflight_direction_scales(
            [
                DirectionScaleRequest(
                    family="diffmean",
                    target_model="decoder_only",
                    mode="positive",
                    concept_id=7,
                    path=path,
                ),
                DirectionScaleRequest(
                    family="diffmean",
                    target_model="decoder_only",
                    mode="positive",
                    concept_id=7,
                    path=path,
                ),
                DirectionScaleRequest(
                    family="linear_probe",
                    target_model="decoder_only",
                    mode="negative",
                    concept_id=8,
                    payload=payload,
                ),
            ],
            expected_layer=6,
            stage_label="test preflight",
        )
        assert result.values == {
            ("diffmean", "decoder_only", "positive", 7): 4.5,
            ("linear_probe", "decoder_only", "negative", 8): 9.0,
        }
        assert result.identity == direction_scale_identity(result.values)

    def test_aggregates_and_names_multiple_bad_artifacts(self, tmp_path):
        first = _write_direction(tmp_path / "first.safetensors", metadata=_metadata(value="-1.0"))
        second = _write_direction(
            tmp_path / "second.safetensors",
            metadata={
                key: value
                for key, value in _metadata().items()
                if key != MAX_ACTIVATION_METADATA_KEY
            },
        )
        requests = [
            DirectionScaleRequest(
                family="diffmean",
                target_model="decoder_only",
                mode="positive",
                concept_id=7,
                path=first,
            ),
            DirectionScaleRequest(
                family="linear_probe",
                target_model="decoder_only",
                mode="positive",
                concept_id=7,
                path=second,
            ),
        ]
        with pytest.raises(AxBenchBaseError, match="direction-scale preflight") as exc:
            preflight_direction_scales(requests, expected_layer=6, stage_label="test preflight")
        message = str(exc.value)
        assert "first.safetensors" in message
        assert "second.safetensors" in message
        assert "negative max_activation" in message
        assert "missing 'max_activation'" in message


def _marker(tmp_path: Path) -> Path:
    return tmp_path / "run" / STEERING_SCALE_MARKER_NAME


class TestEnforceDirectionScaleIdentity:
    def test_fresh_run_stamps_marker(self, tmp_path):
        marker = _marker(tmp_path)
        enforce_steering_scale_identity(marker, identity="dirmaxact:abc", resume_active=False)
        assert marker.read_text() == "dirmaxact:abc"

    def test_matching_marker_is_noop_even_on_resume(self, tmp_path):
        marker = _marker(tmp_path)
        enforce_steering_scale_identity(marker, identity="dirmaxact:abc", resume_active=False)
        enforce_steering_scale_identity(marker, identity="dirmaxact:abc", resume_active=True)
        assert marker.read_text() == "dirmaxact:abc"

    def test_e8_mismatched_direction_identity_raises(self, tmp_path):
        marker = _marker(tmp_path)
        enforce_steering_scale_identity(marker, identity="dirmaxact:abc", resume_active=False)
        with pytest.raises(AxBenchBaseError, match="DIFFERENT steering-scale regime"):
            enforce_steering_scale_identity(marker, identity="dirmaxact:xyz", resume_active=True)

    def test_e8_old_maxact_marker_raises(self, tmp_path):
        marker = _marker(tmp_path)
        marker.parent.mkdir(parents=True)
        marker.write_text("maxact:0123456789abcdef")
        with pytest.raises(AxBenchBaseError, match=r"pre-#1228 SAE max_act regime.*fresh run_tag"):
            enforce_steering_scale_identity(
                marker, identity="dirmaxact:0123456789abcdef", resume_active=True
            )

    def test_e9_resume_without_marker_raises(self, tmp_path):
        marker = _marker(tmp_path)
        marker.parent.mkdir(parents=True)
        with pytest.raises(AxBenchBaseError, match="no steering-scale identity marker"):
            enforce_steering_scale_identity(marker, identity="dirmaxact:abc", resume_active=True)

    def test_fresh_run_without_marker_dir_creates_it(self, tmp_path):
        marker = tmp_path / "fresh" / "nested" / STEERING_SCALE_MARKER_NAME
        enforce_steering_scale_identity(marker, identity="dirmaxact:abc", resume_active=False)
        assert marker.read_text() == "dirmaxact:abc"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_signed_calibration_rejects_nonfinite_later_rows(bad: float) -> None:
    with pytest.raises(AxBenchBaseError, match="non-finite projection"):
        compute_signed_direction_max_activation(
            [t.ones(1, 2), t.tensor([[bad, 0.0]])], t.tensor([1.0, 0.0]), surface="signed"
        )


def test_signed_calibration_rejects_projection_overflow() -> None:
    maximum = t.finfo(t.float32).max
    with pytest.raises(AxBenchBaseError, match="non-finite projection"):
        compute_signed_direction_max_activation(
            [t.full((1, 3), maximum)], t.ones(3) / 3**0.5, surface="overflow"
        )


@pytest.mark.parametrize("rows", [[], [t.empty(0, 2)], [t.ones(1, 3)]])
def test_signed_calibration_rejects_invalid_population(rows: list[t.Tensor]) -> None:
    with pytest.raises(AxBenchBaseError):
        compute_signed_direction_max_activation(rows, t.tensor([1.0, 0.0]), surface="invalid")


def test_scale_guard_preserves_fp32_zero_conversion() -> None:
    direction = t.tensor([1e-100], dtype=t.float64)
    assert compute_direction_max_activation([t.ones(1, 1)], direction, surface="underflow") == 0.0
