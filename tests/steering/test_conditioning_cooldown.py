"""Exact schedule, masking, RNG, and resume contracts for conditioning cooldown."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from scripts.train.conditional_encdec_train import (
    _checkpoint_user_config,
    _resolved_user_config,
)
from steering.conditional.encoder_decoder.conditioning_cooldown import (
    ConditioningCooldown,
    apply_conditioning_cooldown,
    parse_conditioning_cooldown,
    resolve_conditioning_cooldown,
    validate_conditioning_cooldown_resume,
)


def _schedule(raw: dict[str, Any] | None = None, **geometry: Any) -> ConditioningCooldown:
    kwargs = {
        "num_iterations": 100,
        "total_batch_size": 1_000_000_000,
        "seed": 42,
        "world_size": 1,
        "device_batch_size": 1,
        "max_seq_len": 1,
        "device_type": "cpu",
        **geometry,
    }
    result = resolve_conditioning_cooldown(
        parse_conditioning_cooldown({"enabled": True, **(raw or {})}), **kwargs
    )
    assert result is not None
    return result


def _mask(mask: t.Tensor, docs: t.Tensor, **overrides: Any) -> t.Tensor:
    kwargs: dict[str, Any] = {
        "fraction": 0.5,
        "granularity": "segment",
        "seed": 42,
        "step": 90,
        "rank": 0,
        "micro_step": 0,
        **overrides,
    }
    return apply_conditioning_cooldown(mask, docs, **kwargs)


def _checkpoint(raw: dict[str, Any] | None = None) -> tuple[ConditioningCooldown, dict]:
    cfg = {"enabled": True, **(raw or {})}
    schedule = _schedule(cfg)
    return schedule, {
        "step": 90,
        "user_config": {"conditioning_cooldown": cfg},
        "conditioning_cooldown": schedule.metadata_dict(),
    }


def test_default_window_has_exact_endpoints_and_twenty_billion_tokens() -> None:
    schedule = _schedule()
    assert (schedule.start_step, schedule.end_step) == (80, 99)
    assert (
        schedule.end_step - schedule.start_step + 1
    ) * schedule.total_batch_size == 20_000_000_000
    assert schedule.fraction_at(0) == schedule.fraction_at(79) == schedule.fraction_at(80) == 1.0
    assert schedule.fraction_at(99) == schedule.fraction_at(100) == 0.5
    assert schedule.fraction_at(90) == 1 - 0.5 * (10 / 19)


def test_ceil_rounding_covers_at_least_the_requested_tokens() -> None:
    schedule = _schedule(total_batch_size=3_000_000_000)
    assert (schedule.start_step, schedule.end_step) == (93, 99)
    covered = (schedule.end_step - schedule.start_step + 1) * schedule.total_batch_size
    assert covered == 21_000_000_000
    assert covered - 20_000_000_000 < schedule.total_batch_size


def test_token_ceil_does_not_round_through_float() -> None:
    tokens = 2**60 + 1
    schedule = _schedule({"cooldown_tokens": tokens}, total_batch_size=2**59)
    assert schedule.start_step == 97  # three updates, never float-rounded to two


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"start_step": 40, "end_step": 60}, (40, 60)),
        ({"start_step": 40}, (40, 99)),
        ({"end_step": 60}, (41, 60)),
        ({"start_step": 0, "end_step": 1}, (0, 1)),
    ],
)
def test_explicit_axes_resolve_without_a_new_time_origin(raw: dict, expected: tuple) -> None:
    schedule = _schedule(raw)
    assert (schedule.start_step, schedule.end_step) == expected


@pytest.mark.parametrize(("start", "end"), [(1.0, 0.0), (0.25, 0.75), (0.4, 0.4), (0.0, 0.0)])
def test_custom_y_axes_and_clamped_interval(start: float, end: float) -> None:
    schedule = _schedule(
        {"start_step": 10, "end_step": 30, "start_fraction": start, "end_fraction": end}
    )
    assert schedule.fraction_at(0) == schedule.fraction_at(10) == start
    assert schedule.fraction_at(20) == (start + end) / 2
    assert schedule.fraction_at(30) == schedule.fraction_at(99) == end


@pytest.mark.parametrize("raw", [None, {}, {"enabled": False}])
def test_disabled_config_is_absent(raw: object) -> None:
    assert parse_conditioning_cooldown(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        True,
        1,
        "linear",
        [],
        {"enabled": 1},
        {"enabled": "true"},
        {"bogus": 0},
        {"granularity": "row"},
        {"granularity": None},
        {"start_fraction": -0.01},
        {"end_fraction": 1.01},
        {"end_fraction": True},
        {"start_fraction": "1.0"},
        {"start_fraction": float("nan")},
        {"end_fraction": float("inf")},
        {"start_step": -1},
        {"start_step": 0.0},
        {"end_step": True},
        {"cooldown_tokens": 0},
        {"cooldown_tokens": 1.0},
        {"cooldown_tokens": False},
        {"start_step": 5, "end_step": 5},
        {"start_step": 6, "end_step": 5},
    ],
)
def test_malformed_config_is_never_silently_coerced(raw: object) -> None:
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        parse_conditioning_cooldown(raw)


@pytest.mark.parametrize(
    ("raw", "geometry"),
    [
        ({}, {"num_iterations": 19}),
        ({}, {"num_iterations": 1}),
        ({"cooldown_tokens": 1}, {}),
        ({"start_step": 98, "end_step": 100}, {}),
        ({"start_step": 99}, {}),
        ({}, {"num_iterations": True}),
        ({}, {"world_size": 0}),
        ({}, {"device_batch_size": 1.0}),
        ({}, {"max_seq_len": -1}),
        ({}, {"total_batch_size": 0}),
        ({}, {"seed": True}),
        ({}, {"device_type": "bogus"}),
        ({}, {"world_size": 3}),
    ],
)
def test_invalid_resolved_geometry_fails(raw: dict, geometry: dict) -> None:
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        _schedule(raw, **geometry)


def test_document_draw_is_shared_within_doc_and_independent_across_rows() -> None:
    docs = t.tensor([[0, 0, 2, 2, 4, -1]] * 128)
    mask = t.ones(128, 6, 3, dtype=t.bool)
    mask[:, -1] = False
    result = _mask(mask, docs, granularity="document")
    assert t.equal(result[:, 0], result[:, 1])
    assert t.equal(result[:, 2], result[:, 3])
    assert (result[:, 0] != result[:, 2]).any()
    assert (result[0] != result[1:]).any()
    assert not result[:, -1].any()
    assert result.any() and (~result[:, :5]).any()


def test_segment_draw_is_independent_within_the_same_document() -> None:
    mask = t.ones(128, 6, 3, dtype=t.bool)
    result = _mask(mask, t.zeros(128, 6, dtype=t.int64))
    assert (result[:, 0] != result[:, 1]).any()
    assert t.equal(result[..., 0], result[..., 1])
    assert t.equal(result[..., 1], result[..., 2])


@pytest.mark.parametrize("granularity", ["segment", "document"])
@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.9])
def test_no_real_junctions_preserve_exact_scalar_masks(granularity: str, fraction: float) -> None:
    original = t.ones(16, 7, 3, dtype=t.bool)
    original[:, -1] = False
    docs = t.tensor([[0, 0, 1, 1, 2, 2, -1]] * 16)
    flags = t.zeros(16, 7, dtype=t.bool)
    flags[:, -1] = True  # padding cannot count as a protected unit
    expected = _mask(original, docs, fraction=fraction, granularity=granularity)
    actual = _mask(original, docs, fraction=fraction, granularity=granularity, splice_mask=flags)
    assert t.equal(actual, expected)


@pytest.mark.parametrize("granularity", ["segment", "document"])
@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.9])
def test_only_protected_units_preserve_exact_scalar_masks(
    granularity: str, fraction: float
) -> None:
    original = t.ones(16, 7, 3, dtype=t.bool)
    original[:, -1] = False
    docs = t.tensor([[0, 0, 1, 1, 2, 2, -1]] * 16)
    flags = t.ones(16, 7, dtype=t.bool)
    expected = _mask(original, docs, fraction=fraction, granularity=granularity)
    actual = _mask(original, docs, fraction=fraction, granularity=granularity, splice_mask=flags)
    assert t.equal(actual, expected)


@pytest.mark.parametrize(
    ("fraction", "expected_unprotected", "expected_protected"),
    [(0.75, 2 / 3, 1.0), (0.25, 0.0, 1.0), (0.10, 0.0, 0.4)],
)
def test_segment_priority_preserves_expected_overall_retention(
    fraction: float, expected_unprotected: float, expected_protected: float
) -> None:
    original = t.ones(5000, 4, 2, dtype=t.bool)
    docs = t.arange(4).expand(5000, -1)
    flags = t.zeros(5000, 4, dtype=t.bool)
    flags[:, -1] = True
    result = _mask(original, docs, fraction=fraction, splice_mask=flags)
    actual_unprotected = result[:, :3, 0].float().mean().item()
    actual_protected = result[:, 3, 0].float().mean().item()
    assert actual_unprotected == pytest.approx(expected_unprotected, abs=0.015)
    assert actual_protected == pytest.approx(expected_protected, abs=0.015)
    assert result[..., 0].float().mean().item() == pytest.approx(fraction, abs=0.015)
    if expected_protected == 1.0:
        assert result[:, 3].all()
    if expected_unprotected == 0.0:
        assert not result[:, :3].any()


def test_document_priority_protects_the_entire_document_and_counts_it_once() -> None:
    original = t.ones(5000, 6, 2, dtype=t.bool)
    docs = t.tensor([[0, 0, 0, 1, 2, 2]] * 5000)
    flags = t.zeros(5000, 6, dtype=t.bool)
    flags[:, 1] = True  # only the middle of document 0 is a junction
    at_boundary = _mask(original, docs, fraction=1 / 3, granularity="document", splice_mask=flags)
    assert at_boundary[:, :3].all()
    assert not at_boundary[:, 3:].any()
    partial = _mask(original, docs, fraction=0.75, granularity="document", splice_mask=flags)
    assert partial[:, :3].all()
    assert t.equal(partial[:, 4], partial[:, 5])
    # Three equal-probability document units, independent of their 3/1/2 lengths.
    assert partial[:, [3, 4], 0].float().mean().item() == pytest.approx(0.625, abs=0.015)
    low = _mask(original, docs, fraction=0.1, granularity="document", splice_mask=flags)
    assert not low[:, 3:].any()
    assert t.equal(low[:, 0], low[:, 1]) and t.equal(low[:, 1], low[:, 2])


def test_priority_counts_empty_real_segments_but_never_padding() -> None:
    original = t.ones(1000, 4, 2, dtype=t.bool)
    original[:, 1] = False  # real but featureless
    original[:, 3] = False  # padding
    docs = t.tensor([[0, 1, 2, -1]] * 1000)
    flags = t.tensor([[False, False, True, True]] * 1000)
    result = _mask(original, docs, fraction=1 / 3, splice_mask=flags)
    assert not result[:, :2].any()
    assert result[:, 2].all()
    assert not result[:, 3].any()


@pytest.mark.parametrize("flags", [t.zeros(2, 3), t.zeros(2, 4, dtype=t.bool)])
def test_priority_mask_schema_is_checked_at_full_retention(flags: t.Tensor) -> None:
    with pytest.raises(ValueError, match="splice_mask"):
        _mask(
            t.ones(2, 3, 2, dtype=t.bool),
            t.zeros(2, 3, dtype=t.int64),
            fraction=1.0,
            splice_mask=flags,
        )


def test_segment_mask_is_exactly_the_existing_bernoulli_algorithm() -> None:
    original = t.ones(8, 17, 5, dtype=t.bool)
    original[:, 3, 1:] = False
    docs = t.arange(17).expand(8, -1)
    seed = int.from_bytes(
        hashlib.sha256(b"conditioning_cooldown_v1:42:90:0:0").digest()[:8], "little"
    )
    generator = t.Generator().manual_seed(seed)
    # Frozen legacy masking operation from dataloader.py, without its unrelated
    # fallback that re-enables a feature in an entirely empty row.
    expected = original.clone()
    drop_mask = t.rand(8, 17, generator=generator, dtype=t.float32) < 0.5
    expected[drop_mask] = False
    actual = _mask(original, docs)
    assert t.equal(actual, expected)
    assert actual.data_ptr() != original.data_ptr()
    assert original[:, 0].all()  # source batch is unmodified


@pytest.mark.parametrize("granularity", ["segment", "document"])
def test_zero_retention_does_not_restore_legacy_sentinel(granularity: str) -> None:
    original = t.zeros(4, 5, 2, dtype=t.bool)
    original[:, 0, 0] = True
    result = _mask(original, t.zeros(4, 5, dtype=t.int64), fraction=0.0, granularity=granularity)
    assert not result.any()
    assert original[:, 0, 0].all()


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_empty_conditioning_stays_empty(fraction: float) -> None:
    original = t.zeros(3, 5, 2, dtype=t.bool)
    assert not _mask(original, t.zeros(3, 5, dtype=t.int64), fraction=fraction).any()


def test_full_retention_is_identity_and_does_not_construct_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = t.ones(3, 5, 2, dtype=t.bool)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("full retention must not construct a generator")

    monkeypatch.setattr(t, "Generator", forbidden)
    assert _mask(original, t.zeros(3, 5, dtype=t.int64), fraction=1.0) is original


def test_mask_replay_ignores_global_rng_draws_and_default_dtype() -> None:
    original = t.ones(8, 17, 2, dtype=t.bool)
    docs = t.arange(17).expand(8, -1)
    before = t.get_rng_state().clone()
    expected = _mask(original, docs)
    assert t.equal(before, t.get_rng_state())
    t.rand(111)
    after_global_draw = t.get_rng_state().clone()
    original_dtype = t.get_default_dtype()
    try:
        t.set_default_dtype(t.float64)
        actual = _mask(original, docs)
    finally:
        t.set_default_dtype(original_dtype)
    assert t.equal(actual, expected)
    assert t.equal(after_global_draw, t.get_rng_state())


@pytest.mark.parametrize("changed", [{"seed": 43}, {"step": 91}, {"rank": 1}, {"micro_step": 1}])
def test_each_coordinate_selects_a_distinct_random_stream(changed: dict) -> None:
    original = t.ones(8, 17, 2, dtype=t.bool)
    docs = t.arange(17).expand(8, -1)
    assert not t.equal(_mask(original, docs), _mask(original, docs, **changed))


@pytest.mark.parametrize("bad", [-2, 3])
def test_document_ids_cannot_alias_another_draw(bad: int) -> None:
    with pytest.raises(ValueError, match="document IDs"):
        _mask(t.ones(1, 3, 2, dtype=t.bool), t.tensor([[0, 0, bad]]), granularity="document")


@pytest.mark.parametrize("fraction", [-1, 2, True, float("nan"), "0.5"])
def test_apply_rejects_invalid_probability(fraction: object) -> None:
    with pytest.raises(ValueError, match="fraction"):
        _mask(t.ones(1, 2, 2, dtype=t.bool), t.tensor([[0, 0]]), fraction=fraction)


@pytest.mark.parametrize(
    ("mask", "docs"),
    [
        (t.ones(1, 2, 2), t.tensor([[0, 0]])),
        (t.ones(1, 2, dtype=t.bool), t.tensor([[0, 0]])),
        (t.ones(1, 0, 2, dtype=t.bool), t.empty(1, 0, dtype=t.int64)),
        (t.ones(1, 2, 2, dtype=t.bool), t.zeros(1, 2)),
        (t.ones(1, 2, 2, dtype=t.bool), t.zeros(2, 2, dtype=t.int64)),
    ],
)
def test_apply_validates_tensor_contract_even_at_full_retention(
    mask: t.Tensor, docs: t.Tensor
) -> None:
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        _mask(mask, docs, fraction=1.0)


def test_resolved_metadata_json_roundtrip_and_exact_resume() -> None:
    schedule, meta = _checkpoint()
    restored = json.loads(json.dumps(meta))
    assert ConditioningCooldown.from_metadata(restored["conditioning_cooldown"]) == schedule
    validate_conditioning_cooldown_resume(schedule, restored)


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "world_size",
        "device_batch_size",
        "max_seq_len",
        "total_batch_size",
        "num_iterations",
        "start_step",
        "end_step",
        "start_fraction",
        "end_fraction",
        "granularity",
        "device_type",
    ],
)
def test_resume_rejects_every_changed_schedule_or_geometry_field(field: str) -> None:
    schedule, meta = _checkpoint()
    changed = schedule.metadata_dict()
    if field == "granularity":
        changed[field] = "document"
    elif field == "device_type":
        changed[field] = "cuda"
    elif field in ("start_fraction", "end_fraction"):
        changed[field] = 0.25
    elif field in ("world_size", "device_batch_size", "max_seq_len"):
        changed[field] *= 2
    else:
        changed[field] += 1
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        other = ConditioningCooldown.from_metadata(changed)
        validate_conditioning_cooldown_resume(other, meta)


@pytest.mark.parametrize("step", [0, 79, 80])
def test_legacy_checkpoint_can_begin_cooldown_before_or_at_start(step: int) -> None:
    validate_conditioning_cooldown_resume(
        _schedule(), {"step": step, "user_config": {"label_dropout": 0.0}}
    )


def test_legacy_checkpoint_cannot_skip_already_scheduled_updates() -> None:
    with pytest.raises(ValueError, match="after start_step"):
        validate_conditioning_cooldown_resume(_schedule(), {"step": 81})


def test_active_checkpoint_cannot_silently_disable_cooldown() -> None:
    _, meta = _checkpoint()
    with pytest.raises(ValueError, match="preserve"):
        validate_conditioning_cooldown_resume(None, meta)


@pytest.mark.parametrize("missing", ["user_config", "conditioning_cooldown"])
def test_partial_active_checkpoint_is_not_legacy(missing: str) -> None:
    schedule, meta = _checkpoint()
    del meta[missing]
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        validate_conditioning_cooldown_resume(schedule, meta)


def test_saved_raw_config_cannot_disagree_with_resolved_window() -> None:
    schedule, meta = _checkpoint()
    meta["user_config"]["conditioning_cooldown"]["cooldown_tokens"] = 10_000_000_000
    with pytest.raises(ValueError, match="disagrees"):
        validate_conditioning_cooldown_resume(schedule, meta)


@pytest.mark.parametrize(
    "name", ["seed", "device_batch_size", "max_seq_len", "total_batch_size", "num_iterations"]
)
@pytest.mark.parametrize("source", ["user_config", "top_level"])
def test_saved_explicit_geometry_cannot_disagree_with_resolved_record(
    name: str, source: str
) -> None:
    schedule, meta = _checkpoint()
    target = meta["user_config"] if source == "user_config" else meta
    target[name] = getattr(schedule, name) + 1
    with pytest.raises(ValueError, match="disagrees"):
        validate_conditioning_cooldown_resume(schedule, meta)


def test_saved_auto_geometry_uses_validated_resolved_values() -> None:
    schedule, meta = _checkpoint()
    meta["user_config"].update(total_batch_size=-1, num_iterations=-1, device_type="")
    validate_conditioning_cooldown_resume(schedule, meta)


def test_saved_world_size_must_match_dataloader_envelope() -> None:
    schedule, meta = _checkpoint()
    meta["dataloader_state_dict"] = {"ddp_world_size": 2}
    with pytest.raises(ValueError, match="world_size"):
        validate_conditioning_cooldown_resume(schedule, meta)


@pytest.mark.parametrize("step", [100, 101])
def test_resume_requires_remaining_updates(step: int) -> None:
    schedule, meta = _checkpoint()
    meta["step"] = step
    with pytest.raises(ValueError, match="no remaining"):
        validate_conditioning_cooldown_resume(schedule, meta)


@pytest.mark.parametrize(
    "field", ["start_step", "end_step", "num_iterations", "seed", "schema_version"]
)
def test_resolved_metadata_does_not_accept_null_integer_fields(field: str) -> None:
    raw = _schedule().metadata_dict()
    raw[field] = None
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        ConditioningCooldown.from_metadata(raw)


@pytest.mark.parametrize("mutation", ["extra", "missing", "version"])
def test_resolved_metadata_rejects_unknown_schema(mutation: str) -> None:
    raw = _schedule().metadata_dict()
    if mutation == "extra":
        raw["extra"] = 1
    elif mutation == "missing":
        del raw["seed"]
    else:
        raw["schema_version"] = 2
    with pytest.raises(ValueError, match="conditioning_cooldown"):
        ConditioningCooldown.from_metadata(raw)


def test_disabled_metadata_is_exactly_the_historical_config_surface() -> None:
    config_dir = Path(__file__).parents[2] / "configs" / "training"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="conditional_encdec_train")
    old_cfg = copy.deepcopy(cfg)
    with open_dict(old_cfg):
        del old_cfg.conditioning_cooldown
    assert _resolved_user_config(cfg) == _resolved_user_config(old_cfg)
    assert _checkpoint_user_config(cfg) == _checkpoint_user_config(old_cfg)
    assert "conditioning_cooldown" not in _checkpoint_user_config(cfg)
    validate_conditioning_cooldown_resume(None, {"step": 9, "user_config": {}})
