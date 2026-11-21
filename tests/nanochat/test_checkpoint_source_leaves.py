"""Regression tests for leaf-bearing encoder-decoder checkpoint sources."""

import re
from pathlib import Path

import pytest

import nanochat.checkpoint_manager as checkpoint_manager
from nanochat.checkpoint_manager import (
    _SELECTION_LEAF_DIR_RE,
    get_checkpoint_dir,
    load_model,
)
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType

_MODEL_TAG = "golden"
_MASK_LEAF = "bl_0123456789abcdef"
_OTHER_MASK_LEAF = "bl_fedcba9876543210"
_CLUSTER_LEAF = f"fc_{'a' * 64}"
_SELECTION_DIR = "t2048.9"
_CONDITIONED_SEGMENT_DIR = "safety_sft_conditioned_checkpoints"


def _feature_selection() -> ThresholdSelection:
    return ThresholdSelection(ThresholdType.feature_percentile_segment_topk, 2048.9)


def _encdec_selection_root(tmp_path: Path) -> Path:
    return tmp_path / "conditional_encdec_checkpoints" / _MODEL_TAG / "s128" / _SELECTION_DIR


@pytest.mark.parametrize(
    ("source", "expected_suffix"),
    [
        ("base", "base_checkpoints/golden"),
        ("sft", "base_checkpoints/golden/standard_sft_checkpoints"),
        ("safety", "base_checkpoints/golden/safety_sft_checkpoints"),
        ("encdec", "conditional_encdec_checkpoints/golden"),
    ],
)
def test_no_leaf_golden_root_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, expected_suffix: str
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    assert get_checkpoint_dir(source, model_tag=_MODEL_TAG) == (f"{tmp_path}/{expected_suffix}")


@pytest.mark.parametrize(
    ("segment_size", "include_selection", "expected_suffix"),
    [
        (128, False, "conditional_encdec_checkpoints/golden/s128"),
        (None, True, "conditional_encdec_checkpoints/golden/t2048.9"),
        (128, True, "conditional_encdec_checkpoints/golden/s128/t2048.9"),
    ],
)
def test_no_leaf_golden_encdec_parameter_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_size: int | None,
    include_selection: bool,
    expected_suffix: str,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    selection = _feature_selection() if include_selection else None

    assert (
        get_checkpoint_dir(
            "encdec",
            model_tag=_MODEL_TAG,
            segment_size=segment_size,
            feature_selection=selection,
        )
        == f"{tmp_path}/{expected_suffix}"
    )


def test_no_leaf_golden_direct_conditioned_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    expected = _encdec_selection_root(tmp_path) / _CONDITIONED_SEGMENT_DIR
    expected.mkdir(parents=True)
    (expected / "model_000000.pt").touch()

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == (
        f"{tmp_path}/conditional_encdec_checkpoints/golden/s128/"
        f"{_SELECTION_DIR}/{_CONDITIONED_SEGMENT_DIR}"
    )


@pytest.mark.parametrize(
    "leaf",
    [
        _MASK_LEAF,
        "wl_0123456789abcdef",
        _CLUSTER_LEAF,
        "grad_self_mf8",
        "grad_future_half_split_mf2",
        "grad_future_self_full_mf12_fp32",
    ],
)
def test_selection_leaf_pattern_accepts_writer_outputs(leaf: str) -> None:
    assert _SELECTION_LEAF_DIR_RE.fullmatch(leaf) is not None


@pytest.mark.parametrize(
    "leaf",
    [
        "blx_123",
        "bl_0123456789abcde",
        "bl_0123456789ABCDEf",
        f"fc_{'a' * 63}",
        "grad_future_mf8",
        "grad_self_mf0",
        "grad_self_mf08",
        "grad_self_mf8_bf16",
    ],
)
def test_selection_leaf_pattern_rejects_non_writer_names(leaf: str) -> None:
    assert _SELECTION_LEAF_DIR_RE.fullmatch(leaf) is None


def test_resolves_conditioned_segment_through_mask_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    expected = _encdec_selection_root(tmp_path) / _MASK_LEAF / _CONDITIONED_SEGMENT_DIR
    expected.mkdir(parents=True)
    (expected / "model_000000.pt").touch()

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(expected)


def test_resolves_conditioned_segment_through_nested_mask_and_cluster_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    expected = (
        _encdec_selection_root(tmp_path) / _MASK_LEAF / _CLUSTER_LEAF / _CONDITIONED_SEGMENT_DIR
    )
    expected.mkdir(parents=True)
    (expected / "model_000000.pt").touch()

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(expected)


def test_resolves_conditioned_segment_at_maximum_writer_leaf_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    expected = (
        _encdec_selection_root(tmp_path)
        / _MASK_LEAF
        / "grad_future_half_split_mf8"
        / _CLUSTER_LEAF
        / _CONDITIONED_SEGMENT_DIR
    )
    expected.mkdir(parents=True)
    (expected / "model_000000.pt").touch()

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(expected)


def test_reapplies_leaf_discovery_for_nonfinal_encdec_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    expected = (
        _encdec_selection_root(tmp_path)
        / _MASK_LEAF
        / _CONDITIONED_SEGMENT_DIR
        / _CLUSTER_LEAF
        / "standard_rl_checkpoints"
    )
    expected.mkdir(parents=True)
    (expected / "model_000000.pt").touch()

    resolved = get_checkpoint_dir(
        "encdec/safety_cond/rl",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(expected)


@pytest.mark.parametrize("invalid_leaf", ["blx_123", "bl_0123456789abcde"])
def test_does_not_traverse_nonmatching_leaf_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_leaf: str
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    hidden_segment = _encdec_selection_root(tmp_path) / invalid_leaf / _CONDITIONED_SEGMENT_DIR
    hidden_segment.mkdir(parents=True)
    (hidden_segment / "model_000000.pt").touch()
    literal = _encdec_selection_root(tmp_path) / _CONDITIONED_SEGMENT_DIR

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(literal)


def test_ambiguous_sibling_leaves_list_all_candidates_and_disambiguator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    first = _encdec_selection_root(tmp_path) / _MASK_LEAF / _CONDITIONED_SEGMENT_DIR
    second = _encdec_selection_root(tmp_path) / _OTHER_MASK_LEAF / _CONDITIONED_SEGMENT_DIR
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    match = rf"{re.escape(str(first))}.*{re.escape(str(second))}.*checkpoints_dir"

    with pytest.raises(ValueError, match=match):
        get_checkpoint_dir(
            "encdec/safety_cond",
            model_tag=_MODEL_TAG,
            segment_size=128,
            feature_selection=_feature_selection(),
        )


def test_ambiguous_direct_and_nested_leaf_candidates_list_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    first = _encdec_selection_root(tmp_path) / _MASK_LEAF / _CONDITIONED_SEGMENT_DIR
    second = (
        _encdec_selection_root(tmp_path)
        / _MASK_LEAF
        / _CLUSTER_LEAF
        / _CONDITIONED_SEGMENT_DIR
    )
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    match = rf"{re.escape(str(first))}.*{re.escape(str(second))}.*checkpoints_dir"

    with pytest.raises(ValueError, match=match):
        get_checkpoint_dir(
            "encdec/safety_cond",
            model_tag=_MODEL_TAG,
            segment_size=128,
            feature_selection=_feature_selection(),
        )


def test_traverses_symlinked_selection_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root = _encdec_selection_root(tmp_path)
    target = tmp_path / "selection-target" / _MASK_LEAF
    (target / _CONDITIONED_SEGMENT_DIR).mkdir(parents=True)
    root.mkdir(parents=True)
    leaf_link = root / _MASK_LEAF
    leaf_link.symlink_to(target, target_is_directory=True)
    expected = leaf_link / _CONDITIONED_SEGMENT_DIR

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(expected)


def test_non_encdec_root_never_discovers_selection_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    root = tmp_path / "base_checkpoints" / _MODEL_TAG
    (root / _MASK_LEAF / "safety_sft_checkpoints").mkdir(parents=True)
    literal = root / "safety_sft_checkpoints"

    assert get_checkpoint_dir("safety", model_tag=_MODEL_TAG) == str(literal)


def test_zero_leaf_candidates_keep_literal_composed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    literal = _encdec_selection_root(tmp_path) / _CONDITIONED_SEGMENT_DIR

    resolved = get_checkpoint_dir(
        "encdec/safety_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == str(literal)


def test_missing_leaf_segment_fails_downstream_with_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    literal = _encdec_selection_root(tmp_path) / _CONDITIONED_SEGMENT_DIR
    expected = re.escape(f"No checkpoints found in {literal}")

    with pytest.raises(FileNotFoundError, match=expected):
        load_model(
            "encdec/safety_cond",
            "cpu",
            "eval",
            model_tag=_MODEL_TAG,
            segment_size=128,
            feature_selection=_feature_selection(),
            eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_leaves"),
        )


@pytest.mark.parametrize("alias", ["safety_uncond", "capabilities_uncond"])
def test_removed_unconditioned_aliases_are_invalid_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    valid_segments = (
        "['capabilities', 'capabilities_cond', 'rl', 'safety', 'safety_cond', "
        "'sft', 'standard', 'standard_cond']"
    )

    with pytest.raises(ValueError, match=r"Invalid segment") as exc_info:
        get_checkpoint_dir(f"encdec/{alias}", model_tag=_MODEL_TAG)
    message = str(exc_info.value)
    assert valid_segments in message
    listed_segments = message.split("Valid segments:", maxsplit=1)[1]
    assert "safety_uncond" not in listed_segments
    assert "capabilities_uncond" not in listed_segments


def test_standard_cond_resolves_conditioned_standard_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    resolved = get_checkpoint_dir(
        "encdec/standard_cond",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=_feature_selection(),
    )

    assert resolved == (
        f"{tmp_path}/conditional_encdec_checkpoints/golden/s128/t2048.9/"
        "standard_sft_conditioned_checkpoints"
    )


def test_load_model_forwards_encdec_path_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _feature_selection()
    calls = []

    def fake_get_checkpoint_dir(source, model_tag=None, **kwargs):
        calls.append((source, model_tag, kwargs))
        return "/resolved/checkpoint"

    monkeypatch.setattr(checkpoint_manager, "get_checkpoint_dir", fake_get_checkpoint_dir)
    monkeypatch.setattr(checkpoint_manager, "find_last_step", lambda _path: 7)
    monkeypatch.setattr(
        checkpoint_manager,
        "build_encdec_model",
        lambda *_args, **_kwargs: (object(), object(), {}),
    )

    load_model(
        "encdec/safety_cond",
        "cpu",
        "eval",
        model_tag=_MODEL_TAG,
        segment_size=128,
        feature_selection=selection,
        eval_conditioning=EvalConditioning.deferred("test_checkpoint_source_leaves"),
    )

    assert calls == [
        (
            "encdec/safety_cond",
            _MODEL_TAG,
            {"segment_size": 128, "feature_selection": selection},
        )
    ]
