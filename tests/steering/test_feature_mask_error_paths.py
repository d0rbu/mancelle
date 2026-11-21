# tests/steering/test_feature_mask_error_paths.py
"""Targeted negative-path tests for FeatureMaskError raise sites.

Each test is deliberately narrow: it exercises a single error branch and pins
the public error message via ``match=``.

Why a separate file instead of extending an existing one?
  - The sites span three different call paths (FeatureMask.__post_init__,
    allowed_universe_mask, _coerce_block, _parse_ids_file) and three different
    existing files (test_feature_mask.py, test_feature_mask_apply.py,
    test_feature_mask_resolve.py).
  - Grouping them here avoids cross-file conceptual pollution and keeps each
    existing file thematically coherent.
  - A single discoverable file named _error_paths.py is easier to audit than
    five scattered additions across multiple files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from beartype.roar import BeartypeCallHintParamViolation

from errors import FeatureMaskError
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    resolve_feature_mask,
)

# ---------------------------------------------------------------------------
# Site 1 (line 52): FeatureMask.__post_init__ ids type check
#
# NOTE on why we construct FeatureMask directly here:
#   resolve_feature_mask -> _validate_and_canonicalize_ids (line 236) raises
#   BEFORE FeatureMask is even constructed, so the resolve path never reaches
#   line 51-52.  The __post_init__ check at line 51-52 is only reachable by
#   constructing FeatureMask directly with a bad ids tuple (or via
#   feature_mask_leaf_from_cfg which also goes through
#   _validate_and_canonicalize_ids first).  Direct construction is the only
#   public path to line 52.
#
# We pass bad ids through an Any-typed variable so ty does not flag them as
# invalid-argument-type; the point of these tests is the runtime guard, not
# the static type.
# ---------------------------------------------------------------------------


def test_featuremask_rejects_float_id_at_typed_boundary() -> None:
    """The refined constructor rejects a non-integral float in IDs."""
    bad_ids: Any = (1, 3.7)
    with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
        FeatureMask(mode=FeatureMaskMode.blacklist, ids=bad_ids)


def test_featuremask_rejects_integral_float_id_at_typed_boundary() -> None:
    """The refined constructor rejects 3.0 even though it is integral."""
    bad_ids: Any = (3.0,)
    with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
        FeatureMask(mode=FeatureMaskMode.blacklist, ids=bad_ids)


def test_featuremask_rejects_bool_id_at_typed_boundary() -> None:
    """The refined constructor rejects a bool in IDs (the subtle branch).

    In Python ``isinstance(True, int)`` is True. The shared strict-integer
    phantom explicitly excludes bool, preventing True from becoming ID 1.
    """
    bad_ids: Any = (True,)
    with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
        FeatureMask(mode=FeatureMaskMode.blacklist, ids=bad_ids)


def test_featuremask_rejects_false_id_at_typed_boundary() -> None:
    """The refined constructor rejects False despite bool subclassing int."""
    bad_ids: Any = (False,)
    with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
        FeatureMask(mode=FeatureMaskMode.blacklist, ids=bad_ids)


def test_featuremask_rejects_negative_id_at_typed_boundary() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
        FeatureMask(mode=FeatureMaskMode.blacklist, ids=(-1, 3))


# ---------------------------------------------------------------------------
# Site 2: allowed_universe_mask — max id >= n_sae_features
# ---------------------------------------------------------------------------


def test_allowed_universe_rejects_id_ge_n_sae_features() -> None:
    """allowed_universe_mask rejects an ID equal to the SAE width."""
    mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(5,))
    # n_sae_features=5 → id 5 is out of range (valid range is [0, 5))
    with pytest.raises(
        FeatureMaskError,
        match=r"feature_mask id 5 >= n_sae_features 5",
    ):
        mask.allowed_universe_mask(5)


def test_allowed_universe_rejects_id_strictly_greater() -> None:
    """allowed_universe_mask rejects an ID above the SAE width."""
    mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(99,))
    with pytest.raises(
        FeatureMaskError,
        match=r"feature_mask id 99 >= n_sae_features 10",
    ):
        mask.allowed_universe_mask(10)


# ---------------------------------------------------------------------------
# Site 3 (line 187): _coerce_block — feature_mask value is not a mapping
#
# Drive through resolve_feature_mask (the public path): passing a plain Python
# list as the feature_mask value is a non-dict, non-OmegaConf object, so
# OmegaConf.is_config is False and we fall straight to the isinstance check.
# ---------------------------------------------------------------------------


def test_resolve_rejects_non_mapping_config_root() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        resolve_feature_mask(object(), 100)  # ty: ignore[invalid-argument-type]


def test_resolve_rejects_list_as_feature_mask_block() -> None:
    """_coerce_block raises when feature_mask is a list, not a dict (line 187)."""
    with pytest.raises(
        FeatureMaskError,
        match=r"feature_mask must be a mapping, got list",
    ):
        resolve_feature_mask({"feature_mask": [1, 2, 3]}, 100)


def test_resolve_rejects_string_as_feature_mask_block() -> None:
    """_coerce_block raises when feature_mask is a string, not a dict (line 187)."""
    with pytest.raises(
        FeatureMaskError,
        match=r"feature_mask must be a mapping, got str",
    ):
        resolve_feature_mask({"feature_mask": "blacklist"}, 100)


# ---------------------------------------------------------------------------
# Site 4 (line 201): _parse_ids_file — yaml top-level is not a list
#
# Drive through resolve_feature_mask via a path: pointing to a temp yaml
# file whose top-level value is a dict (not a list).
# ---------------------------------------------------------------------------


def test_resolve_path_yaml_not_list(tmp_path: Path) -> None:
    """_parse_ids_file raises when yaml file top-level is not a list (line 201)."""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"a": 1, "b": 2}))
    with pytest.raises(
        FeatureMaskError,
        match=r"manual feature YAML must be a flat list.*got dict",
    ):
        resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(p)}}, 100
        )


def test_resolve_path_yaml_scalar_not_list(tmp_path: Path) -> None:
    """_parse_ids_file raises when yaml file top-level is a scalar (line 201)."""
    p = tmp_path / "scalar.yaml"
    p.write_text("42\n")
    with pytest.raises(
        FeatureMaskError,
        match=r"manual feature YAML must be a flat list.*got int",
    ):
        resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(p)}}, 100
        )


# ---------------------------------------------------------------------------
# Path-based masks use the shared strict YAML validator and canonical text
# feature-ID parser. Both dependency-level errors must be adapted to the public
# FeatureMaskError contract.
# ---------------------------------------------------------------------------


def test_resolve_path_yaml_non_integer_entry_raises(tmp_path: Path) -> None:
    """YAML path entries flow through the shared strict-id validator."""
    p = tmp_path / "bad_entries.yaml"
    p.write_text(yaml.safe_dump([1, "abc", 3]))
    with pytest.raises(
        FeatureMaskError,
        match=r"strict int or quoted digits",
    ):
        resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(p)}}, 100
        )


def test_resolve_path_txt_non_integer_entry_raises(tmp_path: Path) -> None:
    """Canonical text-parser failures are exposed as FeatureMaskError."""
    p = tmp_path / "bad.txt"
    p.write_text("1\nabc\n3\n")
    with pytest.raises(
        FeatureMaskError,
        match=r"non-integer entry",
    ):
        resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(p)}}, 100
        )
