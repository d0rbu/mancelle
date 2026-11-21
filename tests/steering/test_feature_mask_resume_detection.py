"""Feature-mask resume checks use only canonical checkpoint metadata."""

from __future__ import annotations

import pytest

from errors import FeatureMaskError
from steering.conditional.feature_mask import (
    FeatureMask,
    assert_resume_feature_mask_matches,
    canonical_dict,
)

CHECKPOINT_DIR = "/tmp/checkpoints/run"


def _meta_for(mask: FeatureMask | None) -> dict:
    meta = {"step": 42, "user_config": {"feature_mask": {"path": "/raw/path.txt"}}}
    if mask is not None:
        meta["feature_mask_canonical"] = canonical_dict(mask)
    return meta


def test_configured_mask_requires_canonical_metadata() -> None:
    """Raw ``user_config.feature_mask`` is intentionally not re-resolved."""
    configured = FeatureMask("blacklist", (1, 2))
    meta = {"step": 42, "user_config": {"feature_mask": {"ids": [1, 2]}}}

    with pytest.raises(FeatureMaskError, match="feature_mask_canonical"):
        assert_resume_feature_mask_matches(
            configured,
            meta,
            checkpoint_dir=CHECKPOINT_DIR,
        )


def test_configured_mask_rejects_explicit_none_canonical_metadata() -> None:
    configured = FeatureMask("blacklist", (1,))
    meta = {"step": 42, "feature_mask_canonical": None}

    with pytest.raises(FeatureMaskError, match="feature_mask mismatch"):
        assert_resume_feature_mask_matches(
            configured,
            meta,
            checkpoint_dir=CHECKPOINT_DIR,
        )


def test_matching_canonical_metadata_allows_resume() -> None:
    configured = FeatureMask("whitelist", (3, 5, 8))

    assert_resume_feature_mask_matches(
        configured,
        _meta_for(configured),
        checkpoint_dir=CHECKPOINT_DIR,
    )


@pytest.mark.parametrize(
    "checkpoint_mask",
    [
        FeatureMask("blacklist", (3, 5, 8)),
        FeatureMask("whitelist", (3, 5)),
    ],
)
def test_mismatched_canonical_metadata_raises(checkpoint_mask: FeatureMask) -> None:
    configured = FeatureMask("whitelist", (3, 5, 8))

    with pytest.raises(FeatureMaskError, match="feature_mask mismatch"):
        assert_resume_feature_mask_matches(
            configured,
            _meta_for(checkpoint_mask),
            checkpoint_dir=CHECKPOINT_DIR,
        )


def test_no_configured_mask_skips_resume_check() -> None:
    assert_resume_feature_mask_matches(
        None,
        {"step": 42, "feature_mask_canonical": {"mode": "blacklist", "ids": [1]}},
        checkpoint_dir=CHECKPOINT_DIR,
    )
    assert_resume_feature_mask_matches(
        None,
        {"step": 42, "user_config": {}},
        checkpoint_dir=CHECKPOINT_DIR,
    )
