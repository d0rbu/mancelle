"""Config tests for ``configs/eval/build_eval_context_cache_clustered_axbench.yaml``
(PR #739 Track D, D-axbench).

The clustered-AxBench preset over ``build_eval_context_cache``. The cache is a
pure function of (feature_mask, feature_clusters, seed) + context set + SAE, so
it is content-addressed and needs NO checkpoint reference -- it can be built
concurrently with (or before) the training run whose checkpoint will consume
it. These are pure config-composition tests (Hydra compose API, no build, no
I/O) asserting exactly that: the preset composes to the clustered shape,
requires the mask/cluster paths, injects no silent seed default, and pulls in
no checkpoint dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "eval")
_CONFIG_NAME = "build_eval_context_cache_clustered_axbench"


def _compose(hydra_compose, overrides=None):
    return hydra_compose(_CONFIG_DIR, _CONFIG_NAME, overrides or [])


def test_preset_composes(hydra_compose):
    """The preset composes over its base (no missing-default / bad-defaults
    error). Unfilled ``???`` are legal at compose time -- they only raise on
    access -- so composition alone proves the defaults chain is sound."""

    cfg = _compose(hydra_compose)
    assert cfg is not None


def test_feature_mask_mode_is_whitelist_and_delta_null(hydra_compose):
    """The preset fixes the two non-``???`` clustered invariants: a whitelist
    mask and feature_delta=null (clustered builds forbid splice)."""

    cfg = _compose(hydra_compose)
    assert cfg.feature_mask.mode == "whitelist"
    assert cfg.feature_delta is None


@pytest.mark.parametrize("field", ["feature_mask.path", "feature_clusters.path"])
def test_mask_and_cluster_paths_are_mandatory(hydra_compose, field):
    """The two identity-defining path inputs are genuinely mandatory: composing
    without them and then accessing raises MissingMandatoryValue. Together with
    ``seed`` (below) they are the (mask, clusters, seed) triple the cache is
    content-addressed by."""

    cfg = _compose(hydra_compose)
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.select(cfg, field, throw_on_missing=True)


def test_no_silent_seed_default(hydra_compose):
    """The preset injects NO silent seed value: seed is null (absent) by
    default, so a clustered build must supply it -- and a null seed then fails
    loud at runtime (steering/eval/cache_identity.py's clustered-seed guard,
    which requires a non-negative int). A silently-defaulted seed (e.g. 0)
    would be exactly the kind of quiet wrong-cache bug this must avoid."""

    cfg = _compose(hydra_compose)
    assert OmegaConf.select(cfg, "seed", throw_on_missing=False) is None


def test_needs_no_checkpoint_reference(hydra_compose):
    """The cache is checkpoint-agnostic: the preset pulls in NO checkpoint
    dependency, so it can be produced in parallel with training (no sequential
    train -> build -> eval). Every base checkpoint field stays null; the
    optional D-contract cross-check therefore stays dormant unless an operator
    opts in explicitly."""

    cfg = _compose(hydra_compose)
    assert cfg.checkpoint_dir is None
    assert cfg.checkpoint_source is None
    assert cfg.checkpoint_model_tag is None


def test_composes_with_mask_cluster_seed_filled(hydra_compose):
    """Filling the (mask, clusters, seed) triple + stats/manifest resolves to
    the expected clustered shape -- no checkpoint override anywhere. (The base
    config's own ``???`` -- contexts_file, feature_dir, identity.*, etc. --
    stay missing here; this asserts only the preset's own surface.)"""

    cfg = _compose(
        hydra_compose,
        overrides=[
            "seed=42",
            "feature_clusters.path=/tmp/clusters.json",
            "feature_mask.path=/tmp/mask.txt",
            f"stats_source_sha256={'a' * 64}",
            "required_contexts_manifest=/tmp/manifest.json",
        ],
    )
    assert cfg.seed == 42
    assert cfg.feature_clusters.path == "/tmp/clusters.json"
    assert cfg.feature_mask.mode == "whitelist"
    assert cfg.feature_mask.path == "/tmp/mask.txt"
    assert cfg.stats_source_sha256 == "a" * 64
    assert cfg.feature_delta is None
    assert cfg.checkpoint_dir is None
