"""Fail-loud clustered eval-context-cache checkpoint/seed cross-check.

PR #739 Track D (D-contract). D-seed (merged) made a clustered eval-context
cache's configured ``seed`` semantically load-bearing -- both its cache
identity hash (``steering/eval/cache_identity.py``'s clustered branch) and
its actually-sampled content (``build_feature_pipeline_from_cfg_block``) are
now pinned to it -- but added no source of truth to check the configured
value against. A clustered cache is inseparable from the compacted
checkpoint that will consume it, so an operator who mistypes ``seed`` (or
reuses a config across checkpoints without updating it) previously got a
cache that was silently, confidently wrong: distinct from every other seed's
cache (by identity), internally self-consistent (by content), and simply NOT
the checkpoint's real training-time selection.

This module is that check. It resolves the target checkpoint's real
training-time ``seed`` DIRECTLY from the checkpoint's own frozen provenance
(``resolve_checkpoint_pin`` -- the same TOCTOU-safe hash-and-load Track B
uses) and compares it against the configured ``seed`` before the caller does
any shard or context I/O.

Deliberately NARROW (PR #739 D-contract, Option B): this cross-check reads
ONLY the checkpoint's real ``seed`` plus its file provenance. It does NOT run
Track B's full ``resolve_cluster_feature_contract`` agreement (mask / cluster
/ SAE / statistics / selection). That agreement is AxBench-clustered-eval-
specific and, among other things, hard-rejects any checkpoint trained with
``delta_weights`` / ``feature_delta`` / ``feature_prediction`` (see
``steering/eval/cluster_checkpoint_pin.py::_resolve_selection_identity``'s
delta/prediction raises) -- which would reject the one real clustered
production checkpoint this check exists to guard: ``configs/training/
cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml``, a
splice-counterfactual + feature-prediction run (``delta_weights:
percentile_delta``, ``feature_prediction`` enabled) whose fully-resolved
training config is persisted verbatim into ``user_config``
(``scripts/train/conditional_encdec_train.py::_checkpoint_user_config``).

A pure seed cross-check needs none of that agreement: the checkpoint file
digests (model / metadata SHA-256, recorded in ``CheckpointSeedProvenance``)
already pin WHICH checkpoint was checked, and reading ``user_config.seed`` is
delta-agnostic. It also imposes NO constraint on the build's ``feature_mask``
shape -- null, inline ``{mode, ids}``, or path form all pass -- because it
never parses the mask at all.

Ordering is load-bearing, not stylistic: this function's caller
(``scripts/eval/build_eval_context_cache.py::build_cache_from_config``) must
invoke it before ``load_context_records`` / ``glob_features`` -- i.e. before
ANY shard or context I/O -- not merely before the clustered-cache-metadata
helper it also feeds. See that call site's own comment for the exact
ordering constraint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from steering.eval.cluster_checkpoint_pin import (
    resolve_checkpoint_pin,
    resolve_eval_checkpoint_directory,
)
from steering.eval.cluster_contract_identity import ClusterFeatureContractError


@dataclass(frozen=True)
class CheckpointSeedProvenance:
    """Provenance of the checkpoint a clustered eval-context cache's ``seed``
    was cross-checked against.

    Recorded (additively) in the clustered-cache metadata as five
    ``checkpoint_``-prefixed keys via :meth:`metadata_dict`; NOT part of the
    cache identity hash (``steering/eval/cache_identity.py`` never reads any
    ``checkpoint_*`` key). Every field comes straight from the checkpoint's
    frozen ``CheckpointPin`` (resolved tag/step, model/metadata file digests)
    plus its real training-time ``user_config.seed`` -- no Track B contract
    object, so it is valid for delta-aware (splice-CF / feature-prediction)
    clustered checkpoints too.
    """

    resolved_tag: str
    step: int
    model_sha256: str
    metadata_sha256: str
    preselection_seed: int

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_resolved_tag": self.resolved_tag,
            "checkpoint_step": self.step,
            "checkpoint_model_sha256": self.model_sha256,
            "checkpoint_metadata_sha256": self.metadata_sha256,
            "checkpoint_preselection_seed": self.preselection_seed,
        }


def _checkpoint_preselection_seed(metadata: Mapping[str, Any]) -> int:
    """Read the checkpoint's real training-time preselection seed from its
    metadata's ``user_config.seed``.

    Mirrors ``steering/eval/cluster_checkpoint_pin.py::
    _resolve_selection_identity``'s own seed read (``user_config`` must be a
    ``Mapping``; ``type(seed) is int`` -- ``is int`` not ``isinstance``
    rejects ``bool``) so this narrow path pins the identical field with the
    identical strictness the full contract would, minus the AxBench
    agreement.
    """

    user_config = metadata.get("user_config")
    if not isinstance(user_config, Mapping):
        raise ClusterFeatureContractError("checkpoint user_config is missing")
    seed = user_config.get("seed")
    if type(seed) is not int:
        raise ClusterFeatureContractError(
            f"checkpoint preselection seed must be a strict int, got {seed!r}"
        )
    return seed


def assert_clustered_seed_matches_checkpoint(
    cfg: DictConfig,
) -> CheckpointSeedProvenance | None:
    """Cross-check ``cfg.seed`` against a target checkpoint's real seed.

    Opt-in: returns ``None`` immediately, without resolving anything, only
    when NO checkpoint reference field is configured at all (the default:
    every ``checkpoint_*`` cfg field null) -- unclustered builds and
    pre-D-contract clustered builds are unaffected. A PARTIAL reference is
    NOT "no reference": if any ``checkpoint_*`` field is set (even a lone
    ``checkpoint_model_tag``, ``checkpoint_step``, or ``checkpoint_tag`` --
    none of which is an independently sufficient reference; a valid one is
    ``checkpoint_dir`` OR ``checkpoint_source`` + ``checkpoint_model_tag``),
    this proceeds and lets ``resolve_eval_checkpoint_directory`` fail loud on
    the misconfiguration rather than silently skipping the check.

    Raises ``ClusterFeatureContractError`` when the configured seed is
    malformed, the checkpoint reference is half-configured, the checkpoint
    cannot be resolved/loaded, or the configured seed disagrees with the
    checkpoint's real training-time seed. Reads ONLY the checkpoint's real
    ``seed`` + file provenance -- see the module docstring for why it does
    NOT run Track B's full mask/cluster/SAE/selection agreement (and so works
    for delta-aware clustered checkpoints and any ``feature_mask`` shape).
    """

    # Opt-in gate: skip the whole cross-check ONLY when no checkpoint
    # reference field is configured at all. ANY checkpoint_* field being set
    # signals checkpoint-cross-check intent, so a PARTIAL reference (e.g.
    # checkpoint_model_tag without checkpoint_source, or a bare
    # checkpoint_step/checkpoint_tag) must NOT silently return None here --
    # that would bypass the exact guarantee this module provides. It instead
    # proceeds so resolve_eval_checkpoint_directory's own "exactly one of
    # checkpoint_dir or checkpoint_source/checkpoint_model_tag" gate fails
    # loud on the misconfiguration. `is None` (not falsiness) is deliberate:
    # checkpoint_step=0 is an explicit selector, not "unset".
    checkpoint_dir = cfg.get("checkpoint_dir", None)
    checkpoint_source = cfg.get("checkpoint_source", None)
    checkpoint_model_tag = cfg.get("checkpoint_model_tag", None)
    checkpoint_step = cfg.get("checkpoint_step", None)
    checkpoint_tag = cfg.get("checkpoint_tag", None)
    if (
        checkpoint_dir is None
        and checkpoint_source is None
        and checkpoint_model_tag is None
        and checkpoint_step is None
        and checkpoint_tag is None
    ):
        return None

    # Cheapest check first: reject a malformed seed before any filesystem or
    # checkpoint resolution. `type(...) is not int` (not `isinstance`)
    # rejects `bool` (a `bool` is an `int` subclass in Python) -- mirrors
    # steering/eval/cache_identity.py's own clustered-branch seed guard
    # (`_canonical_hash_from_conditioning`'s `seed_raw` check).
    seed_raw = cfg.get("seed", None)
    if type(seed_raw) is not int or seed_raw < 0:
        raise ClusterFeatureContractError(
            "clustered eval-context cache checkpoint cross-check requires "
            f"cfg.seed to be a non-negative strict int; got {seed_raw!r}"
        )

    # Resolve WHICH checkpoint (a direct dir, or the same source/model_tag
    # runtime pair) and freeze + TOCTOU-safe hash-and-load it. resolve_
    # checkpoint_pin is Track B's own checkpoint-pin primitive, but WITHOUT
    # its downstream agreement: no delta/prediction rejection, no mask/cluster/
    # SAE/statistics/selection resolution -- exactly the narrowing this
    # cross-check requires. Both raise ClusterFeatureContractError for a
    # half-configured reference (the resolve_eval_checkpoint_directory XOR
    # gate) or a malformed/corrupt checkpoint; but their strict path
    # resolution (a missing checkpoint_dir, an empty dir with no step, or a
    # nonexistent explicit step) instead raises a bare OSError/
    # FileNotFoundError. Normalize those to ClusterFeatureContractError so
    # this function honors its documented "cannot be resolved/loaded ->
    # ClusterFeatureContractError" contract. A ClusterFeatureContractError
    # raised inside is NOT an OSError, so it propagates unwrapped (e.g. the
    # partial-reference XOR raise the merged opt-in gate relies on).
    try:
        checkpoint_directory, resolved_tag = resolve_eval_checkpoint_directory(
            checkpoint_dir=checkpoint_dir,
            checkpoint_source=checkpoint_source,
            checkpoint_model_tag=checkpoint_model_tag,
        )
        pin, metadata, _ = resolve_checkpoint_pin(
            checkpoint_directory,
            step=checkpoint_step,
            resolved_tag=checkpoint_tag or resolved_tag,
        )
    except OSError as exc:
        raise ClusterFeatureContractError(
            "could not resolve or load the target clustered checkpoint "
            f"(checkpoint_dir={checkpoint_dir!r}, checkpoint_source="
            f"{checkpoint_source!r}, checkpoint_model_tag={checkpoint_model_tag!r}, "
            f"checkpoint_step={checkpoint_step!r}): {exc}"
        ) from exc
    expected_seed = _checkpoint_preselection_seed(metadata)
    if seed_raw != expected_seed:
        raise ClusterFeatureContractError(
            "clustered eval-context cache cfg.seed does not match the target "
            f"checkpoint's real training-time preselection seed: cfg.seed="
            f"{seed_raw}, checkpoint preselection_seed={expected_seed}"
        )
    return CheckpointSeedProvenance(
        resolved_tag=pin.resolved_tag,
        step=pin.step,
        model_sha256=pin.model_file.sha256,
        metadata_sha256=pin.metadata_file.sha256,
        preselection_seed=expected_seed,
    )
