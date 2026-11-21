"""Build Intervention objects for each registry method (incl. the no-steering controls).

Reuses primitives from steering.eval.axbench_base rather than reimplementing.
"""

from __future__ import annotations

import contextlib
import json
import random
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch as t

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_base.baselines.hooks import register_residual_hook
from steering.eval.axbench_base.baselines.lora.loader import attach_lora_from_disk
from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactRequest,
    load_direction_artifact,
)
from steering.eval.axbench_base.direction_scale import (
    normalize_direction,
    require_direction_max_activation,
)
from steering.eval.axbench_base.encoder_input import (
    build_mode_aware_encoder_input,
)
from steering.eval.axbench_base.intervention_dispatch import (
    _load_top_k_concept_fids,
)
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_base.stage4_prompts import (
    prompting_prefix_text as _prompting_prefix_text,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    InterventionKind,
    LoraParams,
    SteeringMode,
    TargetModel,
)
from steering.eval.axbench_bpb.vocab_restriction import (
    UNCOMPACTED_VOCAB_FINGERPRINT,
    checkpoint_vocab_fingerprint,
    feature_vocab_allowed_ids,
    restrict_to_vocab,
)
from steering.eval.axbench_bpb_adv.gcg_family import ALPHA_AXIS_GCG_METHODS
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    VOCAB_FINGERPRINT_META_KEY,
    read_ours_gcg_bag,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    ROOFLINE_PRECOMPUTED_BAG_FAMILIES,
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


def _assert_bag_matches_checkpoint_vocab(
    meta: Mapping[str, str],
    *,
    model: Any,
    method_name: str,
    concept_id: int,
    bag_path: Any,
) -> None:
    """Fail loudly if a bag-backed candidate's stamped ``V`` disagrees with ``model``'s.

    ``meta`` is the metadata mapping :func:`read_ours_gcg_bag` returned. The
    on-disk precomputed-bag cache keys ONLY on ``(concept_id, mode, reg_lambda,
    family, prefix_id)`` under an operator-chosen ``run_tag`` — it carries no
    checkpoint or ``V`` component. So reusing a ``run_tag`` across checkpoints
    with a DIFFERENT compacted vocabulary is otherwise indistinguishable, at read
    time, from a legitimate cache hit, and would silently condition on feature ids
    the currently-loaded checkpoint never saw (some may even be out of its
    embedding table's range). This is the belt to ``FeatureEncoder._to_compact``'s
    suspenders — it fires here, at intervention-build time, rather than deep
    inside a forward pass.

    Back-compat for a bag written before ``vocab_fingerprint`` stamping existed
    (no :data:`VOCAB_FINGERPRINT_META_KEY` entry in ``meta`` at all): we cannot
    prove such a bag was derived for the CURRENTLY loaded checkpoint's ``V``, so
      * against a COMPACTED checkpoint (a real ``V``-fingerprint) — RAISE. Silently
        trusting an unstamped bag here is exactly the silent-wrong-bag risk this
        function exists to close.
      * against a LEGACY checkpoint (``feature_vocab is None``, so
        ``checkpoint_vocab_fingerprint`` returns
        :data:`UNCOMPACTED_VOCAB_FINGERPRINT`) — load unchanged. Every bag written
        before compaction existed was implicitly derived against the full
        (uncompacted) vocabulary, which is exactly what a legacy checkpoint has;
        legacy behavior must stay byte-identical to pre-#695.
    """
    loaded_fingerprint = checkpoint_vocab_fingerprint(model)
    stamped_fingerprint = meta.get(VOCAB_FINGERPRINT_META_KEY)
    if stamped_fingerprint is None:
        if loaded_fingerprint == UNCOMPACTED_VOCAB_FINGERPRINT:
            return
        raise AxBenchBpbError(
            f"precomputed bag at {bag_path} for method={method_name!r} "
            f"concept_id={concept_id} carries no {VOCAB_FINGERPRINT_META_KEY!r} "
            "metadata (written before checkpoint-V stamping existed), and the "
            f"loaded checkpoint is COMPACTED (V-fingerprint={loaded_fingerprint!r}). "
            "Cannot prove this bag was derived for this checkpoint's V -- a bag "
            "derived against a different (or unmasked) vocabulary would silently "
            "condition on the wrong feature ids. Re-derive the bag under this "
            "checkpoint (clear the precomputed-bag cache / use a fresh run_tag)."
        )
    if stamped_fingerprint != loaded_fingerprint:
        raise AxBenchBpbError(
            f"precomputed bag at {bag_path} for method={method_name!r} "
            f"concept_id={concept_id} was derived against checkpoint "
            f"V-fingerprint={stamped_fingerprint!r}, but the loaded checkpoint's "
            f"V-fingerprint={loaded_fingerprint!r}. The on-disk bag cache key "
            "carries no checkpoint/V component, so reusing a run_tag across "
            "checkpoints with a different V would otherwise silently serve a "
            "WRONG bag. Use a fresh run_tag for this checkpoint, or clear the "
            "stale precomputed-bag cache."
        )


def _assert_feature_conditions_within_vocab(
    feature_conditions: Mapping[int, int],
    *,
    allowed_ids: t.Tensor | None,
    method_name: str,
    concept_id: int,
) -> None:
    """Fail if an ``ours`` producer emits a feature outside checkpoint ``V``.

    Selection-time restriction is the primary correctness mechanism. This
    intervention-boundary assertion is an independent producer backstop: it
    catches a missed or corrupt producer before teacher-force scoring builds
    encoder tensors. The tensor-level assertion in ``teacher_force.py`` remains
    the final consumer backstop.
    """
    if allowed_ids is None:
        return
    candidate_ids = t.tensor(
        list(feature_conditions),
        dtype=t.int64,
        device=allowed_ids.device,
    )
    _kept, dropped = restrict_to_vocab(candidate_ids, allowed_ids)
    if dropped.numel() == 0:
        return
    offending_ids = sorted({int(fid) for fid in dropped.tolist()})
    raise AxBenchBpbError(
        f"ours intervention producer invariant violation for "
        f"method={method_name!r} concept_id={concept_id}: final "
        f"feature_conditions contains checkpoint-V-excluded feature ids "
        f"{offending_ids}. Every live conditioning producer must restrict "
        "its output to the loaded checkpoint vocabulary before returning."
    )


def _assert_concept_top_k_within_sae_range(
    top_k: list[int],
    *,
    target_model_meta: Any,
    method_name: str,
    concept_id: int,
    artifact_path: Any,
) -> None:
    """Reject corrupt non-sentinel concept IDs before checkpoint-V filtering."""
    n_sae_features = getattr(target_model_meta, "n_sae_features", None)
    if isinstance(n_sae_features, bool) or not isinstance(n_sae_features, int):
        raise AxBenchBpbError(
            "cannot validate compacted-checkpoint concept feature artifact "
            f"{artifact_path} for method={method_name!r} concept_id={concept_id}: "
            "target_model_meta.n_sae_features must be an integer, got "
            f"{n_sae_features!r}."
        )
    if n_sae_features <= 0:
        raise AxBenchBpbError(
            "cannot validate compacted-checkpoint concept feature artifact "
            f"{artifact_path} for method={method_name!r} concept_id={concept_id}: "
            f"n_sae_features must be positive, got {n_sae_features}."
        )
    offending_ids = sorted(
        {fid for fid in top_k if fid < 0 or fid >= n_sae_features}
    )
    if not offending_ids:
        return
    raise AxBenchBpbError(
        f"concept feature artifact {artifact_path} for method={method_name!r} "
        f"concept_id={concept_id} contains feature ids outside the loaded "
        f"checkpoint SAE range [0, {n_sae_features}): {offending_ids}. "
        "The artifact is corrupt or was produced for a different SAE; "
        "rebuild it instead of silently filtering invalid ids."
    )


def _build_ours_intervention(
    *,
    method: Any,
    concept: Any,
    coefficient: int | float | None,
    source_paths: SourcePaths,
    target_model_meta: Any,
    inactive_sampler: Any | None,
    rng: random.Random,
    adv_paths: Any | None = None,
    core_prefix_id: str | None = None,
    model: Any,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Intervention:
    """Wrap axbench-base's build_mode_aware_encoder_input.

    For precomputed conditioning-bag methods (classified by
    ``precomputed_bag_method_family`` — the ONE home of the method-name →
    family mapping, shared with the derive and the score-stage coverage
    guard, so this loader can never disagree with them about what
    counts as a bag-backed method) this function bypasses
    ``build_mode_aware_encoder_input`` entirely and loads the
    precomputed conditioning bag from ``adv_paths`` instead.  The bag
    path is keyed by ``(concept_id, mode, reg_lambda=coefficient,
    family)``, with an additional ``prefix_id=core_prefix_id`` for
    the roofline family.

    Missing-bag policy (Task-10 choice):
        A missing bag now RAISES ``AxBenchBpbError``. The score stage's
        prefix coverage guard scores a bag-backed candidate ONLY when its
        ``core_prefix_id`` is in the derive's persisted scored set — exactly
        the prefixes whose bag was written.
        So once a row reaches this builder its bag MUST exist; an absent bag
        is a genuine pipeline bug (a stale / never-run derive), not the
        routine empty-optimize-group case the earlier warn+empty path
        tolerated. A corrupt / unreadable artifact (file exists but cannot
        be parsed) also raises, as before.

    ``model`` -- checkpoint-``V`` safety:
        The loaded target encoder-decoder model is REQUIRED for every ``ours``
        path. Non-bag-backed methods restrict concept top-K and inactive
        sampling to its persistent checkpoint vocabulary. Bag-backed methods
        validate their derive-time vocabulary fingerprint against it. Every
        returned mapping is then checked for checkpoint-V membership.
    """
    allowed_ids = feature_vocab_allowed_ids(model)

    # --- precomputed conditioning-bag fast path ---------------------------
    # partition_modifier_name may be a PartitionModifier member OR its plain
    # string value (some Method-like objects / fixtures carry the raw string).
    # str() normalizes both — a StrEnum stringifies to its value, a str is
    # itself — so classification matches the frozenset membership check this
    # replaced (which accepted both via StrEnum hash equality). Using
    # `.value` would crash with AttributeError on the plain-string form.
    modifier_name = getattr(method, "partition_modifier_name", None)
    family = (
        precomputed_bag_method_family(str(modifier_name)) if modifier_name is not None else None
    )
    if family is not None:
        if adv_paths is None:
            raise AxBenchBpbError(
                f"precomputed bag paths must be provided for bag-backed method "
                f"{method.name!r}; pass adv_paths=<paths object> to "
                "build_intervention."
            )
        is_roofline = family in ROOFLINE_PRECOMPUTED_BAG_FAMILIES
        prefix_id = core_prefix_id if is_roofline else None
        is_feature_diffmean = family in {
            PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        }
        # Which axis this method's coefficient indexes. The alpha methods share
        # the fair GCG family but sweep the class-margin alpha, so their
        # coefficient is NOT a reg_lambda; reading it as one would load the
        # lambda-swept bag and report it under the alpha method's name.
        is_alpha_axis = str(modifier_name) in ALPHA_AXIS_GCG_METHODS
        reg_lambda = (
            0.0
            if (is_feature_diffmean or is_alpha_axis)
            else float(coefficient)
            if coefficient is not None
            else 0.0
        )
        objective_alpha = (
            float(coefficient)
            if is_alpha_axis and coefficient is not None
            else DEFAULT_OBJECTIVE_ALPHA
        )
        bag_path = adv_paths.ours_gcg_bag(
            concept.concept_id,
            method.mode,
            reg_lambda,
            family,
            prefix_id=prefix_id,
            objective_alpha=objective_alpha,
        )
        if not bag_path.exists():
            raise AxBenchBpbError(
                f"precomputed bag not found for method={method.name!r} "
                f"concept_id={concept.concept_id} mode={str(method.mode)!r} "
                f"reg_lambda={reg_lambda} objective_alpha={objective_alpha} "
                f"family={family.value!r} "
                f"prefix_id={prefix_id!r} at path={str(bag_path)!r}. The score "
                "stage's coverage guard only scores prefixes the derive wrote a bag "
                "for, so a missing bag here is a genuine pipeline bug — the "
                "precomputed-bag derive did not run (or ran on a "
                "different prefix set) for this (concept, mode)."
            )
        try:
            # The reader owns the identity rule in BOTH directions: with an
            # expectation it must match, and with none a cluster-stamped artifact is
            # rejected as an unclustered read. Only one artifact is consumed here, so
            # unlike the derive flow there is no second read to bind it to (#938).
            bag, meta = read_ours_gcg_bag(
                bag_path,
                expected_feature_contract_fingerprint=(
                    None
                    if cluster_contract is None
                    else cluster_contract.feature_contract_fingerprint
                ),
            )
        except Exception as exc:
            raise AxBenchBpbError(
                f"corrupt or unreadable precomputed bag for method={method.name!r} "
                f"concept_id={concept.concept_id} family={family.value!r} "
                f"prefix_id={prefix_id!r} path={bag_path}: {exc}"
            ) from exc
        _assert_bag_matches_checkpoint_vocab(
            meta,
            model=model,
            method_name=method.name,
            concept_id=concept.concept_id,
            bag_path=bag_path,
        )
        _assert_feature_conditions_within_vocab(
            bag,
            allowed_ids=allowed_ids,
            method_name=method.name,
            concept_id=concept.concept_id,
        )
        return Intervention.ours(feature_conditions=bag)

    # --- existing ours_* path --------------------------------------------
    if cluster_contract is not None:
        raise AxBenchBpbError(
            f"clustered BPB-Adv does not support live ordinary ours method "
            f"{method.name!r}; use a validated feature-DiffMean/GCG bag method"
        )
    if method.mode == SteeringMode.POSITIVE:
        feat_path = source_paths.concept_features()
    else:
        feat_path = source_paths.concept_features_neg()

    top_k = _load_top_k_concept_fids(feat_path, concept.concept_id)
    if allowed_ids is not None:
        _assert_concept_top_k_within_sae_range(
            top_k,
            target_model_meta=target_model_meta,
            method_name=method.name,
            concept_id=concept.concept_id,
            artifact_path=feat_path,
        )
        candidate_ids = t.tensor(
            top_k,
            dtype=t.int64,
            device=allowed_ids.device,
        )
        kept_ids, _dropped_ids = restrict_to_vocab(candidate_ids, allowed_ids)
        top_k = [int(fid) for fid in kept_ids.tolist()]
    feature_conditions = build_mode_aware_encoder_input(
        method=method,
        active={},
        inactive={},
        top_k_concept_fids=top_k,
        coefficient=int(coefficient) if coefficient is not None else 0,
        target_model_meta=target_model_meta,
        inactive_sampler=inactive_sampler,
        rng=rng,
    )
    _assert_feature_conditions_within_vocab(
        feature_conditions,
        allowed_ids=allowed_ids,
        method_name=method.name,
        concept_id=concept.concept_id,
    )
    return Intervention.ours(feature_conditions=feature_conditions)


def _build_direction_intervention(
    *,
    method: Any,
    concept: Any,
    coefficient: float,
    source_paths: SourcePaths,
    diffmean_layer: int,
    apply_max_act_scale: bool,
    cluster_contract: ClusterFeatureContract | None = None,
    direction_payload: bytes | None = None,
) -> Intervention:
    """Load direction safetensors and build a direction-kind Intervention.

    P0.B caveat (slice 17 audit 2026-05-05): this function applies a sign
    flip to the coefficient for negative-mode methods. The `coefficient`
    argument carries the **declared grid value** (always non-negative for
    canonical grids); the **applied** coefficient on the residual hook is
    `-abs(coefficient)` for negative-mode methods. Asymmetry consequence:
    the leaderboard CSV's `coefficient_str` column reflects the declared
    grid value, NOT the sign-flipped applied value. Two grid entries with
    the same `abs(coefficient)` but different signs (rare but possible
    in a hand-tuned grid) would map to the **same** applied coefficient
    under negative mode, producing identical BPB scores under different
    `coefficient_str` keys. The SQLite primary key still discriminates
    them (different `coefficient_str`), so no data loss — but operators
    inspecting the leaderboard should treat negative-mode `coefficient_str`
    as a grid identifier rather than the literal applied value.
    """
    target = TargetModel.coerce(method.target_model)
    mode = method.mode.value
    try:
        baseline = BaselineKind(method.baseline)
    except (TypeError, ValueError) as exc:
        raise AxBenchBpbError(
            f"direction intervention has unsupported baseline={method.baseline!r}"
        ) from exc
    if baseline is BaselineKind.DIFFMEAN:
        path = source_paths.diffmean_direction(target, mode, concept.concept_id)
        kind = InterventionKind.DIFFMEAN
    elif baseline is BaselineKind.LINEAR_PROBE:
        path = source_paths.linear_probe_direction(target, mode, concept.concept_id)
        kind = InterventionKind.LINEAR_PROBE
    else:
        raise AxBenchBpbError(
            f"direction intervention requires diffmean or linear_probe, got {baseline.value!r}"
        )
    request = DirectionArtifactRequest(
        family=baseline.value,
        target_model=target.value,
        mode=mode,
        concept_id=concept.concept_id,
        layer=diffmean_layer,
    )
    if direction_payload is None:
        if cluster_contract is not None:
            raise AxBenchBpbError(
                "clustered diffmean/linear_probe intervention requires the exact "
                "direction bytes retained by source validation; "
                f"method={method.name!r} concept={concept.concept_id}"
            )
        if not path.exists():
            raise AxBenchBpbError(
                f"{baseline.value} direction file missing: {path}"
            )
        loaded = load_direction_artifact(
            path,
            request=request,
            error_type=AxBenchBpbError,
        )
    else:
        loaded = load_direction_artifact(
            direction_payload,
            artifact_path=path,
            request=request,
            error_type=AxBenchBpbError,
        )
    if cluster_contract is not None:
        metadata = loaded.metadata_dict()
        expected_fingerprint = cluster_contract.feature_contract_fingerprint
        actual_fingerprint = metadata.get("feature_contract_fingerprint")
        if actual_fingerprint != expected_fingerprint:
            raise AxBenchBpbError(
                "clustered direction payload feature-contract fingerprint "
                f"mismatch for method={method.name!r} concept={concept.concept_id}: "
                f"expected={expected_fingerprint!r}, actual={actual_fingerprint!r}"
            )
    direction = loaded.direction
    metadata = loaded.metadata_dict()
    # Match axbench-base's `_dispatch_scalar_direction`
    # (intervention_dispatch.py:420-422): negative mode flips the sign so
    # the residual hook subtracts the direction. Without this, every
    # neg_diffmean_* / neg_linear_probe_* method silently inverts the
    # leaderboard sign.
    effective_coefficient = float(coefficient)
    if method.mode == SteeringMode.NEGATIVE:
        effective_coefficient = -abs(effective_coefficient)
    if apply_max_act_scale:
        direction = normalize_direction(direction, error_cls=AxBenchBpbError)
        effective_coefficient *= require_direction_max_activation(
            metadata,
            direction=direction,
            expected_concept_id=request.concept_id,
            expected_mode=request.mode,
            expected_target_model=request.target_model,
            expected_layer=request.layer,
            surface=(
                f"{kind} direction for method={method.name!r} "
                f"concept={concept.concept_id}"
            ),
            error_cls=AxBenchBpbError,
        )
    return Intervention.from_direction(
        kind=kind,
        target_model=target,
        direction=direction,
        direction_layer=diffmean_layer,
        coefficient=effective_coefficient,
    )


def _build_lora_intervention(
    *,
    concept: Any,
    mode: str,
    source_paths: SourcePaths,
) -> Intervention:
    """Resolve the per-concept LoRA adapter path and build a lora-kind Intervention."""
    path = source_paths.lora_adapter(mode, concept.concept_id)
    if not path.exists():
        raise AxBenchBpbError(
            f"lora adapter file missing for concept={concept.concept_id} mode={mode!r}: {path}"
        )
    return Intervention.lora(adapter_path=path)


def _build_prompting_intervention(
    *,
    concept: Any,
    mode: str,
    source_paths: SourcePaths,
    target_model: TargetModel | str,
    tokenizer: Any,
) -> Intervention:
    """Render axbench-base's per-mode prompt template and build a prompting-kind Intervention."""
    path = source_paths.prompting_descriptions(mode)
    if not path.exists():
        raise AxBenchBpbError(f"prompting descriptions file missing for mode={mode!r}: {path}")
    data = json.loads(path.read_text())
    description = data.get(str(concept.concept_id))
    if description is None:
        raise AxBenchBpbError(
            f"no prompting description for concept={concept.concept_id} in {path}"
        )
    # Render the template wrapper to match axbench-base's Stage 4 dispatch.
    prompt_text = _prompting_prefix_text(SteeringMode(mode), description)
    prompt_tokens = tokenizer.encode(prompt_text)
    return Intervention.prompting(
        target_model=target_model,
        prompt_prefix_tokens=prompt_tokens,
    )


def build_intervention(
    *,
    method: Any,
    concept: Any,
    coefficient: int | float | None,
    source_paths: SourcePaths,
    diffmean_layer: int,
    target_model_meta: Any,
    inactive_sampler: Any | None,
    tokenizer_nanochat: Any,
    rng: random.Random,
    apply_max_act_scale: bool,
    adv_paths: Any | None = None,
    core_prefix_id: str | None = None,
    model: Any | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    direction_payload: bytes | None = None,
) -> Intervention:
    """Central dispatcher. Per method.baseline, builds the appropriate Intervention.

    ``adv_paths`` (an ``AxBenchBpbAdvPaths`` instance) and ``core_prefix_id``
    are optional and only consulted for precomputed conditioning-bag methods.
    All other methods ignore them. Callers that know they will only score
    non-bag-backed methods may omit both.

    ``model`` is REQUIRED for every ``baseline == "ours"`` path. It is the
    authoritative source of the checkpoint feature vocabulary used to restrict
    live concept and negative-feature producers, validate precomputed-bag
    fingerprints, and assert final mapping membership. It remains optional and
    unused for non-``ours`` baselines.
    """
    mode_str = method.mode.value
    baseline = method.baseline
    if (
        direction_payload is not None
        and baseline not in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE)
    ):
        raise AxBenchBpbError(
            "direction_payload is valid only for diffmean/linear_probe methods, "
            f"got method={method.name!r}"
        )
    if baseline == BaselineKind.OURS:
        if model is None:
            raise AxBenchBpbError(
                f"the loaded target encdec model must be provided for ours "
                f"method {method.name!r}; pass model=<loaded encdec model> to "
                "build_intervention. Every ours path requires the checkpoint's "
                "authoritative feature vocabulary."
            )
        return _build_ours_intervention(
            method=method,
            concept=concept,
            coefficient=coefficient,
            source_paths=source_paths,
            target_model_meta=target_model_meta,
            inactive_sampler=inactive_sampler,
            rng=rng,
            adv_paths=adv_paths,
            core_prefix_id=core_prefix_id,
            model=model,
            cluster_contract=cluster_contract,
        )
    if baseline in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE):
        return _build_direction_intervention(
            method=method,
            concept=concept,
            coefficient=float(coefficient) if coefficient is not None else 0.0,
            source_paths=source_paths,
            diffmean_layer=diffmean_layer,
            apply_max_act_scale=apply_max_act_scale,
            cluster_contract=cluster_contract,
            direction_payload=direction_payload,
        )
    if baseline == BaselineKind.LORA:
        return _build_lora_intervention(
            concept=concept,
            mode=mode_str,
            source_paths=source_paths,
        )
    if baseline == BaselineKind.PROMPTING:
        return _build_prompting_intervention(
            concept=concept,
            mode=mode_str,
            source_paths=source_paths,
            target_model=method.target_model,
            tokenizer=tokenizer_nanochat,
        )
    if baseline == BaselineKind.CONTROL:
        # NO steering: decoder-only -> unmodified model (kind NONE, no hook);
        # encdec -> empty concept conditioning (kind OURS, feature_conditions={}),
        # so the prompt-slot conditioning still flows but no concept is injected.
        # additive composes its counter-direction on top of either via
        # `with_counter_direction` in the BPB-Adv scorer.
        if method.target_model == TargetModel.ENCDEC:
            return Intervention.ours(feature_conditions={})
        return Intervention.none()
    raise AxBenchBpbError(f"Unknown baseline {baseline!r} for method {method.name!r}")


@contextlib.contextmanager
def apply_intervention(
    model: Any,
    intervention: Intervention,
    *,
    lora: LoraParams | None = None,
) -> Iterator[None]:
    """Install the intervention on ``model`` at enter, tear down at exit.

    Guarantees teardown even if the wrapped block raises.

    Per-kind behaviour:
    - ``none``: no model hook is installed; represents the decoder-only
      no-intervention reference.
    - ``diffmean`` / ``linear_probe``: register a residual-direction hook
      at ``intervention.direction_layer``; ``.remove()`` on exit.
    - ``lora``: attach the adapter at ``intervention.lora_adapter_path``;
      call the returned ``detach_fn`` on exit. Requires the ``lora``
      bundle (``LoraParams``: ``cfg`` + ``layer``) — these are not stored
      on the Intervention since spec §10 only carries the adapter path.
    - ``ours`` / ``prompting``: no model hook is installed
      (payloads are consumed elsewhere).
    - Unknown kind: raise ``AxBenchBpbError``.

    Additionally, IF ``intervention.counter_direction`` is set (the additive
    counter-steering add-on, orthogonal to ``kind``), a residual hook
    ``counter_direction_coefficient * counter_direction`` is registered at
    ``counter_direction_layer`` IN ADDITION to the per-kind behaviour above —
    for both decoder-only and encoder-decoder models (``register_residual_hook``
    supports both). All installs tear down on exit (including on exceptions) via
    an ``ExitStack``.
    """
    kind = intervention.kind
    with contextlib.ExitStack() as stack:
        # 1. The per-kind steering intervention.
        if kind == InterventionKind.NONE:
            pass
        elif kind in (InterventionKind.DIFFMEAN, InterventionKind.LINEAR_PROBE):
            if intervention.direction is None:
                raise AxBenchBpbError(f"{kind} intervention missing direction tensor")
            if intervention.direction_layer is None:
                raise AxBenchBpbError(f"{kind} intervention missing direction_layer")
            if intervention.direction_coefficient is None:
                raise AxBenchBpbError(f"{kind} intervention missing direction_coefficient")
            handle = register_residual_hook(
                model,
                intervention.direction_layer,
                intervention.direction,
                intervention.direction_coefficient,
            )
            stack.callback(handle.remove)
        elif kind == InterventionKind.LORA:
            if intervention.lora_adapter_path is None:
                raise AxBenchBpbError("lora intervention missing lora_adapter_path")
            if lora is None:
                raise AxBenchBpbError(
                    "apply_intervention(lora) requires the lora kwarg (LoraParams)"
                )
            _, detach_fn = attach_lora_from_disk(
                model=model,
                adapter_path=intervention.lora_adapter_path,
                lora_cfg=lora.cfg,
                layer=lora.layer,
            )
            stack.callback(detach_fn)
        elif kind in (InterventionKind.OURS, InterventionKind.PROMPTING):
            pass
        else:
            raise AxBenchBpbError(f"Unknown intervention kind: {kind!r}")

        # 2. The orthogonal additive counter-direction add-on (any kind).
        if intervention.counter_direction is not None:
            if intervention.counter_direction_layer is None:
                raise AxBenchBpbError("counter_direction set but counter_direction_layer is None")
            if intervention.counter_direction_coefficient is None:
                raise AxBenchBpbError(
                    "counter_direction set but counter_direction_coefficient is None"
                )
            counter_handle = register_residual_hook(
                model,
                intervention.counter_direction_layer,
                intervention.counter_direction,
                intervention.counter_direction_coefficient,
            )
            stack.callback(counter_handle.remove)

        yield


__all__ = ["apply_intervention", "build_intervention", "normalize_direction"]
