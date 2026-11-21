"""Fair feature-GCG derive: warm-start + universal optimize over the optimize subset (Task 5).

This module ties Tasks 1–4 together and drives the real feature-GCG optimizer to
produce ONE shared conditioning bag per ``(concept_id, mode, reg_lambda)`` for the
*fair* Phase-2 BPB-Adv methods (``ours_gcg_encdec`` / ``neg_ours_gcg_encdec``).

Per ``(concept_id, mode)`` group:

1. ``split_prefixes_for_concept`` (Task 3) → an *optimize* subset and a *held-out*
   subset.
2. ``build_class_margin_members`` (Task 2) turns every optimize prefix into its
   ``(rich_i, avoid_j)`` :class:`ClassMarginMember`s.
3. ``feature_diffmean_bag`` (Task 4) computes the percentile-space feature-DiffMean
   *warm-start* over the optimize subset's completions (SAE forward on the base
   Gemma model; the nanochat completion tokens are decoded to text first — the
   convention Task 4 documents).
4. For each ``reg_lambda`` in ``cfg.ours_gcg.reg_lambda_grid``,
   ``optimize_feature_conditioning_universal`` GCG-optimizes a universal bag from
   the warm-start, minimizing the class-margin ``g_steer`` jointly over the
   optimize members.  The bag is cached at ``paths.ours_gcg_bag(...,"fair")``.
5. The held-out members' ``g_steer`` under the optimized bag is computed and
   recorded in the bag's safetensors metadata — a generalization measure, NOT
   optimized on.

Sign correctness
----------------
``mean_g_steer`` and the optimizer's objective both come from
:func:`class_margin_objective` with the SAME group ``mode``.  ``g_steer =
loss_desired − loss_contrast = −primary_margin_y``; minimizing it STRENGTHENS
steering (spec §7.2).  Passing the wrong ``mode`` here swaps desired/contrast and
the held-out margin moves the wrong way (pinned by the sign trap in
``tests/axbench_bpb_adv/integration/test_gcg_bags_fair.py``).

SAE-identity guard (carried from the Task-4 review)
---------------------------------------------------
The encdec's ``feature_embed`` is RAW-FID-INDEXED with no remap, so the
warm-start bag's feature ids are only valid if the DiffMean/extraction SAE
(``cfg.neuronpedia_source`` / ``cfg.neuronpedia_model_id``) IS the encdec's
training SAE.  A same-width-DIFFERENT-SAE misconfig silently corrupts every
result, so :func:`assert_sae_matches_encdec` compares the structured SAE identity
recorded at ``model._loaded_meta_data["user_config"]["sae"]`` against the
BPB-Adv config and raises loudly on any mismatch.  Call it before the derive runs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch as t
from loguru import logger

from steering.conditional.feature_clusters import (
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_gcg.config import (
    DEFAULT_MAX_FORWARD_BATCH,
    DEFAULT_OBJECTIVE_ALPHA,
    FeatureGcgConfig,
)
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.objective import (
    ClassMarginItem,
    ClassMarginMember,
    FeatureObjective,
    class_margin_objective,
)
from steering.conditional.feature_gcg.optimize import (
    optimize_feature_conditioning_universal,
)
from steering.eval.axbench_base.config import bare_model_id
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb.vocab_restriction import (
    checkpoint_vocab_fingerprint,
    feature_vocab_allowed_ids,
)
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import (
    DEFAULT_MEAN_G_STEER_BATCH,
    AxBenchBpbAdvConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.feature_diffmean import (
    PrefixedCompletion,
    encdec_dec_sequence_len,
    feature_diffmean_bag,
)
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)
from steering.eval.axbench_bpb_adv.gcg_family import (
    FAIR_GCG_METHOD_MODES,
    ROOFLINE_GCG_METHOD_MODES,
    GcgFamily,
    active_gcg_modes,
)
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.axbench_bpb_adv.paths import (
    AxBenchBpbAdvPaths,
    format_objective_alpha,
    format_reg_lambda,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    VOCAB_FINGERPRINT_META_KEY,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    write_ours_gcg_bag as _write_ours_gcg_bag,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    write_ours_gcg_scored_prefixes as _write_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_identity import (
    BAG_DERIVE_SCHEMA_VERSION,
    BAG_DERIVE_SCHEMA_VERSION_KEY,
    committed_bag_for_reuse,
)
from steering.eval.axbench_bpb_adv.prefix_split import (
    split_prefixes_for_concept,
)
from steering.eval.axbench_bpb_adv.progress import StageProgress as _DeriveProgress
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContractError,
    assert_cluster_contract_loaded_state,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.feature_paths import parse_feature_dir
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    SAERelease,
    coerce_activation_type,
    coerce_sae_width,
    parse_neuronpedia_source,
    resolve_sae,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

# ─────────────────────────────────────────────────────────────────────────────
# Method-activation predicates (consumed by the BUILD_CANDIDATES wiring)
# ─────────────────────────────────────────────────────────────────────────────


def active_fair_gcg_modes(cfg: AxBenchBpbAdvConfig) -> set[SteeringMode]:
    """Return the steering modes whose fair feature-GCG method is active.

    ``cfg.base_methods is None`` means *all* registered methods are active (the
    method whitelist is disabled), so both fair methods derive.  Otherwise only
    the listed method names count.
    """
    return active_gcg_modes(cfg.base_methods, FAIR_GCG_METHOD_MODES)


def fair_gcg_methods_active(cfg: AxBenchBpbAdvConfig) -> bool:
    """True iff a fair feature-GCG method (``ours_gcg_encdec`` / neg) is active."""
    return bool(active_fair_gcg_modes(cfg))


def active_roofline_gcg_modes(cfg: AxBenchBpbAdvConfig) -> set[SteeringMode]:
    """Return the steering modes whose roofline feature-GCG method is active.

    The sibling of :func:`active_fair_gcg_modes` for the per-prefix in-sample
    upper-bound methods (``ours_gcg_roofline_encdec`` /
    ``neg_ours_gcg_roofline_encdec``).  ``cfg.base_methods is None`` means *all*
    registered methods are active, so both roofline methods derive; otherwise only
    the listed method names count.
    """
    return active_gcg_modes(cfg.base_methods, ROOFLINE_GCG_METHOD_MODES)


def roofline_gcg_methods_active(cfg: AxBenchBpbAdvConfig) -> bool:
    """True iff a roofline feature-GCG method (``ours_gcg_roofline_encdec`` / neg) is active."""
    return bool(active_roofline_gcg_modes(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# SAE-identity guard
# ─────────────────────────────────────────────────────────────────────────────


def assert_sae_matches_encdec(
    cfg: AxBenchBpbAdvConfig,
    model: Any,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    """Raise if the BPB-Adv DiffMean/extraction SAE is not the encdec's training SAE.

    The warm-start bag's feature ids are emitted by the SAE loaded from
    ``cfg.neuronpedia_source`` / ``cfg.neuronpedia_model_id`` and indexed RAW into
    the encdec ``feature_embed``.  A same-width-DIFFERENT-SAE misconfig (e.g. a
    different layer / release / activation / base model) keeps every shape valid
    yet silently corrupts every bag, so the structured SAE identity recorded on
    the checkpoint MUST match.

    Compares the canonical ``(model, release, activation_type, layer, width)``
    tuple parsed from the BPB-Adv config against the encdec checkpoint's
    ``model._loaded_meta_data["user_config"]["sae"]`` block, plus a defensive
    ``encoder.config.n_sae_features`` width cross-check.

    Raises
    ------
    AxBenchBpbAdvError
        On a missing/malformed ``user_config["sae"]`` block, an unparseable
        ``cfg.neuronpedia_source``, an SAE-identity mismatch, or a width /
        ``n_sae_features`` disagreement.
    """
    try:
        bpb_release, bpb_act, bpb_layer, bpb_width, bpb_l0 = parse_neuronpedia_source(
            cfg.neuronpedia_source
        )
    except ValueError as exc:
        raise AxBenchBpbAdvError(
            f"cannot parse cfg.neuronpedia_source {cfg.neuronpedia_source!r} for "
            f"the SAE-identity guard: {exc}"
        ) from exc
    bpb_model_source = "neuronpedia_model_id"
    bpb_model = str(cfg.neuronpedia_model_id)
    if cluster_contract is not None:
        feature_extraction_model_id = cfg.feature_extraction_model_id
        if feature_extraction_model_id:
            bpb_model_source = "feature_extraction_model_id"
            bpb_model = str(feature_extraction_model_id)

    meta = getattr(model, "_loaded_meta_data", None)
    if meta is None:
        raise AxBenchBpbAdvError(
            "encdec model lacks _loaded_meta_data; cannot verify the warm-start "
            "fids are valid for this checkpoint. Attach meta_data after "
            "build_encdec_model(): `model._loaded_meta_data = meta_data`."
        )
    if cluster_contract is None:
        reject_feature_clusters_metadata(
            meta,
            "feature-GCG SAE identity checkpoint",
        )
    else:
        try:
            assert_cluster_contract_loaded_state(
                cluster_contract,
                model,
                meta,
                surface="feature-GCG SAE identity checkpoint",
            )
        except ClusterFeatureContractError as exc:
            raise AxBenchBpbAdvError(str(exc)) from exc
        try:
            resolved = resolve_sae(
                # Registry-keyed on BARE ids, while feature_extraction_model_id /
                # neuronpedia_model_id may be org-prefixed (SAELens force-prefixes
                # Gemma-3 with google/). Same class as the two sites in
                # axbench_bpb/runtime_state.py; a real clustered gemma-scope-2
                # load raised "Unsupported model 'google/gemma-3-4b-pt'" against a
                # registry list containing the bare id.
                model=bare_model_id(bpb_model),
                release=bpb_release,
                activation_type=bpb_act,
                layer=bpb_layer,
                width=bpb_width,
                l0=bpb_l0,
            )
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                "cannot resolve the clustered BPB-Adv SAE exactly: "
                f"{exc}"
            ) from exc
        configured_identity = (
            # Both sides bare-normalized: the configured id may be
            # org-prefixed (SAELens force-prefixes Gemma-3) while a
            # contract may record either form. Normalizing one side
            # only moves the mismatch.
            bare_model_id(bpb_model),
            bpb_act.value,
            bpb_layer,
            bpb_width.value,
            resolved.release,
            resolved.sae_id,
            resolved.hook_name,
        )
        contract_identity = (
            bare_model_id(str(cluster_contract.sae.model)),
            cluster_contract.sae.activation_type,
            cluster_contract.sae.layer,
            cluster_contract.sae.width,
            cluster_contract.sae.release,
            cluster_contract.sae.sae_id,
            cluster_contract.sae.hook_name,
        )
        if configured_identity != contract_identity:
            raise AxBenchBpbAdvError(
                "clustered BPB-Adv SAE identity does not match the feature "
                "contract: "
                f"configured={configured_identity!r}, "
                f"contract={contract_identity!r}"
            )
    user_cfg = meta.get("user_config") or {}
    sae_cfg = user_cfg.get("sae")
    if not isinstance(sae_cfg, Mapping):
        raise AxBenchBpbAdvError(
            "encdec user_config has no 'sae' block, so the checkpoint's training "
            "SAE cannot be verified against the BPB-Adv DiffMean SAE "
            f"({cfg.neuronpedia_source!r}). The encdec feature_embed is "
            "RAW-FID-INDEXED — refusing to derive warm-start bags whose feature "
            "ids may not correspond to this checkpoint."
        )
    try:
        enc_release = SAERelease(str(sae_cfg["release"]))
        enc_act = coerce_activation_type(sae_cfg["activation_type"])
        enc_layer = int(sae_cfg["layer"])
        enc_width = coerce_sae_width(sae_cfg["width"])
        enc_model = str(sae_cfg["model"])
    except (KeyError, ValueError, TypeError) as exc:
        raise AxBenchBpbAdvError(
            f"encdec user_config['sae'] is malformed ({dict(sae_cfg)!r}): {exc}"
        ) from exc

    # Both model ids are bare-normalized before comparison. SAELens force-prefixes
    # Gemma-3 ids, so a config naming the canonical ``google/gemma-3-4b-pt``
    # resolves prefixed, while an encdec checkpoint records the bare feature-path
    # token ``gemma-3-4b-pt``. Comparing the raw strings rejects a correctly
    # matched pair. This mirrors ``_model_names_agree`` in cluster_checkpoint_pin,
    # including its limit: an org prefix cannot be verified because the checkpoint
    # records no org, so ``other-org/gemma-3-4b-pt`` is indistinguishable here.
    # Every other SAE coordinate below is still compared exactly.
    bpb_key = (bare_model_id(bpb_model), bpb_release, bpb_act, bpb_layer, bpb_width)
    enc_key = (bare_model_id(enc_model), enc_release, enc_act, enc_layer, enc_width)
    if bpb_key != enc_key:
        raise AxBenchBpbAdvError(
            "SAE-identity mismatch: the BPB-Adv DiffMean/extraction SAE "
            f"(neuronpedia_source={cfg.neuronpedia_source!r}, "
            f"{bpb_model_source}={bpb_model!r}) resolves to {bpb_key}, which "
            "does NOT match the encdec checkpoint's recorded training SAE "
            f"{enc_key}. The encdec feature_embed is RAW-FID-INDEXED, so a "
            "different SAE silently corrupts every warm-start bag. Point "
            f"neuronpedia_source / {bpb_model_source} at the checkpoint's "
            "training SAE."
        )

    expected_n_sae = SAE_WIDTH_TO_NUM_FEATURES[bpb_width]
    try:
        n_sae = int(model.encoder.config.n_sae_features)
    except AttributeError as exc:
        raise AxBenchBpbAdvError(
            f"cannot read encoder.config.n_sae_features from the encdec model: {exc}"
        ) from exc
    if n_sae != expected_n_sae:
        raise AxBenchBpbAdvError(
            f"SAE width mismatch: encdec encoder.config.n_sae_features={n_sae} "
            f"!= the SAE width {bpb_width.value} ({expected_n_sae} features)."
        )

    # ``feature_dir`` sources the percentile_table (via feature_statistics_path);
    # its SAE layer/width must ALSO match the BPB-Adv SAE, else feature_diffmean
    # maps magnitudes through a different SAE's percentiles even when the
    # warm-start fids are valid.  Skipped only when ``feature_dir`` is unset
    # (production config validation requires it; detect_target_model_metadata also
    # parses it, so a genuinely malformed dir surfaces loudly here too).
    if cfg.feature_dir:
        parsed_dir = parse_feature_dir(Path(cfg.feature_dir))
        if (int(parsed_dir.layer), parsed_dir.width) != (bpb_layer, bpb_width):
            raise AxBenchBpbAdvError(
                "feature_dir SAE mismatch: cfg.feature_dir "
                f"(layer={parsed_dir.layer}, width={parsed_dir.width.value}) disagrees "
                f"with cfg.neuronpedia_source (layer={bpb_layer}, "
                f"width={bpb_width.value}); the percentile_table would be built from a "
                "different SAE than the warm-start feature ids."
            )


def _sae_layer_for_cfg(cfg: AxBenchBpbAdvConfig) -> int:
    """Transformer layer the warm-start SAE reads from, from the run's SAE identity.

    ``cfg.neuronpedia_source`` is the SAE's own identity string and is always
    present (``assert_sae_matches_encdec`` parses + validates it before either
    derive runs), so the layer is taken from it rather than re-parsed out of the
    SAE's TransformerLens hook name.  Threaded into ``feature_diffmean_bag`` so
    ``_max_pooled_sae_activations`` needs no hook-name layer parsing.
    """
    _release, _act, sae_layer, _width, _l0 = parse_neuronpedia_source(
        cfg.neuronpedia_source
    )
    return sae_layer


# ─────────────────────────────────────────────────────────────────────────────
# g_steer measurement (forward-only; the generalization metric)
# ─────────────────────────────────────────────────────────────────────────────


def mean_g_steer(
    model: Any,
    members: Sequence[ClassMarginMember],
    bag: FeatureConditioning,
    *,
    mode: SteeringMode,
    device: Any,
    batch_size: int = DEFAULT_MEAN_G_STEER_BATCH,
    max_forward_batch: int = DEFAULT_MAX_FORWARD_BATCH,
) -> float:
    """Mean class-margin ``g_steer`` of *bag* over *members* (lower = stronger steering).

    ``g_steer = loss_desired − loss_contrast`` per member (the single §7.2
    negation), averaged.  Forward-only.  Raises if *members* is empty (an empty
    mean is undefined) — callers never pass an empty member set: the fair derive
    raises on an empty held-out split before measuring, and both derives raise on
    a prefix that produces no members.

    ``batch_size`` bounds how many members are scored per batched forward
    (default :data:`DEFAULT_MEAN_G_STEER_BATCH`; wired to
    ``OursGcgConfig.mean_g_steer_batch`` by the fair/roofline derives).

    ``max_forward_batch`` bounds the members per sub-batched inner forward inside
    the objective (default :data:`DEFAULT_MAX_FORWARD_BATCH`; wired to
    ``OursGcgConfig.max_forward_batch`` by the fair/roofline derives so the same
    knob bounds both the GCG optimizer's inner forwards and this evaluation pass).
    """
    if not members:
        raise AxBenchBpbAdvError("mean_g_steer called with no members")
    if batch_size < 1:
        raise AxBenchBpbAdvError(f"mean_g_steer batch_size must be >= 1, got {batch_size}")
    # #720: NO compaction guard here. Both derives now restrict every bag/context they
    # condition on to the checkpoint's vocabulary V before reaching this forward-only
    # measurement (warm bags via feature_diffmean_bag(allowed_ids=...), optimized bags via
    # the V-restricted universal optimizer), so the encoder's `_to_compact` sees only ids
    # in V — no OOV. On an uncompacted encoder this is unchanged (V is the full range).
    objective = class_margin_objective(model, mode=mode, device=device, max_forward_batch=max_forward_batch)
    items = [ClassMarginItem(member=member, bag=bag) for member in members]
    # The batched objective materialises fp32 decoder logits (batch x seq x vocab)
    # in a SINGLE forward; a concept's full optimize/held-out member set (dozens of
    # rich x avoid pairs) OOMs a 24GB card. Each member conditions on its own
    # encoder input and its scalar_fn masks to its own real positions, so the loss
    # is independent per member -> chunking leaves the mean unchanged (up to the
    # documented bf16 batch-shape rounding). Matches the GCG-step minibatch size.
    values: list[float] = []
    for start in range(0, len(items), batch_size):
        values.extend(objective.losses(items[start : start + batch_size]))
    return float(sum(values) / len(values))


def _objective_factory(
    mode: SteeringMode,
    device: t.device,
    max_forward_batch: int,
    max_grad_batch: int,
    objective_alpha: float = DEFAULT_OBJECTIVE_ALPHA,
) -> Callable[[Any], FeatureObjective]:
    """One (concept, mode, alpha) group's optimizer objective factory.

    The SAME ``class_margin_objective`` (mode / device / max_forward_batch) the
    group's ``mean_g_steer`` measurements use, late-bound to the optimizer's
    model argument — defined ONCE so the fair and roofline derives can never
    drift on how the optimizer's objective is built.  ``class_margin_objective``
    is resolved from the module global at CALL time (tests monkeypatch it).
    Typed to the seam ``optimize_feature_conditioning_universal`` declares
    (``Callable[[Any], FeatureObjective]``) so a factory that stops returning a
    ``FeatureObjective`` is a type error, not a deep-in-the-loop AttributeError.

    ``objective_alpha`` is the SEARCH objective's alpha and belongs to a single
    bag, so a factory must be built per alpha inside the sweep. The reported
    ``*_g_steer`` diagnostics deliberately do NOT take it: those are the plain
    margin, kept comparable across the whole alpha sweep (a per-alpha g_steer
    would rescale the very axis the sweep is plotted against).
    """

    def factory(gcg_model: Any) -> FeatureObjective:
        return class_margin_objective(
            gcg_model,
            mode=mode,
            device=device,
            max_forward_batch=max_forward_batch,
            max_grad_batch=max_grad_batch,
            objective_alpha=objective_alpha,
        )

    return factory


def _group_g_steer(
    model: Any, ours: Any, mode: SteeringMode, device: t.device
) -> Callable[[Sequence[ClassMarginMember], FeatureConditioning], float]:
    """One (concept, mode) group's ``mean_g_steer`` measurement, knobs pre-bound.

    Binds ``mode`` / ``device`` and the batching knobs
    (``ours.mean_g_steer_batch`` / ``ours.max_forward_batch``) ONCE so every
    measurement in the group — warm / optimize / held-out / in-sample — shares
    the same binding.  Defined ONCE (used by BOTH derives) so a per-call kwarg
    bundle at each site — the copy-paste drift that would let one measurement
    run under different batching than the rest — is structurally impossible.
    """
    return functools.partial(
        mean_g_steer,
        model,
        mode=mode,
        device=device,
        batch_size=ours.mean_g_steer_batch,
        max_forward_batch=ours.max_forward_batch,
    )


def _move_module_if_possible(module: Any, device: t.device | str) -> None:
    to_method = getattr(module, "to", None)
    if callable(to_method):
        to_method(device)


def _empty_cuda_cache_if_needed(device: t.device) -> None:
    if device.type == "cuda" and t.cuda.is_available():
        # Backward hooks and their captured activation graphs can form Python
        # cycles. ``empty_cache`` alone cannot release allocator blocks whose
        # tensors are still unreachable-but-uncollected. Collect first at every
        # mutually-exclusive model swap so repeated DiffMean <-> GCG groups do
        # not grow live CUDA state until the next incidental GC cycle. Expected
        # numerical neutrality; finalizer timing on this shared clustered and
        # unclustered path is explicitly accepted.
        release_cuda_iteration_memory()


def _prepare_for_diffmean_forward(
    *,
    model: Any,
    base_model: Any,
    sae: Any,
    device: t.device,
) -> None:
    """Keep only the base Gemma+SAE extractor resident on CUDA for warm-starts."""
    if device.type != "cuda":
        return
    _move_module_if_possible(model, "cpu")
    _empty_cuda_cache_if_needed(device)
    _move_module_if_possible(base_model, device)
    _move_module_if_possible(sae, device)


def _prepare_for_gcg_forward(
    *,
    model: Any,
    base_model: Any,
    sae: Any,
    device: t.device,
) -> None:
    """Keep only the encoder-decoder resident on CUDA for GCG optimization."""
    if device.type != "cuda":
        return
    _move_module_if_possible(base_model, "cpu")
    _move_module_if_possible(sae, "cpu")
    _empty_cuda_cache_if_needed(device)
    _move_module_if_possible(model, device)


# ─────────────────────────────────────────────────────────────────────────────
# The fair derive
# ─────────────────────────────────────────────────────────────────────────────


def _group_prefixes_by_concept_mode(
    prefixes: Sequence[BpbAttackPrefix],
) -> list[tuple[int, SteeringMode, list[BpbAttackPrefix]]]:
    """Group prefixes by ``(concept_id, mode)``, preserving first-seen order."""
    groups: dict[tuple[int, str], list[BpbAttackPrefix]] = {}
    order: list[tuple[int, str]] = []
    for prefix in prefixes:
        key = (int(prefix.concept_id), SteeringMode(prefix.mode).value)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(prefix)
    return [
        (concept_id, SteeringMode(mode_value), groups[(concept_id, mode_value)])
        for concept_id, mode_value in order
    ]


def _members_for_prefixes(
    prefixes: Sequence[BpbAttackPrefix],
    *,
    device: t.device,
    context_segments_by_prefix_hash: dict[str, tuple[ContextFeatureSegment, ...]],
    tokenizer: Any,
    max_seq_len: int,
    allowed_ids: t.Tensor | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> list[ClassMarginMember]:
    """Build every prefix's class-margin members under the score-time layout.

    Each prefix's REAL context-cache segments — keyed by ``prefix.prefix_hash``
    in ``context_segments_by_prefix_hash``, the SAME dict the score stage loads
    via ``load_context_segments_by_prefix_hash`` — become the fixed per-segment
    context conditioning, and ``build_multi_segment_seg_map`` (inside
    ``build_class_margin_members``) routes every prefix token to its own segment
    slot.  So the derive optimizes each member under the identical multi-slot
    layout the score stage applies the bag under, with no GEMMA-token counting
    heuristic in between: the segmentation the score stage will actually use IS
    the input here.  ``max_seq_len`` (the encdec decoder ``sequence_len``) bounds
    each member so a prefix the score stage would left-truncate raises rather
    than optimizing under a divergent untruncated layout.  ``allowed_ids`` (the
    checkpoint's compacted vocabulary ``V``, or ``None`` for an uncompacted
    checkpoint) is threaded straight through to ``build_class_margin_members``,
    which restricts each member's FIXED context bags to ``V``; ``None`` is a
    strict no-op.

    A prefix whose hash is absent from the cache RAISES — a context cache missing
    an optimize/held-out prefix is a real pipeline inconsistency (the score stage
    would condition it, but the derive cannot), never a skip.
    """
    members: list[ClassMarginMember] = []
    for prefix in prefixes:
        if prefix.prefix_hash not in context_segments_by_prefix_hash:
            raise AxBenchBpbAdvError(
                "feature-GCG derive: no context-cache segments for prefix_hash="
                f"{prefix.prefix_hash!r} (core_prefix_id={prefix.core_prefix_id!r}); "
                "the context cache must carry an entry for every optimize/held-out "
                "prefix. A missing entry is a real pipeline inconsistency (the "
                "score stage conditions this prefix, so the derive must too), not "
                "a skippable case."
            )
        segments = context_segments_by_prefix_hash[prefix.prefix_hash]
        if cluster_contract is not None:
            for segment_idx, segment in enumerate(segments):
                segment_ids = t.tensor(segment.feature_ids, dtype=t.int64)
                cluster_contract.consolidation.assert_representatives_only(
                    segment_ids,
                    surface=(
                        "feature-GCG fixed context "
                        f"{prefix.prefix_hash} segment {segment_idx}"
                    ),
                )
                if cluster_contract.consolidation.feature_mask is not None:
                    cluster_contract.consolidation.feature_mask.assert_no_masked(
                        segment_ids
                    )
        members.extend(
            build_class_margin_members(
                prefix,
                prefix_segments=segments,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                device=device,
                # Clustered caches are already contract-validated. Filtering at
                # read time could hide a stale/non-representative producer, so
                # the clustered branch asserts above and passes through intact.
                allowed_ids=None if cluster_contract is not None else allowed_ids,
            )
        )
    return members


def _assert_prefix_tokens_roundtrip(
    prefix: BpbAttackPrefix, *, tokenizer: Any
) -> None:
    """Raise if ``prefix.prefix_tokens`` do not re-encode from ``prefix.prefix_text``.

    A text/tokens drift (corrupt parquet / wrong tokenizer) means the DiffMean
    warm-start (which conditions on the raw TEXT), the score-time segment count
    (also TEXT), and the GCG members (which embed the stored TOKENS) would
    silently describe DIFFERENT prefixes.  Applied to EVERY prefix the derive
    touches — optimize, HELD-OUT, and roofline-selected — so the fair
    generalization metric (held_out_g_steer) can never be measured on a prefix
    whose text and tokens disagree.
    """
    stored_tokens = [int(token_id) for token_id in prefix.prefix_tokens]
    expected_tokens = [
        int(token_id) for token_id in encode_prefix(prefix.prefix_text, tokenizer)
    ]
    if stored_tokens == expected_tokens:
        return
    divergence = next(
        (
            i
            for i, (stored, expected) in enumerate(
                zip(stored_tokens, expected_tokens, strict=False)
            )
            if stored != expected
        ),
        min(len(stored_tokens), len(expected_tokens)),
    )
    raise AxBenchBpbAdvError(
        "feature-GCG derive: prefix_tokens do not round-trip from "
        f"prefix_text for core_prefix_id={prefix.core_prefix_id!r} "
        f"(concept_id={prefix.concept_id}, mode={SteeringMode(prefix.mode).value}): "
        f"stored {len(stored_tokens)} token(s) vs re-encoded "
        f"{len(expected_tokens)}, first divergence at index {divergence}. "
        "The raw text (DiffMean warm-start input) and the stored tokens "
        "(GCG member input) must describe the SAME prefix."
    )


def _prefixed_completions(
    prefixes: Sequence[BpbAttackPrefix],
    *,
    tokenizer: Any,
) -> tuple[list[PrefixedCompletion], list[PrefixedCompletion]]:
    """Pair each prefix's RAW text with its decoded completion texts.

    Task 4's ``feature_diffmean_bag`` consumes :class:`PrefixedCompletion` pairs
    (it re-tokenizes with the Gemma extraction tokenizer, runs the prefix as
    context, and pools ONLY the completion span).  The prefix text is taken from
    ``prefix.prefix_text`` — the RAW text the synth pipeline stored — NEVER from
    ``tokenizer.decode(prefix.prefix_tokens)``: the stored tokens carry the
    nanochat BOS (``encode_prefix`` prepends it) and the nanochat decode
    preserves special tokens, so a decoded prefix starts with a literal
    ``'<|bos|>'`` string.  Under the REAL nanochat tokenizer that literal
    renders as ``'<bos>'``, which the Gemma extraction tokenizer re-encodes
    into a REAL BOS id even with ``add_special_tokens=False`` — the SAE
    forward would then silently run on a DOUBLE-BOS sequence (the prefix-match
    check cannot fire: both sides see the same polluted text).  Completions
    exist only as nanochat token lists (no stored text), so they ARE decoded
    here; completions never carry a BOS.

    Each completion is paired with ITS OWN prefix's text.  Every prefix is
    tokens<->text round-trip checked via :func:`_assert_prefix_tokens_roundtrip`
    (the caller also checks held-out prefixes the same way).

    Raises:
        AxBenchBpbAdvError: any ``prefix.prefix_tokens`` does not round-trip from
            ``prefix.prefix_text`` under *tokenizer*.
    """
    rich_items: list[PrefixedCompletion] = []
    avoid_items: list[PrefixedCompletion] = []
    for prefix in prefixes:
        _assert_prefix_tokens_roundtrip(prefix, tokenizer=tokenizer)
        rich_items.extend(
            PrefixedCompletion(
                prefix_text=prefix.prefix_text,
                completion_text=tokenizer.decode(tokens),
            )
            for tokens in prefix.concept_rich_output_tokens
        )
        avoid_items.extend(
            PrefixedCompletion(
                prefix_text=prefix.prefix_text,
                completion_text=tokenizer.decode(tokens),
            )
            for tokens in prefix.concept_avoiding_output_tokens
        )
    return rich_items, avoid_items


@dataclass(frozen=True)
class _ContractBoundWarmStart:
    """One warm bag plus the contract fingerprint that derived it."""

    bag: FeatureConditioning
    source_feature_contract_fingerprint: str


_CLUSTERED_WARM_START_CACHE_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class ClusteredWarmStartCache:
    """Immutable clustered DiffMean bags shared by every precomputed family.

    A real 65k Gemma SAE and Gemma-3-4B nearly fill a 24 GiB device. Alternating
    them with GCG forwards can also retain CUDA-library state outside PyTorch's
    allocator, so later extractor uploads needlessly fail despite every live
    tensor being offloaded. Clustered production therefore computes all warm
    starts while the extractor is the sole large CUDA owner, retires it once,
    and then reuses these exact bags for fair/roofline DiffMean and GCG.

    Construction is reserved for :func:`build_clustered_warm_start_cache`.
    Every private mapping value retains the fingerprint present when its bag was
    derived, independently of the cache-level label. This makes a routine
    ``dataclasses.replace`` relabel fail closed when a bag is requested.
    """

    feature_contract_fingerprint: str
    _fair: Mapping[tuple[int, str], _ContractBoundWarmStart]
    _roofline: Mapping[tuple[int, str, str], _ContractBoundWarmStart]
    _factory_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _CLUSTERED_WARM_START_CACHE_FACTORY_TOKEN:
            raise TypeError(
                "ClusteredWarmStartCache construction is factory-only; use "
                "build_clustered_warm_start_cache"
            )
        if not all(
            isinstance(bound, _ContractBoundWarmStart)
            for bound in (*self._fair.values(), *self._roofline.values())
        ):
            raise TypeError(
                "ClusteredWarmStartCache mappings must contain factory-bound warm bags"
            )
        object.__setattr__(self, "_fair", MappingProxyType(dict(self._fair)))
        object.__setattr__(self, "_roofline", MappingProxyType(dict(self._roofline)))

    def assert_matches(
        self,
        contract: ClusterFeatureContract | None,
        *,
        surface: str,
    ) -> None:
        if contract is None:
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start cache requires a validated "
                "cluster feature contract"
            )
        actual = contract.feature_contract_fingerprint
        if actual != self.feature_contract_fingerprint:
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start cache feature-contract "
                f"fingerprint mismatch: expected={actual}, "
                f"actual={self.feature_contract_fingerprint}"
            )

    def fair_bag(
        self,
        concept_id: int,
        mode: SteeringMode,
        *,
        contract: ClusterFeatureContract | None,
        surface: str,
    ) -> FeatureConditioning:
        self.assert_matches(contract, surface=surface)
        key = (int(concept_id), SteeringMode(mode).value)
        try:
            bound_bag = self._fair[key]
        except KeyError as exc:
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start cache is missing fair key {key!r}"
            ) from exc
        return self._validated_bag(
            bound_bag,
            contract=contract,
            surface=surface,
        )

    def roofline_bag(
        self,
        concept_id: int,
        mode: SteeringMode,
        prefix_id: str,
        *,
        contract: ClusterFeatureContract | None,
        surface: str,
    ) -> FeatureConditioning:
        self.assert_matches(contract, surface=surface)
        key = (int(concept_id), SteeringMode(mode).value, str(prefix_id))
        try:
            bound_bag = self._roofline[key]
        except KeyError as exc:
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start cache is missing roofline key {key!r}"
            ) from exc
        return self._validated_bag(
            bound_bag,
            contract=contract,
            surface=surface,
        )

    def _validated_bag(
        self,
        bound_bag: _ContractBoundWarmStart,
        *,
        contract: ClusterFeatureContract | None,
        surface: str,
    ) -> FeatureConditioning:
        # This repeats the cross-contract barrier independently of the public
        # lookup methods. A future refactor that omits ``assert_matches`` must
        # still be unable to return a bag under a different contract.
        if contract is None:
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start cache has no validated contract"
            )
        requested_fingerprint = contract.feature_contract_fingerprint
        bag_fingerprint = bound_bag.source_feature_contract_fingerprint
        cache_fingerprint = self.feature_contract_fingerprint
        if len(
            {
                bag_fingerprint,
                cache_fingerprint,
                requested_fingerprint,
            }
        ) != 1:
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start bag/cache/request feature-contract "
                "fingerprints disagree at bag validation: "
                f"bag_bound={bag_fingerprint}, cache={cache_fingerprint}, "
                f"requested={requested_fingerprint}"
            )
        bag = bound_bag.bag
        feature_ids = t.tensor(bag.feature_ids(), dtype=t.int64)
        contract.consolidation.assert_representatives_only(
            feature_ids,
            surface=f"{surface} cached warm start",
        )
        if contract.consolidation.feature_mask is not None:
            contract.consolidation.feature_mask.assert_no_masked(feature_ids)
        if any(bucket >= contract.selection.num_buckets for _fid, bucket in bag.entries):
            raise AxBenchBpbAdvError(
                f"{surface}: clustered warm-start cache contains an out-of-range bucket"
            )
        return bag


def build_clustered_warm_start_cache(
    *,
    prefixes: Sequence[BpbAttackPrefix],
    model: Any,
    base_model: Any,
    sae: Any,
    percentile_table: t.Tensor,
    cfg: AxBenchBpbAdvConfig,
    device: Any,
    tokenizer: Any,
    context_segments_by_prefix_hash: dict[
        str, tuple[ContextFeatureSegment, ...]
    ],
    fair_modes: set[SteeringMode],
    roofline_modes: set[SteeringMode],
    cluster_contract: ClusterFeatureContract,
) -> ClusteredWarmStartCache:
    """Compute every clustered precomputed-family warm start in one CUDA phase."""

    torch_device = t.device(device) if not isinstance(device, t.device) else device
    positive_bucket, negative_bucket = assert_single_positive_bucket(
        model,
        cluster_contract=cluster_contract,
    )
    k_pos, k_neg = resolve_k(
        model,
        cfg.ours_gcg,
        cluster_contract=cluster_contract,
    )
    sae_layer = _sae_layer_for_cfg(cfg)
    allowed_ids = feature_vocab_allowed_ids(model)
    max_seq_len = encdec_dec_sequence_len(model)
    groups = _group_prefixes_by_concept_mode(prefixes)
    fair_tasks: list[
        tuple[
            tuple[int, str],
            SteeringMode,
            list[PrefixedCompletion],
            list[PrefixedCompletion],
        ]
    ] = []
    roofline_tasks: list[
        tuple[
            tuple[int, str, str],
            SteeringMode,
            list[PrefixedCompletion],
            list[PrefixedCompletion],
        ]
    ] = []

    # Validate and materialize every text pair before the first model forward.
    # A malformed later group must not leave an expensive partially computed
    # cache that looks reusable to the caller.
    for concept_id, mode, group_prefixes in groups:
        if mode not in fair_modes and mode not in roofline_modes:
            continue
        optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
            group_prefixes,
            ratio=cfg.ours_gcg.split_ratio,
            seed=cfg.ours_gcg.split_seed,
        )
        if not held_out_prefixes:
            raise AxBenchBpbAdvError(
                "clustered warm-start precompute selected no held-out prefixes for "
                f"concept_id={concept_id} mode={mode.value}"
            )
        # Validate every fixed context, representative set, and score-time
        # sequence layout on CPU before the first expensive SAE forward. The
        # derives rebuild these small members on the target device later; this
        # preflight exists so a stale later group cannot consume extractor work
        # and fail only after some warm starts have already been computed.
        held_out_members = _members_for_prefixes(
            held_out_prefixes,
            device=t.device("cpu"),
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            allowed_ids=allowed_ids,
            cluster_contract=cluster_contract,
        )
        if not held_out_members:
            raise AxBenchBpbAdvError(
                "clustered warm-start precompute produced no held-out members for "
                f"concept_id={concept_id} mode={mode.value}"
            )
        if mode in fair_modes:
            if not optimize_prefixes:
                raise AxBenchBpbAdvError(
                    "clustered warm-start precompute selected no optimize prefixes for "
                    f"concept_id={concept_id} mode={mode.value}"
                )
            optimize_members = _members_for_prefixes(
                optimize_prefixes,
                device=t.device("cpu"),
                context_segments_by_prefix_hash=context_segments_by_prefix_hash,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                allowed_ids=allowed_ids,
                cluster_contract=cluster_contract,
            )
            if not optimize_members:
                raise AxBenchBpbAdvError(
                    "clustered warm-start precompute produced no optimize members for "
                    f"concept_id={concept_id} mode={mode.value}"
                )
            rich_items, avoid_items = _prefixed_completions(
                optimize_prefixes,
                tokenizer=tokenizer,
            )
            fair_tasks.append(
                (
                    (int(concept_id), mode.value),
                    mode,
                    rich_items,
                    avoid_items,
                )
            )
        if mode in roofline_modes:
            for prefix in held_out_prefixes:
                rich_items, avoid_items = _prefixed_completions(
                    [prefix],
                    tokenizer=tokenizer,
                )
                roofline_tasks.append(
                    (
                        (int(concept_id), mode.value, str(prefix.core_prefix_id)),
                        mode,
                        rich_items,
                        avoid_items,
                    )
                )

    if not fair_tasks and not roofline_tasks:
        raise AxBenchBpbAdvError(
            "clustered warm-start precompute received no active fair or roofline tasks"
        )

    fair_bags: dict[tuple[int, str], FeatureConditioning] = {}
    roofline_bags: dict[tuple[int, str, str], FeatureConditioning] = {}
    try:
        # Put the ownership transition inside the protected region too. A
        # partial extractor upload (for example, Gemma moves successfully but
        # the SAE upload raises) must still retire whichever extractor modules
        # moved and restore the target before this function unwinds.
        _prepare_for_diffmean_forward(
            model=model,
            base_model=base_model,
            sae=sae,
            device=torch_device,
        )
        for key, mode, rich_items, avoid_items in fair_tasks:
            if key in fair_bags:
                raise AxBenchBpbAdvError(
                    f"clustered warm-start precompute produced duplicate fair key {key!r}"
                )
            fair_bags[key] = feature_diffmean_bag(
                rich_items,
                avoid_items,
                sae=sae,
                model=base_model,
                sae_layer=sae_layer,
                percentile_table=percentile_table,
                mode=mode,
                k_pos=k_pos,
                k_neg=k_neg,
                positive_bucket=positive_bucket,
                negative_bucket=negative_bucket,
                device=torch_device,
                extraction_batch_size=cfg.feature_diffmean_extraction_batch_size,
                allowed_ids=allowed_ids,
                cluster_contract=cluster_contract,
            )
        for key, mode, rich_items, avoid_items in roofline_tasks:
            if key in roofline_bags:
                raise AxBenchBpbAdvError(
                    f"clustered warm-start precompute produced duplicate roofline key {key!r}"
                )
            roofline_bags[key] = feature_diffmean_bag(
                rich_items,
                avoid_items,
                sae=sae,
                model=base_model,
                sae_layer=sae_layer,
                percentile_table=percentile_table,
                mode=mode,
                k_pos=k_pos,
                k_neg=k_neg,
                positive_bucket=positive_bucket,
                negative_bucket=negative_bucket,
                device=torch_device,
                extraction_batch_size=cfg.feature_diffmean_extraction_batch_size,
                allowed_ids=allowed_ids,
                cluster_contract=cluster_contract,
            )
    finally:
        # This is the sole large-owner transition in clustered production: once
        # all warm starts exist, Gemma+SAE stay on CPU for every subsequent
        # DiffMean measurement and GCG optimize.
        _prepare_for_gcg_forward(
            model=model,
            base_model=base_model,
            sae=sae,
            device=torch_device,
        )

    logger.info(
        "clustered warm-start precompute complete: fair={} roofline={} "
        "feature_contract_fingerprint={}",
        len(fair_bags),
        len(roofline_bags),
        cluster_contract.feature_contract_fingerprint,
    )
    source_fingerprint = cluster_contract.feature_contract_fingerprint
    return ClusteredWarmStartCache(
        feature_contract_fingerprint=source_fingerprint,
        _fair={
            key: _ContractBoundWarmStart(
                bag=bag,
                source_feature_contract_fingerprint=source_fingerprint,
            )
            for key, bag in fair_bags.items()
        },
        _roofline={
            key: _ContractBoundWarmStart(
                bag=bag,
                source_feature_contract_fingerprint=source_fingerprint,
            )
            for key, bag in roofline_bags.items()
        },
        _factory_token=_CLUSTERED_WARM_START_CACHE_FACTORY_TOKEN,
    )


def _lambda_alpha_grid(ours: Any) -> list[tuple[float, float]]:
    """The ``(reg_lambda, objective_alpha)`` product a derive sweeps.

    ``reg_lambda``-major, so the default ``objective_alpha_grid == (1.0,)``
    degenerates to exactly the old ``for reg_lambda in ours.reg_lambda_grid``
    sequence — same bags, same order, same progress totals.

    Named rather than inlined because FOUR call sites need it: each derive's
    loop and each derive's ``_DeriveProgress`` total. A progress total that
    disagreed with its own loop is exactly the drift this prevents.
    """
    return [
        (float(reg_lambda), float(objective_alpha))
        for reg_lambda, objective_alpha in product(
            ours.reg_lambda_grid, ours.objective_alpha_grid
        )
    ]


def _stamp_objective_alpha(identity: dict[str, str], objective_alpha: float) -> None:
    """Stamp the class-margin alpha into ``identity`` — ONLY when non-default.

    Strip-default, deliberately. At ``alpha == 1`` the minimized margin
    ``loss_desired - alpha * loss_contrast`` is arithmetically the plain margin
    every pre-alpha derive used, so an alpha = 1 bag IS the artifact a pre-alpha
    derive would have written. Stamping the key unconditionally would make every
    bag on disk fail its reuse check for a "missing identity key" and force a
    multi-day re-derive that could only reproduce the same bytes -- the exact
    failure mode ``max_grad_batch`` caused when it was added unconditionally.

    The converse still holds where it matters: any alpha != 1 DOES stamp, so a
    bag optimized under a different objective can never be silently reused for
    the plain margin (or for a different alpha).
    """
    if float(objective_alpha) != DEFAULT_OBJECTIVE_ALPHA:
        identity["objective_alpha"] = format_objective_alpha(objective_alpha)


def _roofline_bag_identity(
    *,
    cfg: Any,
    ours: Any,
    concept_id: int,
    mode: SteeringMode,
    reg_lambda: float,
    objective_alpha: float,
    prefix_id: Any,
    prefix_members: Sequence[Any],
    rich_items: Sequence[Any],
    avoid_items: Sequence[Any],
    k_pos: int,
    k_neg: int,
    positive_bucket: int,
    negative_bucket: int,
    cluster_contract: ClusterFeatureContract | None,
) -> dict[str, str]:
    """The input-derived half of a roofline feature-GCG bag's metadata.

    Roofline is keyed one level deeper than fair: per ``(concept, mode, prefix,
    reg_lambda)``, so ``prefix_id`` is part of the identity. Omitting it would let a
    bag derived for one prefix be reused for another.

    ``in_sample_g_steer`` is roofline's measured diagnostic (fair uses
    ``optimize_g_steer`` / ``held_out_g_steer``) and is added after the derive.
    """

    identity = {
        "concept_id": str(int(concept_id)),
        "mode": mode.value,
        "mode_tag": GcgFamily.ROOFLINE,
        "prefix_id": str(prefix_id),
        "reg_lambda": format_reg_lambda(reg_lambda),
        "n_prefix_members": str(len(prefix_members)),
        "n_rich_completions": str(len(rich_items)),
        "n_avoid_completions": str(len(avoid_items)),
        "k_pos": str(int(k_pos)),
        "k_neg": str(int(k_neg)),
        "positive_bucket": str(int(positive_bucket)),
        "negative_bucket": str(int(negative_bucket)),
        "extraction_batch_size": str(cfg.feature_diffmean_extraction_batch_size),
        "master_seed": str(int(ours.master_seed)),
        "split_seed": str(int(ours.split_seed)),
        "split_ratio": repr(float(ours.split_ratio)),
        "scored_split": "held_out",
        "max_steps": str(int(ours.max_steps)),
        "minibatch_size": str(int(ours.minibatch_size)),
        # A derive input, not merely a memory knob: it is bound into
        # class_margin_objective via _objective_factory, and gradient.py documents
        # batch-shape-dependent bf16 rounding. Two derives at different values
        # therefore explore different trajectories, so a bag from one must never be
        # reused for the other. No legacy default: clustered GCG resume always failed
        # before this change (that IS #1050), so no working run_tag depends on a bag
        # written without this key.
        "max_forward_batch": str(int(ours.max_forward_batch)),
        # Stamped for the SAME reason as max_forward_batch (#1054): slicing the
        # fwd+bwd changes the batch shape, and under the real model's GPU bf16
        # that carries the documented batch-shape kernel rounding. A rounding
        # difference can flip a GCG argmax and thus the derived bag, so a resumed
        # run must NOT reuse a bag built under a different value.
        "max_grad_batch": str(int(ours.max_grad_batch)),
    }
    _stamp_objective_alpha(identity, objective_alpha)
    if cluster_contract is not None:
        identity["feature_contract_fingerprint"] = (
            cluster_contract.feature_contract_fingerprint
        )
        identity[BAG_DERIVE_SCHEMA_VERSION_KEY] = BAG_DERIVE_SCHEMA_VERSION
    return identity


def _fair_bag_identity(
    *,
    cfg: Any,
    ours: Any,
    concept_id: int,
    mode: SteeringMode,
    reg_lambda: float,
    objective_alpha: float,
    optimize_prefixes: Sequence[Any],
    held_out_prefixes: Sequence[Any],
    optimize_members: Sequence[Any],
    held_out_members: Sequence[Any],
    rich_items: Sequence[Any],
    avoid_items: Sequence[Any],
    k_pos: int,
    k_neg: int,
    positive_bucket: int,
    negative_bucket: int,
    cluster_contract: ClusterFeatureContract | None,
) -> dict[str, str]:
    """The input-derived half of a fair feature-GCG bag's metadata.

    Split out from the write site so ONE definition serves both the writer and the
    resume-time reuse check (#1050); a second, weaker copy of these keys is exactly
    how a reader ends up validating less than its writer stamped.

    Every value here is computable before the GCG run. ``n_features`` and the measured
    ``*_g_steer`` diagnostics are deliberately absent: the first is result-derived and
    the others are the nondeterministic quantities that made byte-comparison
    unworkable.

    The schema-version stamp is added only on the clustered branch. Adding it
    unconditionally would change unclustered bag bytes, and the unclustered path is
    required to stay byte-identical.
    """

    identity = {
        "concept_id": str(int(concept_id)),
        "mode": mode.value,
        "mode_tag": GcgFamily.FAIR,
        "reg_lambda": format_reg_lambda(reg_lambda),
        "n_optimize_prefixes": str(len(optimize_prefixes)),
        "n_held_out_prefixes": str(len(held_out_prefixes)),
        "n_optimize_members": str(len(optimize_members)),
        "n_held_out_members": str(len(held_out_members)),
        "n_rich_completions": str(len(rich_items)),
        "n_avoid_completions": str(len(avoid_items)),
        "k_pos": str(int(k_pos)),
        "k_neg": str(int(k_neg)),
        "positive_bucket": str(int(positive_bucket)),
        "negative_bucket": str(int(negative_bucket)),
        "extraction_batch_size": str(cfg.feature_diffmean_extraction_batch_size),
        "master_seed": str(int(ours.master_seed)),
        "split_ratio": repr(float(ours.split_ratio)),
        "split_seed": str(int(ours.split_seed)),
        "max_steps": str(int(ours.max_steps)),
        "minibatch_size": str(int(ours.minibatch_size)),
        # A derive input, not merely a memory knob: it is bound into
        # class_margin_objective via _objective_factory, and gradient.py documents
        # batch-shape-dependent bf16 rounding. Two derives at different values
        # therefore explore different trajectories, so a bag from one must never be
        # reused for the other. No legacy default: clustered GCG resume always failed
        # before this change (that IS #1050), so no working run_tag depends on a bag
        # written without this key.
        "max_forward_batch": str(int(ours.max_forward_batch)),
        # Stamped for the SAME reason as max_forward_batch (#1054): slicing the
        # fwd+bwd changes the batch shape, and under the real model's GPU bf16
        # that carries the documented batch-shape kernel rounding. A rounding
        # difference can flip a GCG argmax and thus the derived bag, so a resumed
        # run must NOT reuse a bag built under a different value.
        "max_grad_batch": str(int(ours.max_grad_batch)),
    }
    _stamp_objective_alpha(identity, objective_alpha)
    if cluster_contract is not None:
        identity["feature_contract_fingerprint"] = (
            cluster_contract.feature_contract_fingerprint
        )
        identity[BAG_DERIVE_SCHEMA_VERSION_KEY] = BAG_DERIVE_SCHEMA_VERSION
    return identity


def build_ours_gcg_bags_fair(
    *,
    prefixes: Sequence[BpbAttackPrefix],
    model: Any,
    base_model: Any,
    sae: Any,
    percentile_table: t.Tensor,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    device: Any,
    tokenizer: Any,
    context_segments_by_prefix_hash: dict[str, tuple[ContextFeatureSegment, ...]],
    cluster_contract: ClusterFeatureContract | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache the fair feature-GCG bags for every ``(concept, mode, λ)``.

    For each ``(concept_id, mode)`` group present in *prefixes*: split into an
    optimize / held-out subset (Task 3), build the optimize members (Task 2),
    compute the feature-DiffMean warm-start over the optimize completions
    (Task 4), then for each ``reg_lambda`` GCG-optimize a universal bag from the
    warm-start and write it to ``paths.ours_gcg_bag(concept, mode, λ, "fair")``.
    The held-out members' ``g_steer`` under the optimized bag is recorded in the
    bag metadata (a generalization measure; never optimized on).

    Parameters
    ----------
    prefixes:
        Core ``BpbAttackPrefix`` objects, ALREADY filtered to the modes whose
        fair method is active (the caller drops inactive modes).
    model:
        The grad-enabled encdec model the GCG optimizer scores against.  Must
        carry ``_loaded_meta_data`` (for ``resolve_k`` / ``assert_single_positive_bucket``).
    base_model:
        The base Gemma model for the warm-start SAE forward (Task 4).
    sae:
        The base SAE matching ``cfg.neuronpedia_source`` (validated by
        :func:`assert_sae_matches_encdec`).
    percentile_table:
        ``[n_sae_features, 99]`` percentile table from
        ``build_percentile_table_from_stats``.
    cfg, paths, device, tokenizer:
        Run config, artifact paths, torch device, and the nanochat tokenizer.
        The tokenizer decodes completion tokens for the warm-start AND aligns
        each prefix's tokens to its context-segment char-spans when building the
        per-member seg_map (``build_multi_segment_seg_map``).
    context_segments_by_prefix_hash:
        The prefix-hash → context-cache segments map the score stage also loads
        (``load_context_segments_by_prefix_hash``).  Supplies each member's fixed
        per-segment context conditioning and the multi-segment seg_map, so the
        derive optimizes under the SAME layout scoring applies the bag under.
        Every optimize/held-out prefix's hash MUST be present (a missing entry
        raises).
    """
    torch_device = t.device(device) if not isinstance(device, t.device) else device
    cluster_kwargs = (
        {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    )
    if clustered_warm_starts is not None:
        clustered_warm_starts.assert_matches(
            cluster_contract,
            surface="fair feature-GCG derive",
        )
    positive_bucket, negative_bucket = assert_single_positive_bucket(
        model,
        **cluster_kwargs,
    )
    k_pos, k_neg = resolve_k(model, cfg.ours_gcg, **cluster_kwargs)
    sae_layer = _sae_layer_for_cfg(cfg)
    ours = cfg.ours_gcg
    # The score stage left-truncates each prefix to the encdec decoder's
    # sequence_len before teacher-forcing; the derive builds members on the
    # untruncated prefix and RAISES (inside build_class_margin_members) if a
    # prefix + completion would exceed this window, so the optimized layout can
    # never silently diverge from the scored one.
    max_seq_len = encdec_dec_sequence_len(model)
    # A compacted target checkpoint's allowed SAE feature vocabulary V (None on
    # an uncompacted checkpoint) -- restricts every member's FIXED context bags
    # (_members_for_prefixes -> build_class_margin_members) and the warm
    # feature_diffmean_bag selection to V, so neither can condition on / select
    # an id that trips the encoder's OOV assert.
    allowed_ids = feature_vocab_allowed_ids(model)

    groups = _group_prefixes_by_concept_mode(prefixes)
    logger.info(
        f"fair feature-GCG derive: {len(groups)} (concept,mode) group(s), "
        f"reg_lambda_grid={list(ours.reg_lambda_grid)}, "
        f"objective_alpha_grid={list(ours.objective_alpha_grid)}, k_pos={k_pos} "
        f"k_neg={k_neg} buckets=({positive_bucket},{negative_bucket})"
    )
    progress = _DeriveProgress(
        "fair feature-GCG derive", len(groups) * len(_lambda_alpha_grid(ours)), "bag"
    )

    for concept_id, mode, group_prefixes in groups:
        optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
            group_prefixes, ratio=ours.split_ratio, seed=ours.split_seed
        )
        if not held_out_prefixes:
            # OursGcgConfig bounds split_ratio < 1, which makes an empty held-out
            # arithmetically impossible for the current int(n * ratio) split —
            # this backstop exists so a future split-function change can never
            # regress to burning the full GCG budget and then persisting an
            # *empty* fair scored set (a silent-zero-results run).
            raise AxBenchBpbAdvError(
                "fair feature-GCG derive produced no HELD-OUT prefixes for "
                f"concept_id={concept_id} mode={mode.value} "
                f"(n_prefixes={len(group_prefixes)}, split_ratio={ours.split_ratio}); "
                "the held-out remainder IS the fair scored set, so an empty one "
                "would silently drop every fair-gcg score downstream."
            )
        # Round-trip check the HELD-OUT prefixes too (optimize prefixes are checked
        # inside _prefixed_completions): held_out_g_steer — the fair generalization
        # metric — is measured on these, so their text and tokens must agree.
        for prefix in held_out_prefixes:
            _assert_prefix_tokens_roundtrip(prefix, tokenizer=tokenizer)
        optimize_members = _members_for_prefixes(
            optimize_prefixes,
            device=torch_device,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            allowed_ids=allowed_ids,
            **cluster_kwargs,
        )
        held_out_members = _members_for_prefixes(
            held_out_prefixes,
            device=torch_device,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            allowed_ids=allowed_ids,
            **cluster_kwargs,
        )
        if not optimize_members:
            # An empty optimize split is a real config/data problem (e.g. a
            # split_ratio so small that int(n_prefixes * ratio) == 0), not a routine
            # skip: without optimize members there is nothing to warm-start or
            # GCG-optimize, so the group would silently produce no bag. Raise loudly
            # instead of persisting an empty scored set and continuing.
            raise AxBenchBpbAdvError(
                "fair feature-GCG derive produced no optimize members for "
                f"concept_id={concept_id} mode={mode.value} "
                f"(optimize_prefixes={len(optimize_prefixes)}, "
                f"split_ratio={ours.split_ratio}); an empty optimize split cannot "
                "be GCG-optimized. Increase split_ratio or provide more prefixes."
            )

        rich_items, avoid_items = _prefixed_completions(
            optimize_prefixes, tokenizer=tokenizer
        )
        if clustered_warm_starts is None:
            # Exact legacy/direct-call path. Clustered production passes the
            # phase-separated cache built before any GCG forward.
            _prepare_for_diffmean_forward(
                model=model,
                base_model=base_model,
                sae=sae,
                device=torch_device,
            )
            warm = feature_diffmean_bag(
                rich_items,
                avoid_items,
                sae=sae,
                model=base_model,
                sae_layer=sae_layer,
                percentile_table=percentile_table,
                mode=mode,
                k_pos=k_pos,
                k_neg=k_neg,
                positive_bucket=positive_bucket,
                negative_bucket=negative_bucket,
                device=torch_device,
                extraction_batch_size=cfg.feature_diffmean_extraction_batch_size,
                allowed_ids=allowed_ids,
                **cluster_kwargs,
            )
            _prepare_for_gcg_forward(
                model=model,
                base_model=base_model,
                sae=sae,
                device=torch_device,
            )
        else:
            warm = clustered_warm_starts.fair_bag(
                concept_id,
                mode,
                contract=cluster_contract,
                surface="fair feature-GCG derive",
            )
        # Every g_steer measurement in this group shares ONE knob binding (the
        # helper is used by both derives, so fair/roofline can't drift).
        group_g_steer = _group_g_steer(model, ours, mode, torch_device)
        warm_g_steer = group_g_steer(optimize_members, warm)

        vocab_fingerprint = checkpoint_vocab_fingerprint(model)
        for reg_lambda, objective_alpha in _lambda_alpha_grid(ours):
            objective_factory = _objective_factory(
                mode,
                torch_device,
                ours.max_forward_batch,
                ours.max_grad_batch,
                objective_alpha,
            )
            path = paths.ours_gcg_bag(
                concept_id,
                mode,
                reg_lambda,
                GcgFamily.FAIR,
                objective_alpha=objective_alpha,
            )
            # Identity is assembled BEFORE the derive so a committed bag can be
            # validated without paying for the GCG run (#1050). Every key here is
            # input-derived; n_features and the measured *_g_steer values are added
            # after the derive below.
            identity = _fair_bag_identity(
                cfg=cfg,
                ours=ours,
                concept_id=concept_id,
                mode=mode,
                reg_lambda=reg_lambda,
                objective_alpha=objective_alpha,
                optimize_prefixes=optimize_prefixes,
                held_out_prefixes=held_out_prefixes,
                optimize_members=optimize_members,
                held_out_members=held_out_members,
                rich_items=rich_items,
                avoid_items=avoid_items,
                k_pos=k_pos,
                k_neg=k_neg,
                positive_bucket=positive_bucket,
                negative_bucket=negative_bucket,
                cluster_contract=cluster_contract,
            )
            if cluster_contract is not None:
                committed = committed_bag_for_reuse(
                    path,
                    expected_identity={
                        **identity,
                        VOCAB_FINGERPRINT_META_KEY: vocab_fingerprint,
                    },
                    expected_feature_contract_fingerprint=(
                        cluster_contract.feature_contract_fingerprint
                    ),
                )
                if committed is not None:
                    logger.debug(
                        "fair feature-GCG bag reused: concept_id={} mode={} "
                        "reg_lambda={} alpha={} n_features={} <- {}",
                        concept_id,
                        mode.value,
                        format_reg_lambda(reg_lambda),
                        objective_alpha,
                        len(committed.bag),
                        path,
                    )
                    progress.record(reused=True)
                    continue
            gcg_cfg = FeatureGcgConfig(
                reg_lambda=float(reg_lambda),
                max_steps=ours.max_steps,
                minibatch_size=ours.minibatch_size,
                master_seed=ours.master_seed,
                max_forward_batch=ours.max_forward_batch,
                max_grad_batch=ours.max_grad_batch,
                objective_alpha=float(objective_alpha),
            )
            bag = optimize_feature_conditioning_universal(
                model,
                optimize_members,
                warm_start=warm,
                objective_factory=objective_factory,
                config=gcg_cfg,
                **cluster_kwargs,
            )
            optimize_g_steer = group_g_steer(optimize_members, bag)
            # held_out_members is guaranteed non-empty: the derive raises above on
            # empty held_out_prefixes, and build_class_margin_members returns >= 1
            # member per prefix (or raises). So this is always a real measurement,
            # never nan — the fair scored set is exactly these held-out prefixes.
            held_out_g_steer = group_g_steer(held_out_members, bag)

            metadata = {
                **identity,
                "n_features": str(len(bag)),
                "warm_start_g_steer": repr(float(warm_g_steer)),
                "optimize_g_steer": repr(float(optimize_g_steer)),
                "held_out_g_steer": repr(float(held_out_g_steer)),
            }
            _write_ours_gcg_bag(
                path,
                bag,
                metadata=metadata,
                vocab_fingerprint=vocab_fingerprint,
            )
            logger.info(
                f"fair feature-GCG bag written: concept_id={concept_id} "
                f"mode={mode.value} reg_lambda={format_reg_lambda(reg_lambda)} "
                f"alpha={objective_alpha} "
                f"n_features={len(bag)} warm_g_steer={warm_g_steer:.5f} "
                f"opt_g_steer={optimize_g_steer:.5f} "
                f"held_g_steer={held_out_g_steer:.5f} -> {path}"
            )
            progress.record(reused=False)

        # The fair universal bag covers every HELD-OUT prefix for every lambda, so
        # those are exactly the prefixes the score stage scores (the clean held-out
        # generalization measure). The optimize prefixes are excluded. Written once
        # per group -- the held-out split depends only on split_ratio/split_seed,
        # so it is independent of BOTH sweep axes (lambda and objective_alpha).
        scored_prefix_path = paths.ours_gcg_scored_prefixes(
            concept_id, mode, GcgFamily.FAIR
        )
        _write_ours_gcg_scored_prefixes(
            scored_prefix_path,
            [str(prefix.core_prefix_id) for prefix in held_out_prefixes],
            concept_id=concept_id,
            mode=mode,
            family=GcgFamily.FAIR,
            **(
                {}
                if cluster_contract is None
                else {
                    "feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
        logger.info(
            f"fair feature-GCG scored prefixes: concept_id={concept_id} "
            f"mode={mode.value} held_out={len(held_out_prefixes)} -> "
            f"{scored_prefix_path}"
        )

    logger.info(progress.summary())


# ─────────────────────────────────────────────────────────────────────────────
# The roofline derive (per-prefix, in-sample upper bound — the #575 oracle)
# ─────────────────────────────────────────────────────────────────────────────


def build_ours_gcg_bags_roofline(
    *,
    prefixes: Sequence[BpbAttackPrefix],
    model: Any,
    base_model: Any,
    sae: Any,
    percentile_table: t.Tensor,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    device: Any,
    tokenizer: Any,
    context_segments_by_prefix_hash: dict[str, tuple[ContextFeatureSegment, ...]],
    cluster_contract: ClusterFeatureContract | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache the roofline feature-GCG bags — per held-out PREFIX, per ``λ``.

    The in-sample upper bound and sibling of :func:`build_ours_gcg_bags_fair`.
    For each ``(concept_id, mode)`` group present in *prefixes*, the same
    deterministic optimize/held-out split as the fair methods is built and the
    held-out prefixes are processed. Each held-out prefix is still its OWN oracle
    eval unit: the warm-start is the feature-DiffMean over THAT prefix's own
    rich/avoid completions (the in-sample #575 oracle), and the universal optimizer
    refines a bag over that single prefix's members. Bags land at
    ``paths.ours_gcg_bag(concept, mode, λ, "roofline", prefix_id=<core_prefix_id>)``.
    The persisted scored-prefix set is the fair held-out set, so fair and roofline
    class-margin rows are measured on the same prefix population.

    Sign correctness mirrors the fair derive exactly: the objective comes from
    :func:`class_margin_objective` with the group ``mode`` (the single §7.2
    negation), so minimizing ``g_steer`` STRENGTHENS steering.  The roofline bag's
    in-sample ``g_steer`` on the prefix's own members is recorded as
    ``in_sample_g_steer`` (and dominates the fair universal bag on the same
    members — the ladder invariant pinned in
    ``tests/axbench_bpb_adv/integration/test_gcg_bags_roofline.py``).

    Parameters mirror :func:`build_ours_gcg_bags_fair` (the SAE-identity guard is
    the caller's responsibility, run once before either derive).
    """
    torch_device = t.device(device) if not isinstance(device, t.device) else device
    cluster_kwargs = (
        {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    )
    if clustered_warm_starts is not None:
        clustered_warm_starts.assert_matches(
            cluster_contract,
            surface="roofline feature-GCG derive",
        )
    positive_bucket, negative_bucket = assert_single_positive_bucket(
        model,
        **cluster_kwargs,
    )
    k_pos, k_neg = resolve_k(model, cfg.ours_gcg, **cluster_kwargs)
    sae_layer = _sae_layer_for_cfg(cfg)
    ours = cfg.ours_gcg
    # Same score-time window bound as the fair derive: members are built on the
    # untruncated prefix and raise if a prefix + completion would exceed the
    # encdec decoder sequence_len the score stage left-truncates against.
    max_seq_len = encdec_dec_sequence_len(model)
    # A compacted target checkpoint's allowed SAE feature vocabulary V (None on
    # an uncompacted checkpoint) -- restricts every member's FIXED context bags
    # (_members_for_prefixes -> build_class_margin_members) and the in-sample
    # warm feature_diffmean_bag selection to V, so neither can condition on /
    # select an id that trips the encoder's OOV assert.
    allowed_ids = feature_vocab_allowed_ids(model)

    groups = _group_prefixes_by_concept_mode(prefixes)
    logger.info(
        f"roofline feature-GCG derive: {len(groups)} (concept,mode) group(s), "
        f"reg_lambda_grid={list(ours.reg_lambda_grid)}, "
        f"objective_alpha_grid={list(ours.objective_alpha_grid)}, "
        f"split_ratio={ours.split_ratio}, k_pos={k_pos} "
        f"k_neg={k_neg} buckets=({positive_bucket},{negative_bucket})"
    )
    progress = _DeriveProgress(
        "roofline feature-GCG derive",
        sum(
            len(split_prefixes_for_concept(
                group_prefixes, ratio=ours.split_ratio, seed=ours.split_seed
            )[1])
            for _cid, _mode, group_prefixes in groups
        )
        * len(_lambda_alpha_grid(ours)),
        "bag",
    )

    for concept_id, mode, group_prefixes in groups:
        optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
            group_prefixes, ratio=ours.split_ratio, seed=ours.split_seed
        )
        if not held_out_prefixes:
            # OursGcgConfig bounds split_ratio < 1 and split_prefixes_for_concept
            # rejects <2-prefix groups, so this is a backstop: an empty selection
            # would persist an *empty* roofline scored set and the score stage would
            # silently drop every roofline row.
            raise AxBenchBpbAdvError(
                "roofline feature-GCG derive selected no HELD-OUT prefixes for "
                f"concept_id={concept_id} mode={mode.value} "
                f"(n_prefixes={len(group_prefixes)}, split_ratio={ours.split_ratio}); an empty "
                "selection would silently drop every roofline score downstream."
            )
        logger.info(
            f"roofline feature-GCG derive: concept_id={concept_id} "
            f"mode={mode.value} prefixes={len(group_prefixes)} "
            f"optimize={len(optimize_prefixes)} held_out={len(held_out_prefixes)} "
            f"(split_ratio={ours.split_ratio})"
        )

        # The g_steer knob binding depends only on (model, ours, mode, device) —
        # all group-level — so build it ONCE per group (not per prefix), matching
        # the fair derive; the SAME helpers keep fair/roofline batching from
        # drifting. The optimizer's objective factory is NOT group-level: it
        # carries the sweep's per-bag objective_alpha, so it is built inside the
        # (reg_lambda, alpha) loop below.
        group_g_steer = _group_g_steer(model, ours, mode, torch_device)

        vocab_fingerprint = checkpoint_vocab_fingerprint(model)

        # The roofline family scores each HELD-OUT prefix IN-SAMPLE under its own
        # bag, so the scored set is exactly the fair held-out split that produced a
        # bag for every lambda (a prefix that cannot produce members RAISES
        # mid-derive — there is no skip path — so reaching the write below means ALL
        # prefixes are scorable).
        scored_prefix_ids: list[str] = []
        for prefix in held_out_prefixes:
            rich_items, avoid_items = _prefixed_completions(
                [prefix], tokenizer=tokenizer
            )
            prefix_members = _members_for_prefixes(
                [prefix],
                device=torch_device,
                context_segments_by_prefix_hash=context_segments_by_prefix_hash,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                allowed_ids=allowed_ids,
                **cluster_kwargs,
            )
            if not prefix_members:
                # build_class_margin_members returns >= 1 member or RAISES (a
                # malformed prefix — missing rich/avoid side, absent context-cache
                # entry, or over-length prefix+completion — raises upstream), so an
                # empty list here is a real data/config inconsistency, not a routine
                # skip. The fair derive treats empty members as fatal; raise here too
                # rather than silently dropping the prefix from the scored set.
                raise AxBenchBpbAdvError(
                    "roofline feature-GCG derive produced no class-margin members "
                    f"for concept_id={concept_id} mode={mode.value} prefix_id="
                    f"{prefix.core_prefix_id!r}; build_class_margin_members must "
                    "return >= 1 member (it raises on a malformed prefix), so an "
                    "empty result is a real data/config inconsistency, not a "
                    "skippable case."
                )

            # In-sample warm-start (the #575 oracle): the feature-DiffMean over THIS
            # prefix's OWN rich/avoid completions, NOT a train-derived one.
            if clustered_warm_starts is None:
                # Exact legacy/direct-call path. Clustered production passes the
                # phase-separated cache built before any GCG forward.
                _prepare_for_diffmean_forward(
                    model=model,
                    base_model=base_model,
                    sae=sae,
                    device=torch_device,
                )
                warm = feature_diffmean_bag(
                    rich_items,
                    avoid_items,
                    sae=sae,
                    model=base_model,
                    sae_layer=sae_layer,
                    percentile_table=percentile_table,
                    mode=mode,
                    k_pos=k_pos,
                    k_neg=k_neg,
                    positive_bucket=positive_bucket,
                    negative_bucket=negative_bucket,
                    device=torch_device,
                    extraction_batch_size=cfg.feature_diffmean_extraction_batch_size,
                    allowed_ids=allowed_ids,
                    **cluster_kwargs,
                )
                _prepare_for_gcg_forward(
                    model=model,
                    base_model=base_model,
                    sae=sae,
                    device=torch_device,
                )
            else:
                warm = clustered_warm_starts.roofline_bag(
                    concept_id,
                    mode,
                    str(prefix.core_prefix_id),
                    contract=cluster_contract,
                    surface="roofline feature-GCG derive",
                )
            warm_g_steer = group_g_steer(prefix_members, warm)

            for reg_lambda, objective_alpha in _lambda_alpha_grid(ours):
                objective_factory = _objective_factory(
                    mode,
                    torch_device,
                    ours.max_forward_batch,
                    ours.max_grad_batch,
                    objective_alpha,
                )
                path = paths.ours_gcg_bag(
                    concept_id,
                    mode,
                    reg_lambda,
                    GcgFamily.ROOFLINE,
                    prefix_id=prefix.core_prefix_id,
                    objective_alpha=objective_alpha,
                )
                identity = _roofline_bag_identity(
                    cfg=cfg,
                    ours=ours,
                    concept_id=concept_id,
                    mode=mode,
                    reg_lambda=reg_lambda,
                    objective_alpha=objective_alpha,
                    prefix_id=prefix.core_prefix_id,
                    prefix_members=prefix_members,
                    rich_items=rich_items,
                    avoid_items=avoid_items,
                    k_pos=k_pos,
                    k_neg=k_neg,
                    positive_bucket=positive_bucket,
                    negative_bucket=negative_bucket,
                    cluster_contract=cluster_contract,
                )
                if cluster_contract is not None:
                    committed = committed_bag_for_reuse(
                        path,
                        expected_identity={
                            **identity,
                            VOCAB_FINGERPRINT_META_KEY: vocab_fingerprint,
                        },
                        expected_feature_contract_fingerprint=(
                            cluster_contract.feature_contract_fingerprint
                        ),
                    )
                    if committed is not None:
                        logger.debug(
                            "roofline feature-GCG bag reused: concept_id={} mode={} "
                            "prefix_id={} reg_lambda={} alpha={} n_features={} <- {}",
                            concept_id,
                            mode.value,
                            prefix.core_prefix_id,
                            format_reg_lambda(reg_lambda),
                            objective_alpha,
                            len(committed.bag),
                            path,
                        )
                        progress.record(reused=True)
                        continue
                gcg_cfg = FeatureGcgConfig(
                    reg_lambda=float(reg_lambda),
                    max_steps=ours.max_steps,
                    minibatch_size=ours.minibatch_size,
                    master_seed=ours.master_seed,
                    max_forward_batch=ours.max_forward_batch,
                    max_grad_batch=ours.max_grad_batch,
                    objective_alpha=float(objective_alpha),
                )
                bag = optimize_feature_conditioning_universal(
                    model,
                    prefix_members,
                    warm_start=warm,
                    objective_factory=objective_factory,
                    config=gcg_cfg,
                    **cluster_kwargs,
                )
                in_sample_g_steer = group_g_steer(prefix_members, bag)
                metadata = {
                    **identity,
                    "n_features": str(len(bag)),
                    "warm_start_g_steer": repr(float(warm_g_steer)),
                    "in_sample_g_steer": repr(float(in_sample_g_steer)),
                }
                _write_ours_gcg_bag(
                    path,
                    bag,
                    metadata=metadata,
                    vocab_fingerprint=vocab_fingerprint,
                )
                logger.info(
                    f"roofline feature-GCG bag written: concept_id={concept_id} "
                    f"mode={mode.value} prefix_id={prefix.core_prefix_id} "
                    f"reg_lambda={format_reg_lambda(reg_lambda)} "
                    f"alpha={objective_alpha} "
                    f"n_features={len(bag)} warm_g_steer={warm_g_steer:.5f} "
                    f"in_sample_g_steer={in_sample_g_steer:.5f} -> {path}"
                )
                progress.record(reused=False)

            # The prefix produced a bag for every lambda → it is scorable in-sample.
            scored_prefix_ids.append(str(prefix.core_prefix_id))

        if not scored_prefix_ids:
            raise AxBenchBpbAdvError(
                "roofline feature-GCG derive produced no scored prefixes for "
                f"concept_id={concept_id} mode={mode.value}."
            )
        scored_prefix_path = paths.ours_gcg_scored_prefixes(
            concept_id, mode, GcgFamily.ROOFLINE
        )
        _write_ours_gcg_scored_prefixes(
            scored_prefix_path,
            scored_prefix_ids,
            concept_id=concept_id,
            mode=mode,
            family=GcgFamily.ROOFLINE,
            **(
                {}
                if cluster_contract is None
                else {
                    "feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
        logger.info(
            f"roofline feature-GCG scored prefixes: concept_id={concept_id} "
            f"mode={mode.value} scored={len(scored_prefix_ids)} -> "
            f"{scored_prefix_path}"
        )

    logger.info(progress.summary())


__all__ = [
    "ClusteredWarmStartCache",
    "active_fair_gcg_modes",
    "active_roofline_gcg_modes",
    "assert_sae_matches_encdec",
    "build_clustered_warm_start_cache",
    "build_ours_gcg_bags_fair",
    "build_ours_gcg_bags_roofline",
    "fair_gcg_methods_active",
    "mean_g_steer",
    "roofline_gcg_methods_active",
]
