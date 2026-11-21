"""Checkpoint-derived setup resolution for the chat TUI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch as t
from loguru import logger

from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from steering.conditional.checkpoint_conditioning import (
    assert_conditioning_supported,
    assert_pretrain_conditioning_recoverable,
    read_checkpoint_conditioning,
)
from steering.conditional.checkpoint_feature_contract import (
    resolve_and_assert_model_feature_contract,
    resolve_checkpoint_feature_contract,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.self_conditioned import (
    selection_without_delta,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_mask import feature_mask_from_checkpoint_meta
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.conditional.sampling import (
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    build_percentile_table,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.eval.axbench_upstream.methods import (
    PROMPT_CONTEXT_WINDOW,
    PROMPT_SEGMENT_SIZE,
    MethodSetup,
    _validate_statistics_contract,
    load_method_setup,
)
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAERelease,
    coerce_config_l0,
    coerce_sae_width,
    resolve_sae,
)
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
    segmentation_identity_drift,
)
from steering.stats_io import load_statistics


def _file_sha256(path: Path) -> str:
    """Return the raw-byte digest recorded by existing chat checkpoint lineage."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_step(path: Path) -> int:
    raw_step = path.stem.removeprefix("meta_")
    try:
        return int(raw_step)
    except ValueError as exc:
        raise ValueError(f"ancestor checkpoint metadata path has no integer step: {path}") from exc


def _nearest_ancestor_conditioning(target_meta_path: Path) -> tuple[dict[str, Any], Path]:
    """Find the nearest fail-loud ancestor metadata with conditioning."""

    for directory in target_meta_path.parent.resolve().parents:
        candidates = list(directory.glob("meta_*.json"))
        if not candidates:
            continue
        candidate = max(candidates, key=_metadata_step)
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"could not read ancestor checkpoint metadata {candidate}: {exc}"
            ) from exc
        if not isinstance(loaded, dict) or not isinstance(loaded.get("user_config"), dict):
            raise ValueError(
                f"ancestor checkpoint metadata has no user_config mapping: {candidate}"
            )
        user_config = loaded["user_config"]
        if "conditioning" not in user_config:
            continue
        if not isinstance(user_config["conditioning"], dict):
            raise ValueError(
                "ancestor checkpoint metadata has malformed "
                f"user_config.conditioning mapping: {candidate}"
            )
        return loaded, candidate
    raise ValueError(
        "no ancestor checkpoint metadata with a user_config.conditioning block "
        f"was found above {target_meta_path.parent}"
    )


class _SetupMetadata(Protocol):
    @property
    def meta(self) -> Mapping[str, Any]: ...

    @property
    def target_meta_path(self) -> Path: ...


@dataclass(frozen=True)
class _SetupMetadataRef:
    """Minimal setup-shaped view used while a MethodSetup is being assembled."""

    meta: dict[str, Any]
    target_meta_path: Path


def _conditioning_for_setup(
    setup: _SetupMetadata,
    *,
    purpose: str,
) -> tuple[dict[str, Any], Path, str]:
    user_config = setup.meta.get("user_config")
    if not isinstance(user_config, dict):
        raise ValueError(f"checkpoint metadata has no user_config mapping for {purpose}")
    if "conditioning" in user_config:
        conditioning = user_config["conditioning"]
        if not isinstance(conditioning, dict):
            raise ValueError("checkpoint user_config.conditioning must be a mapping")
        return conditioning, Path(setup.target_meta_path), "leaf"
    ancestor_meta, ancestor_path = _nearest_ancestor_conditioning(Path(setup.target_meta_path))
    return ancestor_meta["user_config"]["conditioning"], ancestor_path, "ancestor"


def resolve_conditioning_seed(setup: _SetupMetadata) -> int:
    """Resolve the checkpoint seed that owns prompt feature sampling."""

    if (
        resolve_checkpoint_sft_segment_mode(
            setup.meta, checkpoint_dir=str(setup.target_meta_path.parent)
        )
        is SegmentMode.turn_delimited_fixed_tokens
    ):
        return OnlineSAELineage.from_checkpoint_meta(setup.meta).selection_seed
    conditioning, meta_path, owner = _conditioning_for_setup(
        setup,
        purpose="conditioning seed",
    )
    if "seed" not in conditioning:
        raise ValueError(f"{owner} conditioning block is present but lacks seed: {meta_path}")
    seed = conditioning["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("checkpoint conditioning seed must be a non-bool integer")
    return seed


def resolve_stats_sha(setup: _SetupMetadata) -> str:
    """Resolve the exact statistics binding validated while loading ``setup``."""

    if (
        resolve_checkpoint_sft_segment_mode(
            setup.meta, checkpoint_dir=str(setup.target_meta_path.parent)
        )
        is SegmentMode.turn_delimited_fixed_tokens
    ):
        return OnlineSAELineage.from_checkpoint_meta(setup.meta).statistics_sha256
    conditioning, meta_path, owner = _conditioning_for_setup(
        setup,
        purpose="statistics binding",
    )
    if "stats_source_sha256" not in conditioning:
        raise ValueError(
            f"{owner} conditioning block is present but lacks stats_source_sha256: {meta_path}"
        )
    stats_sha = conditioning["stats_source_sha256"]
    if (
        not isinstance(stats_sha, str)
        or len(stats_sha) != 64
        or any(character not in "0123456789abcdef" for character in stats_sha)
    ):
        raise ValueError(
            "checkpoint conditioning.stats_source_sha256 must be 64 lowercase hex characters"
        )
    return stats_sha


def _resolve_ancestor_sae(
    ancestor_user_config: dict[str, Any],
    ancestor_meta_path: Path,
) -> tuple[str, int, Any, int, str]:
    sae_config = ancestor_user_config.get("sae")
    if not isinstance(sae_config, dict):
        raise ValueError(f"target metadata has no user_config.sae mapping: {ancestor_meta_path}")
    try:
        sae_release = SAERelease(str(sae_config["release"]))
        sae_model = str(sae_config["model"])
        sae_layer = int(sae_config["layer"])
        activation_type = ActivationType(str(sae_config["activation_type"]))
        sae_width = coerce_sae_width(sae_config["width"])
        sae_l0 = coerce_config_l0(sae_release, sae_config.get("l0"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed user_config.sae in {ancestor_meta_path}: {exc}") from exc
    if activation_type is not ActivationType.residual:
        raise ValueError("AxBench SelfConditioning requires a residual SAE")
    n_sae_features = SAE_WIDTH_TO_NUM_FEATURES[sae_width]
    resolved_sae = resolve_sae(
        model=sae_model,
        release=sae_release,
        activation_type=activation_type,
        layer=sae_layer,
        width=sae_width,
        l0=sae_l0,
    )
    return sae_model, sae_layer, resolved_sae, n_sae_features, sae_width.value


def _sae_label(raw_sae: Any) -> str:
    if not isinstance(raw_sae, Mapping):
        return f"{raw_sae!r}/unknown-width"
    return f"{raw_sae.get('model')}/{raw_sae.get('width')}"


def load_chat_setup(
    target_meta_path: Path,
    feature_statistics_path: Path,
    *,
    pretrain_meta_path: Path | None = None,
) -> MethodSetup:
    """Resolve a pretraining checkpoint or its conditioned-SFT leaf."""

    target_meta_path = Path(target_meta_path)
    try:
        loaded_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read target metadata {target_meta_path}: {exc}") from exc
    if not isinstance(loaded_meta, dict) or not isinstance(loaded_meta.get("user_config"), dict):
        raise ValueError(f"target metadata has no user_config mapping: {target_meta_path}")
    meta: dict[str, Any] = loaded_meta
    if (
        resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=str(target_meta_path.parent))
        is SegmentMode.turn_delimited_fixed_tokens
    ):
        # The one-shot loader owns lineage, raw-byte statistics validation,
        # checkpoint selection and measured cadence. Do not recover a different
        # prompt policy from an ancestor for a self-contained turn leaf.
        resolve_turn_self_cond_segment_size_rustbpe(
            meta, checkpoint_dir=str(target_meta_path.parent)
        )
        return load_method_setup(
            target_meta_path, feature_statistics_path, pretrain_meta_path=pretrain_meta_path
        )
    if pretrain_meta_path is not None:
        raise ValueError("chat TUI pretrain_meta_path is only supported for turn checkpoints")
    require_fixed_checkpoint_segment_mode(
        meta,
        surface="chat TUI",
        checkpoint_dir=target_meta_path.parent,
    )
    user_config: dict[str, Any] = meta["user_config"]

    is_sft_leaf = "encoder_config" in meta and "encoder" not in user_config
    if not is_sft_leaf and "conditioning" in user_config:
        return load_method_setup(target_meta_path, feature_statistics_path)
    if not is_sft_leaf:
        raise ValueError(
            "target metadata is neither pretrain-shaped nor a conditioned-SFT leaf: "
            f"{target_meta_path}"
        )

    assert_pretrain_conditioning_recoverable(
        meta,
        checkpoint_label=str(target_meta_path),
    )
    conditioning = read_checkpoint_conditioning(meta)
    assert_conditioning_supported(
        conditioning,
        checkpoint_label=str(target_meta_path),
    )

    raw_output_selection = user_config.get("feature_selection")
    if not isinstance(raw_output_selection, dict):
        raise ValueError(
            "chat TUI SFT leaf output selection requires a feature_selection mapping; "
            f"got {raw_output_selection!r}"
        )
    parsed_output_selection = parse_feature_selection(raw_output_selection)
    if (
        not isinstance(parsed_output_selection, SamplingSelection)
        or parsed_output_selection.mode is not SamplingMode.top_k_percentile
    ):
        found_mode = getattr(parsed_output_selection, "mode", None)
        raise ValueError(
            "chat TUI SFT leaf output selection requires SamplingSelection with "
            "mode=top_k_percentile; got "
            f"{type(parsed_output_selection).__name__}(mode={found_mode!r})"
        )
    output_selection = selection_without_delta(parsed_output_selection)
    output_negative_selection = parse_negative_selection(conditioning.negative_selection)
    if output_negative_selection is None:
        raise ValueError("chat TUI SFT leaf requires checkpoint negative_selection")
    if meta.get("num_buckets") != 2 or user_config.get("num_buckets") != 2:
        raise ValueError(
            "AxBench SelfConditioning requires two buckets (active bucket 0, inactive bucket 1)"
        )

    ancestor_meta, ancestor_meta_path = _nearest_ancestor_conditioning(target_meta_path)
    ancestor_user_config: dict[str, Any] = ancestor_meta["user_config"]
    ancestor_conditioning: dict[str, Any] = ancestor_user_config["conditioning"]
    (
        sae_model,
        sae_layer,
        resolved_sae,
        n_sae_features,
        sae_width_label,
    ) = _resolve_ancestor_sae(ancestor_user_config, ancestor_meta_path)

    recorded_vocab = meta.get("feature_vocab")
    leaf_n_sae_features = (
        recorded_vocab.get("n_sae_features") if isinstance(recorded_vocab, dict) else None
    )
    if leaf_n_sae_features != n_sae_features:
        raise ValueError(
            "ancestor SAE n_sae_features="
            f"{n_sae_features} disagrees with leaf feature_vocab.n_sae_features="
            f"{leaf_n_sae_features}"
        )
    ancestor_sae = ancestor_user_config.get("sae")
    leaf_sae = user_config.get("sae")
    if leaf_sae != ancestor_sae:
        logger.warning(
            "chat TUI SFT leaf user_config.sae "
            f"{_sae_label(leaf_sae)} differs from ancestor conditioning SAE "
            f"{sae_model}/{sae_width_label}; ignoring the unrelated leaf config default"
        )

    feature_contract = resolve_checkpoint_feature_contract(
        meta,
        n_sae_features=PositiveInt.parse(n_sae_features),
        surface="chat TUI SFT leaf",
        source_path=str(target_meta_path),
    )
    feature_vocab = effective_feature_vocab(feature_contract.universe, n_sae_features)
    actual_vocab = (
        None
        if feature_vocab is None
        else {
            "n_sae_features": feature_vocab.n_sae_features,
            "vocab_size": feature_vocab.size,
            "sha256": feature_vocab.fingerprint(),
        }
    )
    if recorded_vocab != actual_vocab:
        raise ValueError(
            "target checkpoint feature vocabulary does not match its canonical "
            f"mask/cluster contract: recorded={recorded_vocab!r}, derived={actual_vocab!r}"
        )
    allowed_ids = None if feature_vocab is None else feature_vocab.new_to_old_tensor(device="cpu")

    prompt_feature_contract = resolve_checkpoint_feature_contract(
        {
            "feature_mask_canonical": meta.get("feature_mask_canonical"),
            "feature_clusters_descriptor": meta.get("feature_clusters_descriptor"),
            "feature_clusters_canonical": meta.get("feature_clusters_canonical"),
            "user_config": {
                "feature_mask": user_config.get("feature_mask"),
                "feature_clusters": user_config.get("feature_clusters"),
            },
        },
        n_sae_features=PositiveInt.parse(n_sae_features),
        surface="chat TUI prompt conditioning",
        source_path=str(target_meta_path),
    )
    leaf_canonical_mask = feature_mask_from_checkpoint_meta(meta)
    if leaf_canonical_mask != prompt_feature_contract.feature_mask:
        raise ValueError(
            "ambiguous checkpoint prompt feature mask: the leaf canonical mask "
            "disagrees with the resolved prompt feature contract"
        )
    prompt_feature_vocab = effective_feature_vocab(
        prompt_feature_contract.universe,
        n_sae_features,
    )
    if prompt_feature_vocab != feature_vocab:
        raise ValueError(
            "checkpoint prompt conditioning feature universe does not equal the "
            "prediction-head/encoder feature vocabulary"
        )

    setup_metadata = _SetupMetadataRef(meta=meta, target_meta_path=target_meta_path)
    feature_statistics_path = Path(feature_statistics_path)
    expected_stats_sha = resolve_stats_sha(setup_metadata)
    actual_stats_sha = _file_sha256(feature_statistics_path)
    if actual_stats_sha != expected_stats_sha:
        raise ValueError(
            "feature statistics do not match the target checkpoint: "
            f"expected sha256={expected_stats_sha}, got {actual_stats_sha} "
            f"for {feature_statistics_path}"
        )
    statistics = load_statistics(feature_statistics_path)
    _validate_statistics_contract(
        statistics,
        path=feature_statistics_path,
        sae_model=sae_model,
        sae_layer=sae_layer,
        n_sae_features=n_sae_features,
    )
    percentile_table = build_percentile_table(
        statistics.per_feature_stats,
        n_sae_features,
    )
    if ancestor_conditioning.get("n_sae_features") != n_sae_features:
        raise ValueError("checkpoint prompt conditioning SAE width disagrees with the SAE")
    if ancestor_conditioning.get("num_buckets") != 2:
        raise ValueError("checkpoint prompt conditioning must use two buckets")
    seed = resolve_conditioning_seed(setup_metadata)
    cleaned_feature_selection = dict(raw_output_selection)
    cleaned_feature_selection.pop("delta_weights", None)
    raw_negative_selection = conditioning.negative_selection
    if not isinstance(raw_negative_selection, Mapping):
        raise ValueError("chat TUI SFT leaf requires raw checkpoint negative_selection")
    leaf_conditioning = {
        "feature_selection": cleaned_feature_selection,
        "negative_selection": dict(raw_negative_selection),
        # Leaf path markers are advisory provenance. Canonical blobs above own
        # identity; preserving the leaf values avoids mixing ancestor markers.
        "feature_mask": user_config.get("feature_mask"),
        "feature_clusters": user_config.get("feature_clusters"),
        "n_sae_features": n_sae_features,
        "num_buckets": 2,
        "seed": seed,
    }
    prompt_pipeline = build_feature_pipeline_from_cfg_block(
        leaf_conditioning,
        per_feature_stats=statistics.per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=2,
        seed=seed,
        feature_mask=prompt_feature_contract.feature_mask,
    )
    if not isinstance(prompt_pipeline.selection_config, SamplingBundle):
        raise ValueError("checkpoint prompt conditioning must use sampling selection")
    prompt_selection = prompt_pipeline.selection_config.selection
    assert isinstance(prompt_selection, SamplingSelection)
    assert prompt_selection.max_k is not None
    assert output_selection.max_k is not None
    assert (
        prompt_selection.mode,
        int(prompt_selection.min_k),
        int(prompt_selection.max_k),
    ) == (
        output_selection.mode,
        int(output_selection.min_k),
        int(output_selection.max_k),
    ), "leaf prompt and output sampling selections disagree"
    if prompt_pipeline.negative_bundle is None:
        raise ValueError("checkpoint prompt conditioning must sample inactive features")
    if prompt_pipeline.effective_buckets != 1:
        raise ValueError("checkpoint prompt positives must encode as bucket 0 only")

    feature_prediction_config = meta.get("feature_prediction_config")
    if (
        not isinstance(feature_prediction_config, dict)
        or feature_prediction_config.get("loss_fn") != FeaturePredLossFn.BCE.value
    ):
        found = (
            feature_prediction_config.get("loss_fn")
            if isinstance(feature_prediction_config, dict)
            else feature_prediction_config
        )
        raise ValueError(
            f"chat TUI SFT leaf requires feature_prediction_config.loss_fn='bce'; got {found!r}"
        )
    if feature_prediction_config.get("loss_mode") != FeaturePredLossMode.MAX_POOL.value:
        raise ValueError(
            "chat TUI SFT leaf requires feature_prediction_config.loss_mode='max_pool'; "
            f"got {feature_prediction_config.get('loss_mode')!r}"
        )

    return MethodSetup(
        target_meta_path=target_meta_path,
        meta=meta,
        sae_model=sae_model,
        sae_layer=sae_layer,
        resolved_sae=resolved_sae,
        n_sae_features=n_sae_features,
        feature_contract=feature_contract,
        prompt_feature_contract=prompt_feature_contract,
        consolidation=feature_contract.consolidation,
        prompt_consolidation=prompt_feature_contract.consolidation,
        feature_vocab=feature_vocab,
        allowed_ids=allowed_ids,
        output_selection=output_selection,
        output_negative_selection=output_negative_selection,
        prompt_pipeline=prompt_pipeline,
        statistics=statistics,
        percentile_table=percentile_table,
        # ``require_fixed_checkpoint_segment_mode`` above already proved this
        # leaf is a fixed-token checkpoint, so state that shape explicitly
        # rather than letting a defaulted field imply it.
        segment_mode=SegmentMode.fixed_tokens,
        prompt_segment_size=PROMPT_SEGMENT_SIZE,
        prompt_context_window=PROMPT_CONTEXT_WINDOW,
        turn_self_cond_segment_size_rustbpe=None,
        prompt_selection_context=None,
    )


def _resolve_target_dtype(dtype_name: str, device: t.device) -> t.dtype | None:
    """Resolve the CUDA inference dtype; CPU keeps the checkpoint loader's dtype."""

    if device.type != "cuda":
        return None
    if dtype_name == "bfloat16":
        return t.bfloat16
    if dtype_name == "float32":
        return t.float32
    raise ValueError("chat_tui dtype must be one of bfloat16, float32")


@contextmanager
def _temporary_default_dtype(dtype: t.dtype | None):
    """Bound the empty CUDA model shell to the requested inference dtype.

    ``build_encdec_model`` first loads the checkpoint tensors, then creates an
    empty model before assigning those tensors.  Constructing that otherwise
    temporary shell in fp32 nearly doubles the load peak for large checkpoints.
    The state-dict assignment still restores every checkpoint tensor exactly;
    the explicit cast after loading owns the final inference dtype.
    """

    if dtype is None:
        yield
        return
    previous = t.get_default_dtype()
    t.set_default_dtype(dtype)
    try:
        yield
    finally:
        t.set_default_dtype(previous)


def load_chat_target_model(
    setup: MethodSetup,
    device: str,
    *,
    dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    """Load the target checkpoint and enforce the chat feature-head contract."""

    try:
        step = int(setup.meta["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"target metadata has no integer step: {setup.target_meta_path}") from exc
    target_device = t.device(device)
    target_dtype = _resolve_target_dtype(dtype, target_device)
    logger.info(
        "Loading chat TUI target on {} with inference dtype {}",
        target_device,
        target_dtype,
    )
    with _temporary_default_dtype(target_dtype):
        model, tokenizer, loaded_meta = build_encdec_model(
            str(setup.target_meta_path.parent),
            step,
            target_device,
            ModelPhase.EVAL,
            eval_conditioning=EvalConditioning.conditions_within_vocab("chat_tui"),
        )
    loaded_mode = resolve_checkpoint_sft_segment_mode(
        loaded_meta, checkpoint_dir=str(setup.target_meta_path.parent)
    )
    if loaded_mode is not resolve_checkpoint_sft_segment_mode(
        setup.meta, checkpoint_dir=str(setup.target_meta_path.parent)
    ) or segmentation_identity_drift(loaded_meta, setup.meta):
        raise ValueError("loaded target model segmentation identity differs from setup")
    resolve_turn_self_cond_segment_size_rustbpe(
        loaded_meta, checkpoint_dir=str(setup.target_meta_path.parent)
    )
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
        # The empty construction shell and replaced checkpoint tensors leave
        # large inactive blocks in the caching allocator. Release them before
        # the lazy Gemma/SAE stack claims the same device.
        t.cuda.empty_cache()
    contract = resolve_and_assert_model_feature_contract(
        model,
        loaded_meta,
        checkpoint_dir=str(setup.target_meta_path.parent),
        surface="chat TUI",
        missing_metadata_label="canonical feature_mask/feature_clusters metadata",
        missing_metadata_guidance="Use the checkpoint's matching metadata generation.",
    )
    loaded_clusters = contract.resolved_clusters
    setup_clusters = setup.feature_contract.resolved_clusters
    clusters_match = (loaded_clusters is None and setup_clusters is None) or (
        loaded_clusters is not None
        and setup_clusters is not None
        and loaded_clusters.definition == setup_clusters.definition
        and loaded_clusters.descriptor == setup_clusters.descriptor
    )
    if (
        contract.feature_mask != setup.feature_contract.feature_mask
        or not clusters_match
        or model.feature_vocab != setup.feature_vocab
    ):
        raise ValueError("loaded target model feature contract or vocabulary differs from setup")

    feature_prediction = getattr(model, "feature_pred_config", None)
    try:
        compatible_head = (
            feature_prediction is not None
            and FeaturePredLossFn(feature_prediction.loss_fn) is FeaturePredLossFn.BCE
            and FeaturePredLossMode(feature_prediction.loss_mode) is FeaturePredLossMode.MAX_POOL
        )
    except (AttributeError, TypeError, ValueError):
        compatible_head = False
    if not compatible_head:
        raise ValueError("chat TUI requires feature_prediction loss_fn=bce and loss_mode=max_pool")
    if not isinstance(model, EncoderDecoderGPT):
        raise ValueError(
            f"chat TUI requires an EncoderDecoderGPT target model; got {type(model).__name__}"
        )
    if model.feature_pred_head is None:
        raise ValueError("chat TUI requires a live feature_prediction head")
    if int(model.encoder.config.num_buckets) != 2:
        raise ValueError(
            "chat TUI requires two encoder feature buckets; got "
            f"{int(model.encoder.config.num_buckets)} buckets"
        )
    head_width = int(model.feature_pred_head.out_features)
    expected_width = (
        int(setup.feature_vocab.size)
        if setup.feature_vocab is not None
        else int(setup.n_sae_features)
    )
    if head_width != expected_width:
        raise ValueError(
            f"feature_prediction head width={head_width} disagrees with "
            f"setup feature vocabulary width={expected_width}"
        )
    return model, tokenizer


__all__ = [
    "load_chat_setup",
    "load_chat_target_model",
    "resolve_conditioning_seed",
    "resolve_stats_sha",
]
