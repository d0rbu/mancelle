"""Lazy checkpoint-derived online-SAE bootstrap for structured-SFT GRPO."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch as t
from loguru import logger

from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.encoder_decoder.sft_grpo import ConditioningAction, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_training import InitialConditioningMode
from steering.conditional.encoder_decoder.sft_selection import (
    SegmentSelectionContext,
    select_conversation_segments,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_pipeline import (
    ThresholdSelectionConfig,
    build_feature_pipeline,
    negative_sampling_mask,
)
from steering.conditional.missing_percentile import (
    MissingPercentilePolicy,
    MissingPercentileResolver,
    parse_missing_percentile_policy,
    percentile_table_coverage,
)
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.sampling import (
    SamplingBundle,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.thresholding import ThresholdConfig
from steering.eval.probe_activations import get_probe_activations
from steering.eval.probe_model import ProbeModel, load_probe_model
from steering.feature_paths import PoolingMode, parse_feature_dir
from steering.sae_resolver import ActivationType, SAERelease, coerce_config_l0
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    ONLINE_SAE_LINEAGE_META_KEY,
    ONLINE_SAE_LINEAGE_VERSION,
    ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY,
    resolve_checkpoint_sft_segment_mode,
)
from steering.sft.structured_rollout import ConversationEnvironment, stateless_seed
from steering.stats_io import load_statistics

ONLINE_SAE_LINEAGE_KEY = ONLINE_SAE_LINEAGE_META_KEY

_LINEAGE_FIELDS = frozenset(
    {
        "schema_version",
        "model",
        "release",
        "activation_type",
        "layer",
        "width",
        "l0",
        "source_segment_size",
        "pooling",
        "context_independent_features",
        "ignore_bos_features",
        "feature_dir",
        "statistics_path",
        "statistics_sha256",
        "n_sae_features",
        "num_buckets",
        "feature_selection",
        "negative_selection",
        "missing_percentile_policy",
        "selection_seed",
    }
)
_SOURCE_SEGMENT_MODE_FIELD = ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SFTGRPOError(f"online-SAE lineage {field} must be an integer >= {minimum}")
    return value


def _strict_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SFTGRPOError(f"online-SAE lineage {field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class OnlineSAELineage:
    """Complete immutable identity needed to reconstruct C0 online."""

    model: str
    release: str
    activation_type: str
    layer: int
    width: str
    l0: str | int | None
    source_segment_size: int
    feature_dir: str
    statistics_path: str
    statistics_sha256: str
    n_sae_features: int
    num_buckets: int
    feature_selection: Mapping[str, Any]
    negative_selection: Mapping[str, Any] | None
    missing_percentile_policy: str
    selection_seed: int
    source_segment_mode: str

    @classmethod
    def from_checkpoint_meta(cls, meta: Mapping[str, Any]) -> OnlineSAELineage:
        """Parse immutable lineage. Filesystem checks are ``validate_paths()``."""
        raw = meta.get(ONLINE_SAE_LINEAGE_KEY)
        if not isinstance(raw, Mapping):
            raise SFTGRPOError(
                f"online_sae requires checkpoint metadata {ONLINE_SAE_LINEAGE_KEY!r}"
            )
        if set(raw) - {_SOURCE_SEGMENT_MODE_FIELD} != _LINEAGE_FIELDS:
            raise SFTGRPOError(
                "online-SAE lineage fields are incomplete or unknown: "
                f"expected={sorted(_LINEAGE_FIELDS)} with optional "
                f"{_SOURCE_SEGMENT_MODE_FIELD!r}, got={sorted(raw, key=str)}"
            )
        if raw["schema_version"] != ONLINE_SAE_LINEAGE_VERSION:
            raise SFTGRPOError("unsupported online-SAE lineage schema version")
        if raw["pooling"] != PoolingMode.max_pool.value:
            raise SFTGRPOError("online_sae requires checkpoint-recorded max_pool extraction")
        if raw["context_independent_features"] is not True:
            raise SFTGRPOError(
                "online_sae requires checkpoint-recorded context-independent extraction"
            )
        if raw["ignore_bos_features"] is not True:
            raise SFTGRPOError("online_sae requires checkpoint-recorded no-BOS extraction")
        try:
            source_segment_mode = resolve_checkpoint_sft_segment_mode(
                meta,
                checkpoint_dir="<online-SAE checkpoint>",
            ).value
        except ValueError as error:
            raise SFTGRPOError(str(error)) from error
        feature_selection = raw["feature_selection"]
        negative_selection = raw["negative_selection"]
        if not isinstance(feature_selection, Mapping):
            raise SFTGRPOError("online-SAE feature_selection lineage must be a mapping")
        if negative_selection is not None and not isinstance(negative_selection, Mapping):
            raise SFTGRPOError("online-SAE negative_selection lineage must be a mapping or null")
        statistics_sha256 = _strict_string(
            raw["statistics_sha256"], field="statistics_sha256"
        )
        if len(statistics_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in statistics_sha256
        ):
            raise SFTGRPOError("online-SAE statistics_sha256 must be lowercase SHA-256")
        return cls(
            model=_strict_string(raw["model"], field="model"),
            release=_strict_string(raw["release"], field="release"),
            activation_type=_strict_string(raw["activation_type"], field="activation_type"),
            layer=_strict_int(raw["layer"], field="layer"),
            width=_strict_string(raw["width"], field="width"),
            l0=raw["l0"],
            source_segment_size=_strict_int(
                raw["source_segment_size"], field="source_segment_size", minimum=1
            ),
            feature_dir=_strict_string(raw["feature_dir"], field="feature_dir"),
            statistics_path=_strict_string(raw["statistics_path"], field="statistics_path"),
            statistics_sha256=statistics_sha256,
            n_sae_features=_strict_int(raw["n_sae_features"], field="n_sae_features", minimum=1),
            num_buckets=_strict_int(raw["num_buckets"], field="num_buckets", minimum=2),
            feature_selection=dict(feature_selection),
            negative_selection=(None if negative_selection is None else dict(negative_selection)),
            missing_percentile_policy=_strict_string(
                raw["missing_percentile_policy"], field="missing_percentile_policy"
            ),
            selection_seed=_strict_int(raw["selection_seed"], field="selection_seed"),
            source_segment_mode=source_segment_mode,
        )

    def assert_grpo_supported(self) -> None:
        """Apply GRPO's fixed-source policy after metadata parsing."""
        if self.source_segment_mode != SegmentMode.fixed_tokens.value:
            raise SFTGRPOError(
                "online_sae supports fixed_tokens source features only; this "
                f"checkpoint records {self.source_segment_mode!r} source features"
            )

    def validate_paths(self) -> None:
        parsed = parse_feature_dir(self.feature_dir)
        expected = {
            "model": self.model,
            "activation_type": self.activation_type,
            "layer": self.layer,
            "width": self.width,
            "source_segment_size": self.source_segment_size,
            "segment_mode": self.source_segment_mode,
            "pooling": PoolingMode.max_pool.value,
            "context_independent_features": True,
            "ignore_bos_features": True,
        }
        observed = {
            "model": parsed.model,
            "activation_type": parsed.activation_type.value,
            "layer": parsed.layer,
            "width": parsed.width.value,
            "source_segment_size": parsed.segment_size,
            "segment_mode": parsed.segment_mode.value,
            "pooling": parsed.pooling.value,
            "context_independent_features": parsed.context_independent_features,
            "ignore_bos_features": parsed.ignore_bos_features,
        }
        if observed != expected:
            raise SFTGRPOError(
                f"online-SAE feature path conflicts with checkpoint lineage: "
                f"expected={expected}, observed={observed}"
            )
        stats_path = Path(self.statistics_path).resolve(strict=True)
        feature_path = Path(self.feature_dir).resolve(strict=True)
        if stats_path.parent != feature_path:
            raise SFTGRPOError("online-SAE statistics_path must be a direct feature_dir child")
        if _sha256_file(stats_path) != self.statistics_sha256:
            raise SFTGRPOError("online-SAE statistics content digest changed")


def build_online_sae_lineage_metadata(
    *,
    feature_dir: str,
    statistics_path: str | Path,
    segment_mode: SegmentMode,
    sae_config: Mapping[str, Any],
    feature_selection: Mapping[str, Any],
    negative_selection: Mapping[str, Any] | None,
    missing_percentile_policy: str,
    n_sae_features: int,
    num_buckets: int,
    selection_seed: int,
) -> dict[str, Any]:
    """Build the self-contained block a conditioned-SFT checkpoint republishes."""
    required_sae = {"model", "release", "activation_type", "layer", "width"}
    if not required_sae.issubset(sae_config):
        raise SFTGRPOError(
            f"conditioned-SFT SAE config lacks online lineage fields {sorted(required_sae - set(sae_config))}"
        )
    feature_input = Path(feature_dir).resolve(strict=True)
    statistics_input = Path(statistics_path)
    if statistics_input.parent.resolve(strict=True) != feature_input:
        raise SFTGRPOError(
            "conditioned-SFT statistics_path must be a direct feature_dir child"
        )
    # SFT staging directories may consist of symlinks into immutable canonical
    # feature families. Parse and persist the statistics target's canonical
    # parent, which carries the source SAE geometry needed by online bootstrap.
    stats_path = statistics_input.resolve(strict=True)
    canonical_feature_dir = stats_path.parent
    parsed = parse_feature_dir(canonical_feature_dir)
    if parsed.segment_mode is not segment_mode:
        raise SFTGRPOError(
            "conditioned-SFT feature directory leaf disagrees with the shard "
            f"metadata: directory {canonical_feature_dir} encodes "
            f"{parsed.segment_mode.value!r}, discovered artifacts record "
            f"{segment_mode.value!r}. Re-extract into the canonical output_dir."
        )
    expected_sae_identity = {
        "model": parsed.model,
        "activation_type": parsed.activation_type.value,
        "layer": parsed.layer,
        "width": parsed.width.value,
    }
    recorded_sae_identity = {
        "model": str(sae_config["model"]),
        "activation_type": str(sae_config["activation_type"]),
        "layer": int(sae_config["layer"]),
        "width": str(sae_config["width"]),
    }
    if recorded_sae_identity != expected_sae_identity:
        raise SFTGRPOError(
            "conditioned-SFT SAE config conflicts with canonical feature path: "
            f"config={recorded_sae_identity}, path={expected_sae_identity}"
        )
    return {
        ONLINE_SAE_LINEAGE_KEY: {
            "schema_version": ONLINE_SAE_LINEAGE_VERSION,
            "model": str(sae_config["model"]),
            "release": str(sae_config["release"]),
            "activation_type": str(sae_config["activation_type"]),
            "layer": int(sae_config["layer"]),
            "width": str(sae_config["width"]),
            "l0": sae_config.get("l0"),
            "source_segment_size": parsed.segment_size,
            **(
                {_SOURCE_SEGMENT_MODE_FIELD: segment_mode.value}
                if segment_mode is not SegmentMode.fixed_tokens
                else {}
            ),
            "pooling": parsed.pooling.value,
            "context_independent_features": parsed.context_independent_features,
            "ignore_bos_features": parsed.ignore_bos_features,
            "feature_dir": str(canonical_feature_dir),
            "statistics_path": str(stats_path),
            "statistics_sha256": _sha256_file(stats_path),
            "n_sae_features": n_sae_features,
            "num_buckets": num_buckets,
            "feature_selection": dict(feature_selection),
            "negative_selection": (
                None if negative_selection is None else dict(negative_selection)
            ),
            "missing_percentile_policy": missing_percentile_policy,
            "selection_seed": selection_seed,
        }
    }


def build_online_selection_context_factory(
    lineage: OnlineSAELineage,
    feature_contract: CheckpointFeatureContract,
) -> Callable[[int], SegmentSelectionContext]:
    """Load pinned statistics once and rebuild only stateless RNG bundles per row."""
    statistics = load_statistics(lineage.statistics_path)
    per_feature_stats = statistics.per_feature_stats
    percentile_table = build_percentile_table(per_feature_stats, lineage.n_sae_features)
    policy = parse_missing_percentile_policy(lineage.missing_percentile_policy)
    missing = (
        MissingPercentileResolver.error()
        if policy is MissingPercentilePolicy.ERROR
        else MissingPercentileResolver(policy, percentile_table_coverage(percentile_table))
    )
    consolidation = feature_contract.consolidation

    def build(original_index: int) -> SegmentSelectionContext:
        row_seed = stateless_seed(
            {
                "namespace": "online_sae_initial_conditioning",
                "selection_seed": lineage.selection_seed,
                "original_index": original_index,
                "schema_version": ONLINE_SAE_LINEAGE_VERSION,
            }
        )
        pipeline = build_feature_pipeline(
            feature_selection=parse_feature_selection(dict(lineage.feature_selection)),
            negative_selection=parse_negative_selection(
                None
                if lineage.negative_selection is None
                else dict(lineage.negative_selection)
            ),
            per_feature_stats=per_feature_stats,
            n_sae_features=lineage.n_sae_features,
            num_buckets=lineage.num_buckets,
            seed=row_seed,
            feature_mask=feature_contract.feature_mask,
        )
        negative_bundle = pipeline.negative_bundle
        if negative_bundle is not None and consolidation is not None:
            negative_bundle = replace(
                negative_bundle,
                feature_mask=negative_sampling_mask(pipeline, consolidation),
            )
        threshold_cfg = None
        precomputed_thresholds = None
        sampling_bundle = None
        if isinstance(pipeline.selection_config, ThresholdSelectionConfig):
            selection = pipeline.selection_config.selection
            threshold_cfg = ThresholdConfig(selection.threshold_type, selection.threshold)
            precomputed_thresholds = pipeline.selection_config.precomputed_thresholds
            if precomputed_thresholds is not None:
                precomputed_thresholds = missing.mask_precomputed_thresholds(
                    precomputed_thresholds
                )
        else:
            sampling_bundle = pipeline.selection_config
            assert isinstance(sampling_bundle, SamplingBundle)
        return SegmentSelectionContext(
            threshold_cfg=threshold_cfg,
            precomputed_thresholds=precomputed_thresholds,
            sampling_bundle=sampling_bundle,
            negative_bundle=negative_bundle,
            boundaries=pipeline.boundaries,
            fp_percentile_table=percentile_table,
            cluster_threshold_percentile_table=(
                percentile_table if consolidation is not None else None
            ),
            feature_mask=feature_contract.feature_mask,
            consolidation=consolidation,
            missing_percentile=missing,
        )

    return build


ProbeFactory = Callable[[], ProbeModel]
ActivationProvider = Callable[[ProbeModel, t.Tensor], t.Tensor]


class OnlineSAEInitialConditioningProvider:
    """Compute one C0 from the complete first RustBPE segment, lazily per rank."""

    mode = InitialConditioningMode.ONLINE_SAE

    def __init__(
        self,
        lineage: OnlineSAELineage,
        *,
        decode_rustbpe: Callable[[Sequence[int]], str],
        selection_context_factory: Callable[[int], SegmentSelectionContext],
        device: t.device | str,
        probe_factory: ProbeFactory | None = None,
        activation_provider: ActivationProvider | None = None,
    ) -> None:
        self._lineage = lineage
        self._decode_rustbpe = decode_rustbpe
        self._selection_context_factory = selection_context_factory
        self._device = t.device(device)
        self._probe_factory = probe_factory
        self._activation_provider = activation_provider
        self._probe: ProbeModel | None = None
        self.raw_empty_count = 0
        self.selected_empty_count = 0
        self.boundary_char_drift_total = 0

    @property
    def sae_loaded(self) -> bool:
        return self._probe is not None

    def _ensure_probe(self) -> ProbeModel:
        if self._probe is None:
            if self._probe_factory is not None:
                self._probe = self._probe_factory()
            else:
                self._probe = load_probe_model(
                    model_name=self._lineage.model,
                    layer=self._lineage.layer,
                    activation_type=self._lineage.activation_type,
                    sae_release=self._lineage.release,
                    sae_width=self._lineage.width,
                    device=self._device,
                    l0=coerce_config_l0(
                        SAERelease(self._lineage.release), self._lineage.l0
                    ),
                    keep_transformerlens_model=False,
                )
            if self._probe.layer != self._lineage.layer:
                raise SFTGRPOError("loaded online-SAE probe layer conflicts with lineage")
            if self._probe.activation_type is not ActivationType(
                self._lineage.activation_type
            ):
                raise SFTGRPOError(
                    "loaded online-SAE activation type conflicts with lineage"
                )
        return self._probe

    @staticmethod
    def _encode_without_specials(tokenizer: Any, text: str) -> list[int]:
        try:
            raw = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            raw = tokenizer.encode(text)
        return [int(token_id) for token_id in raw]

    @staticmethod
    def _bos_id(tokenizer: Any) -> int:
        value = getattr(tokenizer, "bos_token_id", None)
        if value is None:
            getter = getattr(tokenizer, "get_bos_token_id", None)
            value = getter() if callable(getter) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SFTGRPOError("online-SAE tokenizer does not expose a valid BOS id")
        return value

    def _activations(self, probe: ProbeModel, tokens: t.Tensor) -> t.Tensor:
        if self._activation_provider is not None:
            return self._activation_provider(probe, tokens)
        return get_probe_activations(
            probe.model,
            tokens,
            layer=probe.layer,
            hook_name=probe.hook_name,
            activation_type=probe.activation_type,
            nnterp_model=probe.nnterp_model,
        )

    @t.no_grad()
    def provide(self, environment: ConversationEnvironment) -> ConditioningAction:
        first_segment = environment.segments[0]
        if first_segment.start != 0 or first_segment.end <= 0:
            raise SFTGRPOError("online-SAE first rollout segment must start at target zero")
        # Decode the cumulative authoritative target prefix once. Never concatenate
        # per-token decodes, which is incorrect at UTF-8/BPE boundaries.
        first_text = self._decode_rustbpe(environment.targets[: first_segment.end])
        probe = self._ensure_probe()
        content_ids = self._encode_without_specials(probe.tokenizer, first_text)
        bos_id = self._bos_id(probe.tokenizer)
        token_ids = content_ids if content_ids[:1] == [bos_id] else [bos_id, *content_ids]
        if probe.context_length is not None and len(token_ids) > probe.context_length:
            raise SFTGRPOError(
                "online-SAE first segment exceeds the checkpoint-derived probe context"
            )
        if not token_ids:
            raise SFTGRPOError("online-SAE tokenizer returned no rows including BOS")
        if hasattr(probe.tokenizer, "decode"):
            decoded = str(probe.tokenizer.decode(content_ids))
            drift = len(decoded) - len(first_text)
            self.boundary_char_drift_total += abs(drift)
            if drift:
                logger.warning(
                    "online-SAE tokenizer boundary drift for row {}: {} chars",
                    environment.original_index,
                    drift,
                )
        token_tensor = t.tensor([token_ids], dtype=t.int64, device=probe.device)
        with t.autocast(
            device_type=probe.device.type,
            dtype=t.bfloat16,
            enabled=probe.device.type == "cuda",
        ):
            activations = self._activations(probe, token_tensor)
            if activations.ndim != 3 or activations.shape[:2] != token_tensor.shape:
                raise SFTGRPOError(
                    "online-SAE activation provider must return shape (1, tokens, d_model)"
                )
            encoded = probe.sae.encode(
                activations.reshape(-1, activations.shape[-1])
            )
        if encoded.ndim != 2 or encoded.shape[1] != self._lineage.n_sae_features:
            raise SFTGRPOError("online-SAE width conflicts with checkpoint lineage")
        keep_rows = t.tensor(
            [token_id != bos_id for token_id in token_ids],
            dtype=t.bool,
            device=encoded.device,
        )
        if not bool(keep_rows.any()):
            pooled = t.zeros(
                self._lineage.n_sae_features,
                dtype=t.float32,
                device=encoded.device,
            )
        else:
            pooled = encoded[keep_rows].float().amax(dim=0)
        if not bool(t.isfinite(pooled).all()):
            raise SFTGRPOError("online-SAE pooled activations are non-finite")
        # Feature shards persist fp16 magnitudes and load them as fp32. Make that
        # representation authoritative before applying the shared SFT selector.
        persisted = pooled.to(t.float16).to(t.float32).cpu()
        if not t.equal(persisted, persisted.to(t.float16).to(t.float32)):
            raise SFTGRPOError("online-SAE FP16 persisted magnitude round trip drifted")
        feature_ids = t.nonzero(persisted > 0, as_tuple=False).squeeze(1).to(t.int32)
        feature_mags = persisted[feature_ids.long()]
        if feature_ids.numel() == 0:
            self.raw_empty_count += 1
            logger.error(
                "online-SAE raw extraction empty for row {} (raw_empty_count={})",
                environment.original_index,
                self.raw_empty_count,
            )
        raw = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=feature_ids,
            feature_mags=feature_mags,
            char_start=0,
            char_end=len(first_text),
            token_start=0,
            token_end=first_segment.end,
        )
        selected = select_conversation_segments(
            [raw],
            ctx=self._selection_context_factory(environment.original_index),
            example_idx=environment.original_index,
        )[0]
        selected_ids = tuple(int(value) for value in selected.feature_ids.tolist())
        buckets = tuple(int(value) for value in selected.buckets.tolist())
        if len(selected_ids) != len(buckets):
            raise SFTGRPOError("online-SAE selected IDs and buckets are misaligned")
        if not selected_ids:
            self.selected_empty_count += 1
            logger.error(
                "online-SAE post-selection conditioning empty for row {} "
                "(selected_empty_count={})",
                environment.original_index,
                self.selected_empty_count,
            )
        return ConditioningAction(
            feature_ids=selected_ids,
            magnitudes=buckets,
            mask=(True,) * len(selected_ids),
        )


__all__ = [
    "ONLINE_SAE_LINEAGE_KEY",
    "ONLINE_SAE_LINEAGE_VERSION",
    "OnlineSAEInitialConditioningProvider",
    "OnlineSAELineage",
    "build_online_sae_lineage_metadata",
    "build_online_selection_context_factory",
]
