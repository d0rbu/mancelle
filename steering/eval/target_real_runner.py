"""Shared target_real eval orchestration.

Consolidates helpers previously duplicated across
``scripts/eval/chat_eval.py``, ``scripts/eval/base_eval.py`` and
``steering/eval/dclm_eval.py`` (PR-C target_real path) plus normal shared
context-provider construction.

The dependency direction is: PR-B and PR-C both consume
:func:`build_context_feature_provider`. The PR-B caller passes
``cfg.conditioning.context_cache_dir`` (single shared cache for the
whole eval); the PR-C caller passes ``trc.cache_dir`` (per-target-real-
config cache). The other parameters (identity, n_sae_features,
num_buckets, model_max_seq_len) all come from the shared
``cfg.conditioning`` block in both cases.

PR-D Batch H2 also lifts the per-example target_real loop into
:func:`run_chat_target_real_per_example` (and a thin
:func:`lookup_target_cache_entry_or_raise` plus
:func:`derive_target_char_range_from_mask` helper) so the progression
collect-loop can reuse the same primitive that ``chat_eval`` and
``base_eval`` use, instead of re-deriving the per-example wrapper rebuild
logic.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from loguru import logger
from omegaconf import OmegaConf

from nanochat.common import get_dist_info
from nanochat.report import get_report
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
    feature_selection_hash_leaf,
)
from steering.eval.eval_cache_paths import default_target_cache_dir
from steering.eval.inference_engine import build_wrapper_for_features
from steering.eval.progression_results import (
    TargetRealLockstepSuiteResult,
    TargetRealPooledSuiteResult,
    dump_suite_to_dict,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    ContextRoutingMode,
    LockstepTargetRealConfig,
    PooledTargetRealConfig,
    SingleEntryContextProvider,
    TargetConditioningMode,
    parse_target_real_configs,
    target_real_cache_generation_instructions,
    token_context_key,
)
from steering.eval.target_real_dispatch import (
    build_target_for_real_config,
    build_turn_target_for_real_config,
)
from steering.eval.turn_context_contract import (
    TurnContextContract,
    canonical_turn_payload,
    validate_turn_context_contract,
)
from steering.feature_paths import ACTIVATION_LABELS
from steering.refined_types import PositiveInt
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, ActivationType, SAEWidth
from steering.segmentation import SegmentMode, token_to_char_offsets


class ContextSegmentProvider(Protocol):
    def get(self, token_ids) -> tuple[ContextFeatureSegment, ...] | None: ...


@dataclass(frozen=True)
class TargetRealExampleArtifacts:
    """Artifacts derived while preparing one target-real eval example."""

    target: object
    wrapper: object
    context_token_ids: list[int]
    target_char_start: int
    target_char_end: int


def context_tokens_from_target_mask(
    full_token_ids: list[int],
    mask: list[int],
) -> list[int]:
    """Return the completion prompt prefix ending before the first target token."""
    mask_1_indices = [idx for idx, value in enumerate(mask) if value == 1]
    if not mask_1_indices:
        return list(full_token_ids)
    return list(full_token_ids[: mask_1_indices[0]])


def context_segments_for_target_real_prefix(
    *,
    provider: ContextSegmentProvider,
    context_token_ids: list[int],
    trc_name: str,
) -> tuple[ContextFeatureSegment, ...]:
    """Return the exact prompt/context cache row for a target_real example."""
    exact_context_segments = provider.get(context_token_ids)
    if exact_context_segments is None:
        raise KeyError(
            f"target_real_configs[{trc_name}]: no prompt/context cache row "
            f"found for {token_context_key(context_token_ids)} "
            f"(token count={len(context_token_ids)}). Target-real caches must "
            "contain both the exact full target row and the exact prompt/context "
            "row used by generation/scoring. Rebuild the cache with "
            "evals=chat_target/core_target after re-enumerating contexts.\n\n"
            f"{target_real_cache_generation_instructions()}"
        )
    if len(exact_context_segments) == 0:
        raise KeyError(
            f"target_real_configs[{trc_name}]: prompt/context cache row "
            f"{token_context_key(context_token_ids)} has no segments. Rebuild "
            "the target-real cache from full prompt/context extraction."
        )
    return tuple(exact_context_segments)


def build_context_feature_provider(
    cfg,
    *,
    cache_dir,
    error_prefix: str = "conditioning.context_cache_dir",
    expected_metadata: dict[str, Any] | None = None,
    expected_contract_metadata: dict[str, Any] | None = None,
    turn_contract: TurnContextContract | None = None,
) -> SafetensorsContextFeatureProvider:
    """Build a :class:`SafetensorsContextFeatureProvider` from
    ``cfg.conditioning`` plus a caller-supplied ``cache_dir``.

    The only thing that varies between callers is where ``cache_dir`` comes
    from:

    * PR-B passes ``cfg.conditioning.context_cache_dir`` (single shared
      cache for the eval run).
    * PR-C passes ``trc.cache_dir`` (per-target-real-config cache).

    ``error_prefix`` lets the caller customize error messages so
    operators can tell which surface (PR-B / target_real_configs[name])
    tripped the validation. The caller MUST handle the case where its
    own ``cache_dir`` source is unset; this helper assumes it's set.

    ``feature_selection_hash`` is resolved via
    :func:`resolve_feature_selection_hash`: explicit hash and canonical
    pipeline parameters must agree; otherwise the canonical hash is
    auto-computed when possible.

    ``num_buckets`` defaults to 2 (binary-bucket SAE pipeline) to match
    the YAML defaults; override via ``conditioning.num_buckets`` for
    legacy 10-bucket caches.
    """
    cond = cfg.get("conditioning", None)
    if cond is None:
        raise ValueError(
            f"{error_prefix}: cfg.conditioning is required to "
            f"specify the cache identity (model_id/sae_id/layer/pooling/"
            f"feature_selection_hash)"
        )
    identity_cfg = cond.get("identity", None)
    if identity_cfg is None and turn_contract is None:
        raise ValueError(f"{error_prefix}: cfg.conditioning.identity is required")
    identity = cache_identity_from_config(
        identity_cfg,
        cond,
        prefix=f"{error_prefix}: cfg.conditioning.identity",
        turn_contract=turn_contract,
    )
    model_max_seq_len = int(cfg.get("model_max_seq_len", 1024))
    if turn_contract is not None:
        validate_turn_consumer_config(cfg, turn_contract)
    full_cache_dir = context_length_cache_dir(
        cache_dir,
        model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
    )
    required_metadata = {"model_max_seq_len": model_max_seq_len}
    required_metadata.update(expected_metadata or {})
    required_metadata.update(expected_contract_metadata or {})
    if turn_contract is not None and "feature_mask_canonical" in required_metadata:
        # Schema 5 records this same canonical mask under `feature_mask`.
        # Translate the shared checkpoint expectation, preserving existing caches.
        required_metadata["feature_mask"] = required_metadata.pop("feature_mask_canonical")
    return SafetensorsContextFeatureProvider(
        full_cache_dir,
        identity=identity,
        expected_n_sae_features=(turn_contract.lineage.n_sae_features if turn_contract is not None
                                 else int(cond.get("n_sae_features", 16384))),
        expected_num_buckets=(turn_contract.lineage.num_buckets if turn_contract is not None
                              else int(cond.get("num_buckets", 2))),
        expected_metadata=required_metadata,
        expected_source_segment_mode=(SegmentMode.fixed_tokens if turn_contract is None
                                      else SegmentMode.turn_delimited_fixed_tokens),
        expected_turn_contract=turn_contract,
    )


def validate_turn_consumer_config(cfg, contract: TurnContextContract) -> None:
    """Treat all supplied geometry/selection knobs as checkpoint assertions."""
    if cfg.get("model_max_seq_len") != contract.model_max_seq_len:
        raise ValueError("model_max_seq_len must equal the turn checkpoint sequence_len")
    cond = cfg.get("conditioning") or {}
    assertions = {
        "n_sae_features": contract.lineage.n_sae_features,
        "num_buckets": contract.lineage.num_buckets,
        "source_segment_size": contract.lineage.source_segment_size,
        "source_context_window": contract.semantic["source_context_window"],
        "source_bos_char_len": contract.semantic["source_bos_char_len"],
        "feature_selection": dict(contract.lineage.feature_selection),
        "negative_selection": contract.lineage.negative_selection,
        "seed": contract.lineage.selection_seed,
        "stats_source_sha256": contract.lineage.statistics_sha256,
        "feature_mask": contract.semantic["feature_mask"],
        "missing_percentile_policy": contract.lineage.missing_percentile_policy,
    }
    for field, expected in assertions.items():
        value = cond.get(field)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if value is not None and canonical_turn_payload({field: value}) != canonical_turn_payload({field: expected}):
            raise ValueError(f"conditioning.{field} conflicts with checkpoint turn contract; use chat_eval_turn")
    if cond.get("feature_clusters") is not None or cond.get("feature_delta") is not None:
        raise ValueError("turn ChatCORE does not support clustered/delta selection overrides")


def require_single_process_turn_target_real(*, rank: int = 0, world_size: int = 1) -> None:
    """Reject multiple processes; the outer runner cannot drive other ranks' collectives.

    A one-rank DDP launch is accepted: rank zero and world size one satisfy the
    single-process restriction whether or not DDP initialization was requested.
    """
    _ddp, actual_rank, _local_rank, actual_world_size = get_dist_info()
    if rank != 0 or world_size != 1 or actual_rank != 0 or actual_world_size != 1:
        raise ValueError("turn target-real ChatCORE requires single-process execution")


def require_shared_context_cache_dir(cfg, *, eval_name: str):
    """Return ``conditioning.context_cache_dir`` or raise a clear eval error."""

    cond = cfg.get("conditioning", None)
    if cond is None:
        raise ValueError(f"Encoder-decoder {eval_name} requires a conditioning block")
    cache_root = cond.get("context_cache_dir", None)
    if cache_root is None:
        raise ValueError(
            f"Encoder-decoder {eval_name} requires conditioning.context_cache_dir "
            "so prompts/contexts receive real cached SAE features."
        )
    return cache_root


def build_shared_context_feature_provider(
    cfg,
    *,
    eval_name: str,
) -> SafetensorsContextFeatureProvider:
    """Build the shared full-context provider for normal encoder-decoder evals."""

    return build_context_feature_provider(
        cfg,
        cache_dir=require_shared_context_cache_dir(cfg, eval_name=eval_name),
        error_prefix="conditioning.context_cache_dir is set but conditioning",
    )


def _build_auto_resolver(
    cfg,
    *,
    suite: str,
    model_max_seq_len: int | None = None,
    turn_contract: TurnContextContract | None = None,
) -> Callable[[dict], Path]:
    """Build the auto-resolve callable for null-cache_dir entries.

    Captures ``cfg.conditioning.identity`` + ``model_max_seq_len`` +
    suite name, and returns a closure that resolves a per-entry cache
    path via :func:`default_target_cache_dir`.

    Two cfg layouts are supported (PR-D Batch H7 paranoid review fix):
      1. Top-level cfg with nested ``conditioning.identity`` —
         chat_eval / base_eval / dclm_eval pass the WHOLE cfg.
      2. cond-block passed directly (cfg's top-level keys ARE
         conditioning's keys: ``identity``, ``target_real_configs``,
         etc.) — training_progression passes ``cfg.conditioning`` so the
         parser can read ``target_real_configs`` from a nested location.

    Args:
        cfg: A Hydra/OmegaConf config or plain dict — either layout above.
        suite: Suite name (e.g. ``"core_target"``, ``"chat_target"``)
            embedded in the auto-resolved path.
        model_max_seq_len: Explicit context length used for the cache
            ``c{N}`` directory. When ``None`` (default), falls back to
            ``cfg.get("model_max_seq_len", 1024)`` which preserves the
            base_eval/chat_eval/dclm_eval contract where the field lives
            at the top level of the eval cfg. training_progression must
            pass this explicitly because its ``cfg.conditioning`` block
            doesn't carry ``model_max_seq_len`` (the value lives on
            ``model.dec_config.sequence_len``).
    """
    # Try the nested-conditioning layout first; fall back to direct identity.
    cond = cfg.get("conditioning", None)
    if cond is None:
        # Could be a cond-block passed directly: identity should be at this
        # level. Detect by presence of ``identity``.
        if cfg.get("identity", None) is not None:
            cond = cfg
        else:
            raise ValueError(
                f"target_real_configs auto-resolution (suite={suite!r}) requires "
                f"cfg.conditioning to specify the cache identity "
                f"(model_id/sae_id/layer/pooling/feature_selection_hash)"
            )
    identity_cfg = cond.get("identity", None)
    if identity_cfg is None and turn_contract is None:
        raise ValueError(
            f"target_real_configs auto-resolution (suite={suite!r}) requires "
            f"cfg.conditioning.identity to specify the cache identity "
            f"(model_id/sae_id/layer/pooling/feature_selection_hash)"
        )
    if model_max_seq_len is None:
        raw_seq_len = int(cfg.get("model_max_seq_len", 1024))
    else:
        raw_seq_len = int(model_max_seq_len)
    try:
        resolved_seq_len = PositiveInt.parse(raw_seq_len)
    except TypeError:
        raise ValueError(
            f"model_max_seq_len must be a positive integer; got {raw_seq_len}"
        ) from None
    identity = cache_identity_from_config(identity_cfg, cond, turn_contract=turn_contract)
    if turn_contract is not None and resolved_seq_len != turn_contract.model_max_seq_len:
        raise ValueError("model_max_seq_len must equal the turn checkpoint sequence_len")

    # Turn caches name their activation/width from the checkpoint lineage
    # instead of parsing them back out of the sae_id.
    activation_label = None if turn_contract is None else ACTIVATION_LABELS[
        ActivationType(turn_contract.lineage.activation_type)
    ]
    sae_width = None if turn_contract is None else SAE_WIDTH_TO_NUM_FEATURES[
        SAEWidth(turn_contract.lineage.width)
    ]

    def _resolver(entry: dict) -> Path:
        return default_target_cache_dir(
            suite=suite,
            sae_identity=identity,
            model_max_seq_len=resolved_seq_len,
            activation_label=activation_label,
            sae_width=sae_width,
        )

    return _resolver


def parse_target_real_configs_from_cfg(
    cfg,
    *,
    suite: str | None = None,
    model_max_seq_len: int | None = None,
    turn_contract: TurnContextContract | None = None,
) -> list:
    """Read ``cfg.target_real_configs`` (handling OmegaConf wrapping)
    and parse it into ``LockstepTargetRealConfig`` /
    ``PooledTargetRealConfig`` instances.

    When ``suite`` is provided AND an entry's ``cache_dir`` is ``None``
    (or missing), the cache_dir is auto-resolved via
    :func:`default_target_cache_dir` from ``cfg.conditioning.identity``,
    ``model_max_seq_len``, and the suite name. Per-script call sites
    pass ``suite="chat_target"`` (chat_eval), ``suite="core_target"``
    (base_eval CORE / dclm_eval), or ``suite="samples"`` (base_eval
    Sample). When ``suite`` is ``None``, missing/null cache_dir raises
    the existing 'missing cache_dir' error.

    Args:
        cfg: A Hydra/OmegaConf config or plain dict.
        suite: Optional suite name. When provided, enables
            null-cache_dir auto-resolution. When ``None``, the parser
            requires every entry to have an explicit cache_dir.
        model_max_seq_len: Explicit context length passed to the
            auto-resolver. When ``None`` (default), the resolver falls
            back to ``cfg.get("model_max_seq_len", 1024)``. Pass
            explicitly from training_progression where the cfg layout
            doesn't carry the field; the value should match
            ``model.dec_config.sequence_len``.
    """
    if turn_contract is not None and suite != "chat_target":
        raise ValueError("turn target-real supports only the chat_target suite")
    raw = cfg.get("target_real_configs", [])
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    elif raw is None:
        raw = []
    raw_list = list(raw) if raw else []
    if suite is None:
        return parse_target_real_configs(raw_list)
    # When suite is provided, build the auto-resolver lazily — only when
    # at least one entry actually needs it. This way operators with all
    # explicit cache_dirs don't trip identity-required errors.
    needs_resolver = any(_entry_needs_resolver(entry) for entry in raw_list)
    if not needs_resolver:
        return parse_target_real_configs(raw_list)
    resolver = _build_auto_resolver(
        cfg,
        suite=suite,
        model_max_seq_len=model_max_seq_len,
        turn_contract=turn_contract,
    )
    return parse_target_real_configs(raw_list, auto_resolve_cache_dir=resolver)


def _entry_needs_resolver(entry) -> bool:
    """True when the entry is a dict whose cache_dir is missing/None."""
    if not isinstance(entry, dict):
        return False
    if "cache_dir" not in entry:
        return True
    return entry["cache_dir"] is None


def assert_target_real_requires_encdec(target_real_configs, is_encdec: bool) -> None:
    """Raise ``ValueError`` with a unified, operator-actionable message
    when ``target_real_configs`` is non-empty but the model is not an
    encoder-decoder.
    """
    if target_real_configs and not is_encdec:
        raise ValueError(
            "target_real_configs requires an encoder-decoder model "
            "(use source=encdec or similar). Decoder-only and HuggingFace models "
            "do not support feature conditioning."
        )


def build_target_real_payload(
    trc,
    body_dict: dict,
    *,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> dict:
    """Construct the per-config yaml payload for ad-hoc target_real evals.

    Schema: SuiteResultVariant (matches training_progression's results.json).
    The eval-specific result body is stored under seeds["adhoc"].

    Callers populate ``body_dict`` with eval-kind-specific keys (results /
    centered_results / core_metric / samples / etc.).
    """
    target_conditioning_mode = TargetConditioningMode.parse(target_mode).value
    if isinstance(trc, PooledTargetRealConfig):
        variant = TargetRealPooledSuiteResult(
            name=trc.name,
            target_conditioning_mode=target_conditioning_mode,
            cache_dir=str(trc.cache_dir),
            pooling_op=trc.pooling_op.value,
            seeds={"adhoc": body_dict},
        )
    elif isinstance(trc, LockstepTargetRealConfig):
        variant = TargetRealLockstepSuiteResult(
            name=trc.name,
            target_conditioning_mode=target_conditioning_mode,
            cache_dir=str(trc.cache_dir),
            seeds={"adhoc": body_dict},
        )
    else:
        raise TypeError(f"unsupported TargetRealConfig variant: {type(trc).__name__}")
    return dump_suite_to_dict(variant)


def run_target_real_evals(
    *,
    target_real_configs: Iterable,
    cfg,
    output_dir_base: str,
    ddp_rank: int,
    per_config_eval_fn: Callable[[Any, Any], dict],
    checkpoint_dir: str | None = None,
    report_section_prefix: str = "target_real eval",
    turn_contract: TurnContextContract | None = None,
    expected_contract_metadata: dict[str, Any] | None = None,
    preflight_fn: Callable[[Any], Any] | None = None,
) -> None:
    """Orchestrate the per-target-real-config eval loop.

    Rank-0-only — non-zero ranks skip provider construction AND I/O
    entirely. The per-example loop is sequential by construction;
    fan-out across ranks is a follow-up.

    For each ``trc``:
    * Build the per-config provider via
      :func:`build_context_feature_provider`.
    * Call ``per_config_eval_fn(trc, provider)`` and obtain the result
      dict (per-eval-mode logic lives in the caller).
    * Write the result to ``<output_dir_base>/target_real/<trc.name>.yaml``.
    * If ``checkpoint_dir`` is provided, log a report entry via
      ``get_report(checkpoint_dir).log(...)`` with the ``kind``
      discriminator so that report consumers (analyze_sweep, progression
      plots) can filter by variant.

    The result dict from ``per_config_eval_fn`` is used as-is for the
    yaml payload (caller is responsible for including ``name``,
    ``cache_dir``, and the eval-mode-specific fields). The variant class
    (PooledTargetRealConfig vs LockstepTargetRealConfig) discriminates pooled-vs-lockstep
    internally — callers don't need a separate field for it.

    Args:
        checkpoint_dir: when provided, ``get_report`` is called after each
            successful yaml write to stamp the ``kind`` discriminator onto
            the report.
        report_section_prefix: prefix for the report section name.
            The final section name is ``"<prefix> (<trc.name>)"``.
        preflight_fn: optional cache validation for all configs before any
            evaluation or publication. Prepared providers are reused.
    """
    trc_list = list(target_real_configs)
    if not trc_list:
        return
    if ddp_rank != 0:
        return
    # ChatCORE range coverage must hold for every selected cache before any
    # configuration starts forwarding. Fixed callers retain lazy construction.
    preflight_providers = []
    if preflight_fn is not None:
        for trc in trc_list:
            try:
                provider = build_context_feature_provider(
                    cfg, cache_dir=trc.cache_dir, error_prefix=f"target_real_configs[{trc.name}]",
                    expected_contract_metadata=expected_contract_metadata,
                    turn_contract=turn_contract,
                )
                preflight_fn(provider)
            except Exception as exc:
                raise RuntimeError(f"target_real_configs[{trc.name}] failed before evaluation: {exc}") from exc
            preflight_providers.append(provider)
    output_dir = os.path.join(output_dir_base, "target_real")
    os.makedirs(output_dir, exist_ok=True)
    for index, trc in enumerate(trc_list):
        logger.info(f"\n--- Target-real config: {trc.name} ({type(trc).__name__}) ---")
        try:
            provider = preflight_providers[index] if preflight_fn is not None else build_context_feature_provider(
                cfg,
                cache_dir=trc.cache_dir,
                error_prefix=f"target_real_configs[{trc.name}]",
                expected_contract_metadata=expected_contract_metadata,
                turn_contract=turn_contract,
            )
            result = per_config_eval_fn(trc, provider)
            output_path = os.path.join(output_dir, f"{trc.name}.yaml")
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(result, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            # Tag the failing trc.name into the exception context so the
            # operator can pinpoint which target_real_configs entry tripped
            # the failure. Partial yamls from earlier (successful) trcs may
            # exist on disk — that's intentional (operators can re-run only
            # the failing entry by trimming target_real_configs).
            raise RuntimeError(
                f"target_real_configs[{trc.name}] failed during evaluation; "
                f"partial yamls from prior configs may exist under "
                f"{output_dir}/. Original error: {type(e).__name__}: {e}"
            ) from e
        logger.info(f"target_real['{trc.name}'] results written to {output_path}")
        if checkpoint_dir is not None:
            kind_str = (
                "target_real_pooled"
                if isinstance(trc, PooledTargetRealConfig)
                else "target_real_lockstep"
            )
            section_name = f"{report_section_prefix} ({trc.name})"
            get_report(checkpoint_dir=checkpoint_dir).log(
                section=section_name,
                data=[{"kind": kind_str}],
            )


# ---------------------------------------------------------------------------
# PR-D Batch H2: shared per-example helpers for target_real loops.
#
# The progression collect-loop (Batch H2 fix) needs the same per-example
# wrapper rebuild logic that ``chat_eval._chat_eval_target_real_per_config``
# already implements. Lifting these primitives here lets progression reuse
# the proven path instead of re-deriving it (DRY).
# ---------------------------------------------------------------------------


def lookup_target_cache_entry_or_raise(
    provider: ContextSegmentProvider,
    full_token_ids: list[int],
    *,
    trc_name: str,
) -> ContextCacheEntry:
    """Look up a target cache row by EXACT full-conversation token ids.

    Raises ``KeyError`` with operator-actionable cache-build instructions on
    miss. Mirrors the behavior previously inlined in chat_eval and base_eval
    so all target_real call sites surface the same diagnostic.
    """
    segments = provider.get(full_token_ids)
    if segments is None:
        raise KeyError(
            f"target_real_configs[{trc_name}]: no cache row found for "
            f"{token_context_key(full_token_ids)} "
            f"(token count={len(full_token_ids)}). The target cache must "
            f"contain one row per eval example with EXACT full-conversation "
            f"token ids as the key.\n\n"
            f"{target_real_cache_generation_instructions()}"
        )
    return ContextCacheEntry(
        token_ids=tuple(int(tok) for tok in full_token_ids),
        segments=segments,
    )


def derive_target_char_range_from_mask(
    tokenizer,
    full_token_ids: list[int],
    mask: list[int],
) -> tuple[int, int]:
    """Mirror :func:`enumerate_chat_target_contexts`: target span = char range
    covered by ``mask==1`` positions. Raises ``ValueError`` if no mask=1
    tokens exist (no assistant content)."""
    char_offsets = token_to_char_offsets(tokenizer, full_token_ids)
    mask_1_indices = [i for i, m in enumerate(mask) if m == 1]
    if not mask_1_indices:
        raise ValueError(
            "target_real eval: example has no mask=1 tokens (no assistant "
            "content); cannot derive target_char range"
        )
    target_char_start = int(char_offsets[mask_1_indices[0]])
    target_char_end = int(char_offsets[mask_1_indices[-1] + 1])
    return target_char_start, target_char_end


def core_row_char_spans(
    tokenizer,
    tokens,
    start_idxs,
    end_idxs,
) -> list[tuple[int, int]]:
    """Target char span for each rendered CORE row.

    The span is ``[len(decode(row[:si])), len(decode(row[:ei])))`` — i.e. the
    characters covered by the answer/continuation, which is what
    :func:`steering.eval.target_segments.filter_segments_to_target` slices the
    cached feature segments against.
    """
    return [
        (
            len(tokenizer.decode(list(tok_seq)[:si])),
            len(tokenizer.decode(list(tok_seq)[:ei])),
        )
        for tok_seq, si, ei in zip(tokens, start_idxs, end_idxs, strict=True)
    ]


def should_skip_empty_target_core_example(
    row_char_spans: Sequence[tuple[int, int]],
) -> bool:
    """Return True when any rendered row's continuation decodes to 0 characters.

    ``filter_segments_to_target`` rejects an empty target span, which is correct:
    there are no feature segments to pool. But for a CORE row that is a property
    of the *benchmark data* (a continuation that tokenizes to characterless
    tokens), not an invariant violation, and aborting a 22-task eval over one
    degenerate row is the wrong trade.

    As with :func:`nanochat.core_eval.should_skip_oversized_core_example`, the
    whole example is skipped rather than the offending row: multiple-choice and
    schema examples render several rows, and dropping one would change the
    answer set and corrupt the comparison.

    Callers must count what they skip; a metric computed over a heavily-skipped
    task set is not comparable to a clean run.
    """
    return any(end <= start for start, end in row_char_spans)


def build_target_real_artifacts_for_example(
    *,
    model,
    tokenizer,
    trc,
    provider: ContextSegmentProvider,
    full_token_ids: list[int],
    mask: list[int],
    target_char_range: tuple[int, int] | None = None,
    target_token_loss_mask: Sequence[int] | None = None,
    context_token_ids: list[int] | None = None,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
    turn_contract: TurnContextContract | None = None,
) -> TargetRealExampleArtifacts:
    """Per-example target_real target + wrapper rebuild primitive.

    For one example, this:
      1. Derives ``target_char_start/end`` from ``mask==1`` tokens unless an
         explicit ``target_char_range`` is supplied.
      2. Looks up the target cache entry via ``provider.get(full_token_ids)``.
      3. Builds a target via :func:`build_target_for_real_config`.
      4. Builds a fresh wrapper via :func:`build_wrapper_for_features`
         with the explicit ``target=`` kwarg.

    The exact prompt/context prefix cache row is exposed as a one-row context
    provider. This conditions prompt tokens on real extracted features while
    keeping the target span routed through the explicit pooled/lockstep target
    slots.

    Returns the derived target, wrapper, spans, and context tokens. The caller
    is responsible for forwarding ``input_ids`` / ``target_mask`` /
    ``context_token_ids`` through the wrapper's ``conditioning_context``.

    Raises:
        KeyError: cache row missing for these token_ids.
    """
    provider_contract = getattr(provider, "turn_contract", None)
    if turn_contract is not None:
        if provider_contract is None:
            raise ValueError("turn target-real requires a validated schema-5 provider")
        validate_turn_context_contract(provider_contract.semantic, turn_contract)
        if context_token_ids is None:
            context_token_ids = context_tokens_from_target_mask(full_token_ids, mask)
    elif provider_contract is not None:
        raise ValueError("turn provider requires an explicit checkpoint-derived turn contract")
    if target_char_range is None:
        target_char_start, target_char_end = derive_target_char_range_from_mask(
            tokenizer,
            full_token_ids,
            mask,
        )
    else:
        if len(target_char_range) != 2:
            raise ValueError(
                "target_real target_char_range must contain exactly two "
                f"values, got {target_char_range!r}"
            )
        target_char_start, target_char_end = map(int, target_char_range)
        if target_char_start < 0 or target_char_end <= target_char_start:
            raise ValueError(
                "target_real target_char_range must be a positive range "
                f"[start, end), got [{target_char_start}, {target_char_end})"
            )
    if target_token_loss_mask is None:
        resolved_target_token_loss_mask = list(mask)
    else:
        resolved_target_token_loss_mask = [int(value) for value in target_token_loss_mask]
        if len(resolved_target_token_loss_mask) != len(full_token_ids):
            raise ValueError(
                "target_real target_token_loss_mask length mismatch: "
                f"{len(resolved_target_token_loss_mask)} vs "
                f"{len(full_token_ids)} full token IDs"
            )
    entry = lookup_target_cache_entry_or_raise(
        provider,
        full_token_ids,
        trc_name=trc.name,
    )
    if context_token_ids is None:
        context_token_ids = context_tokens_from_target_mask(full_token_ids, mask)
    if turn_contract is None:
        target = build_target_for_real_config(
            cfg=trc,
            entry=entry,
            target_char_start=target_char_start,
            target_char_end=target_char_end,
            full_token_ids=full_token_ids,
            target_token_loss_mask=resolved_target_token_loss_mask,
            tokenizer=tokenizer,
        )
    else:
        target = build_turn_target_for_real_config(
            cfg=trc,
            entry=entry,
            target_char_start=target_char_start,
            target_char_end=target_char_end,
            full_token_ids=full_token_ids,
            target_token_loss_mask=resolved_target_token_loss_mask,
            tokenizer=tokenizer,
            turn_contract=turn_contract,
            context_token_ids=context_token_ids,
        )
    context_segments = context_segments_for_target_real_prefix(
        provider=provider,
        context_token_ids=context_token_ids,
        trc_name=trc.name,
    )
    context_provider = SingleEntryContextProvider(
        context_token_ids,
        context_segments,
        turn_contract=turn_contract,
    )
    wrapper = build_wrapper_for_features(
        model,
        tokenizer,
        target=target,
        context_provider=context_provider,
        target_mode=target_mode,
        context_routing_mode=(ContextRoutingMode.DECODER_ALIGNED if turn_contract is None
                              else ContextRoutingMode.TURN_RAW_SOURCE),
        turn_contract=turn_contract,
    )
    return TargetRealExampleArtifacts(
        target=target,
        wrapper=wrapper,
        context_token_ids=list(context_token_ids),
        target_char_start=target_char_start,
        target_char_end=target_char_end,
    )
