"""Synth QC predicates for positive-mode and negative-mode continuations.

Public helpers:

- :func:`classify_avoid_continuation` is the strict-zero avoid QC for
  both positive-mode (issue #513) and negative-mode (issue #518). It
  tokenizes ``(prefix, continuation)`` ONCE and returns a tri-state
  :class:`AvoidContinuationOutcome`: ``NO_TOKENS`` (the continuation
  adds no tokens beyond the prefix — the QC trap, never a strict-zero
  PASS), ``INACTIVE`` (the strict-zero PASS condition), or ``ACTIVE``
  (the concept feature fires in the continuation slice).
- :func:`compute_fc_percentiles` preserves the joined-sequence pair-scoring
  API used by diagnostics and tests. Positive Stage 1b instead calls
  :func:`output_fc_percentile_for_continuation` and subtracts the exact
  prefix-alone score already computed by :class:`PrefixPercentileCache`.
- :func:`output_fc_percentiles_for_continuations` and
  :func:`classify_avoid_continuations` are caller-thread ragged-batch seams for
  post-provider Stage 1 QC. They reuse the same tokenizer, BOS/slice contract,
  and extraction backend as the scalar reference functions above.
- :func:`fc_percentile_on_prefix` returns the max-over-prefix-tokens
  percentile for a concept feature.

These helpers share a re-tokenization pattern: the prefix and
continuation are re-tokenized via the extraction tokenizer
(``google/gemma-2-2b`` by default), which is distinct from the
nanochat tokenizer used by seed pools. The prefix is tokenized alone
to compute its token length, then ``prefix + continuation`` is
tokenized as a single string to preserve any merges across the seam.
Exactly one leading BOS is then prepended before every SAE forward
pass (matching :func:`steering.extract_features`'s ``real_acts =
acts[1:]`` convention, via
:func:`steering.eval.axbench_base.extraction._bos_prepended_model_input`),
so every prefix/continuation slice is offset by ``+1`` to exclude the
BOS row. The continuation tokens are the tail of the combined
sequence, after that ``+1`` BOS offset.
Both positive-mode avoid tiers (issue #513) and negative-mode
``concept_avoiding`` QC (issue #518) classify via
:func:`classify_avoid_continuation`.

**Percentile currency (deliberate asymmetry).** Every percentile this module
computes -- via :func:`_max_percentile`, :func:`_max_percentiles_for_feature`,
and the clustered :func:`_max_percentile_for_target` /
:func:`_max_percentiles_for_target` -- is the raw, unquantized ``[0.5, 99.5]``
scalar :func:`steering.conditional.percentile_table.magnitude_to_percentile`
returns. This is intentionally DIFFERENT from the lossy uint8-quantized
round-trip (:func:`steering.conditional.percentile_table.percentile_to_uint8`
/ ``uint8_to_percentile``) that training and packed-rows data use to store
percentile weights compactly on disk (``~0.39%``-wide bins). QC's
accept/reject decisions convert the configured fractional thresholds to raw
percentile points before comparison. Those comparisons and the strict "zero
means never fired" floor are precision-sensitive at exactly the boundary the
uint8 quantization would blur, so QC must NEVER round-trip its scores through
the uint8 encoding. Both the legacy scalar path and the clustered member-set
path share this same unquantized currency
(:func:`_max_percentile_for_target` calls the same
:func:`magnitude_to_percentile` primitive per member, never
``percentile_to_uint8``/``uint8_to_percentile``). Do NOT "fix" QC to uint8; the
asymmetry is by design (see the cross-cutting "percentile unit asymmetry /
QC-unquantized-vs-training-uint8" item in the PR #739 split campaign).
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch as t
from loguru import logger

from steering.conditional.percentile_table import (
    compute_percentile_checksum,
    magnitude_to_percentile,
)
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.extraction import (
    BosPrependedActivations,
    _bos_prepended_model_input,
    _run_batched_extractor,
    _sae_hook_name,
    warn_on_special_id_pollution,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth.response_parser import (
    DEFAULT_EXTRACTION_TOKENIZER_ID,
    get_extraction_tokenizer,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.sae_resolver import ActivationType
from steering.segmentation import resolve_bos_token_id

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureTarget

DEFAULT_PREFIX_PERCENTILE_BATCH_SIZE = 8
CLUSTER_QC_TARGET_ALGORITHM_VERSION = "cluster_target_percentile_amax_v1"


class AvoidContinuationOutcome(enum.StrEnum):
    """Tri-state result of the avoid strict-zero QC (issues #513, #518).

    NO_TOKENS : continuation contributes no tokens beyond the prefix
                (``n_prefix >= len(full_tokens)``); the QC trap — callers must
                NOT treat this as a strict-zero PASS. No SAE forward pass runs.
    INACTIVE  : ``f_c`` does not fire above ``extraction_threshold`` anywhere in
                the continuation slice (the strict-zero PASS condition).
    ACTIVE    : ``f_c`` fires above ``extraction_threshold`` at >= 1 continuation
                token position.
    """

    NO_TOKENS = "no_tokens"
    INACTIVE = "inactive"
    ACTIVE = "active"


@dataclass(frozen=True)
class ContinuationQcInput:
    """One aligned prefix/continuation target for batched Stage 1 QC."""

    prefix_text: str
    continuation_text: str
    feature_id: int
    cluster_target: ClusterFeatureTarget | None = None


@dataclass(frozen=True)
class PrefixPercentileCacheKey:
    """Deterministic identity for a prefix-percentile QC result.

    The cache stores only the final percentile for one
    ``(prefix_tokens, feature_id)`` pair. It never stores or reuses LLM
    responses, so Stage 1b retry cache keys and retry semantics remain
    unchanged.
    """

    model_id: str
    n_ctx: int
    sae_id: str
    sae_layer: int
    tokenizer_id: str
    percentile_namespace: str
    feature_id: int
    token_count: int
    token_sha256: str


@dataclass(frozen=True)
class ClusterPrefixPercentileCacheKey:
    """Cluster-only cache identity; the legacy scalar key remains untouched."""

    model_id: str
    n_ctx: int
    sae_id: str
    sae_layer: int
    tokenizer_id: str
    percentile_namespace: str
    percentile_table_checksum: str
    representative_id: int
    allowed_member_ids: tuple[int, ...]
    target_algorithm_version: str
    feature_contract_fingerprint: str
    token_count: int
    token_sha256: str


def _token_ids_sha256(token_ids: list[int]) -> str:
    h = hashlib.sha256()
    for token_id in token_ids:
        h.update(int(token_id).to_bytes(8, byteorder="little", signed=False))
    return h.hexdigest()


def _metadata_get(obj: Any, key: str) -> Any:
    metadata = getattr(obj, "metadata", None)
    if metadata is None:
        return None
    getter = getattr(metadata, "get", None)
    if callable(getter):
        return getter(key)
    try:
        return metadata[key]
    except (KeyError, TypeError):
        return None


def _model_cache_id(base_model: Any) -> str:
    cfg = getattr(base_model, "cfg", None)
    for attr in (
        "model_name",
        "official_model_name",
        "model_path",
        "name_or_path",
        "model_id",
    ):
        value = getattr(cfg, attr, None) or getattr(base_model, attr, None)
        if value:
            return str(value)
    return type(base_model).__qualname__


def _sae_cache_id(base_sae: Any) -> str:
    cfg = getattr(base_sae, "cfg", None)
    hook_name = _sae_hook_name(base_sae)
    release = (
        getattr(cfg, "release", None)
        or _metadata_get(cfg, "neuronpedia_id")
        or _metadata_get(cfg, "release")
        or "unknown-release"
    )
    sae_id = (
        getattr(cfg, "sae_id", None)
        or getattr(cfg, "id", None)
        or _metadata_get(cfg, "sae_id")
        or _metadata_get(cfg, "id")
        or "unknown-sae"
    )
    n_features = (
        getattr(cfg, "d_sae", None)
        or getattr(cfg, "n_features", None)
        or _metadata_get(cfg, "d_sae")
        or _metadata_get(cfg, "n_features")
        or "unknown-n_features"
    )
    return f"{release}|{sae_id}|{hook_name}|{n_features}"


def _base_context_window(base_model: Any) -> int:
    n_ctx = getattr(getattr(base_model, "cfg", None), "n_ctx", None)
    if n_ctx is None:
        raise AxBenchBaseError(
            "synth QC: base_model has no .cfg.n_ctx attribute; cannot "
            "enforce exact context-window bounds. Verify that base_model is a "
            "HookedTransformer-compatible instance.",
        )
    return int(n_ctx)


def make_prefix_percentile_cache_key(
    *,
    token_ids: list[int],
    f_c: int,
    base_model: Any,
    base_sae: Any,
    sae_layer: int,
    tokenizer_id: str,
    percentile_namespace: str,
) -> PrefixPercentileCacheKey:
    """Build the cache key for a prefix percentile lookup.

    ``token_ids`` must be the exact prepared model input and must fit the
    base-model context window. Callers fail loudly rather than key or score a
    truncated prefix.
    """
    return PrefixPercentileCacheKey(
        model_id=_model_cache_id(base_model),
        n_ctx=_base_context_window(base_model),
        sae_id=_sae_cache_id(base_sae),
        sae_layer=int(sae_layer),
        tokenizer_id=str(tokenizer_id),
        percentile_namespace=str(percentile_namespace),
        feature_id=int(f_c),
        token_count=len(token_ids),
        token_sha256=_token_ids_sha256(token_ids),
    )


def make_cluster_prefix_percentile_cache_key(
    *,
    token_ids: list[int],
    target: ClusterFeatureTarget,
    base_model: Any,
    base_sae: Any,
    sae_layer: int,
    tokenizer_id: str,
    percentile_namespace: str,
    percentile_table_checksum: str,
) -> ClusterPrefixPercentileCacheKey:
    return ClusterPrefixPercentileCacheKey(
        model_id=_model_cache_id(base_model),
        n_ctx=_base_context_window(base_model),
        sae_id=_sae_cache_id(base_sae),
        sae_layer=int(sae_layer),
        tokenizer_id=str(tokenizer_id),
        percentile_namespace=str(percentile_namespace),
        percentile_table_checksum=str(percentile_table_checksum),
        representative_id=int(target.representative_id),
        allowed_member_ids=tuple(int(x) for x in target.allowed_member_ids),
        target_algorithm_version=CLUSTER_QC_TARGET_ALGORITHM_VERSION,
        feature_contract_fingerprint=target.feature_contract_fingerprint,
        token_count=len(token_ids),
        token_sha256=_token_ids_sha256(token_ids),
    )


def _run_sae_on_token_lists(
    token_id_lists: list[list[int]],
    base_model: Any,
    base_sae: Any,
    *,
    sae_layer: int,
) -> list[t.Tensor]:
    """Run one or more token lists through the base Gemma + SAE path.

    This is the shared implementation behind the old single-example
    ``_run_sae_on_full`` helper and the Stage 1b prefix cache's batched
    miss path. Each sequence is clamped exactly as the old single-example
    helper did before forwarding to the extractor.
    """
    if not token_id_lists:
        return []
    n_ctx = _base_context_window(base_model)
    clamped: list[list[int]] = []
    for tokens in token_id_lists:
        if not tokens:
            raise AxBenchBaseError("synth QC: refusing to run SAE on zero tokens")
        clamped_tokens = tokens
        if len(tokens) > n_ctx:
            logger.debug(
                f"synth QC: clamping {len(tokens)} tokens to n_ctx={n_ctx}",
            )
            clamped_tokens = tokens[:n_ctx]
        clamped.append(clamped_tokens)

    token_counts = [len(tokens) for tokens in clamped]
    with profile_region(
        "synth_qc.run_sae_on_full",
        rows=len(clamped),
        token_count=sum(token_counts),
        min_token_count=min(token_counts),
        max_token_count=max(token_counts),
        sae_layer=sae_layer,
    ):
        acts_list = _run_batched_extractor(
            model=base_model,
            sae=base_sae,
            token_id_lists=clamped,
            layer=sae_layer,
            hook_name=_sae_hook_name(base_sae),
            activation_type=ActivationType.residual,
        )
    if not acts_list:
        raise AxBenchBaseError(
            "synth QC: batched extractor returned no activation tensors"
        )
    if len(acts_list) != len(clamped):
        raise AxBenchBaseError(
            "synth QC: batched extractor returned "
            f"{len(acts_list)} activation tensors for {len(clamped)} token lists"
        )
    return acts_list


def _max_percentiles_for_feature(
    acts_list: list[t.Tensor], f_c: int, percentile_table: t.Tensor,
) -> list[float]:
    """Vectorized equivalent of ``_max_percentile(acts[:, f_c], ...)``."""
    if not acts_list:
        return []
    max_mags: list[t.Tensor] = []
    for acts in acts_list:
        if acts.shape[0] == 0:
            device = acts.device if isinstance(acts, t.Tensor) else None
            max_mags.append(t.tensor(0.0, device=device))
        else:
            max_mags.append(acts[:, f_c].max())
    mags_cpu = t.stack(max_mags).detach().to(dtype=t.float32, device="cpu")
    out = t.zeros(len(acts_list), dtype=t.float64)
    positive = mags_cpu > 0
    if positive.any():
        fids = t.full(
            (int(positive.sum().item()),), int(f_c), dtype=t.int64,
        )
        out[positive] = magnitude_to_percentile(
            fids,
            mags_cpu[positive],
            percentile_table,
        )
    return [float(x) for x in out.tolist()]


def _validate_cluster_target(
    target: ClusterFeatureTarget,
    *,
    representative_id: int,
    n_features: int,
) -> tuple[int, ...]:
    if int(target.representative_id) != int(representative_id):
        raise AxBenchBaseError(
            "cluster target representative does not match the public feature ID: "
            f"target={target.representative_id}, public={representative_id}"
        )
    members = tuple(int(x) for x in target.allowed_member_ids)
    if not members or members != tuple(sorted(set(members))):
        raise AxBenchBaseError(
            "cluster target allowed_member_ids must be non-empty, unique, and sorted"
        )
    invalid = [feature_id for feature_id in members if not 0 <= feature_id < n_features]
    if invalid:
        raise AxBenchBaseError(
            "cluster target member IDs are outside the SAE/statistics width "
            f"{n_features}: {invalid[:10]}"
        )
    if int(representative_id) not in members:
        raise AxBenchBaseError(
            "cluster target allowed_member_ids must contain its representative"
        )
    return members


def _max_percentile_for_target(
    activations: t.Tensor,
    target: ClusterFeatureTarget,
    percentile_table: t.Tensor,
) -> float:
    """Per-member own-statistic percentile followed by member ``amax``."""

    members = _validate_cluster_target(
        target,
        representative_id=target.representative_id,
        n_features=int(percentile_table.shape[0]),
    )
    if activations.ndim != 2:
        raise AxBenchBaseError(
            "cluster target percentile requires [tokens, features] activations"
        )
    if int(activations.shape[1]) != int(percentile_table.shape[0]):
        raise AxBenchBaseError(
            "cluster target activation width does not match percentile table"
        )
    if activations.shape[0] == 0:
        return 0.0
    member_tensor = t.tensor(members, dtype=t.int64, device=activations.device)
    max_magnitudes = activations[:, member_tensor].amax(dim=0).detach().to("cpu")
    firing = max_magnitudes > 0
    if not bool(firing.any()):
        return 0.0
    firing_ids = member_tensor.detach().to("cpu")[firing]
    percentiles = magnitude_to_percentile(
        firing_ids,
        max_magnitudes[firing].to(t.float32),
        percentile_table.to("cpu"),
    )
    return float(percentiles.max().item())


def _max_percentiles_for_target(
    acts_list: list[t.Tensor],
    target: ClusterFeatureTarget,
    percentile_table: t.Tensor,
) -> list[float]:
    return [
        _max_percentile_for_target(acts, target, percentile_table)
        for acts in acts_list
    ]


class PrefixPercentileCache:
    """In-memory cache for canonical Stage 1b prefix-percentile scoring."""

    def __init__(
        self,
        *,
        base_model: Any,
        base_sae: Any,
        percentile_table: t.Tensor,
        sae_layer: int,
        tokenizer_id: str,
        percentile_namespace: str,
        batch_size: int = DEFAULT_PREFIX_PERCENTILE_BATCH_SIZE,
        store: Any | None = None,
    ) -> None:
        self.base_model = base_model
        self.base_sae = base_sae
        self.percentile_table = percentile_table
        self.sae_layer = int(sae_layer)
        self.tokenizer_id = str(tokenizer_id)
        self.percentile_namespace = str(percentile_namespace)
        self._percentile_table_checksum: str | None = None
        self.batch_size = max(1, int(batch_size))
        # Optional durable backing. The in-memory dict stays authoritative
        # within a process; the store only lets a hit survive process exit.
        # Scores are a pure function of the key below, so a persisted hit is
        # exactly the float the SAE forward would have produced.
        self._store = store
        self._values: dict[
            PrefixPercentileCacheKey | ClusterPrefixPercentileCacheKey,
            float,
        ] = {}

    def get_many(
        self,
        prefix_texts: list[str],
        f_c: int,
        *,
        cluster_target: ClusterFeatureTarget | None = None,
    ) -> list[float]:
        """Return prefix percentiles in the same order as ``prefix_texts``.

        Injects exactly one leading BOS into each prefix's tokens before
        the SAE forward and excludes that BOS row from the percentile
        pool (``acts[1:, f_c]`` -- see :func:`_max_percentiles_for_feature`
        callers below). The ``<bos>`` token is a large attention-sink
        outlier; pooling it in would be out-of-distribution relative to
        the concept features this is compared against, matching canonical
        feature extraction's ``real_acts = acts[1:]`` convention.
        """
        if not prefix_texts:
            return []

        results: list[float | None] = [None] * len(prefix_texts)
        # Values computed in THIS call, flushed to the durable store once at
        # the end so a batch costs one transaction rather than one per row.
        newly_computed: list[tuple[Any, float]] = []
        pending: dict[
            PrefixPercentileCacheKey | ClusterPrefixPercentileCacheKey,
            tuple[list[int], list[int]],
        ] = {}

        tokenizer = None
        if any(prefix_texts):
            tokenizer = get_extraction_tokenizer(self.tokenizer_id)
        n_ctx: int | None = None
        bos_id: int | None = None
        call_hits = 0

        with profile_region(
            "synth_qc.tokenize_prefix_batch",
            rows=len(prefix_texts),
            feature_id=f_c,
        ):
            for idx, prefix_text in enumerate(prefix_texts):
                # These are CORPUS seeds (decoded from seed prefix_tokens), not
                # concept descriptions or LLM output. A corpus seed should
                # never be empty, so an empty one is a real data bug — raise loudly
                # rather than emitting a silent zero-percentile row that would
                # mis-classify the seed as clean.
                if not prefix_text:
                    raise AxBenchBaseError(
                        "synth QC prefix-percentile cache: a corpus seed has "
                        "empty prefix text; a corpus seed should never be empty."
                    )
                assert tokenizer is not None
                tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                if not tokens:
                    raise AxBenchBaseError(
                        "synth QC prefix-percentile cache: a corpus seed's "
                        f"prefix text tokenized to zero tokens (text: {prefix_text[:80]!r}); "
                        "a corpus seed should never be empty."
                    )
                if n_ctx is None:
                    n_ctx = _base_context_window(self.base_model)
                if bos_id is None:
                    resolved_bos = resolve_bos_token_id(tokenizer)
                    if resolved_bos is None:
                        raise AxBenchBaseError(
                            "synth QC prefix-percentile cache: extraction "
                            "tokenizer exposes no BOS id; the canonical "
                            "feature extraction prepends a BOS and ignores "
                            "its features, so scoring prefixes without one "
                            "would be out-of-distribution. A missing Gemma "
                            "BOS is a real config bug -- refusing to proceed."
                        )
                    bos_id = int(resolved_bos)
                # A literal special-token string in the seed (e.g. '<bos>')
                # re-encodes to the real id even with add_special_tokens=False.
                # These seeds ARE natural corpus text, so we PRESERVE their
                # original tokens and prepend our BOS anyway (a '<bos>'-starting
                # seed intentionally double-BOSes); warn for observability rather
                # than dropping the seed from the cleanliness scoring.
                warn_on_special_id_pollution(
                    tokens,
                    tokenizer=tokenizer,
                    bos_id=bos_id,
                    context="synth QC prefix-percentile cache",
                    text=prefix_text,
                )
                model_input = [bos_id] + [int(tok) for tok in tokens]
                if len(model_input) > n_ctx:
                    raise AxBenchBaseError(
                        "synth QC prefix-percentile cache: prepared prefix is "
                        f"{len(model_input)} tokens including BOS > n_ctx={n_ctx}; "
                        "refusing to score a truncated prefix"
                    )
                if cluster_target is None:
                    key = make_prefix_percentile_cache_key(
                        token_ids=model_input,
                        f_c=f_c,
                        base_model=self.base_model,
                        base_sae=self.base_sae,
                        sae_layer=self.sae_layer,
                        tokenizer_id=self.tokenizer_id,
                        percentile_namespace=self.percentile_namespace,
                    )
                else:
                    _validate_cluster_target(
                        cluster_target,
                        representative_id=f_c,
                        n_features=int(self.percentile_table.shape[0]),
                    )
                    if self._percentile_table_checksum is None:
                        self._percentile_table_checksum = compute_percentile_checksum(
                            self.percentile_table
                        )
                    key = make_cluster_prefix_percentile_cache_key(
                        token_ids=model_input,
                        target=cluster_target,
                        base_model=self.base_model,
                        base_sae=self.base_sae,
                        sae_layer=self.sae_layer,
                        tokenizer_id=self.tokenizer_id,
                        percentile_namespace=self.percentile_namespace,
                        percentile_table_checksum=self._percentile_table_checksum,
                    )
                cached = self._values.get(key)
                if cached is None and self._store is not None:
                    cached = self._store.get(key)
                    if cached is not None:
                        # Promote into memory so the rest of this process
                        # never pays the lookup again.
                        self._values[key] = cached
                if cached is not None:
                    call_hits += 1
                    results[idx] = cached
                    continue
                entry = pending.get(key)
                if entry is None:
                    pending[key] = (model_input, [idx])
                else:
                    entry[1].append(idx)

        miss_items = list(pending.items())
        record_counter(
            "cache.synth_qc.prefix_percentile.hit",
            call_hits,
            feature_id=f_c,
        )
        record_counter(
            "cache.synth_qc.prefix_percentile.miss",
            len(miss_items),
            feature_id=f_c,
        )

        for start in range(0, len(miss_items), self.batch_size):
            chunk = miss_items[start:start + self.batch_size]
            token_lists = [tokens for _key, (tokens, _positions) in chunk]
            acts_list: list[t.Tensor] | None = None
            with profile_region(
                "synth_qc.prefix_percentile_cache_fill",
                rows=len(chunk),
                feature_id=f_c,
                batch_size=self.batch_size,
            ):
                try:
                    acts_list = _run_sae_on_token_lists(
                        token_lists,
                        self.base_model,
                        self.base_sae,
                        sae_layer=self.sae_layer,
                    )
                    # +1: each acts tensor carries the injected BOS at row 0
                    # (token_lists are all BOS-prepended above); exclude it
                    # from the pool, matching canonical feature extraction.
                    if cluster_target is None:
                        pcts = _max_percentiles_for_feature(
                            [acts[1:] for acts in acts_list],
                            f_c,
                            self.percentile_table,
                        )
                    else:
                        pcts = _max_percentiles_for_target(
                            [acts[1:] for acts in acts_list],
                            cluster_target,
                            self.percentile_table,
                        )
                finally:
                    del acts_list
                    release_cuda_iteration_memory()
            for (key, (_tokens, positions)), pct in zip(
                chunk, pcts, strict=True,
            ):
                self._values[key] = pct
                if self._store is not None:
                    newly_computed.append((key, pct))
                for idx in positions:
                    results[idx] = pct

        if self._store is not None and newly_computed:
            self._store.put_many(newly_computed)

        missing = [idx for idx, value in enumerate(results) if value is None]
        if missing:
            raise AxBenchBaseError(
                "synth QC prefix-percentile cache failed to fill result "
                f"indices: {missing[:10]}"
            )
        filled: list[float] = []
        for value in results:
            assert value is not None
            filled.append(float(value))
        return filled


def _tokenize_prefix_and_full(
    prefix_text: str,
    continuation_text: str,
    *,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
) -> tuple[list[int], list[int]]:
    """Tokenize prefix alone and prefix+continuation together.

    Returns ``(prefix_tokens, full_tokens)``. ``prefix_tokens`` is the raw
    (no-BOS) prefix-alone encoding -- callers use only its LENGTH as the
    prefix/continuation boundary. ``full_tokens`` is the encoding of
    ``prefix_text + continuation_text`` as a single string (so a subword
    that would merge across the seam tokenizes the SAME way it will at
    scoring time) with exactly ONE leading BOS prepended.

    Because ``full_tokens`` carries a leading BOS at position 0, callers
    must read the prefix slice as ``acts[1 : 1 + len(prefix_tokens)]`` and
    the continuation slice as ``acts[1 + len(prefix_tokens):]`` -- the
    ``+1`` is the injected BOS, excluded from both spans (matching canonical
    feature extraction's ``real_acts = acts[1:]`` convention).

    Seam merges (a continuation subword gluing onto the prefix's trailing
    token, so ``len(full_ids) <= len(prefix_tokens)``) are a LEGITIMATE,
    already-handled QC outcome, NOT an error: ``classify_avoid_continuation``
    routes ``1 + n_prefix >= len(full_tokens)`` to ``NO_TOKENS``. Unlike the
    derive path (``feature_diffmean``), which RAISES on a seam merge because it
    needs a clean completion span for the DiffMean pool, QC deliberately
    tolerates it — so this function does NOT reject on a seam mismatch (a raise
    here would crash the QC stage on a normal continuation). ``n_prefix`` is
    still the boundary; a partial merge approximates the boundary exactly as
    the pre-BOS code did (out of scope for the BOS fix).

    Raises:
        AxBenchBaseError: if the combined encoding is empty (nothing to
            analyze); or if the extraction tokenizer exposes no BOS id.
    """
    with profile_region("synth_qc.tokenize_prefix_and_full"):
        tok = get_extraction_tokenizer(tokenizer_id)
        prefix_tokens = tok.encode(prefix_text, add_special_tokens=False)
        full_ids = tok.encode(
            prefix_text + continuation_text, add_special_tokens=False,
        )
    if not full_ids:
        raise AxBenchBaseError(
            "synth QC: prefix + continuation tokenized to zero tokens"
        )
    bos_id = resolve_bos_token_id(tok)
    if bos_id is None:
        raise AxBenchBaseError(
            "synth QC: extraction tokenizer exposes no BOS id; the "
            "canonical feature extraction prepends a BOS and ignores its "
            "features, so scoring without one would be out-of-distribution. "
            "A missing Gemma BOS is a real config bug -- refusing to proceed."
        )
    # In-data special-token warning (distinct from a seam merge, which QC also
    # tolerates): a literal '<bos>' string in prefix/continuation re-encodes to
    # the real BOS id even with add_special_tokens=False. These are natural
    # corpus/synth texts, so we PRESERVE their original tokens and prepend our
    # own BOS anyway (position 0, dropped by the acts[1:...] slices; the seed
    # span acts[1:1+n_prefix] and continuation acts[1+n_prefix:] therefore keep
    # any in-data special as data). A '<bos>'-starting text intentionally
    # double-BOSes. Warn for observability rather than dropping the row.
    warn_on_special_id_pollution(
        full_ids,
        tokenizer=tok,
        bos_id=bos_id,
        context="synth QC prefix+continuation",
        text=prefix_text + continuation_text,
    )
    full_tokens = [int(bos_id), *full_ids]  # our BOS at position 0
    return prefix_tokens, full_tokens


def _run_sae_on_full(
    full_tokens: list[int],
    base_model: Any,
    base_sae: Any,
    *,
    sae_layer: int,
) -> t.Tensor:
    """Run the base Gemma + SAE forward pass and return the full activation tensor.

    Returns a ``(seq_len, n_sae_features)`` float tensor. Raises
    ``AxBenchBaseError`` if the extractor returns an empty list.

    Clamps ``full_tokens`` to ``base_model.cfg.n_ctx`` before the forward
    pass — mirrors the clamp in
    :func:`extraction._bos_prepended_model_input`. Without this, a long seed +
    retry-generated long continuation can exceed
    the model's context window and crash mid-stage after paying for
    LLM calls. The ``cfg.extraction_max_tokens`` soft cap is the
    caller's responsibility; we enforce only the hard ``n_ctx`` here
    so QC calls never blow through the model's architectural limit.
    """
    acts_list = _run_sae_on_token_lists(
        [full_tokens], base_model, base_sae, sae_layer=sae_layer,
    )
    return acts_list[0]


def classify_avoid_continuation(
    prefix_text: str,
    continuation_text: str,
    f_c: int,
    base_model: Any,
    base_sae: Any,
    *,
    sae_layer: int,
    extraction_threshold: float = 0.0,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    cluster_target: ClusterFeatureTarget | None = None,
) -> AvoidContinuationOutcome:
    """Tokenize ``(prefix, continuation)`` ONCE and classify the avoid-QC outcome.

    Combines the token-contribution guard (``n_prefix >= len(full_tokens)``)
    and the strict-zero activation check so the pair is tokenized a single
    time (issue #513). Used by positive-mode Stage 1b avoid tiers (issue #513)
    and negative-mode ``concept_avoiding`` synth QC (issue #518).

    - ``NO_TOKENS`` when ``1 + n_prefix >= len(full_tokens)`` — no SAE pass
      (the ``+1`` accounts for ``full_tokens``' leading BOS: the
      continuation contributes no tokens beyond the prefix iff BOS +
      prefix already fill the whole combined-tokenization sequence).
    - ``ACTIVE`` / ``INACTIVE`` apply the strict-``>`` threshold check over
      ``acts[1 + n_prefix:, f_c]`` (the ``+1`` excludes the injected BOS at
      row 0); records the ``synth_qc.fc_active_checks`` counter.
    """
    prefix_tokens, full_tokens = _tokenize_prefix_and_full(
        prefix_text, continuation_text, tokenizer_id=tokenizer_id,
    )
    n_prefix = len(prefix_tokens)
    # +1: full_tokens carries a leading BOS (position 0); the continuation
    # contributes no tokens beyond the prefix iff BOS + prefix already fill
    # the whole sequence.
    if 1 + n_prefix >= len(full_tokens):
        return AvoidContinuationOutcome.NO_TOKENS
    # Loud, NOT graceful: refuse to classify an input the SAE forward would
    # front-clamp to n_ctx (classifying a truncated continuation would score a
    # partial view). This never fires on real synth data (continuations are
    # short); if it ever does it is an anomaly to surface, not silently score.
    n_ctx = _base_context_window(base_model)
    if len(full_tokens) > n_ctx:
        raise AxBenchBaseError(
            f"classify_avoid_continuation: input is {len(full_tokens)} tokens > "
            f"n_ctx={n_ctx}; refusing to classify a continuation the SAE forward "
            "would truncate. Investigate the oversized prefix/continuation."
        )
    acts = _run_sae_on_full(full_tokens, base_model, base_sae, sae_layer=sae_layer)
    bpa = BosPrependedActivations.from_forward(acts, full_tokens, n_prefix=n_prefix)
    continuation_fc: t.Tensor | None = None
    try:
        # completion() excludes the injected BOS (row 0) and the prefix (rows
        # 1..n_prefix): acts[1 + n_prefix:]. The +1 offset lives once in the object.
        if cluster_target is None:
            continuation_fc = bpa.completion()[:, f_c]
        else:
            members = _validate_cluster_target(
                cluster_target,
                representative_id=f_c,
                n_features=int(acts.shape[1]),
            )
            continuation_fc = bpa.completion()[:, list(members)]
        active = bool((continuation_fc > extraction_threshold).any().item())
    finally:
        del continuation_fc
        del acts
        release_cuda_iteration_memory()
    record_counter("synth_qc.fc_active_checks", 1, active=active)
    return (
        AvoidContinuationOutcome.ACTIVE if active
        else AvoidContinuationOutcome.INACTIVE
    )


def output_fc_percentiles_for_continuations(
    inputs: list[ContinuationQcInput],
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    release_memory: bool = True,
) -> list[float]:
    """Score a ragged batch of joined-sequence continuation percentiles.

    This is the batched equivalent of calling
    :func:`output_fc_percentile_for_continuation` once per input. Every item is
    tokenized independently with the canonical seam-aware helper, while the
    prepared token lists share one model/SAE extractor call. Results retain
    input order and may mix feature IDs and cluster targets.
    """
    if not inputs:
        return []

    prepared: list[tuple[int, list[int]]] = []
    for item in inputs:
        prefix_tokens, full_tokens = _tokenize_prefix_and_full(
            item.prefix_text,
            item.continuation_text,
            tokenizer_id=tokenizer_id,
        )
        prepared.append((len(prefix_tokens), full_tokens))

    n_ctx = _base_context_window(base_model)
    for _n_prefix, full_tokens in prepared:
        if len(full_tokens) > n_ctx:
            raise AxBenchBaseError(
                f"synth percentile QC: input is {len(full_tokens)} tokens > "
                f"n_ctx={n_ctx}; refusing to score a continuation the SAE forward "
                "would truncate. Investigate the oversized prefix/continuation."
            )

    acts_list: list[t.Tensor] | None = None
    aligned_acts: t.Tensor | None = None
    acts: t.Tensor | None = None
    bpa: BosPrependedActivations | None = None
    completion: t.Tensor | None = None
    try:
        with profile_region(
            "synth_qc.output_fc_percentiles_for_continuations",
            rows=len(inputs),
            sae_layer=sae_layer,
        ):
            acts_list = _run_sae_on_token_lists(
                [full_tokens for _n_prefix, full_tokens in prepared],
                base_model,
                base_sae,
                sae_layer=sae_layer,
            )
            output: list[float] = []
            for item, (n_prefix, full_tokens), batch_acts in zip(
                inputs,
                prepared,
                acts_list,
                strict=True,
            ):
                aligned_acts = batch_acts
                acts = aligned_acts
                bpa = BosPrependedActivations.from_forward(
                    acts,
                    full_tokens,
                    n_prefix=n_prefix,
                )
                completion = bpa.completion()
                if item.cluster_target is None:
                    percentile = _max_percentile(
                        completion[:, item.feature_id],
                        item.feature_id,
                        percentile_table,
                    )
                else:
                    _validate_cluster_target(
                        item.cluster_target,
                        representative_id=item.feature_id,
                        n_features=int(acts.shape[1]),
                    )
                    percentile = _max_percentile_for_target(
                        completion,
                        item.cluster_target,
                        percentile_table,
                    )
                output.append(float(percentile))
                completion = None
                bpa = None
                acts = None
                aligned_acts = None
                del batch_acts
            return output
    finally:
        del completion
        del bpa
        del acts
        del aligned_acts
        del acts_list
        if release_memory:
            with profile_region(
                "synth_qc.release_cuda_iteration_memory",
                caller="output_fc_percentiles_for_continuations",
            ):
                release_cuda_iteration_memory()


def classify_avoid_continuations(
    inputs: list[ContinuationQcInput],
    base_model: Any,
    base_sae: Any,
    *,
    sae_layer: int,
    extraction_threshold: float = 0.0,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    profile_region_name: str = "synth_qc.classify_avoid_continuations",
) -> list[AvoidContinuationOutcome]:
    """Classify ragged continuation targets with one strict-zero SAE pass.

    Items whose seam-aware combined encoding contributes no continuation token
    are scattered to ``NO_TOKENS`` and omitted from the extractor call. The
    remaining items may mix feature IDs and cluster targets and retain their
    original order in the returned list.
    """
    if not inputs:
        return []

    results: list[AvoidContinuationOutcome | None] = [None] * len(inputs)
    active: list[tuple[int, ContinuationQcInput, int, list[int]]] = []
    for position, item in enumerate(inputs):
        prefix_tokens, full_tokens = _tokenize_prefix_and_full(
            item.prefix_text,
            item.continuation_text,
            tokenizer_id=tokenizer_id,
        )
        n_prefix = len(prefix_tokens)
        if 1 + n_prefix >= len(full_tokens):
            results[position] = AvoidContinuationOutcome.NO_TOKENS
        else:
            active.append((position, item, n_prefix, full_tokens))

    if not active:
        return [
            result
            for result in results
            if result is not None
        ]

    n_ctx = _base_context_window(base_model)
    for _position, _item, _n_prefix, full_tokens in active:
        if len(full_tokens) > n_ctx:
            raise AxBenchBaseError(
                f"classify_avoid_continuation: input is {len(full_tokens)} tokens > "
                f"n_ctx={n_ctx}; refusing to classify a continuation the SAE forward "
                "would truncate. Investigate the oversized prefix/continuation."
            )

    acts_list: list[t.Tensor] | None = None
    aligned_acts: t.Tensor | None = None
    acts: t.Tensor | None = None
    bpa: BosPrependedActivations | None = None
    continuation_fc: t.Tensor | None = None
    try:
        with profile_region(
            profile_region_name,
            rows=len(active),
            sae_layer=sae_layer,
        ):
            acts_list = _run_sae_on_token_lists(
                [full_tokens for _position, _item, _n_prefix, full_tokens in active],
                base_model,
                base_sae,
                sae_layer=sae_layer,
            )
            for (
                position,
                item,
                n_prefix,
                full_tokens,
            ), batch_acts in zip(active, acts_list, strict=True):
                aligned_acts = batch_acts
                acts = aligned_acts
                bpa = BosPrependedActivations.from_forward(
                    acts,
                    full_tokens,
                    n_prefix=n_prefix,
                )
                if item.cluster_target is None:
                    continuation_fc = bpa.completion()[:, item.feature_id]
                else:
                    members = _validate_cluster_target(
                        item.cluster_target,
                        representative_id=item.feature_id,
                        n_features=int(acts.shape[1]),
                    )
                    continuation_fc = bpa.completion()[:, list(members)]
                is_active = bool(
                    (continuation_fc > extraction_threshold).any().item()
                )
                results[position] = (
                    AvoidContinuationOutcome.ACTIVE
                    if is_active
                    else AvoidContinuationOutcome.INACTIVE
                )
                # Match the scalar seam: a completed check is observable even
                # if validating a later item aborts the rest of the batch.
                record_counter(
                    "synth_qc.fc_active_checks",
                    1,
                    active=is_active,
                )
                continuation_fc = None
                bpa = None
                acts = None
                aligned_acts = None
                del batch_acts
    finally:
        del continuation_fc
        del bpa
        del acts
        del aligned_acts
        del acts_list
        release_cuda_iteration_memory()

    missing = [position for position, result in enumerate(results) if result is None]
    if missing:
        raise AxBenchBaseError(
            "synth strict-zero batch failed to classify result indices: "
            f"{missing[:10]}"
        )
    return [result for result in results if result is not None]


def _compute_fc_percentiles_impl(
    prefix_text: str,
    continuation_text: str,
    f_c: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    release_memory: bool = True,
    cluster_target: ClusterFeatureTarget | None = None,
    include_seed: bool,
) -> tuple[float | None, float]:
    """Return joined-sequence output percentile and optionally its prefix score."""

    prefix_tokens, full_tokens = _tokenize_prefix_and_full(
        prefix_text, continuation_text, tokenizer_id=tokenizer_id,
    )
    n_prefix = len(prefix_tokens)
    # Loud, NOT graceful: refuse to score an input the SAE forward would
    # front-clamp to n_ctx (scoring a truncated continuation would misrepresent
    # its feature activation). This never fires on real synth data (continuations
    # are short); if it ever does it is an anomaly to surface, not silently score.
    n_ctx = _base_context_window(base_model)
    if len(full_tokens) > n_ctx:
        raise AxBenchBaseError(
            f"synth percentile QC: input is {len(full_tokens)} tokens > "
            f"n_ctx={n_ctx}; refusing to score a continuation the SAE forward "
            "would truncate. Investigate the oversized prefix/continuation."
        )
    acts = _run_sae_on_full(full_tokens, base_model, base_sae, sae_layer=sae_layer)
    # full_tokens carries a leading BOS (position 0). The +1 BOS-exclusion offset
    # lives ONCE in BosPrependedActivations.from_forward, which ASSERTS acts is
    # 1:1 with full_tokens and 0 <= n_prefix <= len-1: prefix() = acts[1:1+n_prefix]
    # (empty when n_prefix == 0) and completion() = acts[1+n_prefix:], both
    # excluding the injected BOS. A prefix/continuation seam BOUNDARY merge (the
    # continuation contributes no net token, n_prefix == len(full_ids)) leaves the
    # completion span empty -> output_percentile 0.0; a STRICT over-merge
    # (n_prefix > len(full_ids), near-impossible for an appended non-empty
    # continuation) trips from_forward's range assert and RAISES rather than
    # silently scoring a nonsensical boundary.
    bpa = BosPrependedActivations.from_forward(acts, full_tokens, n_prefix=n_prefix)
    prefix_fc: t.Tensor | None = None
    continuation_fc: t.Tensor | None = None
    try:
        with profile_region(
            "synth_qc.compute_fc_percentiles.max_percentile",
            feature_id=f_c,
        ):
            if cluster_target is None:
                continuation_fc = bpa.completion()[:, f_c]
                if include_seed:
                    prefix_fc = bpa.prefix()[:, f_c]
                    seed_percentile = _max_percentile(
                        prefix_fc,
                        f_c,
                        percentile_table,
                    )
                else:
                    seed_percentile = None
                output_percentile = _max_percentile(
                    continuation_fc,
                    f_c,
                    percentile_table,
                )
            else:
                _validate_cluster_target(
                    cluster_target,
                    representative_id=f_c,
                    n_features=int(acts.shape[1]),
                )
                seed_percentile = (
                    _max_percentile_for_target(
                        bpa.prefix(),
                        cluster_target,
                        percentile_table,
                    )
                    if include_seed
                    else None
                )
                output_percentile = _max_percentile_for_target(
                    bpa.completion(),
                    cluster_target,
                    percentile_table,
                )
    finally:
        del continuation_fc
        del prefix_fc
        del acts
        if release_memory:
            with profile_region(
                "synth_qc.release_cuda_iteration_memory",
                caller="compute_fc_percentiles",
            ):
                release_cuda_iteration_memory()
    return seed_percentile, output_percentile


def compute_fc_percentiles(
    prefix_text: str,
    continuation_text: str,
    f_c: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    release_memory: bool = True,
    cluster_target: ClusterFeatureTarget | None = None,
) -> tuple[float, float]:
    """Return joined-sequence prefix and output percentiles for ``f_c``.

    This preserves the existing pair-scoring contract. Positive Stage 1b uses
    :func:`output_fc_percentile_for_continuation` with its separately cached
    canonical prefix-alone baseline.
    """

    seed_percentile, output_percentile = _compute_fc_percentiles_impl(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=sae_layer,
        tokenizer_id=tokenizer_id,
        release_memory=release_memory,
        cluster_target=cluster_target,
        include_seed=True,
    )
    assert seed_percentile is not None
    return seed_percentile, output_percentile


def output_fc_percentile_for_continuation(
    prefix_text: str,
    continuation_text: str,
    f_c: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    release_memory: bool = True,
    cluster_target: ClusterFeatureTarget | None = None,
) -> float:
    """Return only the joined-sequence continuation percentile for ``f_c``."""

    seed_percentile, output_percentile = _compute_fc_percentiles_impl(
        prefix_text,
        continuation_text,
        f_c,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=sae_layer,
        tokenizer_id=tokenizer_id,
        release_memory=release_memory,
        cluster_target=cluster_target,
        include_seed=False,
    )
    assert seed_percentile is None
    return output_percentile


def fc_percentile_on_prefix(
    prefix_text: str,
    f_c: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    tokenizer_id: str = DEFAULT_EXTRACTION_TOKENIZER_ID,
    cluster_target: ClusterFeatureTarget | None = None,
) -> float:
    """Return the max-over-prefix-tokens percentile of feature ``f_c``.

    Used by Stage 1b seed filtering to score which seeds are "cleanest"
    (lowest pre-existing activation of the target concept feature).
    Callers pass ``target_model_meta.sae_layer`` for ``sae_layer``.

    Injects exactly one leading BOS before the SAE forward and excludes
    that BOS row from the percentile pool (``acts[1:, f_c]``), mirroring
    canonical feature extraction's ``real_acts = acts[1:]`` convention.

    Empty prefix -> RAISES. The prefix is a CORPUS seed
    (``seeds_train[...].prefix_text``,
    the same input :meth:`PrefixPercentileCache.get_many` scores); a corpus seed
    should never be empty, so — consistent with ``get_many`` — an empty one is a
    real data bug, refused rather than scored as a silent 0.0 percentile. Empty
    tokenization (a non-empty string the tokenizer maps to ``[]``) raises for the
    same reason.
    """
    if not prefix_text:
        raise AxBenchBaseError(
            "fc_percentile_on_prefix: empty prefix text. This is a CORPUS seed "
            "(seeds_train[...].prefix_text); a corpus seed should never be empty. "
            "Consistent with PrefixPercentileCache.get_many, refuse rather than "
            "return a silent 0.0 percentile that would mis-classify the seed."
        )
    with profile_region("synth_qc.tokenize_prefix"):
        gemma_tok = get_extraction_tokenizer(tokenizer_id)
        tokens = gemma_tok.encode(prefix_text, add_special_tokens=False)
    if not tokens:
        raise AxBenchBaseError(
            "fc_percentile_on_prefix: a corpus seed's prefix text tokenized to "
            f"zero tokens (text: {prefix_text[:80]!r}); a corpus seed should never "
            "be empty."
        )
    n_ctx = _base_context_window(base_model)
    if 1 + len(tokens) > n_ctx:
        raise AxBenchBaseError(
            "fc_percentile_on_prefix: prepared prefix is "
            f"{1 + len(tokens)} tokens including BOS > n_ctx={n_ctx}; "
            "refusing to score a truncated prefix"
        )
    # Reuse the already-computed `tokens` (encoded_ids) so the helper does not
    # re-tokenize; it still runs the in-data special-token warning + BOS-prepend
    # + clamp.
    model_input = _bos_prepended_model_input(
        prefix_text,
        tokenizer=gemma_tok,
        max_tokens=n_ctx,
        n_ctx=n_ctx,
        context="fc_percentile_on_prefix",
        encoded_ids=tokens,
    )
    acts = _run_sae_on_full(model_input, base_model, base_sae, sae_layer=sae_layer)
    bpa = BosPrependedActivations.from_forward(acts, model_input)
    prefix_fc: t.Tensor | None = None
    try:
        if cluster_target is None:
            prefix_fc = bpa.content()[:, f_c]  # acts[1:, f_c] — BOS row 0 excluded
            percentile = _max_percentile(prefix_fc, f_c, percentile_table)
        else:
            _validate_cluster_target(
                cluster_target,
                representative_id=f_c,
                n_features=int(acts.shape[1]),
            )
            percentile = _max_percentile_for_target(
                bpa.content(),
                cluster_target,
                percentile_table,
            )
    finally:
        del prefix_fc
        del acts
        release_cuda_iteration_memory()
    return percentile


def _max_percentile(
    activations_for_fc: t.Tensor, f_c: int, percentile_table: t.Tensor,
) -> float:
    """Compute the max percentile of a 1-D activation tensor for feature ``f_c``.

    Returns ``0.0`` in two cases — both mean "feature did not fire":

    - Empty input (``numel() == 0``): no tokens to score.
    - Max activation is zero (``max_mag <= 0``): every token had zero
      activation, the feature is genuinely silent.

    Without the second check, a silent feature would be floored to
    ``0.5`` by :func:`magnitude_to_percentile` (its "below p1"
    sentinel), which is indistinguishable from "feature fired at the
    very bottom of its observed distribution". Threshold-based callers
    at or near ``0.0`` would then admit truly-inactive continuations.
    Returning ``0.0`` here collapses "never fires" into the same code
    path as "no tokens", so a ``threshold=0.0`` caller filters out
    both — the semantically correct behavior.

    Any non-zero activation still routes through
    :func:`magnitude_to_percentile` and floors to ``0.5`` when below
    the observed p1. SAE activations are ReLU'd so ``max_mag`` is
    always ``>= 0``; ``<= 0`` is a defensive check rather than a
    negative-value path.
    """
    if activations_for_fc.numel() == 0:
        return 0.0
    max_mag = activations_for_fc.max().item()
    if max_mag <= 0:
        return 0.0
    fid_tensor = t.tensor([f_c], dtype=t.int64)
    mag_tensor = t.tensor([max_mag], dtype=t.float32)
    pct_tensor = magnitude_to_percentile(fid_tensor, mag_tensor, percentile_table)
    return float(pct_tensor.item())
