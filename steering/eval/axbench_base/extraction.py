"""Shared SAE extraction helpers for AxBench.

The live Stage 0c, Stage 2, synth-QC, negative-prefix, and feature-diffmean
paths reuse the BOS-aware input/span helpers, batched extractor, and
statistics-derived percentile caches defined here.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
from loguru import logger

from steering.conditional.percentile_table import (
    build_percentile_table,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
)
from steering.extract_features import (
    _get_sae_activations_batched,
)
from steering.sae_resolver import ActivationType
from steering.segmentation import (
    resolve_bos_token_id,
    special_token_ids,
)
from steering.stats_io import load_statistics

_StatsFileCacheKey = tuple[str, int, int, str]


def _run_batched_extractor(
    *,
    model: Any,
    sae: Any,
    token_id_lists: list[list[int]],
    layer: int,
    hook_name: str | None,
    activation_type: ActivationType = ActivationType.residual,
    batch_size: int | None = None,
    per_item_reduce: Callable[[t.Tensor, int], t.Tensor] | None = None,
) -> list[t.Tensor]:
    """Thin keyword-only wrapper around ``_get_sae_activations_batched``.

    Exists so tests can ``monkeypatch.setattr(extraction,
    "_run_batched_extractor", stub)`` without touching the underlying
    extract_features module.

    Forwards an optional native or nnterp extraction model when ``model``
    carries the corresponding side-channel attribute. nnterp retains priority
    if both attributes exist. With neither, the TransformerLens path remains
    unchanged.

    ``per_item_reduce`` (#1240) collapses each item's activations **as its
    sub-batch is produced**, receiving ``(acts, global_index)`` and returning the
    tensor to accumulate in its place. Without it, ``batch_size`` bounds only the
    transient forward: every item's full ``[seq_len, n_features]`` tensor is
    appended to the returned list and stays resident, so peak memory scales with
    the ITEM COUNT, not the batch. At a 65k-wide SAE and ~432 tokens that is
    ~113 MiB per item -- ~45 GiB for 400 items, against a ~105 MiB pooled result.
    Every caller reduces per item immediately, so nothing needs the raw tensors
    simultaneously. Passing a reduction makes peak scale with ``batch_size``,
    which is what the parameter already implies.

    The reduction is applied inside the recursion so a monkeypatched stub still
    intercepts each sub-batch, preserving the test seam described above.
    """
    if batch_size is not None and batch_size < 1:
        raise AxBenchBaseError(f"batch_size must be >= 1 when set, got {batch_size}")
    if batch_size is not None and len(token_id_lists) > batch_size:
        results: list[t.Tensor] = []
        for start in range(0, len(token_id_lists), batch_size):
            # Rebase the reduction's index onto the GLOBAL item position so
            # callers can key per-item metadata (completion offsets, input ids)
            # off it exactly as they would over the unbatched list.
            chunk_reduce = (
                None
                if per_item_reduce is None
                else (lambda acts, i, _o=start: per_item_reduce(acts, _o + i))
            )
            results.extend(
                _run_batched_extractor(
                    model=model,
                    sae=sae,
                    token_id_lists=token_id_lists[start : start + batch_size],
                    layer=layer,
                    hook_name=hook_name,
                    activation_type=activation_type,
                    batch_size=None,
                    per_item_reduce=chunk_reduce,
                )
            )
        return results

    nnterp_model = getattr(model, "nnterp_model", None)
    native_model = getattr(model, "native_model", None) if nnterp_model is None else None
    raw = _get_sae_activations_batched(
        model=model,
        sae=sae,
        token_id_lists=token_id_lists,
        layer=layer,
        nnterp_model=nnterp_model,
        **({"native_model": native_model} if native_model is not None else {}),
        hook_name=hook_name,
        activation_type=activation_type,
    )
    if per_item_reduce is None:
        return raw
    reduced = [per_item_reduce(acts, index) for index, acts in enumerate(raw)]
    # Drop the sub-batch's raw tensors before returning so the caller's
    # accumulation never holds them alongside the next sub-batch's forward.
    del raw
    return reduced


@functools.lru_cache(maxsize=4)
def _stats_file_sha256_cached(
    stats_path: str,
    mtime_ns: int,
    size: int,
) -> str:
    """Return a content fingerprint for a stats file.

    ``mtime_ns`` and ``size`` are part of the cache key, so replacing a
    stats file in a long-lived Stage 4 process invalidates this helper's
    LRU entry before the file content is read.
    """
    del mtime_ns, size  # Included only for LRU invalidation.
    digest = hashlib.sha256()
    with Path(stats_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats_file_cache_key(stats_path: str) -> _StatsFileCacheKey:
    path = Path(stats_path).resolve()
    stat = path.stat()
    mtime_ns = int(stat.st_mtime_ns)
    size = int(stat.st_size)
    sha256 = _stats_file_sha256_cached(str(path), mtime_ns, size)
    return str(path), mtime_ns, size, sha256


def feature_statistics_file_sha256(stats_path: str) -> str:
    """Return the content identity already used by the percentile-table cache.

    Calling this after :func:`build_percentile_table_from_stats` reuses the
    warmed content-hash entry rather than streaming the statistics file a
    second time for a downstream fingerprint.
    """

    return _stats_file_cache_key(stats_path)[3]


@functools.lru_cache(maxsize=4)
def _load_per_feature_stats_cached(
    stats_path: str,
    mtime_ns: int,
    size: int,
    sha256: str,
) -> dict:
    """LRU-cached loader for per-feature stats dict.

    The cache key includes file metadata and sha256 content so replacing
    a stats file at the same path in a long-lived process cannot reuse a
    stale parsed stats dict.
    """
    del mtime_ns, size, sha256
    return load_statistics(Path(stats_path)).per_feature_stats


@functools.lru_cache(maxsize=4)
def _build_percentile_table_for_stats_cached(
    stats_path: str,
    mtime_ns: int,
    size: int,
    sha256: str,
    n_sae_features: int,
) -> t.Tensor:
    per_feature_stats = _load_per_feature_stats_cached(
        stats_path, mtime_ns, size, sha256,
    )
    return build_percentile_table(per_feature_stats, n_sae_features)


def build_percentile_table_from_stats(
    stats_path: str, n_sae_features: int,
) -> t.Tensor:
    """LRU-cached wrapper around :func:`build_percentile_table`.

    The effective cache key includes a sha256 fingerprint of the stats
    file, so repeated calls reuse the percentile table only while the
    path still points at the same content.
    """
    return _build_percentile_table_for_stats_cached(
        *_stats_file_cache_key(stats_path),
        n_sae_features,
    )


def _sae_hook_name(base_sae: Any) -> str:
    """Return the SAE's TransformerLens hook name.

    Handles both pre- and post-6.37 SAELens cfg layouts:

    * Older releases exposed ``cfg.hook_name`` directly on the config.
    * SAELens 6.37+ nests per-SAE metadata under a ``cfg.metadata``
      dict-like (:class:`sae_lens.SAEMetadata`); the hook name lives at
      ``cfg.metadata["hook_name"]``.

    The direct attribute takes precedence when both are present, so
    callers that carry a legacy cfg keep their existing behaviour.

    Raises:
        AxBenchBaseError: if ``base_sae`` has no ``.cfg`` attribute, or
            if the hook name is missing from both locations. A missing
            hook_name is an upstream bug (wrong SAE type, failed load)
            that should surface immediately rather than being silently
            papered over with a ``"default"`` fallback that would
            collide in the extraction cache across distinct SAEs.
    """
    cfg = getattr(base_sae, "cfg", None)
    if cfg is None:
        raise AxBenchBaseError(
            "base_sae has no .cfg attribute; cannot read hook_name. "
            "Verify that base_sae is a valid SAELens SAE instance."
        )
    hook_name = getattr(cfg, "hook_name", None) or _cfg_metadata_get(
        cfg, "hook_name",
    )
    if not hook_name:
        raise AxBenchBaseError(
            "base_sae.cfg has no hook_name attribute (or it is empty) "
            "and cfg.metadata['hook_name'] is also unset; cannot "
            "identify the SAE's TransformerLens hook. Fix the SAE cfg "
            "upstream or file a bug."
        )
    return str(hook_name)


def _cfg_metadata_get(cfg: Any, key: str) -> Any:
    """Read ``key`` from ``cfg.metadata`` if it behaves like a mapping.

    SAELens 6.37+ stores per-SAE fields (``hook_name``, ``model_name``,
    ``neuronpedia_id``, ...) in a dict-like ``SAEMetadata`` object hung
    off the cfg. Older releases do not have this attribute at all.
    Returns :data:`None` when either the attribute is missing or the
    key is absent so callers can compose it with ``or``-based
    fallbacks without handling a separate exception path.
    """
    metadata = getattr(cfg, "metadata", None)
    if metadata is None:
        return None
    getter = getattr(metadata, "get", None)
    if callable(getter):
        return getter(key)
    try:
        return metadata[key]
    except (KeyError, TypeError):
        return None


def _infer_n_sae_features(base_sae: Any) -> int:
    """Read ``n_sae_features`` off the SAE config.

    The SAE's feature-count field is named differently across SAELens
    versions (``d_sae`` vs ``n_features``); the accessor tries both.
    This is the single canonical helper for inferring SAE feature
    counts across the axbench_base pipeline --
    :mod:`concept_feature_mapping`, :mod:`synth.positive_mode`, and
    :mod:`negative_prefixes` all import it.

    Raises:
        AxBenchBaseError: if ``base_sae`` has no ``.cfg`` attribute, or
            if neither ``cfg.d_sae`` nor ``cfg.n_features`` is set. A
            missing cfg or both attrs unset indicates an upstream bug
            (wrong model type, failed load) that should surface
            immediately rather than being silently papered over with
            a ``16384`` default.
    """
    cfg = getattr(base_sae, "cfg", None)
    if cfg is None:
        raise AxBenchBaseError(
            "base_sae has no .cfg attribute; cannot infer n_sae_features. "
            "Verify that base_sae is a valid SAELens SAE instance."
        )
    n_features = getattr(cfg, "d_sae", None) or getattr(cfg, "n_features", None)
    if n_features is None:
        raise AxBenchBaseError(
            "base_sae.cfg has neither d_sae nor n_features attribute; "
            "cannot infer n_sae_features."
        )
    return int(n_features)


def warn_on_special_id_pollution(
    ids: list[int], *, tokenizer: Any, bos_id: int, context: str, text: str,
) -> None:
    """Log a warning (but PROCEED) if ``ids`` (an ``add_special_tokens=False``
    encoding of ``text``) contains any special-token id.

    A literal special-token string in the DATA (e.g. ``'<bos>'`` left by a
    specials-preserving decode upstream) re-encodes to the REAL id even with
    specials disabled.  We PRESERVE the document's original tokens and run the
    SAE forward as-is: the leading BOS the pooling drops is our OWN injected one
    (position 0), so a genuine in-data ``<bos>`` stays in the pooled span as data,
    and a document that already starts with ``<bos>`` intentionally yields a
    double-BOS.  The in-data special is a benign signal, not corruption, so we do
    NOT refuse — but we log it so a run with unexpectedly many polluted inputs is
    visible.  ``context`` names the calling site.
    """
    polluting = special_token_ids(tokenizer, bos_id).intersection(ids)
    if polluting:
        logger.warning(
            "{}: input text encodes in-data special token id(s) {} even with "
            "add_special_tokens=False (a literal special-token string such as "
            "'<bos>'). Preserving the document's original tokens and running the "
            "SAE forward as-is (the dropped leading BOS is our injected one; any "
            "in-data special stays in the pooled span). text: {!r}",
            context,
            sorted(polluting),
            text[:80],
        )


def _bos_prepended_model_input(
    text: str,
    *,
    tokenizer: Any,
    max_tokens: int,
    n_ctx: int,
    context: str = "AxBench SAE input",
    encoded_ids: list[int] | None = None,
) -> list[int]:
    """Tokenize ``text`` and prepend exactly one leading BOS.

    Mirrors the BOS handling in
    :func:`steering.eval.axbench_bpb_adv.feature_diffmean._max_pooled_sae_activations`
    (the proven reference pattern): tokenize with ``add_special_tokens=False``
    so the tokenizer adds no specials of its own, WARN (but proceed) if the TEXT
    itself contains a literal special-token string (see
    :func:`warn_on_special_id_pollution`), then prepend exactly one BOS and
    front-clamp to ``min(max_tokens, n_ctx)``. Position 0 of the result is
    always OUR injected BOS; a genuine in-data ``<bos>`` is preserved at
    position 1+, so a ``<bos>``-starting document intentionally double-BOSes and
    keeps its original tokens in the pooled span.

    ``context`` names the calling function for error messages. ``encoded_ids``
    lets a caller that already tokenized ``text`` (e.g. for an emptiness
    pre-check) pass the ids to avoid a redundant second encode.

    Raises:
        AxBenchBaseError: the tokenizer returns zero tokens for ``text``, or the
            tokenizer exposes no discoverable BOS id (a real Gemma config bug).
            An in-data special-token string is WARNED, not raised — the
            document's original tokens are preserved and the forward runs as-is.
    """
    ids = (
        encoded_ids
        if encoded_ids is not None
        else tokenizer.encode(text, add_special_tokens=False)
    )
    if not ids:
        raise AxBenchBaseError(
            f"{context}: tokenizer produced zero tokens for input text"
        )
    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise AxBenchBaseError(
            f"{context}: extraction tokenizer exposes no BOS id; the "
            "canonical feature extraction prepends a BOS and ignores its features, "
            "so running without one is out-of-distribution. A missing Gemma BOS is "
            "a real config bug — refusing to proceed."
        )
    warn_on_special_id_pollution(
        ids, tokenizer=tokenizer, bos_id=bos_id, context=context, text=text,
    )
    return ([int(bos_id)] + [int(i) for i in ids])[: min(int(max_tokens), int(n_ctx))]


@dataclass(frozen=True)
class BosPrependedActivations:
    """SAE activations for a model input whose row 0 is our injected BOS.

    Centralizes the single ``+1`` BOS-exclusion offset so callers read NAMED
    spans (:meth:`content` / :meth:`prefix` / :meth:`completion`) instead of
    hand-writing ``acts[1:]`` / ``acts[1 + n_prefix:]`` at each SAE-forward site
    — the ``+1`` (and the invariants that make it valid) live here, once.

    Row 0 is the BOS the SAE-forward sites prepend (an attention-sink outlier);
    it is NEVER part of any span, matching ``extract_features``'s
    ``real_acts = acts[1:]`` convention.  ``n_prefix`` is the number of prefix
    tokens AFTER that BOS (the split point for prefix-vs-completion pooling);
    it defaults to 0 for a whole-text pool that only needs :meth:`content`.

    Build via :meth:`from_forward`, which asserts the activations are 1:1 with
    the BOS-prepended model input (so the extractor injected no BOS of its own)
    and the prefix boundary is in range — the two preconditions every manual
    ``acts[1 + n_prefix:]`` slice silently assumed. Callers that must NOT let a
    clamped/truncated input slip through silently (the synth QC sites) guard the
    length loudly BEFORE the forward, so ``acts`` is always 1:1 here.
    """

    acts: t.Tensor  # (seq_len, n_features); row 0 == the injected BOS
    n_prefix: int = 0

    @classmethod
    def from_forward(
        cls, acts: t.Tensor, model_input: list[int], *, n_prefix: int = 0
    ) -> BosPrependedActivations:
        """Validate ``acts`` is 1:1 with ``model_input`` and wrap it.

        Raises:
            AxBenchBaseError: ``acts`` is not 2-D; its row count differs from
                ``len(model_input)`` (the extractor injected a BOS, so the
                ``acts[1:]`` exclusion would be off); or ``n_prefix`` is out of
                ``[0, len(model_input) - 1]`` (the prefix span would overflow the
                non-BOS rows).
        """
        n_input = len(model_input)
        if acts.ndim != 2:
            raise AxBenchBaseError(
                "BosPrependedActivations: acts must be 2-D (seq_len, n_features), "
                f"got shape {tuple(acts.shape)}."
            )
        if acts.shape[0] != n_input:
            raise AxBenchBaseError(
                f"BosPrependedActivations: extractor returned {acts.shape[0]} rows "
                f"for a {n_input}-token BOS-prepended input; acts must be 1:1 (no "
                "BOS injection in the extractor) for the acts[1:] BOS-exclusion "
                "slice to be valid."
            )
        if not (0 <= n_prefix <= n_input - 1):
            raise AxBenchBaseError(
                f"BosPrependedActivations: n_prefix={n_prefix} out of range "
                f"[0, {n_input - 1}] for a {n_input}-token input (row 0 is the "
                "injected BOS; the prefix span acts[1:1+n_prefix] must fit the "
                "remaining rows)."
            )
        return cls(acts=acts, n_prefix=n_prefix)

    def content(self) -> t.Tensor:
        """All non-BOS rows (prefix + completion): ``acts[1:]``."""
        return self.acts[1:]

    def prefix(self) -> t.Tensor:
        """The prefix span, excluding the leading BOS: ``acts[1 : 1 + n_prefix]``."""
        return self.acts[1 : 1 + self.n_prefix]

    def completion(self) -> t.Tensor:
        """The completion span, excluding BOS + prefix: ``acts[1 + n_prefix:]``."""
        return self.acts[1 + self.n_prefix :]
