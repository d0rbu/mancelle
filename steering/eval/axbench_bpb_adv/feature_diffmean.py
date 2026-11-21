"""Percentile-space feature-space DiffMean warm-start bag (Task 4).

This is the feature-space analog of residual DiffMean: instead of a direction
in residual space, it produces a DISCRETE bag of ``(feature_id, bucket)``
conditioning entries (:class:`FeatureConditioning`) that the feature-GCG
optimizer (Tasks 5/6) warm-starts from.  Correctness here is SIGN-CRITICAL: a
flipped sign silently corrupts every downstream BPB-Adv result.

The sign chain (every link pinned by a mutation trap in
``tests/axbench_bpb_adv/unit/test_feature_diffmean.py``)
---------------------------------------------------------------------------
For each SAE feature, in **percentile space**::

    diff[fid] = mean_rich_pct[fid] - mean_avoid_pct[fid]

``+`` ⇒ the feature fires at a higher percentile on the concept-RICH
completions (concept-indicating); ``−`` ⇒ higher on the concept-AVOIDING
completions.

* **POSITIVE mode** (push a neutral prefix TOWARD the concept):
  - top ``k_pos`` features by ``+diff``  → ``(fid, positive_bucket=0)``
  - bottom ``k_neg`` features by ``-diff`` → ``(fid, negative_bucket=1)``
* **NEGATIVE mode** (push a rich prefix AWAY from the concept): the **MIRROR**
  - top ``k_pos`` features by ``+diff``  → ``(fid, negative_bucket=1)``
  - bottom ``k_neg`` features by ``-diff`` → ``(fid, positive_bucket=0)``

If a feature lands in BOTH the top-k and bottom-k selections (possible only
when ``k_pos + k_neg`` exceeds the feature count), it is kept on the side whose
signed score is larger -- ``max(+diff, -diff) == |diff|`` -- i.e. the top side
when ``diff >= 0`` and the bottom side when ``diff < 0``.  This guarantees no
duplicate feature id in the returned :class:`FeatureConditioning` and that the
surviving assignment matches the feature's diff sign.

Percentile source (REUSED, never hand-rolled)
----------------------------------------------
Each completion's per-feature magnitude is mapped to a percentile in
``[0.5, 99.5]`` via :func:`steering.conditional.percentile_table.magnitude_to_percentile`
against the caller-supplied ``percentile_table`` -- the SAME table the training
pipeline builds with ``build_percentile_table_from_stats(feature_statistics_path,
n_sae_features)`` and the SAME function the encoder-input selection
(``select_top_k_by_percentile``) uses.  The diff-of-means is computed on these
percentiles, never on raw magnitudes; percentile normalization is what makes the
per-feature scales comparable across the SAE.

Prefix context, completion-only pool (SIGN-ADJACENT: what feeds the diff)
------------------------------------------------------------------------
Each completion is scored WITH its prefix as context but the SAE activations are
pooled over ONLY the completion-bearing tokens.  Concretely, per completion the
base Gemma model + SAE run on ``BOS + prefix_text + completion_text`` — the
prefix and completion are joined in CHARACTER space and tokenized as ONE string
(splice-safe: boundary subwords tokenize as they will at scoring time), then a
SINGLE BOS is prepended to match the main feature extraction (which prepends a
BOS and ignores its features).  The prefix CONDITIONS the activations; the
max-pool then covers only joint-token positions whose character span contains at
least one completion character (plus the leading BOS offset) -- isolating the
completion's concept expression from the BOS + prefix context while still
including a token that merges across the prefix/completion boundary.  This
changes WHICH activations feed ``mean_rich_pct`` / ``mean_avoid_pct``
(completion-in-context, not completion-alone); the diff/selection/bucket sign
chain above is UNCHANGED.

Text vs tokens (this function consumes TEXT pairs)
--------------------------------------------------
``BpbAttackPrefix`` carries ``prefix_tokens`` / ``concept_rich_output_tokens`` /
``concept_avoiding_output_tokens`` as token-id lists, but those are **nanochat
(encoder-decoder target) tokens** -- ``encode_prefix``/``encode_output`` use the
``get_bos_token_id()`` method form, the nanochat convention, not Gemma's
``bos_token_id`` attribute.  The SAE runs on the **base Gemma** model, which
needs Gemma tokenization, so feeding nanochat token ids to the SAE forward would
be wrong.  Therefore the caller (Tasks 5/6) passes :class:`PrefixedCompletion`
``(prefix_text, completion_text)`` pairs here: the prefix text is the RAW
``BpbAttackPrefix.prefix_text`` (never a specials-preserving decode of
``prefix_tokens``, whose leading nanochat BOS would re-encode into a REAL Gemma
BOS — the double-BOS trap), and each completion is decoded from its nanochat
tokens (completions carry no BOS).  This module re-tokenizes each pair with the
base model's OWN tokenizer (``model.tokenizer`` via :func:`base_model_tokenizer`
— the exact tokenizer ``extract_document_features`` uses at score time, so the
DiffMean forward and the score stage tokenize identically) inside
:func:`_max_pooled_sae_activations`, mirroring AxBench Base's live dense SAE
pooling path (base-model + SAE forward, then max-pool -- here over the
completion span only),
and WARNS (but proceeds, preserving the document's tokens) on any in-data special
token id found in the encoded text.

The function is pure/deterministic given its inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch as t
from loguru import logger

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.percentile_table import (
    compute_percentile_checksum,
    magnitude_to_percentile,
)
from steering.eval.axbench_base.extraction import (
    _run_batched_extractor,
    _sae_hook_name,
)
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.vocab_restriction import restrict_to_vocab
from steering.segmentation import resolve_bos_token_id, special_token_ids

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


@dataclass(frozen=True)
class PrefixedCompletion:
    """A completion paired with the prefix context it was generated under.

    :func:`feature_diffmean_bag` runs the base model + SAE on
    ``prefix_text + completion_text`` (the prefix CONDITIONS the activations) but
    max-pools over ONLY the completion span.  Carrying the two texts as named
    fields -- rather than a bare ``(str, str)`` tuple -- makes the
    prefix/completion order impossible to swap silently in this sign-critical
    warm-start path.
    """

    prefix_text: str
    completion_text: str


@dataclass(frozen=True)
class _JointTokenization:
    token_ids: list[int]
    completion_start_token: int


def feature_diffmean_bag(
    rich_items: Sequence[PrefixedCompletion],
    avoid_items: Sequence[PrefixedCompletion],
    *,
    sae: Any,
    model: Any,
    sae_layer: int,
    percentile_table: t.Tensor,
    mode: SteeringMode | str,
    k_pos: int,
    k_neg: int,
    positive_bucket: int,
    negative_bucket: int,
    device: t.device | str,
    extraction_batch_size: int | None = None,
    allowed_ids: t.Tensor | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureConditioning:
    """Build the percentile-space feature-DiffMean warm-start bag.

    Steps:

    1. Extract per-completion SAE activations for every rich and avoid item via
       the base Gemma model + SAE, running each item's prefix as CONTEXT and
       max-pooling over ONLY the completion tokens (see
       :func:`_max_pooled_sae_activations`).
    2. Map each feature's pooled magnitude to its percentile via
       ``percentile_table`` (the percentile authority).
    3. ``diff = mean_rich_pct - mean_avoid_pct`` over the per-feature
       percentiles.
    4. Take the top ``k_pos`` features by ``+diff`` and the bottom ``k_neg`` by
       ``-diff``, restricted to ``allowed_ids`` when given (see
       :func:`_select_bag`).
    5. Assign buckets per the sign chain (mirrored for NEGATIVE mode), dedup any
       feature appearing on both ends by sign of its diff, and return a
       :class:`FeatureConditioning`.

    Args:
        rich_items: Concept-rich :class:`PrefixedCompletion` pairs; the prefix
            conditions the SAE forward, the completion is what gets pooled.
        avoid_items: Concept-avoiding :class:`PrefixedCompletion` pairs.
        sae: Base SAE (``runtime_state.base_gemma_and_sae()[1]``).
        model: Base Gemma model (``runtime_state.base_gemma_and_sae()[0]``).
        sae_layer: Transformer layer the SAE reads from (the derive threads
            this in from ``cfg.neuronpedia_source`` — the SAE's own identity —
            so no hook-name re-parsing is needed here).
        percentile_table: ``[n_sae_features, 99]`` float64 table from
            ``build_percentile_table_from_stats``.
        mode: ``SteeringMode.POSITIVE`` or ``SteeringMode.NEGATIVE``.
        k_pos: Number of top-positive (rich-indicating) features to select.
        k_neg: Number of bottom-negative (avoid-indicating) features to select.
        positive_bucket: The active/positive bucket index (0).
        negative_bucket: The inactive/negative bucket index (``num_buckets - 1``).
        device: Device for the SAE forward + tensor construction.
        extraction_batch_size: Optional subbatch size for the base Gemma+SAE
            forward used to pool completion activations.
        allowed_ids: The compacted target checkpoint's allowed SAE feature ids
            ``V`` (see
            :func:`steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids`),
            or ``None`` for an uncompacted checkpoint. Restricts the top/bottom-
            ``k`` ranking to ``V`` so a selected id can never trip the encoder's
            OOV assert on a compacted checkpoint; ``None`` reproduces the
            unrestricted (full ``n_sae``) selection byte-for-byte.
        cluster_contract: Optional validated clustered-evaluation contract.
            When present, max-project raw float64 percent ranks to the
            representative universe before class averaging; ``None`` preserves
            the legacy path exactly.

    Returns:
        A :class:`FeatureConditioning` warm-start bag (canonical sorted entries,
        no duplicate feature ids).

    Raises:
        FeatureGcgError: empty ``rich_items`` or ``avoid_items``; a
            ``percentile_table`` whose row count disagrees with the SAE feature
            count; a completion tokenizing to zero tokens or a prefix that fills
            the whole context window; a prefix/completion TEXT that encodes a
            special token id (e.g. a literal ``'<bos>'`` left by a
            specials-preserving decode upstream — see
            :func:`_max_pooled_sae_activations`); or an unknown ``mode``.
    """
    resolved_mode = _resolve_mode(mode)
    if not rich_items:
        raise FeatureGcgError(
            "feature_diffmean_bag: rich_items is empty; cannot compute a "
            "diff-of-means warm-start without at least one rich completion."
        )
    if not avoid_items:
        raise FeatureGcgError(
            "feature_diffmean_bag: avoid_items is empty; cannot compute a "
            "diff-of-means warm-start without at least one avoid completion."
        )

    rich_acts = _max_pooled_sae_activations(
        rich_items,
        model=model,
        sae=sae,
        sae_layer=sae_layer,
        device=device,
        extraction_batch_size=extraction_batch_size,
    )
    avoid_acts = _max_pooled_sae_activations(
        avoid_items,
        model=model,
        sae=sae,
        sae_layer=sae_layer,
        device=device,
        extraction_batch_size=extraction_batch_size,
    )

    n_features = rich_acts.shape[1]
    table_rows = int(percentile_table.shape[0])
    if n_features != table_rows:
        raise FeatureGcgError(
            "feature_diffmean_bag: percentile_table row count "
            f"({table_rows}) disagrees with the SAE feature count "
            f"({n_features}). They must be built from the same statistics."
        )

    selection_allowed_ids = allowed_ids
    if cluster_contract is None:
        # Exact legacy branch: do not quantize, mask, project, or alter the
        # unrestricted/V-restricted ranking currency.
        mean_rich_pct = _mean_percentiles(rich_acts, percentile_table)
        mean_avoid_pct = _mean_percentiles(avoid_acts, percentile_table)
    else:
        consolidation = cluster_contract.consolidation
        if consolidation.n_sae_features != n_features:
            raise FeatureGcgError(
                "feature_diffmean_bag: cluster contract original SAE width "
                f"{consolidation.n_sae_features} disagrees with extracted width "
                f"{n_features}"
            )
        actual_percentile_checksum = compute_percentile_checksum(
            percentile_table.detach().to(device="cpu", dtype=t.float64)
        )
        if (
            actual_percentile_checksum
            != cluster_contract.statistics.percentile_table_checksum
        ):
            raise FeatureGcgError(
                "feature_diffmean_bag: percentile table does not match the "
                "cluster feature contract statistics: "
                f"expected={cluster_contract.statistics.percentile_table_checksum}, "
                f"actual={actual_percentile_checksum}"
            )
        representative_ids = t.tensor(
            cluster_contract.representative_universe,
            dtype=t.int64,
        )
        if representative_ids.numel() == 0:
            raise FeatureGcgError(
                "feature_diffmean_bag: cluster contract has an empty "
                "representative universe"
            )
        if allowed_ids is not None:
            missing = representative_ids[
                ~t.isin(representative_ids, allowed_ids.to("cpu"))
            ]
            if missing.numel() > 0:
                raise FeatureGcgError(
                    "feature_diffmean_bag: representative universe contains IDs "
                    "outside the checkpoint FeatureVocab: "
                    f"{[int(x) for x in missing[:10].tolist()]}"
                )
        # For a contract-passing checkpoint, four enforced links prove that
        # representative_universe is exactly V: it equals nonzero(cluster_
        # allowed_universe), effective_feature_vocab derives those same original
        # IDs, checkpoint.feature_vocab is pinned to that result, and the loaded
        # model is pinned to the checkpoint vocabulary. Narrowing to
        # representatives is therefore a no-op in production. Keep the explicit
        # check and narrowing as defense-in-depth against forged/desynced state.
        selection_allowed_ids = representative_ids
        if consolidation.is_identity_after_mask:
            # Redundant explicit guarantee: both branches use raw float64
            # percent ranks, and consolidate_segment is the identity map when
            # no cluster retains two mask-surviving members. Keep the direct
            # path so an inert contract is visibly guaranteed to match the
            # no-cluster path while still querying only mask-surviving rows.
            try:
                mean_rich_pct = _mean_percentiles_for_feature_ids(
                    rich_acts,
                    percentile_table,
                    feature_ids=representative_ids,
                )
                mean_avoid_pct = _mean_percentiles_for_feature_ids(
                    avoid_acts,
                    percentile_table,
                    feature_ids=representative_ids,
                )
            except ValueError as exc:
                raise FeatureGcgError(
                    "feature_diffmean_bag: identity-after-mask allowed feature "
                    f"percentile statistics are invalid: {exc}"
                ) from exc
        else:
            mean_rich_pct = _cluster_projected_mean_percentiles(
                rich_acts,
                percentile_table,
                cluster_contract=cluster_contract,
            )
            mean_avoid_pct = _cluster_projected_mean_percentiles(
                avoid_acts,
                percentile_table,
                cluster_contract=cluster_contract,
            )
    diff = mean_rich_pct - mean_avoid_pct  # [n_features] float64

    bag = _select_bag(
        diff,
        mode=resolved_mode,
        k_pos=k_pos,
        k_neg=k_neg,
        positive_bucket=positive_bucket,
        negative_bucket=negative_bucket,
        allowed_ids=selection_allowed_ids,
    )
    if cluster_contract is not None:
        cluster_contract.consolidation.assert_representatives_only(
            t.tensor(bag.feature_ids(), dtype=t.int64),
            surface="clustered feature-DiffMean bag",
        )
    return bag


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction (the model-dependent seam tests monkeypatch)
# ─────────────────────────────────────────────────────────────────────────────


def _max_pooled_sae_activations(
    items: Sequence[PrefixedCompletion],
    *,
    model: Any,
    sae: Any,
    sae_layer: int,
    device: t.device | str,
    extraction_batch_size: int | None = None,
) -> t.Tensor:
    """Return ``[len(items), n_sae_features]`` completion-span max-pooled acts.

    Tokenization mirrors the MAIN feature extraction
    (:func:`steering.extract_features`) so the DiffMean activations are consistent
    with the concept features the encdec was trained on:

    1. **Splice-safe combined tokenization.** The prefix and completion are joined
       in CHARACTER space FIRST and the whole string is tokenized once
       (``add_special_tokens=False``), so a subword that merges across the
       prefix/completion boundary is tokenized the SAME way it is at scoring time.
       For production tokenizers, the completion span is located from the joint
       tokenization's character offsets: the first pooled token is the first token
       whose character span contains any completion character.  Thus a
       boundary-crossing token is included in the completion pool instead of
       being treated as unrecoverable.  Tokenizers without offset mappings fall
       back to the old prefix-token match guard and raise on a boundary merge
       rather than guessing.
    2. **Leading BOS (ours).** ``resolve_bos_token_id`` (the SAME helper
       ``extract_features`` uses) supplies the BOS prepended at position 0;
       ``add_special_tokens=False`` above guarantees the tokenizer ADDS none.
       ``add_special_tokens=False`` does NOT protect against the text itself
       CONTAINING a literal special-token string: Gemma tokenizers re-encode a
       literal ``'<bos>'`` substring into the REAL BOS id even with specials
       disabled.  That is DATA — we PRESERVE the document's tokens and prepend our
       own BOS anyway (a ``'<bos>'``-starting prefix intentionally double-BOSes),
       WARNING for observability rather than refusing.  It is harmless here
       because only the completion span is pooled (§4): our injected BOS and the
       whole prefix — including any in-data special — are pooled OUT.  A tokenizer
       with no discoverable BOS is a real config bug (Gemma always has one) →
       raise rather than silently run without it.
    3. **Front clamp to ``n_ctx``.** BOS + prefix are CONTEXT, so clamping the
       whole sequence from the front lets a long prefix truncate only the
       completion TAIL; if the clamp leaves zero completion positions → raise.
    4. **Completion-only max-pool.** With the prepended BOS, the completion-bearing
       tokens occupy positions ``[1 + completion_start_token, len(model_input))`` —
       the leading ``+1`` is the BOS at position 0.  The BOS and pure-prefix tokens
       are context and are POOLED OUT; the trailing completion-bearing span is
       max-pooled.

    ``_run_batched_extractor`` returns per-token acts 1:1 with ``model_input`` (it
    does NOT inject its own BOS — asserted by a length check below), so the
    completion positions are exactly ``acts[completion_start:]``.  ``sae_layer`` is
    supplied explicitly by the caller; the SAE hook name is still read for the
    extractor's ``hook_name`` argument.  The tokenizer is the base model's
    OWN (``model.tokenizer`` via :func:`base_model_tokenizer`) — the exact one the
    forward runs under and the one ``extract_document_features`` uses at score
    time — so this never tokenizes with a different vocabulary than the model it
    forwards through (no separate ``extraction_tokenizer_id`` to drift).

    This is the single seam unit tests stub out to inject synthetic activations;
    the percentile mapping and the selection logic above run for real.

    Raises:
        FeatureGcgError: the extraction tokenizer exposes no BOS id; a tokenizer
            without offsets sees a boundary merge; a completion contributes zero
            completion-bearing tokens; the prefix context fills ``model.cfg.n_ctx``
            so no completion token remains; or an extractor whose acts are not
            1:1 with the input.  (An in-data special-token string is WARNED, not
            raised — the document's tokens are preserved and the forward runs
            as-is.)
    """
    tokenizer = base_model_tokenizer(model)
    hook_name = _sae_hook_name(sae)
    n_ctx = model_n_ctx(model)

    # The BOS the MAIN extraction prepends (and whose features it ignores). The
    # DiffMean SAE forward must see the SAME leading BOS or its activations would
    # be inconsistent with the training-time concept features. This is the ONE
    # documented home for the no-BOS-encode + prepend-BOS handling (review #2c);
    # the AxBench Base dense pool operates on whole text, so a shared wrapper is
    # not forced.
    bos_id = resolve_bos_token_id(tokenizer)
    if bos_id is None:
        raise FeatureGcgError(
            "feature_diffmean_bag: the extraction tokenizer exposes no BOS id, but "
            "the main feature extraction prepends a BOS and ignores its features. "
            "Running the DiffMean SAE forward without that BOS would make these "
            "activations inconsistent with the concept features. A missing Gemma "
            "BOS is a real config bug — refusing to proceed without one."
        )

    # In-data special-token detection (for the warning below): with
    # add_special_tokens=False the tokenizer ADDS none, but a literal
    # special-token string in the text (e.g. '<bos>' left by a specials-preserving
    # decode upstream) still re-encodes to the REAL id. Build the FULL special-id
    # set the same way `_tokenizer_stop_token_ids` does: the named bos/eos/pad
    # attrs UNION `all_special_ids`, PLUS the resolved `bos_id` unconditionally.
    # This closes two gaps a bare `all_special_ids or (bos_id,)` fallback left
    # open: a tokenizer WITHOUT `all_special_ids` would narrow to BOS-only (a
    # literal '<eos>' would sail through), and a tokenizer WHOSE `all_special_ids`
    # omits its BOS (BOS discovered via the get_bos_token_id() method probe, not
    # registered as an HF special) would be missed.
    special_ids = special_token_ids(tokenizer, bos_id)

    token_id_lists: list[list[int]] = []
    completion_starts: list[int] = []
    for item in items:
        # (1) Splice-safe: combine in CHARACTER space, THEN tokenize the whole thing.
        joint = _joint_tokenization_with_completion_start(
            tokenizer,
            prefix_text=item.prefix_text,
            completion_text=item.completion_text,
        )
        full_ids = joint.token_ids
        polluting_ids = special_ids.intersection(full_ids)
        if polluting_ids:
            # PRESERVE the document's original tokens and run the DiffMean forward
            # as-is: only the COMPLETION span is pooled (the prefix — including our
            # injected BOS at position 0 and any in-data special — is pooled OUT),
            # so an in-data special is a benign signal, not corruption. Warn for
            # observability rather than refusing.
            logger.warning(
                "feature_diffmean_bag: the prefix/completion text encodes "
                f"in-data special token id(s) {sorted(polluting_ids)} even "
                "with add_special_tokens=False (a literal special-token string "
                "such as '<bos>'). Preserving the document's tokens and running "
                "the DiffMean SAE forward as-is (only the completion is pooled). "
                f"prefix (first 80 chars): {item.prefix_text[:80]!r}; "
                f"completion (first 80 chars): {item.completion_text[:80]!r}."
            )
        # (2) Prepend EXACTLY ONE BOS (add_special_tokens=False plus the
        #     special-id pollution check above ⇒ full_ids carries none).
        # (3) Front-clamp to n_ctx: BOS + prefix are context, so a long prefix can
        #     only truncate the completion TAIL, never displace it.
        model_input = ([bos_id] + list(full_ids))[:n_ctx]

        # (4) OFF-BY-ONE: completion-bearing tokens occupy
        #     [1 + completion_start_token, len). The leading +1 is the prepended
        #     BOS at position 0; everything before that span is context and is
        #     pooled OUT.
        completion_start = 1 + joint.completion_start_token
        if completion_start >= len(model_input):
            raise FeatureGcgError(
                "feature_diffmean_bag: after prepending the BOS and front-clamping "
                f"to n_ctx={n_ctx}, no completion tokens remain (BOS + prefix-context "
                f"tokens fill the window: 1 + completion_start_token="
                f"{joint.completion_start_token} >= n_ctx). prefix (first 80 chars): "
                f"{item.prefix_text[:80]!r}."
            )
        token_id_lists.append(model_input)
        completion_starts.append(completion_start)

    # Each entry is [seq_len_i, n_features], 1:1 with model_input (the extractor
    # injects NO BOS of its own — asserted here so a future regression raises
    # loudly instead of silently shifting the pooled span). Max-pool ONLY the
    # completion span acts[completion_start:], excluding the BOS (position 0) and
    # pure-prefix context tokens.
    #
    # The pool runs as a per-item reduction INSIDE the extractor (#1240) rather
    # than over a fully materialized list. Accumulating the raw
    # [seq_len, n_features] tensors made peak memory scale with the item count
    # instead of `extraction_batch_size`: ~113 MiB per item at a 65k-wide SAE and
    # ~432 tokens, i.e. ~45 GiB for 400 items, to produce a ~105 MiB result. The
    # arithmetic per item is unchanged, so the output is bitwise identical.
    def _pool_completion_span(acts: t.Tensor, index: int) -> t.Tensor:
        input_ids = token_id_lists[index]
        if acts.shape[0] != len(input_ids):
            raise FeatureGcgError(
                "feature_diffmean_bag: the extractor returned "
                f"{acts.shape[0]} activation rows for a {len(input_ids)}-token "
                "input; acts must be 1:1 with the input (no BOS injection on this "
                "path) for the completion span to be valid."
            )
        return acts[completion_starts[index] :].max(dim=0).values

    pooled = _run_batched_extractor(
        model=model,
        sae=sae,
        token_id_lists=token_id_lists,
        layer=sae_layer,
        hook_name=hook_name,
        batch_size=extraction_batch_size,
        per_item_reduce=_pool_completion_span,
    )
    if len(pooled) != len(token_id_lists):
        raise FeatureGcgError(
            "feature_diffmean_bag: the extractor returned "
            f"{len(pooled)} pooled rows for {len(token_id_lists)} inputs."
        )
    return t.stack(pooled, dim=0).to(device)


def _joint_tokenization_with_completion_start(
    tokenizer: Any,
    *,
    prefix_text: str,
    completion_text: str,
) -> _JointTokenization:
    """Tokenize ``prefix_text + completion_text`` and locate completion tokens.

    Production Gemma tokenizers expose character offsets for the joint
    tokenization.  Those offsets are the authoritative way to find the
    completion span because a token can merge across the prefix/completion text
    boundary.  When offsets are unavailable (mainly tiny unit-test stubs), keep
    the old prefix-token equality guard so an unalignable boundary merge still
    raises loudly instead of producing a guessed span.
    """
    full_text = prefix_text + completion_text
    encoded_with_offsets = _try_encode_with_offsets(tokenizer, full_text)
    if encoded_with_offsets is not None:
        full_ids, offsets = encoded_with_offsets
        prefix_char_end = len(prefix_text)
        for idx, (char_start, char_end) in enumerate(offsets):
            if char_start > char_end:
                raise FeatureGcgError(
                    "feature_diffmean_bag: tokenizer returned an invalid offset "
                    f"span [{char_start}, {char_end}) for token index {idx}; "
                    "cannot isolate the completion span."
                )
            if char_start < 0 or char_end > len(full_text):
                raise FeatureGcgError(
                    "feature_diffmean_bag: tokenizer returned an out-of-range "
                    f"offset span [{char_start}, {char_end}) for text length "
                    f"{len(full_text)}; cannot isolate the completion span."
                )
            if char_end > prefix_char_end:
                return _JointTokenization(
                    token_ids=full_ids,
                    completion_start_token=idx,
                )
        raise FeatureGcgError(
            "feature_diffmean_bag: the joint prefix+completion tokenization has "
            "zero completion-bearing tokens beyond the prefix; there is nothing "
            f"to pool. completion (first 80 chars): {completion_text[:80]!r}."
        )

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    if list(full_ids[:n_prefix]) != list(prefix_ids):
        raise FeatureGcgError(
            "feature_diffmean_bag: the tokenizer does not expose character offsets "
            "and the combined prefix+completion tokenization does not start with "
            "the prefix's own tokens (a subword merged across the boundary), so "
            "the completion span cannot be isolated. prefix (first 80 chars): "
            f"{prefix_text[:80]!r}; completion (first 80 chars): "
            f"{completion_text[:80]!r}."
        )
    if len(full_ids) <= n_prefix:
        raise FeatureGcgError(
            "feature_diffmean_bag: the combined prefix+completion tokenization "
            "has zero completion tokens beyond the prefix; there is nothing to "
            f"pool. completion (first 80 chars): {completion_text[:80]!r}."
        )
    return _JointTokenization(token_ids=list(full_ids), completion_start_token=n_prefix)


def _try_encode_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    # Preserve the legacy instance-attribute lookup exactly. ``callable(tokenizer)``
    # differs for pathological objects that expose an instance-level ``__call__``.
    call = getattr(tokenizer, "__call__", None)  # noqa: B004
    if not callable(call):
        return None
    try:
        encoded = call(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        return None
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise FeatureGcgError(
            "feature_diffmean_bag: tokenizer returned "
            f"{len(offsets)} offset span(s) for {len(input_ids)} token id(s); "
            "cannot isolate the completion span."
        )
    return input_ids, offsets


def model_n_ctx(model: Any) -> int:
    """The base model's context window (``model.cfg.n_ctx``), or raise loudly.

    Bounds the DiffMean SAE forward (the front-clamp on how many prefix tokens
    the base model sees) to the SAME window the extraction uses.
    """
    cfg = getattr(model, "cfg", None)
    n_ctx = getattr(cfg, "n_ctx", None) if cfg is not None else None
    if n_ctx is None:
        raise FeatureGcgError(
            "base model has no cfg.n_ctx; cannot bound the SAE forward "
            "without the model's context window."
        )
    return int(n_ctx)


def encdec_dec_sequence_len(model: Any) -> int:
    """The encdec decoder's max sequence length (``model.dec_config.sequence_len``).

    This is the EXACT ``max_seq_len`` the score stage uses to left-truncate a
    prefix before teacher-forcing (``runtime_state.py`` binds
    ``max_seq_len=model.dec_config.sequence_len``;
    ``teacher_force.score_sample`` truncates against it).  The feature-GCG
    derive threads this into ``build_class_margin_members`` so it can RAISE
    (rather than silently optimize under a divergent, untruncated layout) if a
    prefix + completion would exceed it — the derive does not yet mirror
    score-time left-truncation, and a member built past ``sequence_len`` would
    be scored under a shorter, truncated prefix, breaking derive/score layout
    agreement.  Raises loudly if the attribute chain is absent so a model-shape
    drift can never silently disable the guard.
    """
    dec_config = getattr(model, "dec_config", None)
    sequence_len = (
        getattr(dec_config, "sequence_len", None) if dec_config is not None else None
    )
    if sequence_len is None:
        raise FeatureGcgError(
            "encdec model has no dec_config.sequence_len; cannot bound the "
            "feature-GCG member length against the score-time max_seq_len "
            "(score left-truncates the prefix to this window, so the derive "
            "must know it to keep the member layout in agreement)."
        )
    return int(sequence_len)


def base_model_tokenizer(model: Any) -> Any:
    """The base model's OWN tokenizer (``model.tokenizer``), or raise loudly.

    This is the tokenizer the model runs its forward under, and the exact one
    :func:`steering.extract_features.extract_document_features` uses at score
    time (``model.tokenizer``).  The feature-GCG derive tokenizes the DiffMean
    SAE forward with THIS tokenizer, never a separately-configured
    ``extraction_tokenizer_id`` — so the derive can never tokenize with a
    different vocabulary than the base model it forwards through / than the score
    stage conditions under. Removing the config-id seam removes the whole
    "half-updated config" drift class.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise FeatureGcgError(
            "base model has no .tokenizer; the DiffMean SAE forward must tokenize "
            "with the base model's OWN tokenizer (the one extract_document_features "
            "uses at score time), so a model without one is a real config/loader bug."
        )
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Percentile mapping + selection (pure, sign-critical)
# ─────────────────────────────────────────────────────────────────────────────


def _mean_percentiles(acts: t.Tensor, percentile_table: t.Tensor) -> t.Tensor:
    """Mean over completions of each feature's percentile.

    ``acts`` is ``[n_completions, n_features]`` raw magnitudes.  For every
    completion, every feature's magnitude is mapped to its percentile via
    ``magnitude_to_percentile`` (the percentile authority), then averaged over
    completions.  Returns ``[n_features]`` float64.
    """
    table_device = percentile_table.device
    n_features = acts.shape[1]
    feature_ids = t.arange(n_features, dtype=t.int64, device=table_device)
    per_completion = [
        magnitude_to_percentile(
            feature_ids,
            acts[i].to(table_device),
            percentile_table,
        )
        for i in range(acts.shape[0])
    ]
    stacked = t.stack(per_completion, dim=0)  # [n_completions, n_features]
    return stacked.mean(dim=0)


def _mean_percentiles_for_feature_ids(
    acts: t.Tensor,
    percentile_table: t.Tensor,
    *,
    feature_ids: t.Tensor,
) -> t.Tensor:
    """Unquantized mean percentiles for selected IDs, scattered to full width."""

    table_device = percentile_table.device
    selected_ids = feature_ids.to(device=table_device, dtype=t.int64)
    acts_on_table_device = acts.to(table_device)
    selected_per_completion = [
        magnitude_to_percentile(
            selected_ids,
            acts_on_table_device[i][selected_ids],
            percentile_table,
        )
        for i in range(acts.shape[0])
    ]
    selected_mean = t.stack(selected_per_completion, dim=0).mean(dim=0)
    full_width_mean = t.zeros(
        acts.shape[1],
        dtype=selected_mean.dtype,
        device=table_device,
    )
    full_width_mean[selected_ids] = selected_mean
    return full_width_mean


def _cluster_projected_mean_percentiles(
    acts: t.Tensor,
    percentile_table: t.Tensor,
    *,
    cluster_contract: ClusterFeatureContract,
) -> t.Tensor:
    """Raw-percent-rank max-before-mean projection for clustered DiffMean."""

    consolidation = cluster_contract.consolidation
    n_features = int(acts.shape[1])
    all_ids = t.arange(n_features, dtype=t.int64)
    if consolidation.feature_mask is not None:
        allowed_mask = consolidation.feature_mask.keep_mask(all_ids)
        allowed_ids = all_ids[allowed_mask]
    else:
        allowed_ids = all_ids

    table_cpu = percentile_table.to("cpu")
    total = t.zeros(n_features, dtype=t.float64)
    for completion_idx in range(int(acts.shape[0])):
        try:
            percentile_ranks = magnitude_to_percentile(
                allowed_ids,
                acts[completion_idx]
                .detach()
                .to(dtype=t.float32, device="cpu")[allowed_ids],
                table_cpu,
            )
        except ValueError as exc:
            raise FeatureGcgError(
                "feature_diffmean_bag: allowed cluster feature percentile "
                f"statistics are invalid: {exc}"
            ) from exc
        # DiffMean reproduces no packed training artifact: its bag carries only
        # feature IDs and configured buckets, so uint8 conversion would merely
        # coarsen 20,001 percent ranks to 254 levels before ranking. Consolidate
        # on the raw float64 [0.5, 99.5] ranks. The generic primitive reserves
        # its first channel for normalized percentiles, so carry the raw ranks
        # as an aligned channel and return that channel's amax. The
        # amax-before-mean ordering below is load-bearing; this scale-only
        # validation channel is not.
        representatives, _, projected = consolidation.consolidate_segment(
            allowed_ids,
            percentile_ranks / 100.0,
            percentile_ranks,
        )
        row = t.zeros(n_features, dtype=t.float64)
        row[representatives] = projected.to(t.float64)
        # Deliberately accumulate in input order in float64. Projecting class
        # means first would implement mean-before-max, a different statistic.
        total += row
    return total / float(acts.shape[0])


def _select_bag(
    diff: t.Tensor,
    *,
    mode: SteeringMode,
    k_pos: int,
    k_neg: int,
    positive_bucket: int,
    negative_bucket: int,
    allowed_ids: t.Tensor | None = None,
) -> FeatureConditioning:
    """Select top/bottom features by ``diff`` and assign buckets per the sign chain.

    Deterministic: ties on ``diff`` are broken by ascending feature id.

    ``allowed_ids`` restricts the ranking to a compacted checkpoint's allowed SAE
    feature vocabulary ``V`` (see
    :func:`steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids`):
    every feature index NOT in ``allowed_ids`` is masked to ``-inf`` for the
    top-``k`` ordering and to ``+inf`` for the bottom-``k`` ordering BEFORE
    ranking, so an out-of-``V`` feature can never outrank an in-``V`` feature no
    matter how large its ``|diff|`` is. ``allowed_ids=None`` (an uncompacted
    checkpoint) reproduces the unrestricted, full-``n_sae`` selection
    byte-for-byte. If ``V`` has fewer than ``k_pos`` (or ``k_neg``) members, the
    masked-out padding that would otherwise fill the slice is dropped rather
    than selected -- the returned bag never contains an id outside ``V``, and
    selecting from a small or empty ``V`` never crashes or pads.
    """
    diff_values: list[float] = [float(v) for v in diff.tolist()]
    n_features = len(diff_values)

    k_pos = max(0, int(k_pos))
    k_neg = max(0, int(k_neg))

    allowed_set: set[int] | None = (
        None if allowed_ids is None else {int(fid) for fid in allowed_ids.tolist()}
    )

    def _in_vocab(fid: int) -> bool:
        return allowed_set is None or fid in allowed_set

    # top-k by +diff: largest diff first, ascending fid tie-break. A feature
    # outside V is masked to -inf so it can never outrank a real in-V diff.
    top_values = (
        diff_values
        if allowed_set is None
        else [v if _in_vocab(f) else float("-inf") for f, v in enumerate(diff_values)]
    )
    order_desc = sorted(range(n_features), key=lambda f: (-top_values[f], f))
    top_fids = [fid for fid in order_desc[:k_pos] if _in_vocab(fid)]

    # bottom-k by -diff: smallest diff first, ascending fid tie-break. A feature
    # outside V is masked to +inf so it can never outrank a real in-V diff.
    bottom_values = (
        diff_values
        if allowed_set is None
        else [v if _in_vocab(f) else float("inf") for f, v in enumerate(diff_values)]
    )
    order_asc = sorted(range(n_features), key=lambda f: (bottom_values[f], f))
    bottom_fids = [fid for fid in order_asc[:k_neg] if _in_vocab(fid)]

    if mode is SteeringMode.POSITIVE:
        top_bucket, bottom_bucket = positive_bucket, negative_bucket
    else:  # SteeringMode.NEGATIVE -> the MIRROR.
        top_bucket, bottom_bucket = negative_bucket, positive_bucket

    top_set = set(top_fids)
    bag: dict[int, int] = dict.fromkeys(top_fids, top_bucket)
    for fid in bottom_fids:
        if fid in top_set:
            # Overlap: keep the side whose signed score is larger == |diff|.
            #   diff > 0 -> top side wins; diff < 0 -> bottom side wins;
            #   diff == 0 -> tie, keep the top side (deterministic convention).
            if diff_values[fid] < 0.0:
                bag[fid] = bottom_bucket
        else:
            bag[fid] = bottom_bucket

    if allowed_set is not None and bag:
        # Fail-loud belt-and-suspenders: the masking above should make this
        # unreachable. Reuses Task 1's restrict_to_vocab (the same kept/dropped
        # split used at the model-load boundary) rather than a bespoke check.
        bag_ids = t.tensor(sorted(bag.keys()), dtype=t.int64)
        _kept, dropped = restrict_to_vocab(bag_ids, allowed_ids)
        if dropped.numel() > 0:
            raise FeatureGcgError(
                "_select_bag: internal invariant violated -- selected feature "
                f"id(s) {[int(x) for x in dropped.tolist()]} fall outside the "
                f"allowed vocabulary V (|V|={len(allowed_set)}); this should be "
                "unreachable after masking non-V indices to +/-inf before "
                "ranking."
            )

    return FeatureConditioning.from_dict(bag)


def _resolve_mode(mode: SteeringMode | str) -> SteeringMode:
    try:
        return SteeringMode(mode)
    except ValueError as exc:
        raise FeatureGcgError(
            f"feature_diffmean_bag: unknown steering mode {mode!r}; expected "
            f"one of {[m.value for m in SteeringMode]}."
        ) from exc
