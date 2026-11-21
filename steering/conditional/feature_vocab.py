"""The encoder's compacted feature dictionary (issue #695).

``FeatureEncoder`` allocates one embedding row per effective SAE feature across
``feature_embed`` and every ``value_embeds[*]`` table. Rows excluded by a
``feature_mask`` or consolidated into another static-cluster representative are
**exactly-dead weight**: they have zero effect on outputs and receive exactly-zero
gradient, because legal batches cannot address them and cross-attention masks encoder
padding positions as *keys* (softmax weight exactly 0). A ``FeatureVocab`` is the sorted
list of original SAE feature ids that survive, and it sizes those tables.

Two things make this safe, and both are easy to get wrong:

1. **Padding.** Every producer pads ``enc_feature_ids`` with literal ``0`` — a real
   SAE id a whitelist may exclude — and ``FeatureEncoder.forward`` embeds padding
   slots anyway. So the remap must be MASK-AWARE: padding maps to compact row 0
   (provably harmless), and only ``mask=True`` slots are validated.

2. **``-1`` is not a safe sentinel under torch.compile.** ``nn.Embedding(-1)`` raises
   in eager but *silently returns the last row* when compiled. The encoder therefore
   carries a ``torch._assert_async`` (sync-free, fires in both modes). This module
   supplies the maps; the encoder supplies the guard.

The single load-bearing invariant:

    ``FeatureVocab.from_mask(mask, n_sae)`` is a PURE, DETERMINISTIC function of
    ``(mask, n_sae)``; for cluster-aware models, ``effective_feature_vocab`` is
    likewise pure in ``(mask, clusters, n_sae)``.

That is what lets the existing ``bl_``/``wl_<16hex>`` mask leaf together with the
conditional ``fc_<64hex>`` cluster leaf uniquely pin the vocab with **no new
selection-hash component** — preserving invariant I1. Without clusters, the literal
mask-only construction and identity remain unchanged.

And the invariant that keeps the ``None`` sentinel honest:

    A **non-``None``** ``FeatureVocab`` ALWAYS means "genuinely compacted" — ``V < n_sae``.

Neither constructor can ever return an identity vocab, but they refuse it *differently*, and
the asymmetry is deliberate:

* :meth:`FeatureVocab.from_mask` **normalizes** an identity to ``None``. An all-allowing mask
  is a legitimate config — a WHITELIST of every id; note ``{mode: blacklist, ids: []}`` is NOT
  one, ``FeatureMask`` rejects it ("resolved to an empty set") — and it must build a
  full-width, byte-identical encoder, which is invariant I5.
* :func:`feature_vocab_from_state_dict` **raises**. A checkpoint carrying identity buffers is
  not a legitimate state at all: because ``from_mask`` normalizes, the model that would have
  registered those buffers never registers them, so no writer here can emit one. Such a file
  came from a writer we do not understand, and guessing what it meant would either
  reinterpret every embedding row or silently discard information. Anomaly ⇒ raise.

If either ever returned an identity, it would be simultaneously *valid* and *"compacted"*,
poisoning every consumer that reads ``vocab is not None`` as "this model was compacted".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch as t
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from errors import FeatureVocabError
from steering.conditional.feature_mask import FeatureMask
from steering.refined_types import (
    NonEmptyNonNegativeIntTupleLike,
    PositiveInt,
    PositiveIntLike,
)

# How many offending ids an out-of-vocab error may list before it truncates. The count is
# always reported in full — a truncated message must never understate the problem.
_MAX_REPORTED_OOV_IDS = 8

# state_dict keys. The single source of truth for both the encoder (which registers
# these buffers) and the loaders (which read them back). PR B wires the encoder side.
VOCAB_IDS_KEY = "encoder.feature_vocab_ids"
VOCAB_N_SAE_KEY = "encoder.feature_vocab_n_sae"
FEATURE_EMBED_KEY = "encoder.feature_embed.weight"

# meta_*.json key mirroring the buffers (a fingerprint, not the id list).
META_KEY = "feature_vocab"


@beartype
@dataclass(frozen=True)
class FeatureVocab:
    """The encoder's compacted feature dictionary.

    ``original_ids[j]`` is the ORIGINAL SAE feature id of compact embedding row ``j``.
    Stored sorted + de-duplicated (like :class:`FeatureMask`), so compact id == rank of
    the original id, and two logically-identical vocabs are ``==`` and hash alike.

    ``n_sae_features`` is the ORIGINAL dictionary width and stays the public number.
    It is NOT recoverable from a compacted table's row count (which is ``V``), which is
    why it is persisted explicitly.
    """

    n_sae_features: PositiveIntLike
    original_ids: NonEmptyNonNegativeIntTupleLike

    def __post_init__(self) -> None:
        canonical_ids = tuple(sorted(set(self.original_ids)))
        if canonical_ids[-1] >= self.n_sae_features:
            raise FeatureVocabError(
                f"feature vocab id {canonical_ids[-1]} is out of range "
                f"[0, {self.n_sae_features}) for "
                f"n_sae_features={self.n_sae_features}"
            )
        object.__setattr__(self, "original_ids", canonical_ids)

    # -- shape -------------------------------------------------------------

    @property
    def size(self) -> PositiveInt:
        """V — the number of compact embedding rows."""
        return PositiveInt.parse(len(self.original_ids))

    @property
    def is_identity(self) -> bool:
        return self.original_ids == tuple(range(self.n_sae_features))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mask(cls, mask: FeatureMask | None, n_sae_features: int) -> FeatureVocab | None:
        """Resolve the mask-only vocab. ``None`` ⇒ ``None`` (no compaction).

        Works identically for blacklist and whitelist: the vocab is exactly
        ``mask.allowed_universe_mask(n_sae_features)``.

        This remains the public mask-only constructor. Cluster-aware production paths
        must use ``effective_feature_vocab`` so consolidated-away rows are excluded;
        that function literally delegates here when clusters are absent.

        An identity result is normalized to ``None`` so the byte-identical
        (zero-added-ops) encoder path cannot be entered accidentally.
        """
        if mask is None:
            return None
        allowed = mask.allowed_universe_mask(n_sae_features)
        ids = tuple(int(i) for i in allowed.nonzero().flatten().tolist())
        if not ids:
            raise FeatureVocabError(
                f"feature_mask (mode={mask.mode.value}) allows zero features out of "
                f"{n_sae_features} — the encoder would have no embedding rows. "
                "Widen the mask until at least one feature is allowed. "
                "(A frequency band that excludes everything usually means the band and the "
                "statistics disagree about which SAE they describe.)"
            )
        vocab = cls(n_sae_features=n_sae_features, original_ids=ids)
        return None if vocab.is_identity else vocab

    # -- tensor maps -------------------------------------------------------

    def old_to_new_tensor(self, *, device: Any = None) -> t.Tensor:
        """int64[n_sae_features]: original id → compact id, ``-1`` where out-of-vocab."""
        out = t.full((self.n_sae_features,), -1, dtype=t.int64, device=device)
        out[self.new_to_old_tensor(device=device)] = t.arange(
            self.size, dtype=t.int64, device=device
        )
        return out

    def new_to_old_tensor(self, *, device: Any = None) -> t.Tensor:
        """int64[V]: compact id → original id."""
        return t.tensor(self.original_ids, dtype=t.int64, device=device)

    # -- id translation ----------------------------------------------------

    def to_compact(self, ids: t.Tensor) -> t.Tensor:
        """Original ids → compact ids. An out-of-vocab id ALWAYS raises.

        There is exactly ONE codepath. This method never returns ``-1``, and there is no
        option, keyword or config that makes it.

        Two distinct failures, both loud:

        * An id outside ``[0, n_sae_features)`` is not merely out-of-vocab — it is not an
          SAE feature id at all.
        * An id that is in range but was masked OUT of the vocabulary has no embedding row.

        **Why there is no sentinel escape hatch.** ``-1`` is a LETHAL return value:

            Under ``torch.compile``, ``nn.Embedding(-1)`` does not raise — it SILENTLY
            RETURNS THE LAST ROW.

        So a caller handed ``-1`` does not get an error, it gets the WRONG SAE feature's
        embedding, and conditions on and scores that feature with nothing downstream to
        catch it. An opt-in ``on_missing="sentinel"`` briefly existed here. It is gone: an
        escape hatch for a silent failure is still a codepath, and the next caller in a
        hurry reaches for it. The strict path is the only path.

        ``FeatureEncoder._to_compact`` (the identically-named *private* method) does use the
        sentinel internally — it can, because it is MASK-AWARE (it knows which slots are
        dead padding) and it guards the value with ``torch._assert_async`` before it reaches
        an embedding. A public caller has neither the mask nor the guard, which is exactly
        why it does not get the option.

        Usable from tensor code: the out-of-vocab test is one fused reduction, the same
        shape as the range check above it, and it only syncs to the host on the failure
        path. (It is *not* for use inside a compiled region — that is what the encoder's
        ``_assert_async`` remap exists for.)
        """
        if ids.dtype.is_floating_point or ids.is_complex() or ids.dtype == t.bool:
            raise FeatureVocabError(
                f"feature ids must be an integer tensor, got {ids.dtype}"
            )
        if ids.numel() == 0:
            return ids.to(t.int64)
        ids64 = ids.to(t.int64)
        if bool(((ids64 < 0) | (ids64 >= self.n_sae_features)).any()):
            raise FeatureVocabError(
                f"feature id out of range [0, {self.n_sae_features}): "
                f"min={int(ids64.min())}, max={int(ids64.max())}"
            )
        compact = self.old_to_new_tensor(device=ids.device)[ids64]
        if bool((compact < 0).any()):
            raise FeatureVocabError(self._oov_message(ids64[compact < 0]))
        return compact

    def _oov_message(self, offenders: t.Tensor) -> str:
        """Name the offending ids AND the remedy. Truncates the list, never the count."""
        distinct = t.unique(offenders)  # unique() returns them sorted
        total = int(distinct.numel())
        shown = [int(i) for i in distinct[:_MAX_REPORTED_OOV_IDS].tolist()]
        listed = ", ".join(str(i) for i in shown)
        if total > _MAX_REPORTED_OOV_IDS:
            listed += f", … (+{total - _MAX_REPORTED_OOV_IDS} more)"
        return (
            f"{total} feature id(s) are valid SAE ids but are NOT in this encoder's "
            f"compacted vocabulary (V={self.size} of n_sae={self.n_sae_features}), so it "
            f"has no embedding row for them: [{listed}].\n"
            "This RAISES instead of returning the -1 sentinel because -1 is not safe to "
            "return: under torch.compile, nn.Embedding(-1) silently returns the LAST row, "
            "so you would condition on — and score — the WRONG SAE feature, with no error "
            "anywhere.\n"
            "Remedy: these ids were excluded by the feature_mask and/or consolidated "
            "away by feature_clusters this model was trained with, so drop them from "
            "your input. Inspect mask provenance with `uv run python -m scripts.analysis."
            "dump_feature_mask <checkpoint_dir>` and configure the same mask (if any) "
            "and clusters from the training record.\n"
            "There is deliberately NO option to get the -1 sentinel back instead: an "
            "escape hatch for a silent failure is still a codepath, and this one ends in "
            "conditioning on the wrong SAE feature."
        )

    def to_original(self, ids: t.Tensor) -> t.Tensor:
        """Compact ids → original ids. Raises on ids outside ``[0, V)``."""
        if ids.dtype.is_floating_point or ids.is_complex() or ids.dtype == t.bool:
            raise FeatureVocabError(
                f"feature ids must be an integer tensor, got {ids.dtype}"
            )
        if ids.numel() == 0:
            return ids.to(t.int64)
        ids64 = ids.to(t.int64)
        if bool(((ids64 < 0) | (ids64 >= self.size)).any()):
            raise FeatureVocabError(
                f"compact feature id out of range [0, {self.size}): "
                f"min={int(ids64.min())}, max={int(ids64.max())}"
            )
        return self.new_to_old_tensor(device=ids.device)[ids64]

    # -- row selection (slice-init + W_dec subsetting) ---------------------

    def select_rows(self, full: t.Tensor) -> t.Tensor:
        """Take the vocab's rows out of a full ``(n_sae_features, ...)`` tensor.

        A pure gather — ``out[j]`` IS ``full[original_ids[j]]``, bit for bit. This is
        what makes slice-init byte-exact: draw the FULL table (consuming the IDENTICAL
        RNG stream, which is why the model is built on ``meta`` — see
        ``FeatureEncoder._init_feature_indexed_``), then keep only these rows. Every
        surviving row of ``init_weights()`` then matches the uncompacted run exactly.

        Two boundaries sit outside that guarantee:

        * **FP8.** The compact table's row 0 is ``feature_embed[original_ids[0]]`` where
          the reference's is ``feature_embed[0]`` (they coincide only when SAE id 0
          survives the mask), and padding embeds row 0. torchao's TENSORWISE amax is a
          whole-tensor reduction that includes those padded rows, so a different padding
          row perturbs the quantization scale of every REAL activation. Not a correctness
          defect — padding contributes exactly zero to the output and exactly zero
          gradient in both models (it is masked as a cross-attention KEY) — but not
          bitwise.
        * **SAE-init ``svd``.** Its eigenbasis is the covariance of the rows fed to it,
          and requirement 6 feeds it the sub-dictionary deliberately, so a compacted
          ``svd`` init cannot be the row-subset of an uncompacted one.

        SAE-init ``random`` — what the queued run uses — **is** bitwise: it projects the
        full ``(n_sae, d)`` matrix, rescales against the FULL table's norm, and calls this
        method LAST. Subsetting *before* the projection would be mathematically equal but
        would run a different-shaped GEMM; subsetting last means the compacted and
        reference runs execute the identical one.

        Both boundaries are pinned by tests; see ``tests/steering/
        test_encdec_compaction_byte_exact_e2e.py::TestByteExactnessBoundaries``.

        Also used to subset the SAE decoder matrix ``W_dec``.
        """
        if full.shape[0] != self.n_sae_features:
            raise FeatureVocabError(
                f"select_rows expects a tensor with n_sae_features={self.n_sae_features} "
                f"rows, got shape {tuple(full.shape)}"
            )
        return full[self.new_to_old_tensor(device=full.device)]

    # -- serialization -----------------------------------------------------

    def canonical_dict(self) -> dict:
        """Canonical hash/metadata form."""
        return {
            "n_sae_features": self.n_sae_features,
            "original_ids": list(self.original_ids),
        }

    def fingerprint(self) -> str:
        """sha256 of the canonical form.

        Covers ``n_sae_features`` as well as the ids — two vocabs with identical ids
        but different original widths are DIFFERENT vocabs, and the ``bl_``/``wl_``
        checkpoint leaf does not distinguish them (its hash covers only mode + ids).
        """
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def metadata_dict(self) -> dict:
        """The ``meta_*.json`` mirror: a fingerprint, not the id list.

        The persistent buffers in the checkpoint are the load-bearing record; this
        exists so tooling can compare a checkpoint's vocab without loading the weights.
        """
        return {
            "n_sae_features": self.n_sae_features,
            "vocab_size": self.size,
            "sha256": self.fingerprint(),
        }


# ===========================================================================
# Checkpoint helpers
# ===========================================================================


def feature_vocab_from_state_dict(
    model_data: Mapping[str, Any],
) -> tuple[int, FeatureVocab | None]:
    """Recover ``(n_sae_features, vocab)`` from a checkpoint state dict ALONE.

    No training config, no sidecar — the checkpoint describes itself (requirement 1).

    A legacy checkpoint carries no vocab buffers; for an identity vocab the old
    shape-inference (``feature_embed.weight.shape[0]``) is still CORRECT, so legacy
    loads are unchanged (requirement 5).

    Upholds the module's ``None`` invariant — a non-``None`` vocab ALWAYS means genuinely
    compacted — but it upholds it by REFUSING, not by normalizing: buffers describing an
    IDENTITY vocab (``V == n_sae``) RAISE. No writer in this repo can emit them
    (:meth:`FeatureVocab.from_mask` normalizes an identity to ``None``, so the model never
    registers the buffers), so such a file came from a writer we do not understand and we
    must not guess what it meant. See the module docstring for why the two constructors
    refuse it differently.

    **This is a PURE QUERY.** It does not mutate ``model_data``. An earlier version did — it
    normalized an identity to ``None`` and DELETED the two buffers from the caller's dict, so
    that the uncompacted model this implied could still ``strict``-load. That leniency is
    gone, and the mutation went with it: a reader that silently deletes its caller's keys is
    a trap regardless of how well-intentioned. Pinned by
    ``test_the_function_is_a_PURE_QUERY_and_does_not_mutate_the_state_dict``.
    """
    if FEATURE_EMBED_KEY not in model_data:
        raise FeatureVocabError(
            f"state dict is missing {FEATURE_EMBED_KEY!r} — is this an encoder-decoder checkpoint?"
        )
    rows = int(model_data[FEATURE_EMBED_KEY].shape[0])

    has_ids = VOCAB_IDS_KEY in model_data
    has_n_sae = VOCAB_N_SAE_KEY in model_data
    if has_ids != has_n_sae:
        present, missing = (
            (VOCAB_IDS_KEY, VOCAB_N_SAE_KEY) if has_ids else (VOCAB_N_SAE_KEY, VOCAB_IDS_KEY)
        )
        raise FeatureVocabError(
            f"corrupt checkpoint: {present!r} is present but {missing!r} is missing. "
            "A compacted checkpoint must carry both vocab buffers."
        )

    if not has_ids:
        # Legacy / uncompacted: identity vocab, and rows == n_sae_features.
        return rows, None

    ids_buffer = model_data[VOCAB_IDS_KEY]
    n_sae_buffer = model_data[VOCAB_N_SAE_KEY]

    # Reject a non-integral buffer rather than TRUNCATING it. `int(3.7)` is 3, so a float
    # buffer would be read as a different, entirely plausible vocabulary — every feature
    # silently mapped to the wrong embedding row, with nothing downstream to catch it.
    # `torch.bool` is the same trap wearing a different dtype: it is neither
    # floating-point nor complex, so it survives those two checks, and `int(True)` is 1 —
    # a bool buffer would silently coerce to a plausible-looking 0/1 vocabulary instead of
    # raising. The typed ``FeatureVocab`` constructor already guards the Python-int
    # boundary; the tensor buffer needs its own explicit guard for the same reason.
    # Unsorted/duplicate ids are checked below on exactly the same reasoning (they are
    # unreachable from the production encoder, which is *why* they are checked rather than
    # assumed); a non-integer dtype is unreachable by the same argument and must not be the
    # one corruption we quietly accept.
    for key, buf in ((VOCAB_IDS_KEY, ids_buffer), (VOCAB_N_SAE_KEY, n_sae_buffer)):
        if (
            not isinstance(buf, t.Tensor)
            or buf.dtype.is_floating_point
            or buf.is_complex()
            or buf.dtype == t.bool
        ):
            raise FeatureVocabError(
                f"corrupt checkpoint: {key!r} must be an integer tensor, got "
                f"{getattr(buf, 'dtype', type(buf).__name__)}. The vocabulary buffers are "
                "the authoritative record of which ORIGINAL SAE id each compact embedding "
                "row holds; a non-integer value (torch.bool included — it silently coerces "
                "True/False to 1/0) cannot be one, and coercing it would silently produce a "
                "different, plausible-looking vocabulary."
            )

    # RANK/SHAPE, before the reads below can turn it into a bare PyTorch error. Both
    # callers document and depend on the FeatureVocabError contract (dump_feature_mask's
    # docstring promises it; build_encdec_model raises it for the meta-mirror mismatch
    # immediately after this call), and FeatureVocabError is a NanoGPTError — NOT a
    # RuntimeError and NOT a TypeError — so what PyTorch throws here could never be caught
    # by a handler written against that contract:
    #
    #   int(n_sae_buffer.item())            on >1 element  → RuntimeError: a Tensor with N
    #                                                        elements cannot be converted
    #                                                        to Scalar
    #   tuple(int(i) for i in ids.tolist()) on a 0-dim     → TypeError: 'int' object is not
    #                                                        iterable
    #                                       on a 2-D       → TypeError: int() argument must
    #                                                        be ... not 'list'
    #
    # Unreachable from the production encoder, which registers a 1-D ids buffer and a 0-dim
    # scalar — which is exactly the reason it is checked rather than assumed, like every
    # sibling check here.
    if ids_buffer.ndim != 1:
        raise FeatureVocabError(
            f"corrupt checkpoint: {VOCAB_IDS_KEY!r} must be a 1-D tensor listing one "
            f"ORIGINAL SAE id per compact embedding row, got shape "
            f"{tuple(ids_buffer.shape)} (ndim={ids_buffer.ndim}). The buffer's order IS "
            "the row mapping, so a tensor of the wrong rank does not describe one at all. "
            "Re-save the checkpoint from a model built by the current encoder, or recover "
            "the vocabulary with `uv run python -m scripts.analysis.dump_feature_mask "
            "<checkpoint_dir>`."
        )
    if n_sae_buffer.numel() != 1:
        raise FeatureVocabError(
            f"corrupt checkpoint: {VOCAB_N_SAE_KEY!r} must be a scalar (the production "
            f"encoder registers a 0-dim tensor) holding the ORIGINAL SAE dictionary "
            f"width, got shape {tuple(n_sae_buffer.shape)} with {n_sae_buffer.numel()} "
            "elements. A width is one number; this is not one. Re-save the checkpoint from "
            "a model built by the current encoder, or recover the vocabulary with "
            "`uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>`."
        )

    ids = tuple(int(i) for i in ids_buffer.tolist())
    n_sae = int(n_sae_buffer.item())

    # The buffer's ORDER *is* the row mapping: compact row j means original id ids[j].
    # It must be taken LITERALLY, never re-derived.
    #
    # ``FeatureVocab.__post_init__`` canonicalises (sorts + de-duplicates), so handing
    # it an unsorted buffer would silently re-sort it — and compact row j would then be
    # reinterpreted as a DIFFERENT feature than the weights encode, with no error
    # anywhere. Reject instead. Unreachable from the production encoder (which writes
    # sorted ids), which is exactly why it is checked rather than assumed.
    if list(ids) != sorted(ids):
        raise FeatureVocabError(
            f"corrupt checkpoint: {VOCAB_IDS_KEY!r} is not sorted ascending. The buffer's "
            "order IS the embedding-row mapping (row j == original id ids[j]); a "
            "re-sorted vocabulary would silently reinterpret every row."
        )
    if len(set(ids)) != len(ids):
        raise FeatureVocabError(
            f"corrupt checkpoint: {VOCAB_IDS_KEY!r} contains duplicate feature ids."
        )

    try:
        vocab = FeatureVocab(n_sae_features=n_sae, original_ids=ids)
    except BeartypeCallHintParamViolation as exc:
        if n_sae <= 0:
            detail = f"n_sae_features must be a positive int, got {n_sae!r}"
        elif not ids:
            detail = "feature vocab is empty — the encoder would have no embedding rows"
        else:
            detail = f"feature vocab ids must be non-negative, got {ids[0]}"
        raise FeatureVocabError(f"corrupt checkpoint: {detail}") from exc
    if rows != vocab.size:
        raise FeatureVocabError(
            f"corrupt checkpoint: {FEATURE_EMBED_KEY!r} has {rows} rows but the vocab "
            f"declares {vocab.size} features."
        )

    # An identity is NOT a compaction, and a checkpoint carrying identity buffers is an
    # ANOMALY: no writer in this repo can produce one. ``FeatureVocab.from_mask`` normalizes
    # an identity to ``None``, so the model that would have registered those buffers never
    # registers them. So the file came from a writer we do not understand, and we must not
    # guess what it meant.
    #
    # This RAISES rather than normalizing. An earlier version warned, normalized to ``None``
    # and DELETED the two buffers from the caller's state dict (so the uncompacted model it
    # implies could still strict-load). That was leniency in the one place leniency is least
    # affordable: it silently reinterpreted a checkpoint from an unknown source, and it made
    # this function MUTATE a dict its callers reasonably read as a pure query.
    #
    # Runs LAST, so every integrity check above still applies first — a permuted ``range()``
    # is rejected as unsorted rather than reaching here and being called an identity.
    if vocab.is_identity:
        raise FeatureVocabError(
            f"checkpoint carries feature-vocab buffers describing the IDENTITY vocabulary "
            f"(V == n_sae == {n_sae}). That is not a compaction, and NO writer in this repo "
            f"emits it: FeatureVocab.from_mask normalizes an identity mask to None, so a "
            f"model with an all-allowing mask registers no vocab buffers at all.\n"
            f"This checkpoint therefore came from a writer we do not understand, and its "
            f"{VOCAB_IDS_KEY!r} / {VOCAB_N_SAE_KEY!r} buffers may not mean what we would "
            "assume. Guessing would either reinterpret every embedding row or silently drop "
            "information the writer intended.\n"
            "If the model genuinely has no mask, the checkpoint should carry NO vocab "
            "buffers — strip them and it will load as an ordinary uncompacted checkpoint. "
            "If it does have one, the buffers are wrong and the file is corrupt."
        )

    return n_sae, vocab


def assert_checkpoint_vocab_compatible(
    model_vocab: FeatureVocab | None,
    model_data: Mapping[str, Any],
    *,
    checkpoint_dir: str,
) -> None:
    """Pre-``load_state_dict`` guard.

    ``load_state_dict`` *does* shape-check even with ``assign=True``, so a mismatch
    already fails loudly — but with PyTorch's bare ``size mismatch``, which tells an
    operator nothing. This raises an ACTIONABLE error instead, and additionally
    catches the two cases where the shapes happen to agree but the dictionaries do
    not.

    Inherits :func:`feature_vocab_from_state_dict`'s refusal of an identity-vocab checkpoint,
    which is correct here: this runs immediately before a ``load_state_dict(..., strict=True)``
    that such a file could not survive anyway. It does NOT mutate ``model_data`` — that
    function is a pure query.
    """
    _, ckpt_vocab = feature_vocab_from_state_dict(model_data)

    if model_vocab is None and ckpt_vocab is None:
        return

    if model_vocab is None and ckpt_vocab is not None:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} is COMPACTED (vocab of {ckpt_vocab.size} of "
            f"{ckpt_vocab.n_sae_features} features) but this run's feature_mask and "
            "feature_clusters produce a full-size encoder. Configure the same mask and "
            "clusters the checkpoint was trained with — run "
            f"`uv run python -m scripts.analysis.dump_feature_mask {checkpoint_dir}` "
            "to recover it."
        )

    assert model_vocab is not None
    if ckpt_vocab is None:
        raise FeatureVocabError(
            f"checkpoint {checkpoint_dir} predates encoder feature-dictionary "
            "compaction: it carries full-size embedding tables and no vocab, but this "
            f"run resolved a feature_mask and/or feature_clusters and built a COMPACTED "
            "encoder "
            f"({model_vocab.size} of {model_vocab.n_sae_features} features). "
            "Resuming would require reshaping the checkpoint.\n"
            "START FRESH. That is the remedy, and it is the only one that works.\n"
            "Read this BEFORE reaching for the mask knobs. This checkpoint is PHYSICALLY "
            "uncompacted, so loading it genuinely does require that no feature_mask "
            "resolve — but you cannot get there by editing the config, because the SAME "
            "mask that sizes the tables also puts the bl_/wl_<16hex> leaf in the path you "
            "are resuming FROM. checkpoint_dir is composed through append_mask_leaf(), "
            "which returns the path UNCHANGED when the mask is None. So setting "
            "feature_mask_freq_min: null AND feature_mask_freq_max: null (and "
            "feature_mask: null) does NOT hand you an uncompacted resume of THIS "
            "checkpoint: it drops the leaf, points the run at a DIFFERENT directory, and "
            "it will not find this checkpoint there at all. One input decides both the "
            "table shape and the path — there is no knob that keeps the leaf while "
            "dropping compaction.\n"
            "If these exact weights are load-bearing, the checkpoint must be re-trained. "
            "(It could in principle be reshaped offline — gather the mask's allowed rows "
            "out of the full-size tables, which is exact, since the rows being dropped are "
            "provably dead — but no tool in this repo does that today, and the optimizer "
            "state would need the same treatment.)"
        )

    if model_vocab.n_sae_features != ckpt_vocab.n_sae_features:
        raise FeatureVocabError(
            f"n_sae_features mismatch on load from {checkpoint_dir}: checkpoint was "
            f"trained against a {ckpt_vocab.n_sae_features}-feature SAE, this run is "
            f"configured for {model_vocab.n_sae_features}. The bl_/wl_ checkpoint leaf "
            "hashes only (mode, ids) and cannot distinguish these — which is exactly "
            "why this is checked explicitly."
        )

    if model_vocab.original_ids != ckpt_vocab.original_ids:
        raise FeatureVocabError(
            f"feature vocabulary mismatch on load from {checkpoint_dir}: the "
            f"checkpoint's {ckpt_vocab.size} features differ from this run's "
            f"{model_vocab.size}. Resuming with a different mask or clusters would "
            "silently reinterpret every embedding row. Use the same feature_mask and "
            "feature_clusters, or start fresh. Possible causes include the same mask "
            "with different clusters, or a checkpoint written before cluster-aware "
            "compaction."
        )


def assert_resume_feature_vocab_matches(
    configured_vocab: FeatureVocab | None,
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Verify a resumed checkpoint's *metadata* does not CONTRADICT ``configured_vocab``.

    Mirrors :func:`steering.conditional.feature_mask.assert_resume_feature_mask_matches`,
    including its asymmetry: no configured vocab means there is nothing to check (the
    state-dict guard :func:`assert_checkpoint_vocab_compatible` covers that direction).

    **An ABSENT mirror is fine; only DISAGREEMENT is a corruption signal.** The persistent
    ``encoder.feature_vocab_ids`` / ``feature_vocab_n_sae`` buffers are the load-bearing
    record — a compacted checkpoint describes itself and needs no sidecar to be interpreted —
    and by the time this runs the resume path has ALREADY validated them against the
    configured vocab (``assert_checkpoint_vocab_compatible`` → ``load_state_dict(strict=True)``
    → ``assert_vocab_consistent()``). So an absent ``feature_vocab`` block leaves nothing
    unverified, and raising on it would only throw away a checkpoint that is provably
    recoverable — metas routinely get thinned, which is why ``_recover_model_config`` and
    ``_iter_ancestor_metas`` exist at all.

    This is the contract the other two readers already implement, verbatim:
    ``nanochat/checkpoint_manager.py`` ("ABSENCE of the mirror is fine; only DISAGREEMENT is a
    corruption signal. Raising on absence would make the sidecar mandatory") and
    ``dump_feature_mask._assert_meta_vocab_agrees`` ("ABSENCE is fine. Its *disagreement* is
    not."). This function used to be the odd one out, and it made the sidecar mandatory on
    every resume.
    """
    if configured_vocab is None:
        return

    if checkpoint_meta is None or META_KEY not in checkpoint_meta:
        return  # the buffers are the record, and they have already been checked

    recorded = checkpoint_meta[META_KEY]
    expected = configured_vocab.metadata_dict()
    if recorded != expected:
        raise FeatureVocabError(
            f"feature vocab mismatch on resume from {checkpoint_dir}: checkpoint "
            f"records {recorded!r}, configured {expected!r}. Resuming with a different "
            "vocabulary would silently reinterpret every encoder embedding row."
        )
