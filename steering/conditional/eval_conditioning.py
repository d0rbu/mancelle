"""Load-time declaration of how a load conditions an encoder-decoder model at EVAL (issue #695).

A checkpoint may be COMPACTED (``model.feature_vocab is not None``: the encoder holds one embedding
row per allowed feature ``V``, not ``n_sae``, and conditioning on an SAE id outside that vocabulary
raises). Evaluating a compacted checkpoint is deferred to a follow-up PR, UNLESS that eval declares
``conditions_within_vocab`` — an explicit claim that it restricts its own feature selection to ``V``
(enforced at the eval's own selection sites, e.g. ``steering.eval.axbench_bpb.vocab_restriction``, not
here). This type is the contract that makes the deferral (a) per-eval and actionable, and (b)
impossible to forget: it is a REQUIRED argument to ``build_encdec_model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from errors import EvalConditioningError
from steering.conditional.feature_vocab import FeatureVocab


class _Stance(Enum):
    DEFERRED = auto()
    NOT_EVALUATING = auto()
    CONDITIONS_WITHIN_VOCAB = auto()


@dataclass(frozen=True)
class EvalConditioning:
    """How a load conditions the encoder at EVAL. Construct via the classmethods; never directly."""

    _stance: _Stance
    _eval_name: str | None = None

    @classmethod
    def deferred(cls, eval_name: str) -> EvalConditioning:
        """This eval is NOT yet compaction-aware. On a compacted checkpoint ``assert_loadable``
        raises ``NotImplementedError`` naming ``eval_name``; on a legacy checkpoint it is a no-op.
        Enabling the eval on compacted checkpoints is a follow-up PR."""
        if not isinstance(eval_name, str) or not eval_name:
            raise EvalConditioningError("deferred(eval_name) requires a non-empty eval name")
        return cls(_Stance.DEFERRED, eval_name)

    @classmethod
    def not_evaluating(cls) -> EvalConditioning:
        """The load performs NO encoder feature-conditioning eval — a training/resume load, or a
        decoder-only model (even at EVAL phase, e.g. plain chat, which never reaches the encoder
        build). If this stance reaches the EVAL conditioning path of an encoder-decoder build, the
        caller mis-declared and ``assert_loadable`` raises ``EvalConditioningError``."""
        return cls(_Stance.NOT_EVALUATING, None)

    @classmethod
    def conditions_within_vocab(cls, eval_name: str) -> EvalConditioning:
        """This eval restricts its OWN feature selection to the checkpoint's compacted
        vocabulary ``V`` — by construction at its selection sites, then asserted there — so,
        unlike ``deferred``, a COMPACTED checkpoint is PERMITTED to load. ``assert_loadable``
        never raises for this stance: at load time no selection has happened yet, so there is
        nothing here to verify against; the safety is downstream, at the eval's own
        feature-selection call sites (see ``steering.eval.axbench_bpb.vocab_restriction``).
        A legacy checkpoint (``feature_vocab is None``) is unaffected, exactly as for
        ``deferred``."""
        if not isinstance(eval_name, str) or not eval_name:
            raise EvalConditioningError(
                "conditions_within_vocab(eval_name) requires a non-empty eval name"
            )
        return cls(_Stance.CONDITIONS_WITHIN_VOCAB, eval_name)

    def assert_loadable(self, feature_vocab: FeatureVocab | None, *, checkpoint_dir: str) -> None:
        """Called by ``build_encdec_model`` ONLY on the EVAL path (reaching here ⇒ an encoder-decoder
        EVAL load). Raises to forbid the load; returns ``None`` to allow it."""
        if self._stance is _Stance.NOT_EVALUATING:
            raise EvalConditioningError(
                f"not_evaluating() was declared for an encoder-decoder EVAL load ({checkpoint_dir}). "
                "That stance is for training/resume or decoder-only loads; an eval must declare an "
                "eval stance (currently: EvalConditioning.deferred(<eval name>))."
            )
        if self._stance is _Stance.CONDITIONS_WITHIN_VOCAB:
            return  # permits both compacted and legacy; V-restriction is enforced downstream
        # DEFERRED
        if feature_vocab is None:
            return  # legacy / uncompacted checkpoint: nothing to defer
        raise NotImplementedError(
            f"{self._eval_name} does not yet support a COMPACTED encoder-decoder checkpoint "
            f"(feature vocabulary of {feature_vocab.size} of {feature_vocab.n_sae_features} SAE "
            f"features, from {checkpoint_dir}). Compacted-checkpoint eval support lands in a "
            f"follow-up PR right after this one. Recover this checkpoint's feature_mask with: "
            f"uv run python -m scripts.analysis.dump_feature_mask {checkpoint_dir} . Training and "
            f"resume from this checkpoint work today."
        )
