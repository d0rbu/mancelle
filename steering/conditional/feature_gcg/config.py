from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.refined_types import (
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    PositiveFiniteNumberLike,
    PositiveIntLike,
    StrictIntLike,
)

# The ONE home of the max_forward_batch default (bounds peak GPU memory of the
# sub-batched loss forwards). FeatureGcgConfig, the BPB-Adv OursGcgConfig, and
# gradient._batched_masked_losses / the class-margin objectives all read THIS
# constant, so tuning it can never leave a config default silently behind.
DEFAULT_MAX_FORWARD_BATCH = 8

# The ONE home of the max_grad_batch default, the same discipline applied to the
# GRADIENT path. That path is strictly heavier than the loss path it mirrors:
# `_batched_masked_losses` runs under `torch.no_grad()`, while
# `_compute_feature_grads_batched` must RETAIN every layer's activations for the
# backward. Before #1245 it was the one unbounded batch in the optimizer -- it
# ran a single fwd+bwd over all 2N (desired, contrast) specs -- which is why
# lowering `max_forward_batch` 8->4 did not move peak GPU memory at all: that
# knob does not reach this path. Default 4, not 8, for the same reason.
DEFAULT_MAX_GRAD_BATCH = 4

# Weight on the CONTRAST (unsteered) loss in the class margin the feature-GCG
# search minimizes: ``g_steer = loss_desired - alpha * loss_contrast``, i.e. the
# search MAXIMIZES ``alpha * unsteered - steered``. alpha = 1 is the plain
# margin -- every value and gradient is then bit-for-bit what it was before this
# knob existed, which is why 1.0 is the default and why identity payloads strip
# it (see `axbench_bpb_adv.gcg_bags`). Lowering alpha buys steering strength
# (``loss_desired`` down) at less cost to the model's unsteered behaviour
# (``loss_contrast`` no longer worth inflating), so alpha < 1 trades margin for
# a smaller base-BPB hit. alpha = 0 ignores the contrast class entirely.
DEFAULT_OBJECTIVE_ALPHA = 1.0


@beartype
@dataclass(frozen=True)
class FeatureGcgConfig:
    """Configuration for the feature-space GCG optimizer.

    Unary scalar domains are declared in the field annotations and enforced by
    :mod:`beartype`. ``__post_init__`` retains only the cross-field partition
    relationships.

    The ``universal_minibatch_size`` property is a pass-through alias for
    ``minibatch_size`` so that the generic ``run_gcg`` loop can read it via
    ``int(cfg.universal_minibatch_size)`` without modification.
    """

    max_steps: PositiveIntLike = 64
    candidate_batch_size: PositiveIntLike = 8
    top_k: PositiveIntLike = 32
    temperature: PositiveFiniteNumberLike = 1.0
    master_seed: StrictIntLike = 0
    minibatch_size: PositiveIntLike = 16
    # Max members per sub-batched loss forward in ``_batched_masked_losses`` (bounds
    # peak GPU memory: the batched objective materialises fp32 decoder logits in one
    # forward). Threaded to the objective by the feature-GCG optimizer; a caller's
    # large candidate x member batch is processed in ``max_forward_batch`` slices.
    max_forward_batch: PositiveIntLike = DEFAULT_MAX_FORWARD_BATCH
    # Max members per sub-batched fwd+bwd in ``_compute_feature_grads_batched``.
    # Memory-only, like ``max_forward_batch``: members are independent and the
    # backed-off objective is a SUM over them, so each member's gradient equals
    # its single-forward value and is unchanged by how the members are sliced.
    max_grad_batch: PositiveIntLike = DEFAULT_MAX_GRAD_BATCH
    reg_lambda: NonNegativeFiniteNumberLike = 0.0  # L1 count-penalty weight (penalty = reg_lambda * len(bag)); 0 = off, sparsity opt-in via the lambda-sweep
    # Weight on the contrast (unsteered) loss in the minimized margin
    # ``loss_desired - objective_alpha * loss_contrast``. See
    # :data:`DEFAULT_OBJECTIVE_ALPHA`. NOT a sparsity knob (that is
    # ``reg_lambda``): this one trades steering margin against the unsteered
    # base BPB. Non-negative -- a negative alpha would REWARD lowering the
    # contrast loss, inverting the class margin into a nonsense objective.
    objective_alpha: NonNegativeFiniteNumberLike = DEFAULT_OBJECTIVE_ALPHA
    # Per-partition count caps (search bounds only; max_k EXCLUSIVE ⇒ realized max = max_k - 1).
    min_k_pos: NonNegativeIntLike = 0
    max_k_pos: PositiveIntLike = 100
    min_k_neg: NonNegativeIntLike = 0
    max_k_neg: PositiveIntLike = 100

    def __post_init__(self) -> None:
        for name, min_k, max_k in (
            ("pos", self.min_k_pos, self.max_k_pos),
            ("neg", self.min_k_neg, self.max_k_neg),
        ):
            if max_k <= min_k:
                raise FeatureGcgError(
                    f"max_k_{name} ({max_k}) must be > min_k_{name} ({min_k})"
                )

    @property
    def universal_minibatch_size(self) -> int:
        """Alias for ``minibatch_size``; read by the generic ``run_gcg`` loop."""
        return self.minibatch_size
