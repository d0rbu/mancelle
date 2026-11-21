"""Encoder conditioning for encoder-decoder baseline direction fitting (#1249).

A DiffMean/LinearProbe direction for an ``encdec`` target must be fit under the
feature conditioning that target is actually steered with. Before #1249 the fit
used an all-masked bag, which its own docstring described as "effectively a
decoder-only residual trajectory on the encdec backbone" -- the direction
described activations the model never produces when steered, and was then used
to counter-steer a decoder that IS conditioned.

This module owns the value passed down the capture chain. It is deliberately a
TYPE rather than a bare tensor triple so that:

* the bag it came from travels with it (``bag_digest``), and can be stamped into
  the direction artifact -- a direction fit under one bag must never be silently
  reused against another;
* the decoder-only path cannot accidentally receive one (``_run_capture_forward``
  rejects it), since a decoder-only capture has nothing to condition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch as t

from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.eval.axbench_base.types import AxBenchBaseError


@dataclass(frozen=True)
class EncoderConditioning:
    """One (concept, mode) feature bag, ready to condition a capture forward.

    ``entries`` is the bag as ``(feature_id, bucket)`` pairs, sorted, matching
    ``FeatureConditioning.entries``. It is kept as a plain tuple so this type has
    no dependency on the feature-GCG bag class -- ``axbench_base`` must not grow
    an import edge toward the adv packages (today: base -> bpb_adv is 0 imports,
    bpb_adv -> base is 24 files).
    """

    entries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            raise AxBenchBaseError(
                f"EncoderConditioning.entries must be a tuple, got {type(self.entries).__name__}"
            )
        for entry in self.entries:
            if len(entry) != 2 or any(not isinstance(v, int) for v in entry):
                raise AxBenchBaseError(
                    f"EncoderConditioning entry must be (int feature_id, int bucket), got {entry!r}"
                )
        feature_ids = [fid for fid, _ in self.entries]
        if len(set(feature_ids)) != len(feature_ids):
            raise AxBenchBaseError(
                "EncoderConditioning has duplicate feature_ids; a bag maps each "
                "feature to exactly one bucket"
            )
        if list(feature_ids) != sorted(feature_ids):
            raise AxBenchBaseError(
                "EncoderConditioning entries must be sorted by feature_id so the "
                "digest is order-independent"
            )

    @property
    def bag_digest(self) -> str:
        """Stable digest of the bag, for stamping into the direction artifact.

        Order-independent by construction (``entries`` is sorted), so two bags
        with the same content always produce the same digest regardless of how
        the caller assembled them.
        """
        payload = ";".join(f"{fid}:{bucket}" for fid, bucket in self.entries)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_tensors(self, device: t.device | str) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        """Build ``(enc_feature_ids, enc_magnitudes, enc_mask)`` for one segment.

        A capture forward scores one continuation under one bag, so there is
        exactly ONE encoder segment; ``_run_capture_forward`` pairs this with an
        all-zero ``seg_map``.
        """
        if not self.entries:
            raise AxBenchBaseError(
                "EncoderConditioning is empty; an empty bag is the unconditioned "
                "forward that #1249 removed, not a valid conditioning"
            )
        return build_segmented_feature_encoder_inputs(
            [{int(fid): int(bucket) for fid, bucket in self.entries}], device
        )
