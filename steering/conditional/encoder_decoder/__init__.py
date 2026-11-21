"""Encoder-decoder package shared exports.

Hosts cross-slice constants that would otherwise force importers to reach into
specific submodules. Slice 8 (encoder), slice 9 (cross-attention), and slice 12
(encdec_train) all need ``FP8_ALIGNMENT``; centralising it here avoids the
``encoder.py imports cross_attention.py just for the constant`` anti-pattern.
"""

# Padding alignment for FP8 ``scaled_mm`` requirement: B*n_seg*max_seg_size
# (and analogous encoder feature counts) must be divisible by 16 for FP8
# backward to satisfy CuTe alignment. Research-specific constant; see
# ``docs/reference/architecture.md`` (Cross-attention + FP8 padding sections).
FP8_ALIGNMENT: int = 16

__all__ = ["FP8_ALIGNMENT"]
