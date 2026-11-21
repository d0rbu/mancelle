"""Test-owned thresholds for cross-backend feature-equivalence acceptance.

The tight defaults mirror ``steering.validate_features``:
``feature_count_rtol=0.01``, ``jaccard_threshold=0.99``, and magnitude
``rtol=1e-2`` / ``atol=1e-5``. Production does not consume these values.
"""

FEATURE_COUNT_RTOL = 0.01
JACCARD_THRESHOLD = 0.99
MAG_RTOL = 1e-2
MAG_ATOL = 1e-5

# --- The slow Layer-4 native-vs-TL MAGNITUDE-tolerance envelope ------------- #
# These are NOT the module defaults above (MAG_RTOL/MAG_ATOL = the tight 1e-2/
# 1e-5 every legacy caller keeps). They are the documented, DATA-GROUNDED
# native-vs-TL cross-backend magnitude envelope the slow Layer-4 gate
# (test_native_integration.py) passes explicitly — the magnitude analogue of
# the structural ``_BF16_NATIVE_VS_TL_JACCARD``. Kept here so the slow gate
# and the non-vacuous unit tests share one test-owned source (no drift or
# hardcoded copy).
#
# fp32 native-vs-TL (TIGHT — NOT loosened): rtol 0.04 / atol 0.15.
#   Measured (2026-05-17, gemma-2-2b@6 + gemma-3-1b-pt@7, GemmaScope 16k,
#   residual, fp32): max rel err 0.01474, max abs 0.08516 (gemma-2 eager-vs-
#   SDPA + attn_logit_softcapping=50.0; a pre-existing TL-vs-HF property, NOT
#   a native bug). rtol 0.04 = 2.7× safety on the 1.47% max; atol 0.15 covers
#   the near-zero features where the 0.085 abs error blows up rel err.
#   matched-fp32 native-vs-TL = 0.00397, well inside this — the fp32 path is
#   fully tight and unchanged.
#
# bf16 native-vs-TL (DATA-GROUNDED, bf16-ULP root-caused): rtol 0.05 / atol
#   5.0. Measured (gemma-3-1b-pt@7 GemmaScope-2 16k residual, realistic
#   1215-token ragged corpus, 4090, 2026-05-18; seg-0 envelope over 284
#   common features): max_abs=16.0, p99(abs)=4.0, mean(abs)=0.3639, max rel
#   err ≈ 16/1328 ≈ 0.0121. Mechanism: PURE bf16 ULP rounding of the huge
#   GemmaScope-2 residual features the realistic corpus fires (bf16 relative
#   ULP = 2⁻⁷ ≈ 0.0078; top offender feat 118: native_bf16≈1312 vs
#   TL_fp32≈1328, |Δ|≈16 ≈ 1.5–2 bf16 ULP at magnitude ~1300). PROVEN not a
#   bug: native ≡ nnterp BIT-IDENTICAL at bf16 (assert_bit_identical, no
#   tolerance), nnterp-vs-TL reproduces the SAME 16, and matched-fp32
#   native-vs-TL = 0.00397. The error is ∝ feature magnitude (∝ bf16 ULP), so
#   a RELATIVE tol is the physically-correct model:
#     RTOL = 0.05 — ≈4.1× the observed max rel 0.0121 (mirrors the fp32 tol's
#       "2.7× safety" data-grounded style); the relative term 0.05·|b| is what
#       carries the HUGE features (feat 118: 0.05·1328 = 66 ≫ 16).
#     ATOL = 5.0 — MINIMAL, deliberately NOT vacuous: ≈1.25× the measured
#       p99(abs)=4.0 small-feature absolute floor. ``t.allclose`` is the
#       ELEMENT-WISE ∀i |aᵢ-bᵢ| ≤ atol+rtol·|bᵢ|; a SMALL-|b| common feature
#       (JumpReLU near-threshold / sub-p99, where rtol·|b| is negligible) —
#       NOT the global-max_abs feat 118 — is what tripped the fp32 0.04/0.15
#       (feat 118 alone satisfies 16 ≤ 0.15+0.04·1328=53). atol covers that
#       small-feature floor. Crucially 5.0 ≪ max_abs=16 (0.31×): a blanket
#       "atol ≥ 16" would be VACUOUS (every feature passes on atol alone), so
#       the relative term MUST do the work for the huge features and a real
#       bug shifting a huge feature by >0.05·|b| (or ANY feature by >5.0
#       absolute) is still caught. Asymmetric oracle: ONLY the weaker bf16
#       native-vs-TL oracle (kernel + precision boundary) is loosened; the
#       strong native≡nnterp anchor stays bit-identical at bf16.
FP32_NATIVE_VS_TL_MAG_RTOL = 0.04  # TIGHT, unchanged
FP32_NATIVE_VS_TL_MAG_ATOL = 0.15  # TIGHT, unchanged
BF16_NATIVE_VS_TL_MAG_RTOL = 0.05  # data-grounded bf16 ULP envelope
BF16_NATIVE_VS_TL_MAG_ATOL = 5.0  # data-grounded small-feature floor
