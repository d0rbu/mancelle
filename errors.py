"""Custom exception hierarchy for nano-gpt-pretrain-steer.

This module lives at the project root so all layers (tasks/, nanochat/,
steering/, scripts/) can import it without circular dependencies.
"""


class NanoGPTError(Exception):
    """Base exception for the project."""


class DataLoadError(NanoGPTError):
    """Data file missing, corrupted, or has unexpected schema."""


class ExtractionError(NanoGPTError):
    """Feature extraction produced invalid/incomplete results."""


class BosOnlySegmentError(ExtractionError):
    """A segment contained only <bos> tokens, so ignore_bos_features removed
    every poolable position. Typically segment_size=1 on pretraining, or a
    degenerate single-<bos> document. Raised (not asserted) so it survives
    python -O, consistent with the typed CSR-invariant raises."""


class EmptyExtractionSegmentError(ExtractionError):
    """A base-extracted shard emitted a segment with zero active features
    (offsets[i] == offsets[i+1]). Per issue #522, if extraction produced a
    segment record it must carry >=1 active extracted feature; an empty
    segment indicates a pooling/threshold/serialization or source-alignment
    problem. Raised (not asserted) so it survives python -O, consistent with
    the other typed CSR-invariant raises."""


class CompileGateError(ExtractionError):
    """The mandatory torch.compile gate failed: the compiled forward degrades
    vs the fp32-eager (ideal-math) reference more than bf16-eager itself does
    (beyond the SLACK band), or the gate could not build its ideal-math
    reference; refusing to emit potentially-corrupted activations and refusing
    to silently fall back to eager (the gate is mandatory, no opt-out)."""


class RowPackGateError(ExtractionError):
    """The mandatory S2 row-packing gate failed: the packed forward degrades
    vs the fp32-eager (ideal-math) reference more than the unpacked
    model_dtype path itself does (beyond the SLACK band), or the gate could
    not build its ideal-math reference. Packing is always-on with no unpacked
    production fallback, so a gate-fail HARD-RAISES rather than silently
    emitting potentially row-packing-corrupted (e.g. cross-document-leaked)
    activations (the gate is mandatory, no opt-out)."""


class GradientAttributionError(ExtractionError):
    """Gradient-attribution producer/replay/gate failure: re-tokenization
    drift vs base shards, packed-replay verification mismatch, the
    decoder-linearity gate failed, an unsupported SAE configuration
    (architecture/normalization/scaling/hook site), incompatible base shards
    (context-independent), or a missing/mismatched grad overlay or provenance
    artifact at consume time. Raised (not asserted) so it survives python -O,
    consistent with the other typed extraction raises."""


class PackedRowInvariantError(ExtractionError):
    """Packed-row producer, replay, or artifact-identity invariant failure."""


class GradientRankingUnsupportedError(NanoGPTError):
    """Gradient-attribution ranking was configured on a code path that
    performs LIVE feature selection (live dataloader, SFT, self-conditioned
    generation, eval-side selection). Gradient ranking is offline-only:
    precompute sel/ shards via scripts.preprocess.precompute_selected_features
    (after running precompute_gradient_attribution; packed rows carry their
    segment provenance natively). Mirrors the feature_delta fail-fast
    precedent in the opposite direction."""


class AcceleratedBackendError(NanoGPTError):
    """An accelerated backend (native/nnterp) was required but unavailable or
    failed, and silent TransformerLens fallback is disallowed. Fatal: raised
    so the run aborts with a non-zero exit instead of silently degrading to
    (often CPU-resident, ~100x slower, numerically divergent) TransformerLens.
    Use ``backend=transformerlens`` to run TransformerLens intentionally."""


class ModelWeightLoadError(NanoGPTError):
    """A model load left weights missing / unexpected / mismatched (silently
    random-initialised). FATAL and deliberately NOT an ExtractionError subclass,
    so it is never swallowed by the backend-fallback chain. See issue #549."""


class DegenerateFeatureStatisticsError(NanoGPTError):
    """Feature statistics show the degenerate signature of a random-weight /
    mis-loaded model (a fixed feature set fires in ~every segment). Fatal at the
    aggregation stage as a second gate behind the load guard. See issue #549."""


class ValidationError(NanoGPTError):
    """An invariant or postcondition was violated."""


class ParseError(NanoGPTError):
    """Failed to parse expected format (judge response, YAML, regex, etc.)."""


class FeatureFilterError(NanoGPTError):
    """Base for SAE feature interpretability pipeline errors."""


class FeatureMaskError(NanoGPTError):
    """Invalid feature blacklist/whitelist config, file, or resolved set."""


class FeatureVocabError(NanoGPTError):
    """Invalid encoder feature vocabulary, or a feature id outside it.

    Deliberately a SIBLING of ``FeatureMaskError``, not a subclass: an
    out-of-vocabulary id (the encoder has no embedding row for it) and an invalid
    mask (the config is wrong) are different failures, and an
    ``except FeatureMaskError`` handler must not silently swallow the former.
    """


class SelfCondTrainingError(NanoGPTError):
    """Invalid self-conditioning post-training config, state, or operation."""


class FeatureClustersError(NanoGPTError):
    """Invalid static feature-cluster input, identity, or consolidation state.

    This is deliberately separate from both ``FeatureMaskError`` and the
    ``RemovedFeatureError`` used for the legacy clustering pipeline. Static
    preselection consolidation is a supported, narrow consumer; malformed
    cluster files and stale representative-only artifacts are correctness
    failures rather than requests for the removed producer/runtime system.
    """


class RemovedFeatureError(NanoGPTError):
    """A removed feature was requested by current config or legacy metadata."""


class AutoInterpFilterError(NanoGPTError):
    """Base for local intruder-detection autointerp filtering errors."""


class ManualFeatureSelectionError(NanoGPTError):
    """Invalid manual feature-selection input (parse, duplicate, or out-of-range)."""


class EvalConditioningError(NanoGPTError):
    """A caller declared an EvalConditioning stance that is invalid for the load it performed
    (a programmer/contract error, distinct from operator-facing vocab/mask errors)."""
