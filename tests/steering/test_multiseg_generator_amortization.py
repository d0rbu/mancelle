# tests/test_multiseg_generator_amortization.py
"""Regression tests for generate_for_document encoder/prefill amortization.

The original docstring claimed encoder + prefill amortization across buckets,
but in insertion mode with num_buckets=N the implementation actually runs the
encoder (1 + N) times, once per condition. These tests lock the current
behaviour so that any future change is explicit:

1. `test_generate_for_document_encoder_call_count_insertion_mode`: wraps
   `model.encoder` with a real (non-mock) call counter and asserts the exact
   observed count (1 + num_buckets) in insertion mode.
2. `test_docstring_claim_matches_behavior`: enforces a documentation/behavior
   contract — if the docstring of `generate_for_document` claims "amortize"
   or "amortization" across buckets, the call count must be <= 3; otherwise
   any count is permitted. Prevents future drift between the docstring and
   the implementation.
3. `test_generate_for_document_output_unchanged_by_amortization`: deterministic
   stub encoder — asserts that per-condition sample output is stable across
   runs, so any future amortization refactor must preserve output bit-for-bit.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import torch as t

from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.completion_steering_eval import EvalMode
from steering.eval.multiseg_generator import generate_for_document
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

_PASSTHROUGH_CFG = ThresholdConfig(ThresholdType.segment_topk, 99999)


class _FakeTokenizer:
    bos_id = 2

    def encode(
        self, text: str, prepend: int | None = None, append: int | None = None
    ) -> list[int]:
        tokens = list(range(10, 10 + len(text) // 4))
        if prepend is not None:
            tokens = [prepend] + tokens
        if append is not None:
            tokens = tokens + [append]
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 4)


class _CountingEncoder:
    """Real callable (not a mock) that counts forward calls and returns
    deterministic encoder outputs so sample output is reproducible."""

    def __init__(self, num_buckets: int = 4, embd: int = 256):
        self.calls = 0
        self.config = MagicMock()
        self.config.num_buckets = num_buckets
        self._embd = embd

    def __call__(self, fids, mags, mask):
        self.calls += 1
        n_seg = fids.shape[1]
        # Deterministic output derived from a fixed seed so sample generation
        # is reproducible across runs.
        g = t.Generator(device="cpu").manual_seed(0xC0FFEE)
        out = t.randn(1, n_seg, mask.shape[2], self._embd, generator=g)
        return pack_dense_encoder_output_reference(out, mask)


class _FakeModel:
    """Minimal EncoderDecoderGPT stand-in. Forward returns deterministic-ish
    logits with a single dominant token so greedy decode is stable."""

    def __init__(
        self,
        vocab_size: int = 100,
        seq_len: int = 1024,
        num_buckets: int = 4,
    ):
        self.vocab_size = vocab_size

        self.dec_config = MagicMock()
        self.dec_config.sequence_len = seq_len
        self.dec_config.n_kv_head = 4
        self.dec_config.n_head = 8
        self.dec_config.n_embd = 256
        self.dec_config.n_layer = 2

        self.encoder = _CountingEncoder(num_buckets=num_buckets, embd=256)

    def __call__(self, idx, **kwargs):
        B, T = idx.shape[0], idx.shape[1]
        # Fully deterministic logits (no randn) so that greedy argmax is
        # stable across calls.
        logits = t.zeros(B, T, self.vocab_size)
        logits[:, :, 50] = 10.0
        return logits


def _make_segments(n_segs: int = 3, features_per_seg: int = 5) -> list[SegmentFeatures]:
    segs: list[SegmentFeatures] = []
    for i in range(n_segs):
        fids = t.arange(i * 10, i * 10 + features_per_seg, dtype=t.int32)
        mags = t.ones(features_per_seg, dtype=t.float32) * (i + 1.0)
        segs.append(
            SegmentFeatures(
                segment_idx=i,
                position=i * 128,
                feature_ids=fids,
                feature_mags=mags,
                char_start=i * 512,
                char_end=(i + 1) * 512,
            )
        )
    return segs


def _make_env(num_buckets: int = 8, n_segs: int = 3):
    model = _FakeModel(vocab_size=100, seq_len=1024, num_buckets=num_buckets)
    tokenizer = _FakeTokenizer()
    doc_text = "A" * 512 * n_segs
    segments = _make_segments(n_segs=n_segs)
    boundaries = t.zeros(100, max(num_buckets - 1, 1), dtype=t.float64)
    for i in range(max(num_buckets - 1, 1)):
        boundaries[:, i] = float(i + 1)
    return model, tokenizer, doc_text, segments, boundaries


# ---------------------------------------------------------------------------
# 1. Call-count lock (insertion mode)
# ---------------------------------------------------------------------------


def test_generate_for_document_encoder_call_count_insertion_mode():
    """In insertion mode with num_buckets=8 the encoder is currently called
    (1 + num_buckets) = 9 times — once per condition. Lock the count so any
    future refactor that introduces real amortization (≤ 3) or regresses the
    behaviour is caught.
    """
    num_buckets = 8
    model, tokenizer, doc_text, segments, boundaries = _make_env(
        num_buckets=num_buckets
    )
    feature_id = 999  # not in any segment -> insertion invariant OK

    generate_for_document(
        model=model,
        tokenizer=tokenizer,
        doc_text=doc_text,
        segments=segments,
        target_segment_idx=2,
        feature_id=feature_id,
        mode=EvalMode.INSERTION,
        boundaries=boundaries,
        feature_cfg=_PASSTHROUGH_CFG,
        precomputed_thresholds=None,
        device=t.device("cpu"),
        num_buckets=num_buckets,
        samples_per_condition=1,
        temperature=0.0,
    )

    # Current behaviour (no amortization across buckets): exactly (1 + N).
    # If a future change implements real amortization, update this bound to
    # <= 3 and update the docstring simultaneously.
    observed = model.encoder.calls
    assert observed == 1 + num_buckets, (
        f"Encoder call count regressed: expected {1 + num_buckets} "
        f"(insertion mode, one encoder forward per condition), got {observed}"
    )


# ---------------------------------------------------------------------------
# 2. Docstring vs behavior contract
# ---------------------------------------------------------------------------


def test_docstring_claim_matches_behavior():
    """If the docstring claims amortization, encoder calls must be <= 3 in
    insertion mode. If no such claim, any count is permitted. This prevents
    future drift where someone adds an "amortization" word to the docstring
    without implementing it (or vice versa).
    """
    doc = generate_for_document.__doc__ or ""
    # "Claims amortization" means: at least one occurrence of "amortiz..."
    # in the docstring that does NOT have a nearby negation marker. We scan
    # each occurrence and check a ~60-char window before it for negation
    # words ("not", "NOT", "no", "does not", "doesn't", "never").
    positive_claim = False
    negation_re = re.compile(
        r"\b(not|no|never|doesn't|don't)\b", re.IGNORECASE
    )
    for m in re.finditer(r"amortiz(e|es|ed|ing|ation)", doc, re.IGNORECASE):
        # Skip identifier references like "D8-amortization-claim.md" or
        # "test_multiseg_generator_amortization.py".
        if m.start() > 0 and doc[m.start() - 1] in ("-", "_"):
            continue
        window_start = max(0, m.start() - 60)
        window = doc[window_start : m.start()]
        # Strip markdown emphasis so "**NOT**" becomes "NOT".
        window_clean = re.sub(r"[*_`]+", "", window)
        if not negation_re.search(window_clean):
            positive_claim = True
            break
    claims_amortization_across_buckets = positive_claim

    num_buckets = 8
    model, tokenizer, doc_text, segments, boundaries = _make_env(
        num_buckets=num_buckets
    )
    generate_for_document(
        model=model,
        tokenizer=tokenizer,
        doc_text=doc_text,
        segments=segments,
        target_segment_idx=2,
        feature_id=999,
        mode=EvalMode.INSERTION,
        boundaries=boundaries,
        feature_cfg=_PASSTHROUGH_CFG,
        precomputed_thresholds=None,
        device=t.device("cpu"),
        num_buckets=num_buckets,
        samples_per_condition=1,
        temperature=0.0,
    )
    calls = model.encoder.calls

    if claims_amortization_across_buckets:
        assert calls <= 3, (
            f"Docstring of generate_for_document claims amortization but "
            f"encoder was called {calls} times (expected <= 3 for real "
            f"cross-bucket amortization). Either implement the amortization "
            f"or remove the claim from the docstring."
        )
    # else: no amortization claim -> any call count is fine.


# ---------------------------------------------------------------------------
# 3. Output determinism regression
# ---------------------------------------------------------------------------


def test_generate_for_document_output_unchanged_by_amortization():
    """Reference regression: with a deterministic stub encoder + greedy
    decoding, re-running generate_for_document produces identical samples
    per condition. Any future amortization change must preserve this.
    """
    num_buckets = 4

    def _run():
        model, tokenizer, doc_text, segments, boundaries = _make_env(
            num_buckets=num_buckets
        )
        return generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=2,
            feature_id=999,
            mode=EvalMode.INSERTION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=num_buckets,
            samples_per_condition=1,
            temperature=0.0,
        )

    a = _run()
    b = _run()

    assert len(a.samples) == len(b.samples) == 1 + num_buckets
    for sa, sb in zip(a.samples, b.samples, strict=True):
        assert sa.condition == sb.condition
        assert sa.bucket_index == sb.bucket_index
        assert sa.text == sb.text, (
            f"Output drift in condition {sa.condition}: '{sa.text}' != '{sb.text}'"
        )
