"""Integration tests for full-sequence conditioned encoder-decoder SFT.

Covers what was previously a thin coverage gap: the conditioned SFT path is
tested piecewise (shape checks, cross-tokenizer segments, multi-turn loss
supervision) but nothing ran an end-to-end loop with synthetic features *and*
proved that gradients actually flow through the encoder and cross-attention.

This file fills that gap across several layers:

* **Unit:** helpers that build synthetic tasks, tokenisers, and features
  deterministically, plus a context manager that monkey-patches the disk
  loader used by ``make_sft_data_generator``.
* **Functional:** ``make_sft_data_generator`` with full-sequence features —
  tensor shapes, ``seg_map`` contents, multi-turn full-sequence supervision.
* **Integration (tiny CPU):** forward + backward through a tiny
  ``EncoderDecoderGPT`` built via ``tests.encdec_test_helpers.build_tiny_model``,
  asserting that encoder + cross-attention parameters receive non-zero
  gradients.
* **Missing-feature invariants:** rows missing from ``sft_features`` are never
  trained with empty routing; any missing or truncated feature row raises with
  enough context to rerun extraction.
* **Contiguity invariants:** ``inputs`` and ``targets`` must always be
  contiguous regardless of device or dtype so that ``F.cross_entropy`` can
  call ``.view(-1)`` on them.
* **Slow GPU (``@pytest.mark.slow``):** real nanochat tokenizer + real
  ``EncoderDecoderGPT`` forward-backward exercising flash-attn-4 / fp8 paths.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Protocol, cast
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
import scripts.train.conditional_encdec_sft as conditional_encdec_sft
from nanochat.flash_attention import FlashAttnImpl
from scripts.preprocess.convert_sft_to_parquet import prepare_messages_for_full_nanochat_render
from scripts.train.conditional_encdec_sft import (
    SFTState,
    make_sft_data_generator,
)
from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer
from steering.conditional.encoder_decoder.full_sequence_routing import (
    source_aligned_char_offsets,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft.mixture_fingerprint import MixtureFingerprint
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_tiny_model,
    setup_optimizer_simple,
)
from tests.encdec_test_helpers import (
    VOCAB_SIZE as TINY_VOCAB_SIZE,
)

# ---------------------------------------------------------------------------
# Attention backend override
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    """Force SDPA attention for every test in this module.

    ``flex_attention`` does not support backward on CPU — the Triton backend
    emits an Inductor ``IndexError`` inside ``_compiled_flex``.  We patch
    ``nanochat.flash_attention._override_impl`` directly (rather than going
    through the ``NANOCHAT_ATTN_IMPL`` env var) because the module may have
    already been imported and cached by earlier tests in the suite.

    Mirrors the fixture in ``tests/test_encdec_gradient_flow.py``.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _tokenizer_available() -> bool:
    """Check whether the nanochat tokenizer pickle is on disk.

    Used to gate slow tests that need the real tokenizer rather than the
    ``SFTMockTokenizer`` defined in this file.  Worktrees may not have the
    pickle downloaded, so we skip gracefully instead of failing hard.
    """
    from pathlib import Path

    from nanochat.common import get_base_dir

    return (Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl").exists()


# ---------------------------------------------------------------------------
# Mock tokenizer and synthetic tasks
# ---------------------------------------------------------------------------


class SFTMockTokenizer:
    """Deterministic char-level tokenizer with ``render_conversation``.

    Vocab is sized to ``TINY_VOCAB_SIZE`` (matching
    ``tests/encdec_test_helpers.build_tiny_model``) so the rendered token ids
    always index into the tiny model's embedding without overflow.  Every
    character encodes to a distinct token id, so ``decode`` is effectively a
    1-to-1 inverse and ``token_to_char_offsets`` yields one offset per token.

    Conversation rendering for the *single*-turn case:

    * ``ids[0]`` is the BOS marker (id 0).
    * The next ``user_len`` tokens encode the user message (mask = 0).
    * The next ``asst_len + 1`` tokens are the assistant turn: one
      assistant-start marker followed by the assistant content. The returned
      mask marks those tokens with 1 for nanochat-template compatibility, but
      full-sequence SFT trains all non-padding tokens.

    Multi-turn rendering repeats ``user_len + 1`` unmasked tokens and
    ``asst_len + 1`` masked tokens per turn.
    """

    vocab_size: int = TINY_VOCAB_SIZE
    _next_token_id: int = 2  # 0=BOS, 1=reserved; content starts at 2

    def __init__(self) -> None:
        self._next_token_id = SFTMockTokenizer._next_token_id

    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(max(32, i)) for i in ids)

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        """Render ``conversation["messages"]`` into ``(ids, mask)``.

        Each user message contributes ``user_len + 1`` unmasked tokens (the
        +1 represents the role marker).  Each assistant message contributes
        ``asst_len + 1`` masked tokens (role marker + content).  The
        conversation always starts with a BOS token and the first user turn's
        role marker is folded into the leading unmasked run.
        """
        ids: list[int] = [self.get_bos_token_id()]
        mask: list[int] = [0]
        next_tok = self._next_token_id
        for msg in conversation["messages"]:
            content = msg["content"]
            role_marker = next_tok
            next_tok = (next_tok + 1) % self.vocab_size or 2
            content_tokens: list[int] = []
            for _ in content:
                content_tokens.append(next_tok)
                next_tok = (next_tok + 1) % self.vocab_size or 2
            run_tokens = [role_marker, *content_tokens]
            ids.extend(run_tokens)
            mark = 1 if msg["role"] == "assistant" else 0
            mask.extend([mark] * len(run_tokens))
        return ids, mask


class _SyntheticFeatureTokenizer(Protocol):
    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]: ...

    def decode(self, ids: list[int]) -> str: ...


class SingleTurnTask(Task):
    """Minimal single-turn task producing deterministic user/assistant pairs."""

    def __init__(
        self,
        n: int,
        user_chars: int = 4,
        asst_chars: int = 8,
        prefix: str = "s",
        **task_kwargs,
    ) -> None:
        super().__init__(**task_kwargs)
        self._n = n
        self._user_chars = user_chars
        self._asst_chars = asst_chars
        self._prefix = prefix

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        user = f"u{self._prefix}{index:03d}"[: self._user_chars].ljust(self._user_chars, ".")
        asst = f"a{self._prefix}{index:03d}"[: self._asst_chars].ljust(self._asst_chars, ".")
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ]
        }


class MultiTurnTask(Task):
    """Multi-turn task whose entire rendered sequence is conditioned."""

    def __init__(
        self,
        n: int,
        user_chars: int = 3,
        asst_chars: int = 6,
        **task_kwargs,
    ) -> None:
        super().__init__(**task_kwargs)
        self._n = n
        self._user_chars = user_chars
        self._asst_chars = asst_chars

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        u1 = f"q{index:03d}"[: self._user_chars].ljust(self._user_chars, ".")
        a1 = f"p{index:03d}"[: self._asst_chars].ljust(self._asst_chars, ".")
        u2 = f"r{index:03d}"[: self._user_chars].ljust(self._user_chars, ".")
        a2 = f"x{index:03d}"[: self._asst_chars].ljust(self._asst_chars, ".")
        return {
            "messages": [
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ]
        }


class TrailingUserTask(Task):
    """Multi-turn task whose conversations END on an unanswered user turn.

    SmolTalk rows legitimately look like this, and the converter truncates
    them after the last assistant message.  No other fixture in this file
    exercises that shape, which is why issue #1465 survived to production.
    """

    def __init__(
        self,
        n: int,
        user_chars: int = 3,
        asst_chars: int = 6,
        tail_chars: int = 9,
        **task_kwargs,
    ) -> None:
        super().__init__(**task_kwargs)
        self._n = n
        self._user_chars = user_chars
        self._asst_chars = asst_chars
        self._tail_chars = tail_chars

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        u1 = f"q{index:03d}"[: self._user_chars].ljust(self._user_chars, ".")
        a1 = f"p{index:03d}"[: self._asst_chars].ljust(self._asst_chars, ".")
        u2 = f"thanks{index:03d}"[: self._tail_chars].ljust(self._tail_chars, ".")
        return {
            "messages": [
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
            ]
        }


# ---------------------------------------------------------------------------
# Synthetic feature construction
# ---------------------------------------------------------------------------


def _full_sequence_char_count(tokenizer: _SyntheticFeatureTokenizer, conversation: dict) -> int:
    """Return the source-coordinate character length of the CONVERTED sample.

    Feature extraction segments the parquet ``text`` column, which
    ``convert_sft_to_parquet`` renders only up to the last assistant turn.
    Measuring the untrimmed conversation here would fabricate coverage the
    real pipeline never produces and make trailing-user tests vacuous
    (issue #1465).
    """
    messages, skip_reason = prepare_messages_for_full_nanochat_render(conversation)
    assert messages is not None, (
        f"fixture conversation is one the converter would skip ({skip_reason}); "
        "pass its index in build_synthetic_features(skip_indices=...) instead"
    )
    ids, _mask = tokenizer.render_conversation({"messages": messages})
    return source_aligned_char_offsets(tokenizer, ids)[-1]


def build_synthetic_features(
    task: Task,
    tokenizer: _SyntheticFeatureTokenizer,
    *,
    n_segments_per_conv: int = 1,
    features_per_segment: int = 3,
    n_sae_features: int = N_SAE_FEATURES,
    skip_indices: frozenset[int] = frozenset(),
) -> dict[int, list[SegmentFeatures]]:
    """Build a deterministic ``{example_idx: [SegmentFeatures, ...]}`` dict.

    For each conversation, the full rendered sequence is partitioned into
    ``n_segments_per_conv`` equal slices along the feature-source character
    axis, and each slice is populated with a unique set of
    ``features_per_segment`` feature ids and monotonically increasing
    magnitudes. Indices listed in ``skip_indices`` are deliberately omitted so
    coverage tests can assert both the exclusion and hard-fail paths.
    """
    assert n_segments_per_conv >= 1
    assert features_per_segment >= 1
    assert features_per_segment * n_segments_per_conv <= n_sae_features, (
        "Need enough SAE features to give each segment a distinct feature set"
    )
    out: dict[int, list[SegmentFeatures]] = {}
    feature_cursor = 0
    for idx in range(len(task)):
        if idx in skip_indices:
            continue
        conv = task[idx]
        total_chars = _full_sequence_char_count(tokenizer, conv)
        assert total_chars >= n_segments_per_conv, (
            f"Need at least {n_segments_per_conv} chars in the rendered sample "
            f"to build that many segments (got {total_chars})"
        )
        seg_list: list[SegmentFeatures] = []
        seg_size = total_chars // n_segments_per_conv
        for seg_i in range(n_segments_per_conv):
            char_start = seg_i * seg_size
            char_end = total_chars if seg_i == n_segments_per_conv - 1 else (seg_i + 1) * seg_size
            fids = t.tensor(
                [(feature_cursor + j) % n_sae_features for j in range(features_per_segment)],
                dtype=t.int32,
            )
            # Magnitudes are descending so segment_topk keeps the ordering stable.
            fmags = t.tensor(
                [float(features_per_segment - j) for j in range(features_per_segment)],
                dtype=t.float32,
            )
            feature_cursor = (feature_cursor + features_per_segment) % n_sae_features
            seg_list.append(
                SegmentFeatures(
                    segment_idx=seg_i,
                    position=seg_i,
                    feature_ids=fids,
                    feature_mags=fmags,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        out[idx] = seg_list
    return out


def build_synthetic_sft_store(
    features: dict[int, list[SegmentFeatures]],
    *,
    split: SFTSplit | None = None,
    mixture: str | None = None,
    parquet_fingerprint: MixtureFingerprint | None = None,
    feature_fingerprint: MixtureFingerprint | None = None,
    texts: dict[int, str] | None = None,
    conversion_total_conversations: int | None = None,
    parquet_original_indices: Iterable[int] | None = None,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> dict:
    """Build the synthetic feature store ``patch_load_sft_features`` hands out.

    Exposed separately so a caller that installs its own ``load_sft_features``
    replacement (for example a train/val dispatcher) gets the same store shape
    without entering — and having to hold open — the patch context.

    ``parquet_original_indices`` is the reference document set the real store
    reads from the parquet's ``original_index`` column. It defaults to the
    synthetic feature keys (the aligned case); pass it explicitly to build a
    store where features and parquet disagree in either direction.
    """

    coverage = conversion_total_conversations
    reference_indices = frozenset(
        features if parquet_original_indices is None else parquet_original_indices
    )

    class _SyntheticSFTFeatureStore(dict):
        segment_mode = SegmentMode.fixed_tokens
        conversion_split = split
        conversion_mixture = mixture
        conversion_total_conversations = coverage
        parquet_original_indices = reference_indices
        parquet_mixture_fingerprint = parquet_fingerprint
        feature_mixture_fingerprint = feature_fingerprint

        def load_texts(self, original_indices: list[int]) -> dict[int, str]:
            if texts is None:
                raise RuntimeError("synthetic SFT store has no probe texts")
            return {index: texts[index] for index in original_indices}

    store = _SyntheticSFTFeatureStore(features)
    store.segment_mode = segment_mode
    return store


@contextmanager
def patch_load_sft_features(
    features: dict[int, list[SegmentFeatures]],
    *,
    split: SFTSplit | None = None,
    mixture: str | None = None,
    parquet_fingerprint: MixtureFingerprint | None = None,
    feature_fingerprint: MixtureFingerprint | None = None,
    texts: dict[int, str] | None = None,
    enforce_fingerprints: bool = False,
    conversion_total_conversations: int | None = None,
    parquet_original_indices: Iterable[int] | None = None,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> Iterator[dict]:
    """Replace ``scripts.train.conditional_encdec_sft.load_sft_features`` for a block.

    The SFT script imports ``load_sft_features`` at module level, so we
    monkey-patch the module attribute rather than
    ``steering.conditional.encoder_decoder.sft_features.load_sft_features``
    (which would miss the already-imported reference).
    """

    store = build_synthetic_sft_store(
        features,
        split=split,
        mixture=mixture,
        parquet_fingerprint=parquet_fingerprint,
        feature_fingerprint=feature_fingerprint,
        texts=texts,
        conversion_total_conversations=conversion_total_conversations,
        parquet_original_indices=parquet_original_indices,
        segment_mode=segment_mode,
    )
    fingerprint_guard = (
        nullcontext()
        if enforce_fingerprints
        else patch(
            "scripts.train.conditional_encdec_sft._assert_sft_mixture_fingerprint_matches_task",
            return_value=None,
        )
    )
    with (
        patch(
            "scripts.train.conditional_encdec_sft.load_sft_features",
            return_value=store,
        ),
        fingerprint_guard,
    ):
        yield store


def make_lazy_patched_sft_generator[T](
    features: dict[int, list[SegmentFeatures]],
    factory: Callable[[], Iterator[T]],
) -> Iterator[T]:
    """Build an SFT generator whose feature patch cannot leak across yields.

    ``make_sft_data_generator`` is lazy and performs its loader lookup while
    producing its first batch.  Wrapping it as ``with patch(...): yield from``
    therefore suspends the patch for the entire lifetime of the returned
    generator.  This helper keeps the outer generator lazy (construction does
    not advance state), advances the inner generator once under the patch,
    and releases the patch before yielding that first batch.  Every subsequent
    batch is already outside the patch as well.

    The first ``next`` is intentionally performed inside the outer generator,
    rather than while this function is called.  SFT tests rely on
    ``SFTState.consumed`` remaining unchanged until the caller consumes a
    batch.
    """

    def _outer() -> Iterator[T]:
        with patch_load_sft_features(features):
            inner = factory()
            try:
                first = next(inner)
            except StopIteration:
                return
        yield first
        yield from inner

    return _outer()


def _make_boundaries(
    n_sae_features: int = N_SAE_FEATURES,
    num_buckets: int = NUM_BUCKETS,
) -> t.Tensor:
    """Build a ``(n_sae_features, num_buckets - 1)`` per-feature boundary table.

    ``discretize_activations`` indexes ``boundaries[0]`` as the SAE feature
    dimension, so a 1D tensor would implicitly cap the feature vocabulary at
    its length.  We give every feature the same ``num_buckets - 1`` linearly
    spaced thresholds so the test is agnostic to which feature ids the
    synthetic builder emits.
    """
    per_feature = t.linspace(0.5, float(num_buckets) - 0.5, num_buckets - 1)
    return per_feature.unsqueeze(0).expand(n_sae_features, -1).contiguous()


def _make_gen(
    task: Task,
    tokenizer: SFTMockTokenizer,
    *,
    B: int = 2,
    T: int = 64,
    num_buckets: int = NUM_BUCKETS,
    threshold: int = 1000,
    device: str = "cpu",
    sft_state: SFTState | None = None,
    loss_mask_mode: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
    alignment_probe_samples: int,
    mixture_name: str = "capabilities",
    expected_split: SFTSplit = SFTSplit.TRAIN,
    feature_dir: str = "/dummy/features",
    parquet_dir: str = "/dummy/parquet",
    feature_source_bos_char_len: int = 0,
    single_pass: bool = False,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    buffer_size: int = 100,
    grad_accum_steps: int = 1,
    min_loss_tokens: int = 1,
    profile_markers: bool = False,
    return_seg_doc_id: bool = False,
):
    """Build a conditioned ``make_sft_data_generator`` for tests.

    Centralising the boilerplate so individual tests can read as behavior
    assertions.  The feature pipeline is a permissive ``segment_topk`` with a
    threshold large enough to keep every synthetic feature.
    """
    threshold_cfg = ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=threshold,
    )
    boundaries = _make_boundaries(num_buckets=num_buckets)
    return make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device=device,
        sft_state=sft_state or SFTState(),
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        buffer_size=buffer_size,
        grad_accum_steps=grad_accum_steps,
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        threshold_cfg=threshold_cfg,
        boundaries=boundaries,
        min_loss_tokens=min_loss_tokens,
        loss_mask_mode=loss_mask_mode,
        alignment_probe_samples=alignment_probe_samples,
        mixture_name=mixture_name,
        expected_split=expected_split,
        feature_source_bos_char_len=feature_source_bos_char_len,
        single_pass=single_pass,
        profile_markers=profile_markers,
        return_seg_doc_id=return_seg_doc_id,
    )


# ---------------------------------------------------------------------------
# Functional: shapes and seg_map contents
# ---------------------------------------------------------------------------


class TestConditionedSftShapes:
    """``make_sft_data_generator`` produces the right conditioned tensors."""

    def test_shape_contract(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        B, T = 2, 64
        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=B, T=T)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = next(gen)

        assert inputs.shape == (B, T), f"inputs: {inputs.shape}"
        assert targets.shape == (B, T), f"targets: {targets.shape}"
        assert seg_map.shape == (B, T), f"seg_map: {seg_map.shape}"
        assert enc_fids.ndim == 3 and enc_fids.shape[0] == B, f"enc_fids: {enc_fids.shape}"
        assert enc_mags.shape == enc_fids.shape
        assert enc_mask.shape == enc_fids.shape

    def test_completion_mask_runs_after_padding_mask(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The completion helper must receive targets with padding already ignored."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=1, user_chars=3, asst_chars=4)
        features = build_synthetic_features(task, tokenizer)
        ids, _ = tokenizer.render_conversation(task[0])
        padding_start = len(ids) - 1
        helper_entered = False
        real_apply = conditional_encdec_sft.apply_completion_only_row_masks

        def spy(targets: t.Tensor, row_masks: list[list[int]]) -> None:
            nonlocal helper_entered
            helper_entered = True
            assert (targets[0, padding_start:] == -1).all()
            real_apply(targets, row_masks)

        monkeypatch.setattr(conditional_encdec_sft, "apply_completion_only_row_masks", spy)
        with patch_load_sft_features(features):
            gen = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=1,
                T=16,
                loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
            )
            next(gen)

        assert helper_entered

    def test_enc_tensors_contain_features(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer, features_per_segment=3)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            _inp, _tgt, enc_fids, _enc_mags, enc_mask, _seg, _fp = next(gen)

        assert enc_mask.any(), "Some encoder positions should be masked True"
        assert (enc_fids > 0).any(), "Encoder should receive at least one feature id"

    def test_seg_map_non_zero_over_full_sequence(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=12)
        features = build_synthetic_features(task, tokenizer)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            _inp, _tgt, _fids, _mags, _mask, seg_map, _fp = next(gen)

        assert (seg_map > 0).any(), (
            "Expected full-sequence routing to produce non-zero seg_map entries"
        )

    def test_full_sequence_supervises_role_user_and_assistant_tokens(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=1, user_chars=6, asst_chars=6)
        features = build_synthetic_features(task, tokenizer)
        ids, role_mask = tokenizer.render_conversation(task[0])
        supervised_len = len(ids) - 1

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=supervised_len + 5)
            inputs, targets, _fids, _mags, _mask, _seg, _fp = next(gen)

        assert t.equal(inputs[0, :supervised_len], t.tensor(ids[:-1], dtype=t.int32))
        assert t.equal(targets[0, :supervised_len], t.tensor(ids[1:], dtype=t.int64))
        assert (targets[0, supervised_len:] == -1).all()

        non_assistant_positions = [
            i for i, mask_value in enumerate(role_mask[:-1]) if mask_value == 0
        ]
        assistant_positions = [i for i, mask_value in enumerate(role_mask[:-1]) if mask_value == 1]
        assert non_assistant_positions, "test setup should include BOS/user positions"
        assert assistant_positions, "test setup should include assistant positions"
        assert (targets[0, non_assistant_positions] != -1).all()
        assert (targets[0, assistant_positions] != -1).all()

    def test_packed_row_targets_are_shifted_full_text_until_padding(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=2, asst_chars=2)
        features = build_synthetic_features(task, tokenizer)
        expected_row: list[int] = []
        for i in range(3):
            ids, _role_mask = tokenizer.render_conversation(task[i])
            expected_row.extend(ids)
        T = len(expected_row) - 1

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=T)
            inputs, targets, _fids, _mags, _mask, _seg, _fp = next(gen)

        assert t.equal(inputs[0], t.tensor(expected_row[:-1], dtype=t.int32))
        assert t.equal(targets[0], t.tensor(expected_row[1:], dtype=t.int64))
        assert (targets != -1).all()


# ---------------------------------------------------------------------------
# Functional: multi-turn segment mapping
# ---------------------------------------------------------------------------


class TestConditionedSftMultiTurn:
    """Multi-turn conversations are conditioned and supervised as full sequences."""

    def test_multi_turn_full_sequence_has_nonzero_seg_map(self) -> None:
        """With one exact-fit conversation per row, every input token is routed."""
        tokenizer = SFTMockTokenizer()
        task = MultiTurnTask(n=4, user_chars=5, asst_chars=16)
        features = build_synthetic_features(task, tokenizer)
        ids, _mask = tokenizer.render_conversation(task[0])

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=len(ids) - 1)
            _inputs, _targets, _fids, _mags, _mask, seg_map, _fp = next(gen)

        assert (seg_map[0] > 0).all(), f"Unrouted positions: {seg_map[0].tolist()}"

    def test_multi_turn_all_targets_are_supervised(self) -> None:
        """All non-padding target positions contribute to the LM loss."""
        tokenizer = SFTMockTokenizer()
        task = MultiTurnTask(n=4, user_chars=5, asst_chars=16)
        features = build_synthetic_features(task, tokenizer)
        ids, _mask = tokenizer.render_conversation(task[0])

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=len(ids) - 1)
            _inputs, targets, _fids, _mags, _mask, seg_map, _fp = next(gen)

        assert (targets[0] != -1).all()
        assert (seg_map[0] > 0).all()

    def test_single_turn_full_sequence_is_conditioned(self) -> None:
        """Every exact-fit single-turn input position should have ``seg_map > 0``."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=6)
        features = build_synthetic_features(task, tokenizer)
        ids, _mask = tokenizer.render_conversation(task[0])

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=len(ids) - 1)
            _inputs, targets, _fids, _mags, _mask, seg_map, _fp = next(gen)

        unmasked = targets[0] != -1
        if unmasked.any():
            unmasked_seg_map = seg_map[0][unmasked]
            assert (unmasked_seg_map > 0).all(), (
                "Single-turn conversations must have every supervised position "
                "covered by a feature segment"
            )


# ---------------------------------------------------------------------------
# Integration (tiny CPU): forward + backward + gradient flow
# ---------------------------------------------------------------------------


def _run_one_training_step(
    model: EncoderDecoderGPT,
    batch: tuple,
    optimizer: t.optim.Optimizer,
) -> float:
    """Forward + backward + step on a single SFT batch.

    Returns the scalar loss for the step.  Gradients are left on the
    parameters so callers can inspect them after this returns; the
    optimizer step is also applied to make sure nothing crashes
    end-to-end.
    """
    inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
    loss = result.loss if hasattr(result, "loss") else result
    loss.backward()
    loss_value = float(loss.detach().item())
    optimizer.step()
    return loss_value


def _build_tiny_encdec_matching_sft() -> EncoderDecoderGPT:
    """Build a tiny EncoderDecoderGPT compatible with this file's synthetic data.

    The model must use the same ``n_sae_features`` and ``num_buckets`` values
    that ``_make_gen`` passes to the feature pipeline, otherwise the encoder's
    feature embedding table won't cover the synthetic feature ids.
    """
    return build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        n_layer=2,
        n_embd=64,
        seq_len=64,
        xattn_frequency=1,  # insert cross-attn on every decoder layer
    )


class TestConditionedSftForward:
    """Forward pass through a tiny encoder-decoder with conditioned batches."""

    def test_forward_produces_finite_loss(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        model.eval()

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp_targets = next(gen)

        with t.no_grad():
            result = model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
        loss = float((result.loss if hasattr(result, "loss") else result).item())
        assert math.isfinite(loss), f"Forward loss must be finite, got {loss}"
        assert loss > 0, f"Forward loss should be positive on a random init, got {loss}"


class TestConditionedSftGradientFlow:
    """Backward pass through a conditioned SFT batch must populate gradients
    in *both* the encoder and the cross-attention layers.

    These tests are the core value of PR 1: they prove that the conditioned
    code path is actually training the pieces it's supposed to train.  If the
    feature pipeline silently fell back to empty segments, or if cross-attn
    was accidentally detached from the autograd graph, these assertions
    would flip to zero and catch the regression.
    """

    def _get_param(self, model: EncoderDecoderGPT, dotted_name: str) -> t.Tensor:
        current: object = model
        for part in dotted_name.split("."):
            if part.isdigit():
                # Parent is a ModuleList/ModuleDict at this point — both support
                # subscripting — but ty cannot prove that from `object`.
                current = current[int(part)]  # ty: ignore[not-subscriptable]
            else:
                current = getattr(current, part)
        assert isinstance(current, t.nn.Parameter), (
            f"Expected Parameter at {dotted_name}, got {type(current)}"
        )
        return current

    def test_encoder_feature_embed_receives_gradient(self) -> None:
        """Feature embedding rows that correspond to *conditioning* feature
        ids must receive gradient — not just row 0 (the reserved empty segment).

        If the conditioning path were accidentally broken to always look up
        row 0, ``feat_embed.grad[1:].abs().sum()`` would be zero.  The stricter
        check on non-zero rows guards against that silent collapse.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            batch = next(gen)

        _run_one_training_step(model, batch, optimizer)

        feat_embed = self._get_param(model, "encoder.feature_embed.weight")
        assert feat_embed.grad is not None, "feature_embed.weight should have a grad"
        # Rows 1..N must receive gradient (those are the real feature ids;
        # row 0 may light up from empty-segment lookups regardless).
        non_zero_row_grad = float(feat_embed.grad[1:].abs().sum().item())
        assert non_zero_row_grad > 0.0, (
            f"feature_embed rows 1.. must receive gradient under conditioning, "
            f"got cumulative magnitude {non_zero_row_grad}"
        )

    def test_encoder_magnitude_embed_receives_gradient(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            batch = next(gen)

        _run_one_training_step(model, batch, optimizer)

        mag_embed = self._get_param(model, "encoder.magnitude_embed.weight")
        assert mag_embed.grad is not None
        grad_norm = float(mag_embed.grad.abs().sum().item())
        assert grad_norm > 0.0, f"magnitude_embed.weight gradient must be non-zero, got {grad_norm}"

    def test_cross_attention_layers_receive_gradient(self) -> None:
        """Every cross-attention layer's ``c_q``, ``c_kv``, and ``c_proj``
        weights must receive gradients from a conditioned batch.

        ``c_proj`` is the most interesting one because the layer sits on a
        residual path: if any earlier cross-attention wiring got
        accidentally detached from the autograd graph, ``c_proj.weight``
        would silently stop receiving grad."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            batch = next(gen)

        _run_one_training_step(model, batch, optimizer)

        xattn_items = list(model.cross_attn_layers.items())
        assert xattn_items, "Tiny model must have at least one cross-attn layer"

        for layer_key, layer in xattn_items:
            for weight_name in ("c_q", "c_kv", "c_proj"):
                param = getattr(layer, weight_name).weight
                assert param.grad is not None, f"xattn layer {layer_key} {weight_name} missing grad"
                grad_norm = float(param.grad.abs().sum().item())
                assert grad_norm > 0.0, (
                    f"xattn layer {layer_key} {weight_name} grad must be non-zero, got {grad_norm}"
                )

    def test_decoder_receives_gradient_under_conditioning(self) -> None:
        """Decoder parameters still get gradient signal even when cross-attn
        is the only source of conditioning — i.e. conditioning must not zero
        out the main loss path.  A useful sanity check on top of the encoder
        and xattn checks."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            batch = next(gen)

        _run_one_training_step(model, batch, optimizer)

        any_decoder_grad = False
        for name, param in model.named_parameters():
            if name.startswith("transformer.") and param.grad is not None:
                if float(param.grad.abs().sum().item()) > 0.0:
                    any_decoder_grad = True
                    break
        assert any_decoder_grad, "At least one decoder parameter must receive a non-zero gradient"


# ---------------------------------------------------------------------------
# Functional: multi-step overfit smoke test
# ---------------------------------------------------------------------------


class TestConditionedSftOverfit:
    """Multi-step training must decrease loss on a stable conditioned dataset.

    This isn't a correctness theorem — tiny models with random init and
    random feature ids may wobble — but it's a strong smoke test that the
    entire conditioned SFT loop produces a learnable gradient.  If the
    forward is well-formed but the backward is disconnected, the loss would
    oscillate around the random-init value instead of trending down.
    """

    @pytest.mark.parametrize("loss_mask_mode", list(LossMaskMode))
    def test_short_run_makes_progress(self, loss_mask_mode: LossMaskMode) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        model = _build_tiny_encdec_matching_sft()
        optimizer = setup_optimizer_simple(model, lr=0.01)

        with patch_load_sft_features(features):
            gen = _make_gen(
                task,
                tokenizer,
                alignment_probe_samples=0,
                B=2,
                T=64,
                loss_mask_mode=loss_mask_mode,
            )
            initial_batch = next(gen)
            losses: list[float] = []
            losses.append(_run_one_training_step(model, initial_batch, optimizer))
            for _ in range(15):
                batch = next(gen)
                losses.append(_run_one_training_step(model, batch, optimizer))

        # Not a strict monotonic decrease — just assert the final half of the
        # run has clearly lower mean loss than the first half.  This is
        # robust to tiny-model oscillations while still catching a
        # disconnected backward pass that would keep losses flat.
        first_half = sum(losses[: len(losses) // 2]) / max(1, len(losses) // 2)
        second_half = sum(losses[len(losses) // 2 :]) / max(1, len(losses) - len(losses) // 2)
        assert math.isfinite(first_half) and math.isfinite(second_half)
        assert any(
            param.grad is not None and bool((param.grad != 0).any())
            for param in model.encoder.parameters()
        ), f"encoder gradients missing under {loss_mask_mode.value}"
        assert any(
            param.grad is not None and bool((param.grad != 0).any())
            for param in model.cross_attn_layers.parameters()
        ), f"cross-attention gradients missing under {loss_mask_mode.value}"
        assert second_half < first_half, (
            f"Loss must trend down over a short conditioned run; "
            f"first half mean {first_half:.4f}, second half mean {second_half:.4f}, "
            f"losses {losses}"
        )


# ---------------------------------------------------------------------------
# Missing SFT feature coverage
# ---------------------------------------------------------------------------


class TestConditionedSftMissingFeatureRows:
    """Missing feature rows must never become empty-routed training rows.

    Full-sequence SFT requires exact feature coverage over the SFT task
    mixture. Any missing row raises immediately with enough detail to rerun
    extraction over the right slice.
    """

    def test_missing_coverage_raises(self) -> None:
        """Features for only half the task fail before packing."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        # Build features for only half the examples.
        features = build_synthetic_features(
            task,
            tokenizer,
            skip_indices=frozenset({1, 3, 5, 7}),
        )
        assert len(features) == 4

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            with pytest.raises((AssertionError, ValueError), match="features|missing features"):
                next(gen)

    def test_single_missing_feature_row_raises(self) -> None:
        """Even one missing feature row fails instead of filtering the dataset."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=20, user_chars=3, asst_chars=10)
        features = build_synthetic_features(
            task,
            tokenizer,
            skip_indices=frozenset({0}),
        )

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=64)
            with pytest.raises((AssertionError, ValueError), match="missing features"):
                next(gen)

    def test_empty_coverage_raises(self) -> None:
        """``sft_features`` dict is empty — all examples missing — must
        raise at first ``next(gen)``."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer, skip_indices=frozenset(range(4)))
        assert len(features) == 0

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=64)
            with pytest.raises((AssertionError, ValueError), match="features|missing features"):
                next(gen)

    def test_empty_feature_segment_row_raises(self) -> None:
        """A present feature row with no segments is not valid coverage."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        features[0] = []

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=64)
            with pytest.raises(ValueError, match="feature row has no segments"):
                next(gen)

    def test_truncated_feature_coverage_raises(self) -> None:
        """A present feature row that does not cover the rendered text fails.

        This catches stale extraction shards whose row keys exist but whose
        feature text was truncated or rendered with different settings.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=80)
        features = build_synthetic_features(task, tokenizer)
        first = features[0][0]
        features[0] = [replace(first, char_end=1)]

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=1, T=96)
            with pytest.raises(ValueError, match="coverage ends"):
                next(gen)

    def test_full_coverage_does_not_raise(self) -> None:
        """Features cover every example → generator yields normally.

        This is the happy-path control for the failure tests above: without it,
        the failing tests could accidentally also match a pre-existing
        assertion somewhere else in the pipeline.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=6, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        assert len(features) == 6

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            batch = next(gen)
        assert batch is not None

    def test_error_message_lists_missing_indices(self) -> None:
        """The assertion message must name the missing example indices so
        the user can re-run extraction over the correct slice.  We check
        that the message contains at least one specific missing index."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=10, user_chars=3, asst_chars=10)
        missing_indices = {2, 4, 6}
        features = build_synthetic_features(
            task, tokenizer, skip_indices=frozenset(missing_indices)
        )

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            with pytest.raises((AssertionError, ValueError)) as excinfo:
                next(gen)
        message = str(excinfo.value)
        # At least one of the missing indices must appear in the message.
        assert any(str(idx) in message for idx in missing_indices), (
            f"Expected at least one missing index from {sorted(missing_indices)} "
            f"in the error message, got: {message}"
        )
        # And the count of missing examples must appear.
        assert str(len(missing_indices)) in message, (
            f"Error message must mention the count of missing examples; got: {message}"
        )


# ---------------------------------------------------------------------------
# Targets / inputs contiguity invariant
# ---------------------------------------------------------------------------


class TestSFTGeneratorContiguity:
    """The SFT data generator's ``inputs`` and ``targets`` must be contiguous.

    ``F.cross_entropy`` inside ``EncoderDecoderGPT.forward`` calls
    ``targets.view(-1)``, which refuses a non-contiguous tensor with:

        RuntimeError: view size is not compatible with input tensor's size
        and stride (at least one dimension spans across two contiguous
        subspaces). Use .reshape(...) instead.

    The generator previously built these via ``batch_tensor[:, :-1].to(...)``
    and ``batch_tensor[:, 1:].to(...)`` which, on CPU with an already-int64
    source tensor, left the outputs as non-contiguous slices.  Production
    on GPU hid the bug because ``.to(device='cuda')`` triggers a copy; the
    CPU integration tests in this file exposed it.  The fix makes both
    tensors contiguous explicitly.
    """

    def test_targets_tensor_is_contiguous(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            _inputs, targets, _fids, _mags, _mask, _seg, _fp = next(gen)

        assert targets.is_contiguous(), (
            "targets tensor must be contiguous so that F.cross_entropy can "
            f"call .view(-1) on it (current stride={targets.stride()}, "
            f"shape={tuple(targets.shape)})"
        )

    def test_inputs_tensor_is_contiguous(self) -> None:
        """Parallel invariant for the inputs tensor.  Under the old
        ``.to(dtype=t.int32)`` path this would accidentally come out
        contiguous because the dtype change forced a copy, but after the
        fix we make contiguity explicit so neither branch depends on an
        implicit PyTorch optimization."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=64)
            inputs, _targets, _fids, _mags, _mask, _seg, _fp = next(gen)

        assert inputs.is_contiguous(), (
            "inputs tensor must be contiguous for the same reason "
            f"(current stride={inputs.stride()}, shape={tuple(inputs.shape)})"
        )


# ---------------------------------------------------------------------------
# Slow tier: real nanochat tokenizer on GPU with flash-attn-4
# ---------------------------------------------------------------------------


class TestConditionedSftSlowGpu:
    """GPU-only tests that exercise the real nanochat tokenizer and the
    flash-attn-4 code path (not SDPA).

    These tests complement the fast CPU tier above: the CPU tests prove the
    logic is right with a mock tokenizer and SDPA, and these slow tests
    prove the logic is also right with the real production dependencies.
    They are marked ``@pytest.mark.slow`` so the fast CI tier excludes them
    (the suite default is ``-m 'not slow'``) and are skipped when no CUDA
    is available.  Run with ``uv run pytest -m slow ...``.
    """

    @pytest.fixture(autouse=True)
    def _do_not_override_backend(self):
        """Undo the module-level SDPA override for this class.

        These slow tests deliberately exercise flash-attn-4 (the production
        backend).  We restore the original ``_override_impl`` for the
        duration of each slow test so ``_active_impl()`` resolves to
        whatever the real GPU offers.
        """
        original = _fa_mod._override_impl
        _fa_mod._override_impl = None
        yield
        _fa_mod._override_impl = original

    @pytest.mark.slow
    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required for flash-attn path")
    @pytest.mark.skipif(
        not _tokenizer_available(),
        reason="nanochat tokenizer pickle missing — run the tokenizer download step",
    )
    def test_gradient_flow_with_real_tokenizer_on_gpu(self) -> None:
        """Real nanochat tokenizer + synthetic features + GPU → gradients
        must still flow through encoder and cross-attention.

        This is the strongest integration test in this file: it uses every
        production dependency except the real feature extraction, and proves
        the conditioned SFT loop is end-to-end functional under the same
        attention backend production uses.
        """
        from nanochat.tokenizer import get_tokenizer

        device = "cuda"
        tokenizer = get_tokenizer()

        # Build a toy conversation list that mirrors what the Alpaca task
        # would hand the SFT generator.  Keep the strings short so the
        # tokenizer produces sub-64-token renderings and the whole test
        # fits in a single row at T=128.
        class _InlineTask(Task):
            _convs = [
                {
                    "messages": [
                        {"role": "user", "content": f"Question {i}?"},
                        {"role": "assistant", "content": f"Answer {i}."},
                    ]
                }
                for i in range(6)
            ]

            def num_examples(self) -> int:
                return len(self._convs)

            def get_example(self, index: int) -> dict:
                return self._convs[index]

        task = _InlineTask()

        # Build features that tile the full rendered sequence for each example.
        features: dict[int, list[SegmentFeatures]] = {}
        for idx in range(len(task)):
            conv = task[idx]
            ids, _mask = tokenizer.render_conversation(conv)
            total_chars = source_aligned_char_offsets(tokenizer, ids)[-1]
            if total_chars <= 0:
                continue
            features[idx] = [
                SegmentFeatures(
                    segment_idx=0,
                    position=0,
                    feature_ids=t.tensor([1, 2, 3], dtype=t.int32),
                    feature_mags=t.tensor([3.0, 2.0, 1.0], dtype=t.float32),
                    char_start=0,
                    char_end=total_chars,
                )
            ]

        assert len(features) == len(task), (
            f"Every example must have features for this test to work "
            f"(got {len(features)}/{len(task)})"
        )

        model = _build_tiny_encdec_matching_sft().to(device)
        optimizer = setup_optimizer_simple(model)

        with patch_load_sft_features(features):
            gen = _make_gen(task, tokenizer, alignment_probe_samples=0, B=2, T=128, device=device)
            batch = next(gen)

        _run_one_training_step(model, batch, optimizer)

        feat_embed = model.encoder.feature_embed.weight
        assert feat_embed.grad is not None
        non_zero_row_grad = float(feat_embed.grad[1:].abs().sum().item())
        assert non_zero_row_grad > 0.0, (
            "feature_embed rows 1.. must receive gradient under conditioning "
            "on GPU with flash-attn-4 backend"
        )

        # Every cross-attention c_proj must have moved away from its init
        # after one training step.
        for layer_key, layer in model.cross_attn_layers.items():
            xattn = cast(CrossAttentionLayer, layer)
            g = xattn.c_proj.weight.grad
            assert g is not None, f"xattn {layer_key} c_proj missing grad"
            assert float(g.abs().sum().item()) > 0.0, (
                f"xattn {layer_key} c_proj grad must be non-zero"
            )
