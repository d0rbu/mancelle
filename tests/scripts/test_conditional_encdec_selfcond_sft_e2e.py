"""CPU end-to-end coverage for source-aware self-conditioning on SFT data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
import scripts.train.conditional_encdec_selfcond as selfcond_script
import scripts.train.conditional_encdec_sft as sft_script
from errors import DataLoadError, SelfCondTrainingError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_selfcond import (
    SELFCOND_ACTION_RANKING_POLICY,
    SelfCondSftEvalConfig,
    build_selfcond_sft_data,
    resolve_identity_config,
    resolve_selfcond_sft_sources,
    resolve_selfcond_train_config,
    run_sft_validation_cycle,
)
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.sft_grpo_online_sae import (
    build_online_sae_lineage_metadata,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_vocab import FeatureVocab
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    sft_phase_for,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft.segment_mode_identity import (
    ONLINE_SAE_LINEAGE_META_KEY,
    SFT_SEGMENT_MODE_META_KEY,
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    resolve_turn_self_cond_segment_size_rustbpe,
)
from steering.sft_schema import SFTSplit
from steering.stats_io import save_tokenizer_checksum
from tasks.common import Task
from tasks.mixtures import STEERED_MIXTURES, SFTMixture, sft_val_family
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.encdec_test_helpers import SEQ_LEN, VOCAB_SIZE, build_tiny_model
from tests.scripts.test_conditional_encdec_selfcond_sft_sources import (
    _compact_checkpoint_meta,
)
from tests.scripts.test_conditional_encdec_selfcond_startup import _sft_shaped_meta
from tests.steering.test_encdec_sft_conditioned_integration import (
    MultiTurnTask,
    SFTMockTokenizer,
    build_synthetic_features,
    patch_load_sft_features,
)
from tests.steering.test_sft_features import _write_parquet, _write_shard

N_SAE_FEATURES = 64
NUM_BUCKETS = 2
BASE_STEP = 100
FRESH_STEP = 2
RESUMED_STEP = 4
TOKENIZER_CHECKSUM = "e2e-test-checksum"


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_fa_mod, "_override_impl", FlashAttnImpl.SDPA)


class _RecordingTask(Task):
    """Transparent task proxy that records every requested dataset index."""

    def __init__(self, delegate: Task) -> None:
        super().__init__()
        self.delegate = delegate
        self.indices: list[int] = []

    def num_examples(self) -> int:
        return len(self.delegate)

    def get_example(self, index: int) -> dict:
        self.indices.append(index)
        return self.delegate[index]


@dataclass(frozen=True)
class _E2ECase:
    dataset: SFTMixture
    steered_jsonl: str | None
    base_dir: Path
    checkpoint_leaf: Path
    output_dir: Path
    train_feature_dir: Path
    val_feature_dir: Path
    train_parquet_dir: Path
    val_parquet_dir: Path
    tokenizer: SFTMockTokenizer
    train_task: MultiTurnTask
    val_task: MultiTurnTask
    train_features: dict
    val_features: dict
    base_meta: dict[str, Any]
    feature_mask: FeatureMask | None = None
    consolidation: FeatureClusterConsolidation | None = None
    feature_vocab: FeatureVocab | None = None


# --- Turn-mode fixture geometry --------------------------------------------
#
# The turn fixture emulates a real turn-delimited extraction: every logical
# turn is chunked into pieces of at most `_TURN_SEGMENT_SIZE` SOURCE (Gemma)
# tokens, and the source tokenizer merges a fixed number of render characters
# into one source token. That width alternates with `(turn index + document
# index) % 2`, so full assistant windows come in two different RustBPE widths
# and the measured cadence is a real mean rather than a constant that a
# copy-of-segment_size mutation could also produce. It also makes the fixture
# carry both exclusions on its own: a FULL window inside a user turn, and a
# PARTIAL window (a turn shorter than one full source window).
_TURN_SEGMENT_SIZE = 3
_TURN_SOURCE_TOKEN_CHARS = (2, 3)
_TURN_USER_CHARS = 4
_TURN_ASSISTANT_CHARS = 16
_TURN_SPECIAL_IDS = {
    "<user_start>": 1,
    "<user_end>": 2,
    "<assistant_start>": 3,
    "<assistant_end>": 4,
    "<output_start>": 5,
    "<output_end>": 6,
}
_TURN_CONTENT_ID_BASE = 10


class _TurnMockTokenizer(SFTMockTokenizer):
    """``SFTMockTokenizer`` rendering the legacy nanochat turn structure.

    ``legacy_turn_spans`` recovers logical roles from the chat special-token
    ids, so a turn fixture has to emit them; the base mock's positional role
    markers carry no role at all. Content ids stay clear of the special ids,
    and one id still decodes to exactly one character, so render position
    ``p >= 1`` covers source characters ``[p - 1, p)``.
    """

    def encode_special(self, name: str) -> int:
        return _TURN_SPECIAL_IDS[name]

    def encode(self, text: str) -> list[int]:
        span = self.vocab_size - _TURN_CONTENT_ID_BASE
        return [(ord(char) % span) + _TURN_CONTENT_ID_BASE for char in text]

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        ids: list[int] = [self.get_bos_token_id()]
        mask: list[int] = [0]
        for message in conversation["messages"]:
            assistant = message["role"] == "assistant"
            role = "assistant" if assistant else "user"
            run = [
                self.encode_special(f"<{role}_start>"),
                *self.encode(message["content"]),
                self.encode_special(f"<{role}_end>"),
            ]
            ids.extend(run)
            mask.extend([1 if assistant else 0] * len(run))
        return ids, mask


def _turn_role_spans(ids: list[int]) -> list[tuple[str, int, int]]:
    """``(role, char_start, char_end)`` per turn, scanned without production code."""
    starts: list[int] = []
    roles: list[str] = []
    for index, token_id in enumerate(ids[1:]):
        if token_id == _TURN_SPECIAL_IDS["<user_start>"]:
            starts.append(index)
            roles.append("user")
        elif token_id == _TURN_SPECIAL_IDS["<assistant_start>"]:
            starts.append(index)
            roles.append("assistant")
    ends = [*starts[1:], len(ids) - 1]
    return list(zip(roles, starts, ends, strict=True))


def _turn_segment_plan(ids: list[int], document_id: int) -> list[tuple[int, int, int]]:
    """``(char_start, char_end, source_tokens)`` for every emulated segment."""
    plan: list[tuple[int, int, int]] = []
    for turn_index, (_role, start, end) in enumerate(_turn_role_spans(ids)):
        width = _TURN_SOURCE_TOKEN_CHARS[(turn_index + document_id) % 2]
        cursor = start
        while cursor < end:
            char_end = min(cursor + _TURN_SEGMENT_SIZE * width, end)
            length = char_end - cursor
            assert length % width == 0, "turn lengths must tile the source token width"
            plan.append((cursor, char_end, length // width))
            cursor = char_end
    return plan


def _build_turn_synthetic_features(
    task: Task,
    tokenizer: _TurnMockTokenizer,
    *,
    features_per_segment: int = 3,
) -> dict[int, list[SegmentFeatures]]:
    """Turn-delimited feature rows tiling each render at turn boundaries."""
    out: dict[int, list[SegmentFeatures]] = {}
    feature_cursor = 0
    for index in range(len(task)):
        ids, _mask = tokenizer.render_conversation(task[index])
        position = 0
        segments: list[SegmentFeatures] = []
        for segment_idx, (char_start, char_end, source_tokens) in enumerate(
            _turn_segment_plan(ids, index)
        ):
            fids = t.tensor(
                [
                    (feature_cursor + offset) % N_SAE_FEATURES
                    for offset in range(features_per_segment)
                ],
                dtype=t.int32,
            )
            fmags = t.tensor(
                [float(features_per_segment - offset) for offset in range(features_per_segment)],
                dtype=t.float32,
            )
            feature_cursor = (feature_cursor + features_per_segment) % N_SAE_FEATURES
            segments.append(
                SegmentFeatures(
                    segment_idx=segment_idx,
                    position=position,
                    feature_ids=fids,
                    feature_mags=fmags,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            position += source_tokens
        out[index] = segments
    return out


def _expected_turn_cadence(case: _E2ECase, *, row_capacity: int = SEQ_LEN + 1) -> int:
    """Recompute K from the fixture in character space, without the estimator.

    The generator truncates every render to its packing row before routing, so
    only windows inside the retained prefix (one character per token past BOS)
    count; the last retained window is the clipped/trailing one and is skipped.
    """
    widths: list[int] = []
    for index in range(len(case.train_task)):
        full_ids, mask = case.tokenizer.render_conversation(case.train_task[index])
        ids = full_ids[:row_capacity]
        supervised = (
            sum(mask[1:row_capacity])
            if case.base_meta.get("sft_loss_mask") == LossMaskMode.COMPLETION_ONLY.value
            else len(ids) - 1
        )
        if supervised < int(case.base_meta["user_config"]["min_loss_tokens"]):
            continue
        retained_chars = len(ids) - 1
        assistant = {
            char
            for role, start, end in _turn_role_spans(ids)
            if role == "assistant"
            for char in range(start, end)
        }
        plan = [
            window for window in _turn_segment_plan(full_ids, index) if window[0] < retained_chars
        ]
        for char_start, char_end, source_tokens in plan[:-1]:
            if source_tokens != _TURN_SEGMENT_SIZE:
                continue
            if all(char in assistant for char in range(char_start, char_end)):
                widths.append(char_end - char_start)
    assert widths, "turn fixture must contain full assistant windows"
    return round(Fraction(sum(widths), len(widths)))


def _canonical_feature_dir(
    base_dir: Path,
    phase: DatasetPhase,
    *,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    segment_size: int = 128,
) -> Path:
    # Turn-delimited extraction exists only as a context-independent no-BOS
    # family, and the checkpoint's online-SAE lineage parser enforces that;
    # the fixed fixture keeps its historical leaf-less layout.
    turn = segment_mode is SegmentMode.turn_delimited_fixed_tokens
    return build_feature_dir(
        root=base_dir / "features",
        segment_size=segment_size,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=phase,
        context_independent_features=turn,
        ignore_bos_features=turn,
        segment_mode=segment_mode,
    )


def _small_feature_selection() -> dict[str, Any]:
    return {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "min_k": 1,
        "max_k": 4,
        "mode": "top_k_percentile",
        "delta_weights": None,
    }


def _small_negative_selection() -> dict[str, Any]:
    return {
        "distribution": "geometric",
        "mean": 2.0,
        "std": 2.0,
        "min_k": 0,
        "max_k": 3,
    }


def _mask_allowed_features(
    features: dict[int, list[SegmentFeatures]],
    mask_ids: tuple[int, ...],
) -> dict[int, list[SegmentFeatures]]:
    """Constrain the shared synthetic harness to the MASK-ALLOWED universe —
    deliberately including cluster MEMBERS, not only representatives, so the
    generator's ``consolidate_segment`` performs real member→representative
    folding inside ``main()`` (review finding: pre-converting features to
    representatives left raw-sibling consolidation untested end-to-end)."""
    allowed = t.tensor(mask_ids, dtype=t.int64)
    remapped: dict[int, list[SegmentFeatures]] = {}
    for doc_id, segments in features.items():
        remapped[doc_id] = [
            replace(
                segment,
                feature_ids=allowed[segment.feature_ids.to(t.int64).remainder(len(mask_ids))].to(
                    segment.feature_ids.dtype
                ),
            )
            for segment in segments
        ]
    return remapped


def _write_steered_jsonl(path: Path, rows: int) -> str:
    path.write_text(
        "".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt {index}"},
                    {"role": "assistant", "content": f"steered answer {index}"},
                ]
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return str(path)


def _build_e2e_case(
    tmp_path: Path,
    *,
    compact: bool,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    dataset: SFTMixture = SFTMixture.SAFETY,
    head_block: FeaturePredictionBlock = FeaturePredictionBlock.MLP,
    head_layers: int = 0,
) -> _E2ECase:
    base_dir = tmp_path / "base"
    val_family = sft_val_family(dataset)
    turn = segment_mode is SegmentMode.turn_delimited_fixed_tokens
    segment_size = _TURN_SEGMENT_SIZE if turn else 128
    train_feature_dir = _canonical_feature_dir(
        base_dir,
        sft_phase_for(dataset.value, SFTSplit.TRAIN),
        segment_mode=segment_mode,
        segment_size=segment_size,
    )
    val_feature_dir = _canonical_feature_dir(
        base_dir,
        sft_phase_for(val_family.value, SFTSplit.VAL),
        segment_mode=segment_mode,
        segment_size=segment_size,
    )
    train_feature_dir.mkdir(parents=True)
    val_feature_dir.mkdir(parents=True)

    stats_path = write_steering_stats(
        train_feature_dir / "feature_statistics.safetensors",
        n_sae_features=N_SAE_FEATURES,
    )
    save_tokenizer_checksum(stats_path, TOKENIZER_CHECKSUM)

    train_parquet_dir = base_dir / "sft_data" / dataset.value
    val_parquet_dir = base_dir / "sft_data" / f"{val_family.value}_val"
    train_parquet_dir.mkdir(parents=True)
    val_parquet_dir.mkdir(parents=True)
    _write_parquet(train_parquet_dir, "shard_0000", list(range(20)))
    _write_parquet(val_parquet_dir, "shard_0000", list(range(6)))
    _write_shard(
        val_feature_dir,
        "shard_00000",
        doc_ids=list(range(6)),
        fid_base=0,
    )

    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
        block=head_block,
        head_layers=head_layers,
    )
    feature_mask: FeatureMask | None = None
    consolidation: FeatureClusterConsolidation | None = None
    feature_vocab: FeatureVocab | None = None
    base_meta: dict[str, Any]
    if compact:
        base_meta, feature_mask, consolidation, feature_vocab = _compact_checkpoint_meta(
            n_sae_features=N_SAE_FEATURES,
            mask_ids=tuple(range(32)),
            clusters=(
                (0, 1, 2),
                (4, 5),
                (8, 9, 10),
                (12, 13, 14, 15),
            ),
        )
    else:
        base_meta = _sft_shaped_meta()
    base_meta["dataset"] = dataset.value
    steered_jsonl = (
        _write_steered_jsonl(base_dir / "steered.jsonl", rows=4)
        if dataset in STEERED_MIXTURES
        else None
    )

    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=fp_config,
        n_layer=2,
        xattn_frequency=1,
        feature_vocab=feature_vocab,
    )
    base_meta["model_config"] = asdict(model.dec_config)
    base_meta["encoder_config"] = asdict(model.enc_config)
    base_meta["xattn_config"] = asdict(model.xattn_config)
    base_meta["num_buckets"] = NUM_BUCKETS
    base_meta[SFT_SEGMENT_MODE_META_KEY] = segment_mode.value
    fp_metadata = {
        "block": head_block.value,
        "head_layers": head_layers,
        "weight": fp_config.weight,
        "layer": fp_config.layer,
        "loss_mode": fp_config.loss_mode.value,
        "loss_fn": fp_config.loss_fn.value,
    }
    base_meta["feature_prediction_config"] = fp_metadata
    pretrain_conditioning: Any = base_meta["pretrain_conditioning"]
    user_config: Any = base_meta["user_config"]
    assert isinstance(pretrain_conditioning, dict)
    assert isinstance(user_config, dict)
    user_config["feature_prediction"] = dict(fp_metadata)
    pretrain_conditioning["negative_selection"] = _small_negative_selection()
    user_config.update(
        {
            "feature_dir": str(train_feature_dir),
            # The SFT trainer persists its whole composed cfg, so a steered
            # leaf carries the JSONL self-conditioning must rebuild with.
            "steered_jsonl": steered_jsonl,
            "feature_selection": _small_feature_selection(),
            "feature_source_bos_char_len": 0,
            "alignment_probe_samples": 0,
            "seed": 42,
            "label_dropout": 0.0,
            "next_segment_conditioning_offset": False,
            "use_precomputed_batches": False,
            "segment_size": segment_size,
        }
    )
    if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        # The conditioned-SFT trainer republishes its online-SAE lineage on
        # every checkpoint, and turn-mode post-training requires it. Build it
        # with the trainer's own producer against the fixture's real artifacts
        # so the digest, canonical paths, and source mode are genuine.
        base_meta.update(
            build_online_sae_lineage_metadata(
                feature_dir=str(train_feature_dir),
                statistics_path=stats_path,
                segment_mode=segment_mode,
                sae_config=user_config["sae"],
                feature_selection=user_config["feature_selection"],
                negative_selection=pretrain_conditioning["negative_selection"],
                missing_percentile_policy=str(base_meta["missing_percentile_policy"]),
                n_sae_features=N_SAE_FEATURES,
                num_buckets=NUM_BUCKETS,
                selection_seed=sft_script.SFT_FEATURE_PIPELINE_SEED,
            )
        )

    checkpoint_leaf = base_dir / "checkpoints" / f"{dataset.value}_sft_conditioned_checkpoints"
    save_checkpoint(
        str(checkpoint_leaf),
        BASE_STEP,
        model.state_dict(),
        None,
        base_meta,
        rank=0,
    )

    tokenizer: SFTMockTokenizer
    train_features: dict[int, list[SegmentFeatures]]
    val_features: dict[int, list[SegmentFeatures]]
    if turn:
        tokenizer = _TurnMockTokenizer()
        train_task = MultiTurnTask(
            20,
            user_chars=_TURN_USER_CHARS,
            asst_chars=_TURN_ASSISTANT_CHARS,
        )
        val_task = MultiTurnTask(
            6,
            user_chars=_TURN_USER_CHARS,
            asst_chars=_TURN_ASSISTANT_CHARS,
        )
        train_features = _build_turn_synthetic_features(train_task, tokenizer)
        val_features = _build_turn_synthetic_features(val_task, tokenizer)
    else:
        tokenizer = SFTMockTokenizer()
        train_task = MultiTurnTask(20)
        val_task = MultiTurnTask(6)
        train_features = build_synthetic_features(
            train_task,
            tokenizer,
            n_segments_per_conv=2,
            n_sae_features=N_SAE_FEATURES,
        )
        val_features = build_synthetic_features(
            val_task,
            tokenizer,
            n_segments_per_conv=2,
            n_sae_features=N_SAE_FEATURES,
        )
    if feature_vocab is not None:
        assert consolidation is not None
        mask_ids = tuple(range(16))
        train_features = _mask_allowed_features(train_features, mask_ids)
        val_features = _mask_allowed_features(val_features, mask_ids)
        representative_ids = set(feature_vocab.original_ids)
        member_ids = set(mask_ids) - representative_ids
        assert member_ids, "fixture must exercise real member->representative folding"
        saw_member = False
        for store in (train_features, val_features):
            for segments in store.values():
                for segment in segments:
                    ids = set(segment.feature_ids.tolist())
                    assert ids <= set(mask_ids)
                    saw_member = saw_member or bool(ids & member_ids)
        # The raw features MUST contain non-representative siblings so that
        # main()'s generator performs real consolidation (review finding).
        assert saw_member
    output_dir = checkpoint_leaf / (
        f"selfcond_{SELFCOND_ACTION_RANKING_POLICY}_head_only_dense_checkpoints"
    )
    return _E2ECase(
        dataset=dataset,
        steered_jsonl=steered_jsonl,
        base_dir=base_dir,
        checkpoint_leaf=checkpoint_leaf,
        output_dir=output_dir,
        train_feature_dir=train_feature_dir,
        val_feature_dir=val_feature_dir,
        train_parquet_dir=train_parquet_dir,
        val_parquet_dir=val_parquet_dir,
        tokenizer=tokenizer,
        train_task=train_task,
        val_task=val_task,
        train_features=train_features,
        val_features=val_features,
        base_meta=base_meta,
        feature_mask=feature_mask,
        consolidation=consolidation,
        feature_vocab=feature_vocab,
    )


@pytest.fixture
def e2e_case(tmp_path: Path) -> _E2ECase:
    return _build_e2e_case(tmp_path, compact=False)


_LONG_TURN_COVERAGE_CHARS = 70


class _LongTailTurnTask(MultiTurnTask):
    """The turn task plus one conversation longer than the decoder's packing row."""

    def num_examples(self) -> int:
        return super().num_examples() + 2

    def get_example(self, index: int) -> dict:
        if index < super().num_examples():
            return super().get_example(index)
        if index == super().num_examples():
            # 2 markers + 181 chars tile the width-3 source tokens of this turn
            # parity, and the extra full windows past the row move the
            # untruncated rounded mean (8) away from the truncated one (7).
            return {
                "messages": [
                    {"role": "user", "content": "long"},
                    {"role": "assistant", "content": "z" * 181},
                ]
            }
        # A user turn that fills the whole row: after truncation no assistant
        # token survives, so the generator skips this row for min_loss_tokens
        # before it ever checks coverage.
        return {
            "messages": [
                {"role": "user", "content": "u" * 181},
                {"role": "assistant", "content": "ok"},
            ]
        }


@pytest.fixture
def long_turn_e2e_case(tmp_path: Path) -> _E2ECase:
    """A turn case whose last TRAIN conversation exceeds the packing row.

    Its features stop at ``_LONG_TURN_COVERAGE_CHARS``, well past the row the
    generator retains but more than the end-marker window short of the full
    render: exactly what head-truncated extraction leaves for a long SmolTalk
    conversation.
    """
    case = _build_e2e_case(
        tmp_path,
        compact=False,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    train_task = _LongTailTurnTask(
        20,
        user_chars=_TURN_USER_CHARS,
        asst_chars=_TURN_ASSISTANT_CHARS,
    )
    train_features = _build_turn_synthetic_features(train_task, case.tokenizer)
    for long_index in (train_task.num_examples() - 2, train_task.num_examples() - 1):
        kept = [
            segment
            for segment in train_features[long_index]
            if segment.char_start < _LONG_TURN_COVERAGE_CHARS
        ]
        kept[-1] = replace(kept[-1], char_end=min(kept[-1].char_end, _LONG_TURN_COVERAGE_CHARS))
        train_features[long_index] = kept
    # Completion-only supervision is what makes the user-heavy row one the
    # generator skips (under full_sequence every input position supervises).
    # A threshold that is NOT the production default keeps the oracle honest:
    # it must read what the run reads, not what the code would fall back to.
    base_meta = {
        **case.base_meta,
        "sft_loss_mask": LossMaskMode.COMPLETION_ONLY.value,
        "user_config": {**case.base_meta["user_config"], "min_loss_tokens": 7},
    }
    meta_path = case.checkpoint_leaf / f"meta_{BASE_STEP:06d}.json"
    meta_path.write_text(json.dumps(base_meta), encoding="utf-8")
    return replace(case, train_task=train_task, train_features=train_features, base_meta=base_meta)


@pytest.fixture
def compact_e2e_case(tmp_path: Path) -> _E2ECase:
    return _build_e2e_case(tmp_path, compact=True)


@pytest.fixture
def steered_e2e_case(tmp_path: Path) -> _E2ECase:
    return _build_e2e_case(
        tmp_path,
        compact=False,
        dataset=SFTMixture.STANDARD_STEERED,
    )


@pytest.fixture
def turn_e2e_case(tmp_path: Path) -> _E2ECase:
    return _build_e2e_case(
        tmp_path,
        compact=False,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )


@contextmanager
def _patched_sft_runtime(
    monkeypatch: pytest.MonkeyPatch,
    case: _E2ECase,
    *,
    train_task: Task | None = None,
    val_task: Task | None = None,
    train_segment_mode: SegmentMode | None = None,
    val_segment_mode: SegmentMode | None = None,
    bos_checks: list[tuple[str, str, int, str]] | None = None,
    mixture_overrides: list[dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Install the real-generator fixture stores and tiny runtime surfaces."""
    active_train_task = case.train_task if train_task is None else train_task
    active_val_task = case.val_task if val_task is None else val_task
    recorded_mode = SegmentMode(case.base_meta[SFT_SEGMENT_MODE_META_KEY])
    resolved_train_mode = recorded_mode if train_segment_mode is None else train_segment_mode
    resolved_val_mode = recorded_mode if val_segment_mode is None else val_segment_mode
    with (
        patch_load_sft_features(
            case.train_features,
            split=None,
            mixture=case.dataset.value,
            segment_mode=resolved_train_mode,
        ) as train_store,
        patch_load_sft_features(
            case.val_features,
            split=SFTSplit.VAL,
            mixture=sft_val_family(case.dataset).value,
            segment_mode=resolved_val_mode,
        ) as val_store,
    ):

        def load_sft_features(
            feature_dir: str,
            parquet_dir: str,
            **_kwargs: Any,
        ) -> dict:
            del feature_dir
            resolved_parquet = Path(parquet_dir).resolve()
            if resolved_parquet == case.train_parquet_dir.resolve():
                return train_store
            if resolved_parquet == case.val_parquet_dir.resolve():
                return val_store
            raise AssertionError(f"unexpected SFT parquet directory requested: {parquet_dir}")

        def record_bos_check(
            feature_dir: str,
            parquet_dir: str,
            *,
            configured: int,
            source: str,
        ) -> int:
            if bos_checks is not None:
                bos_checks.append((feature_dir, parquet_dir, configured, source))
            return configured

        monkeypatch.setattr(selfcond_script, "get_base_dir", lambda: str(case.base_dir))
        monkeypatch.setattr(
            selfcond_script,
            "get_tokenizer",
            lambda: case.tokenizer,
        )
        monkeypatch.setattr(
            selfcond_script,
            "get_token_bytes",
            lambda device="cpu": t.ones(
                VOCAB_SIZE,
                dtype=t.int64,
                device=device,
            ),
        )

        def fake_build_task_mixtures(_dataset: SFTMixture, **overrides: Any) -> tuple[Task, Task]:
            if mixture_overrides is not None:
                mixture_overrides.append(overrides)
            return active_train_task, active_val_task

        monkeypatch.setattr(
            selfcond_script,
            "build_task_mixtures",
            fake_build_task_mixtures,
        )
        monkeypatch.setattr(
            selfcond_script,
            "assert_sft_source_bos_char_len_matches",
            record_bos_check,
        )
        with (
            patch.object(sft_script, "load_sft_features", new=load_sft_features),
            patch.object(
                sft_script,
                "_assert_sft_mixture_fingerprint_matches_task",
                new=lambda **_kwargs: None,
            ),
        ):
            yield


def _main_argv(
    case: _E2ECase,
    *,
    num_iterations: int,
    resume_from_step: int = -1,
    extra_overrides: tuple[str, ...] = (),
) -> list[str]:
    argv = [
        "prog",
        "config_name=conditional_encdec_selfcond",
        f"base_checkpoint_dir={case.checkpoint_leaf}",
        "selfcond.data_source=sft",
        "device_type=cpu",
        "run=dummy",
        f"num_iterations={num_iterations}",
        "eval_every=2",
        "save_every=2",
        "eval_tokens=256",
        "device_batch_size=2",
        "total_batch_size=128",
    ]
    if resume_from_step >= 0:
        argv.append(f"resume_from_step={resume_from_step}")
    argv.extend(extra_overrides)
    return argv


def _invoke_main(
    monkeypatch: pytest.MonkeyPatch,
    case: _E2ECase,
    *,
    num_iterations: int,
    resume_from_step: int = -1,
    extra_overrides: tuple[str, ...] = (),
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(
            case,
            num_iterations=num_iterations,
            resume_from_step=resume_from_step,
            extra_overrides=extra_overrides,
        ),
    )
    selfcond_script.main()


def _expected_selfcond_data(case: _E2ECase) -> dict[str, Any]:
    identity = {
        "dataset": case.dataset.value,
        "train_parquet_dir": str(case.train_parquet_dir),
        "train_feature_dir": str(case.train_feature_dir),
        "val_parquet_dir": str(case.val_parquet_dir),
        "val_feature_dir": str(case.val_feature_dir),
        "sft_loss_mask": "full_sequence",
        SFT_SEGMENT_MODE_META_KEY: SegmentMode.fixed_tokens.value,
        "min_loss_tokens": 8,
        "feature_source_bos_char_len": 0,
        "tokenizer_checksum": TOKENIZER_CHECKSUM,
        # write_steering_stats embeds no percentile checksum; the identity
        # block records that absence explicitly rather than omitting the key.
        "percentile_checksum": None,
        "feature_selection": _small_feature_selection(),
        "negative_selection": _small_negative_selection(),
        "missing_percentile_policy": "error",
    }
    if case.feature_mask is not None:
        mask_canonical = case.feature_mask.canonical_dict()
        mask_payload = json.dumps(
            mask_canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        identity.update(
            {
                "feature_mask_canonical_sha256": {
                    "sha256": hashlib.sha256(mask_payload.encode("utf-8")).hexdigest(),
                    "mode": case.feature_mask.mode.value,
                    "id_count": len(case.feature_mask.ids),
                },
                "feature_clusters_descriptor": case.base_meta["feature_clusters_descriptor"],
                "feature_vocab": case.base_meta["feature_vocab"],
            }
        )
    return identity


# The base identity blocks each fixture family actually records. Stated here
# rather than derived from INHERITED_BASE_META_KEYS so that DROPPING a key from
# the production tuple fails these tests instead of quietly shrinking them.
_UNCLUSTERED_INHERITED_KEYS = (
    "dataset",
    "sft_loss_mask",
    "missing_percentile_policy",
    "pretrain_conditioning",
)
_COMPACT_INHERITED_KEYS = _UNCLUSTERED_INHERITED_KEYS + (
    "feature_mask_canonical",
    "feature_clusters_descriptor",
    "feature_clusters_canonical",
    "feature_vocab",
    "feature_prediction_width",
)


def _assert_inherited_base_identity(
    case: _E2ECase,
    meta: dict[str, Any],
    *,
    step: int,
    expected_keys: tuple[str, ...],
) -> None:
    """Every block a downstream consumer reads off the checkpoint it was handed.

    ``run_methods._target_checkpoint_identity`` needs a top-level ``step`` to
    locate ``model_{step:06d}.pt``; ``methods.resolve_raw_feature_delta`` falls
    back to ``pretrain_conditioning``; and
    ``resolve_feature_clusters_from_checkpoint_meta`` requires BOTH cluster
    blocks. Post-training used to publish none of them.
    """
    assert meta["step"] == step
    # The self-conditioning step, not a copy of the base's (which is why `step`
    # is written from loop_state instead of being inherited).
    assert meta["step"] != case.base_meta["step"]
    assert (
        tuple(key for key in selfcond_script.INHERITED_BASE_META_KEYS if key in case.base_meta)
        == expected_keys
    )
    for key in expected_keys:
        assert meta[key] == case.base_meta[key], key


def _load_meta(case: _E2ECase, step: int) -> dict[str, Any]:
    _model_data, _optimizer_data, meta = load_checkpoint(
        str(case.output_dir),
        step,
        "cpu",
        load_optimizer=False,
    )
    return meta


def _load_trained_model(case: _E2ECase, step: int):
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=fp_config,
        n_layer=2,
        xattn_frequency=1,
        feature_vocab=case.feature_vocab,
    )
    model_data, _optimizer_data, _meta = load_checkpoint(
        str(case.output_dir),
        step,
        "cpu",
        load_optimizer=False,
    )
    model.load_state_dict(model_data, strict=True)
    model.eval()
    return model


def _direct_validation_cycle(case: _E2ECase, step: int) -> tuple[float, float]:
    selfcond_block = {
        "data_source": "sft",
        "sft_dataset": None,
        "sft_feature_dir": None,
        "sft_parquet_dir": None,
        "val_sft_feature_dir": None,
        "val_sft_parquet_dir": None,
        "train_scope": "head_only",
        "ste_form": "dense",
        "pool_mode": None,
        "positive_counts": None,
        "negative_counts": None,
    }
    sources = resolve_selfcond_sft_sources(
        selfcond_block,
        case.base_meta,
        str(case.base_dir),
        base_checkpoint_dir=str(case.checkpoint_leaf),
        eval_every=2,
        segment_size=128,
    )
    identity = resolve_identity_config({}, case.base_meta, set())
    identity["negative_selection"] = _small_negative_selection()
    sft_data = build_selfcond_sft_data(
        sources=sources,
        base_meta=case.base_meta,
        identity=identity,
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        ddp_rank=0,
        tokenizer=case.tokenizer,
        state_feature_vocab=case.feature_vocab,
    )
    selfcond_cfg = resolve_selfcond_train_config(
        case.base_meta,
        identity,
        selfcond_block,
        N_SAE_FEATURES,
    )
    return run_sft_validation_cycle(
        model=_load_trained_model(case, step),
        sources=sources,
        sft_data=sft_data,
        val_task=case.val_task,
        tokenizer=case.tokenizer,
        cfg_eval=SelfCondSftEvalConfig(B=2, T=64, eval_steps=2, seed=42),
        selfcond_cfg=selfcond_cfg,
        ddp_rank=0,
        ddp_world_size=1,
        device="cpu",
        token_bytes=t.ones(VOCAB_SIZE, dtype=t.int64),
    )


def test_main_fresh_run_and_buffered_cursor_resume(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    with _patched_sft_runtime(monkeypatch, e2e_case):
        _invoke_main(monkeypatch, e2e_case, num_iterations=FRESH_STEP)

    assert (e2e_case.output_dir / "model_000002.pt").is_file()
    assert (e2e_case.output_dir / "meta_000002.json").is_file()
    fresh_meta = _load_meta(e2e_case, FRESH_STEP)
    assert fresh_meta["selfcond"]["data_source"] == "sft"
    assert fresh_meta["selfcond_data"] == _expected_selfcond_data(e2e_case)
    assert "dataloader_state_dict" not in fresh_meta
    fresh_loop = fresh_meta["loop_state"]
    assert fresh_loop["sft_cursor_semantics"] == selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS
    committed_consumed = fresh_loop["sft_consumed"]
    assert isinstance(committed_consumed, int) and committed_consumed > 0
    dataloader_state = fresh_loop["sft_dataloader_state"]
    assert isinstance(dataloader_state, dict)
    assert dataloader_state["consumed"] == committed_consumed
    assert isinstance(fresh_loop["sft_artifact_coverage"], dict)
    assert fresh_loop["sft_artifact_coverage"]["dataset_size"] == len(e2e_case.train_task)
    assert math.isfinite(fresh_loop["min_val_bpb"])
    assert math.isfinite(fresh_loop["min_val_bpb_selfcond"])

    recording_train = _RecordingTask(e2e_case.train_task)
    recording_val = _RecordingTask(e2e_case.val_task)
    with _patched_sft_runtime(
        monkeypatch,
        e2e_case,
        train_task=recording_train,
        val_task=recording_val,
    ):
        _invoke_main(
            monkeypatch,
            e2e_case,
            num_iterations=RESUMED_STEP,
            resume_from_step=FRESH_STEP,
        )

    assert recording_train.indices
    assert recording_train.indices[0] == committed_consumed
    assert (e2e_case.output_dir / "model_000004.pt").is_file()
    resumed_meta = _load_meta(e2e_case, RESUMED_STEP)
    assert resumed_meta["selfcond_data"] == fresh_meta["selfcond_data"]
    assert (
        resumed_meta["loop_state"]["sft_artifact_coverage"] == fresh_loop["sft_artifact_coverage"]
    )
    assert resumed_meta["loop_state"]["sft_consumed"] > committed_consumed

    with _patched_sft_runtime(monkeypatch, e2e_case):
        with pytest.raises(
            SelfCondTrainingError,
            match=r"sft_dataset.*capabilities.*recorded dataset.*safety",
        ):
            _invoke_main(
                monkeypatch,
                e2e_case,
                num_iterations=RESUMED_STEP,
                resume_from_step=FRESH_STEP,
                extra_overrides=("selfcond.sft_dataset=capabilities",),
            )


def test_transformer_head_survives_real_selfcond_sft_save_pool_override_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_e2e_case(
        tmp_path, compact=True, head_block=FeaturePredictionBlock.TRANSFORMER, head_layers=2,
    )
    overrides = ("selfcond.pool_mode=last_token",)
    with _patched_sft_runtime(monkeypatch, case):
        _invoke_main(monkeypatch, case, num_iterations=FRESH_STEP, extra_overrides=overrides)
        _invoke_main(
            monkeypatch, case, num_iterations=RESUMED_STEP,
            resume_from_step=FRESH_STEP, extra_overrides=overrides,
        )
    for step in (FRESH_STEP, RESUMED_STEP):
        meta = _load_meta(case, step)
        head = meta["feature_prediction_config"]
        assert head == meta["user_config"]["feature_prediction"]
        assert (head["block"], head["head_layers"], head["loss_mode"]) == ("transformer", 2, "last_token")
        state = t.load(case.output_dir / f"model_{step:06d}.pt", weights_only=True)
        for layer in range(2):
            for branch in ("attn", "mlp"):
                assert t.count_nonzero(state[f"feature_pred_head.residual_blocks.{layer}.{branch}.c_proj.weight"])


def test_main_rejects_a_steered_checkpoint_before_building_any_mixture(
    monkeypatch: pytest.MonkeyPatch,
    steered_e2e_case: _E2ECase,
) -> None:
    """A steered SFT leaf is refused outright, however well provisioned it is.

    The SFT generator opens a NEW feature-prediction group at each steered
    conversation's assistant turn; the rollout would have to gate its slots the
    same way to stay aligned with the objective the head was trained under.
    This fixture records a valid ``steered_jsonl`` and has every artifact in
    place, so nothing but the categorical rejection can be raising here -- and
    it raises during source resolution, before ``build_task_mixtures`` and
    before any checkpoint is written.
    """
    assert steered_e2e_case.steered_jsonl is not None
    overrides: list[dict[str, Any]] = []
    with (
        _patched_sft_runtime(
            monkeypatch,
            steered_e2e_case,
            mixture_overrides=overrides,
        ),
        pytest.raises(
            SelfCondTrainingError,
            match=(
                "self-conditioning SFT mode does not support steered mixtures "
                + re.escape(f"(dataset={steered_e2e_case.dataset.value!r})")
            ),
        ),
    ):
        _invoke_main(monkeypatch, steered_e2e_case, num_iterations=FRESH_STEP)

    assert overrides == []
    assert not steered_e2e_case.output_dir.exists()


def test_main_rejects_train_store_mode_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    with _patched_sft_runtime(
        monkeypatch,
        e2e_case,
        train_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    ):
        with pytest.raises(
            SelfCondTrainingError,
            match=(
                rf"train feature store segmentation mismatch.*"
                rf"{e2e_case.train_feature_dir}.*fixed_tokens.*"
                r"turn_delimited_fixed_tokens"
            ),
        ):
            _invoke_main(monkeypatch, e2e_case, num_iterations=FRESH_STEP)


def test_main_rejects_validation_store_mode_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    turn_e2e_case: _E2ECase,
) -> None:
    with _patched_sft_runtime(
        monkeypatch,
        turn_e2e_case,
        val_segment_mode=SegmentMode.fixed_tokens,
    ):
        with pytest.raises(
            SelfCondTrainingError,
            match=(
                rf"val feature store segmentation mismatch.*"
                rf"{turn_e2e_case.val_feature_dir}.*"
                r"turn_delimited_fixed_tokens.*fixed_tokens"
            ),
        ):
            _invoke_main(monkeypatch, turn_e2e_case, num_iterations=FRESH_STEP)


def test_main_accepts_matching_turn_train_and_validation_stores(
    monkeypatch: pytest.MonkeyPatch,
    turn_e2e_case: _E2ECase,
) -> None:
    bos_checks: list[tuple[str, str, int, str]] = []
    with _patched_sft_runtime(
        monkeypatch,
        turn_e2e_case,
        bos_checks=bos_checks,
    ):
        _invoke_main(monkeypatch, turn_e2e_case, num_iterations=FRESH_STEP)
    assert bos_checks == [
        (
            str(turn_e2e_case.train_feature_dir),
            str(turn_e2e_case.train_parquet_dir),
            0,
            SFTSplit.TRAIN.value,
        ),
        (
            str(turn_e2e_case.val_feature_dir),
            str(turn_e2e_case.val_parquet_dir),
            0,
            SFTSplit.VAL.value,
        ),
    ]

    meta = _load_meta(turn_e2e_case, FRESH_STEP)
    _assert_inherited_base_identity(
        turn_e2e_case,
        meta,
        step=FRESH_STEP,
        expected_keys=_UNCLUSTERED_INHERITED_KEYS,
    )
    assert meta[SFT_SEGMENT_MODE_META_KEY] == (SegmentMode.turn_delimited_fixed_tokens.value)
    assert meta["selfcond_data"][SFT_SEGMENT_MODE_META_KEY] == (
        SegmentMode.turn_delimited_fixed_tokens.value
    )
    # kills: recording a configured/constant cadence, copying the Gemma
    # segment_size, or measuring over anything but the full assistant windows
    # -- the fixture's two window widths make K a mean no other rule produces.
    expected_cadence = _expected_turn_cadence(turn_e2e_case)
    assert expected_cadence == 7
    assert expected_cadence != _TURN_SEGMENT_SIZE
    assert meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] == expected_cadence
    assert TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY not in meta["selfcond"]
    assert (
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir=str(turn_e2e_case.output_dir),
        )
        == expected_cadence
    )

    with _patched_sft_runtime(monkeypatch, turn_e2e_case):
        _invoke_main(
            monkeypatch,
            turn_e2e_case,
            num_iterations=RESUMED_STEP,
            resume_from_step=FRESH_STEP,
        )
    resumed_meta = _load_meta(turn_e2e_case, RESUMED_STEP)
    assert resumed_meta[SFT_SEGMENT_MODE_META_KEY] == meta[SFT_SEGMENT_MODE_META_KEY]
    assert resumed_meta["selfcond_data"] == meta["selfcond_data"]
    assert resumed_meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] == expected_cadence


# kills: dropping the resume cadence comparison, which would let a run resume
# against artifacts whose measured cadence differs from the one the resumed
# weights (and every downstream consumer of the checkpoint) were minted with.
# The float case kills comparing the raw recorded value: 7.0 == 7 would resume
# and silently normalise the checkpoint on its next save, while the inference
# reader rejects the same metadata.
@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        pytest.param(
            lambda measured: measured + 1,
            r"turn_self_cond_segment_size_rustbpe mismatch on resume.*records {plus_one}.*measure {measured}",
            id="drifted-artifacts",
        ),
        pytest.param(
            lambda measured: float(measured),
            r"turn_self_cond_segment_size_rustbpe must be a positive integer, got {as_float}",
            id="float-recorded-cadence",
        ),
    ],
)
def test_main_rejects_a_resumed_cadence_the_artifacts_no_longer_measure(
    monkeypatch: pytest.MonkeyPatch,
    turn_e2e_case: _E2ECase,
    corrupt,
    expected: str,
) -> None:
    with _patched_sft_runtime(monkeypatch, turn_e2e_case):
        _invoke_main(monkeypatch, turn_e2e_case, num_iterations=FRESH_STEP)

    meta_path = turn_e2e_case.output_dir / f"meta_{FRESH_STEP:06d}.json"
    corrupted = json.loads(meta_path.read_text(encoding="utf-8"))
    measured = corrupted[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY]
    corrupted[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] = corrupt(measured)
    meta_path.write_text(json.dumps(corrupted), encoding="utf-8")

    with _patched_sft_runtime(monkeypatch, turn_e2e_case):
        with pytest.raises(
            SelfCondTrainingError,
            match=expected.format(
                measured=measured, plus_one=measured + 1, as_float=float(measured)
            ),
        ):
            _invoke_main(
                monkeypatch,
                turn_e2e_case,
                num_iterations=RESUMED_STEP,
                resume_from_step=FRESH_STEP,
            )


# The generator truncates a long conversation to the packing row and trains on
# it; the cadence pass must measure that same prefix instead of aborting on the
# full render's coverage gap (the first production 8000-conversation run died
# here) or counting windows the generator never sees.
# kills: checking coverage against the untruncated render (abort), and routing
# the untruncated render (K moves from the truncated oracle's value).
def test_main_measures_a_row_truncated_conversation_as_the_generator_trains_it(
    monkeypatch: pytest.MonkeyPatch,
    long_turn_e2e_case: _E2ECase,
) -> None:
    case = long_turn_e2e_case
    for long_index in (case.train_task.num_examples() - 2, case.train_task.num_examples() - 1):
        ids, _mask = case.tokenizer.render_conversation(case.train_task[long_index])
        assert len(ids) > SEQ_LEN + 1
        full_render_gap = (len(ids) - 1) - case.train_features[long_index][-1].char_end
        assert full_render_gap > 20
    _ids, mask = case.tokenizer.render_conversation(
        case.train_task[case.train_task.num_examples() - 1]
    )
    assert sum(mask[1 : SEQ_LEN + 1]) < int(case.base_meta["user_config"]["min_loss_tokens"]), (
        "the user-heavy row must be one the generator skips"
    )
    truncated = _expected_turn_cadence(case)
    assert truncated != _expected_turn_cadence(case, row_capacity=len(ids))

    with _patched_sft_runtime(monkeypatch, case):
        _invoke_main(monkeypatch, case, num_iterations=FRESH_STEP)

    meta = json.loads((case.output_dir / f"meta_{FRESH_STEP:06d}.json").read_text(encoding="utf-8"))
    assert meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] == truncated


# kills: reinstating the removed operator-declared cadence knob -- a config
# value must be refused, not silently override the measurement.
def test_main_rejects_a_configured_turn_cadence(
    monkeypatch: pytest.MonkeyPatch,
    turn_e2e_case: _E2ECase,
) -> None:
    with _patched_sft_runtime(monkeypatch, turn_e2e_case):
        with pytest.raises(
            SelfCondTrainingError,
            match=r"is not a config knob.*measured from the training artifacts",
        ):
            _invoke_main(
                monkeypatch,
                turn_e2e_case,
                num_iterations=FRESH_STEP,
                extra_overrides=(f"+selfcond.{TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY}=17",),
            )


# kills: dropping either half of main()'s base-lineage gate -- the
# turn-base-must-carry-a-lineage refusal (mutating its `segment_mode=` argument
# to fixed_tokens) and the malformed-lineage refusal -- both of which must fire
# before any SFT task index is touched.
@pytest.mark.parametrize(
    ("lineage", "match"),
    [
        (None, r"base checkpoint.*turn-delimited but has no"),
        (
            {
                "schema_version": 1,
                "source_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value,
            },
            r"base checkpoint.*malformed online-SAE lineage.*incomplete",
        ),
    ],
    ids=["absent", "incomplete"],
)
def test_main_rejects_malformed_lineage_before_sft_data_activity(
    monkeypatch: pytest.MonkeyPatch,
    turn_e2e_case: _E2ECase,
    lineage: dict[str, object] | None,
    match: str,
) -> None:
    meta_path = turn_e2e_case.checkpoint_leaf / f"meta_{BASE_STEP:06d}.json"
    base_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if lineage is None:
        del base_meta[ONLINE_SAE_LINEAGE_META_KEY]
    else:
        base_meta[ONLINE_SAE_LINEAGE_META_KEY] = lineage
    meta_path.write_text(json.dumps(base_meta), encoding="utf-8")
    train_task = _RecordingTask(turn_e2e_case.train_task)
    val_task = _RecordingTask(turn_e2e_case.val_task)

    with _patched_sft_runtime(
        monkeypatch,
        turn_e2e_case,
        train_task=train_task,
        val_task=val_task,
    ):
        with pytest.raises(SelfCondTrainingError, match=match):
            _invoke_main(monkeypatch, turn_e2e_case, num_iterations=FRESH_STEP)

    assert train_task.indices == []
    assert val_task.indices == []


# kills: dropping the `except ValueError -> SelfCondTrainingError` wrap around
# the base-checkpoint segment-mode resolution in main(). Without it the run
# aborts with a bare ValueError that no caller of this trainer expects.
def test_main_wraps_corrupt_base_segment_mode_as_a_training_error(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    meta_path = e2e_case.checkpoint_leaf / f"meta_{BASE_STEP:06d}.json"
    base_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base_meta[SFT_SEGMENT_MODE_META_KEY] = "bogus"
    meta_path.write_text(json.dumps(base_meta), encoding="utf-8")
    train_task = _RecordingTask(e2e_case.train_task)
    val_task = _RecordingTask(e2e_case.val_task)

    with _patched_sft_runtime(
        monkeypatch,
        e2e_case,
        train_task=train_task,
        val_task=val_task,
    ):
        with pytest.raises(
            SelfCondTrainingError,
            match=r"unknown sft_segment_mode='bogus'",
        ):
            _invoke_main(monkeypatch, e2e_case, num_iterations=FRESH_STEP)

    assert train_task.indices == []
    assert val_task.indices == []


# kills: dropping the `except ValueError -> SelfCondTrainingError` wrap around
# the RESUME checkpoint's segment-mode resolution in main(), and weakening that
# resolution from assert_resume_sft_segment_mode_matches(base_sft_segment_mode,
# ...) to a bare resolve that never compares base and resume. The resume policy
# assert passes either way (the selfcond block is untouched).
def test_main_wraps_corrupt_resume_segment_mode_as_a_training_error(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    with _patched_sft_runtime(monkeypatch, e2e_case):
        _invoke_main(monkeypatch, e2e_case, num_iterations=FRESH_STEP)

    resume_meta_path = e2e_case.output_dir / f"meta_{FRESH_STEP:06d}.json"
    resume_meta = json.loads(resume_meta_path.read_text(encoding="utf-8"))
    assert resume_meta[SFT_SEGMENT_MODE_META_KEY] == SegmentMode.fixed_tokens.value

    # Corrupt both witnesses at once, so the second case resolves cleanly to a
    # VALID but wrong mode instead of tripping the conflict rule.
    for recorded, match in (
        ("bogus", r"unknown sft_segment_mode='bogus'"),
        (
            SegmentMode.turn_delimited_fixed_tokens.value,
            r"sft_segment_mode mismatch on resume",
        ),
    ):
        resume_meta[SFT_SEGMENT_MODE_META_KEY] = recorded
        resume_meta["selfcond_data"][SFT_SEGMENT_MODE_META_KEY] = recorded
        resume_meta_path.write_text(json.dumps(resume_meta), encoding="utf-8")
        with _patched_sft_runtime(monkeypatch, e2e_case):
            with pytest.raises(SelfCondTrainingError, match=match):
                _invoke_main(
                    monkeypatch,
                    e2e_case,
                    num_iterations=RESUMED_STEP,
                    resume_from_step=FRESH_STEP,
                )


def test_main_timing_profile_covers_train_eval_and_save_without_checkpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    profile_dir = e2e_case.base_dir / "timing-profile"
    with _patched_sft_runtime(monkeypatch, e2e_case):
        _invoke_main(
            monkeypatch,
            e2e_case,
            num_iterations=2,
            extra_overrides=(
                "profile_timing=true",
                f"profile_output_dir={profile_dir}",
            ),
        )

    timing_path = profile_dir / "selfcond_timing_rank0.csv"
    assert timing_path.is_file()
    with timing_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["name"] for row in rows]
    assert names.count("selfcond/train_step") == 2
    assert names.count("selfcond/validation") == 1
    assert names.count("selfcond/checkpoint_save") == 1
    assert {row["step"] for row in rows} == {"0", "1"}
    assert all(row["rank"] == "0" for row in rows)
    assert {
        "sft/refill_buffer",
        "sft/render_conversation",
        "sft/build_text_tensors",
        "sft/build_conv_segments",
    } <= set(names)

    meta = _load_meta(e2e_case, 2)
    profile_keys = {
        "profile_timing",
        "profile_trace",
        "profile_trace_steps",
        "profile_output_dir",
    }
    assert profile_keys.isdisjoint(meta)
    assert profile_keys.isdisjoint(meta["selfcond"])


def test_main_trace_is_bounded_to_training_step_and_exports_real_chrome_json(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    profile_dir = e2e_case.base_dir / "trace-profile"
    with _patched_sft_runtime(monkeypatch, e2e_case):
        _invoke_main(
            monkeypatch,
            e2e_case,
            num_iterations=1,
            extra_overrides=(
                "profile_trace=0",
                "profile_trace_steps=1",
                f"profile_output_dir={profile_dir}",
            ),
        )

    trace_path = profile_dir / "selfcond_trace_step0_rank0.json"
    payload = json.loads(trace_path.read_text())
    assert isinstance(payload, dict)
    events = payload.get("traceEvents")
    assert isinstance(events, list) and events
    names = {
        event.get("name")
        for event in events
        if isinstance(event, dict) and isinstance(event.get("name"), str)
    }
    assert {
        "selfcond/train_step",
        "selfcond/rollout/model_forward",
        "selfcond/pass1/model_forward",
        "selfcond/pass2/lm_forward",
        "selfcond/backward",
        "selfcond/optimizer_step",
    } <= names
    assert "selfcond/validation" not in names
    assert "selfcond/checkpoint_save" not in names


def test_sft_mode_isolates_val_artifacts_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    for parquet_path in e2e_case.val_parquet_dir.glob("shard_*.parquet"):
        parquet_path.unlink()

    with _patched_sft_runtime(monkeypatch, e2e_case):
        with pytest.raises(DataLoadError) as exc_info:
            _invoke_main(monkeypatch, e2e_case, num_iterations=FRESH_STEP)
    message = str(exc_info.value)
    assert str(e2e_case.val_parquet_dir) in message
    assert str(e2e_case.val_feature_dir) in message

    _write_parquet(e2e_case.val_parquet_dir, "shard_0000", list(range(6)))
    packed_val_dir = (
        e2e_case.train_feature_dir
        / "pct"
        / "pretraining"
        / "tok"
        / "pretraining"
        / "c64"
        / "val_sall"
    )
    packed_val_dir.mkdir(parents=True)
    (packed_val_dir / "shard_00000.safetensors.zst").write_bytes(b"packed")

    def reject_pretraining_discovery(**_kwargs: Any) -> None:
        raise AssertionError("pretraining discovery must not run in SFT mode")

    monkeypatch.setattr(
        selfcond_script,
        "discover_packed_selection",
        reject_pretraining_discovery,
    )
    with _patched_sft_runtime(monkeypatch, e2e_case):
        _invoke_main(monkeypatch, e2e_case, num_iterations=FRESH_STEP)

    with _patched_sft_runtime(monkeypatch, e2e_case):
        first = _direct_validation_cycle(e2e_case, FRESH_STEP)
        second = _direct_validation_cycle(e2e_case, FRESH_STEP)
    assert math.isfinite(first[0]) and math.isfinite(first[1])
    assert first == second


def test_main_rejects_prefetch_for_buffered_sft_cursor(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    with _patched_sft_runtime(monkeypatch, e2e_case):
        with pytest.raises(SelfCondTrainingError, match="buffered_v2 cursor contract"):
            _invoke_main(
                monkeypatch,
                e2e_case,
                num_iterations=FRESH_STEP,
                extra_overrides=("train_prefetch_batches=2",),
            )


def test_main_rejects_sft_base_under_pretraining_default(
    monkeypatch: pytest.MonkeyPatch,
    e2e_case: _E2ECase,
) -> None:
    argv = _main_argv(e2e_case, num_iterations=FRESH_STEP)
    argv.remove("selfcond.data_source=sft")
    monkeypatch.setattr(sys, "argv", argv)

    with _patched_sft_runtime(monkeypatch, e2e_case):
        with pytest.raises(
            SelfCondTrainingError,
            match="this base is an SFT checkpoint",
        ):
            selfcond_script.main()


def test_compact_clustered_main_fresh_resume_determinism_and_threading(
    monkeypatch: pytest.MonkeyPatch,
    compact_e2e_case: _E2ECase,
) -> None:
    case = compact_e2e_case
    assert case.feature_mask is not None
    assert case.consolidation is not None
    assert case.feature_vocab is not None
    assert case.feature_vocab.size < N_SAE_FEATURES

    generator_calls: list[tuple[str, dict[str, Any]]] = []
    train_consolidations: list[FeatureClusterConsolidation | None] = []
    eval_consolidations: list[FeatureClusterConsolidation | None] = []
    real_train_generator = selfcond_script.make_sft_data_generator
    real_val_generator = sft_script.make_sft_data_generator
    real_train_step = selfcond_script.run_selfcond_train_step
    real_selfcond_eval = selfcond_script.evaluate_selfcond_bpb_sft

    def recording_train_generator(**kwargs):
        generator_calls.append(("train", kwargs))
        return real_train_generator(**kwargs)

    def recording_val_generator(**kwargs):
        generator_calls.append(("val", kwargs))
        return real_val_generator(**kwargs)

    def recording_train_step(*args, **kwargs):
        train_consolidations.append(kwargs.get("consolidation"))
        return real_train_step(*args, **kwargs)

    def recording_selfcond_eval(*args, **kwargs):
        eval_consolidations.append(kwargs.get("consolidation"))
        return real_selfcond_eval(*args, **kwargs)

    monkeypatch.setattr(
        selfcond_script,
        "make_sft_data_generator",
        recording_train_generator,
    )
    monkeypatch.setattr(
        sft_script,
        "make_sft_data_generator",
        recording_val_generator,
    )
    monkeypatch.setattr(
        selfcond_script,
        "run_selfcond_train_step",
        recording_train_step,
    )
    monkeypatch.setattr(
        selfcond_script,
        "evaluate_selfcond_bpb_sft",
        recording_selfcond_eval,
    )

    with _patched_sft_runtime(monkeypatch, case):
        _invoke_main(monkeypatch, case, num_iterations=FRESH_STEP)

    fresh_meta = _load_meta(case, FRESH_STEP)
    _assert_inherited_base_identity(
        case,
        fresh_meta,
        step=FRESH_STEP,
        expected_keys=_COMPACT_INHERITED_KEYS,
    )
    assert fresh_meta["selfcond"]["data_source"] == "sft"
    assert fresh_meta["selfcond_data"] == _expected_selfcond_data(case)
    assert "dataloader_state_dict" not in fresh_meta
    fresh_loop = fresh_meta["loop_state"]
    assert fresh_loop["sft_cursor_semantics"] == selfcond_script.SELFCOND_SFT_CURSOR_SEMANTICS
    assert fresh_loop["sft_consumed"] > 0
    assert isinstance(fresh_loop["sft_dataloader_state"], dict)
    assert math.isfinite(fresh_loop["min_val_bpb"])
    assert math.isfinite(fresh_loop["min_val_bpb_selfcond"])

    representative_mask = case.consolidation.effective_negative_mask()
    train_calls = [kwargs for kind, kwargs in generator_calls if kind == "train"]
    val_calls = [kwargs for kind, kwargs in generator_calls if kind == "val"]
    assert len(train_calls) == 1
    assert len(val_calls) == 2
    for kind, call in generator_calls:
        assert call["feature_mask"] == case.feature_mask
        assert call["consolidation"] is not None
        assert call["cluster_threshold_percentile_table"] is None
        negative_bundle = call["negative_bundle"]
        assert negative_bundle is not None
        assert negative_bundle.feature_mask == representative_mask
        assert call["expected_split"] is (SFTSplit.TRAIN if kind == "train" else SFTSplit.VAL)
    train_negative_mask = train_calls[0]["negative_bundle"].feature_mask
    assert all(call["negative_bundle"].feature_mask is train_negative_mask for call in val_calls)
    assert train_consolidations
    assert all(item is not None for item in train_consolidations)
    assert eval_consolidations
    assert all(item is not None for item in eval_consolidations)

    committed_consumed = fresh_loop["sft_consumed"]
    recording_train = _RecordingTask(case.train_task)
    recording_val = _RecordingTask(case.val_task)
    with _patched_sft_runtime(
        monkeypatch,
        case,
        train_task=recording_train,
        val_task=recording_val,
    ):
        _invoke_main(
            monkeypatch,
            case,
            num_iterations=RESUMED_STEP,
            resume_from_step=FRESH_STEP,
        )

    assert recording_train.indices
    assert recording_train.indices[0] == committed_consumed
    resumed_meta = _load_meta(case, RESUMED_STEP)
    assert resumed_meta["selfcond_data"] == fresh_meta["selfcond_data"]
    assert resumed_meta["loop_state"]["sft_consumed"] > committed_consumed
    assert (
        resumed_meta["loop_state"]["sft_artifact_coverage"] == fresh_loop["sft_artifact_coverage"]
    )

    with _patched_sft_runtime(monkeypatch, case):
        first = _direct_validation_cycle(case, RESUMED_STEP)
        second = _direct_validation_cycle(case, RESUMED_STEP)
    assert math.isfinite(first[0]) and math.isfinite(first[1])
    assert first == second
