"""End-to-end: a steered SFT mixture from JSONL to a feature-prediction loss.

Every other test of the steered prediction-group split stubs something — the
feature store, the tokenizer, or both. This one runs the real chain with the
real nanochat tokenizer:

    steered JSONL + base task stubs
      -> build_task_mixtures(standard_steered)
      -> convert_mixture_to_parquet(split=train)      (real renders + turn starts)
      -> run_extraction_loop(turn_delimited_fixed_tokens)
      -> compute_statistics
      -> load_sft_features                            (real store, real segments)
      -> make_sft_data_generator(steered_positions=...)
      -> EncoderDecoderGPT with a feature-prediction head

and asserts the emitted ``seg_doc_id`` equals a row-exact expectation computed
independently from the LIVE renders' turn boundaries, for every packed
conversation, until every steered row has been served.
"""

from __future__ import annotations

import json
import math
from itertools import count
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
import tasks.mixtures as mixtures_module
from nanochat.flash_attention import FlashAttnImpl
from scripts.preprocess.convert_sft_to_parquet import convert_mixture_to_parquet
from scripts.train.conditional_encdec_sft import SFTState, make_sft_data_generator
from steering.compute_statistics import compute_statistics
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    compute_feature_pred_loss,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.extract_features import (
    _load_sft_conversion_fingerprint_json,
    run_extraction_loop,
)
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import SegmentMode
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans
from steering.sft_schema import SFTSplit
from tasks.mixtures import SFTMixture, build_task_mixtures
from tests.conftest import get_tokenizer_or_skip
from tests.encdec_test_helpers import N_SAE_FEATURES, NUM_BUCKETS, build_tiny_model
from tests.steering.test_encdec_sft_feature_prediction import _make_per_feature_stats
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE
from tests.steering.test_extract_features_loop import _staging, tiny_params
from tests.steering.test_fp_loss_steered_group_oracle import _oracle_loss

STEERED_ROWS = 6
# The tiny extraction model's tokenizer prepends a BOS that decodes to exactly
# one character, so the feature-source coordinate system is the parquet text
# shifted by one. (The real Gemma source model's is 5.)
SOURCE_BOS_CHARS = 1
SEGMENT_SIZE = 12
B = 2
T = 120

# Every HuggingFace-backed task the standard mixture builds. Stubbed so the
# conversion is deterministic, offline and small.
MIXTURE_TASK_NAMES = (
    "CoCoNot",
    "CustomJSON",
    "GSM8K",
    "MMLU",
    "MetaMathQA",
    "NemotronSafety",
    "OpenCoder",
    "OrcaMath",
    "SimpleSpelling",
    "SmolTalk",
    "SpellingBee",
    "WildGuardMix",
    "WildJailbreak",
)


class _StubTask:
    """Two single-turn conversations whose text identifies the task uniquely."""

    def __init__(self, prefix: str) -> None:
        self.conversations = [
            {
                "messages": [
                    {"role": "user", "content": f"{prefix} question {index}"},
                    {"role": "assistant", "content": f"{prefix} answer {index} text"},
                ]
            }
            for index in range(2)
        ]
        self.start, self.stop, self.step = 0, None, 1

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict:
        return self.conversations[index]

    def num_examples(self) -> int:
        return len(self.conversations)

    def get_example(self, index: int) -> dict:
        return self.conversations[index]


def _write_steered_jsonl(path: Path) -> str:
    path.write_text(
        "".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt number {index}"},
                    {
                        "role": "assistant",
                        "content": f"steered response number {index} continues here",
                    },
                ]
            )
            + "\n"
            for index in range(STEERED_ROWS)
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def steered_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the real convert -> extract -> statistics -> load chain once."""
    tokenizer = get_tokenizer_or_skip()
    # A unique prefix PER CONSTRUCTION: the standard mixture builds some task
    # classes more than once, and two identical renders would make the packed
    # row ambiguous to decode.
    serial = count()
    for task_name in MIXTURE_TASK_NAMES:
        monkeypatch.setattr(
            mixtures_module,
            task_name,
            lambda *args, _name=task_name, **kwargs: _StubTask(f"{_name}-{next(serial)}"),
        )
    monkeypatch.setattr(mixtures_module, "get_base_dir", lambda: str(tmp_path / "fake-base"))

    steered_jsonl = _write_steered_jsonl(tmp_path / "steered.jsonl")
    task, _val_task = build_task_mixtures(SFTMixture.STANDARD_STEERED, steered_jsonl=steered_jsonl)
    assert len(task.steered_positions) == STEERED_ROWS

    parquet_dir = tmp_path / "parquet"
    convert_mixture_to_parquet(
        task,
        parquet_dir,
        conversations_per_shard=PositiveInt.parse(64),
        mixture_name=SFTMixture.STANDARD_STEERED.value,
        split=SFTSplit.TRAIN,
    )
    feature_dir = tmp_path / "features"
    run_extraction_loop(
        tiny_params(
            feature_dir,
            model=TinyHookedTransformer(),
            sae=TinySAE(n_features=int(N_SAE_FEATURES)),
            dataset_phase=f"{SFTMixture.STANDARD_STEERED.value}_sft",
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            segment_size=PositiveInt.parse(SEGMENT_SIZE),
            context_window=PositiveInt.parse(512),
            sft_mixture_fingerprint_json=_load_sft_conversion_fingerprint_json(parquet_dir),
        ),
        [str(parquet_dir / "shard_00000.parquet")],
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "staging", 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "profile"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )
    compute_statistics(
        feature_dir=str(feature_dir),
        output_path=str(feature_dir / "feature_statistics.safetensors"),
        exclude_last_shard=False,
        skip_per_feature_detail=True,
        streaming=False,
        allow_degenerate_statistics=True,
    )
    store = load_sft_features(str(feature_dir), str(parquet_dir))
    assert store.segment_mode is SegmentMode.turn_delimited_fixed_tokens
    return tokenizer, task, store, feature_dir, parquet_dir


def _live_geometry(tokenizer, task) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Per position: the rendered ids, and the assistant turn's source char.

    Computed from the LIVE task through ``legacy_turn_spans``, independently of
    anything the generator or the feature store did.
    """
    renders: dict[int, list[int]] = {}
    assistant_char: dict[int, int] = {}
    specials = ChatSpecialTokens.from_tokenizer(tokenizer)
    for position in range(len(task)):
        ids, _mask = tokenizer.render_conversation(task[position])
        renders[position] = list(ids)
        spans = legacy_turn_spans(ids, specials)
        offsets = source_aligned_char_offsets(tokenizer, ids, source_bos_char_len=SOURCE_BOS_CHARS)
        assistant_char[position] = offsets[spans[1].start + 1]
    return renders, assistant_char


def _packed_positions(
    inputs_row: list[int],
    targets_row: list[int],
    renders: dict[int, list[int]],
) -> list[int]:
    """Recover which conversations a packed row holds, in packing order.

    ``targets`` is ``-1`` from the padding boundary on, which gives the row's
    exact content length; the greedy walk then consumes it conversation by
    conversation. Every rendered conversation is unique (its text names its
    task and index) and each is ~20 tokens, so the walk stays unambiguous even
    where a completely full row's last token falls outside the
    ``inputs = row[:-1]`` slice.
    """
    padding = [index for index, value in enumerate(targets_row) if value == -1]
    content_len = len(inputs_row) + 1 if not padding else padding[0] + 1
    packed: list[int] = []
    offset = 0
    while offset < content_len:
        remaining = inputs_row[offset : min(content_len, len(inputs_row))]
        matches = [
            position
            for position, ids in renders.items()
            if ids[: len(remaining)] == remaining[: len(ids)]
        ]
        assert len(matches) == 1, f"ambiguous packed conversation at offset {offset}"
        packed.append(matches[0])
        offset += len(renders[matches[0]])
    assert offset == content_len, "the decoded conversations must tile the row exactly"
    return packed


def _routed_segment_count(tokenizer, ids: list[int], segments) -> int:
    """How many encoder segments this conversation contributes to a row.

    Routing drops trailing segments that no decoder token reaches, so the count
    is not simply ``len(segments)``. Only the COUNT comes from production here;
    WHICH group each segment lands in is computed independently below, and the
    count itself is already pinned by the non-steered ``seg_doc_id`` tests.
    """
    routing = build_full_sequence_segment_routing(
        ids=ids,
        tokenizer=tokenizer,
        seg_features=(
            [segment.feature_ids for segment in segments],
            [segment.feature_ids for segment in segments],
            [segment.char_start for segment in segments],
            [segment.char_end for segment in segments],
        ),
        source_bos_char_len=SOURCE_BOS_CHARS,
    )
    return len(routing.per_seg_fids) - 1


def _expected_seg_doc_id(
    tokenizer,
    packed: list[int],
    renders: dict[int, list[int]],
    store,
    assistant_char: dict[int, int],
    steered_positions: frozenset[int],
    width: int,
) -> list[int]:
    """The prediction-group row this packing must produce.

    Built from the store's real segment ``char_start``s and the independently
    recovered assistant-turn boundary: a steered conversation's segments split
    at the first one that starts at or after that boundary.
    """
    expected = [-1]
    group = 0
    for position in packed:
        segments = store[position]
        count = _routed_segment_count(tokenizer, renders[position], segments)
        starts = [segment.char_start for segment in segments][:count]
        if position in steered_positions:
            boundary = assistant_char[position]
            expected.extend(group + int(start >= boundary) for start in starts)
            group = expected[-1]
        else:
            expected.extend([group] * len(starts))
        group += 1
    return expected + [-1] * (width - len(expected))


def test_a_steered_mixture_splits_its_groups_all_the_way_to_the_fp_loss(
    steered_artifacts,
) -> None:
    tokenizer, task, store, feature_dir, parquet_dir = steered_artifacts
    renders, assistant_char = _live_geometry(tokenizer, task)
    steered_positions = frozenset(task.steered_positions)

    # Anchor the boundary to an INDEPENDENT coordinate: the parquet's own
    # ``turn_char_starts`` column, written by the converter from the same
    # renders the extractor segmented. ``_live_geometry`` and the generator
    # both re-derive ``offsets[spans[1].start + 1]``, so a SHARED error there
    # (a dropped ``+1``, the wrong ``source_bos_char_len``) would move
    # expectation and production together and stay invisible. The parquet
    # column is in BOS-free text coordinates; the feature-source system is the
    # same text shifted by the source model's BOS.
    table = pq.read_table(str(parquet_dir / "shard_00000.parquet"))
    truth = dict(
        zip(
            table.column("original_index").to_pylist(),
            table.column("turn_char_starts").to_pylist(),
            strict=True,
        )
    )
    for position in sorted(steered_positions):
        assert len(truth[position]) == 2, "a steered row has exactly two logical turns"
        assert assistant_char[position] == truth[position][1] + SOURCE_BOS_CHARS
    boundaries = (
        t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    generator = make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device="cpu",
        sft_state=SFTState(),
        buffer_size=len(task),
        feature_dir=str(feature_dir),
        parquet_dir=str(parquet_dir),
        threshold_cfg=ThresholdConfig(threshold_type=ThresholdType.segment_topk, threshold=1000),
        boundaries=boundaries,
        n_sae_features=N_SAE_FEATURES,
        fp_percentile_table=build_percentile_table(
            _make_per_feature_stats(N_SAE_FEATURES), N_SAE_FEATURES
        ),
        return_seg_doc_id=True,
        min_loss_tokens=4,
        alignment_probe_samples=2,
        feature_source_bos_char_len=SOURCE_BOS_CHARS,
        mixture_name=SFTMixture.STANDARD_STEERED.value,
        expected_split=SFTSplit.TRAIN,
        steered_positions=steered_positions,
    )

    served: set[int] = set()
    split_rows = 0
    batch = None
    for _ in range(len(task)):
        batch = next(generator)
        seg_doc_id = batch[-1]
        for row in range(B):
            packed = _packed_positions(
                [int(value) for value in batch[0][row].tolist()],
                [int(value) for value in batch[1][row].tolist()],
                renders,
            )
            assert packed, "every row must hold at least one conversation"
            expected = _expected_seg_doc_id(
                tokenizer,
                packed,
                renders,
                store,
                assistant_char,
                steered_positions,
                seg_doc_id.shape[1],
            )
            assert [int(value) for value in seg_doc_id[row].tolist()] == expected
            served.update(packed)
            split_rows += sum(1 for position in packed if position in steered_positions)
        if steered_positions <= served:
            break

    assert steered_positions <= served, "every steered row must be served"
    assert split_rows >= STEERED_ROWS

    # Non-vacuity: the expectation above is not "one group per conversation".
    # A steered conversation really does contribute two groups here.
    sample = next(iter(steered_positions))
    sample_starts = [segment.char_start for segment in store[sample]]
    assert min(sample_starts) < assistant_char[sample] <= max(sample_starts)

    assert batch is not None
    inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets, seg_doc_id = batch
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.BEST_TOKEN,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=fp_config,
        n_layer=2,
        n_embd=64,
        seq_len=T,
        xattn_frequency=1,
        vocab_size=tokenizer.get_vocab_size(),
    )
    with t.no_grad():
        result = model(
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=fp_targets,
        )
        fp_logits = model(
            inputs,
            None,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_logits_only=True,
        )
    assert math.isfinite(float(result.loss))
    assert math.isfinite(float(result.feature_pred_loss))

    # The model's own FP loss on a SPLIT seg_doc_id, against the hand-written
    # oracle: the split is honoured all the way through the objective.
    oracle = _oracle_loss(
        fp_logits,
        seg_map,
        fp_targets,
        seg_doc_id,
        fp_config.loss_mode,
        fp_config.loss_fn,
    )
    t.testing.assert_close(
        compute_feature_pred_loss(
            fp_logits, seg_map, fp_targets, seg_doc_id, fp_config.loss_mode, fp_config.loss_fn
        ),
        oracle,
        rtol=1e-5,
        atol=1e-6,
    )
    t.testing.assert_close(result.feature_pred_loss, oracle, rtol=1e-5, atol=1e-6)


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    """flex_attention has no CPU backward; SDPA is what the CPU suites use."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old
