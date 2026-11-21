import pytest
import torch as t
from hypothesis import given
from hypothesis import strategies as st

from errors import ValidationError
from steering.compute_statistics import compute_segment_lengths, compute_statistics
from steering.feature_io import save_features
from steering.segmentation import SegmentMode
from steering.stats_io import load_statistics

TURN_MODE = SegmentMode.turn_delimited_fixed_tokens


def _legacy_compute_segment_lengths(shard_data: dict) -> list[int]:
    """Legacy implementation copied verbatim from source commit a18873736."""
    seg_mode = shard_data["segment_mode"]
    seg_size = shard_data["segment_size"]
    if seg_mode != "fixed_tokens":
        raise ValidationError(
            f"compute_segment_lengths requires segment_mode='fixed_tokens', got {seg_mode!r}"
        )
    if seg_size <= 0:
        raise ValidationError(f"compute_segment_lengths requires segment_size > 0, got {seg_size}")

    doc_ids = shard_data["doc_ids"]
    n_segments = shard_data["n_segments"]
    ctx_win = shard_data["context_window"]
    doc_tokens = shard_data["doc_token_counts"]

    lengths: list[int] = []
    prev_doc = -1
    for seg_idx in range(n_segments):
        cur_doc = int(doc_ids[seg_idx].item())
        if cur_doc != prev_doc:
            # Fix up last segment of previous document
            if prev_doc >= 0 and lengths:
                doc_n_tokens = int(doc_tokens[prev_doc].item())
                kept_len = min(doc_n_tokens, ctx_win) if ctx_win > 0 else doc_n_tokens
                remainder = kept_len % seg_size
                if remainder > 0:
                    lengths[-1] = remainder
            prev_doc = cur_doc
        lengths.append(seg_size)

    # Fix up last segment of final document
    if prev_doc >= 0 and lengths:
        doc_n_tokens = int(doc_tokens[prev_doc].item())
        kept_len = min(doc_n_tokens, ctx_win) if ctx_win > 0 else doc_n_tokens
        remainder = kept_len % seg_size
        if remainder > 0:
            lengths[-1] = remainder

    return lengths


def _fixed_shard(segment_size, token_counts, context_window):
    doc_ids = []
    positions = []
    for doc_id, token_count in enumerate(token_counts):
        kept = min(token_count, context_window) if context_window > 0 else token_count
        doc_positions = list(range(0, kept, segment_size))
        doc_ids.extend([doc_id] * len(doc_positions))
        positions.extend(doc_positions)
    return {
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_positions": t.tensor(positions, dtype=t.int32),
        "n_segments": len(positions),
        "context_window": context_window,
        "doc_token_counts": t.tensor(token_counts, dtype=t.int32),
    }


@given(
    segment_size=st.integers(min_value=1, max_value=32),
    token_counts=st.lists(st.integers(min_value=1, max_value=128), min_size=1, max_size=8),
    context_window=st.one_of(st.just(0), st.integers(min_value=1, max_value=128)),
)
def test_fixed_valid_shards_match_legacy(segment_size, token_counts, context_window):
    shard = _fixed_shard(segment_size, token_counts, context_window)
    assert compute_segment_lengths(shard) == _legacy_compute_segment_lengths(shard)


def test_positions_fix_nonuniform_fixed_geometry():
    shard = {
        "segment_mode": "fixed_tokens",
        "segment_size": 4,
        "doc_ids": t.tensor([0, 0, 0], dtype=t.int32),
        "segment_positions": t.tensor([0, 3, 8], dtype=t.int32),
        "n_segments": 3,
        "context_window": 0,
        "doc_token_counts": t.tensor([10], dtype=t.int32),
    }
    assert _legacy_compute_segment_lengths(shard) == [4, 4, 2]
    assert compute_segment_lengths(shard) == [3, 5, 2]


def test_positions_need_not_start_at_segment_size_multiple():
    shard = {
        "segment_mode": "fixed_tokens",
        "segment_size": 4,
        "doc_ids": t.tensor([0, 0], dtype=t.int32),
        "segment_positions": t.tensor([2, 6], dtype=t.int32),
        "n_segments": 2,
        "context_window": 0,
        "doc_token_counts": t.tensor([9], dtype=t.int32),
    }
    assert _legacy_compute_segment_lengths(shard) == [4, 1]
    assert compute_segment_lengths(shard) == [4, 3]


@pytest.mark.parametrize(
    "positions,doc_ids,counts,expected",
    [
        ([0, 4, 7, 0, 2], [0, 0, 0, 1, 1], [10, 5], [4, 3, 3, 2, 3]),
        ([0, 3, 6], [0, 0, 0], [8], [3, 3, 2]),
    ],
)
def test_turn_lengths_are_hand_computed(positions, doc_ids, counts, expected):
    shard = {
        "segment_mode": TURN_MODE.value,
        "segment_size": 4,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_positions": t.tensor(positions, dtype=t.int32),
        "n_segments": len(positions),
        "context_window": 0,
        "doc_token_counts": t.tensor(counts, dtype=t.int32),
    }
    assert compute_segment_lengths(shard) == expected


def _minimal_shard(**overrides):
    shard = _fixed_shard(4, [8], 0)
    shard.update(overrides)
    return shard


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"segment_mode": "sentences"}, "segment_mode"),
        ({"segment_size": 0}, "segment_size"),
        (
            {"segment_positions": t.tensor([0, 0], dtype=t.int32)},
            "non-positive segment length",
        ),
    ],
)
def test_invalid_segment_length_inputs_raise(overrides, match):
    with pytest.raises(ValidationError, match=match):
        compute_segment_lengths(_minimal_shard(**overrides))


def test_non_positive_length_reports_first_offending_segment():
    """A vectorized implementation must retain the first-index diagnostic."""
    shard = _minimal_shard(
        doc_ids=t.tensor([0, 0, 0], dtype=t.int32),
        segment_positions=t.tensor([0, 0, 0], dtype=t.int32),
        n_segments=3,
    )
    with pytest.raises(
        ValidationError,
        match=r"Computed non-positive segment length 0 at segment 0 .*position=0",
    ):
        compute_segment_lengths(shard)


def test_compute_statistics_writes_turn_mode_metadata(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    save_features(
        {
            "offsets": t.tensor([0, 1, 2], dtype=t.int64),
            "feature_ids": t.tensor([1, 2], dtype=t.int32),
            "feature_mags": t.tensor([1.0, 2.0], dtype=t.float32),
            "doc_ids": t.tensor([0, 0], dtype=t.int32),
            "segment_indices": t.tensor([0, 1], dtype=t.int32),
            "segment_positions": t.tensor([0, 3], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 3], dtype=t.int32),
            "segment_char_ends": t.tensor([3, 6], dtype=t.int32),
            "doc_token_counts": t.tensor([6], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 2,
            "layer": 8,
            "context_window": 0,
            "pooling": "max_pool",
            "segment_mode": TURN_MODE.value,
            "segment_size": 4,
        },
        feature_dir / "shard_00000_features",
    )
    output = tmp_path / "feature_statistics.safetensors"
    compute_statistics(
        str(feature_dir),
        str(output),
        streaming=False,
        skip_per_feature_detail=True,
        allow_degenerate_statistics=True,
    )
    assert load_statistics(output).metadata.segment_mode is TURN_MODE
