"""Pin the kept_indices contract of align_char_ranges_to_decoded_prefix.

Full-sequence routing, target-segment helpers, and multiseg generation use
``kept_indices`` to filter parallel arrays (feature ids, magnitudes, and
``SegmentFeatures`` lists) in lockstep with returned char ranges.
"""

from __future__ import annotations

from steering.segmentation import align_char_ranges_to_decoded_prefix


def test_kept_indices_equal_returns_full_range():
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3], char_ends=[3, 6], decoded_len=6,
    )
    assert kept == [0, 1]


def test_kept_indices_forward_extension_returns_full_range():
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3], char_ends=[3, 6], decoded_len=8,  # delta=2 within window
    )
    assert kept == [0, 1]


def test_kept_indices_reverse_clip_no_filtering_returns_full_range():
    """All segments overlap decoded prefix; last is clipped."""
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3], char_ends=[3, 10], decoded_len=6,  # last clipped
    )
    assert kept == [0, 1]


def test_kept_indices_reverse_filter_drops_tail_segments():
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3, 6, 9], char_ends=[3, 6, 9, 12], decoded_len=4,
    )
    # Indices 0 (start=0), 1 (start=3) keep; 2 (start=6), 3 (start=9) drop.
    assert kept == [0, 1]


def test_kept_indices_reverse_filter_drops_only_after_clip_target():
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 5, 10], char_ends=[5, 10, 15], decoded_len=7,
    )
    # Index 0 (start=0), 1 (start=5) keep; 2 (start=10) drop.
    assert kept == [0, 1]
    assert kept[-1] == 1


def test_kept_indices_lockstep_filter_example():
    """Demonstrate the lockstep filter pattern used by migration consumers."""
    starts = [0, 5, 10, 15]
    ends = [5, 10, 15, 20]
    fids = ["seg0", "seg1", "seg2", "seg3"]  # caller's parallel array
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=12,
    )
    # Lockstep filter:
    new_fids = [fids[i] for i in kept]
    assert kept == [0, 1, 2]
    assert new_s == [0, 5, 10]
    assert new_e == [5, 10, 12]  # last clipped
    assert new_fids == ["seg0", "seg1", "seg2"]
