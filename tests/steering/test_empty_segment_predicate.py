import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from errors import EmptyExtractionSegmentError, ExtractionError
from steering.feature_io import empty_segment_indices


def test_no_empty_segments_returns_empty_list():
    assert empty_segment_indices(t.tensor([0, 2, 5, 7], dtype=t.int64)) == []


def test_leading_middle_trailing_empties_detected():
    # segments: [0,0)=empty, [0,3), [3,3)=empty, [3,5), [5,5)=empty
    offsets = t.tensor([0, 0, 3, 3, 5, 5], dtype=t.int64)
    assert empty_segment_indices(offsets) == [0, 2, 4]


def test_n_segments_zero_is_exempt():
    assert empty_segment_indices(t.tensor([0], dtype=t.int64)) == []
    assert empty_segment_indices(t.tensor([], dtype=t.int64)) == []


@settings(derandomize=True, max_examples=200)
@given(st.lists(st.integers(min_value=0, max_value=4), min_size=0, max_size=30))
def test_predicate_matches_diff_zero(counts):
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    off = t.tensor(offsets, dtype=t.int64)
    expected = [i for i, c in enumerate(counts) if c == 0]
    assert empty_segment_indices(off) == expected


def test_error_type_is_extraction_error():
    assert issubclass(EmptyExtractionSegmentError, ExtractionError)
