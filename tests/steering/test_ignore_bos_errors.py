"""Unit tests for the BosOnlySegmentError typed exception."""

from errors import BosOnlySegmentError, ExtractionError, NanoGPTError


def test_bos_only_segment_error_is_extraction_error():
    assert issubclass(BosOnlySegmentError, ExtractionError)
    assert issubclass(BosOnlySegmentError, NanoGPTError)


def test_bos_only_segment_error_message_roundtrips():
    err = BosOnlySegmentError("segment 0 is all <bos>")
    assert "segment 0 is all <bos>" in str(err)


def test_bos_only_segment_error_is_raisable_and_catchable_as_extraction_error():
    try:
        raise BosOnlySegmentError("x")
    except ExtractionError as e:
        assert isinstance(e, BosOnlySegmentError)
