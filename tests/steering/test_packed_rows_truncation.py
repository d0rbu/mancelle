"""Tests for packed rows truncation when precomputed boundaries don't cover all tokens.

When the nanochat tokenizer is more efficient than Gemma (e.g. for number-heavy
documents), 2048 nanochat tokens can cover MORE text than 2048 Gemma tokens.
The precomputed token boundaries only cover the Gemma context window, so the
tail nanochat tokens have no feature coverage.

The fix truncates documents to the last precomputed boundary end, with tolerance
checks to distinguish expected tail gaps from fundamentally broken boundaries.
"""

from dataclasses import dataclass, field

import pytest
import torch as t
from loguru import logger

from scripts.preprocess.precompute_packed_rows import (
    assert_boundaries_covered,
    truncate_to_boundary_coverage,
)
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.refined_types import PositiveInt


@dataclass
class _LogRecord:
    level: str
    message: str


@dataclass
class LogCapture:
    records: list[_LogRecord] = field(default_factory=list)

    def __call__(self, message):
        record = message.record
        self.records.append(_LogRecord(
            level=record["level"].name,
            message=record["message"],
        ))


@pytest.fixture()
def log_capture():
    """Capture loguru log records."""
    capture = LogCapture()
    handler_id = logger.add(capture, format="{message}", level="DEBUG")
    yield capture
    logger.remove(handler_id)


def _make_seg_features(
    boundaries: list[tuple[int, int]],
    n_features: int = 3,
) -> list[SegFeatures]:
    """Create synthetic SegFeatures with precomputed token boundaries.

    Args:
        boundaries: List of (token_start, token_end) pairs.
        n_features: Number of dummy features per segment.
    """
    segs = []
    for tok_start, tok_end in boundaries:
        segs.append(SegFeatures(
            feature_ids=t.arange(n_features, dtype=t.int64),
            buckets=t.zeros(n_features, dtype=t.int64),
            percentiles=t.ones(n_features, dtype=t.float32) * 0.5,
            char_start=tok_start * 5,  # dummy char offsets
            char_end=tok_end * 5,
            token_start=tok_start,
            token_end=tok_end,
        ))
    return segs


CONTEXT_LEN = 2048
SEGMENT_SIZE: PositiveInt = PositiveInt.parse(128)
# max_gap = int(doc_token_count * 0.42)
# For full-context docs (2048 tokens): 860
MAX_GAP_FULL = int(CONTEXT_LEN * 0.42)  # 860


class TestFullyCovered:
    """Test case (a): precomputed boundaries cover all tokens."""

    def test_no_truncation_when_fully_covered(self):
        """Document with 2048 tokens fully covered by boundaries → no truncation."""
        boundaries = [(i * SEGMENT_SIZE, (i + 1) * SEGMENT_SIZE) for i in range(16)]
        # Last boundary ends at 2048, matching doc length
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == CONTEXT_LEN

    def test_no_truncation_short_doc_fully_covered(self):
        """Short document (500 tokens) fully covered → no truncation."""
        boundaries = [(i * SEGMENT_SIZE, (i + 1) * SEGMENT_SIZE) for i in range(4)]
        # Last boundary ends at 512, doc has 500 tokens → covered
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(500))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == 500

    def test_no_truncation_when_boundaries_exceed_doc_len(self):
        """Boundaries extend past doc length → no truncation needed."""
        boundaries = [(i * SEGMENT_SIZE, (i + 1) * SEGMENT_SIZE) for i in range(16)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(1500))  # shorter than coverage

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == 1500


class TestGapWithinTolerance:
    """Test case (b): gap within 2*segment_size of context window end."""

    def test_truncates_to_last_boundary_end(self):
        """Doc 240 scenario: 2048 tokens, boundaries end at 1938 → truncate to 1938."""
        # 15 full segments + partial last = 1938
        boundaries = [(i * SEGMENT_SIZE, (i + 1) * SEGMENT_SIZE) for i in range(15)]
        boundaries.append((15 * SEGMENT_SIZE, 1938))
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == 1938

    def test_gap_above_50pct_logs_error(self, log_capture):
        """Gap=500 (500/716=0.70 > 0.5) → error logged."""
        last_end = CONTEXT_LEN - 500  # 1548
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )

        error_msgs = [r for r in log_capture.records if r.level == "ERROR"]
        assert len(error_msgs) >= 1
        assert "400" in error_msgs[0].message or "gap" in error_msgs[0].message.lower()

    def test_gap_above_25pct_logs_warning(self, log_capture):
        """Gap=250 (250/716=0.35 > 0.25) → warning logged."""
        last_end = CONTEXT_LEN - 250  # 1798
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )

        warn_msgs = [r for r in log_capture.records if r.level == "WARNING"]
        assert len(warn_msgs) >= 1
        assert "150" in warn_msgs[0].message or "gap" in warn_msgs[0].message.lower()

    def test_small_gap_no_log(self, log_capture):
        """Gap=100 (100/512=0.20 < 0.25) → no warning or error."""
        last_end = CONTEXT_LEN - 100  # 1948
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )

        log_msgs = [r for r in log_capture.records if r.level in ("WARNING", "ERROR")]
        assert len(log_msgs) == 0

    def test_gap_exactly_max_gap_logs_error(self, log_capture):
        """Gap=512 (exactly max_gap) → error logged."""
        last_end = CONTEXT_LEN - MAX_GAP_FULL  # 1536
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )

        error_msgs = [r for r in log_capture.records if r.level == "ERROR"]
        assert len(error_msgs) >= 1

    def test_truncation_preserves_token_values(self):
        """Truncated tokens are the prefix of the original."""
        last_end = 1938
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert result == tokens[:1938]


class TestGapExceedsTolerance:
    """Test case (c): boundaries don't reach close to context window end."""

    def test_raises_when_gap_exceeds_max(self):
        """Gap=948 > max_gap=860 → ValueError."""
        boundaries = [(0, 1100)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        with pytest.raises(ValueError, match="max_gap"):
            truncate_to_boundary_coverage(
                tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
            )

    def test_raises_with_diagnostic_info(self):
        """ValueError message includes last_precomputed_end, doc_token_count, threshold."""
        boundaries = [(0, 1000)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        with pytest.raises(ValueError) as exc_info:
            truncate_to_boundary_coverage(
                tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
            )
        msg = str(exc_info.value)
        assert "1000" in msg  # last_precomputed_end
        assert "2048" in msg  # doc_token_count
        assert "max_gap=" in msg


class TestShortDocumentWithGap:
    """Test case: short documents where gap detection must still work."""

    def test_short_doc_tiny_gap_truncates(self):
        """500-token doc, boundaries cover 499 → gap=1 ≤ 4 → silently truncate."""
        boundaries = [(0, 499)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(500))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == 499

    def test_short_doc_large_gap_raises(self):
        """500-token doc, boundaries cover 250 → gap=250 > max_gap(210) → ValueError."""
        boundaries = [(0, 250)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(500))

        with pytest.raises(ValueError, match="max_gap"):
            truncate_to_boundary_coverage(
                tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
            )

    def test_short_doc_fully_covered_ok(self):
        """500-token doc, boundaries cover 500 → fully covered, no issue."""
        boundaries = [(0, 500)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(500))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == 500


class TestNoneSegFeatures:
    """Documents with no precomputed segments should pass through unchanged."""

    def test_none_seg_features_no_truncation(self):
        """No seg_features → return tokens unchanged."""
        tokens = list(range(CONTEXT_LEN))
        result = truncate_to_boundary_coverage(
            tokens, None, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == CONTEXT_LEN

    def test_empty_seg_features_no_truncation(self):
        """Empty seg_features list → return tokens unchanged."""
        tokens = list(range(CONTEXT_LEN))
        result = truncate_to_boundary_coverage(
            tokens, [], CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == CONTEXT_LEN


class TestNoPrecomputedBoundaries:
    """Segments without token_start/token_end should raise by default."""

    def test_missing_token_boundaries_raises(self):
        """Segments with token_start=None → ValueError."""
        segs = [SegFeatures(
            feature_ids=t.arange(3, dtype=t.int64),
            buckets=t.zeros(3, dtype=t.int64),
            percentiles=t.ones(3, dtype=t.float32) * 0.5,
            char_start=0,
            char_end=1000,
            token_start=None,
            token_end=None,
        )]
        tokens = list(range(CONTEXT_LEN))

        with pytest.raises(ValueError, match="token_start=None"):
            truncate_to_boundary_coverage(
                tokens, segs, CONTEXT_LEN, SEGMENT_SIZE,
            )

    def test_missing_boundary_on_middle_segment_raises(self):
        """First segment has boundaries but second doesn't → ValueError."""
        segs = [
            SegFeatures(
                feature_ids=t.arange(3, dtype=t.int64),
                buckets=t.zeros(3, dtype=t.int64),
                percentiles=t.ones(3, dtype=t.float32) * 0.5,
                char_start=0, char_end=500,
                token_start=0, token_end=128,
            ),
            SegFeatures(
                feature_ids=t.arange(3, dtype=t.int64),
                buckets=t.zeros(3, dtype=t.int64),
                percentiles=t.ones(3, dtype=t.float32) * 0.5,
                char_start=500, char_end=1000,
                token_start=None, token_end=None,
            ),
        ]
        tokens = list(range(CONTEXT_LEN))

        with pytest.raises(ValueError, match="segment 1/2"):
            truncate_to_boundary_coverage(
                tokens, segs, CONTEXT_LEN, SEGMENT_SIZE,
            )



class TestEdgeCases:
    """Edge cases for truncation logic."""

    def test_doc_longer_than_context_len_truncated_first(self):
        """Doc with 5000 tokens → truncated to context_len first, then boundary check."""
        boundaries = [(i * SEGMENT_SIZE, (i + 1) * SEGMENT_SIZE) for i in range(16)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(5000))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == CONTEXT_LEN

    def test_gap_exactly_at_max(self):
        """Gap exactly equals max_gap (512 for 2048-token doc) → within tolerance."""
        last_end = CONTEXT_LEN - MAX_GAP_FULL  # 1536
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == last_end

    def test_gap_one_over_max_raises(self):
        """Gap = max_gap + 1 → ValueError."""
        last_end = CONTEXT_LEN - MAX_GAP_FULL - 1  # 1535
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        with pytest.raises(ValueError, match="max_gap"):
            truncate_to_boundary_coverage(
                tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
            )

    def test_out_of_order_segments_uses_max(self):
        """Segments out of order — max() finds the true last boundary, not [-1]."""
        # Second segment ends earlier than the first
        boundaries = [(128, 1938), (0, 128)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        # max(token_end) = 1938, so truncation to 1938
        assert len(result) == 1938

    def test_exact_boundary_doc_equals_context_len(self):
        """doc_token_count == last_precomputed_end == context_len → fully covered."""
        boundaries = [(i * SEGMENT_SIZE, (i + 1) * SEGMENT_SIZE) for i in range(16)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
        )
        assert len(result) == CONTEXT_LEN
        assert result == tokens


class TestDocIdentityInLogs:
    """Log messages must include shard/doc identity for debugging."""

    def test_error_log_includes_doc_id(self, log_capture):
        """Error log contains the doc_id string."""
        last_end = CONTEXT_LEN - 500  # gap=500, > 0.5*716 → error
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
            doc_id="shard_00000:240",
        )

        error_msgs = [r for r in log_capture.records if r.level == "ERROR"]
        assert len(error_msgs) >= 1
        assert "shard_00000:240" in error_msgs[0].message

    def test_warning_log_includes_doc_id(self, log_capture):
        """Warning log contains the doc_id string."""
        last_end = CONTEXT_LEN - 250  # gap=250, > 0.25*716 → warning
        boundaries = [(0, last_end)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        truncate_to_boundary_coverage(
            tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
            doc_id="shard_00000:42",
        )

        warn_msgs = [r for r in log_capture.records if r.level == "WARNING"]
        assert len(warn_msgs) >= 1
        assert "shard_00000:42" in warn_msgs[0].message

    def test_valueerror_includes_doc_id(self):
        """ValueError message contains the doc_id string."""
        boundaries = [(0, 1000)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(CONTEXT_LEN))

        with pytest.raises(ValueError, match="shard_00000:99"):
            truncate_to_boundary_coverage(
                tokens, seg_features, CONTEXT_LEN, SEGMENT_SIZE,
                doc_id="shard_00000:99",
            )


class TestAssertBoundariesCovered:
    """Post-truncation assertion: first segment starts at 0, at least one in range.

    The relaxed assertion allows segments with token_end > n_tokens because
    _build_row_segments_char_level clamps tok_hi = min(sf.token_end, n_doc_tokens).
    """

    def test_passes_when_all_boundaries_within_tokens(self):
        """All segments within token count → no error."""
        boundaries = [(0, 128), (128, 256)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(256))
        assert_boundaries_covered(tokens, seg_features)

    def test_passes_with_none_seg_features(self):
        """None seg_features → no error."""
        tokens = list(range(CONTEXT_LEN))
        assert_boundaries_covered(tokens, None)

    def test_passes_with_empty_seg_features(self):
        """Empty seg_features → no error."""
        tokens = list(range(CONTEXT_LEN))
        assert_boundaries_covered(tokens, [])

    def test_passes_when_token_end_exceeds_token_count(self):
        """Segment token_end > len(tokens) is OK — downstream clamps."""
        boundaries = [(0, 300)]  # token_end=300 but only 256 tokens
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(256))
        # Should NOT raise — relaxed assertion allows this
        assert_boundaries_covered(tokens, seg_features)

    def test_fails_when_first_segment_not_at_zero(self):
        """First segment token_start != 0 → AssertionError."""
        boundaries = [(5, 128)]  # starts at 5, not 0
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(256))
        with pytest.raises(AssertionError, match="token_start"):
            assert_boundaries_covered(tokens, seg_features)

    def test_fails_when_no_segment_starts_within_range(self):
        """All segments start beyond token count → AssertionError."""
        boundaries = [(500, 600)]  # starts at 500, but only 256 tokens
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(256))
        with pytest.raises(AssertionError, match="token_start"):
            assert_boundaries_covered(tokens, seg_features)

    def test_passes_with_no_precomputed_boundaries(self):
        """Segments without token_start/token_end → no check needed."""
        segs = [SegFeatures(
            feature_ids=t.arange(3, dtype=t.int64),
            buckets=t.zeros(3, dtype=t.int64),
            percentiles=t.ones(3, dtype=t.float32) * 0.5,
            char_start=0, char_end=1000,
            token_start=None, token_end=None,
        )]
        tokens = list(range(100))
        assert_boundaries_covered(tokens, segs)
