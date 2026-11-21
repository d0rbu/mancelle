"""Strict tests for full-document character coordinate invariants.

The extraction pipeline stores character boundaries per segment. For rendered
SFT and pretraining documents alike, boundaries are document-relative positions
over the extracted text prefix, not assistant-relative spans.
"""

from steering.extract_features import _precompute_batch_cpu_work
from steering.segmentation import SegmentMode


class SimpleTokenizer:
    """Tokenizer where each ASCII character is a separate token."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids)

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}


class TestFullDocumentCharCoordinates:
    """Char offsets must be document-relative over the kept text prefix."""

    def test_no_truncation_offsets_start_at_zero(self):
        """When a document fits in the context window, char offsets start at 0."""
        tokenizer = SimpleTokenizer()
        text = "Hello world test"  # 16 chars = 16 tokens
        token_ids = tokenizer.encode(text)
        batch = [(token_ids, len(token_ids))]  # no truncation

        results = _precompute_batch_cpu_work(
            batch,
            tokenizer,
            SegmentMode.fixed_tokens,
            8,
        )

        _, _, char_starts, char_ends = results[0]
        assert char_starts[0] == 0, (
            "First segment must start at char 0 when no truncation"
        )
        assert char_ends[-1] == len(text), (
            f"Last segment must end at {len(text)}, got {char_ends[-1]}"
        )

    def test_head_truncation_offsets_start_at_zero(self):
        """When head-truncated, char offsets still start at 0."""
        tokenizer = SimpleTokenizer()
        # 100-char text, context_window=40 -> keep first 40 tokens
        text = "A" * 40 + "B" * 60  # head=40 'A's, tail=60 'B's
        token_ids = tokenizer.encode(text)
        context_window = 40
        kept_ids = token_ids[:context_window]

        batch = [(kept_ids, len(token_ids))]

        results = _precompute_batch_cpu_work(
            batch,
            tokenizer,
            SegmentMode.fixed_tokens,
            16,
        )

        _, _, char_starts, char_ends = results[0]

        # With head truncation, first segment starts at 0
        assert char_starts[0] == 0
        # Last segment ends at 40 (only kept first 40 chars)
        assert char_ends[-1] == 40

    def test_head_truncation_offsets_cover_correct_text(self):
        """Char offsets should index into the kept prefix of the original text."""
        tokenizer = SimpleTokenizer()
        text = "AAAA" + "BBBB" + "CCCC" + "DDDD" + "EEEE"  # 20 chars
        token_ids = tokenizer.encode(text)
        context_window = 12  # keep first 12 -> "AAAABBBBCCCC"
        kept_ids = token_ids[:context_window]

        batch = [(kept_ids, len(token_ids))]

        results = _precompute_batch_cpu_work(
            batch,
            tokenizer,
            SegmentMode.fixed_tokens,
            4,
        )

        _, _, char_starts, char_ends = results[0]

        # 3 segments of 4 tokens each, starting at char 0
        assert char_starts == [0, 4, 8]
        assert char_ends == [4, 8, 12]

        for start, end in zip(char_starts, char_ends, strict=True):
            segment_text = text[start:end]
            assert len(segment_text) == 4

    def test_segments_tile_kept_region(self):
        """Segments must tile the kept character region with no gaps or overlaps."""
        tokenizer = SimpleTokenizer()
        text = "X" * 200
        context_window = 128
        token_ids = tokenizer.encode(text)
        kept_ids = token_ids[:context_window]

        batch = [(kept_ids, len(token_ids))]

        results = _precompute_batch_cpu_work(
            batch,
            tokenizer,
            SegmentMode.fixed_tokens,
            32,
        )

        _, _, char_starts, char_ends = results[0]

        for i in range(1, len(char_starts)):
            assert char_starts[i] == char_ends[i - 1]

        assert char_starts[0] == 0
        assert char_ends[-1] == context_window
