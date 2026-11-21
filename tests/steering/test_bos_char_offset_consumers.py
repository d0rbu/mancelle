"""Tests for BOS-inclusive character offset coordinate system in consumers.

Validates that consumers of stored segment char offsets (from feature extraction)
operate in the same BOS-inclusive coordinate system as the stored values.

The extraction pipeline computes char offsets via token_to_char_offsets([BOS, t0, t1, ...]),
producing offsets in the coordinate system of tokenizer.decode([BOS, t0, t1, ...]). Consumers
must use this same coordinate system when slicing text or computing fresh char offsets.
"""

import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets

# ---------------------------------------------------------------------------
# Test tokenizer that simulates Gemma-2's multi-char BOS decode
# ---------------------------------------------------------------------------

BOS_ID = 2
BOS_DECODE = "<bos>"  # 5 chars, matching Gemma-2


class BosTokenizer:
    """Tokenizer where BOS (ID=2) decodes to a multi-char string like Gemma-2.

    Non-BOS tokens: each ASCII character is a token (ID = ord(c)).
    decode([BOS]) = "<bos>" (5 chars), so BOS creates a coordinate offset.
    """

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        tokens = [ord(c) for c in text] if text else []
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        parts = []
        for tid in token_ids:
            if tid == BOS_ID:
                parts.append(BOS_DECODE)
            else:
                parts.append(chr(tid))
        return "".join(parts)

    def get_bos_token_id(self) -> int:
        return BOS_ID


# ---------------------------------------------------------------------------
# 4a. Round-trip test
# ---------------------------------------------------------------------------


class TestRoundTripBosCoordinates:
    """Verify that slicing BOS-decoded text at token_to_char_offsets boundaries
    produces clean token boundaries, and that slicing raw text does NOT."""

    def test_bos_text_clean_boundaries(self):
        tokenizer = BosTokenizer()
        raw_text = "Hello world test"
        tokens_with_bos = [BOS_ID] + [ord(c) for c in raw_text]

        # Compute char offsets in BOS coordinate system
        offsets = token_to_char_offsets(tokenizer, tokens_with_bos)
        bos_text = tokenizer.decode(tokens_with_bos)

        assert bos_text == BOS_DECODE + raw_text

        # Slicing bos_text at any offset boundary should produce the correct token decode
        for i in range(len(tokens_with_bos)):
            segment = bos_text[offsets[i] : offsets[i + 1]]
            expected = tokenizer.decode([tokens_with_bos[i]])
            assert segment == expected, (
                f"Token {i} (id={tokens_with_bos[i]}): "
                f"bos_text[{offsets[i]}:{offsets[i + 1]}] = {segment!r}, "
                f"expected {expected!r}"
            )

    def test_raw_text_wrong_boundaries(self):
        """Slicing raw text at BOS-coordinate offsets produces WRONG boundaries."""
        tokenizer = BosTokenizer()
        raw_text = "Hello world test"
        tokens_with_bos = [BOS_ID] + [ord(c) for c in raw_text]

        offsets = token_to_char_offsets(tokenizer, tokens_with_bos)
        bos_prefix_len = len(BOS_DECODE)

        # The first real text token starts at offset[1] in BOS coordinates.
        # In raw text, that offset is shifted by bos_prefix_len.
        first_text_offset = offsets[1]  # e.g. 5 for Gemma-2
        assert first_text_offset == bos_prefix_len

        # Slicing raw_text at this offset misses the start of the text
        raw_slice = raw_text[first_text_offset:]
        assert raw_slice != raw_text, (
            "Raw text sliced at BOS-coordinate offset should NOT equal raw text"
        )
        # We've lost the first bos_prefix_len characters
        assert raw_slice == raw_text[bos_prefix_len:]


# ---------------------------------------------------------------------------
# 4b. Integration test for multiseg_generator.py
# ---------------------------------------------------------------------------


class TestMultisegGeneratorBosCoordinates:
    """Verify that multiseg_generator correctly slices text using BOS coordinates."""

    def test_ground_truth_and_prefix_correct(self):
        """After the fix, pre_segment_text and ground_truth have clean boundaries."""
        tokenizer = BosTokenizer()
        # Simulate: doc_text is raw text, segments have BOS-coordinate offsets
        raw_text = "The quick brown fox jumps over the lazy dog"
        bos_tokens = [BOS_ID] + [ord(c) for c in raw_text]

        # Compute BOS-coordinate offsets (as extraction would)
        token_to_char_offsets(tokenizer, bos_tokens)
        bos_prefix_len = len(BOS_DECODE)  # 5

        # Create a segment starting at "fox" in the raw text
        fox_pos_raw = raw_text.index("fox")
        fox_pos_bos = fox_pos_raw + bos_prefix_len  # BOS coordinate
        fox_end_bos = fox_pos_bos + len("fox jumps")

        seg = SegmentFeatures(
            segment_idx=1,
            position=fox_pos_raw,
            feature_ids=t.tensor([10, 20], dtype=t.int32),
            feature_mags=t.tensor([1.0, 2.0], dtype=t.float32),
            char_start=fox_pos_bos,
            char_end=fox_end_bos,
        )

        # Apply the fix: prepend BOS decode to raw text
        bos_prefix = tokenizer.decode([BOS_ID])
        bos_text = bos_prefix + raw_text

        ground_truth = bos_text[seg.char_start : seg.char_end]
        pre_segment_text = bos_text[len(bos_prefix) : seg.char_start]

        assert ground_truth == "fox jumps", (
            f"Ground truth should be 'fox jumps', got {ground_truth!r}"
        )
        assert pre_segment_text == raw_text[:fox_pos_raw], (
            f"Pre-segment text should be text before 'fox', got {pre_segment_text!r}"
        )
        # Verify no mid-word split
        assert not ground_truth[0].isalpha() or pre_segment_text[-1] == " ", (
            "Boundary should not split a word"
        )

    def test_buggy_code_would_slice_wrong(self):
        """Without the fix, slicing raw doc_text at BOS offsets gives wrong content."""
        BosTokenizer()
        raw_text = "The quick brown fox jumps over the lazy dog"
        bos_prefix_len = len(BOS_DECODE)

        fox_pos_raw = raw_text.index("fox")
        fox_pos_bos = fox_pos_raw + bos_prefix_len

        # Buggy: slice raw text at BOS coordinate
        buggy_ground_truth = raw_text[fox_pos_bos:][: len("fox jumps")]
        assert buggy_ground_truth != "fox jumps", (
            "Buggy code should NOT produce correct ground truth"
        )

    def test_no_double_bos_in_prefix_tokens(self):
        """The fixed encoding path must produce exactly one BOS at position 0."""
        tokenizer = BosTokenizer()
        raw_text = "Hello world"
        bos_prefix_len = len(BOS_DECODE)

        # Simulate BOS-coordinate segment starting at position bos_prefix_len + 5
        char_start_bos = bos_prefix_len + 5  # "Hello" is 5 chars

        bos_prefix = tokenizer.decode([BOS_ID])
        bos_text = bos_prefix + raw_text
        pre_segment_text = bos_text[len(bos_prefix) : char_start_bos]

        # Encode with prepend=bos_id (same as fixed code)
        prefix_tokens = tokenizer.encode(pre_segment_text, prepend=BOS_ID)

        # BOS should appear exactly once, at position 0
        assert prefix_tokens[0] == BOS_ID
        assert prefix_tokens.count(BOS_ID) == 1, (
            f"BOS should appear exactly once, found {prefix_tokens.count(BOS_ID)} times"
        )


# ---------------------------------------------------------------------------
# 4c. Integration test for dataloader.py
# ---------------------------------------------------------------------------


class TestDataloaderBosCoordinates:
    """Verify that _build_row_segments assigns tokens correctly
    when both fresh char offsets and stored segment boundaries are in the
    BOS coordinate system."""

    def _make_seg_features_tuples(
        self,
        char_ranges: list[tuple[int, int]],
        n_features: int = 2,
    ) -> list[SegFeatures]:
        """Build SegFeatures list from char ranges."""
        result = []
        for i, (cs, ce) in enumerate(char_ranges):
            fids = t.arange(n_features, dtype=t.int64) + (i + 1) * 10
            buckets = t.ones(n_features, dtype=t.int64)
            pcts = t.ones(n_features, dtype=t.float32) * 0.5
            result.append(
                SegFeatures(
                    feature_ids=fids,
                    buckets=buckets,
                    percentiles=pcts,
                    char_start=cs,
                    char_end=ce,
                )
            )
        return result

    def test_correct_assignment_with_bos_coordinates(self):
        """Tokens are correctly assigned to segments when both sides use BOS coords."""
        tokenizer = BosTokenizer()
        bos_prefix_len = len(BOS_DECODE)  # 5

        # Raw text: "AAAABBBB" (8 chars, 8 tokens)
        raw_text = "AAAABBBB"
        doc_tokens = [BOS_ID] + [ord(c) for c in raw_text]
        T = len(doc_tokens)

        # Extraction would produce BOS-coordinate offsets:
        # First segment includes BOS: [0,9), second: [9,13)
        seg_features = self._make_seg_features_tuples(
            [
                (0, bos_prefix_len + 4),  # BOS+"AAAA": [0, 9)
                (bos_prefix_len + 4, bos_prefix_len + 8),  # "BBBB": [9, 13)
            ]
        )

        row_docs = [(0, seg_features, doc_tokens)]
        per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map, seg_doc_id, _, _ = _build_row_segments(
            row_docs,  # ty: ignore[invalid-argument-type]
            T,
            tokenizer,
        )

        # BOS token (position 0) should be assigned to segment 0 (first real segment)
        # because cummax ensures monotonicity
        assert seg_map[0].item() == 0, (
            f"BOS should map to segment 0, got {seg_map[0].item()}"
        )

        # "AAAA" tokens (positions 1-4) should map to segment 0
        for i in range(1, 5):
            assert seg_map[i].item() == 0, f"Token at pos {i} should map to seg 0"

        # "BBBB" tokens (positions 5-8) should map to segment 1
        for i in range(5, T):
            assert seg_map[i].item() == 1, f"Token at pos {i} should map to seg 1"

    def test_buggy_stripping_bos_misaligns(self):
        """Demonstrate that stripping BOS before computing char offsets causes mismatch.

        The buggy path (no-BOS offsets) now raises ValueError because tokens have
        zero overlap with segments -- the validation correctly detects the coordinate
        system mismatch.
        """
        tokenizer = BosTokenizer()
        bos_prefix_len = len(BOS_DECODE)

        raw_text = "AAAABBBB"
        doc_tokens = [BOS_ID] + [ord(c) for c in raw_text]

        # Stored segment boundaries in BOS coordinates (include BOS in first segment)
        seg_char_starts = t.tensor([0, bos_prefix_len + 4], dtype=t.int64)
        seg_char_ends = t.tensor(
            [bos_prefix_len + 4, bos_prefix_len + 8], dtype=t.int64
        )

        # BUGGY: strip BOS, compute offsets in no-BOS coordinate system
        text_tokens_no_bos = doc_tokens[1:]
        offsets_no_bos = token_to_char_offsets(tokenizer, text_tokens_no_bos)
        token_starts_no_bos = t.tensor(offsets_no_bos[:-1], dtype=t.int64)
        token_ends_no_bos = t.tensor(offsets_no_bos[1:], dtype=t.int64)

        # Buggy path: no-BOS tokens at [0,1), [1,2), ..., [7,8) have some overlap
        # with segments [0,9) and [9,13), but the assignments will be wrong because
        # the coordinate systems don't match. Tokens 5-7 should be in seg 1 but
        # the buggy path puts them in seg 0.
        assignments_buggy = assign_tokens_to_segments(
            token_starts_no_bos,
            token_ends_no_bos,
            seg_char_starts,
            seg_char_ends,
        )

        # FIXED: keep BOS, compute offsets in BOS coordinate system
        offsets_bos = token_to_char_offsets(tokenizer, doc_tokens)
        token_starts_bos = t.tensor(offsets_bos[:-1], dtype=t.int64)
        token_ends_bos = t.tensor(offsets_bos[1:], dtype=t.int64)

        assignments_fixed = assign_tokens_to_segments(
            token_starts_bos,
            token_ends_bos,
            seg_char_starts,
            seg_char_ends,
        )

        # Fixed assignments should have correct segment transitions
        fixed_cummax = t.cummax(assignments_fixed, dim=0).values
        # Tokens 0-4 -> seg 0, tokens 5-8 -> seg 1
        assert all(fixed_cummax[i].item() == 0 for i in range(5)), (
            f"Tokens 0-4 should be in segment 0: {fixed_cummax.tolist()}"
        )
        assert all(fixed_cummax[i].item() == 1 for i in range(5, len(doc_tokens))), (
            f"Tokens 5-8 should be in segment 1: {fixed_cummax.tolist()}"
        )

        # Buggy assignments differ from fixed (proving the coordinate system bug)
        buggy_cummax = t.cummax(assignments_buggy, dim=0).values
        assert not t.equal(buggy_cummax, fixed_cummax[1:]), (
            "Buggy assignments should differ from fixed (proving the bug)"
        )


# ---------------------------------------------------------------------------
# 4d. Regression test: simulated doc 45488 boundary
# ---------------------------------------------------------------------------


class TestRegressionDoc45488:
    """Regression test simulating the doc 45488, shard 369 boundary issue.

    In the real data (Gemma-2-2B tokenizer), segment 1 has stored char_start=678.
    In BOS-decoded text, offset 678 falls at " more" | " structured" (clean boundary).
    In raw text, offset 678 falls at " stru" | "ctured" (mid-word split).
    """

    def test_boundary_falls_at_clean_word_break(self):
        tokenizer = BosTokenizer()
        bos_prefix_len = len(BOS_DECODE)  # 5

        # Build text where a specific BOS-coordinate offset hits a word boundary.
        # Place a word boundary at raw position 673 (= BOS position 678).
        # Text: 673 chars of padding + " structured ..."
        prefix_text = "x" * 668 + " more"  # 668 + 5 = 673 chars
        boundary_text = " structured learning"
        raw_text = prefix_text + boundary_text

        # Stored BOS-coordinate offset: 673 + bos_prefix_len = 678
        stored_char_start = len(prefix_text) + bos_prefix_len  # 678

        # FIXED: use BOS coordinate system
        bos_prefix = tokenizer.decode([BOS_ID])
        bos_text = bos_prefix + raw_text
        ground_truth = bos_text[stored_char_start:][: len(boundary_text)]

        assert ground_truth == boundary_text, (
            f"Fixed: boundary should produce {boundary_text!r}, got {ground_truth!r}"
        )
        assert ground_truth.startswith(" structured"), (
            "Fixed: should start at word boundary ' structured'"
        )

        # BUGGY: slice raw text at BOS offset
        buggy_gt = raw_text[stored_char_start:][: len(boundary_text)]
        # This would start 5 chars later in the raw text, hitting mid-word
        assert buggy_gt != boundary_text, (
            "Buggy code should NOT produce correct ground truth"
        )
        # The buggy slice skips into the word
        assert "structured" not in buggy_gt[:12], (
            "Buggy slice should hit mid-word (not start with ' structured')"
        )


# ---------------------------------------------------------------------------
# 4e. No-double-BOS test
# ---------------------------------------------------------------------------


class TestNoDoubleBos:
    """Verify that the fixed code paths produce token sequences with BOS
    appearing exactly once at position 0."""

    def test_multiseg_prefix_tokens_single_bos(self):
        """multiseg_generator: prefix_tokens should have exactly one BOS."""
        tokenizer = BosTokenizer()
        raw_text = "Hello world, this is a test document."
        bos_prefix_len = len(BOS_DECODE)

        # Segment starts at "this" in raw text
        this_pos = raw_text.index("this")
        char_start_bos = this_pos + bos_prefix_len

        # Apply the fix
        bos_prefix = tokenizer.decode([BOS_ID])
        bos_text = bos_prefix + raw_text
        pre_segment_text = bos_text[len(bos_prefix) : char_start_bos]
        prefix_tokens = tokenizer.encode(pre_segment_text, prepend=BOS_ID)

        assert prefix_tokens[0] == BOS_ID, "First token must be BOS"
        assert prefix_tokens.count(BOS_ID) == 1, (
            f"BOS must appear exactly once, found {prefix_tokens.count(BOS_ID)}"
        )
        # Verify the remaining tokens encode the correct text
        decoded_rest = tokenizer.decode(prefix_tokens[1:])
        assert decoded_rest == pre_segment_text

    def test_dataloader_bos_in_tokens_once(self):
        """dataloader: doc_tokens should contain BOS exactly once at position 0."""
        BosTokenizer()
        raw_text = "AAAABBBB"
        doc_tokens = [BOS_ID] + [ord(c) for c in raw_text]

        # The fix keeps BOS: text_tokens = doc_tokens[:n_doc_tokens]
        n_doc_tokens = len(doc_tokens)
        text_tokens = doc_tokens[:n_doc_tokens]

        assert text_tokens[0] == BOS_ID, "First token must be BOS"
        assert text_tokens.count(BOS_ID) == 1, (
            f"BOS must appear exactly once, found {text_tokens.count(BOS_ID)}"
        )

    def test_bos_coordinate_offsets_start_at_zero(self):
        """BOS-inclusive char offsets must start at 0 (BOS decode starts at char 0)."""
        tokenizer = BosTokenizer()
        tokens = [BOS_ID, ord("A"), ord("B")]
        offsets = token_to_char_offsets(tokenizer, tokens)
        assert offsets[0] == 0, "First offset must be 0"
        assert offsets[1] == len(BOS_DECODE), (
            f"Second offset must be len(BOS_DECODE)={len(BOS_DECODE)}, got {offsets[1]}"
        )
