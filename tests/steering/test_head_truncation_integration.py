"""Integration tests for head truncation through the full extraction pipeline.

Exercises truncation end-to-end using TinyHookedTransformer + TinySAE,
verifying that documents exceeding the context window are correctly
head-truncated and produce valid CSR output.

Run with:
    uv run pytest tests/test_head_truncation_integration.py -v --model-scale=tiny
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.extract_features import (
    PoolingMode,
    extract_document_features,
    process_shard,
)
from steering.feature_io import load_features
from steering.shard_validation import validate_shard
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
    model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")
    return model, sae


def _make_parquet(path: str, texts: list[str]):
    table = pa.table({"text": texts})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# process_shard with truncation
# ---------------------------------------------------------------------------


class TestTruncationThroughProcessShard:
    """Long documents through the full parquet -> CSR pipeline."""

    def test_long_doc_produces_valid_output(self, tiny_model, tmp_path):
        """200-char doc with context_window=10 is truncated and valid."""
        model, sae = tiny_model
        parquet_path = str(tmp_path / "long.parquet")
        _make_parquet(parquet_path, ["A" * 200])

        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(tmp_path),
            context_window=10,
        )
        assert result.n_docs == 1
        vr = validate_shard(result.output_path)
        assert vr.valid, f"Violations: {vr.violations}"

    def test_mixed_short_and_long_docs(self, tiny_model, tmp_path):
        """Batch with docs that fit and docs that don't."""
        model, sae = tiny_model
        texts = [
            "Short",  # 5 chars -> fits in window=10
            "A" * 200,  # 200 chars -> truncated
            "Hello World!",  # 12 chars -> truncated to 10
            "B" * 100,  # 100 chars -> truncated
        ]
        parquet_path = str(tmp_path / "mixed.parquet")
        _make_parquet(parquet_path, texts)

        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(tmp_path),
            context_window=10,
        )
        assert result.n_docs == 4
        vr = validate_shard(result.output_path)
        assert vr.valid, f"Violations: {vr.violations}"

    def test_truncated_doc_fewer_segments(self, tiny_model, tmp_path):
        """Smaller context_window -> fewer segments for the same document."""
        model, sae = tiny_model
        parquet_path = str(tmp_path / "doc.parquet")
        _make_parquet(parquet_path, ["X" * 200])

        small_dir = tmp_path / "small"
        small_dir.mkdir()
        large_dir = tmp_path / "large"
        large_dir.mkdir()

        result_small = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(small_dir),
            context_window=5,
            segment_size=4,
        )
        result_large = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(large_dir),
            context_window=100,
            segment_size=4,
        )

        data_small = load_features(result_small.output_path)
        data_large = load_features(result_large.output_path)
        assert data_small["n_segments"] < data_large["n_segments"]

    def test_sae_cap_limits_extraction(self, tiny_model, tmp_path):
        """model.cfg.n_ctx caps tokens even when context_window is large."""
        model, sae = tiny_model
        model.cfg.n_ctx = 10  # very small SAE cap

        parquet_path = str(tmp_path / "cap.parquet")
        _make_parquet(parquet_path, ["A" * 200])

        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(tmp_path),
            context_window=100,
        )
        assert result.n_docs == 1
        vr = validate_shard(result.output_path)
        assert vr.valid, f"Violations: {vr.violations}"

        # Segments should be very few (only 10 tokens kept)
        data = load_features(result.output_path)
        # With segment_size=128 (default), 10 tokens -> 1 segment
        assert data["n_segments"] <= 2

    def test_all_docs_short_no_truncation(self, tiny_model, tmp_path):
        """When all docs fit, output matches behavior without truncation logic."""
        model, sae = tiny_model
        texts = ["Hi", "Hello", "OK"]
        parquet_path = str(tmp_path / "short.parquet")
        _make_parquet(parquet_path, texts)

        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(tmp_path),
            context_window=100,
        )
        assert result.n_docs == 3
        vr = validate_shard(result.output_path)
        assert vr.valid

    def test_doc_token_counts_report_full_lengths(self, tiny_model, tmp_path):
        """doc_token_counts in CSR output should report full (pre-truncation) counts."""
        model, sae = tiny_model
        texts = ["A" * 200, "B" * 10]
        parquet_path = str(tmp_path / "counts.parquet")
        _make_parquet(parquet_path, texts)

        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=8,
            output_dir=str(tmp_path),
            context_window=10,
        )
        data = load_features(result.output_path)
        counts = data["doc_token_counts"].tolist()
        # TinyHookedTransformer: per-char tokenizer with BOS prepend, so
        # 200 chars -> 201 tokens, 10 chars -> 11 tokens
        assert counts[0] == 201
        assert counts[1] == 11

    def test_all_pooling_modes_with_truncation(self, tiny_model, tmp_path):
        """Every pooling mode works with truncated documents."""
        model, sae = tiny_model
        parquet_path = str(tmp_path / "modes.parquet")
        _make_parquet(parquet_path, ["A" * 200, "B" * 100])

        for mode in PoolingMode:
            out_dir = tmp_path / mode.value
            out_dir.mkdir(exist_ok=True)
            result = process_shard(
                parquet_path,
                model,
                sae,
                layer=8,
                output_dir=str(out_dir),
                context_window=10,
                pooling=mode,
            )
            assert result.n_docs == 2
            vr = validate_shard(result.output_path)
            assert vr.valid, f"{mode.value}: {vr.violations}"


# ---------------------------------------------------------------------------
# extract_document_features with truncation
# ---------------------------------------------------------------------------


class TestTruncationThroughExtractDocFeatures:
    """Head truncation through extract_document_features."""

    def test_truncated_features_match_head_only(self, tiny_model):
        """Features from truncated doc match features from just the head text."""
        model, sae = tiny_model
        full_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars

        # context_window=20 SAE tokens -> BOS + 19 chars kept
        head_text = full_text[:19]

        # Use explicit segment_size so both calls segment identically
        ids_trunc, mags_trunc, pos_trunc, _, _, _ = extract_document_features(
            model,
            sae,
            full_text,
            layer=8,
            context_window=20,
            segment_size=8,
        )
        ids_head, mags_head, pos_head, _, _, _ = extract_document_features(
            model,
            sae,
            head_text,
            layer=8,
            context_window=100,
            segment_size=8,
        )

        assert len(ids_trunc) == len(ids_head)
        for a, b in zip(ids_trunc, ids_head, strict=True):
            assert t.equal(a, b)
        for a, b in zip(mags_trunc, mags_head, strict=True):
            assert t.equal(a, b)

    def test_n_tokens_reports_full_count(self, tiny_model):
        """n_tokens is the full document token count, not the kept count."""
        model, sae = tiny_model
        text = "A" * 200  # 200 chars + BOS prepend = 201 SAE tokens

        _, _, _, _, _, n_tokens = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=20,
        )
        assert n_tokens == 201

    def test_determinism_with_truncation(self, tiny_model):
        """Truncated extraction is deterministic."""
        model, sae = tiny_model
        text = "A" * 200

        r1 = extract_document_features(model, sae, text, layer=8, context_window=20)
        r2 = extract_document_features(model, sae, text, layer=8, context_window=20)

        assert len(r1[0]) == len(r2[0])
        for a, b in zip(r1[0], r2[0], strict=True):
            assert t.equal(a, b)
        assert r1[5] == r2[5]

    def test_char_boundaries_within_head_region(self, tiny_model):
        """Char boundaries span only the kept head, not the full document."""
        model, sae = tiny_model
        text = "A" * 200

        _, _, _, char_starts, char_ends, _ = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=20,
            segment_size=4,
        )

        # context_window=20 SAE tokens -> 20 chars kept
        for cs, ce in zip(char_starts, char_ends, strict=True):
            assert cs >= 0
            assert ce <= 20, f"char_end {ce} exceeds head region (20 chars)"

    def test_no_truncation_when_doc_fits(self, tiny_model):
        """Short document: all tokens kept, features are the same as large window."""
        model, sae = tiny_model
        text = "Hello"  # 5 chars + BOS = 6 tokens

        r_small = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=6,  # exactly 6 tokens -> no truncation
        )
        r_large = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=100,
        )

        assert len(r_small[0]) == len(r_large[0])
        for a, b in zip(r_small[0], r_large[0], strict=True):
            assert t.equal(a, b)
        assert r_small[5] == r_large[5]  # n_tokens matches


# ---------------------------------------------------------------------------
# Char boundary invariants after truncation
# ---------------------------------------------------------------------------


class TestCharBoundaryInvariantsIntegration:
    """Property-like tests for char boundaries through the real pipeline."""

    def test_segments_tile_head_region(self, tiny_model):
        """Adjacent segments: char_ends[i] == char_starts[i+1]."""
        model, sae = tiny_model
        text = "The quick brown fox jumps over the lazy dog " * 5

        _, _, _, char_starts, char_ends, _ = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=20,
            segment_size=4,
        )

        for i in range(len(char_starts) - 1):
            assert char_ends[i] == char_starts[i + 1], (
                f"Gap at segment {i}: end={char_ends[i]}, next start={char_starts[i + 1]}"
            )

    def test_first_segment_starts_at_zero(self, tiny_model):
        """First segment always starts at char 0, truncated or not."""
        model, sae = tiny_model

        for text in ["Short", "A" * 200, "Hello World! " * 50]:
            _, _, _, char_starts, _, _ = extract_document_features(
                model,
                sae,
                text,
                layer=8,
                context_window=20,
                segment_size=4,
            )
            assert char_starts[0] == 0, f"Failed for text length {len(text)}"

    def test_no_negative_boundaries(self, tiny_model):
        """All char boundaries are non-negative and end >= start."""
        model, sae = tiny_model
        text = "A" * 300

        _, _, _, char_starts, char_ends, _ = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=20,
            segment_size=4,
        )

        for cs, ce in zip(char_starts, char_ends, strict=True):
            assert cs >= 0
            assert ce >= 0
            assert ce >= cs

    def test_last_segment_ends_at_kept_boundary(self, tiny_model):
        """Last segment's char_end equals the decoded length of kept tokens."""
        model, sae = tiny_model
        text = "A" * 200

        _, _, _, _, char_ends, _ = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=20,
            segment_size=4,
        )

        # context_window=20 SAE tokens -> 20 chars
        assert char_ends[-1] == 20


# ---------------------------------------------------------------------------
# Dual truncation (context_window + SAE cap both active)
# ---------------------------------------------------------------------------


class TestDualTruncation:
    """Both context_window and SAE max context apply."""

    def test_sae_cap_wins_when_smaller(self, tiny_model):
        """When SAE cap < context_window-allowed tokens, SAE cap takes precedence."""
        model, sae = tiny_model
        model.cfg.n_ctx = 5  # only 5 SAE tokens allowed
        text = "A" * 200

        _, _, _, _, char_ends, _ = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=100,  # context_window allows all
            segment_size=4,
        )
        # Only 5 SAE tokens (per-char) -> 5 chars
        assert char_ends[-1] <= 5

    def test_context_window_wins_when_smaller(self, tiny_model):
        """When context_window is more restrictive, it takes precedence."""
        model, sae = tiny_model
        model.cfg.n_ctx = 1000  # SAE allows a lot
        text = "A" * 200

        _, _, _, _, char_ends, _ = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=20,
            segment_size=4,
        )
        assert char_ends[-1] <= 20

    def test_both_limits_inactive_for_short_doc(self, tiny_model):
        """Short doc: neither limit applies, all tokens kept."""
        model, sae = tiny_model
        model.cfg.n_ctx = 1000
        text = "Hello"  # 5 chars + BOS prepend = 6 tokens

        _, _, _, _, _, n_tokens = extract_document_features(
            model,
            sae,
            text,
            layer=8,
            context_window=100,
        )
        assert n_tokens == 6
