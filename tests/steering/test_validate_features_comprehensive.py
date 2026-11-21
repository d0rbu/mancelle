"""Comprehensive validation tests for feature shard files.

Tests segment tiling, character boundaries, document grouping,
source-data cross-referencing, directory consistency, and
GPU activation spot-checks.
"""

import hypothesis.strategies as st
import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from hypothesis import assume, given, settings

from steering.extract_features import PoolingMode, extract_document_features
from steering.feature_io import load_features, save_features
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, segment_tokens, token_to_char_offsets
from steering.shard_validation import ValidationResult, validate_shard_data
from steering.validate_features import (
    _select_spot_check_docs,
    validate_feature_directory,
    validate_shard_activations,
    validate_shard_against_source,
    validate_shard_segments,
)
from tests.conftest import MockTokenizer
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE


class _ProductionMockTokenizer(MockTokenizer):
    """Expose the callable mapping contract used by production extraction."""

    def __call__(self, text: str, *, padding: bool = False) -> dict[str, list[int]]:  # ty: ignore[invalid-method-override]
        assert padding is False
        return {"input_ids": self.encode(text)}  # ty: ignore[invalid-return-type]


def _make_valid_shard_dict(
    n_docs: int = 3,
    segs_per_doc: list[int] | None = None,
    feats_per_seg: int = 5,
    max_feature_id: int = 16383,
    segment_size: int = 128,
    context_window: int = 1024,
) -> dict:
    """Build a shard dict satisfying ALL validation invariants."""
    if segs_per_doc is None:
        segs_per_doc = [3, 2, 4]
    assert len(segs_per_doc) == n_docs

    n_segments = sum(segs_per_doc)
    n_features = n_segments * feats_per_seg
    offsets = [i * feats_per_seg for i in range(n_segments + 1)]

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    doc_token_counts = []
    chars_per_token = 4

    for doc_id in range(n_docs):
        n_segs = segs_per_doc[doc_id]
        total_tokens = (n_segs - 1) * segment_size + segment_size
        doc_token_counts.append(total_tokens)
        for seg_idx in range(n_segs):
            doc_ids.append(doc_id)
            seg_indices.append(seg_idx)
            seg_positions.append(seg_idx * segment_size)
            seg_char_starts.append(seg_idx * segment_size * chars_per_token)
            seg_char_ends.append((seg_idx + 1) * segment_size * chars_per_token)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.randint(0, max_feature_id + 1, (n_features,), dtype=t.int32),
        "feature_mags": t.rand(n_features, dtype=t.float32).abs() + 0.01,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": context_window,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
    }


def _make_parquet_and_shard(tmp_path, n_docs=3, segment_size=4, context_window=64):
    """Create a synthetic parquet file and matching valid shard.

    Uses MockTokenizer-compatible text (ASCII chars) so re-tokenization
    produces predictable token counts. Each character is one token.
    """
    tokenizer = _ProductionMockTokenizer()

    texts = []
    doc_token_counts = []
    all_seg_positions = []
    all_seg_char_starts = []
    all_seg_char_ends = []
    all_doc_ids = []
    all_seg_indices = []
    all_feat_ids = []
    all_feat_mags = []

    seg_counter = 0
    for doc_id in range(n_docs):
        # Each doc has a different length
        n_chars = 20 + doc_id * 8  # 20, 28, 36 chars
        text = "A" * n_chars
        texts.append(text)

        # MockTokenizer: each char -> 1 token (no BOS)
        token_ids = tokenizer.encode(text)
        n_tokens = len(token_ids)
        doc_token_counts.append(n_tokens)

        # Segment: fixed_tokens
        n_kept = min(n_tokens, context_window)
        kept_ids = token_ids[:n_kept]
        n_segs = (n_kept + segment_size - 1) // segment_size

        char_offsets = token_to_char_offsets(tokenizer, kept_ids)  # ty: ignore[invalid-argument-type]

        for seg_idx in range(n_segs):
            start_tok = seg_idx * segment_size
            end_tok = min(start_tok + segment_size, n_kept)
            all_doc_ids.append(doc_id)
            all_seg_indices.append(seg_idx)
            all_seg_positions.append(start_tok)
            all_seg_char_starts.append(char_offsets[start_tok])
            all_seg_char_ends.append(char_offsets[end_tok])
            # Generate some features for this segment
            n_feats = 3
            all_feat_ids.append(t.randint(0, 100, (n_feats,), dtype=t.int32))
            all_feat_mags.append(t.rand(n_feats, dtype=t.float32).abs() + 0.01)
            seg_counter += 1

    # Write parquet
    pq_path = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": texts}), str(pq_path))

    # Build shard dict
    n_segments = seg_counter
    offsets = [0]
    for fids in all_feat_ids:
        offsets.append(offsets[-1] + len(fids))

    shard = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.cat(all_feat_ids)
        if all_feat_ids
        else t.tensor([], dtype=t.int32),
        "feature_mags": t.cat(all_feat_mags)
        if all_feat_mags
        else t.tensor([], dtype=t.float32),
        "doc_ids": t.tensor(all_doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(all_seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(all_seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(all_seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(all_seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": context_window,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
    }
    return pq_path, shard, tokenizer


def _make_valid_directory(tmp_path, n_shards=3, n_docs_per_shard=2, segment_size=128):
    """Create a directory with multiple valid feature shards."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for i in range(n_shards):
        shard = _make_valid_shard_dict(
            n_docs=n_docs_per_shard,
            segs_per_doc=[2] * n_docs_per_shard,
            segment_size=segment_size,
        )
        save_features(shard, feature_dir / f"shard_{i:05d}")
    return feature_dir


def _make_shard_from_extraction(
    texts, model, sae, segment_size=4, context_window=64, pooling="max_pool"
):
    """Run extraction on texts and build a shard dict."""
    all_feat_ids = []
    all_feat_mags = []
    doc_ids_list = []
    seg_indices_list = []
    seg_positions_list = []
    char_starts_list = []
    char_ends_list = []
    doc_token_counts = []

    for doc_id, text in enumerate(texts):
        feat_ids, feat_mags, positions, char_starts, char_ends, n_tokens = (
            extract_document_features(
                model,
                sae,
                text,
                layer=8,
                context_window=context_window,
                pooling=PoolingMode(pooling),
                segment_mode=SegmentMode.fixed_tokens,
                segment_size=segment_size,
            )
        )
        n_segs = len(feat_ids)
        all_feat_ids.extend(feat_ids)
        all_feat_mags.extend(feat_mags)
        doc_ids_list.extend([doc_id] * n_segs)
        seg_indices_list.extend(range(n_segs))
        seg_positions_list.extend(positions)
        char_starts_list.extend(char_starts)
        char_ends_list.extend(char_ends)
        doc_token_counts.append(n_tokens)

    n_segments = len(all_feat_ids)
    feat_counts = t.tensor([len(fids) for fids in all_feat_ids], dtype=t.int64)
    offsets = t.cat([t.zeros(1, dtype=t.int64), feat_counts.cumsum(0)])

    return {
        "offsets": offsets,
        "feature_ids": t.cat(all_feat_ids)
        if all_feat_ids
        else t.tensor([], dtype=t.int32),
        "feature_mags": t.cat(all_feat_mags)
        if all_feat_mags
        else t.tensor([], dtype=t.float32),
        "doc_ids": t.tensor(doc_ids_list, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices_list, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions_list, dtype=t.int32),
        "segment_char_starts": t.tensor(char_starts_list, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends_list, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": len(texts),
        "n_segments": n_segments,
        "layer": 8,
        "context_window": context_window,
        "pooling": pooling,
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
    }


class TestShardSegmentValidation:
    """Tests for validate_shard_segments."""

    def test_valid_shard_passes(self):
        data = _make_valid_shard_dict()
        result = validate_shard_segments(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_missing_char_starts(self):
        data = _make_valid_shard_dict()
        del data["segment_char_starts"]
        result = validate_shard_segments(data)
        assert not result.valid
        assert any("segment_char_starts" in v for v in result.violations)

    def test_missing_char_ends(self):
        data = _make_valid_shard_dict()
        del data["segment_char_ends"]
        result = validate_shard_segments(data)
        assert not result.valid
        assert any("segment_char_ends" in v for v in result.violations)

    def test_detects_out_of_order_segment_indices(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[3])
        data["segment_indices"] = t.tensor([0, 2, 1], dtype=t.int32)
        result = validate_shard_segments(data)
        assert not result.valid
        assert any("segment_indices" in v for v in result.violations)

    def test_detects_non_contiguous_positions(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[3], segment_size=128)
        data["segment_positions"] = t.tensor([0, 128, 384], dtype=t.int32)
        result = validate_shard_segments(data)
        assert not result.valid
        assert any("position" in v.lower() for v in result.violations)

    def test_detects_char_end_less_than_start(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
        data["segment_char_ends"][0] = data["segment_char_starts"][0] - 1
        result = validate_shard_segments(data)
        assert not result.valid
        assert any(
            "char_end" in v.lower() or "char" in v.lower() for v in result.violations
        )

    def test_detects_char_boundary_tiling_gap(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[3], segment_size=128)
        data["segment_char_ends"][0] = data["segment_char_starts"][1] - 10
        result = validate_shard_segments(data)
        assert not result.valid
        assert any(
            "tile" in v.lower() or "char" in v.lower() for v in result.violations
        )

    def test_detects_negative_char_start(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
        data["segment_char_starts"][0] = -1
        result = validate_shard_segments(data)
        assert not result.valid

    def test_detects_non_contiguous_doc_ids(self):
        data = _make_valid_shard_dict(n_docs=2, segs_per_doc=[2, 2])
        data["doc_ids"] = t.tensor([0, 1, 0, 1], dtype=t.int32)
        result = validate_shard_segments(data)
        assert not result.valid
        assert any(
            "contiguous" in v.lower() or "doc_id" in v.lower()
            for v in result.violations
        )

    def test_detects_non_positive_doc_token_counts(self):
        data = _make_valid_shard_dict()
        data["doc_token_counts"][0] = 0
        result = validate_shard_segments(data)
        assert not result.valid

    def test_detects_doc_id_out_of_range(self):
        data = _make_valid_shard_dict(n_docs=2, segs_per_doc=[2, 2])
        data["doc_ids"] = t.tensor([0, 0, 5, 5], dtype=t.int32)
        result = validate_shard_segments(data)
        assert not result.valid
        assert any(
            "doc_id" in v.lower() and "n_docs" in v.lower() for v in result.violations
        )

    def test_detects_more_distinct_doc_ids_than_n_docs(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
        data["doc_ids"] = t.tensor([0, 1], dtype=t.int32)
        result = validate_shard_segments(data)
        assert not result.valid

    def test_single_segment_document(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[1])
        result = validate_shard_segments(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_single_doc_shard(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[5])
        result = validate_shard_segments(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_short_last_segment(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[3], segment_size=128)
        data["doc_token_counts"][0] = 300
        result = validate_shard_segments(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_first_pretraining_segment_starts_at_char_zero(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[3])
        data["segment_char_starts"][0] = 10
        result = validate_shard_segments(data)
        assert not result.valid
        assert any("char" in v.lower() and "0" in v for v in result.violations)

    def test_sft_first_segment_nonzero_position_valid(self):
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2], segment_size=128)
        data["segment_positions"] = t.tensor([64, 192], dtype=t.int32)
        data["segment_char_starts"] = t.tensor([0, 512], dtype=t.int32)
        data["segment_char_ends"] = t.tensor([512, 1024], dtype=t.int32)
        result = validate_shard_segments(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_detects_last_segment_beyond_context_window(self):
        data = _make_valid_shard_dict(
            n_docs=1, segs_per_doc=[3], segment_size=128, context_window=256
        )
        data["doc_token_counts"][0] = 384
        result = validate_shard_segments(data)
        assert not result.valid
        assert any(
            "context_window" in v.lower() or "exceed" in v.lower()
            for v in result.violations
        )


@st.composite
def valid_shard_strategy(draw):
    """Generate random valid shard dicts that should pass all checks."""
    n_docs = draw(st.integers(1, 5))
    segs_per_doc = [draw(st.integers(1, 8)) for _ in range(n_docs)]
    segment_size = draw(st.sampled_from([32, 64, 128, 256]))
    feats_per_seg = draw(st.integers(1, 10))
    context_window = segment_size * max(segs_per_doc) + segment_size
    return _make_valid_shard_dict(
        n_docs=n_docs,
        segs_per_doc=segs_per_doc,
        feats_per_seg=feats_per_seg,
        segment_size=segment_size,
        context_window=context_window,
    )


class TestShardSegmentPropertyBased:
    """Property-based tests for validate_shard_segments."""

    @given(data=valid_shard_strategy())
    @settings(max_examples=100)
    def test_valid_shards_always_pass(self, data):
        result = validate_shard_segments(data)
        assert result.valid, f"Violations: {result.violations}"

    @given(data=valid_shard_strategy())
    @settings(max_examples=50)
    def test_swapping_two_segment_indices_detected(self, data):
        n_segs = data["n_segments"]
        assume(n_segs >= 2)
        indices = data["segment_indices"].clone()
        # Only meaningful when the two values differ (same-doc with multiple segs);
        # if both are 0 the swap is a no-op and nothing is broken.
        assume(indices[0].item() != indices[1].item())
        indices[0], indices[1] = indices[1].item(), indices[0].item()
        data["segment_indices"] = indices
        result = validate_shard_segments(data)
        assert not result.valid

    @given(data=valid_shard_strategy())
    @settings(max_examples=50)
    def test_negating_char_start_detected(self, data):
        assume(data["n_segments"] > 0)
        data["segment_char_starts"][0] = -1
        result = validate_shard_segments(data)
        assert not result.valid


class TestRoundTripValidation:
    """Validate that save -> load -> validate passes for valid shards."""

    def test_roundtrip_passes_all_checks(self, tmp_path):
        data = _make_valid_shard_dict()
        path = save_features(data, tmp_path / "shard_00000")
        loaded = load_features(path)
        result0 = validate_shard_data(loaded)
        assert result0.valid, f"validate_shard_data violations: {result0.violations}"
        result1 = validate_shard_segments(loaded)
        assert result1.valid, (
            f"validate_shard_segments violations: {result1.violations}"
        )

    def test_roundtrip_sft_style_shard(self, tmp_path):
        """SFT-style shard with nonzero first position."""
        data = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2], segment_size=128)
        data["segment_positions"] = t.tensor([64, 192], dtype=t.int32)
        data["segment_char_starts"] = t.tensor([0, 512], dtype=t.int32)
        data["segment_char_ends"] = t.tensor([512, 1024], dtype=t.int32)
        path = save_features(data, tmp_path / "shard_00000")
        loaded = load_features(path)
        result0 = validate_shard_data(loaded)
        assert result0.valid, f"validate_shard_data violations: {result0.violations}"
        result1 = validate_shard_segments(loaded)
        assert result1.valid, (
            f"validate_shard_segments violations: {result1.violations}"
        )


class TestSourceValidation:
    """Tests for validate_shard_against_source."""

    def test_valid_shard_against_source_passes(self, tmp_path):
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path)
        result = validate_shard_against_source(
            shard,
            pq_path,
            tokenizer,
            context_window=shard["context_window"],
            segment_size=shard["segment_size"],
        )
        assert result.valid, f"Violations: {result.violations}"

    def test_detects_doc_count_mismatch(self, tmp_path):
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path, n_docs=3)
        shard["n_docs"] = 5  # Wrong count
        result = validate_shard_against_source(
            shard,
            pq_path,
            tokenizer,
            context_window=shard["context_window"],
            segment_size=shard["segment_size"],
        )
        assert not result.valid
        assert any(
            "doc" in v.lower() and "count" in v.lower() for v in result.violations
        )

    def test_detects_token_count_mismatch(self, tmp_path):
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path)
        shard["doc_token_counts"][0] = 999  # Wrong
        result = validate_shard_against_source(
            shard,
            pq_path,
            tokenizer,
            context_window=shard["context_window"],
            segment_size=shard["segment_size"],
        )
        assert not result.valid
        assert any("token" in v.lower() for v in result.violations)

    def test_detects_char_boundary_beyond_text(self, tmp_path):
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path)
        # Set a char_end beyond the decoded text length
        shard["segment_char_ends"][-1] = 99999
        result = validate_shard_against_source(
            shard,
            pq_path,
            tokenizer,
            context_window=shard["context_window"],
            segment_size=shard["segment_size"],
        )
        assert not result.valid
        assert any("char" in v.lower() for v in result.violations)

    def test_sft_without_chat_template_skips_gracefully(self, tmp_path):
        """SFT parquet with a tokenizer lacking apply_chat_template skips SFT checks."""
        tokenizer = MockTokenizer()  # No apply_chat_template
        pq_path = tmp_path / "shard_00000.parquet"
        pq.write_table(
            pa.table(
                {
                    "text": ["hello"],
                    "messages": [[{"role": "user", "content": "hi"}]],
                    "response": ["hello"],
                }
            ),
            str(pq_path),
        )
        # Minimal shard
        shard = _make_valid_shard_dict(n_docs=1, segs_per_doc=[1], segment_size=4)
        shard["doc_token_counts"] = t.tensor([5], dtype=t.int32)
        result = validate_shard_against_source(
            shard,
            pq_path,
            tokenizer,
            context_window=64,
            segment_size=4,
        )
        # Should not crash — SFT checks skipped with warning
        assert isinstance(result, ValidationResult)


class TestDirectoryValidation:
    """Tests for validate_feature_directory."""

    def test_valid_directory_passes(self, tmp_path):
        feature_dir = _make_valid_directory(tmp_path)
        result = validate_feature_directory(feature_dir)
        assert result.valid, f"Violations: {result.violations}"

    def test_detects_metadata_mismatch_layer(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        # Shard 0: layer=8
        s0 = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
        s0["layer"] = 8
        save_features(s0, feature_dir / "shard_00000")
        # Shard 1: layer=12
        s1 = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
        s1["layer"] = 12
        save_features(s1, feature_dir / "shard_00001")
        result = validate_feature_directory(feature_dir)
        assert not result.valid
        assert any("layer" in v.lower() for v in result.violations)

    def test_detects_missing_shard_for_parquet(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        parquet_dir = tmp_path / "parquets"
        parquet_dir.mkdir()
        # Create 2 parquet files but only 1 feature shard
        for i in range(2):
            pq.write_table(
                pa.table({"text": ["hello"]}),
                str(parquet_dir / f"shard_{i:05d}.parquet"),
            )
        s = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
        save_features(s, feature_dir / "shard_00000")
        result = validate_feature_directory(
            feature_dir,
            parquet_dir=parquet_dir,
            tokenizer=_ProductionMockTokenizer(),
        )
        assert not result.valid
        assert any("missing" in v.lower() for v in result.violations)

    def test_detects_orphan_shard(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        parquet_dir = tmp_path / "parquets"
        parquet_dir.mkdir()
        # Create 1 parquet file but 2 feature shards
        pq.write_table(
            pa.table({"text": ["hello"]}), str(parquet_dir / "shard_00000.parquet")
        )
        for i in range(2):
            s = _make_valid_shard_dict(n_docs=1, segs_per_doc=[2])
            save_features(s, feature_dir / f"shard_{i:05d}")
        result = validate_feature_directory(
            feature_dir,
            parquet_dir=parquet_dir,
            tokenizer=_ProductionMockTokenizer(),
        )
        assert not result.valid
        assert any(
            "orphan" in v.lower() or "no parquet" in v.lower()
            for v in result.violations
        )

    def test_empty_directory(self, tmp_path):
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        result = validate_feature_directory(feature_dir)
        assert not result.valid
        assert any(
            "no shard" in v.lower() or "empty" in v.lower() for v in result.violations
        )


class TestActivationValidation:
    """Tests for validate_shard_activations using TinyHookedTransformer + TinySAE."""

    def _get_model_and_sae(self):
        model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
        sae = TinySAE(d_model=64, n_features=256, device="cpu")
        return model, sae

    def test_valid_extraction_passes(self, tmp_path):
        model, sae = self._get_model_and_sae()
        texts = [
            "Hello world this is a test document with enough tokens",
            "Another doc here",
        ]
        pq_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))

        shard = _make_shard_from_extraction(
            texts, model, sae, segment_size=8, context_window=64
        )
        result = validate_shard_activations(
            shard,
            pq_path,
            model,
            sae,
            layer=8,
            context_window=64,
            segment_size=8,
            pooling="max_pool",
            n_spot_check=2,
        )
        assert result.valid, f"Violations: {result.violations}"

    def test_n_spot_check_zero_skips(self, tmp_path):
        model, sae = self._get_model_and_sae()
        texts = ["Hello world"]
        pq_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))

        shard = _make_shard_from_extraction(
            texts, model, sae, segment_size=4, context_window=64
        )
        result = validate_shard_activations(
            shard,
            pq_path,
            model,
            sae,
            layer=8,
            context_window=64,
            segment_size=4,
            pooling="max_pool",
            n_spot_check=0,
        )
        assert result.valid  # Trivially passes — no checks run

    def test_detects_tampered_magnitudes(self, tmp_path):
        model, sae = self._get_model_and_sae()
        texts = ["Hello world this is a test document with tokens"]
        pq_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))

        shard = _make_shard_from_extraction(
            texts, model, sae, segment_size=8, context_window=64
        )
        # Tamper: multiply all magnitudes by 100
        shard["feature_mags"] = shard["feature_mags"] * 100.0
        result = validate_shard_activations(
            shard,
            pq_path,
            model,
            sae,
            layer=8,
            context_window=64,
            segment_size=8,
            pooling="max_pool",
            n_spot_check=1,
        )
        assert not result.valid
        assert any(
            "magnitude" in v.lower() or "mismatch" in v.lower()
            for v in result.violations
        )

    def test_detects_tampered_feature_ids(self, tmp_path):
        model, sae = self._get_model_and_sae()
        texts = ["Hello world this is a test document with tokens"]
        pq_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))

        shard = _make_shard_from_extraction(
            texts, model, sae, segment_size=8, context_window=64
        )
        # Tamper: shift all feature IDs by 1
        if len(shard["feature_ids"]) > 0:
            shard["feature_ids"] = (shard["feature_ids"] + 1) % 256
        result = validate_shard_activations(
            shard,
            pq_path,
            model,
            sae,
            layer=8,
            context_window=64,
            segment_size=8,
            pooling="max_pool",
            n_spot_check=1,
        )
        assert not result.valid

    def test_length_biased_sampling(self):
        """Verify the document selection biases toward longer documents."""
        doc_token_counts = t.tensor([10, 100, 1000, 50, 500], dtype=t.int32)
        n_docs = 5
        # Call the selection helper directly
        selected = _select_spot_check_docs(doc_token_counts, n_docs, n_spot_check=4)
        # First and last should always be included
        assert 0 in selected
        assert n_docs - 1 in selected
        # Doc 2 (1000 tokens) should be very likely in the remaining picks
        assert len(selected) == 4
        assert len(set(selected)) == 4  # No duplicates


def ref_validate_shard(
    shard_data: dict,
    texts: list[str],
    tokenizer,
    segment_size: int,
    context_window: int,
) -> list[str]:
    """Independent reference validator that re-derives shard contents from scratch.

    Re-tokenizes each text, re-segments, re-computes char boundaries, and
    compares against stored shard data. Returns a list of violation strings
    (empty if valid).
    """
    violations: list[str] = []
    n_docs = shard_data["n_docs"]
    if n_docs != len(texts):
        violations.append(f"n_docs={n_docs} != len(texts)={len(texts)}")
        return violations

    doc_ids = shard_data["doc_ids"]
    seg_indices = shard_data["segment_indices"]
    seg_positions = shard_data["segment_positions"]
    char_starts = shard_data["segment_char_starts"]
    char_ends = shard_data["segment_char_ends"]
    doc_token_counts = shard_data["doc_token_counts"]

    expected_total_segs = 0
    for doc_id in range(n_docs):
        text = texts[doc_id]
        token_ids = tokenizer.encode(text)
        n_tokens = len(token_ids)

        # Check token count
        stored_count = int(doc_token_counts[doc_id].item())
        if stored_count != n_tokens:
            violations.append(
                f"Doc {doc_id}: ref token count={n_tokens}, stored={stored_count}"
            )
            continue

        # Head truncation
        n_kept = min(n_tokens, context_window)
        kept_ids = token_ids[:n_kept]

        # Re-segment (fixed_tokens mode: just need length and segment_size)
        segments = segment_tokens(
            kept_ids,
            segment_size=PositiveInt.parse(segment_size),
        )
        n_segs = len(segments)
        expected_total_segs += n_segs

        # Check segments for this doc
        mask = doc_ids == doc_id
        stored_n_segs = int(mask.sum().item())
        if stored_n_segs != n_segs:
            violations.append(
                f"Doc {doc_id}: ref n_segs={n_segs}, stored={stored_n_segs}"
            )
            continue

        doc_seg_positions = seg_positions[mask]
        doc_seg_indices = seg_indices[mask]
        doc_char_starts = char_starts[mask]
        doc_char_ends = char_ends[mask]

        # Re-compute char offsets
        ref_char_offsets = token_to_char_offsets(tokenizer, kept_ids)

        for seg_idx, (start_tok, end_tok) in enumerate(segments):
            # Position check
            stored_pos = int(doc_seg_positions[seg_idx].item())
            if stored_pos != start_tok:
                violations.append(
                    f"Doc {doc_id}, seg {seg_idx}: ref pos={start_tok}, stored={stored_pos}"
                )

            # Index check
            stored_idx = int(doc_seg_indices[seg_idx].item())
            if stored_idx != seg_idx:
                violations.append(
                    f"Doc {doc_id}, seg {seg_idx}: ref idx={seg_idx}, stored={stored_idx}"
                )

            # Char boundary check
            ref_cs = ref_char_offsets[start_tok]
            ref_ce = ref_char_offsets[end_tok]
            stored_cs = int(doc_char_starts[seg_idx].item())
            stored_ce = int(doc_char_ends[seg_idx].item())
            if stored_cs != ref_cs:
                violations.append(
                    f"Doc {doc_id}, seg {seg_idx}: ref char_start={ref_cs}, stored={stored_cs}"
                )
            if stored_ce != ref_ce:
                violations.append(
                    f"Doc {doc_id}, seg {seg_idx}: ref char_end={ref_ce}, stored={stored_ce}"
                )

    # Total segments check
    if expected_total_segs != shard_data["n_segments"]:
        violations.append(
            f"ref total_segs={expected_total_segs}, stored n_segments={shard_data['n_segments']}"
        )

    return violations


class TestReferenceValidation:
    """Test the reference validator against synthetic data."""

    def test_ref_validates_synthetic_shard(self, tmp_path):
        """Shard created by _make_parquet_and_shard should pass ref validation."""
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path)
        table = pq.read_table(str(pq_path))
        texts = table.column("text").to_pylist()
        violations = ref_validate_shard(
            shard,
            texts,
            tokenizer,
            segment_size=shard["segment_size"],
            context_window=shard["context_window"],
        )
        assert violations == [], f"Violations: {violations}"

    def test_ref_detects_wrong_position(self, tmp_path):
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path)
        table = pq.read_table(str(pq_path))
        texts = table.column("text").to_pylist()
        shard["segment_positions"][0] = 999
        violations = ref_validate_shard(
            shard,
            texts,
            tokenizer,
            segment_size=shard["segment_size"],
            context_window=shard["context_window"],
        )
        assert len(violations) > 0
        assert any("pos" in v.lower() for v in violations)

    def test_ref_detects_wrong_char_boundary(self, tmp_path):
        pq_path, shard, tokenizer = _make_parquet_and_shard(tmp_path)
        table = pq.read_table(str(pq_path))
        texts = table.column("text").to_pylist()
        shard["segment_char_starts"][0] = 999
        violations = ref_validate_shard(
            shard,
            texts,
            tokenizer,
            segment_size=shard["segment_size"],
            context_window=shard["context_window"],
        )
        assert len(violations) > 0
        assert any("char" in v.lower() for v in violations)
