# tests/test_batch_tokenization.py
"""Tests for batched tokenization in precompute_token_boundaries.

Verifies:
1. Batch encode matches sequential encode.
2. Pre-tokenized boundary computation handles edge cases.
3. The live shard path preserves its pinned tensor output.
4. Pooled and sequential boundary computation agree.
"""

import hashlib

import torch as t

from scripts.preprocess.precompute_token_boundaries import (
    compute_doc_token_boundaries_from_tokens,
    precompute_shard,
)
from steering.feature_io import TensorKey
from steering.overlay_io import (
    OverlayKind,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    write_manifest,
)
from steering.segmentation import token_to_char_offsets
from tests.steering.test_precomputed_token_boundaries import (
    SimpleTokenizer,
    _make_parallel_test_shard,
    _worker_pool,
)


class TestEdgeCases:
    """Edge cases for compute_doc_token_boundaries_from_tokens."""

    def test_empty_text(self):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text_tokens = tok.encode("", prepend=bos)

        starts, ends = compute_doc_token_boundaries_from_tokens(
            text_tokens,
            [],
            [],
            tok,
            source_bos_char_len=len(tok.decode([bos])),
        )
        assert starts.numel() == 0
        assert ends.numel() == 0

    def test_single_word(self):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text = "hello"
        text_tokens = tok.encode(text, prepend=bos)

        char_offsets = token_to_char_offsets(tok, text_tokens)
        seg_char_starts = [char_offsets[0]]
        seg_char_ends = [char_offsets[-1]]

        starts, ends = compute_doc_token_boundaries_from_tokens(
            text_tokens,
            seg_char_starts,
            seg_char_ends,
            tok,
            source_bos_char_len=len(tok.decode([bos])),
        )
        assert starts.tolist() == [0]
        assert ends.tolist() == [len(text_tokens)]

    def test_long_text(self):
        """10K+ words."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text = " ".join(f"word{i}" for i in range(10_000))
        text_tokens = tok.encode(text, prepend=bos)

        char_offsets = token_to_char_offsets(tok, text_tokens)
        # 10 segments
        n_toks = len(text_tokens)
        seg_size = n_toks // 10
        seg_char_starts = [char_offsets[i * seg_size] for i in range(10)]
        seg_char_ends = [
            char_offsets[min((i + 1) * seg_size, len(char_offsets) - 1)]
            for i in range(10)
        ]

        starts, ends = compute_doc_token_boundaries_from_tokens(
            text_tokens,
            seg_char_starts,
            seg_char_ends,
            tok,
            source_bos_char_len=len(tok.decode([bos])),
        )
        assert starts.numel() == 10
        assert ends.numel() == 10
        assert t.all(starts[1:] >= starts[:-1])
        assert t.all(ends[1:] >= ends[:-1])
        assert t.all(starts <= ends)
        assert ends[-1] <= len(text_tokens)

    def test_no_segments(self):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text_tokens = tok.encode("hello world", prepend=bos)

        starts, ends = compute_doc_token_boundaries_from_tokens(
            text_tokens,
            [],
            [],
            tok,
            source_bos_char_len=len(tok.decode([bos])),
        )
        assert starts.numel() == 0
        assert ends.numel() == 0


class TestBatchEncodeEquivalence:
    """tokenizer.encode(list_of_texts) == [tokenizer.encode(t) for t in texts]."""

    def test_simple_tokenizer_batch(self):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()

        texts = [
            "hello world foo bar baz",
            " ".join(f"w{i}" for i in range(30)),
            "single",
            "",
            " ".join(f"word{i}" for i in range(100)),
        ]

        batch_result = tok.encode(texts, prepend=bos)
        sequential_result = [tok.encode(text, prepend=bos) for text in texts]

        assert len(batch_result) == len(sequential_result)
        for i, (batch_ids, seq_ids) in enumerate(zip(batch_result, sequential_result, strict=True)):
            assert batch_ids == seq_ids, (
                f"Document {i}: batch != sequential\n"
                f"  batch: {batch_ids[:20]}...\n  seq: {seq_ids[:20]}..."
            )


class TestShardOutput:
    """The live batched path preserves its reviewed tensor output."""

    def test_boundary_tensor_hash(self, tmp_path):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        checksum = "parity" + "0" * 58

        shard, parquet, _ = _make_parallel_test_shard(
            tmp_path, n_docs=12, suffix="_baseline",
        )
        output = overlay_dir(shard.parent, OverlayKind.SEG_BOUNDARY, checksum)
        write_manifest(output, checksum, OverlayKind.SEG_BOUNDARY)

        precompute_shard(
            shard,
            parquet,
            tok,
            bos,
            output,
            "shard_00000",
            source_bos_char_len=len(tok.decode([bos])),
            source_segment_size=16,
        )

        data = load_overlay_shard(overlay_shard_path(output, "shard_00000"))
        starts = data[TensorKey.SEGMENT_TOKEN_STARTS].contiguous().numpy().tobytes()
        ends = data[TensorKey.SEGMENT_TOKEN_ENDS].contiguous().numpy().tobytes()
        assert hashlib.sha256(starts + ends).hexdigest() == (
            "50740ab1480d2abfdaa8a5d82053eaaa0936f3d9e4aec44c790c30eba9044320"
        )


class TestParallelEquivalence:
    """Pooled and sequential boundary computation agree."""

    def test_parallel_vs_sequential(self, tmp_path):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        checksum = "parallel_batch_test" + "0" * 45

        shard_par, pq_par, _ = _make_parallel_test_shard(
            tmp_path, n_docs=10, suffix="_par2",
        )
        shard_seq, pq_seq, _ = _make_parallel_test_shard(
            tmp_path, n_docs=10, suffix="_seq2",
        )

        out_par = overlay_dir(shard_par.parent, OverlayKind.SEG_BOUNDARY, checksum)
        out_seq = overlay_dir(shard_seq.parent, OverlayKind.SEG_BOUNDARY, checksum)
        write_manifest(out_par, checksum, OverlayKind.SEG_BOUNDARY)
        write_manifest(out_seq, checksum, OverlayKind.SEG_BOUNDARY)

        with _worker_pool(tok, len(tok.decode([bos])), 16, n_workers=4) as pool:
            precompute_shard(
                shard_par,
                pq_par,
                tok,
                bos,
                out_par,
                "shard_00000",
                source_bos_char_len=len(tok.decode([bos])),
                source_segment_size=16,
                worker_pool=pool,
            )
        precompute_shard(
            shard_seq,
            pq_seq,
            tok,
            bos,
            out_seq,
            "shard_00000",
            source_bos_char_len=len(tok.decode([bos])),
            source_segment_size=16,
        )

        data_par = load_overlay_shard(overlay_shard_path(out_par, "shard_00000"))
        data_seq = load_overlay_shard(overlay_shard_path(out_seq, "shard_00000"))

        assert t.equal(
            data_par[TensorKey.SEGMENT_TOKEN_STARTS],
            data_seq[TensorKey.SEGMENT_TOKEN_STARTS],
        )
        assert t.equal(
            data_par[TensorKey.SEGMENT_TOKEN_ENDS],
            data_seq[TensorKey.SEGMENT_TOKEN_ENDS],
        )
