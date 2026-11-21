"""Tests for `scripts.utils.validate_token_boundary_overlays`.

Structural checks (no tokenizer) live in `TestValidateShardStructural`.
Deep (recompute) checks and the CLI end-to-end test live in later classes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tiktoken
import torch as t
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer, get_tokenizer
from scripts.preprocess.precompute_token_boundaries import (
    compute_doc_token_boundaries_from_tokens,
    precompute_shard,
)
from scripts.utils.validate_token_boundary_overlays import (
    IssueKind,
    ShardValidationResult,
    ValidationIssue,
    iter_issues_by_kind,
    run,
    validate_shard,
    validate_shard_deep,
    validate_shard_structural,
)
from steering.feature_io import TensorKey, empty_segment_indices, save_features
from steering.overlay_io import (
    OverlayKind,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.segmentation import (
    _HYBRID_CHUNK_SIZE,
    _token_to_char_offsets_progressive,
    compute_tokenizer_checksum,
    token_to_char_offsets,
)


def _tensor(values: list[int]) -> t.Tensor:
    return t.tensor(values, dtype=t.int32)


def _one_feature_per_segment_payload(n_segments: int) -> dict[str, t.Tensor]:
    """Build a minimal base-shard CSR payload with no empty segment rows."""
    return {
        "offsets": t.arange(n_segments + 1, dtype=t.int64),
        "feature_ids": t.zeros(n_segments, dtype=t.int32),
        "feature_mags": t.ones(n_segments, dtype=t.float32),
    }


@pytest.mark.parametrize("n_segments", [0, 1, 4])
def test_one_feature_per_segment_payload_respects_base_shard_invariant(n_segments):
    payload = _one_feature_per_segment_payload(n_segments)
    offsets = payload["offsets"]

    if n_segments > 0:
        assert t.all(offsets[1:] > offsets[:-1])
    assert len(payload["feature_ids"]) == len(payload["feature_mags"])
    assert len(payload["feature_ids"]) == int(offsets[-1].item())
    assert empty_segment_indices(offsets) == []


class _WordTokenizer:
    """Deterministic test tokenizer.

    BOS decodes to ``"<bos>"`` (5 chars). Non-BOS tokens decode to fixed
    5-char strings ``"tNNN "`` so char offsets are predictable. ``encode``
    splits ``text`` on spaces and assigns monotonically increasing ids.

    This matches the ``SimpleTokenizer`` pattern used elsewhere in the
    test suite so that ``compute_doc_token_boundaries_from_tokens`` runs
    on deterministic token-to-char offsets.
    """

    BOS_ID = 0

    def get_bos_token_id(self) -> int:
        return self.BOS_ID

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(item, prepend=prepend) for item in text]
        words = text.split()
        ids = [i + 1 for i in range(len(words))]
        if prepend is not None:
            ids.insert(0, prepend if isinstance(prepend, int) else self.BOS_ID)
        return ids

    def decode(self, token_ids):
        parts: list[str] = []
        for tid in token_ids:
            if tid == self.BOS_ID:
                parts.append("<bos>")
            else:
                parts.append(f"t{tid:03d} ")
        return "".join(parts)


def _make_two_doc_setup():
    """Construct a two-doc synthetic shard + the expected overlay.

    Returns a dict with everything a deep-validation test needs:
    doc_ids / segment_indices / segment_char_starts / segment_char_ends as
    int32 tensors, a list of texts, and the freshly-computed overlay
    tensors that the current pipeline would write.
    """
    tok = _WordTokenizer()
    bos = tok.get_bos_token_id()
    source_bos_char_len = 5  # "<bos>"
    # Two docs, 2 segments each; content chosen so that each segment gets
    # a non-trivial char range.
    texts = [
        " ".join(f"w{i:03d}" for i in range(8)),   # 8 words → 8 tokens
        " ".join(f"w{i:03d}" for i in range(10)),  # 10 words → 10 tokens
    ]
    doc_ids_rows: list[int] = []
    seg_indices_rows: list[int] = []
    char_starts_rows: list[int] = []
    char_ends_rows: list[int] = []
    tok_starts_rows: list[int] = []
    tok_ends_rows: list[int] = []
    for doc_id, text in enumerate(texts):
        text_tokens = tok.encode(text, prepend=bos)
        # token_to_char_offsets: 5-char BOS + 5-char words.
        # Split this doc into two equal-sized segments in CHAR space:
        total_chars = 5 + 5 * (len(text_tokens) - 1)  # BOS + (N-1)*5
        mid = total_chars // 2
        doc_char_starts = [0, mid]
        doc_char_ends = [mid, total_chars]
        expected_starts, expected_ends = compute_doc_token_boundaries_from_tokens(
            text_tokens,
            doc_char_starts,
            doc_char_ends,
            tok,
            source_bos_char_len=source_bos_char_len,
        )
        for seg_local, (cs, ce, ts, te) in enumerate(
            zip(
                doc_char_starts,
                doc_char_ends,
                expected_starts.tolist(),
                expected_ends.tolist(), strict=True,
            )
        ):
            doc_ids_rows.append(doc_id)
            seg_indices_rows.append(seg_local)
            char_starts_rows.append(cs)
            char_ends_rows.append(ce)
            tok_starts_rows.append(int(ts))
            tok_ends_rows.append(int(te))
    return {
        "tokenizer": tok,
        "bos_token": bos,
        "source_bos_char_len": source_bos_char_len,
        "texts": texts,
        "doc_ids": _tensor(doc_ids_rows),
        "segment_indices": _tensor(seg_indices_rows),
        "segment_char_starts": _tensor(char_starts_rows),
        "segment_char_ends": _tensor(char_ends_rows),
        "overlay_token_starts": _tensor(tok_starts_rows),
        "overlay_token_ends": _tensor(tok_ends_rows),
    }


class TestValidateShardStructural:
    """Unit tests for the structural-only validator.

    The validator takes the base shard's ``doc_ids`` and ``segment_indices``
    tensors plus the overlay's ``token_starts``/``token_ends`` tensors and
    returns a list of :class:`ValidationIssue`. It never raises.
    """

    def test_valid_overlay_passes(self):
        """A correct overlay with two docs produces no issues."""
        # doc 0: 2 segments, tokens [0, 3) and [3, 5)
        # doc 1: 3 segments, tokens [0, 4), [4, 7), [7, 10)
        doc_ids = _tensor([0, 0, 1, 1, 1])
        segment_indices = _tensor([0, 1, 0, 1, 2])
        token_starts = _tensor([0, 3, 0, 4, 7])
        token_ends = _tensor([3, 5, 4, 7, 10])

        issues = validate_shard_structural(
            "shard_00000", doc_ids, segment_indices, token_starts, token_ends
        )
        assert issues == []

    def test_length_mismatch_flagged_and_short_circuits(self):
        """Wrong overlay length is flagged once; no per-segment checks run."""
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([0, 1])
        token_starts = _tensor([0, 3, 5])  # one extra
        token_ends = _tensor([3, 5, 8])

        issues = validate_shard_structural(
            "shard_00001", doc_ids, segment_indices, token_starts, token_ends
        )
        assert len(issues) == 1
        assert issues[0].kind is IssueKind.LENGTH_MISMATCH
        assert issues[0].shard_stem == "shard_00001"
        assert issues[0].doc_id is None
        assert issues[0].segment_idx is None

    def test_length_mismatch_on_ends_flagged(self):
        """Length mismatch on token_ends is also flagged."""
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([0, 1])
        token_starts = _tensor([0, 3])
        token_ends = _tensor([3, 5, 8])

        issues = validate_shard_structural(
            "shard_00002", doc_ids, segment_indices, token_starts, token_ends
        )
        assert len(issues) == 1
        assert issues[0].kind is IssueKind.LENGTH_MISMATCH

    def test_empty_segment_flagged(self):
        """token_end == token_start → EMPTY_SEGMENT for that segment only."""
        doc_ids = _tensor([0, 0, 0])
        segment_indices = _tensor([0, 1, 2])
        token_starts = _tensor([0, 3, 3])  # seg 1 empty
        token_ends = _tensor([3, 3, 6])

        issues = validate_shard_structural(
            "shard_00003", doc_ids, segment_indices, token_starts, token_ends
        )
        kinds = [i.kind for i in issues]
        assert IssueKind.EMPTY_SEGMENT in kinds
        empty = next(i for i in issues if i.kind is IssueKind.EMPTY_SEGMENT)
        assert empty.doc_id == 0
        assert empty.segment_idx == 1

    def test_non_monotonic_flagged(self):
        """token_end < token_start → NON_MONOTONIC."""
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([0, 1])
        token_starts = _tensor([0, 5])
        token_ends = _tensor([3, 4])  # 4 < 5

        issues = validate_shard_structural(
            "shard_00004", doc_ids, segment_indices, token_starts, token_ends
        )
        non_monotonic = [i for i in issues if i.kind is IssueKind.NON_MONOTONIC]
        assert len(non_monotonic) == 1
        assert non_monotonic[0].doc_id == 0
        assert non_monotonic[0].segment_idx == 1

    def test_negative_start_flagged(self):
        """token_start < 0 → NEGATIVE_START."""
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([0, 1])
        token_starts = _tensor([-1, 3])
        token_ends = _tensor([3, 6])

        issues = validate_shard_structural(
            "shard_00005", doc_ids, segment_indices, token_starts, token_ends
        )
        neg = [i for i in issues if i.kind is IssueKind.NEGATIVE_START]
        assert len(neg) == 1
        assert neg[0].doc_id == 0
        assert neg[0].segment_idx == 0

    def test_non_contiguous_within_doc_flagged(self):
        """Same doc but token_end[i] != token_start[i+1] → NON_CONTIGUOUS."""
        doc_ids = _tensor([0, 0, 0])
        segment_indices = _tensor([0, 1, 2])
        token_starts = _tensor([0, 3, 7])  # gap between seg 1 (end 5) and seg 2
        token_ends = _tensor([3, 5, 10])

        issues = validate_shard_structural(
            "shard_00006", doc_ids, segment_indices, token_starts, token_ends
        )
        gaps = [i for i in issues if i.kind is IssueKind.NON_CONTIGUOUS_WITHIN_DOC]
        assert len(gaps) == 1
        assert gaps[0].doc_id == 0
        assert gaps[0].segment_idx == 2

    def test_contiguity_is_per_doc_not_global(self):
        """Different docs may have independent token ranges without flagging."""
        doc_ids = _tensor([0, 0, 1, 1])
        segment_indices = _tensor([0, 1, 0, 1])
        # doc 0 ends at 5, doc 1 starts at 0 — intentional
        token_starts = _tensor([0, 3, 0, 4])
        token_ends = _tensor([3, 5, 4, 8])

        issues = validate_shard_structural(
            "shard_00007", doc_ids, segment_indices, token_starts, token_ends
        )
        assert issues == []

    def test_first_segment_of_doc_must_start_at_zero(self):
        """The seg_idx=0 segment of a doc must start at token 0 (BOS)."""
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([0, 1])
        token_starts = _tensor([1, 4])  # seg 0 starts at 1 — BOS missing
        token_ends = _tensor([4, 8])

        issues = validate_shard_structural(
            "shard_00008", doc_ids, segment_indices, token_starts, token_ends
        )
        nonzero_first = [i for i in issues if i.kind is IssueKind.FIRST_SEGMENT_NONZERO]
        assert len(nonzero_first) == 1
        assert nonzero_first[0].doc_id == 0
        assert nonzero_first[0].segment_idx == 0

    def test_docs_without_segment_idx_zero_are_not_flagged_for_bos(self):
        """If a doc's first recorded seg has seg_idx > 0 (cropped doc), don't
        flag it for missing BOS — only seg_idx==0 must start at token 0.
        """
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([3, 4])  # cropped; no seg 0/1/2 in shard
        token_starts = _tensor([10, 14])
        token_ends = _tensor([14, 18])

        issues = validate_shard_structural(
            "shard_00009", doc_ids, segment_indices, token_starts, token_ends
        )
        assert [i for i in issues if i.kind is IssueKind.FIRST_SEGMENT_NONZERO] == []

    def test_multi_issue_single_segment_reports_each(self):
        """A segment can trigger several issues at once; all are reported."""
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([0, 1])
        token_starts = _tensor([-2, 8])   # seg 0 negative
        token_ends = _tensor([-2, 5])     # seg 0 empty+negative; seg 1 non-monotonic

        issues = validate_shard_structural(
            "shard_00010", doc_ids, segment_indices, token_starts, token_ends
        )
        kinds = {i.kind for i in issues}
        assert IssueKind.NEGATIVE_START in kinds
        assert IssueKind.EMPTY_SEGMENT in kinds
        assert IssueKind.NON_MONOTONIC in kinds

    def test_zero_segment_shard_is_noop(self):
        """Empty shard (no segments) produces no issues."""
        doc_ids = _tensor([])
        segment_indices = _tensor([])
        token_starts = _tensor([])
        token_ends = _tensor([])

        issues = validate_shard_structural(
            "shard_00011", doc_ids, segment_indices, token_starts, token_ends
        )
        assert issues == []

    def test_unordered_segments_within_doc_are_handled(self):
        """Segments within a doc can appear in any order in the shard; the
        validator must sort them by seg_idx before checking contiguity.
        """
        # segments of doc 0 appear in order: seg 1 (row 0), seg 0 (row 1)
        doc_ids = _tensor([0, 0])
        segment_indices = _tensor([1, 0])
        # Row order values: seg 1 rows first, so row 0 is seg 1 tokens, row 1 is seg 0
        token_starts = _tensor([3, 0])
        token_ends = _tensor([5, 3])

        issues = validate_shard_structural(
            "shard_00012", doc_ids, segment_indices, token_starts, token_ends
        )
        assert issues == []

    def test_issues_carry_shard_stem(self):
        """Every emitted issue carries the passed shard_stem."""
        doc_ids = _tensor([0])
        segment_indices = _tensor([0])
        token_starts = _tensor([2])
        token_ends = _tensor([2])  # empty

        issues = validate_shard_structural(
            "shard_XYZ", doc_ids, segment_indices, token_starts, token_ends
        )
        assert all(i.shard_stem == "shard_XYZ" for i in issues)


class TestValidateShardDeep:
    """Tests for ``validate_shard_deep`` — the recompute-and-compare path."""

    def test_fresh_overlay_passes(self):
        """An overlay written by the current pipeline matches the recompute
        exactly and produces no issues."""
        setup = _make_two_doc_setup()
        issues = validate_shard_deep(
            shard_stem="shard_00000",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=setup["overlay_token_starts"],
            overlay_token_ends=setup["overlay_token_ends"],
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        assert issues == []

    def test_corrupted_overlay_flagged(self):
        """Perturbing one overlay row produces exactly one RECOMPUTE_MISMATCH
        for the right (doc_id, segment_idx)."""
        setup = _make_two_doc_setup()
        overlay_starts = setup["overlay_token_starts"].clone()
        overlay_ends = setup["overlay_token_ends"].clone()
        # Row 1 is doc 0 segment 1. Shift its end by 1 — guaranteed wrong.
        overlay_ends[1] = int(overlay_ends[1].item()) - 1

        issues = validate_shard_deep(
            shard_stem="shard_corrupt",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=overlay_starts,
            overlay_token_ends=overlay_ends,
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        mismatches = [i for i in issues if i.kind is IssueKind.RECOMPUTE_MISMATCH]
        assert len(mismatches) == 1
        assert mismatches[0].doc_id == 0
        assert mismatches[0].segment_idx == 1
        assert mismatches[0].shard_stem == "shard_corrupt"

    def test_multiple_corrupted_rows_each_flagged(self):
        """Each mismatched row yields its own issue — nothing batched away."""
        setup = _make_two_doc_setup()
        overlay_starts = setup["overlay_token_starts"].clone()
        overlay_ends = setup["overlay_token_ends"].clone()
        # Corrupt doc 0 seg 0 start AND doc 1 seg 1 end
        overlay_starts[0] = int(overlay_starts[0].item()) + 1
        overlay_ends[3] = int(overlay_ends[3].item()) + 2

        issues = validate_shard_deep(
            shard_stem="shard_multi",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=overlay_starts,
            overlay_token_ends=overlay_ends,
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        mismatches = sorted(
            (i.doc_id, i.segment_idx)
            for i in issues
            if i.kind is IssueKind.RECOMPUTE_MISMATCH
        )
        assert mismatches == [(0, 0), (1, 1)]

    def test_recompute_exception_is_captured(self):
        """If the current pipeline raises for a doc, the deep validator
        must capture it as a RECOMPUTE_EXCEPTION issue instead of letting
        the exception propagate — so a partial shard still produces a
        complete report."""
        setup = _make_two_doc_setup()
        # Break doc 0's segment char ranges so the recompute raises:
        # send segment 0 past the document's total char length.
        char_starts = setup["segment_char_starts"].clone()
        char_ends = setup["segment_char_ends"].clone()
        char_starts[0] = 10_000
        char_ends[0] = 10_050
        char_starts[1] = 10_050
        char_ends[1] = 10_100

        issues = validate_shard_deep(
            shard_stem="shard_raises",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=char_starts,
            segment_char_ends=char_ends,
            overlay_token_starts=setup["overlay_token_starts"],
            overlay_token_ends=setup["overlay_token_ends"],
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        exceptions = [i for i in issues if i.kind is IssueKind.RECOMPUTE_EXCEPTION]
        assert len(exceptions) >= 1
        # Every exception issue carries the failing doc_id (never None).
        assert all(i.doc_id is not None for i in exceptions)

    def test_empty_shard_returns_empty(self):
        """Zero segments ⇒ no issues and no tokenizer calls."""
        issues = validate_shard_deep(
            shard_stem="shard_empty",
            doc_ids=_tensor([]),
            segment_indices=_tensor([]),
            segment_char_starts=_tensor([]),
            segment_char_ends=_tensor([]),
            overlay_token_starts=_tensor([]),
            overlay_token_ends=_tensor([]),
            texts=[],
            tokenizer=_WordTokenizer(),
            bos_token=_WordTokenizer.BOS_ID,
            source_bos_char_len=5,
        )
        assert issues == []

    def test_issues_carry_shard_stem(self):
        """Every deep-validator issue must carry the passed shard_stem."""
        setup = _make_two_doc_setup()
        overlay_ends = setup["overlay_token_ends"].clone()
        overlay_ends[1] = int(overlay_ends[1].item()) + 5

        issues = validate_shard_deep(
            shard_stem="shard_labeled",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=setup["overlay_token_starts"],
            overlay_token_ends=overlay_ends,
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        assert issues
        assert all(i.shard_stem == "shard_labeled" for i in issues)


class TestValidateShard:
    """Tests for ``validate_shard`` — the per-shard orchestrator that
    runs structural + optional deep checks and returns a
    ``ShardValidationResult``.
    """

    def test_clean_shard_structural_only(self):
        """A clean shard with ``deep=False`` runs structural checks only and
        returns an ``ok`` result."""
        setup = _make_two_doc_setup()
        result = validate_shard(
            shard_stem="shard_00000",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=setup["overlay_token_starts"],
            overlay_token_ends=setup["overlay_token_ends"],
            n_docs=len(setup["texts"]),
            deep=False,
            texts=None,
            tokenizer=None,
            bos_token=None,
            source_bos_char_len=None,
        )
        assert isinstance(result, ShardValidationResult)
        assert result.shard_stem == "shard_00000"
        assert result.ok
        assert result.n_segments == int(setup["doc_ids"].numel())
        assert result.n_docs == len(setup["texts"])

    def test_clean_shard_deep(self):
        """A clean shard with ``deep=True`` runs both checks and still
        returns ``ok``."""
        setup = _make_two_doc_setup()
        result = validate_shard(
            shard_stem="shard_00000",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=setup["overlay_token_starts"],
            overlay_token_ends=setup["overlay_token_ends"],
            n_docs=len(setup["texts"]),
            deep=True,
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        assert result.ok

    def test_structural_failure_short_circuits_deep(self):
        """If structural checks find a length mismatch, the deep checks
        must not run (they'd index into a wrong-sized overlay and crash)."""
        setup = _make_two_doc_setup()
        # Drop the last overlay row so length mismatches base shard.
        result = validate_shard(
            shard_stem="shard_bad_len",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=setup["overlay_token_starts"][:-1],
            overlay_token_ends=setup["overlay_token_ends"][:-1],
            n_docs=len(setup["texts"]),
            deep=True,
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        kinds = [i.kind for i in result.issues]
        assert IssueKind.LENGTH_MISMATCH in kinds
        assert IssueKind.RECOMPUTE_MISMATCH not in kinds
        assert IssueKind.RECOMPUTE_EXCEPTION not in kinds

    def test_deep_detects_corruption(self):
        """A corrupted overlay is passed by structural but caught by deep."""
        setup = _make_two_doc_setup()
        overlay_starts = setup["overlay_token_starts"].clone()
        overlay_ends = setup["overlay_token_ends"].clone()
        # Perturb both boundaries of row 2 (doc 1, seg 0) keeping it
        # structurally contiguous: widen seg 0 by 1 token and match seg 1.
        overlay_ends[2] = int(overlay_ends[2].item()) + 1
        overlay_starts[3] = int(overlay_starts[3].item()) + 1

        result = validate_shard(
            shard_stem="shard_corrupt",
            doc_ids=setup["doc_ids"],
            segment_indices=setup["segment_indices"],
            segment_char_starts=setup["segment_char_starts"],
            segment_char_ends=setup["segment_char_ends"],
            overlay_token_starts=overlay_starts,
            overlay_token_ends=overlay_ends,
            n_docs=len(setup["texts"]),
            deep=True,
            texts=setup["texts"],
            tokenizer=setup["tokenizer"],
            bos_token=setup["bos_token"],
            source_bos_char_len=setup["source_bos_char_len"],
        )
        mismatches = [i for i in result.issues if i.kind is IssueKind.RECOMPUTE_MISMATCH]
        assert mismatches, "expected recompute mismatch for corrupted row"

    def test_deep_requires_texts_when_enabled(self):
        """``deep=True`` without ``texts`` is a programming error."""
        setup = _make_two_doc_setup()
        with pytest.raises(ValueError, match="texts"):
            validate_shard(
                shard_stem="shard_00000",
                doc_ids=setup["doc_ids"],
                segment_indices=setup["segment_indices"],
                segment_char_starts=setup["segment_char_starts"],
                segment_char_ends=setup["segment_char_ends"],
                overlay_token_starts=setup["overlay_token_starts"],
                overlay_token_ends=setup["overlay_token_ends"],
                n_docs=len(setup["texts"]),
                deep=True,
                texts=None,
                tokenizer=setup["tokenizer"],
                bos_token=setup["bos_token"],
                source_bos_char_len=setup["source_bos_char_len"],
            )


class TestIterIssuesByKind:
    """Unit tests for the issue-grouping helper."""

    def test_groups_by_kind(self):
        i1 = ValidationIssue(IssueKind.EMPTY_SEGMENT, "s", 0, 1, "")
        i2 = ValidationIssue(IssueKind.EMPTY_SEGMENT, "s", 0, 2, "")
        i3 = ValidationIssue(IssueKind.NEGATIVE_START, "s", 1, 0, "")
        grouped = iter_issues_by_kind([i1, i2, i3])
        assert grouped[IssueKind.EMPTY_SEGMENT] == [i1, i2]
        assert grouped[IssueKind.NEGATIVE_START] == [i3]

    def test_empty_input_returns_empty_dict(self):
        assert iter_issues_by_kind([]) == {}


# ---------------------------------------------------------------------------
# End-to-end CLI tests (use real nanochat tokenizer; slow but realistic)
# ---------------------------------------------------------------------------


def _encoding_with_nanochat_specials() -> tiktoken.Encoding:
    base = tiktoken.get_encoding("gpt2")
    special_tokens = {
        token: len(base._mergeable_ranks) + i
        for i, token in enumerate(SPECIAL_TOKENS)
    }
    return tiktoken.Encoding(
        name="nanochat-token-boundary-test",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )


@pytest.fixture
def nanochat_tokenizer_cache(tmp_path: Path, monkeypatch):
    base_dir = tmp_path / "nanochat_cache"
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer = RustBPETokenizer(_encoding_with_nanochat_specials(), "<bos>")
    tokenizer.save(tokenizer_dir)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))
    return base_dir


def _build_synthetic_feature_dir(tmp_path: Path, *, corrupt_overlay: bool = False):
    """Build a tiny feature dir with one shard, one parquet, and one
    overlay that was written by the real ``precompute_shard`` path.

    When ``corrupt_overlay=True``, the overlay's first segment's
    ``token_ends`` entry is shifted by +1 token to deliberately introduce
    a recompute mismatch.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    checksum = compute_tokenizer_checksum(tokenizer)

    # Two short docs, a few segments each.
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A second simple sentence for testing validators",
    ]
    shard_stem = "shard_00000"
    pq_path = data_dir / f"{shard_stem}.parquet"
    pq.write_table(pa.table({"text": texts}), str(pq_path))

    # Build the base shard metadata with 2 segments per doc, each
    # covering roughly half the document's nanochat char range.
    doc_ids_rows: list[int] = []
    seg_indices_rows: list[int] = []
    char_starts_rows: list[int] = []
    char_ends_rows: list[int] = []
    doc_token_counts_list: list[int] = []
    total_segs = 0
    for doc_id, text in enumerate(texts):
        tokens = tokenizer.encode(text, prepend=bos_token)
        char_offsets = token_to_char_offsets(tokenizer, tokens)
        total = char_offsets[-1]
        mid = total // 2
        # Pin mid to a token boundary so the first segment covers a
        # whole-number of tokens (avoids spurious split-inside-a-token
        # issues in the test fixture).
        mid = min(
            char_offsets,
            key=lambda off: (abs(off - mid), off),
        )
        if mid in (0, total):
            mid = char_offsets[len(char_offsets) // 2]
        doc_ids_rows.extend([doc_id, doc_id])
        seg_indices_rows.extend([0, 1])
        char_starts_rows.extend([0, mid])
        char_ends_rows.extend([mid, total])
        doc_token_counts_list.append(len(tokens))
        total_segs += 2

    shard_data = {
        **_one_feature_per_segment_payload(total_segs),
        "doc_ids": t.tensor(doc_ids_rows, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices_rows, dtype=t.int32),
        "segment_positions": t.zeros(total_segs, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts_list, dtype=t.int32),
        "segment_char_starts": t.tensor(char_starts_rows, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends_rows, dtype=t.int32),
        "n_docs": len(texts),
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 16,
    }
    save_features(shard_data, feature_dir / f"{shard_stem}_features")

    # Run the real precompute path to populate the overlay directory.
    output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    write_manifest(output, checksum, OverlayKind.SEG_BOUNDARY)
    shard_path = next(feature_dir.glob(f"{shard_stem}_features.*"))
    precompute_shard(
        shard_path=shard_path,
        parquet_path=pq_path,
        tokenizer=tokenizer,
        bos_token=bos_token,
        output_dir=output,
        shard_stem=shard_stem,
        source_bos_char_len=len(tokenizer.decode([bos_token])),
        source_segment_size=16,
        dry_run=False,
        # Relax the default multiplier-based bounds so our tiny synthetic
        # docs (5-10 tokens/segment) don't trip the validator's floors.
        min_nanochat_tokens_per_segment=1,
        max_nanochat_tokens_per_segment=1024,
    )

    if corrupt_overlay:
        overlay_path = overlay_shard_path(output, shard_stem)
        overlay = load_overlay_shard(overlay_path)
        starts = overlay[TensorKey.SEGMENT_TOKEN_STARTS].clone()
        ends = overlay[TensorKey.SEGMENT_TOKEN_ENDS].clone()
        # Shift doc 0 seg 0 end by +1 token (and doc 0 seg 1 start by +1)
        # so the overlay stays structurally contiguous but diverges from
        # the recompute.
        ends[0] = int(ends[0].item()) + 1
        starts[1] = int(starts[1].item()) + 1
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: starts,
                TensorKey.SEGMENT_TOKEN_ENDS: ends,
            },
            overlay_path,
        )

    return feature_dir, data_dir


def _invoke_cli(feature_dir, data_dir, *, deep: bool) -> subprocess.CompletedProcess:
    """Invoke the CLI entry via ``python -m`` so argv/Hydra parsing drift is caught."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.utils.validate_token_boundary_overlays",
            f"feature_dir={feature_dir}",
            f"data_dir={data_dir}",
            f"deep={'true' if deep else 'false'}",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ.copy(),
        check=False,
    )


@pytest.mark.usefixtures("nanochat_tokenizer_cache")
class TestCliEndToEnd:
    """End-to-end tests for the CLI over a real synthetic feature dir.

    Tests invoke the script via ``python -m`` (subprocess) rather than
    calling ``run()`` directly so regressions in the Hydra entrypoint,
    argv parsing, and module-scope definition order are all caught.
    """

    def test_run_on_clean_overlay_returns_zero(self, tmp_path):
        """A freshly-computed overlay validates successfully; CLI exits 0."""
        feature_dir, data_dir = _build_synthetic_feature_dir(tmp_path)
        proc = _invoke_cli(feature_dir, data_dir, deep=True)
        assert proc.returncode == 0, (
            f"CLI failed (rc={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

    def test_run_on_corrupted_overlay_returns_nonzero(self, tmp_path):
        """Deliberately corrupted overlay: deep-path CLI exits non-zero."""
        feature_dir, data_dir = _build_synthetic_feature_dir(
            tmp_path, corrupt_overlay=True
        )
        proc = _invoke_cli(feature_dir, data_dir, deep=True)
        assert proc.returncode != 0

    def test_run_structural_only_on_corrupted_overlay_still_passes(self, tmp_path):
        """Structural-only CLI still exits 0 on a deep-only corruption."""
        feature_dir, data_dir = _build_synthetic_feature_dir(
            tmp_path, corrupt_overlay=True
        )
        proc = _invoke_cli(feature_dir, data_dir, deep=False)
        assert proc.returncode == 0, (
            f"CLI failed (rc={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

    def test_run_missing_overlay_returns_nonzero(self, tmp_path):
        """Missing overlay dir: CLI exits non-zero."""
        feature_dir, data_dir = _build_synthetic_feature_dir(tmp_path)
        shutil.rmtree(feature_dir / "tok")
        proc = _invoke_cli(feature_dir, data_dir, deep=True)
        assert proc.returncode != 0


@pytest.mark.usefixtures("nanochat_tokenizer_cache")
class TestModuleCanBeInvokedAsMain:
    """Regression test: the script must work when run as ``python -m`` and
    not just when imported.

    A previous edit accidentally placed ``validate_shard_deep`` after the
    ``if __name__ == "__main__": main()`` block. Importing the module
    succeeded (because all top-level statements run before pytest inspects
    the module), so every existing pytest passed — but invoking the script
    as ``python -m scripts.utils.validate_token_boundary_overlays`` raised
    ``NameError: name 'validate_shard_deep' is not defined`` partway
    through the run, after a worker had already started chewing on real
    data.

    This test catches that class of bug by actually spawning a subprocess
    and checking the script gets through enough of the pipeline without
    crashing.
    """

    def test_python_m_invocation_smoke_test(self, tmp_path):
        feature_dir, data_dir = _build_synthetic_feature_dir(tmp_path)
        proc = _invoke_cli(feature_dir, data_dir, deep=True)
        assert proc.returncode == 0, (
            f"`python -m` invocation failed (rc={proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
        # Catch any definition-order class of bug.
        assert "NameError" not in proc.stderr
        assert "AttributeError" not in proc.stderr


class TestValidatorCatchesZeroWidthMiddleSegment:
    """Confirm that the deep validator catches an overlay backed by a base
    shard with a zero-width source segment in the middle of a document.

    This is the failure mode that ``shard_00355/doc_id=15373`` exhibited
    in production: a chunked-offset bug in
    ``_token_to_char_offsets_chunked`` produced a phantom ``[char, char)``
    segment, and the previously-removed "Shape 2" repair path silently
    filled it with an arbitrary nanochat token. With Shape 2 gone, the
    recompute can't assign anything to a zero-width segment, so
    ``_extract_ranges_from_monotonic`` raises and the validator must
    capture it as a ``RECOMPUTE_EXCEPTION`` issue (not let it crash).
    """

    def test_zero_width_middle_segment_flagged_as_recompute_exception(self):
        tok = _WordTokenizer()
        bos = tok.get_bos_token_id()
        source_bos_char_len = 5  # "<bos>"

        # Single doc with 4 source segments. The doc has 8 words → 9
        # nanochat tokens total. Char ranges in BOS-prepended coords:
        #   tok 0 (BOS) covers [0, 5)
        #   tok 1 (w000) covers [5, 10)
        #   tok 2 (w001) covers [10, 15)
        #   tok 3 (w002) covers [15, 20)
        #   tok 4 (w003) covers [20, 25)
        #   tok 5 (w004) covers [25, 30)
        #   tok 6 (w005) covers [30, 35)
        #   tok 7 (w006) covers [35, 40)
        #   tok 8 (w007) covers [40, 45)
        # Source segments split the char range [0, 45):
        #   seg 0: [0, 15)   covers BOS, w000, w001
        #   seg 1: [15, 25)  covers w002, w003
        #   seg 2: [25, 25)  ← ZERO-WIDTH PHANTOM SEGMENT
        #   seg 3: [25, 45)  covers w004..w007
        text = " ".join(f"w{i:03d}" for i in range(8))
        texts = [text]
        doc_ids = _tensor([0, 0, 0, 0])
        seg_indices = _tensor([0, 1, 2, 3])
        seg_char_starts = _tensor([0, 15, 25, 25])
        seg_char_ends = _tensor([15, 25, 25, 45])
        # Overlay tensors are arbitrary — they don't matter because the
        # recompute will raise before they're inspected. Use plausible
        # contiguous values so structural checks would also pass.
        overlay_token_starts = _tensor([0, 3, 5, 5])
        overlay_token_ends = _tensor([3, 5, 5, 9])

        issues = validate_shard_deep(
            shard_stem="shard_zero_width",
            doc_ids=doc_ids,
            segment_indices=seg_indices,
            segment_char_starts=seg_char_starts,
            segment_char_ends=seg_char_ends,
            overlay_token_starts=overlay_token_starts,
            overlay_token_ends=overlay_token_ends,
            texts=texts,
            tokenizer=tok,
            bos_token=bos,
            source_bos_char_len=source_bos_char_len,
        )
        # Recompute MUST fail to assign tokens to seg 2; the validator
        # must capture it (not propagate) and label it RECOMPUTE_EXCEPTION
        # for the doc that triggered it.
        exception_issues = [
            i for i in issues if i.kind is IssueKind.RECOMPUTE_EXCEPTION
        ]
        assert len(exception_issues) == 1, (
            f"expected exactly 1 RECOMPUTE_EXCEPTION, got: {issues!r}"
        )
        assert exception_issues[0].doc_id == 0
        # The exception is at the doc level — not pinned to a specific seg
        # because the failure happens during contiguous-run extraction.
        assert exception_issues[0].segment_idx is None


# ---------------------------------------------------------------------------
# Pre-fix snapshot of _token_to_char_offsets_chunked
#
# This is byte-for-byte the function as it existed on the parent commit
# (310e707) before being replaced by the bail-on-divergence fix in
# steering/segmentation.py. We keep it here so the realistic real-Gemma
# end-to-end test below can produce the same buggy offsets the production
# extractor was producing for shard_00355/doc 15373.
# ---------------------------------------------------------------------------


def _buggy_chunked_pre_fix(tokenizer, token_ids, per_token_texts):
    """Snapshot of `_token_to_char_offsets_chunked` from before the fix.

    Reproduces the chunk-boundary walk-back over-correction that produced
    zero-width source segments in production. Do not use outside tests.
    """
    n = len(token_ids)
    chunk_size = _HYBRID_CHUNK_SIZE
    per_tok_lens = [len(s) for s in per_token_texts]

    offsets = [0] * (n + 1)
    chunk_start = 0
    n_divergent = 0

    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_size, n)

        optimistic_end = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            optimistic_end += per_tok_lens[i]

        actual_prefix_len = len(tokenizer.decode(token_ids[:chunk_end]))

        if actual_prefix_len == optimistic_end:
            pos = offsets[chunk_start]
            for i in range(chunk_start, chunk_end):
                pos += per_tok_lens[i]
                offsets[i + 1] = pos
        else:
            n_divergent += 1
            n_chunks_done = (chunk_end + chunk_size - 1) // chunk_size
            if n_divergent >= 2 and n_chunks_done <= 4:
                return _token_to_char_offsets_progressive(tokenizer, token_ids)

            for i in range(chunk_start, chunk_end):
                prefix = tokenizer.decode(token_ids[: i + 1])
                new_len = len(prefix)

                if new_len < offsets[i]:
                    base = max(0, new_len - 1)
                    for j in range(i, 0, -1):
                        if offsets[j] <= base:
                            break
                        offsets[j] = base

                offsets[i + 1] = new_len

        chunk_start = chunk_end

    return offsets


@pytest.fixture(scope="module")
def gemma_tok():
    """Real Gemma tokenizer; skip the test if it can't be loaded."""
    try:
        return AutoTokenizer.from_pretrained("google/gemma-2-2b")
    except Exception:
        pytest.skip("Gemma tokenizer not available")


@pytest.mark.usefixtures("nanochat_tokenizer_cache")
class TestRealGemmaValidatorCatchesProductionBug:
    """End-to-end realistic reproduction of the shard_00355 production bug.

    Uses the real Gemma tokenizer + the real Old Italic inscription text
    that triggered the chunked-offset walk-back over-correction. Produces
    the exact same buggy offsets the pre-fix extractor would have written
    by snapshotting the pre-fix ``_token_to_char_offsets_chunked``
    implementation, builds a real feature directory + parquet + overlay
    around it, and runs the validator's full ``run()`` pipeline against
    them.

    This is the bridge between:
      1. ``test_chunked_agrees_with_progressive_on_long_byte_fallback_run``
         (in ``tests/test_token_to_char_offsets.py``) — proves the bug
         was real and the fix resolves it, against the real Gemma
         tokenizer + Old Italic text.
      2. ``test_zero_width_middle_segment_flagged_as_recompute_exception``
         (above) — proves the validator catches a zero-width middle
         source segment, using a synthetic mock tokenizer for
         deterministic test data.

    The realistic chain proven by THIS test:
      real Gemma tokenizer
        → real Old Italic text from shard_00355
        → pre-fix buggy chunked offsets (snapshot)
        → real base shard with zero-width segment 6
        → real validator.run() invocation
        → validator catches the failure as RECOMPUTE_EXCEPTION
    """

    # Same fixture text as TestChunkedVsProgressiveAgreement uses for the
    # chunked-vs-progressive disagreement reproduction.
    KHAROSHTHI_REPRO_TEXT = (
        "x " + "𐌇𐌀𐌓𐌉𐌙𐌀𐌔𐌕𐌉𐌕𐌄𐌉𐌅𐌀" * 4 + " y"
    )

    def test_validator_catches_real_gemma_zero_width_segment(
        self, tmp_path, gemma_tok
    ):
        # ── Step 1: tokenize the real Old Italic text with real Gemma ──
        gemma_ids = gemma_tok(
            self.KHAROSHTHI_REPRO_TEXT, return_tensors=None
        )["input_ids"]
        per_token_texts = [
            gemma_tok.decode([tid], skip_special_tokens=False) for tid in gemma_ids
        ]

        # Wrap so the snapshot helper sees an object with a `decode(ids)` method.
        class _GemmaAdapter:
            def __init__(self, t):
                self._t = t
            def decode(self, ids):
                return self._t.decode(ids, skip_special_tokens=False)

        adapter = _GemmaAdapter(gemma_tok)

        # ── Step 2: produce the buggy offsets the pre-fix code would have ──
        buggy_offsets = _buggy_chunked_pre_fix(adapter, gemma_ids, per_token_texts)

        # Sanity: the buggy offsets must contain at least one segment-sized
        # zero-width range. With segment_size=32 we look at every 32-token
        # window. If this assertion fails, the test fixture text is no longer
        # tickling the bug and the test would be a no-op.
        SEG = 32
        n_segs = len(gemma_ids) // SEG
        zero_width = [
            i for i in range(n_segs)
            if buggy_offsets[i * SEG] == buggy_offsets[(i + 1) * SEG]
        ]
        assert zero_width, (
            "fixture regression: pre-fix chunked offsets no longer produce "
            "any zero-width segment for the Old Italic text. Either the "
            "Gemma tokenizer changed or the snapshot drifted."
        )

        # ── Step 3: build a real feature dir + parquet + base shard ──
        nanochat_tok = get_tokenizer()
        checksum = compute_tokenizer_checksum(nanochat_tok)

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        feature_dir = tmp_path / "features"
        feature_dir.mkdir(parents=True)

        shard_stem = "shard_00000"
        pq_path = data_dir / f"{shard_stem}.parquet"
        # Single-doc parquet containing the real Old Italic text.
        pq.write_table(
            pa.table({"text": [self.KHAROSHTHI_REPRO_TEXT]}), str(pq_path)
        )

        # Build the base shard with one doc, n_segs source segments. The
        # char ranges come straight from the buggy offsets, so segment 6
        # (or whichever index is in zero_width) really has start == end.
        seg_char_starts = [buggy_offsets[i * SEG] for i in range(n_segs)]
        seg_char_ends = [buggy_offsets[(i + 1) * SEG] for i in range(n_segs)]
        n_tokens_in_doc = n_segs * SEG  # we drop the trailing partial seg

        shard_data = {
            **_one_feature_per_segment_payload(n_segs),
            "doc_ids": t.zeros(n_segs, dtype=t.int32),  # all in doc 0
            "segment_indices": t.arange(n_segs, dtype=t.int32),
            "segment_positions": t.tensor(
                [i * SEG for i in range(n_segs)], dtype=t.int32
            ),
            "doc_token_counts": t.tensor([n_tokens_in_doc], dtype=t.int32),
            "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
            "n_docs": 1,
            "n_segments": n_segs,
            "layer": 8,
            "context_window": 2048,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": SEG,
        }
        save_features(shard_data, feature_dir / f"{shard_stem}_features")

        # ── Step 4: build a structurally-plausible overlay ──
        # The exact values don't matter; the recompute will raise before
        # comparing them. Use a contiguous monotone range so structural
        # checks pass cleanly and the deep stage runs.
        overlay_starts = t.zeros(n_segs, dtype=t.int32)
        overlay_ends = t.zeros(n_segs, dtype=t.int32)
        cursor = 0
        per_seg_tokens = max(1, n_tokens_in_doc // n_segs)
        for i in range(n_segs):
            overlay_starts[i] = cursor
            cursor += per_seg_tokens
            overlay_ends[i] = cursor

        output = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
        write_manifest(output, checksum, OverlayKind.SEG_BOUNDARY)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: overlay_starts,
                TensorKey.SEGMENT_TOKEN_ENDS: overlay_ends,
            },
            overlay_shard_path(output, shard_stem),
        )

        # ── Step 5: run the validator end-to-end ──
        cfg = OmegaConf.create(
            {
                "feature_dir": str(feature_dir),
                "data_dir": str(data_dir),
                "deep": True,
                "shard_start": 0,
                "shard_end": -1,
                "max_shards": None,
                "fail_fast": False,
                "max_issues_per_shard": 50,
            }
        )
        rc = run(cfg)
        # Validator must surface the zero-width-segment failure as a non-zero
        # exit code.  rc == 0 means it silently accepted a broken shard.
        assert rc != 0, (
            "Validator failed to catch a zero-width middle segment produced "
            "by the real pre-fix chunked offset bug on real Old Italic text. "
            "This is the production bug from shard_00355/doc_id=15373."
        )
