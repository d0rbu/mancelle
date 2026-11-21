"""Tests for BOS-prepend precondition probe at splice_feature_shard entry (FIX-10).

Verifies that splice_feature_shard raises TokenizerContractError when a tokenizer
advertises a BOS token but does not prepend it by default, before any shard I/O.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from nanochat.atomic_io import atomic_write_parquet
from steering.feature_io import save_features
from steering.feature_shard_splice import (
    TokenizerContractError,
    splice_feature_shard,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

# Real tokenizer: BOS=500 and CharMergeTokenizer always prepends BOS.
_GOOD_TOK = CharMergeTokenizer(bos_id=500)

# BOS-less tokenizer (no bos_id): probe must be skipped entirely.
_NOBOS_TOK = CharMergeTokenizer(bos_id=None)

SEGMENT_SIZE = 2
CONTEXT_WINDOW = 128


class _BosNotPrependedTokenizer:
    """Advertises bos_token_id=999 but never prepends it.

    Encoding: chars → ord values (no BOS prepend).
    Decoding: chars from ord values.
    """

    bos_token_id = 999

    def __call__(self, texts, *, padding=False, add_special_tokens=True):
        if isinstance(texts, list):
            return {"input_ids": [[ord(c) for c in text] for text in texts]}
        return {"input_ids": [ord(c) for c in texts]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


_BAD_TOK = _BosNotPrependedTokenizer()


def _make_fixtures(tmp_path, tok, segment_size=SEGMENT_SIZE, context_window=CONTEXT_WINDOW):
    """Create a base shard + parquet for the given tokenizer."""
    texts = ["abcde"]
    base_dict = make_base_shard(texts, tok, segment_size, context_window=context_window)
    base_path = save_features(base_dict, tmp_path / "shard_00000")
    parquet_path = tmp_path / "shard_00000.parquet"
    atomic_write_parquet(pa.table({"text": texts}), parquet_path)
    return base_path, parquet_path, texts


# ---------------------------------------------------------------------------
# Test (i): gemma-2-2b style passes (BOS defined and prepended)
# ---------------------------------------------------------------------------


def test_bos_prepend_probe_passes_for_good_tokenizer(tmp_path):
    """TokenizerContractError is NOT raised when BOS is correctly prepended."""
    base_path, parquet_path, _ = _make_fixtures(tmp_path, _GOOD_TOK)
    # Should not raise
    splice_feature_shard(
        base_path, parquet_path, _GOOD_TOK,
        model="test", cut_fraction=0.0, seed=0,
        shard_index=0, shard_name="shard_00000",
        segment_size=SEGMENT_SIZE, context_window=CONTEXT_WINDOW, sae_max_context=None,
    )


# ---------------------------------------------------------------------------
# Test (ii): CharMergeTokenizer with bos passes
# ---------------------------------------------------------------------------


def test_bos_prepend_probe_passes_for_charmarge_bos(tmp_path):
    """TokenizerContractError is NOT raised for CharMergeTokenizer with bos_id."""
    tok = CharMergeTokenizer(bos_id=42)
    base_path, parquet_path, _ = _make_fixtures(tmp_path, tok)
    splice_feature_shard(
        base_path, parquet_path, tok,
        model="test", cut_fraction=0.0, seed=0,
        shard_index=0, shard_name="shard_00000",
        segment_size=SEGMENT_SIZE, context_window=CONTEXT_WINDOW, sae_max_context=None,
    )


# ---------------------------------------------------------------------------
# Test (iii): tokenizer with BOS defined but not prepended -> raises
# ---------------------------------------------------------------------------


def test_bos_prepend_probe_raises_when_bos_not_prepended(tmp_path):
    """TokenizerContractError raised when BOS defined but not prepended by default."""
    # Build fixtures with the GOOD tokenizer (for a valid base shard).
    # The probe checks the TOKENIZER BEHAVIOR, not the shard content.
    # Use _GOOD_TOK for shard so alignment doesn't fail before probe would run.
    # But the probe runs BEFORE load_source_segments, so using _BAD_TOK's own fixture.
    # Actually we must use _BAD_TOK for the base shard too, else the probe would fail
    # before any I/O error. But that's fine — the probe fires first.
    # Let's just make a base shard with the good tok and see if probe fires.
    # The probe is at the TOP of splice_feature_shard BEFORE loading the shard.
    base_path, parquet_path, _ = _make_fixtures(tmp_path, _GOOD_TOK)
    with pytest.raises(TokenizerContractError) as exc_info:
        splice_feature_shard(
            base_path, parquet_path, _BAD_TOK,
            model="test", cut_fraction=0.0, seed=0,
            shard_index=0, shard_name="shard_00000",
            segment_size=SEGMENT_SIZE, context_window=CONTEXT_WINDOW, sae_max_context=None,
        )
    msg = str(exc_info.value)
    # Error must mention: tokenizer name/class, bos_id, and first observed id
    assert "999" in msg, f"bos_id not in error: {msg!r}"


# ---------------------------------------------------------------------------
# Test (iv): BOS is None -> no probe, no error even for non-prepending tokenizer
# ---------------------------------------------------------------------------


def test_bos_probe_skipped_when_bos_is_none(tmp_path):
    """No probe when tokenizer has no BOS (bos_id=None)."""
    # _NOBOS_TOK has no BOS; make base shard with it.
    texts = ["abcde"]
    base_dict = make_base_shard(texts, _NOBOS_TOK, SEGMENT_SIZE, context_window=CONTEXT_WINDOW)
    base_path = save_features(base_dict, tmp_path / "shard_00000")
    parquet_path = tmp_path / "shard_00000.parquet"
    atomic_write_parquet(pa.table({"text": texts}), parquet_path)
    # No TokenizerContractError — BOS-less tokenizers are legal
    # (recombine_segments will raise SpliceInternalError instead when cut_fraction>0,
    #  but with cut_fraction=0.0, identity result is returned)
    splice_feature_shard(
        base_path, parquet_path, _NOBOS_TOK,
        model="test", cut_fraction=0.0, seed=0,
        shard_index=0, shard_name="shard_00000",
        segment_size=SEGMENT_SIZE, context_window=CONTEXT_WINDOW, sae_max_context=None,
    )


# ---------------------------------------------------------------------------
# Test (v): probe precedes any shard work (failing probe leaves no output files)
# ---------------------------------------------------------------------------


def test_bos_probe_fires_before_any_shard_work(tmp_path):
    """TokenizerContractError fires before any shard I/O so no partial output."""
    # The base shard is valid; the tokenizer contract is violated.
    # The test verifies: the error is raised, and it is TokenizerContractError
    # (not AlignmentError, not SourceDocNotStableError, etc.).
    base_path, parquet_path, _ = _make_fixtures(tmp_path, _GOOD_TOK)
    with pytest.raises(TokenizerContractError):
        splice_feature_shard(
            base_path, parquet_path, _BAD_TOK,
            model="test", cut_fraction=0.0, seed=0,
            shard_index=0, shard_name="shard_00000",
            segment_size=SEGMENT_SIZE, context_window=CONTEXT_WINDOW, sae_max_context=None,
        )
    # No output .safetensors.zst created for the derived shard
    derived_files = list(tmp_path.glob("*derived*"))
    assert len(derived_files) == 0, f"unexpected derived files: {derived_files}"
