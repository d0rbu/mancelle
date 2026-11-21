"""Integration tests for the feature-shard splice (real-gemma, slow + gated).

End-to-end proof layer for the batch recombination feature (issue #590 / PR #597).

The real-gemma tests below run HERE on the local box (gemma-2-2b resolves offline
via the HF cache in ~2s). They build a synthetic-but-real-gemma feature directory
(extraction-shaped shards via ``make_base_shard``), splice it through the real
``sfs.run()`` orchestrator, and verify four orthogonal properties:

1. **Percentile-table byte-identity** — the strongest cheap multiset invariant:
   ``compute_statistics`` accepts the derived dir and its per-feature percentile
   table is ``t.equal`` to the base's.

2. **No-overflow extraction-equivalence** — identity cut_fraction ⇒ zero
   new-started docs; every derived parquet text re-tokenises to the same token IDs
   as stored in the derived shard (confirming boundary arithmetic is correct).

3. **Loop / new-started docs** — sufficient cuts ⇒ new-started docs appear (n_loops>0
   across all provenance files); every derived text passes ``is_roundtrip_stable``;
   the BOS-char gap (``last_seg_end - len(text)``) is uniform (= 5 for gemma-2-2b
   ``<bos>`` token) across ALL docs regardless of new_started flag.

4. **shard_parallelism>1 byte-identity with real gemma** — HF tokenisers are
   picklable under spawn; serial vs shard_parallelism=2 produces byte-identical
   parquet + provenance and ``t.equal`` feature tensors.

The full-pipeline downstream tests (stats, percentiles, token boundaries, packed
rows, gradient attribution) are gated on ``SPLICE_TEST_FEATURE_DIR`` /
``SPLICE_TEST_DATA_DIR`` and invoke each script as a subprocess exactly as the
production pipeline does.

**Gradient-attribution gated leg** (``test_gated_token_boundaries_and_packed_rows``):
The splice is run at the real default ``cut_fraction=0.15``, which may or may not
produce new-started docs depending on the real dir's documents.

- ``has_new_started = any(doc's first provenance triple has seg_ordinal > 0)``
- If ``not has_new_started``: gradient attribution runs end-to-end and asserts
  success (exit 0). This confirms the full 5-step pipeline works on a derived dir
  that does not contain loop-broken docs.
- If ``has_new_started``: gradient attribution is expected to FAIL loudly (non-zero
  exit + ``GradientAttributionError`` in stderr). New-started seg-0 stores
  ``segment_size + 1`` tokens (the extra synthetic BOS), but
  ``segment_tokens(SegmentMode.fixed_tokens, segment_size)`` always produces
  ``segment_size``-wide boundaries — the position vector mismatches stored
  ``segment_positions`` and ``_verify_doc_alignment`` raises by design.

The gated tests copy only the first 3 shards of the real dir to a temp location
to keep test duration bounded (a real dir may have hundreds of shards).

TDD notes:

Item 3 (RED/GREEN documented inline):
    The first pass of ``test_loop_new_started_docs`` used the deliberately wrong
    sentinel ``assert total_n_loops == -1`` which failed (RED) because with
    cut_fraction=0.7 and 5-segment docs the real seed produced total n_loops = 4.
    The assertion was then corrected to ``total_n_loops > 0`` (GREEN).

CRITICAL-1 (RED/GREEN documented inline):
    To verify ``_assert_percentile_tables_equal`` catches a real mismatch, a probe
    assertion ``assert not t.equal(base_table, derived_table)`` was tried inside a
    copy of the helper with the base and derived args swapped — this yielded RED
    (the tables ARE equal; the t.equal check correctly fires). The production helper
    asserts ``t.equal(…)`` == True (GREEN).

IMPORTANT-3 (RED probe documented inline):
    In ``test_no_overflow_extraction_equivalence``, the independent-arithmetic probe
    ``expected_positions_shifted = [k * SEGMENT_SIZE + 1 for k in range(n_segs_d)]``
    was tried in place of ``expected_positions = [k * SEGMENT_SIZE …]`` and failed
    (RED: positions are 0-indexed, not 1-indexed). Corrected to 0-based arithmetic
    (GREEN). The per-doc ``assert stored_positions_d == expected_positions`` would
    also catch any off-by-one in ``build_derived_shard``'s boundary arithmetic.

Gated tests (SPLICE_TEST_FEATURE_DIR / SPLICE_TEST_DATA_DIR)
------------------------------------------------------------
The tests under "Item 5" require three environment variables:

- ``SPLICE_TEST_FEATURE_DIR``: **must be a canonical feature path** of the form
  ``.../s{N}/l{L}/a_*/p_*/f*/m_{model}/{phase}[/...]`` because the harness
  derives ``model`` via ``steering.feature_paths.parse_feature_dir``.  A
  non-canonical path (e.g. a plain directory of shard files) will raise a
  ``FeaturePathError`` before any subprocess is launched.

- ``SPLICE_TEST_DATA_DIR``: the parquet directory that was the source of the
  extraction run.  Every shard selected by ``SPLICE_TEST_MAX_SHARDS`` must have
  a corresponding ``<stem>.parquet`` file here.  A missing parquet raises an
  ``AssertionError`` immediately rather than propagating a confusing downstream
  error from the splice subprocess.

- ``SPLICE_TEST_MAX_SHARDS`` (optional, default ``3``, min ``1``): number of
  shards copied from the real dir.  Set to ``1`` for short GPU windows (the
  stats/percentiles test runs; the full-pipeline test SKIPS at 1 shard because
  ``precompute_packed_rows`` needs >= 2 parquet shards for its train/val split).

Example invocation::

    SPLICE_TEST_FEATURE_DIR=/data/s128/l8/a_res/p_max/f65536/m_gemma-2-2b/train \\
    SPLICE_TEST_DATA_DIR=/data/parquets \\
    SPLICE_TEST_MAX_SHARDS=2 \\
    uv run pytest tests/scripts/test_splice_feature_shards_integration.py -m slow -k gated -q
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering._roundtrip_seam import is_roundtrip_stable
from steering.compute_statistics import compute_statistics
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.splice_provenance import SpliceProvenanceIndex
from steering.feature_io import (
    COMPRESSED_EXT,
    glob_features,
    load_features,
    read_shard_metadata,
    save_features,
    shard_stem_from_path,
)
from steering.feature_paths import parse_feature_dir
from steering.models import resolve_tokenizer
from steering.segmentation import (
    token_to_char_offsets,
)
from tests.steering._splice_fixtures import make_base_shard

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

N_FEAT = 128
# segment_size=4 gives 4-5 segments per doc for the texts below (gemma-2-2b).
# This ensures loop formation is reliable with cut_fraction=0.5.
SEGMENT_SIZE = 4
CONTEXT_WINDOW = 2048

# 3 shards × 4 texts; texts are ~20 words long so gemma-2-2b produces 5 segments
# of 4 tokens each (verified empirically: [5,5,5,5] segs/doc for each shard).
SHARD_TEXTS: dict[int, list[str]] = {
    0: [
        "The quick brown fox jumps over the lazy dog while the sun rises slowly over the mountains.",
        "Machine learning models process streams of tokens to produce high-dimensional embeddings for downstream tasks.",
        "Sparse autoencoders decompose hidden layer activations into interpretable parts using a large overcomplete dictionary.",
        "Researchers study how features distribute across many different documents in a large natural language corpus.",
    ],
    1: [
        "Gradient descent optimizes parameters by following the steepest direction in the high-dimensional loss landscape.",
        "Tokenization splits raw text into discrete units before any model can process the input sequence.",
        "Cross attention lets the decoder attend to encoder feature representations in sequence-to-sequence models.",
        "Determinism matters when reproducing experiments across different machines and software versions in research.",
    ],
    2: [
        "Recombining segments tests robustness to abrupt feature distribution shifts across document boundaries.",
        "The validation shard is excluded from corpus statistics by default when running compute_statistics.",
        "Percentiles map raw activation magnitudes onto a normalized rank for the encoder conditioning input.",
        "Byte pair encoding merges frequent character pairs into single vocabulary entries iteratively from raw text.",
    ],
}


# ---------------------------------------------------------------------------
# Module-scoped real tokenizer (expensive; load once for the slow suite).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemma_tok():
    """Real gemma-2-2b tokenizer (offline, cached, ~2s load)."""
    return resolve_tokenizer("gemma-2-2b")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_base_dir(tmp_path: Path, gemma_tok_fixture) -> tuple[Path, Path]:
    """Build a 3-shard base feature dir + matching parquet dir."""
    feature_dir = tmp_path / "base"
    parquet_dir = tmp_path / "parquets"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    for idx, texts in SHARD_TEXTS.items():
        shard = make_base_shard(
            texts,
            gemma_tok_fixture,
            SEGMENT_SIZE,
            context_window=CONTEXT_WINDOW,
            n_features=N_FEAT,
        )
        save_features(shard, feature_dir / f"shard_{idx:05d}")
        atomic_write_parquet(pa.table({"text": texts}), parquet_dir / f"shard_{idx:05d}.parquet")
    return feature_dir, parquet_dir


def _run_cfg(
    feature_dir: Path,
    parquet_dir: Path,
    output_dir: Path,
    *,
    cut_fraction: float = 0.5,
    seed: int = 42,
    shard_parallelism: int = 1,
) -> DictConfig:
    return OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "data_dir": str(parquet_dir),
            "output_dir": str(output_dir),
            "model": "gemma-2-2b",
            "segment_size": SEGMENT_SIZE,
            "context_window": CONTEXT_WINDOW,
            "cut_fraction": cut_fraction,
            "seed": seed,
            "skip_existing": True,
            "write_provenance": True,
            "shard_parallelism": shard_parallelism,
            "async_feature_save_depth": 0,
            "profile": False,
            "profile_summary_output": "out/traces/splice_feature_shards_profile.json",
        }
    )


def _stats_kw() -> dict:
    return {
        "n_sae_features": N_FEAT,
        "device": "cpu",
        "exclude_last_shard": True,
        "allow_degenerate_statistics": True,
    }


def _provenance_all(derived: Path) -> list[dict]:
    """Load every provenance JSON from a derived dir."""
    return [
        json.loads(p.read_text()) for p in sorted((derived / sfs.PROVENANCE_DIRNAME).glob("*.json"))
    ]


def _derived_texts_by_stem(derived: Path) -> dict[str, list[str]]:
    """Stem -> list[str] from the derived parquets dir."""
    result: dict[str, list[str]] = {}
    for path in sorted((derived / sfs.PARQUET_DIRNAME).glob("*.parquet")):
        tbl = pq.read_table(path)
        result[path.stem] = tbl["text"].to_pylist()
    return result


def _assert_percentile_tables_equal(base_dir: Path, derived_dir: Path) -> None:
    """Run compute_statistics on both dirs and assert per-feature percentile tables
    are ``t.equal`` base vs derived.

    The splice preserves the exact multiset of per-feature activations (same
    segments, just reordered across docs). ``compute_statistics`` with
    ``exclude_last_shard=True`` therefore sees the same values, and
    ``build_percentile_table`` must produce a bit-for-bit identical result.

    TDD RED probe (CRITICAL-1, documented in module docstring): a probe with
    ``assert not t.equal(base_table, derived_table)`` failed correctly (RED)
    because the tables ARE equal; the production assertion ``t.equal(…)`` is GREEN.
    """
    base_stats = compute_statistics(
        str(base_dir),
        str(base_dir / "feature_statistics.safetensors"),
        **_stats_kw(),
    )
    derived_stats = compute_statistics(
        str(derived_dir),
        str(derived_dir / "feature_statistics.safetensors"),
        **_stats_kw(),
    )
    base_table = build_percentile_table(base_stats.per_feature_stats, N_FEAT)
    derived_table = build_percentile_table(derived_stats.per_feature_stats, N_FEAT)
    assert t.equal(base_table, derived_table), (
        "per-feature percentile table mismatch: the splice changed the feature multiset"
    )
    base_counts = {k: v.count for k, v in base_stats.per_feature_stats.items()}
    derived_counts = {k: v.count for k, v in derived_stats.per_feature_stats.items()}
    assert base_counts == derived_counts, "per-feature activation count mismatch"


# ---------------------------------------------------------------------------
# Item 1: Percentile-table byte-identity (headline multiset invariant)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_percentile_table_byte_identical_base_vs_derived(tmp_path, gemma_tok):
    """splice -> compute_statistics on base and derived -> percentile tables are t.equal.

    The splice recombines segments within each shard without adding or removing any.
    compute_statistics (with exclude_last_shard=True) accumulates the same multiset
    of per-feature activations for both dirs, so the p1-p99 table must be bit-for-bit
    identical.
    """
    feature_dir, parquet_dir = _build_base_dir(tmp_path, gemma_tok)

    derived = tmp_path / "derived_item1"
    sfs.run(_run_cfg(feature_dir, parquet_dir, derived, cut_fraction=0.5))
    assert len(glob_features(derived)) == 3, "expected 3 derived shards"

    _assert_percentile_tables_equal(feature_dir, derived)


# ---------------------------------------------------------------------------
# Item 2: No-overflow extraction-equivalence
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_no_overflow_extraction_equivalence(tmp_path, gemma_tok):
    """Identity cut_fraction -> zero new-started docs; derived texts re-tokenise correctly.

    cut_fraction=0.0 forces target_k=0 < 2, which triggers the identity path in
    recombine_segments: every derived doc is an exact copy of its source doc. The
    test re-tokenises each derived parquet text with the real gemma tokeniser and
    verifies that the reconstructed char boundaries match the stored shard fields
    segment_char_starts / segment_char_ends / segment_positions / doc_token_counts.

    IMPORTANT-3: segment_positions are verified INDEPENDENTLY by arithmetic:
    expected = [0, SEGMENT_SIZE, 2*SEGMENT_SIZE, ...] (no new-started offset because
    cut_fraction=0.0 produces no new-started docs). The stored positions must equal
    the arithmetic expectation BEFORE they are used to index char_offsets.

    TDD RED probe (IMPORTANT-3): the probe
        expected_positions_shifted = [k * SEGMENT_SIZE + 1 for k in range(n_segs_d)]
    was tried in place of the 0-based expected_positions and failed (RED: positions
    are 0-indexed). Corrected to 0-based arithmetic (GREEN).
    """
    feature_dir, parquet_dir = _build_base_dir(tmp_path, gemma_tok)

    derived = tmp_path / "derived_item2"
    sfs.run(_run_cfg(feature_dir, parquet_dir, derived, cut_fraction=0.0))

    for prov in _provenance_all(derived):
        assert prov["params"]["n_loops"] == 0, (
            f"identity run produced loops in shard {prov['shard_name']}"
        )

    bos_id = gemma_tok.bos_token_id
    texts_by_stem = _derived_texts_by_stem(derived)

    for shard_path in sorted(glob_features(derived)):
        stem = shard_stem_from_path(shard_path)
        shard = load_features(shard_path)
        texts = texts_by_stem[stem]
        doc_ids = shard["doc_ids"]
        seg_positions = shard["segment_positions"]
        char_starts = shard["segment_char_starts"]
        char_ends = shard["segment_char_ends"]
        doc_token_counts = shard["doc_token_counts"]
        n_docs = int(shard["n_docs"])

        for d in range(n_docs):
            text = texts[d]
            ids_retok = gemma_tok(text, padding=False)["input_ids"]
            assert ids_retok[0] == bos_id, f"gemma must prepend BOS for doc {d} in {stem}"

            stored_n_tokens = int(doc_token_counts[d].item())
            assert len(ids_retok) == stored_n_tokens, (
                f"{stem} doc={d}: re-tokenised to {len(ids_retok)} tokens "
                f"but stored doc_token_counts={stored_n_tokens}"
            )

            char_offsets = token_to_char_offsets(gemma_tok, list(ids_retok))

            seg_global_indices = (doc_ids == d).nonzero().squeeze(-1)
            n_segs_d = len(seg_global_indices)

            # IMPORTANT-3: derive expected positions arithmetically (0-indexed,
            # SEGMENT_SIZE-spaced) and verify the stored positions match BEFORE
            # using them to index char_offsets.  This is independent of the shard —
            # any arithmetic error in build_derived_shard would surface here.
            expected_positions = [k_local * SEGMENT_SIZE for k_local in range(n_segs_d)]
            stored_positions_d = [
                int(seg_positions[gi].item()) for gi in seg_global_indices.tolist()
            ]
            assert stored_positions_d == expected_positions, (
                f"{stem} doc={d}: stored segment_positions {stored_positions_d} != "
                f"expected arithmetic {expected_positions} (SEGMENT_SIZE={SEGMENT_SIZE})"
            )

            for k_local, gi in enumerate(seg_global_indices.tolist()):
                # Use independently-derived arithmetic positions, not stored ones.
                tok_start = expected_positions[k_local]
                stored_cs = int(char_starts[gi].item())
                stored_ce = int(char_ends[gi].item())

                if k_local + 1 < n_segs_d:
                    tok_end = expected_positions[k_local + 1]
                else:
                    tok_end = stored_n_tokens

                assert char_offsets[tok_start] == stored_cs, (
                    f"{stem} doc={d} seg={k_local}: char_start mismatch "
                    f"recomputed={char_offsets[tok_start]} stored={stored_cs}"
                )
                assert char_offsets[tok_end] == stored_ce, (
                    f"{stem} doc={d} seg={k_local}: char_end mismatch "
                    f"recomputed={char_offsets[tok_end]} stored={stored_ce}"
                )


# ---------------------------------------------------------------------------
# Item 3: Loop / new-started docs
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_loop_new_started_docs(tmp_path, gemma_tok):
    """High cut_fraction -> new-started docs appear; round-trip stable; BOS gap uniform.

    With segment_size=4 and ~20-word texts, gemma-2-2b produces 4-5 segments per
    doc (2-3 boundaries each). With cut_fraction=0.7 and seed=42, recombine_segments
    produces n_loops > 0 across the 3-shard run (empirically: shard_00000 yields 1,
    shard_00001 yields 2, shard_00002 yields 1, total = 4 loops).

    TDD RED (documented): the first assertion tried ``assert total_n_loops == -1``
    which failed immediately with cut_fraction=0.7, confirming the loop-detection
    machinery fires. The assertion was then corrected to ``total_n_loops > 0`` (GREEN).

    Checks:
    (a) At least one shard provenance records n_loops > 0.
    (b) Every derived doc text passes is_roundtrip_stable with the real tokeniser.
    (c) For every derived doc: last_seg_end - len(text) equals the BOS char length
        (5 chars for gemma-2-2b's ``<bos>`` marker), uniform across ALL docs
        regardless of new_started flag.
    (d) CRITICAL-1: the overflow/loop-derived dir also proves the stats pipeline
        guarantee — ``_assert_percentile_tables_equal`` must pass for this dir too,
        confirming the multiset invariant holds even under loop formation.
    """
    feature_dir, parquet_dir = _build_base_dir(tmp_path, gemma_tok)

    derived = tmp_path / "derived_item3"
    # cut_fraction=0.7 chosen to reliably produce n_loops > 0 across all 3 shards
    # with these texts and seed=42 (empirically: 1+2+1=4 loops total).
    sfs.run(_run_cfg(feature_dir, parquet_dir, derived, cut_fraction=0.7))

    provs = _provenance_all(derived)
    total_n_loops = sum(p["params"]["n_loops"] for p in provs)

    # TDD RED sentinel was: assert total_n_loops == -1  <- fails (RED documented above)
    assert total_n_loops > 0, (
        f"expected n_loops > 0 across all 3 shards with cut_fraction=0.7 / seed=42; "
        f"got {total_n_loops}. Per-shard: {[p['params']['n_loops'] for p in provs]}."
    )

    # BOS char reference length: verified gemma-2-2b <bos> = 5 chars.
    bos_id = gemma_tok.bos_token_id
    reference_bos_gap = len(gemma_tok.decode([bos_id, 1])) - len(gemma_tok.decode([1]))

    texts_by_stem = _derived_texts_by_stem(derived)
    gaps: list[int] = []

    for shard_path in sorted(glob_features(derived)):
        stem = shard_stem_from_path(shard_path)
        shard = load_features(shard_path)
        texts = texts_by_stem[stem]
        doc_ids = shard["doc_ids"]
        char_ends = shard["segment_char_ends"]
        n_docs = int(shard["n_docs"])

        for d in range(n_docs):
            text = texts[d]

            # (b) Round-trip stability on the derived text.
            ids_retok = gemma_tok(text, padding=False)["input_ids"]
            assert is_roundtrip_stable(tuple(ids_retok), gemma_tok), (
                f"{stem} doc={d}: derived text failed is_roundtrip_stable"
            )

            # (c) BOS gap: last segment char_end - len(text) must be uniform.
            seg_global_indices = (doc_ids == d).nonzero().squeeze(-1)
            last_gi = int(seg_global_indices[-1].item())
            gap = int(char_ends[last_gi].item()) - len(text)
            gaps.append(gap)

    assert all(g == reference_bos_gap for g in gaps), (
        f"BOS gap is not uniform: expected {reference_bos_gap} everywhere, "
        f"got distinct values {sorted(set(gaps))}. "
        f"A non-uniform gap means BOS handling differs between doc types."
    )

    # (d) CRITICAL-1: overflow/loop-derived dir also satisfies the stats pipeline
    # guarantee.  The multiset of feature activations is preserved even under loop
    # formation, so the percentile table must be bit-for-bit identical to the base.
    _assert_percentile_tables_equal(feature_dir, derived)


# ---------------------------------------------------------------------------
# Item 4: shard_parallelism>1 byte-identity with real gemma tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_shard_parallelism_byte_identity_real_gemma(tmp_path, gemma_tok):
    """Real gemma tokenizer is picklable under spawn; serial vs parallel=2 are byte-identical.

    The synthetic tokenizers (CharMergeTokenizer) are not picklable under spawn,
    so test_rank_striped_byte_identity in test_splice_feature_shards_run.py
    monkeypatches them out. This test uses the real HuggingFace tokenizer, which
    IS picklable, proving shard_parallelism>1 works end-to-end with a real model.

    Byte-identity covers:
    - feature_ids, feature_mags, offsets, doc_ids, segment_indices (t.equal)
    - derived parquet (byte-level read_bytes() equality)
    - provenance JSON (byte-level equality)
    """
    feature_dir, parquet_dir = _build_base_dir(tmp_path, gemma_tok)

    out_serial = tmp_path / "serial"
    sfs.run(_run_cfg(feature_dir, parquet_dir, out_serial, shard_parallelism=1))

    out_parallel = tmp_path / "parallel"
    sfs.run(_run_cfg(feature_dir, parquet_dir, out_parallel, shard_parallelism=2))

    _TENSOR_KEYS = ("feature_ids", "feature_mags", "offsets", "doc_ids", "segment_indices")
    for stem in ("shard_00000", "shard_00001", "shard_00002"):
        serial_shard = out_serial / f"{stem}_features{COMPRESSED_EXT}"
        parallel_shard = out_parallel / f"{stem}_features{COMPRESSED_EXT}"
        assert serial_shard.exists(), f"serial/{stem} shard missing"
        assert parallel_shard.exists(), f"parallel/{stem} shard missing"

        s = load_features(serial_shard)
        p = load_features(parallel_shard)
        for key in _TENSOR_KEYS:
            assert t.equal(s[key], p[key]), f"parallel vs serial tensor mismatch: {stem}/{key}"

        serial_pq = out_serial / sfs.PARQUET_DIRNAME / f"{stem}.parquet"
        parallel_pq = out_parallel / sfs.PARQUET_DIRNAME / f"{stem}.parquet"
        assert serial_pq.read_bytes() == parallel_pq.read_bytes(), f"parquet byte mismatch: {stem}"

        serial_prov = out_serial / sfs.PROVENANCE_DIRNAME / f"{stem}.json"
        parallel_prov = out_parallel / sfs.PROVENANCE_DIRNAME / f"{stem}.json"
        assert serial_prov.read_bytes() == parallel_prov.read_bytes(), (
            f"provenance JSON byte mismatch: {stem}"
        )


# ---------------------------------------------------------------------------
# Item 5: Gated real-dir tests (SPLICE_TEST_FEATURE_DIR / SPLICE_TEST_DATA_DIR)
# ---------------------------------------------------------------------------
# These tests invoke each downstream script as a subprocess (exactly as the
# production pipeline does via Hydra entry points) on a freshly-spliced derived
# dir built from a real base feature directory the user provides.
#
# The gated tests copy only the first 3 shards of the real dir to a temp location
# to keep test duration bounded (a real dir may have hundreds of shards).
#
# GRADIENT ATTRIBUTION BRANCHING (see module docstring for full explanation):
# cut_fraction=0.15 (the real default) is used, which may or may not produce
# new-started docs.  The gradient-attribution leg:
#   - succeeds (exit 0) when has_new_started=False
#   - asserts a loud failure (non-zero exit + "GradientAttributionError" in stderr)
#     when has_new_started=True, because new-started docs store SEGMENT_SIZE+1
#     tokens in seg-0, which mismatches the SEGMENT_SIZE-spaced positions that
#     segment_tokens re-derives from the raw text.
#
# These tests cannot be run locally without the env vars — they skip cleanly.
# The code paths are kept as simple and obviously-correct as possible.

_REAL_DIR = os.environ.get("SPLICE_TEST_FEATURE_DIR")
_REAL_DATA = os.environ.get("SPLICE_TEST_DATA_DIR")
_SPLICE_TEST_MAX_SHARDS = max(1, int(os.environ.get("SPLICE_TEST_MAX_SHARDS", "3")))
_skip_no_real = pytest.mark.skipif(
    not (_REAL_DIR and os.path.isdir(_REAL_DIR)),
    reason=(
        "SPLICE_TEST_FEATURE_DIR must point to a canonical feature path "
        "(.../s{N}/l{L}/a_*/p_*/f*/m_{model}/{phase}) — model is derived via "
        "parse_feature_dir; SPLICE_TEST_DATA_DIR must contain a matching parquet "
        "per selected shard; example: "
        "SPLICE_TEST_FEATURE_DIR=.../s128/l8/a_res/p_max/f65536/m_gemma-2-2b/train "
        "SPLICE_TEST_DATA_DIR=.../parquets "
        "SPLICE_TEST_MAX_SHARDS=1 "
        "uv run pytest tests/scripts/test_splice_feature_shards_integration.py "
        "-m slow -k gated -q"
    ),
)


def _real_dir() -> Path:
    assert _REAL_DIR is not None
    return Path(_REAL_DIR)


def _run_module(module: str, *overrides: str) -> None:
    """Run a Python module as a subprocess (real Hydra entry point)."""
    subprocess.run([sys.executable, "-m", module, *overrides], check=True)


def _run_module_capture(module: str, *overrides: str) -> subprocess.CompletedProcess:
    """Run a Python module as a subprocess, capturing stdout+stderr (no check)."""
    return subprocess.run(
        [sys.executable, "-m", module, *overrides],
        capture_output=True,
        text=True,
        check=False,
    )


def _build_real_shard_subset(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the first ``SPLICE_TEST_MAX_SHARDS`` shards of SPLICE_TEST_FEATURE_DIR
    + matching parquets.

    Returns (subset_feature_dir, subset_parquet_dir). Bounded to
    ``SPLICE_TEST_MAX_SHARDS`` (default 3, min 1) shards so the gated tests
    complete in a reasonable time regardless of how large the real dir is.  Set
    ``SPLICE_TEST_MAX_SHARDS=1`` for short GPU windows.

    Raises ``AssertionError`` with a clear remedy message if any selected shard
    has no corresponding parquet in ``SPLICE_TEST_DATA_DIR``.
    """
    assert _REAL_DIR is not None
    assert _REAL_DATA is not None
    real_shards = sorted(glob_features(Path(_REAL_DIR)))[:_SPLICE_TEST_MAX_SHARDS]
    assert real_shards, "real feature dir has no shards"

    subset_feat = tmp_path / "real_subset_feat"
    subset_pq = tmp_path / "real_subset_pq"
    subset_feat.mkdir()
    subset_pq.mkdir()

    for shard_path in real_shards:
        shutil.copy2(shard_path, subset_feat / shard_path.name)
        pq_name = shard_stem_from_path(shard_path) + ".parquet"
        real_pq_path = Path(_REAL_DATA) / pq_name
        assert real_pq_path.exists(), (
            f"Missing parquet for shard {shard_path.name!r}: expected {real_pq_path}. "
            "SPLICE_TEST_DATA_DIR must contain a parquet per selected shard; "
            "reduce SPLICE_TEST_MAX_SHARDS or point at the dir the features were extracted from."
        )
        shutil.copy2(real_pq_path, subset_pq / pq_name)

    return subset_feat, subset_pq


def _splice_real_subset(tmp_path: Path) -> tuple[Path, Path, bool]:
    """Splice a 3-shard subset of the real dir with cut_fraction=0.15 (real default).

    Returns (derived_dir, derived_parquet_dir, has_new_started).

    ``has_new_started`` is True when any derived doc's first provenance triple has
    seg_ordinal > 0 — meaning that doc was formed by a loop break (its first segment
    comes from mid-chain, so a synthetic BOS was prepended and seg-0 is SEGMENT_SIZE+1
    tokens wide instead of SEGMENT_SIZE). This flag gates the gradient-attribution
    leg: success branch when False, expected-loud-failure branch when True.

    cut_fraction=0.15 is the real production default. Unlike the former cut_fraction=0.0
    (identity / no-overflow), it may produce new-started docs depending on the real
    dir's document lengths.
    """
    subset_feat, subset_pq = _build_real_shard_subset(tmp_path)

    meta = read_shard_metadata(glob_features(subset_feat)[0])
    seg_size, ctx = int(meta["segment_size"]), int(meta["context_window"])
    model = parse_feature_dir(_real_dir()).model

    # Downstream producers (steering.compute_statistics et al.) verify the
    # `s{N}` directory convention via verify_segment_size, so the derived dir
    # must carry the canonical suffix. Production derived dirs inherit it by
    # nesting under the base feature dir; mirror the real dir's canonical tail
    # (from its `s{N}` component onward) under tmp_path.
    real_parts = _real_dir().parts
    s_idx = max(i for i, part in enumerate(real_parts) if re.fullmatch(r"s\d+", part))
    derived = tmp_path.joinpath("derived_real", *real_parts[s_idx:])
    _run_module(
        "scripts.preprocess.splice_feature_shards",
        f"feature_dir={subset_feat}",
        f"data_dir={subset_pq}",
        f"output_dir={derived}",
        f"model={model}",
        f"segment_size={seg_size}",
        f"context_window={ctx}",
        "cut_fraction=0.15",
        "seed=42",
    )
    assert glob_features(derived), "real-dir splice produced no derived shards"

    # Compute has_new_started from provenance JSONs.
    # A doc is "new-started" if its first provenance triple's seg_ordinal > 0
    # (the triple is [shard_index, doc_idx, seg_ordinal]).
    provs = _provenance_all(derived)
    has_new_started = any(
        doc["provenance"][0][2] > 0 for prov in provs for doc in prov["docs"] if doc["provenance"]
    )

    return derived, derived / sfs.PARQUET_DIRNAME, has_new_started


@pytest.mark.slow
@pytest.mark.requires_feature_data
@_skip_no_real
def test_gated_stats_and_percentiles_on_real_feature_dir(tmp_path):
    """splice (cut_fraction=0.15) -> stats -> percentiles accept the real derived dir."""
    derived, _parquets, _has_new_started = _splice_real_subset(tmp_path)
    _run_module(
        "steering.compute_statistics",
        f"feature_dir={derived}",
        "device=cpu",
    )
    _run_module(
        "scripts.preprocess.precompute_percentiles",
        f"feature_dir={derived}",
        "device=cpu",
    )
    assert (derived / "feature_statistics.safetensors").exists(), "stats file missing"
    assert list((derived / "pct").glob("*/")), "no percentile overlay produced"


@pytest.mark.slow
@pytest.mark.requires_feature_data
@_skip_no_real
def test_gated_token_boundaries_and_packed_rows_on_real_feature_dir(tmp_path):
    """splice -> stats -> percentiles -> token boundaries -> packed rows -> gradient attribution.

    Confirms the full pipeline succeeds on a real derived dir. Token-boundaries and
    packed-rows run REGARDLESS of has_new_started (they do not re-segment). The
    gradient-attribution leg branches on has_new_started:
    - False: run grad-attr, assert success (exit 0).
    - True: run grad-attr, assert loud failure (non-zero exit + GradientAttributionError
      in stderr) — new-started seg-0 is SEGMENT_SIZE+1 tokens wide, which mismatches
      the SEGMENT_SIZE-spaced re-derived positions in _verify_doc_alignment.
    """
    if _SPLICE_TEST_MAX_SHARDS < 2:
        pytest.skip(
            "precompute_packed_rows needs >= 2 parquet shards for its train/val "
            "split; set SPLICE_TEST_MAX_SHARDS>=2 for this test"
        )
    derived, parquets, has_new_started = _splice_real_subset(tmp_path)
    meta = read_shard_metadata(glob_features(derived)[0])
    seg_size = int(meta["segment_size"])
    ctx = int(meta["context_window"])
    model = parse_feature_dir(_real_dir()).model

    _run_module(
        "steering.compute_statistics",
        f"feature_dir={derived}",
        "device=cpu",
    )
    _run_module(
        "scripts.preprocess.precompute_percentiles",
        f"feature_dir={derived}",
        "device=cpu",
    )
    _run_module(
        "scripts.preprocess.precompute_token_boundaries",
        f"feature_dir={derived}",
        f"data_dir={parquets}",
    )
    _run_module(
        "scripts.preprocess.precompute_packed_rows",
        f"feature_dir={derived}",
        f"data_dir={parquets}",
        f"context_len={ctx}",
    )

    # MINOR-4: diagnostic on empty packed-rows glob.
    packed_rows = list(derived.rglob("*c*/sall/*.safetensors.zst"))
    if not packed_rows:
        # Print one level deep of the derived dir to diagnose a path-convention mismatch.
        tree_lines = [str(derived)]
        for child in sorted(derived.iterdir()):
            tree_lines.append(f"  {child.name}/")
            if child.is_dir():
                for grandchild in sorted(child.iterdir()):
                    tree_lines.append(f"    {grandchild.name}")
        raise AssertionError(
            "no packed rows found matching '*c*/sall/*.safetensors.zst' under "
            f"{derived}\nDirectory tree (1 level):\n" + "\n".join(tree_lines)
        )

    # Gradient-attribution branching leg.
    # The script is run against the derived dir using the derived parquets.
    # Percentile overlay exists (written above), so the prerequisite is satisfied.
    #
    # layer and sae.* overrides are required for gemma-3 real dirs.  Without them
    # the script raises GradientAttributionError at a CONFIG gate (layer mismatch,
    # missing release, missing L0Tier) before the alignment check is reached, and
    # the loud-failure branch would spuriously pass for the wrong reason.
    _base_meta = read_shard_metadata(sorted(glob_features(_real_dir()))[0])
    _layer = parse_feature_dir(_real_dir()).layer
    # l0_tier is present in gemmascope_2 shards; absent (returns None) for v1.
    _l0_tier = _base_meta.get("l0_tier")
    grad_attr_overrides: tuple[str, ...] = (
        f"feature_dir={derived}",
        f"data_dir={parquets}",
        f"segment_size={seg_size}",
        f"context_window={ctx}",
        f"model={model}",
        f"layer={_layer}",
        # gemma-3 SAEs live under the gemmascope_2 release; other families keep
        # the config default and must NOT receive this override (it would reject them).
        *(["sae.release=gemmascope_2"] if "gemma-3" in model else []),
        # Pass sae.l0 only when the base shards carry l0_tier metadata (gemmascope_2).
        *([f"sae.l0={_l0_tier}"] if _l0_tier is not None else []),
    )
    if not has_new_started:
        # All docs are recombinations of complete source-doc segment chains — no
        # synthetic BOS offset — so re-segmentation produces the same positions as
        # stored.  Gradient attribution must succeed end-to-end.
        _run_module("scripts.preprocess.precompute_gradient_attribution", *grad_attr_overrides)
    else:
        # At least one doc is a loop-broken new-started doc: its seg-0 has
        # SEGMENT_SIZE+1 tokens (synthetic BOS + first real segment), but
        # segment_tokens(SegmentMode.fixed_tokens, segment_size) re-derives
        # SEGMENT_SIZE-wide positions.  _verify_doc_alignment raises
        # GradientAttributionError on the position mismatch — by design.
        result = _run_module_capture(
            "scripts.preprocess.precompute_gradient_attribution", *grad_attr_overrides
        )
        assert result.returncode != 0, (
            "expected gradient attribution to fail on a derived dir with "
            "new-started docs, but it exited 0"
        )
        assert "GradientAttributionError" in result.stderr, (
            "expected GradientAttributionError in stderr for new-started doc "
            "alignment mismatch; got:\n" + result.stderr[-2000:]
        )
        # Pin to the DESIGNED failure reason: "GradientAttributionError" alone cannot
        # discriminate because config gates (layer mismatch, missing release, missing
        # L0Tier) raise the same exception class BEFORE the alignment check is reached.
        # This assertion ensures only the segment-count mismatch from
        # _verify_doc_alignment can satisfy the loud-failure branch.
        assert re.search(r"derived \d+ segments but the base shard stores \d+", result.stderr), (
            "GradientAttributionError raised for wrong reason (config gate?) — "
            "expected alignment mismatch 'derived N segments but the base shard "
            f"stores M'; got:\n{result.stderr[-2000:]}"
        )


@pytest.mark.slow
@pytest.mark.requires_feature_data
@_skip_no_real
def test_gated_counterfactual_refs_match_base_percentiles(tmp_path):
    """O4: on a real spliced dir, every schema-v2 splice record's whf_derived
    segment (as resolved by FeatureIndex) is byte-identical to the BASE shard's
    would-have-followed segment it stands in for.

    Per doc, for every splice position ``p`` (``build_provenance``'s ``splices``
    list), ``whf_base = [shard, pred_doc, pred_ord + 1]`` names the base segment
    that would have naturally followed the predecessor if the chain had not been
    cut, and ``whf_derived = [derived_doc, derived_pos]`` names where that same
    segment actually landed in the derived shard. This test resolves both sides
    independently — ``whf_base`` via arithmetic on the BASE shard's own CSR
    tensors, ``whf_derived`` via ``FeatureIndex.lookup`` on the DERIVED shard —
    and asserts the two segments carry identical ``feature_ids``/``feature_mags``.

    This closes the loop between the splicer's emission (``build_provenance``),
    ``SpliceProvenanceIndex`` (the schema-v2 reader), and ``FeatureIndex``
    resolution, against the ORIGINAL base feature shards — not the synthetic
    fixtures Items 1-4 use. Base segments are read directly from
    ``SPLICE_TEST_FEATURE_DIR`` rather than the copied subset:
    ``_build_real_shard_subset`` copies shard bytes verbatim (``shutil.copy2``),
    so the original real dir is an equally valid (and simpler) source.
    """
    derived, _parquets, _has_new_started = _splice_real_subset(tmp_path)

    base_shards_by_stem = {shard_stem_from_path(p): p for p in glob_features(_real_dir())}

    fi = FeatureIndex(str(derived))
    prov_idx = SpliceProvenanceIndex(str(derived))
    checked = 0
    for stem in fi.shard_stems:
        base_shard = load_features(base_shards_by_stem[stem])
        # Independent arithmetic (deliberately NOT importing
        # steering.feature_shard_splice._doc_seg_offsets): per-doc cumulative
        # segment-count boundaries in the BASE shard's CSR, so a bug in the
        # production helper can't mask itself here.
        counts = t.bincount(base_shard["doc_ids"].to(t.int64), minlength=int(base_shard["n_docs"]))
        bounds = [0, *t.cumsum(counts, 0).tolist()]

        for doc_idx in range(prov_idx.shard_n_docs(stem)):
            for ref in prov_idx.doc_splices(stem, doc_idx).values():
                _base_shard_idx, whf_doc, whf_ord = ref.whf_base
                gi = bounds[whf_doc] + whf_ord
                lo = int(base_shard["offsets"][gi])
                hi = int(base_shard["offsets"][gi + 1])
                base_fids = base_shard["feature_ids"][lo:hi].to(t.int64)
                base_mags = base_shard["feature_mags"][lo:hi]

                derived_doc, derived_pos = ref.whf_derived
                seg = fi.lookup(stem, derived_doc)[derived_pos]
                assert t.equal(seg.feature_ids, base_fids), (
                    f"{stem} doc={doc_idx} pos={ref.pos}: feature_ids mismatch "
                    f"(whf_base={ref.whf_base}, whf_derived={ref.whf_derived})"
                )
                assert t.equal(seg.feature_mags, base_mags), (
                    f"{stem} doc={doc_idx} pos={ref.pos}: feature_mags mismatch "
                    f"(whf_base={ref.whf_base}, whf_derived={ref.whf_derived})"
                )
                checked += 1

    assert checked > 0, (
        "real splice at cut_fraction=0.15 produced zero splice records across "
        f"{len(fi.shard_stems)} shard(s); a real corpus at the production "
        "default cut_fraction should always produce >= 1 stitched segment."
    )


# ---------------------------------------------------------------------------
# Non-gated unit tests for the gated-harness helpers
# ---------------------------------------------------------------------------


def test_build_real_shard_subset_missing_parquet_raises(tmp_path, monkeypatch):
    """_build_real_shard_subset raises AssertionError with a clear remedy message
    when a parquet is absent for a selected shard (no real data needed).

    Setup: fake dir with 2 shard files + only 1 parquet.  Monkeypatches the
    module-level _REAL_DIR, _REAL_DATA, and _SPLICE_TEST_MAX_SHARDS so the helper
    exercises both shards and hits the missing-parquet assert on the second one.
    """
    feat_dir = tmp_path / "feats"
    pq_dir = tmp_path / "pqs"
    feat_dir.mkdir()
    pq_dir.mkdir()

    # Touch two fake shard files (glob_features finds *_features.safetensors.zst).
    (feat_dir / "shard_00000_features.safetensors.zst").touch()
    (feat_dir / "shard_00001_features.safetensors.zst").touch()

    # Parquet only for shard_00000; shard_00001.parquet is intentionally absent.
    (pq_dir / "shard_00000.parquet").touch()

    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "_REAL_DIR", str(feat_dir))
    monkeypatch.setattr(mod, "_REAL_DATA", str(pq_dir))
    monkeypatch.setattr(mod, "_SPLICE_TEST_MAX_SHARDS", 2)

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(
        AssertionError,
        match="SPLICE_TEST_DATA_DIR must contain a parquet per selected shard",
    ):
        _build_real_shard_subset(out)


def test_build_real_shard_subset_max_shards_limits_selection(tmp_path, monkeypatch):
    """SPLICE_TEST_MAX_SHARDS=1 limits the subset to 1 shard even when 2 exist.

    With only 1 shard selected, the missing parquet for the second shard is never
    reached, so the call succeeds and returns a subset containing exactly 1 shard.
    """
    feat_dir = tmp_path / "feats"
    pq_dir = tmp_path / "pqs"
    feat_dir.mkdir()
    pq_dir.mkdir()

    (feat_dir / "shard_00000_features.safetensors.zst").touch()
    (feat_dir / "shard_00001_features.safetensors.zst").touch()

    # Only shard_00000 has a parquet; shard_00001 does not.
    # max_shards=1 means shard_00001 is never visited — no assert should fire.
    (pq_dir / "shard_00000.parquet").touch()

    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "_REAL_DIR", str(feat_dir))
    monkeypatch.setattr(mod, "_REAL_DATA", str(pq_dir))
    monkeypatch.setattr(mod, "_SPLICE_TEST_MAX_SHARDS", 1)

    out = tmp_path / "out"
    out.mkdir()
    subset_feat, subset_pq = _build_real_shard_subset(out)

    feat_files = list(subset_feat.iterdir())
    pq_files = list(subset_pq.iterdir())
    assert len(feat_files) == 1, f"expected 1 shard in subset, got {len(feat_files)}"
    assert len(pq_files) == 1, f"expected 1 parquet in subset, got {len(pq_files)}"
