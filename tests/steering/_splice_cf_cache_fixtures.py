"""Shared fixture: a REAL multi-source interleaved splice-counterfactual
preselection corpus, small enough for the default test suite.

Builds a derived feature dir (real splicer), all-ones percentile + real token
overlays, a saved statistics file, and a real packed-row corpus with a small
packing buffer so packed provenance genuinely interleaves source shards — the
pathological access pattern issue #740 optimizes. ``run_selection_main`` drives
the production ``precompute_selected_features.main.__wrapped__`` with a frozen
``created_at`` and returns every ``sel/`` file's bytes for byte-identity checks.
"""

from __future__ import annotations

import itertools
import json
import string
from pathlib import Path

import torch as t
import yaml
from omegaconf import OmegaConf

from scripts.preprocess import precompute_selected_features as psf
from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from steering import overlay_io
from steering.conditional.feature_index import FeatureIndex
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    load_overlay_shard,
    write_manifest,
)
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)
from tests.steering._splice_cf_fixtures import build_derived_env
from tests.steering.test_splice_counterfactual_preselected import (
    _write_all_ones_percentile_overlay,
    _write_token_boundary_overlay,
)

SELECTION_N_SAE = 4096
SELECTION_SEG_SIZE = 2

# Seven source shards, two short (5-char "XY ZW") docs each; every doc string is
# globally distinct so splicing produces real would-have-followed references
# across shards. Doc length matters here, not content: CharMergeTokenizer (see
# tests/steering/_splice_test_tokenizers.py) is configured with NO merge pairs,
# so it tokenizes 1 char == 1 token — every doc below is exactly 5 tokens, short
# enough that several docs fit in one packed row (context_len=16 below), which
# is what makes the packer's small buffer_size actually interleave source
# shards instead of draining one shard before touching the next. An earlier,
# longer-text 5-shard draft (one doc ~= one whole packed row) empirically
# plateaued at only 2 distinct stems / 2 alternations per packed shard,
# regardless of buffer_size or shard count, because a single doc already fills
# almost an entire row, leaving no room for a 3rd shard's doc to interleave
# in — see tests/steering/test_splice_preselection_parity.py's guard test.
_TEXTS_BY_SHARD = [
    ["aa bb", "cc dd"],
    ["ee ff", "gg hh"],
    ["ii jj", "kk ll"],
    ["mm nn", "oo pp"],
    ["qq rr", "ss tt"],
    ["uu vv", "ww xx"],
    ["yy zz", "ab cd"],
]


def _save_statistics(
    feature_dir: Path, n_shards: int, *, texts_by_shard: list[list[str]] | None = None,
) -> None:
    """Write a feature_statistics file covering SELECTION_N_SAE features.

    ``n_shards`` selects the corpus-stats prefix of ``texts_by_shard`` (default
    ``_TEXTS_BY_SHARD``, build_interleaved_env's corpus) that was ACTUALLY
    built by the caller, so ``n_shards_processed``/``n_docs`` never silently
    disagree with a caller that builds fewer than all shards -- or a
    different corpus entirely (e.g. build_multisplice_env's
    ``_MULTISPLICE_TEXTS_BY_SHARD``).
    """
    percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    per_feature_stats = {
        str(i): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(percentiles),
            histogram=None,
        )
        for i in range(SELECTION_N_SAE)
    }
    built_shards = (texts_by_shard if texts_by_shard is not None else _TEXTS_BY_SHARD)[:n_shards]
    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=SELECTION_SEG_SIZE,
            context_window=2048,
            layer=8,
            n_sae_features=SELECTION_N_SAE,
            generated_at="2026-01-01T00:00:00",
            activation_percentile=0.95,
        ),
        corpus=CorpusStats(
            n_shards_processed=len(built_shards),
            n_docs=sum(len(s) for s in built_shards),
            n_tokens_gpt2=1000,
            mean_tokens_per_doc=10.0,
            median_tokens_per_doc=10,
        ),
        features=SegmentFeatureStats(
            n_segments=100,
            n_features_seen=SELECTION_N_SAE,
            mean_active_per_segment=2.0,
            median_active_per_segment=2,
            p90_active_per_segment=4,
            max_active_per_segment=8,
            min_active_per_segment=1,
            n_total_sae_features=SELECTION_N_SAE,
            pct_features_active=0.5,
            count_percentiles={"p50": 2},
            magnitude_stats={"mean": 1.0},
            active_per_segment_histogram={"bins": [0, 1], "counts": [1]},
            active_per_segment_percentiles={"p50": 2.0},
        ),
        # total_bytes/mean_bytes_per_doc/mean_bytes_per_token are all required
        # (no dataclass defaults) — StorageStats(total_bytes=1000) alone raises.
        storage=StorageStats(
            total_bytes=1000, mean_bytes_per_doc=100.0, mean_bytes_per_token=10.0
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
    )
    save_statistics(stats, feature_dir / "feature_statistics")


def _finish_derived_env(
    tmp_path,
    env: dict,
    *,
    n_shards: int,
    context_len: int,
    manifest_name: str,
    texts_by_shard: list[list[str]] | None = None,
    buffer_size: int = 3,
    rows_per_shard: int = 2,
) -> dict:
    """Shared tail for build_interleaved_env/build_multisplice_env: statistics,
    percentile + token overlays, a real FeatureIndex, real packed rows, and a
    packed-dir manifest — everything ``run_selection_main`` expects, given an
    ``env`` dict already produced by ``build_derived_env``.
    """
    feature_dir = Path(env["feature_dir"])
    _save_statistics(feature_dir, n_shards, texts_by_shard=texts_by_shard)
    pct_checksum = _write_all_ones_percentile_overlay(feature_dir)
    tok_checksum = _write_token_boundary_overlay(
        feature_dir=feature_dir,
        data_dir=env["data_dir"],
        tokenizer=env["tokenizer"],
        segment_size=int(env["seg_size"]),
    )
    feature_index = FeatureIndex(
        str(feature_dir),
        percentile_checksum=pct_checksum,
        tokenizer_checksum=tok_checksum,
    )
    packed_dir = Path(tmp_path) / "packed"
    total_rows, max_n_seg, max_seg_size = precompute_packed_rows(
        parquet_paths=[
            str(path) for path in sorted(Path(env["data_dir"]).glob("*.parquet"))
        ],
        tokenizer=env["tokenizer"],
        feature_index=feature_index,
        context_len=context_len,
        output_dir=packed_dir,
        buffer_size=buffer_size,
        rows_per_shard=rows_per_shard,
        compress_workers=1,
    )
    assert total_rows > 0
    write_manifest(
        packed_dir,
        manifest_name,
        OverlayKind.PACKED_ROW,
        metadata={
            "base_shard_stems": feature_index.shard_stems,
            "provenance_shards": len(glob_overlay_shards(provenance_dir(packed_dir))),
            "max_n_seg": max_n_seg,
            "max_seg_size": max_seg_size,
            "context_len": context_len,
            "pct_checksum": pct_checksum,
            "tok_checksum": tok_checksum,
        },
    )
    return {
        "feature_dir": str(feature_dir),
        "packed_row_dir": packed_dir,
        "data_dir": env["data_dir"],
        "tokenizer": env["tokenizer"],
        "seg_size": int(env["seg_size"]),
        "n_sae": SELECTION_N_SAE,
        "pct_checksum": pct_checksum,
        "tok_checksum": tok_checksum,
    }


def build_interleaved_env(tmp_path, *, n_shards: int = 7, cut_fraction: float = 0.6) -> dict:
    env = build_derived_env(
        tmp_path, cut_fraction=cut_fraction, texts_by_shard=_TEXTS_BY_SHARD[:n_shards]
    )
    # buffer_size=3, rows_per_shard=2, context_len=16 (_finish_derived_env's
    # defaults) with the 7-shard, 5-token-doc corpus above achieves
    # (max_distinct_stems, max_alternations) = (4, 4) in the worst-case packed
    # shard — verified via worst_case_stem_alternations() and asserted by
    # test_splice_preselection_parity.py::test_fixture_exhibits_interleaved_access.
    return _finish_derived_env(
        tmp_path, env, n_shards=n_shards, context_len=16, manifest_name="packed-interleaved",
    )


# --- Multi-splice-per-doc corpus (issue #740 hardening item 1) -------------
#
# build_interleaved_env's docs are exactly 5 tokens (2-word "aa bb") — short
# enough that recombine_segments' target_k = round(cut_fraction *
# n_splice_points) never selects more than one splice point per derived doc's
# root chain (verified empirically: every derived doc there has 0 or 1
# splice positions, never >= 2 — see steering/feature_shard_splice.py's
# recombine_segments/build_provenance for why: a chain only accumulates a
# SECOND splice position if the segment it lands on after one stitch is
# ITSELF another cut's source, which needs more segments per doc than a
# 5-token doc has). Longer, multi-segment docs give the real splicer room to
# chain more than one stitch into a single derived doc before it ends. 6-word
# (17-token) docs at the SAME cut_fraction=0.6 default reliably clear this
# bar: empirically, every one of 7 shards independently produces at least one
# derived doc with >= 3 splice positions (see
# test_fixture_exhibits_multisplice_docs).
_MULTISPLICE_WORDS_PER_DOC = 6
_MULTISPLICE_DOCS_PER_SHARD = 2
_MULTISPLICE_N_SHARDS = 7


def _multisplice_texts_by_shard(
    n_shards: int, docs_per_shard: int, words_per_doc: int,
) -> list[list[str]]:
    """``n_shards`` groups of ``docs_per_shard`` docs, each a space-joined
    string of ``words_per_doc`` GLOBALLY DISTINCT 2-char words (matching
    _TEXTS_BY_SHARD's global-distinctness property above, so splicing
    produces real would-have-followed references)."""
    total_words = n_shards * docs_per_shard * words_per_doc
    words = [
        "".join(pair)
        for pair in itertools.islice(
            itertools.product(string.ascii_lowercase, repeat=2), total_words
        )
    ]
    texts_by_shard: list[list[str]] = []
    idx = 0
    for _ in range(n_shards):
        shard_docs: list[str] = []
        for _ in range(docs_per_shard):
            shard_docs.append(" ".join(words[idx : idx + words_per_doc]))
            idx += words_per_doc
        texts_by_shard.append(shard_docs)
    return texts_by_shard


_MULTISPLICE_TEXTS_BY_SHARD = _multisplice_texts_by_shard(
    _MULTISPLICE_N_SHARDS, _MULTISPLICE_DOCS_PER_SHARD, _MULTISPLICE_WORDS_PER_DOC
)


def build_multisplice_env(tmp_path, *, n_shards: int = 7, cut_fraction: float = 0.6) -> dict:
    """Like build_interleaved_env, but with longer per-doc source text so the
    real splicer genuinely produces >= 2 splice positions within a SINGLE
    derived doc (issue #740 hardening item 1) — a dimension
    build_interleaved_env's short docs never reach. See
    test_fixture_exhibits_multisplice_docs for the self-checking guard.
    """
    env = build_derived_env(
        tmp_path,
        cut_fraction=cut_fraction,
        texts_by_shard=_MULTISPLICE_TEXTS_BY_SHARD[:n_shards],
    )
    # context_len=64 (vs build_interleaved_env's 16): a derived doc here can
    # run up to ~2x a 17-token source doc plus a BOS when a chain stitches
    # most of both of a shard's docs together; 64 comfortably avoids
    # truncate_to_boundary_coverage trimming a doc that would otherwise carry
    # one of the >= 2 splice positions this fixture exists to exercise.
    return _finish_derived_env(
        tmp_path,
        env,
        n_shards=n_shards,
        context_len=64,
        manifest_name="packed-multisplice",
        texts_by_shard=_MULTISPLICE_TEXTS_BY_SHARD,
    )


def splice_doc_stats(feature_dir: str | Path) -> tuple[int, int, int]:
    """(total_derived_docs, zero_splice_docs, max_splices_in_one_doc) across
    every ``_splice_provenance`` sidecar under ``feature_dir``.

    The shared witness for two issue #740 hardening dimensions: the no-op /
    zero-splice case (build_interleaved_env; a doc with ``splices == []``,
    see test_fixture_exhibits_zero_splice_docs) and the multi-splice-per-doc
    case (build_multisplice_env; a doc with ``len(splices) >= 2``, see
    test_fixture_exhibits_multisplice_docs).
    """
    prov_dir = Path(feature_dir) / PROVENANCE_DIRNAME
    total = 0
    zero = 0
    max_splices = 0
    for p in sorted(prov_dir.glob("*.json")):
        raw = json.loads(p.read_text())
        for doc in raw["docs"]:
            n = len(doc["splices"])
            total += 1
            zero += n == 0
            max_splices = max(max_splices, n)
    return total, zero, max_splices


def worst_case_stem_alternations(packed_row_dir: Path) -> tuple[int, int]:
    """Max distinct source stems and max stem alternations across packed shards.

    Reconstructs the exact reference group order (sorted by first packed-segment
    appearance) used by ``select_splice_counterfactual_features_for_shard``: that
    function groups packed segments by ``(base_shard_idx, doc_idx)`` and visits
    groups in ascending order of each group's first packed-segment index — the
    same LRU-1 ``FeatureIndex`` shard-cache access sequence reproduced here via
    ``PROVENANCE_BASE_SHARD_IDX_KEY``/``PROVENANCE_DOC_IDX_KEY`` directly (no
    manifest ``base_shard_stems`` lookup needed: alternation/distinct counts on
    the integer ``base_shard_idx`` are identical to counting on the resolved
    stem strings, since that mapping is a bijection).
    """
    prov_dir = provenance_dir(packed_row_dir)
    max_distinct = 0
    max_alt = 0
    for pp in sorted(prov_dir.glob("*.safetensors.zst")):
        prov = load_overlay_shard(pp)
        b = prov[PROVENANCE_BASE_SHARD_IDX_KEY].to(t.int64)
        d = prov[PROVENANCE_DOC_IDX_KEY].to(t.int64)
        key = b * 1_000_000 + d
        uniq, inv = t.unique(key, return_inverse=True)
        first = t.full((uniq.numel(),), t.iinfo(t.int64).max, dtype=t.int64)
        first.scatter_reduce_(0, inv, t.arange(key.numel(), dtype=t.int64), reduce="amin")
        group_stems = (uniq[first.argsort(stable=True)] // 1_000_000).tolist()
        max_distinct = max(max_distinct, len(set(group_stems)))
        # Count transitions between adjacent source stems.
        alt = sum(
            1 for a, c in itertools.pairwise(group_stems) if a != c
        )
        max_alt = max(max_alt, alt)
    return max_distinct, max_alt


def run_selection_main(
    env: dict,
    *,
    cache_shards: int | None = None,
    sidecar_shards: int | None = None,
    prefetch_stage_shards: int | None = None,
    num_workers: int | None = None,
    shared_store: object = False,
    shared_store_capacity: int | None = None,
    shm_dir: str | None = None,
    packed_spool_dir: str | None = None,
    selection_resume: object = True,
    splice_batch_size: object = None,
    mask: dict | None = None,
    clusters_path: str | None = None,
    monkeypatch=None,
) -> dict[str, bytes]:
    """Drive precompute_selected_features.main.__wrapped__ on the fixture with a
    frozen created_at, returning {basename: bytes} for every sel/ output file.

    cache_shards/sidecar_shards/prefetch_stage_shards are omitted from cfg when
    None so this helper works before PR-2/PR-3 add the keys (pre-change golden).
    """
    cfg_dict: dict = {
        "packed_row_dir": str(env["packed_row_dir"]),
        "feature_dir": env["feature_dir"],
        "segment_size": SELECTION_SEG_SIZE,
        "feature_selection": {
            "distribution": "geometric",
            "mean": 12.0,
            "std": 12.0,
            "min_k": 1,
            "max_k": 49,
            "mode": "top_k_percentile",
            "delta_weights": "percentile_delta",
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 12.0,
            "min_k": 0,
            "max_k": 49,
        },
        "feature_delta": {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
        "feature_mask_freq_min": None,
        "feature_mask_freq_max": None,
        "selection_resume": selection_resume,
        "num_buckets": 2,
        "seed": 42,
        "sae": {
            "release": "gemmascope_2",
            "model": "gemma-3-4b-pt",
            "activation_type": "residual",
            "layer": 8,
            "width": "65k",
            "l0": "medium",
        },
        "feature_root": None,
        "pooling": "max_pool",
        "dataset_phase": None,
    }
    if mask is not None:
        cfg_dict["feature_mask"] = mask
    if clusters_path is not None:
        cfg_dict["feature_clusters"] = {"path": clusters_path}
    if cache_shards is not None:
        cfg_dict["selection_source_cache_shards"] = cache_shards
    if sidecar_shards is not None:
        cfg_dict["selection_sidecar_cache_shards"] = sidecar_shards
    if prefetch_stage_shards is not None:
        cfg_dict["selection_prefetch_stage_shards"] = prefetch_stage_shards
    if num_workers is not None:
        cfg_dict["selection_num_workers"] = num_workers
    if shared_store is not False:
        cfg_dict["selection_shared_store"] = shared_store
        if shared_store_capacity is not None:
            cfg_dict["selection_shared_store_capacity"] = shared_store_capacity
    if packed_spool_dir is not None:
        cfg_dict["selection_packed_spool_dir"] = packed_spool_dir
    if splice_batch_size is not None:
        cfg_dict["selection_splice_batch_size"] = splice_batch_size

    if monkeypatch is not None:
        monkeypatch.setattr(
            overlay_io.time, "strftime", lambda *a, **k: "2026-01-01T00:00:00Z"
        )
        # Point the shared store at a test-local tmpfs-substitute so it never touches the
        # real /dev/shm; spawn workers inherit the env var.
        if shm_dir is not None:
            monkeypatch.setenv("NANOCHAT_SHM_DIR", shm_dir)

    sel_root = env["packed_row_dir"] / "sel"
    prior_sel_dirs = set(sel_root.iterdir()) if sel_root.is_dir() else set()
    cfg = OmegaConf.create(cfg_dict)
    psf.main.__wrapped__(cfg)

    sel_dirs = [p for p in sel_root.iterdir() if p.is_dir()]
    new_sel_dirs = set(sel_dirs) - prior_sel_dirs
    if len(new_sel_dirs) == 1:
        selected_dir = new_sel_dirs.pop()
    else:
        assert len(sel_dirs) == 1, f"expected one unambiguous sel dir, got {sel_dirs}"
        selected_dir = sel_dirs[0]
    return {p.name: p.read_bytes() for p in sorted(selected_dir.glob("*"))}


# Every derived shard this fixture builds produces the IDENTICAL active-id
# set {0, 37, 42, 74, 79, 84, 111, 116, 121, 126, 148, 185, 190} -- verified
# empirically (see tests/steering/_splice_fixtures.py::_fabricate_fids: fids
# depend only on the per-shard segment counter ``gi``, which restarts at 0
# for every one of build_interleaved_env's source shards). write_cluster_yaml
# below draws its "real" cluster members from this set so mask/cluster
# parity arms exercise genuine consolidation on live segments rather than a
# structurally-valid-but-never-triggered configuration.
#
# Whitelist covering all real ids < 150 (0, 37, 42, 74, 79, 84, 111, 116,
# 121, 126, 148) while excluding 185/190 -- a genuine partial filter, not a
# no-op -- and excluding the decoy ids (>= 150) write_cluster_yaml uses for
# the identity_after_mask branch below.
WHITELIST_MASK_BELOW_150 = {"mode": "whitelist", "ids": list(range(150))}


def write_cluster_yaml(tmp_path, *, identity_after_mask: bool = False) -> str:
    """Write a schema-v1 static feature-cluster file for the parity matrix.

    Pairs with ``WHITELIST_MASK_BELOW_150`` (ids 0-149).

    identity_after_mask=False -> clusters of REAL co-occurring active ids
        (37+42 co-occur in one segment of every source shard; 74+79+84 in
        another; 111+116+121+126 in another -- see the module comment
        above) -- all members < 150, so all survive the paired whitelist and
        FeatureClusterConsolidation genuinely consolidates multiple
        surviving members on real segments (is_identity_after_mask=False).
    identity_after_mask=True -> each cluster pairs one real, whitelisted id
        with a decoy id >= 150 (excluded by WHITELIST_MASK_BELOW_150, still
        < SELECTION_N_SAE so cluster-file validation accepts it) -> exactly
        one survivor per cluster -> no consolidation fires
        (is_identity_after_mask=True) even though every cluster is a
        structurally valid (>= 2 member) definition.

    NOTE on a brief bug fixed here: the original task-2.4-brief draft used
    singleton clusters (``[[10], [20], [30]]``) for the identity_after_mask
    case. ``FeatureClusterDefinition.__post_init__`` rejects any cluster
    with fewer than 2 members ("must contain at least two source IDs"), so
    that draft raises FeatureClustersError immediately and was never
    reachable. It also drew ids (10, 11, 12, 20, 21, 30, 31, 32, 33) that
    never occur as active features in this fixture regardless (see the
    module comment above), which would have made the "active" arm
    degenerate (structurally valid but never actually consolidating on real
    data) even after fixing the member-count bug. Both are fixed by using
    real, empirically-verified active ids paired with a two-member (not
    singleton) decoy construction.
    """
    if identity_after_mask:
        clusters = [[real, 150 + i] for i, real in enumerate((37, 74, 111))]
    else:
        clusters = [[37, 42], [74, 79, 84], [111, 116, 121, 126]]
    path = Path(tmp_path) / f"clusters_{'identity' if identity_after_mask else 'active'}.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "clusters": clusters}))
    return str(path)
