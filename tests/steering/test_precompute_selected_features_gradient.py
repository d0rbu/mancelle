"""Gradient-attribution ranking in precompute_selected_features.

Two layers:

1. SYNTHETIC: hand-built PackedRowShard + stubbed score joins drive
   ``select_features_for_shard(gradient_join=...)`` so the selection
   semantics are pinned surgically — strict score>0 eligibility (no
   percentile fallback, no random fill), ranking by score with the drawn k
   capped at the eligible pool, empty selections for zero-eligible segments,
   percentile-valued ``sel_pcts``, rerun determinism, and misalignment
   refusals.

2. FULL PIPELINE: the tiny corpus → REAL grad overlay (stage-2 producer) →
   REAL packed rows (whose packer records provenance natively) →
   ``resolve_gradient_artifacts`` + selection — the integration capstone,
   including non-vacuity vs a percentile-ranked run and every artifact
   refusal (missing overlay / missing provenance / legacy packed dirs).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.preprocess.precompute_gradient_attribution as grad_producer_mod
from errors import GradientAttributionError
from scripts.preprocess.precompute_selected_features import (
    GradScoreJoiner,
    ShardGradientJoin,
    resolve_gradient_artifacts,
    select_features_for_shard,
)
from steering.conditional.feature_pipeline import build_feature_pipeline
from steering.conditional.sampling import parse_feature_selection
from steering.gradient_overlay import (
    GRAD_DOC_SEG_OFFSETS_KEY,
    GRAD_OFFSETS_KEY,
    GRAD_SCORES_KEY,
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
)
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import load_packed_row_shard, load_selection_shard
from tests.steering.test_gradient_attribution_producer_integration import (  # noqa: F401
    TinyTok,
    _corpus_template,
    corpus,
    patched_builders,
)
from tests.steering.test_precompute_packed_provenance import (  # noqa: F401
    packed,
)
from tests.steering.test_precompute_selected_features_mask import (
    N_SAE_FEATURES,
    _build_base_shard,
    _make_per_feature_stats,
)

_N_SEGS = 6
_N_FEATS_PER_SEG = 8


def _gradient_pipeline(seed: int = 1, min_k: int = 1, max_k: int | None = 25,
                       mean: float = 5.0):
    fs = parse_feature_selection(
        {
            "distribution": "geometric",
            "mean": mean,
            "mode": "top_k_percentile",
            "min_k": min_k,
            "max_k": max_k,
            "gradient": {"mode": "future_half_split", "model_dtype": "float32",
                         "min_final_segment_tokens": 4},
        }
    )
    return build_feature_pipeline(
        feature_selection=fs,
        negative_selection=None,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        seed=seed,
    )


class _StubJoiner:
    """GradScoreJoiner stand-in: scores keyed by (base_idx, doc_idx, ord)."""

    def __init__(self, table: dict[tuple[int, int, int], t.Tensor]):
        self.table = table

    def retain(self, base_indices: set[int]) -> None:
        del base_indices  # stub holds everything

    def scores(self, b: int, d: int, s: int) -> t.Tensor:
        return self.table[(b, d, s)].to(t.float32)


def _join_for(shard, seg_scores: list[t.Tensor]) -> ShardGradientJoin:
    """Build a ShardGradientJoin whose segment s yields seg_scores[s].
    A ``None`` entry leaves a negative (-1) triple — the packer never emits
    those, so it models CORRUPTION (ShardGradientJoin raises on construction).
    Tests of valid behavior pass all-real scores."""
    n = shard.total_segments
    base_idx = t.full((n,), -1, dtype=t.int32)
    doc_idx = t.full((n,), -1, dtype=t.int32)
    seg_ord = t.full((n,), -1, dtype=t.int32)
    table: dict[tuple[int, int, int], t.Tensor] = {}
    for s, scores in enumerate(seg_scores):
        if scores is None:
            continue
        base_idx[s] = 0
        doc_idx[s] = 0
        seg_ord[s] = s
        table[(0, 0, s)] = scores
    prov = {
        PROVENANCE_BASE_SHARD_IDX_KEY: base_idx,
        PROVENANCE_DOC_IDX_KEY: doc_idx,
        PROVENANCE_SEG_ORDINAL_KEY: seg_ord,
    }
    return ShardGradientJoin(cast(Any, _StubJoiner(table)), prov)


def test_gradscore_joiner_bounds_raises(tmp_path):
    """The REAL GradScoreJoiner.scores offset arithmetic (the production tests
    use _StubJoiner, which bypasses it). Out-of-range doc_idx / negative or
    out-of-range seg_ordinal each raise a typed error rather than silently
    reading a neighbouring segment's scores."""
    grad_dir = tmp_path / "grad"
    grad_dir.mkdir()
    # one base doc with 2 segments; seg 0 has 3 feats, seg 1 has 2.
    save_overlay_shard(
        {
            GRAD_DOC_SEG_OFFSETS_KEY: t.tensor([0, 2], dtype=t.int32),
            GRAD_OFFSETS_KEY: t.tensor([0, 3, 5], dtype=t.int64),
            GRAD_SCORES_KEY: t.arange(5, dtype=t.float16),
        },
        overlay_shard_path(grad_dir, "s0"),
    )
    joiner = GradScoreJoiner(grad_dir, ["s0"])
    # happy path: seg 0 → 3 scores, seg 1 → 2 scores.
    assert joiner.scores(0, 0, 0).shape[0] == 3
    assert joiner.scores(0, 0, 1).shape[0] == 2
    with pytest.raises(GradientAttributionError, match="doc_idx 1 out of range"):
        joiner.scores(0, 1, 0)  # only 1 doc
    with pytest.raises(GradientAttributionError, match="negative"):
        joiner.scores(0, 0, -1)
    with pytest.raises(GradientAttributionError, match="seg_ordinal 2 out of range"):
        joiner.scores(0, 0, 2)  # doc 0 has only segs 0,1


class TestGradientSelectionSemantics:
    def _run(self, tmp_path, seg_scores, seed=1, **pipe_kw):
        shard, shard_path = _build_base_shard(tmp_path)
        out_dir = tmp_path / "sel"
        out_dir.mkdir(exist_ok=True)
        pipeline = _gradient_pipeline(seed=seed, **pipe_kw)
        join = _join_for(shard, seg_scores)
        select_features_for_shard(
            shard_path, pipeline, out_dir, "shard_00000", gradient_join=join
        )
        sel = load_selection_shard(out_dir / "shard_00000.safetensors.zst")
        return shard, sel

    def test_strict_eligibility_and_ranking(self, tmp_path):
        """Only score>0 features are selectable; the selected set is the top
        of the eligible pool by score (min selected score >= max unselected
        eligible score); zero-score features never appear."""
        gen = t.Generator().manual_seed(3)
        seg_scores = []
        for _ in range(_N_SEGS):
            scores = t.rand(_N_FEATS_PER_SEG, generator=gen)
            scores[::2] = 0.0  # half ineligible
            seg_scores.append(scores)
        shard, sel = self._run(tmp_path, seg_scores, max_k=4)

        offsets = shard.seg_feat_offsets.tolist()
        sel_off = sel.sel_offsets.tolist()
        checked = 0
        for s in range(_N_SEGS):
            fids = shard.feature_ids[offsets[s] : offsets[s + 1]].to(t.int64)
            scores = seg_scores[s]
            by_fid = {int(f): float(v) for f, v in zip(fids, scores, strict=True)}
            chosen = sel.sel_fids[sel_off[s] : sel_off[s + 1]].to(t.int64).tolist()
            assert chosen, f"segment {s} selected nothing despite eligible features"
            for fid in chosen:
                assert by_fid[fid] > 0, f"segment {s} selected zero-score fid {fid}"
            unchosen_elig = [
                v for f, v in by_fid.items() if v > 0 and f not in set(chosen)
            ]
            if unchosen_elig:
                assert min(by_fid[f] for f in chosen) >= max(unchosen_elig), (
                    f"segment {s}: selection is not the top of the eligible "
                    f"pool by gradient score"
                )
                checked += 1
        assert checked > 0, "ranking property never exercised (k too large?)"

    def test_capped_at_eligible_pool(self, tmp_path):
        """2 eligible features with a huge drawn k => exactly those 2 selected
        (k reduced to what's available, no fill)."""
        seg_scores = []
        for _ in range(_N_SEGS):
            scores = t.zeros(_N_FEATS_PER_SEG)
            scores[0] = 0.5
            scores[3] = 0.25
            seg_scores.append(scores)
        shard, sel = self._run(tmp_path, seg_scores, mean=50.0, max_k=None, min_k=10)
        offsets = shard.seg_feat_offsets.tolist()
        sel_off = sel.sel_offsets.tolist()
        for s in range(_N_SEGS):
            fids = shard.feature_ids[offsets[s] : offsets[s + 1]].to(t.int64)
            chosen = set(sel.sel_fids[sel_off[s] : sel_off[s + 1]].tolist())
            assert chosen == {int(fids[0]), int(fids[3])}, (
                f"segment {s}: expected exactly the 2 eligible features, got {chosen}"
            )

    def test_zero_eligible_yields_empty_selection(self, tmp_path):
        seg_scores = [t.zeros(_N_FEATS_PER_SEG) for _ in range(_N_SEGS)]
        _, sel = self._run(tmp_path, seg_scores)
        assert sel.total_selected == 0
        assert sel.sel_offsets.tolist() == [0] * (_N_SEGS + 1)

    def test_negative_provenance_refused(self, tmp_path):
        """The packer only writes non-negative triples (every packed segment
        has a real base segment — there is no sentinel/filler value), so a
        negative field is corruption and must raise loud, never silently
        become an empty selection."""
        seg_scores: list = [None] * _N_SEGS  # all-(-1) triples => corruption
        with pytest.raises(GradientAttributionError, match="negative"):
            self._run(tmp_path, seg_scores)

    def test_sel_pcts_stay_percentile_valued(self, tmp_path):
        """The stored sel_pcts must be the segment's ACTIVATION percentiles
        (uint8 200 in the fixture), never gradient-derived."""
        gen = t.Generator().manual_seed(5)
        seg_scores = [t.rand(_N_FEATS_PER_SEG, generator=gen) for _ in range(_N_SEGS)]
        _, sel = self._run(tmp_path, seg_scores)
        assert sel.total_selected > 0
        assert set(sel.sel_pcts.tolist()) == {200}

    def test_rerun_determinism(self, tmp_path):
        gen = t.Generator().manual_seed(7)
        seg_scores = [t.rand(_N_FEATS_PER_SEG, generator=gen) for _ in range(_N_SEGS)]
        _, sel_a = self._run(tmp_path / "a", seg_scores, seed=9)
        _, sel_b = self._run(tmp_path / "b", seg_scores, seed=9)
        assert t.equal(sel_a.sel_fids, sel_b.sel_fids)
        assert t.equal(sel_a.sel_offsets, sel_b.sel_offsets)

    def test_score_misalignment_refused(self, tmp_path):
        shard, shard_path = _build_base_shard(tmp_path)
        out_dir = tmp_path / "sel"
        out_dir.mkdir(exist_ok=True)
        bad = [t.rand(_N_FEATS_PER_SEG + 1) for _ in range(_N_SEGS)]  # wrong length
        join = _join_for(shard, bad)
        with pytest.raises(GradientAttributionError, match="misalignment"):
            select_features_for_shard(
                shard_path, _gradient_pipeline(), out_dir, "shard_00000",
                gradient_join=join,
            )

    def test_provenance_segment_count_mismatch_refused(self, tmp_path):
        shard, shard_path = _build_base_shard(tmp_path)
        out_dir = tmp_path / "sel"
        out_dir.mkdir(exist_ok=True)
        # Build a VALID join (all non-negative), then truncate its arrays so
        # total_segments < the shard's — the count check, not the corruption
        # guard, is what we exercise here.
        join = _join_for(shard, [t.rand(_N_FEATS_PER_SEG) for _ in range(_N_SEGS)])
        join.base_idx = join.base_idx[:-1]
        join.doc_idx = join.doc_idx[:-1]
        join.seg_ordinal = join.seg_ordinal[:-1]
        with pytest.raises(GradientAttributionError, match="provenance has"):
            select_features_for_shard(
                shard_path, _gradient_pipeline(), out_dir, "shard_00000",
                gradient_join=join,
            )

# ---------------------------------------------------------------------------
# Full pipeline: corpus -> grad overlay -> packed + provenance -> selection
# ---------------------------------------------------------------------------

SEL_CFG_SAE = {
    "model": "gemma-2-2b",
    "layer": 2,
    "release": "gemmascope",
    "width": "16k",
    "l0": None,
    "activation_type": "residual",
}


def _sel_cfg(**sae_overrides):
    sae = dict(SEL_CFG_SAE)
    sae.update(sae_overrides)
    return OmegaConf.create({"sae": sae})


@pytest.fixture()
def grad_ready(corpus, packed, patched_builders):  # noqa: F811
    """Grad overlay (stage-2 producer, future_half_split/min_final=4/fp32)
    over the shared tiny corpus; packed rows (from the ``packed`` fixture)
    already carry their provenance natively."""
    grad_cfg = OmegaConf.create(
        {
            "feature_dir": str(corpus["feature_dir"]),
            "data_dir": str(corpus["data_dir"]),
            "dataset_phase": None,
            "model": "gemma-2-2b",
            "activation_type": "residual",
            "layer": 2,
            "pooling": "max_pool",
            "segment_size": 8,
            "sae_width": "16k",
            "feature_root": "unused",
            "sae": {"release": "gemmascope", "l0": None},
            "context_window": 64,
            "attribution_mode": "future_half_split",
            "min_final_segment_tokens": 4,
            "device": "cpu",
            "model_dtype": "float32",
            "batch_size": 3,
            "resume": False,
            "force": False,
            "dry_run": False,
            "zstd_level": None,
        }
    )
    assert grad_producer_mod.run(grad_cfg) == 0
    return packed


class TestSelOverlayStalenessGuard:
    """#2a: a sel dir built against a DIFFERENT grad overlay must be refused
    (the sel hash is config-only; the per-shard skip would otherwise serve
    selections from the old overlay's scores)."""

    def test_refuses_sel_dir_from_different_overlay(self, tmp_path):
        from scripts.preprocess.precompute_selected_features import (
            _assert_sel_overlay_unchanged,
        )
        from steering.overlay_io import OverlayKind, write_manifest

        d = tmp_path / "sel" / "h"
        d.mkdir(parents=True)
        # No gradient run / no manifest yet => no-op.
        _assert_sel_overlay_unchanged(d, None)
        _assert_sel_overlay_unchanged(d, "A" * 64)

        write_manifest(
            d, "selhash", OverlayKind.PACKED_ROW,
            metadata={"gradient_attribution": {"overlay_checksum": "A" * 64}},
        )
        _assert_sel_overlay_unchanged(d, "A" * 64)  # same overlay => ok
        _assert_sel_overlay_unchanged(d, None)  # non-gradient run => always ok
        with pytest.raises(GradientAttributionError, match="stale"):
            _assert_sel_overlay_unchanged(d, "B" * 64)  # different overlay


class TestFullPipelineSelection:
    def _spec(self) -> Any:
        fs = parse_feature_selection(
            {
                "distribution": "geometric",
                "mean": 5.0,
                "mode": "top_k_percentile",
                "min_k": 1,
                "max_k": 25,
                "gradient": {
                    "mode": "future_half_split",
                    "model_dtype": "float32",
                    "min_final_segment_tokens": 4,
                },
            }
        )
        return fs

    def test_resolve_and_select_end_to_end(self, grad_ready, tmp_path):
        fs = self._spec()
        joiner, prov_dir, checksum = resolve_gradient_artifacts(
            _sel_cfg(),
            grad_ready["feature_dir"],
            grad_ready["train_dir"],
            fs.gradient,
        )
        assert isinstance(joiner, GradScoreJoiner)
        assert prov_dir.is_dir()
        assert len(checksum) == 64

        pipeline = build_feature_pipeline(
            feature_selection=fs,
            negative_selection=None,
            per_feature_stats=_make_per_feature_stats(48),
            n_sae_features=48,
            num_buckets=2,
            seed=11,
        )
        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        from steering.overlay_io import load_overlay_shard

        total_selected = 0
        for shard_path in glob_overlay_shards(grad_ready["train_dir"]):
            name = shard_path.stem.split(".")[0]
            join = ShardGradientJoin(
                joiner, load_overlay_shard(prov_dir / shard_path.name)
            )
            select_features_for_shard(
                shard_path, pipeline, out_dir, name, gradient_join=join
            )
            sel = load_selection_shard(out_dir / shard_path.name)
            packed_shard = load_packed_row_shard(shard_path)
            assert sel.total_segments == packed_shard.total_segments
            total_selected += sel.total_selected
        assert total_selected > 0, "gradient selection selected nothing at all"

    def test_differs_from_percentile_ranking(self, grad_ready, tmp_path):
        """Same seed, gradient vs percentile ranking => different selections
        somewhere (non-vacuity)."""
        fs_grad = self._spec()
        joiner, prov_dir, _ = resolve_gradient_artifacts(
            _sel_cfg(), grad_ready["feature_dir"], grad_ready["train_dir"],
            fs_grad.gradient,
        )
        fs_pct = parse_feature_selection(
            {
                "distribution": "geometric",
                "mean": 5.0,
                "mode": "top_k_percentile",
                "min_k": 1,
                "max_k": 25,
            }
        )
        from steering.overlay_io import load_overlay_shard

        results = {}
        for label, fs, with_join in [("grad", fs_grad, True), ("pct", fs_pct, False)]:
            pipeline = build_feature_pipeline(
                feature_selection=fs,
                negative_selection=None,
                per_feature_stats=_make_per_feature_stats(48),
                n_sae_features=48,
                num_buckets=2,
                seed=11,
            )
            out_dir = tmp_path / label
            out_dir.mkdir()
            shard_path = glob_overlay_shards(grad_ready["train_dir"])[0]
            join = (
                ShardGradientJoin(
                    joiner, load_overlay_shard(prov_dir / shard_path.name)
                )
                if with_join
                else None
            )
            select_features_for_shard(
                shard_path, pipeline, out_dir,
                shard_path.stem.split(".")[0], gradient_join=join,
            )
            results[label] = load_selection_shard(out_dir / shard_path.name)
        assert not t.equal(results["grad"].sel_fids, results["pct"].sel_fids) or not (
            t.equal(results["grad"].sel_offsets, results["pct"].sel_offsets)
        ), "gradient ranking produced identical selections to percentile ranking"

    def test_missing_overlay_names_producer(self, grad_ready):
        fs: Any = parse_feature_selection(
            {
                "distribution": "geometric",
                "mean": 5.0,
                "mode": "top_k_percentile",
                "gradient": {"mode": "self", "model_dtype": "float32",
                             "min_final_segment_tokens": 4},
            }
        )
        with pytest.raises(
            GradientAttributionError, match="precompute_gradient_attribution"
        ):
            resolve_gradient_artifacts(
                _sel_cfg(), grad_ready["feature_dir"], grad_ready["train_dir"],
                fs.gradient,
            )

    def test_missing_provenance_names_packer_heal(self, grad_ready):
        import shutil

        fs = self._spec()
        shutil.rmtree(grad_ready["train_dir"] / "provenance")
        with pytest.raises(
            GradientAttributionError, match="precompute_packed_rows"
        ):
            resolve_gradient_artifacts(
                _sel_cfg(), grad_ready["feature_dir"], grad_ready["train_dir"],
                fs.gradient,
            )

    def test_legacy_packed_dir_without_stems_refused(self, grad_ready):
        """A packed dir regenerated by pre-integration code (manifest without
        base_shard_stems) must fail loud, naming the heal command."""
        fs = self._spec()
        manifest = read_manifest(grad_ready["train_dir"])
        assert manifest is not None
        legacy_meta = {
            k: v
            for k, v in manifest["metadata"].items()
            if k not in ("base_shard_stems", "provenance_shards")
        }
        write_manifest(
            grad_ready["train_dir"],
            manifest["checksum"],
            OverlayKind.PACKED_ROW,
            metadata=legacy_meta,
        )
        with pytest.raises(
            GradientAttributionError,
            match="predates packer-integrated provenance",
        ):
            resolve_gradient_artifacts(
                _sel_cfg(), grad_ready["feature_dir"], grad_ready["train_dir"],
                fs.gradient,
            )

    def test_incomplete_provenance_refused(self, grad_ready):
        """A provenance dir missing a shard (count mismatch vs packed shards)
        must fail loud at resolve time, not at first join."""
        fs = self._spec()
        victim = glob_overlay_shards(
            grad_ready["train_dir"] / "provenance"
        )[0]
        victim.unlink()
        with pytest.raises(
            GradientAttributionError, match="incomplete or stale provenance"
        ):
            resolve_gradient_artifacts(
                _sel_cfg(), grad_ready["feature_dir"], grad_ready["train_dir"],
                fs.gradient,
            )
