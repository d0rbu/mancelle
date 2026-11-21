"""Tests for the splice-preselection source-shard locality analyzer (issue #740).

``scripts.analysis.splice_preselection_locality`` reconstructs the exact source-shard
access sequence ``select_splice_counterfactual_features_for_shard`` visits during
splice-counterfactual preselection, from provenance shards alone (never the multi-GB
feature shards), then simulates LRU cache load counts per capacity. These tests exercise
the LRU simulator in isolation, the group-order reconstruction against hand-built
provenance, ``analyze()``/CLI behavior, and an internal-consistency check against the
shared Task-1.1 interleaved fixture (an independently written implementation of the same
group-order technique). A separate real-corpus regression test (gated ``slow`` +
``requires_feature_data``) pins the exact counts measured against the production l22
corpus while planning this work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch as t

from scripts.analysis.splice_preselection_locality import (
    _group_stem_order,
    analyze,
    main,
    simulate_lru_loads,
)
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    load_overlay_shard,
    save_overlay_shard,
    write_manifest,
)
from tests.steering._splice_cf_cache_fixtures import (
    build_interleaved_env,
    worst_case_stem_alternations,
)

# The real production packed-row corpus this analyzer was verified against while writing
# the issue #740 plan. Only consulted by TestRealCorpusRegression, which skips cleanly
# when this box doesn't have it.
_PROD_PACKED_ROW_DIR = Path(
    "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/out/data/features/s128/l22/a_res/"
    "p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/splice/cut_frac0.15/"
    "seed42/pct/623c372d3f121b18/tok/dc6308e6116cbfe0/c2048/sall"
)


def _write_prov(packed_dir: Path, stem: str, base_idx, doc_idx):
    prov_dir = provenance_dir(packed_dir)
    prov_dir.mkdir(parents=True, exist_ok=True)
    n = len(base_idx)
    save_overlay_shard(
        {
            PROVENANCE_BASE_SHARD_IDX_KEY: t.tensor(base_idx, dtype=t.int32),
            PROVENANCE_DOC_IDX_KEY: t.tensor(doc_idx, dtype=t.int32),
            PROVENANCE_SEG_ORDINAL_KEY: t.zeros(n, dtype=t.int32),
        },
        prov_dir / stem,
    )


def _reference_stem_sequence(packed_row_dir: Path) -> list[int]:
    """Test-only reconstruction used to check ``analyze`` across shards."""
    sequence: list[int] = []
    for path in sorted(provenance_dir(packed_row_dir).glob("*.safetensors.zst")):
        provenance = load_overlay_shard(path)
        sequence.extend(
            _group_stem_order(
                provenance[PROVENANCE_BASE_SHARD_IDX_KEY],
                provenance[PROVENANCE_DOC_IDX_KEY],
            )
        )
    return sequence


def test_simulate_lru_loads_alternation() -> None:
    # strict A/B alternation: capacity 1 reloads every step; capacity 2 loads each once
    seq = [0, 1, 0, 1, 0, 1]
    assert simulate_lru_loads(seq, 1) == 6
    assert simulate_lru_loads(seq, 2) == 2


def test_reference_sequence_matches_group_order(tmp_path: Path) -> None:
    packed = tmp_path / "packed"
    packed.mkdir()
    # one packed shard: segments interleave (base0,doc0),(base1,doc0),(base0,doc0)
    _write_prov(packed, "shard_00000", base_idx=[0, 1, 0], doc_idx=[0, 0, 0])
    write_manifest(
        packed, "m", OverlayKind.PACKED_ROW,
        metadata={"base_shard_stems": ["shard_00000", "shard_00001"], "provenance_shards": 1},
    )
    # groups keyed (base,doc); first-appearance order: (0,0)@seg0, (1,0)@seg1
    assert _reference_stem_sequence(packed) == [0, 1]


class TestSimulateLruLoads:
    """Unit coverage of the plain textbook LRU simulator (no I/O)."""

    def test_rejects_nonpositive_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            simulate_lru_loads([0, 1], 0)
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            simulate_lru_loads([0, 1], -1)

    def test_empty_sequence_is_zero_loads_at_any_capacity(self) -> None:
        assert simulate_lru_loads([], 1) == 0
        assert simulate_lru_loads([], 8) == 0

    def test_capacity_covering_working_set_loads_each_stem_once(self) -> None:
        seq = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        assert simulate_lru_loads(seq, 3) == 3

    def test_capacity_below_working_set_thrashes_completely(self) -> None:
        # 3 distinct stems visited cyclically; capacity 2 can never retain the stem
        # that will be needed 2 steps ahead, so every single access misses.
        seq = [0, 1, 2, 0, 1, 2]
        assert simulate_lru_loads(seq, 2) == 6

    def test_monotonic_non_increasing_in_capacity(self) -> None:
        # LRU is a "stack algorithm" (Mattson et al. 1970): the contents held at
        # capacity k are always a subset of the contents held at capacity k+1 for the
        # same access sequence, so load counts can only fall (or hold) as capacity
        # grows -- unlike FIFO, LRU never exhibits Belady's anomaly.
        seq = [0, 1, 2, 0, 3, 1, 2, 0, 4, 4, 1]
        loads = [simulate_lru_loads(seq, c) for c in range(1, 6)]
        assert loads == sorted(loads, reverse=True)
        assert loads[-1] == len(set(seq))


class TestReferenceStemSequence:
    """Group-order reconstruction from provenance shards alone."""

    def test_concatenates_multiple_shards_in_sorted_glob_order(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed"
        packed.mkdir()
        # Written out of order on purpose: the reference must sort by name.
        _write_prov(packed, "shard_00001", base_idx=[2], doc_idx=[0])
        _write_prov(packed, "shard_00000", base_idx=[0, 1], doc_idx=[0, 0])
        write_manifest(packed, "m", OverlayKind.PACKED_ROW, metadata={})
        assert _reference_stem_sequence(packed) == [0, 1, 2]

    def test_one_entry_per_group_not_per_segment(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed"
        packed.mkdir()
        # 5 packed segments but only 2 (base,doc) groups: (0,0) has 3 segments
        # interleaved with (1,0)'s 2 -- the sequence must still be length 2.
        _write_prov(packed, "shard_00000", base_idx=[0, 0, 1, 0, 1], doc_idx=[0, 0, 0, 0, 0])
        write_manifest(packed, "m", OverlayKind.PACKED_ROW, metadata={})
        assert _reference_stem_sequence(packed) == [0, 1]

    def test_distinguishes_same_base_shard_different_docs(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed"
        packed.mkdir()
        # same base_shard_idx, two different docs within it -- these are TWO groups,
        # not one, since the group key is (base_shard_idx, doc_idx).
        _write_prov(packed, "shard_00000", base_idx=[0, 0], doc_idx=[0, 1])
        write_manifest(packed, "m", OverlayKind.PACKED_ROW, metadata={})
        assert _reference_stem_sequence(packed) == [0, 0]

    def test_doc_idx_at_or_above_stride_raises_loud(self) -> None:
        """The fail-loud guard: a doc_idx >= _DOC_KEY_STRIDE must raise rather than
        silently collide two distinct (base, doc) groups into the same composite
        key -- the exact silent-wrong-answer failure mode this project forbids
        (issues #588/#602). Calls _group_stem_order directly since this is a
        property of the key encoding, independent of any provenance shard I/O."""
        with pytest.raises(ValueError, match="stride"):
            _group_stem_order(t.tensor([0, 0]), t.tensor([0, 1_000_000]))


class TestAnalyze:
    def test_structure_and_load_counts(self, tmp_path: Path) -> None:
        packed = tmp_path / "packed"
        packed.mkdir()
        _write_prov(packed, "shard_00000", base_idx=[0, 1], doc_idx=[0, 0])
        _write_prov(packed, "shard_00001", base_idx=[2, 0], doc_idx=[0, 0])
        write_manifest(packed, "m", OverlayKind.PACKED_ROW, metadata={})

        result = analyze(packed, capacities=[1, 3])

        assert result["n_packed_shards"] == 2
        assert result["unique_source_stems"] == 3
        assert result["distinct_stems"]["max"] == 2  # each shard groups to 2 distinct stems
        # analyze() re-derives full_seq internally; cross-check both assemblies.
        assert result["loads_by_capacity"][1] == simulate_lru_loads(
            _reference_stem_sequence(packed), 1
        )
        assert result["loads_by_capacity"][3] == 3  # capacity covers the whole working set
        assert [s["shard"] for s in result["per_shard"]] == ["shard_00000", "shard_00001"]
        assert json.dumps(result)  # every field must survive JSON round-trip (--json-out)

    def test_on_shared_interleaved_fixture_is_internally_consistent(
        self, tmp_path: Path
    ) -> None:
        """Cross-checks against the Task-1.1 fixture helper (an independent
        implementation of the identical group-order technique) on a REAL packed
        corpus with genuine multi-source interleaving, not a hand-built sequence."""
        env = build_interleaved_env(tmp_path)
        packed_dir = Path(env["packed_row_dir"])
        max_distinct, max_alt = worst_case_stem_alternations(packed_dir)
        assert max_distinct >= 3 and max_alt >= 4  # this really is the interleaved fixture

        seq = _reference_stem_sequence(packed_dir)
        ideal = len(set(seq))
        result = analyze(packed_dir, capacities=[1, 2, 1000])

        assert result["unique_source_stems"] == ideal
        assert result["loads_by_capacity"][1] == simulate_lru_loads(seq, 1)
        assert result["loads_by_capacity"][2] == simulate_lru_loads(seq, 2)
        # A capacity covering every distinct stem ever seen can never reload one.
        assert result["loads_by_capacity"][1000] == ideal
        # LRU is a stack algorithm: strictly more cache never causes strictly more loads.
        assert result["loads_by_capacity"][1] >= result["loads_by_capacity"][2] >= ideal


class TestMainCli:
    def _build_packed_dir(self, tmp_path: Path) -> Path:
        packed = tmp_path / "packed"
        packed.mkdir()
        _write_prov(packed, "shard_00000", base_idx=[0, 1, 0], doc_idx=[0, 0, 0])
        write_manifest(packed, "m", OverlayKind.PACKED_ROW, metadata={})
        return packed

    def test_prints_summary_for_requested_capacities(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        packed = self._build_packed_dir(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["splice_preselection_locality.py", str(packed), "--capacities", "1,2"]
        )
        main()
        out = capsys.readouterr().out
        assert "packed shards: 1" in out
        assert "unique source stems: 2" in out
        assert "capacity   1:" in out
        assert "capacity   2:" in out

    def test_default_capacities_are_1_2_4_5_8_16(self, tmp_path: Path, capsys, monkeypatch) -> None:
        packed = self._build_packed_dir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["splice_preselection_locality.py", str(packed)])
        main()
        out = capsys.readouterr().out
        for cap in (1, 2, 4, 5, 8, 16):
            assert f"capacity {cap:3d}:" in out

    def test_json_out_matches_analyze(self, tmp_path: Path, monkeypatch) -> None:
        packed = self._build_packed_dir(tmp_path)
        json_out = tmp_path / "result.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "splice_preselection_locality.py",
                str(packed),
                "--capacities",
                "1,2",
                "--json-out",
                str(json_out),
            ],
        )
        main()
        on_disk = json.loads(json_out.read_text())
        # JSON round-tripping stringifies the loads_by_capacity int keys, so compare
        # against analyze()'s own output pushed through the identical round-trip.
        expected = json.loads(json.dumps(analyze(packed, [1, 2])))
        assert on_disk == expected

    def test_raises_when_no_packed_row_manifest(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["splice_preselection_locality.py", str(tmp_path)])
        with pytest.raises(SystemExit, match="no packed-row manifest"):
            main()


@pytest.mark.slow
@pytest.mark.requires_feature_data
class TestRealCorpusRegression:
    """Pins the exact counts this analyzer was verified against while writing the
    issue #740 plan. If this fails, the analyzer no longer reconstructs the SAME
    group order the production selection code uses -- a correctness regression, not
    corpus drift, since the packed corpus + provenance are immutable once written."""

    def _skip_unless_prod_corpus(self) -> Path:
        if not (_PROD_PACKED_ROW_DIR / "manifest.json").is_file():
            pytest.skip(f"production packed-row corpus missing: {_PROD_PACKED_ROW_DIR}")
        return _PROD_PACKED_ROW_DIR

    def test_matches_issue_740_plan_measurement(self) -> None:
        packed_dir = self._skip_unless_prod_corpus()
        result = analyze(packed_dir, capacities=[1, 5])
        assert result["n_packed_shards"] == 423
        assert result["unique_source_stems"] == 369
        assert result["loads_by_capacity"][1] == 170280
        assert result["loads_by_capacity"][5] == 1586
