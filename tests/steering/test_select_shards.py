"""Tests for select_shard_paths — shard selection with start_shard offset.

Run with:
    uv run pytest tests/test_select_shards.py -v
"""

from pathlib import Path
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

import steering.extract_features as extract_module
from steering.feature_io import feature_path_exists
from steering.parquet_io import select_shard_paths as _select_shard_paths
from steering.refined_types import NonNegativeInt, PositiveInt


def _paths(n: int) -> list[str]:
    """Generate n fake parquet paths."""
    return [f"/data/shard_{i:05d}.parquet" for i in range(n)]


def select_shard_paths(
    parquet_paths: list[str],
    num_shards: int,
    start_shard: int,
    rank: int,
    world_size: int,
) -> list[str]:
    """Test adapter that makes each successful refinement explicit."""
    return _select_shard_paths(
        parquet_paths,
        NonNegativeInt.parse(num_shards),
        NonNegativeInt.parse(start_shard),
        NonNegativeInt.parse(rank),
        PositiveInt.parse(world_size),
    )


def _call_with_untyped_scalars(**updates: Any) -> list[str]:
    """Bypass static refinement only for negative runtime-boundary tests."""
    kwargs: dict[str, Any] = {
        "parquet_paths": _paths(5),
        "num_shards": 5,
        "start_shard": 0,
        "rank": 0,
        "world_size": 1,
    }
    kwargs.update(updates)
    return _select_shard_paths(**kwargs)


class TestSelectShardPathsBasic:
    """Core behavior of select_shard_paths."""

    def test_default_start_shard_zero(self):
        """start_shard=0 should return all shards up to num_shards."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=5, start_shard=0, rank=0, world_size=1
        )
        assert result == paths[:5]

    def test_start_shard_slices_from_offset(self):
        """start_shard=3, num_shards=8 should return shards 3-7."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=8, start_shard=3, rank=0, world_size=1
        )
        assert result == paths[3:8]

    def test_num_shards_caps_at_available(self):
        """num_shards exceeding available paths should cap at len(paths)."""
        paths = _paths(5)
        result = select_shard_paths(
            paths, num_shards=100, start_shard=0, rank=0, world_size=1
        )
        assert result == paths

    def test_start_shard_with_num_shards_cap(self):
        """start_shard=2, num_shards=100 with 5 paths -> shards 2-4."""
        paths = _paths(5)
        result = select_shard_paths(
            paths, num_shards=100, start_shard=2, rank=0, world_size=1
        )
        assert result == paths[2:]

    def test_start_shard_equals_num_shards_returns_empty(self):
        """start_shard == num_shards -> no shards to process."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=5, start_shard=5, rank=0, world_size=1
        )
        assert result == []

    def test_start_shard_exceeds_num_shards_returns_empty(self):
        """start_shard > num_shards -> no shards to process."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=5, start_shard=7, rank=0, world_size=1
        )
        assert result == []

    def test_start_shard_exceeds_available_paths_returns_empty(self):
        """start_shard beyond all available paths -> empty."""
        paths = _paths(5)
        result = select_shard_paths(
            paths, num_shards=100, start_shard=10, rank=0, world_size=1
        )
        assert result == []

    def test_single_shard_at_end(self):
        """start_shard=num_shards-1 should return exactly one shard."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=5, start_shard=4, rank=0, world_size=1
        )
        assert result == [paths[4]]

    def test_empty_input_paths(self):
        """Empty parquet_paths should always return empty."""
        result = select_shard_paths(
            [], num_shards=10, start_shard=0, rank=0, world_size=1
        )
        assert result == []


class TestSelectShardPathsDDP:
    """Round-robin DDP distribution with start_shard."""

    def test_round_robin_rank0_of_2(self):
        """Rank 0 of 2 should get every other shard from the selected range."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=10, start_shard=0, rank=0, world_size=2
        )
        assert result == [paths[i] for i in range(0, 10, 2)]

    def test_round_robin_rank1_of_2(self):
        """Rank 1 of 2 should get the other shards."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=10, start_shard=0, rank=1, world_size=2
        )
        assert result == [paths[i] for i in range(1, 10, 2)]

    def test_round_robin_with_start_shard(self):
        """start_shard=4 with 2 ranks: range is [4..9], then round-robin."""
        paths = _paths(10)
        selected = paths[4:10]  # shards 4,5,6,7,8,9
        r0 = select_shard_paths(
            paths, num_shards=10, start_shard=4, rank=0, world_size=2
        )
        r1 = select_shard_paths(
            paths, num_shards=10, start_shard=4, rank=1, world_size=2
        )
        assert r0 == [selected[i] for i in range(0, 6, 2)]  # shards 4,6,8
        assert r1 == [selected[i] for i in range(1, 6, 2)]  # shards 5,7,9

    def test_all_ranks_cover_full_range(self):
        """Union of all ranks should equal the full selected range."""
        paths = _paths(20)
        world_size = 4
        all_assigned = []
        for rank in range(world_size):
            assigned = select_shard_paths(
                paths, num_shards=15, start_shard=5, rank=rank, world_size=world_size
            )
            all_assigned.extend(assigned)
        expected = paths[5:15]
        assert sorted(all_assigned) == sorted(expected)

    def test_no_duplicates_across_ranks(self):
        """No shard should be assigned to more than one rank."""
        paths = _paths(20)
        world_size = 3
        all_assigned = []
        for rank in range(world_size):
            assigned = select_shard_paths(
                paths, num_shards=20, start_shard=2, rank=rank, world_size=world_size
            )
            all_assigned.extend(assigned)
        assert len(all_assigned) == len(set(all_assigned))

    def test_single_rank_gets_all(self):
        """world_size=1 should give everything to rank 0."""
        paths = _paths(10)
        result = select_shard_paths(
            paths, num_shards=10, start_shard=3, rank=0, world_size=1
        )
        assert result == paths[3:10]

    def test_more_ranks_than_shards(self):
        """When world_size > num selected shards, some ranks get nothing."""
        paths = _paths(10)
        results = [
            select_shard_paths(
                paths, num_shards=10, start_shard=8, rank=r, world_size=4
            )
            for r in range(4)
        ]
        assert results[0] == [paths[8]]
        assert results[1] == [paths[9]]
        assert results[2] == []
        assert results[3] == []


class TestSelectShardPathsValidation:
    """Edge cases and validation."""

    def test_negative_start_shard_raises(self):
        """start_shard < 0 should fail at the refined boundary."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=r"parameter start_shard=.*NonNegativeInt",
        ):
            _call_with_untyped_scalars(start_shard=-1)

    def test_negative_num_shards_raises(self):
        """num_shards < 0 should fail at the refined boundary."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=r"parameter num_shards=.*NonNegativeInt",
        ):
            _call_with_untyped_scalars(num_shards=-1)

    def test_zero_world_size_raises(self):
        """world_size < 1 should fail at the refined boundary."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=r"parameter world_size=.*PositiveInt",
        ):
            _call_with_untyped_scalars(world_size=0)

    def test_negative_rank_raises(self):
        """rank < 0 should fail at the refined boundary."""
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=r"parameter rank=.*NonNegativeInt",
        ):
            _call_with_untyped_scalars(rank=-1, world_size=2)

    @pytest.mark.parametrize(
        ("name", "value", "type_name"),
        [
            ("num_shards", True, "NonNegativeInt"),
            ("start_shard", False, "NonNegativeInt"),
            ("rank", 1.0, "NonNegativeInt"),
            ("world_size", True, "PositiveInt"),
        ],
    )
    def test_non_strict_integer_scalars_fail_at_refined_boundary(
        self, name: str, value: object, type_name: str
    ) -> None:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=rf"parameter {name}=.*{type_name}",
        ):
            _call_with_untyped_scalars(**{name: value})

    def test_rank_equals_world_size_raises(self):
        """rank == world_size should raise ValueError."""
        with pytest.raises(ValueError, match="rank must be in"):
            select_shard_paths(
                _paths(5), num_shards=5, start_shard=0, rank=2, world_size=2
            )

    def test_backward_compatible_with_original_logic(self):
        """start_shard=0 should produce identical results to the old code path.

        Old logic: paths[:num_shards][rank::world_size]
        New logic: select_shard_paths(paths, num_shards, start_shard=0, rank, world_size)
        """
        paths = _paths(20)
        for num_shards in [1, 5, 15, 20, 100]:
            for world_size in [1, 2, 4]:
                for rank in range(world_size):
                    end = min(num_shards, len(paths))
                    old = paths[:end][rank::world_size]
                    new = select_shard_paths(
                        paths,
                        num_shards,
                        start_shard=0,
                        rank=rank,
                        world_size=world_size,
                    )
                    assert new == old, (
                        f"Mismatch: num_shards={num_shards}, rank={rank}, ws={world_size}"
                    )


def test_extraction_entrypoint_refines_empty_shard_window_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke the real Hydra entrypoint through selection without loading weights."""
    paths = _paths(3)
    seen: dict[str, Any] = {}
    real_select = extract_module.select_shard_paths

    def tracked_select(*args: Any, **kwargs: Any) -> list[str]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return real_select(*args, **kwargs)

    def forbidden_model_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("empty shard window must return before model loading")

    monkeypatch.setattr(
        extract_module,
        "_assert_context_window_within_sae_max",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(extract_module, "resolve_data_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(extract_module, "list_parquet_files", lambda *_args: paths)
    monkeypatch.setattr(extract_module, "select_shard_paths", tracked_select)
    monkeypatch.setattr(extract_module, "load_sae_and_model", forbidden_model_load)

    output_dir = tmp_path / "features"
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "model": "gemma-2-2b",
            "layer": 1,
            "context_window": 8,
            "dataset_phase": None,
            "context_independent_features": False,
            "data_dir": str(tmp_path),
            "output_dir": str(output_dir),
            "activation_type": "residual",
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 1,
            "gc_mem_threshold": 0.9,
            "gpu_batch_size": 1,
            "cpu_batch_size": 1,
            "zstd_level": 3,
            "num_shards": 2,
            "start_shard": 2,
            "staging": {
                "scratch_dir": str(tmp_path / "scratch"),
                "output_depth": 0,
                "input_depth": 0,
                "min_free_gb": 0.0,
                "est_shard_gb": 0.001,
            },
        }
    )

    assert extract_module.main.__wrapped__(cfg) is None
    assert output_dir.is_dir()
    assert seen["args"][1:] == (2, 2)
    assert seen["kwargs"] == {"rank": 0, "world_size": 1}


class TestStartShardWithResume:
    """Integration: select_shard_paths + feature_path_exists resume logic."""

    def _simulate_extraction_loop(
        self,
        parquet_paths: list[str],
        output_dir: str,
        num_shards: int,
        start_shard: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[list[str], int]:
        """Simulate the main() shard loop with select + resume.

        Returns (processed_shard_names, n_skipped).
        """
        selected = select_shard_paths(
            parquet_paths, num_shards, start_shard, rank, world_size
        )
        processed = []
        n_skipped = 0
        for path in selected:
            shard_name = Path(path).stem
            if feature_path_exists(output_dir, shard_name):
                n_skipped += 1
                continue
            processed.append(shard_name)
        return processed, n_skipped

    def test_start_shard_skips_existing_in_range(self, tmp_path):
        """start_shard=105, shards 100-120 exist -> skip 105-120, process from 121."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = _paths(300)
        for i in range(100, 121):
            (output_dir / f"shard_{i:05d}_features.safetensors.zst").write_bytes(b"x")

        processed, n_skipped = self._simulate_extraction_loop(
            paths,
            str(output_dir),
            num_shards=300,
            start_shard=105,
        )
        assert n_skipped == 16  # shards 105-120
        assert processed[0] == "shard_00121"
        assert len(processed) == 300 - 121  # 121..299

    def test_start_shard_zero_with_existing_shards(self, tmp_path):
        """Default start_shard=0 with existing shards 100-120."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = _paths(300)
        for i in range(100, 121):
            (output_dir / f"shard_{i:05d}_features.safetensors.zst").write_bytes(b"x")

        processed, n_skipped = self._simulate_extraction_loop(
            paths,
            str(output_dir),
            num_shards=300,
            start_shard=0,
        )
        assert n_skipped == 21  # shards 100-120
        assert len(processed) == 300 - 21
        assert "shard_00000" in processed
        assert "shard_00099" in processed
        assert "shard_00100" not in processed
        assert "shard_00121" in processed

    def test_start_shard_beyond_all_existing(self, tmp_path):
        """start_shard=200, existing shards 100-120 -> no skips, process 200+."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = _paths(300)
        for i in range(100, 121):
            (output_dir / f"shard_{i:05d}_features.safetensors.zst").write_bytes(b"x")

        processed, n_skipped = self._simulate_extraction_loop(
            paths,
            str(output_dir),
            num_shards=300,
            start_shard=200,
        )
        assert n_skipped == 0
        assert len(processed) == 100  # 200..299
        assert processed[0] == "shard_00200"

    def test_start_shard_with_ddp_and_resume(self, tmp_path):
        """2 ranks, start_shard=4, shards 4-5 exist -> both ranks skip correctly."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = _paths(10)
        (output_dir / "shard_00004_features.safetensors.zst").write_bytes(b"x")
        (output_dir / "shard_00005_features.safetensors.zst").write_bytes(b"x")

        # Range [4,10) = shards 4,5,6,7,8,9
        # Rank 0 gets: 4,6,8 — skip 4 -> process 6,8
        proc0, skip0 = self._simulate_extraction_loop(
            paths,
            str(output_dir),
            num_shards=10,
            start_shard=4,
            rank=0,
            world_size=2,
        )
        assert skip0 == 1
        assert proc0 == ["shard_00006", "shard_00008"]

        # Rank 1 gets: 5,7,9 — skip 5 -> process 7,9
        proc1, skip1 = self._simulate_extraction_loop(
            paths,
            str(output_dir),
            num_shards=10,
            start_shard=4,
            rank=1,
            world_size=2,
        )
        assert skip1 == 1
        assert proc1 == ["shard_00007", "shard_00009"]

    def test_all_in_range_already_extracted(self, tmp_path):
        """Every shard in [start_shard, num_shards) exists -> nothing to process."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = _paths(10)
        for i in range(5, 10):
            (output_dir / f"shard_{i:05d}_features.safetensors.zst").write_bytes(b"x")

        processed, n_skipped = self._simulate_extraction_loop(
            paths,
            str(output_dir),
            num_shards=10,
            start_shard=5,
        )
        assert n_skipped == 5
        assert processed == []


class TestSelectShardPathsByIndex:
    """#8: selection is by parsed shard INDEX (not list position), so gaps in the
    shard numbering are handled correctly; non-shard filenames fall back to the
    historical position-based window."""

    def test_gapped_dir_selects_by_index_not_position(self):
        # shards 0, 5, 10 present (gaps). num_shards=10 -> indices in [0,10) = {0,5}.
        # Position-based slicing would have returned all 3 (paths[0:10]) — the bug.
        paths = [f"/d/shard_{i:05d}.parquet" for i in (0, 5, 10)]
        result = select_shard_paths(paths, num_shards=10, start_shard=0, rank=0, world_size=1)
        assert result == ["/d/shard_00000.parquet", "/d/shard_00005.parquet"]

    def test_gapped_dir_start_shard_by_index(self):
        paths = [f"/d/shard_{i:05d}.parquet" for i in (0, 5, 10)]
        # indices in [5, 11) = {5, 10}
        result = select_shard_paths(paths, num_shards=11, start_shard=5, rank=0, world_size=1)
        assert result == ["/d/shard_00005.parquet", "/d/shard_00010.parquet"]

    def test_gapped_dir_ddp_partition_disjoint_exhaustive(self):
        paths = [f"/d/shard_{i:05d}.parquet" for i in (0, 3, 7, 12, 20)]
        r0 = select_shard_paths(paths, num_shards=100, start_shard=0, rank=0, world_size=2)
        r1 = select_shard_paths(paths, num_shards=100, start_shard=0, rank=1, world_size=2)
        assert sorted(r0 + r1) == sorted(paths)       # exhaustive
        assert not (set(r0) & set(r1))                 # disjoint

    def test_unsorted_input_ordered_by_index(self):
        # Non-zero-padded names sort alphabetically as shard_1, shard_10, shard_2;
        # index selection orders by NUMBER instead.
        paths = ["/d/shard_2.parquet", "/d/shard_10.parquet", "/d/shard_1.parquet"]
        result = select_shard_paths(paths, num_shards=11, start_shard=0, rank=0, world_size=1)
        assert result == ["/d/shard_1.parquet", "/d/shard_2.parquet", "/d/shard_10.parquet"]

    def test_non_shard_filenames_fall_back_to_position(self):
        paths = ["/d/alpha.parquet", "/d/beta.parquet", "/d/gamma.parquet"]
        result = select_shard_paths(paths, num_shards=2, start_shard=0, rank=0, world_size=1)
        assert result == ["/d/alpha.parquet", "/d/beta.parquet"]   # paths[0:2]

    def test_stray_file_excluded_shards_still_indexed(self):
        # A stray non-shard file does NOT revert the whole selection to position;
        # the shards are index-selected and the stray is excluded.
        paths = ["/d/shard_00000.parquet", "/d/val.parquet", "/d/shard_00002.parquet"]
        result = select_shard_paths(paths, num_shards=2, start_shard=0, rank=0, world_size=1)
        assert result == ["/d/shard_00000.parquet"]   # idx 0 in [0,2); idx 2 out; val excluded

    def test_regex_anchored_no_substring_false_match(self):
        # Names merely CONTAINING shard_N (not equal to shard_<digits>) must not
        # parse as shards; with no real shards, selection falls back to position.
        paths = ["/d/data_shard_3.parquet", "/d/shard_7_backup.parquet"]
        result = select_shard_paths(paths, num_shards=10, start_shard=0, rank=0, world_size=1)
        assert result == paths   # neither is a shard -> position fallback returns the window

    def test_contiguous_dir_byte_identical_to_position(self):
        # Regression guard: for a contiguous shard_NNNNN dir, index-based selection
        # is byte-identical to the OLD position-based slice for every window/stripe.
        paths = [f"/d/shard_{i:05d}.parquet" for i in range(20)]
        for num_shards in (1, 5, 15, 20, 100):
            for start_shard in (0, 3, 19, 25):
                for world_size in (1, 2, 4):
                    for rank in range(world_size):
                        end = min(num_shards, len(paths))
                        old = paths[start_shard:end][rank::world_size]
                        new = select_shard_paths(paths, num_shards, start_shard, rank, world_size)
                        assert new == old, f"ns={num_shards} ss={start_shard} ws={world_size} r={rank}"
