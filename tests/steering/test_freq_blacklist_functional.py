"""Functional tests: real feature_statistics.safetensors on disk + script."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from errors import FeatureMaskError
from scripts.preprocess import precompute_feature_blacklist as producer
from scripts.preprocess.precompute_feature_blacklist import (
    _resolve_feature_dir,
)
from scripts.preprocess.precompute_feature_blacklist import (
    run as script_run,
)
from steering.conditional import freq_blacklist as fb
from tests.steering._freq_blacklist_fixtures import write_stats


def _expected_deterministic_meta(result: fb.BlacklistResult) -> dict[str, object]:
    return {
        "generator_version": fb.GENERATOR_VERSION,
        "source": "feature_statistics.safetensors",
        "freq_min": "null" if result.freq_min is None else repr(result.freq_min),
        "freq_max": "null" if result.freq_max is None else repr(result.freq_max),
        "n_sae_features": result.n_sae_features,
        "n_segments": result.n_segments,
        "n_active_features": result.n_active_features,
        "stats_generated_at": result.stats_generated_at,
        "n_blacklisted": len(result.ids),
        "empty": result.is_empty,
    }


def _assert_exact_typed_meta(
    actual: dict[str, object], expected: dict[str, object]
) -> None:
    for key, expected_value in expected.items():
        assert key in actual
        assert type(actual[key]) is type(expected_value)
        assert actual[key] == expected_value


def test_standalone_auto_resolver_targets_no_bos_leaf(tmp_path):
    with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
        cfg = compose(
            config_name="precompute_feature_blacklist",
            overrides=[
                f"feature_root={tmp_path}",
                "dataset_phase=base",
            ],
        )

    resolved = Path(_resolve_feature_dir(cfg))
    assert resolved.name == "no_bos"
    assert resolved.parent.name == "base"


def test_generate_then_no_rewrite_when_unchanged(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    out = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert out is not None
    txt, res = out
    assert res.ids == (1,)  # 8/10=0.8 > 0.4
    expected_txt, meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.4)
    assert txt == expected_txt
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (txt, meta)
    }

    # second call: same stats+threshold -> identical ids -> NO rewrite
    real_write = fb.write_blacklist
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real_write(*a, **k)

    monkeypatch.setattr(fb, "write_blacklist", counting)
    out2 = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert calls["n"] == 0  # unchanged -> not rewritten
    assert out2 is not None and out2[1].ids == (1,)
    for path in (txt, meta):
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before[path]


def test_write_blacklist_fixed_clock_bytes_match_existing_wire_format(
    tmp_path, monkeypatch
):
    class FixedDatetime:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return datetime(2026, 5, 21, 12, 34, 56, tzinfo=UTC)

    result = fb.BlacklistResult(
        ids=(7, 1),
        n_sae_features=16,
        n_segments=10,
        n_active_features=2,
        freq_min=None,
        freq_max=0.4,
        stats_generated_at="stats-generation",
    )
    txt_path, meta_path = fb.blacklist_paths(
        tmp_path, freq_min=None, freq_max=0.4
    )
    monkeypatch.setattr(fb, "datetime", FixedDatetime)

    fb.write_blacklist(txt_path, meta_path, result)

    assert txt_path.read_bytes() == (
        b"# activation-frequency blacklist (generator v1, source: "
        b"feature_statistics.safetensors)\n"
        b"# freq_min: null  (blacklist iff count <= freq_min * n_segments)\n"
        b"# freq_max: 0.4  (blacklist iff count >  freq_max * n_segments)\n"
        b"# n_sae_features: 16\n"
        b"# n_segments: 10\n"
        b"# n_active_features: 2\n"
        b"# stats_generated_at: stats-generation\n"
        b"# n_blacklisted: 2\n"
        b"1\n"
        b"7\n"
    )
    assert meta_path.read_bytes() == (
        b'{\n'
        b'  "empty": false,\n'
        b'  "freq_max": "0.4",\n'
        b'  "freq_min": "null",\n'
        b'  "generated_at": "2026-05-21T12:34:56+00:00",\n'
        b'  "generator_version": 1,\n'
        b'  "n_active_features": 2,\n'
        b'  "n_blacklisted": 2,\n'
        b'  "n_sae_features": 16,\n'
        b'  "n_segments": 10,\n'
        b'  "source": "feature_statistics.safetensors",\n'
        b'  "stats_generated_at": "stats-generation"\n'
        b"}"
    )


def test_invalid_meta_json_self_heals_and_returns_current_provenance(tmp_path):
    write_stats(
        tmp_path,
        n_segments=10,
        counts={1: 8, 2: 1},
        n_sae_features=16,
        generated_at="2026-05-20T00:00:00+00:00",
    )
    first = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )
    assert first is not None
    txt_path, meta_path = fb.blacklist_paths(
        tmp_path, freq_min=None, freq_max=0.4
    )
    intended_txt = txt_path.read_bytes()
    meta_path.write_text("{not json")

    repaired = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )

    assert repaired is not None
    _assert_exact_typed_meta(
        json.loads(meta_path.read_text()),
        _expected_deterministic_meta(repaired[1]),
    )
    assert repaired[1].stats_generated_at == "2026-05-20T00:00:00+00:00"
    assert txt_path.read_bytes() == intended_txt


def test_torn_meta_write_is_repaired_from_unchanged_generation_2_stats(
    tmp_path, monkeypatch
):
    write_stats(
        tmp_path,
        n_segments=10,
        counts={1: 8},
        n_sae_features=16,
        generated_at="generation-1",
    )
    first = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )
    assert first is not None
    txt_path, meta_path = fb.blacklist_paths(
        tmp_path, freq_min=None, freq_max=0.4
    )

    write_stats(
        tmp_path,
        n_segments=20,
        counts={1: 16, 2: 1},
        n_sae_features=16,
        generated_at="generation-2",
    )
    real_atomic_write = fb.atomic_write_text

    def tear_after_txt(content, path):
        if Path(path) == meta_path:
            raise OSError("simulated generation-2 meta write failure")
        return real_atomic_write(content, path)

    with monkeypatch.context() as patch:
        patch.setattr(fb, "atomic_write_text", tear_after_txt)
        with pytest.raises(OSError, match="generation-2 meta write failure"):
            fb.generate_or_load_blacklist(
                tmp_path, 16, freq_min=None, freq_max=0.4
            )

    assert "# n_segments: 20\n" in txt_path.read_text()
    assert json.loads(meta_path.read_text())["stats_generated_at"] == "generation-1"

    repaired = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )
    assert repaired is not None
    repaired_meta = json.loads(meta_path.read_text())
    _assert_exact_typed_meta(
        repaired_meta, _expected_deterministic_meta(repaired[1])
    )
    assert repaired[1].n_segments == 20
    assert repaired[1].stats_generated_at == "generation-2"


@pytest.mark.parametrize(
    ("field", "corrupt_value"),
    [
        ("generator_version", True),
        ("source", ["feature_statistics.safetensors"]),
        ("freq_min", None),
        ("freq_max", 0.4),
        ("n_sae_features", "16"),
        ("n_segments", "10"),
        ("n_active_features", "2"),
        ("stats_generated_at", ["2026-05-19T00:00:00+00:00"]),
        ("n_blacklisted", True),
        ("empty", 0),
        ("generated_at", 123),
    ],
)
def test_each_meta_field_requires_exact_type_and_value(
    tmp_path, monkeypatch, field, corrupt_value
):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    first = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )
    assert first is not None
    txt_path, meta_path = fb.blacklist_paths(
        tmp_path, freq_min=None, freq_max=0.4
    )
    intended_txt = txt_path.read_bytes()
    corrupt_meta = json.loads(meta_path.read_text())
    corrupt_meta[field] = corrupt_value
    meta_path.write_text(json.dumps(corrupt_meta, sort_keys=True, indent=2))

    real_write = fb.write_blacklist
    calls = 0

    def counting_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(fb, "write_blacklist", counting_write)
    repaired = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )

    assert repaired is not None
    assert calls == 1
    repaired_meta = json.loads(meta_path.read_text())
    _assert_exact_typed_meta(
        repaired_meta, _expected_deterministic_meta(repaired[1])
    )
    assert type(repaired_meta["generated_at"]) is str
    assert txt_path.read_bytes() == intended_txt


def test_missing_deterministic_meta_field_triggers_one_repair(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    first = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )
    assert first is not None
    _txt_path, meta_path = fb.blacklist_paths(
        tmp_path, freq_min=None, freq_max=0.4
    )
    corrupt_meta = json.loads(meta_path.read_text())
    del corrupt_meta["n_segments"]
    meta_path.write_text(json.dumps(corrupt_meta, sort_keys=True, indent=2))
    real_write = fb.write_blacklist
    calls = 0

    def counting_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(fb, "write_blacklist", counting_write)
    repaired = fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4
    )

    assert repaired is not None
    assert calls == 1
    _assert_exact_typed_meta(
        json.loads(meta_path.read_text()),
        _expected_deterministic_meta(repaired[1]),
    )


def test_ddp_rank1_corrupt_meta_names_file_and_chains_parse_cause(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=0
    )
    _txt_path, meta_path = fb.blacklist_paths(
        tmp_path, freq_min=None, freq_max=0.4
    )
    meta_path.write_text("{not json")

    with pytest.raises(FeatureMaskError) as exc_info:
        fb.generate_or_load_blacklist(
            tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=1
        )

    assert str(meta_path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_force_rewrites(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    calls = {"n": 0}
    real_write = fb.write_blacklist

    def counting(*a, **k):
        calls["n"] += 1
        return real_write(*a, **k)

    monkeypatch.setattr(fb, "write_blacklist", counting)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4, force=True)
    assert calls["n"] == 1


def test_stale_stats_auto_regenerates(tmp_path):
    """If the statistics are recomputed (different counts), the next call
    auto-corrects the on-disk blacklist (ids differ -> rewrite)."""
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    out1 = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert out1[1].ids == (1,)  # ty: ignore[not-subscriptable]
    # recompute stats: now feature 2 also fires a lot
    write_stats(
        tmp_path, n_segments=10, counts={1: 8, 2: 9}, n_sae_features=16,
        generated_at="2026-05-20T00:00:00+00:00",
    )
    out2 = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert out2[1].ids == (1, 2)  # auto-corrected  # ty: ignore[not-subscriptable]
    assert out2[1].stats_generated_at == "2026-05-20T00:00:00+00:00"  # ty: ignore[not-subscriptable]


def test_empty_returns_none_but_writes_artifact(tmp_path):
    write_stats(
        tmp_path, n_segments=100, counts={1: 1, 2: 2}, n_sae_features=16
    )
    assert fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.9) is None
    _txt, meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.9)
    assert meta.is_file() and '"empty": true' in meta.read_text()
    # second call still None, no error
    assert fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.9) is None


def test_distinct_threshold_distinct_files(tmp_path):
    write_stats(
        tmp_path, n_segments=10, counts={1: 8, 2: 5}, n_sae_features=16
    )
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.6)
    files = sorted(p.name for p in (tmp_path / "blacklist").glob("*.txt"))
    assert files == ["freq_minnull_max0.4.txt", "freq_minnull_max0.6.txt"]


def test_ddp_nonzero_rank_waits_and_reads(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    state = {"barriers": 0}

    def barrier():
        state["barriers"] += 1

    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=0, barrier_fn=barrier)

    def _explode(*a, **k):
        raise AssertionError("rank>0 must not compute")

    monkeypatch.setattr(
        fb, "compute_activation_frequency_blacklist", _explode
    )
    out = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=1, barrier_fn=barrier
    )
    assert out is not None and out[1].ids == (1,)
    assert state["barriers"] >= 2


def test_ddp_missing_artifact_raises(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    with pytest.raises(FeatureMaskError, match="artifact missing"):
        fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=1, barrier_fn=None
        )


def test_ddp_rank0_error_releases_barrier_then_reraises(tmp_path, monkeypatch):
    """M1: rank-0 compute raising must NOT strand other ranks. Rank 0 still
    calls the barrier, then re-raises the REAL error (no NCCL-timeout hang)."""
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    state = {"barriers": 0}

    def barrier():
        state["barriers"] += 1

    def _boom(*a, **k):
        raise RuntimeError("stats read exploded")

    monkeypatch.setattr(fb, "compute_activation_frequency_blacklist", _boom)
    with pytest.raises(RuntimeError, match="stats read exploded"):
        fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=0, barrier_fn=barrier
        )
    assert state["barriers"] == 1  # barrier reached despite the failure


def test_ddp_rank0_missing_stats_does_not_hang(tmp_path):
    """M1: rank-0 FileNotFound-equivalent (no stats file) -> real error
    surfaces (caught + re-raised after the barrier), not a collective hang."""
    state = {"barriers": 0}

    def barrier():
        state["barriers"] += 1

    with pytest.raises(FeatureMaskError, match="feature_statistics"):
        fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=0, barrier_fn=barrier
        )
    assert state["barriers"] == 1  # barrier reached before the re-raise


def _cfg_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs" / "extraction")


def test_standalone_script(tmp_path):
    write_stats(
        tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16384
    )
    with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
        cfg = compose(
            config_name="precompute_feature_blacklist",
            overrides=[
                f"feature_dir={tmp_path}",
                "feature_mask_freq_min=null",
                "feature_mask_freq_max=0.4",
                "n_sae_features=16384",
            ],
        )
    path = script_run(cfg)
    assert path is not None and Path(path).is_file()
    assert script_run(cfg) == path  # idempotent rerun


def test_standalone_script_reads_n_sae_from_stats(tmp_path):
    # no n_sae_features override -> read from the stats file (metadata=64)
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=64)
    with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
        cfg = compose(
            config_name="precompute_feature_blacklist",
            overrides=[
                f"feature_dir={tmp_path}",
                "feature_mask_freq_min=null",
                "feature_mask_freq_max=0.4",
            ],
        )
    path = script_run(cfg)
    assert path is not None and Path(path).is_file()


def test_standalone_script_disabled_via_null_bounds(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16384)
    with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
        cfg = compose(
            config_name="precompute_feature_blacklist",
            overrides=[
                f"feature_dir={tmp_path}",
                "feature_mask_freq_min=null",
                "feature_mask_freq_max=null",
            ],
        )
    assert script_run(cfg) is None


def _snapshot_script_cfg(tmp_path, *overrides):
    with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
        return compose(config_name="precompute_feature_blacklist", overrides=[
            f"feature_dir={tmp_path}", "feature_mask_freq_min=null",
            "feature_mask_freq_max=0.4", *overrides])


def _count_snapshot_loads(monkeypatch):
    original = fb.load_statistics
    calls: list[str] = []
    def counted(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(producer, "load_statistics", counted)
    monkeypatch.setattr(fb, "load_statistics", counted)
    return calls


@pytest.mark.parametrize("force", [False, True])
def test_cli_snapshot_loads_once_even_for_existing_artifact(tmp_path, monkeypatch, force):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    cfg = _snapshot_script_cfg(tmp_path, f"force={str(force).lower()}")
    loads = _count_snapshot_loads(monkeypatch)
    original_write = fb.write_blacklist
    writes: list[int] = []
    def counted_write(txt_path, meta_path, result):
        writes.append(1)
        return original_write(txt_path, meta_path, result)
    monkeypatch.setattr(fb, "write_blacklist", counted_write)
    path = script_run(cfg)
    assert path is not None
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(path).parent.iterdir()}
    assert script_run(cfg) == path
    assert len(loads) == 2
    assert len(writes) == (2 if force else 1)
    assert fb._load_blacklist(*fb.blacklist_paths(tmp_path, None, 0.4)).ids == (1,)
    if not force:
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before


def test_cli_snapshot_rereads_changed_statistics_next_invocation(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    cfg = _snapshot_script_cfg(tmp_path)
    loads = _count_snapshot_loads(monkeypatch)
    script_run(cfg)
    assert fb._load_blacklist(*fb.blacklist_paths(tmp_path, None, 0.4)).ids == (1,)
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 9}, n_sae_features=16,
                generated_at="2026-05-20T00:00:00+00:00")
    script_run(cfg)
    result = fb._load_blacklist(*fb.blacklist_paths(tmp_path, None, 0.4))
    assert result.ids == (1, 2)
    assert result.stats_generated_at == "2026-05-20T00:00:00+00:00"
    assert len(loads) == 2


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_snapshot_explicit_width_and_disabled_keep_old_loads(tmp_path, monkeypatch, disabled):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    cfg = _snapshot_script_cfg(tmp_path, "n_sae_features=16")
    if disabled:
        cfg.feature_mask_freq_max = None
    loads = _count_snapshot_loads(monkeypatch)
    result = script_run(cfg)
    assert len(loads) == (0 if disabled else 1)
    assert (result is None) is disabled
    assert (tmp_path / "blacklist").exists() is not disabled


def test_cli_snapshot_still_parses_unused_malformed_field_before_info(tmp_path, monkeypatch):
    path = write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    tensors = load_file(str(path))
    with safe_open(path, framework="pt") as stream:
        metadata = stream.metadata()
    assert metadata is not None
    metadata["extrap_target_tokens"] = "invalid-int"
    save_file(tensors, str(path), metadata=metadata)
    messages: list[str] = []
    monkeypatch.setattr(producer, "logger", SimpleNamespace(info=messages.append))
    with pytest.raises(ValueError, match="invalid-int"):
        script_run(_snapshot_script_cfg(tmp_path))
    assert messages == []
    assert not (tmp_path / "blacklist").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_snapshot_missing_statistics_preserves_fallback_error(tmp_path, explicit):
    cfg = _snapshot_script_cfg(tmp_path)
    if explicit:
        cfg.n_sae_features = 16
    else:
        cfg.feature_dir = None
        cfg.feature_root = str(tmp_path)
    with pytest.raises(FeatureMaskError, match="feature_statistics.safetensors not found"):
        script_run(cfg)


def test_cli_snapshot_uses_first_graph_if_logger_replaces_statistics(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    def info(message):
        if message.startswith("Generating freq-blacklist"):
            write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 9}, n_sae_features=16)
    monkeypatch.setattr(producer, "logger", SimpleNamespace(info=info))
    cfg = _snapshot_script_cfg(tmp_path)
    script_run(cfg)
    assert fb._load_blacklist(*fb.blacklist_paths(tmp_path, None, 0.4)).ids == (1,)
    script_run(cfg)
    assert fb._load_blacklist(*fb.blacklist_paths(tmp_path, None, 0.4)).ids == (1, 2)


def test_generic_generator_keeps_exact_public_compute_keywords(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    original = fb.compute_activation_frequency_blacklist
    calls: list[int] = []
    def strict(feature_dir, n_sae_features, *, freq_min, freq_max):
        calls.append(1)
        return original(feature_dir, n_sae_features, freq_min=freq_min, freq_max=freq_max)
    monkeypatch.setattr(fb, "compute_activation_frequency_blacklist", strict)
    assert fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4) is not None
    assert calls == [1]


def test_snapshot_range_error_reaches_barrier_before_reraise(tmp_path):
    path = write_stats(tmp_path, n_segments=10, counts={17: 8}, n_sae_features=16)
    snapshot = fb.load_statistics(path)
    barriers: list[int] = []
    with pytest.raises(FeatureMaskError, match="out of range"):
        fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4,
            _statistics=snapshot, barrier_fn=lambda: barriers.append(1))
    assert barriers == [1]
    assert not (tmp_path / "blacklist").exists()
