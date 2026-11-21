"""Tests for the completion TUI auto-discovery module."""

import json
import os

from scripts.completion_tui.discovery import (
    _enumerate_model_tags,
    _find_all_checkpoint_leaves,
    _find_last_step_safe,
    _load_meta,
    _nested_path,
    _scan_feature_dirs,
    discover_configs,
    prompt_selection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(d: str, step: int = 100, user_config: dict | None = None) -> None:
    """Create a minimal checkpoint (model_*.pt + meta_*.json) in directory d."""
    os.makedirs(d, exist_ok=True)
    # Dummy model file (empty)
    with open(os.path.join(d, f"model_{step:06d}.pt"), "w") as f:
        f.write("")
    # Meta JSON
    meta = {
        "model_config": {
            "sequence_len": 2048,
            "vocab_size": 32768,
            "n_layer": 26,
            "n_head": 13,
            "n_kv_head": 13,
            "n_embd": 1664,
        },
        "user_config": user_config
        or {
            "num_buckets": 11,
            "segment_size": 128,
        },
    }
    with open(os.path.join(d, f"meta_{step:06d}.json"), "w") as f:
        json.dump(meta, f)


def _make_feature_dir(base: str, seg_size: int = 128, layer: int = 8) -> str:
    """Create a minimal feature directory with a statistics file."""
    path = os.path.join(
        base,
        f"s{seg_size}",
        f"l{layer}",
        "a_res",
        "p_max_pool",
        "f16k",
        "m_gemma-2-2b",
        "base",
    )
    os.makedirs(path, exist_ok=True)
    # Dummy statistics file
    with open(os.path.join(path, "feature_statistics.safetensors"), "w") as f:
        f.write("")
    return path


def _make_parquet_dir(base_dir: str) -> str:
    """Create a base_data dir with a dummy parquet file."""
    data_dir = os.path.join(base_dir, "base_data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "shard_000.parquet"), "w") as f:
        f.write("")
    return data_dir


# ---------------------------------------------------------------------------
# Tests: _enumerate_model_tags
# ---------------------------------------------------------------------------


class TestEnumerateModelTags:
    def test_empty_dir(self, tmp_path):
        d = str(tmp_path / "empty")
        os.makedirs(d)
        assert _enumerate_model_tags(d) == []

    def test_nonexistent_dir(self, tmp_path):
        assert _enumerate_model_tags(str(tmp_path / "nope")) == []

    def test_returns_sorted(self, tmp_path):
        d = str(tmp_path / "ckpts")
        os.makedirs(os.path.join(d, "d27"))
        os.makedirs(os.path.join(d, "d26"))
        os.makedirs(os.path.join(d, "d10"))
        assert _enumerate_model_tags(d) == ["d10", "d26", "d27"]

    def test_skips_files(self, tmp_path):
        d = str(tmp_path / "ckpts")
        os.makedirs(d)
        with open(os.path.join(d, "notes.txt"), "w") as f:
            f.write("")
        os.makedirs(os.path.join(d, "d26"))
        assert _enumerate_model_tags(d) == ["d26"]


# ---------------------------------------------------------------------------
# Tests: _find_all_checkpoint_leaves
# ---------------------------------------------------------------------------


class TestFindAllCheckpointLeaves:
    def test_flat_checkpoint(self, tmp_path):
        d = str(tmp_path / "d26")
        _make_checkpoint(d)
        assert _find_all_checkpoint_leaves(d) == [d]

    def test_nested_checkpoint(self, tmp_path):
        leaf = str(tmp_path / "d26" / "s128" / "t2048.9")
        _make_checkpoint(leaf)
        assert _find_all_checkpoint_leaves(str(tmp_path / "d26")) == [leaf]

    def test_multiple_leaves(self, tmp_path):
        leaf1 = str(tmp_path / "d26" / "s64" / "t1024")
        leaf2 = str(tmp_path / "d26" / "s128" / "t2048")
        _make_checkpoint(leaf1)
        _make_checkpoint(leaf2)
        result = _find_all_checkpoint_leaves(str(tmp_path / "d26"))
        assert sorted(result) == sorted([leaf1, leaf2])

    def test_no_checkpoints(self, tmp_path):
        d = str(tmp_path / "empty")
        os.makedirs(d)
        assert _find_all_checkpoint_leaves(d) == []


# ---------------------------------------------------------------------------
# Tests: _find_last_step
# ---------------------------------------------------------------------------


class TestFindLastStepSafe:
    def test_single_step(self, tmp_path):
        _make_checkpoint(str(tmp_path), step=42)
        assert _find_last_step_safe(str(tmp_path)) == 42

    def test_multiple_steps(self, tmp_path):
        d = str(tmp_path)
        _make_checkpoint(d, step=100)
        _make_checkpoint(d, step=200)
        assert _find_last_step_safe(d) == 200

    def test_no_checkpoints(self, tmp_path):
        os.makedirs(str(tmp_path), exist_ok=True)
        assert _find_last_step_safe(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tests: _load_meta
# ---------------------------------------------------------------------------


class TestLoadMeta:
    def test_valid_meta(self, tmp_path):
        _make_checkpoint(str(tmp_path), step=100, user_config={"num_buckets": 11})
        meta = _load_meta(str(tmp_path), 100)
        assert meta is not None
        assert meta["user_config"]["num_buckets"] == 11

    def test_missing_meta(self, tmp_path):
        assert _load_meta(str(tmp_path), 999) is None

    def test_corrupt_json(self, tmp_path):
        os.makedirs(str(tmp_path), exist_ok=True)
        meta_path = os.path.join(str(tmp_path), "meta_000100.json")
        with open(meta_path, "w") as f:
            f.write("not valid json{{{")
        assert _load_meta(str(tmp_path), 100) is None


# ---------------------------------------------------------------------------
# Tests: _nested_path
# ---------------------------------------------------------------------------


class TestNestedPath:
    def test_same_dir(self):
        assert _nested_path("/a/b", "/a/b") == ""

    def test_nested(self):
        assert _nested_path("/a/b", "/a/b/s128/t2048") == "s128/t2048"


# ---------------------------------------------------------------------------
# Tests: _scan_feature_dirs
# ---------------------------------------------------------------------------


class TestScanFeatureDirs:
    def test_finds_feature_dirs(self, tmp_path):
        feat_root = str(tmp_path / "features")
        path = _make_feature_dir(feat_root, seg_size=128)
        result = _scan_feature_dirs(feat_root)
        assert len(result) == 1
        assert str(result[0]) == path

    def test_empty_root(self, tmp_path):
        assert _scan_feature_dirs(str(tmp_path / "nope")) == []

    def test_multiple_feature_dirs(self, tmp_path):
        feat_root = str(tmp_path / "features")
        _make_feature_dir(feat_root, seg_size=64)
        _make_feature_dir(feat_root, seg_size=128)
        result = _scan_feature_dirs(feat_root)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: discover_configs (integration with mock filesystem)
# ---------------------------------------------------------------------------


class TestDiscoverConfigs:
    def _feat_root(self, tmp_path) -> str:
        return str(tmp_path / "out" / "data" / "features")

    def test_no_checkpoints(self, tmp_path):
        """No checkpoints → empty list."""
        assert discover_configs(str(tmp_path), feature_root=self._feat_root(tmp_path)) == []

    def test_no_matching_features(self, tmp_path):
        """Checkpoint exists but no feature dirs match segment_size."""
        encdec = tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128" / "t2048"
        _make_checkpoint(
            str(encdec),
            step=100,
            user_config={
                "num_buckets": 11,
                "segment_size": 128,
            },
        )
        # Feature dir has segment_size=64, not 128
        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root, seg_size=64)
        _make_parquet_dir(str(tmp_path))

        result = discover_configs(str(tmp_path), feature_root=feat_root)
        assert result == []

    def test_single_match(self, tmp_path, monkeypatch):
        """One checkpoint, one matching feature dir → one config."""
        # Checkpoint
        encdec = tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128" / "t2048"
        _make_checkpoint(
            str(encdec),
            step=150,
            user_config={
                "num_buckets": 11,
                "segment_size": 128,
            },
        )

        # Matching feature dir
        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root, seg_size=128, layer=8)

        # Parquet data
        _make_parquet_dir(str(tmp_path))

        # Patch get_base_dir to return tmp_path so resolve_data_dir finds base_data
        monkeypatch.setattr(
            "steering.feature_paths.get_base_dir",
            lambda: str(tmp_path),
        )

        result = discover_configs(str(tmp_path), feature_root=feat_root)
        assert len(result) == 1
        cfg = result[0]
        assert cfg.model_tag == "d26"
        assert cfg.step == 150
        assert cfg.segment_size == 128
        assert cfg.num_buckets == 11
        assert not cfg.has_negatives

    def test_has_negatives(self, tmp_path, monkeypatch):
        """negative_selection present → has_negatives=True."""
        encdec = tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128" / "t2048"
        _make_checkpoint(
            str(encdec),
            step=100,
            user_config={
                "num_buckets": 11,
                "segment_size": 128,
                "negative_selection": {"mode": "random"},
            },
        )

        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root, seg_size=128)
        _make_parquet_dir(str(tmp_path))

        monkeypatch.setattr(
            "steering.feature_paths.get_base_dir",
            lambda: str(tmp_path),
        )

        result = discover_configs(str(tmp_path), feature_root=feat_root)
        assert len(result) == 1
        assert result[0].has_negatives is True

    def test_missing_num_buckets_skipped(self, tmp_path):
        """Checkpoint without num_buckets in user_config is skipped."""
        encdec = tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128" / "t2048"
        _make_checkpoint(
            str(encdec),
            step=100,
            user_config={
                "segment_size": 128,
                # no num_buckets
            },
        )

        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root, seg_size=128)
        _make_parquet_dir(str(tmp_path))

        assert discover_configs(str(tmp_path), feature_root=self._feat_root(tmp_path)) == []

    def test_sorted_by_tag_then_path(self, tmp_path, monkeypatch):
        """Multiple configs are sorted by model_tag, then nested_path."""
        user_cfg = {"num_buckets": 11, "segment_size": 128}
        for tag in ["d27", "d26"]:
            leaf = tmp_path / "conditional_encdec_checkpoints" / tag / "s128" / "t2048"
            _make_checkpoint(str(leaf), step=100, user_config=user_cfg)

        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root, seg_size=128)
        _make_parquet_dir(str(tmp_path))

        monkeypatch.setattr(
            "steering.feature_paths.get_base_dir",
            lambda: str(tmp_path),
        )

        result = discover_configs(str(tmp_path), feature_root=feat_root)
        assert len(result) == 2
        assert result[0].model_tag == "d26"
        assert result[1].model_tag == "d27"

    def test_cluster_count_is_appended_to_label(self, tmp_path, monkeypatch):
        encdec = tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128"
        _make_checkpoint(str(encdec), step=100)
        meta_path = encdec / "meta_000100.json"
        meta = json.loads(meta_path.read_text())
        meta["feature_clusters_descriptor"] = {"schema_version": 1}
        meta["feature_clusters_canonical"] = {
            "schema_version": 1,
            "clusters": [[1, 2], [3, 4]],
        }
        meta_path.write_text(json.dumps(meta))
        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root)
        _make_parquet_dir(str(tmp_path))
        monkeypatch.setattr("steering.feature_paths.get_base_dir", lambda: str(tmp_path))
        result = discover_configs(str(tmp_path), feature_root=feat_root)
        assert len(result) == 1
        assert ", 2 clusters)" in result[0].label

    def test_partial_cluster_metadata_does_not_break_discovery(self, tmp_path, monkeypatch):
        encdec = tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128"
        _make_checkpoint(str(encdec), step=100)
        meta_path = encdec / "meta_000100.json"
        meta = json.loads(meta_path.read_text())
        meta["feature_clusters_descriptor"] = {"schema_version": 1}
        meta_path.write_text(json.dumps(meta))
        feat_root = str(tmp_path / "out" / "data" / "features")
        _make_feature_dir(feat_root)
        _make_parquet_dir(str(tmp_path))
        monkeypatch.setattr("steering.feature_paths.get_base_dir", lambda: str(tmp_path))
        result = discover_configs(str(tmp_path), feature_root=feat_root)
        assert len(result) == 1
        assert "clusters" not in result[0].label


# ---------------------------------------------------------------------------
# Tests: prompt_selection
# ---------------------------------------------------------------------------


class TestPromptSelection:
    """Legacy ``builtins.input`` mocks removed.

    ``prompt_selection`` reads directly via ``termios``/``tty``/``select.select``
    on ``sys.stdin`` — patching ``builtins.input`` has no effect. The real
    termios path is covered in ``test_prompt_selection_termios.py`` using
    ``pty.openpty``.
    """

    def test_placeholder(self) -> None:
        # Keep the class importable for downstream tooling; real coverage is
        # in the pty-based module.
        assert prompt_selection is not None
