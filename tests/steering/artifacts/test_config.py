"""Hydra config loading for HF bucket sync."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from steering.artifacts.categories import resolve_category
from steering.artifacts.config import ArtifactsConfig, CategoryConfig


@pytest.fixture(autouse=True)
def _reset_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose_default():
    cfg_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "configs" / "artifacts")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="default")


def test_default_config_loads():
    raw = _compose_default()
    cfg = ArtifactsConfig.from_config(raw)
    assert cfg.bucket_owner
    assert cfg.bucket_name
    assert set(cfg.categories) == {
        "base_checkpoints", "sft_checkpoints", "encdec_checkpoints", "features",
    }


def test_sync_options_have_fixed_defaults_but_filters_remain_cli_additions():
    raw = _compose_default()

    assert raw.dry_run is False
    assert raw.delete is False
    assert raw.no_default_include is False
    assert raw.no_default_exclude is False
    assert "include" not in raw
    assert "exclude" not in raw


def test_category_required_fields():
    raw = _compose_default()
    cfg = ArtifactsConfig.from_config(raw)
    cat = cfg.categories["base_checkpoints"]
    assert isinstance(cat, CategoryConfig)
    assert cat.local_root  # str template, expanded at use time
    assert cat.remote_subpath
    # default_exclude is a list (may be empty) of glob strings
    assert isinstance(cat.default_exclude, list)
    assert isinstance(cat.default_include, list)


@pytest.mark.parametrize(
    "category",
    ["base_checkpoints", "sft_checkpoints", "encdec_checkpoints"],
)
def test_checkpoint_categories_exclude_real_optimizer_filenames(category: str) -> None:
    raw = _compose_default()
    cfg = ArtifactsConfig.from_config(raw)
    assert cfg.categories[category].default_exclude == ["**/optim_*.pt"]


def test_resolve_category_local_root_uses_env_var(monkeypatch, tmp_path):
    """Category resolution expands NANOCHAT_BASE_DIR when set."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    raw = _compose_default()
    cfg = ArtifactsConfig.from_config(raw)
    resolved = resolve_category("base_checkpoints", cfg).local_root
    # OmegaConf has already substituted $NANOCHAT_BASE_DIR at compose time,
    # so the resolved Path should be tmp_path / "base_checkpoints"
    assert resolved == tmp_path / "base_checkpoints"


def test_from_config_missing_categories_key_raises_with_context():
    with pytest.raises(KeyError, match="missing required key"):
        ArtifactsConfig.from_config(
            OmegaConf.create({"bucket_owner": "o", "bucket_name": "b"})
        )


def test_from_config_malformed_category_raises_with_context():
    with pytest.raises(TypeError, match="malformed category spec"):
        ArtifactsConfig.from_config(
            OmegaConf.create(
                {
                    "bucket_owner": "o",
                    "bucket_name": "b",
                    "categories": {
                        "bad": {"local_root": "x", "unknown_field": True}
                    },
                }
            )
        )


# ---------------------------------------------------------------------------
# Coverage + robustness
# ---------------------------------------------------------------------------


def test_resolve_category_unset_env_var_keeps_literal():
    """Category resolution passes through unset env vars without raising.

    This documents the current behavior: a CategoryConfig constructed with
    a literal `$VAR` string returns a Path containing the unexpanded text.
    Downstream sync_bucket would fail; the contract is that callers should
    use Hydra's resolution at compose time, not raw $VAR strings."""
    import os
    cat = CategoryConfig(local_root="$NANO_GPT_DEFINITELY_UNSET/foo", remote_subpath="x")
    # Make sure the env var really isn't set
    assert "NANO_GPT_DEFINITELY_UNSET" not in os.environ
    cfg = ArtifactsConfig(bucket_owner="o", bucket_name="b", categories={"c": cat})
    p = resolve_category("c", cfg).local_root
    # Path keeps the literal $-token; no exception
    assert "$NANO_GPT_DEFINITELY_UNSET" in str(p)


def test_resolve_category_expands_tilde():
    """Category resolution expands ~ to $HOME."""
    import os
    cat = CategoryConfig(local_root="~/nano_test_path", remote_subpath="x")
    cfg = ArtifactsConfig(bucket_owner="o", bucket_name="b", categories={"c": cat})
    p = resolve_category("c", cfg).local_root
    assert "~" not in str(p)
    assert str(p).startswith(os.path.expanduser("~"))


def test_dataclasses_are_frozen():
    """Both config dataclasses are immutable (frozen) — protects against accidental mutation."""
    cat = CategoryConfig(local_root="/x", remote_subpath="x")
    cfg = ArtifactsConfig(bucket_owner="o", bucket_name="b", categories={"c": cat})
    with pytest.raises(FrozenInstanceError):
        cat.local_root = "/y"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        cfg.bucket_owner = "x"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
