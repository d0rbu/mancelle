"""Pin chat_eval.yaml defaults.

Default feature_configs is empty, so chat_eval does not create an implicit
empty static target-delta pass. target_real is opt-in so default eval
invocations do not trip over cache placeholders.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_CHAT_EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval"


def test_chat_eval_yaml_default_feature_configs():
    """Default feature_configs does not request a static target-delta pass."""
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="chat_eval", overrides=["source=base"])
    fcs = OmegaConf.to_container(cfg.feature_configs, resolve=True)
    assert fcs == []


def test_chat_eval_yaml_target_real_configs_default_empty():
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="chat_eval", overrides=["source=base"])
    trcs = OmegaConf.to_container(cfg.target_real_configs, resolve=False)
    assert trcs == []


def test_chat_eval_yaml_target_real_configs_can_be_disabled_on_cli():
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="chat_eval",
            overrides=[
                "source=base",
                "target_real_configs=null",
            ],
        )
    assert cfg.target_real_configs is None


def test_chat_eval_yaml_target_real_default_parses_to_empty_list():
    from steering.eval.target_real_runner import parse_target_real_configs_from_cfg

    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="chat_eval", overrides=["source=base"])

    assert parse_target_real_configs_from_cfg(cfg, suite="chat_target") == []


def test_base_and_dclm_eval_yaml_target_real_defaults_parse_to_empty_list():
    from steering.eval.target_real_runner import parse_target_real_configs_from_cfg

    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        base_cfg = compose(config_name="base_eval", overrides=["source=base"])
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        dclm_cfg = compose(config_name="dclm_eval", overrides=["source=base"])

    assert OmegaConf.to_container(base_cfg.target_real_configs, resolve=False) == []
    assert OmegaConf.to_container(dclm_cfg.target_real_configs, resolve=False) == []
    assert parse_target_real_configs_from_cfg(base_cfg, suite="core_target") == []
    assert parse_target_real_configs_from_cfg(dclm_cfg, suite="core_target") == []


def test_chat_eval_yaml_feature_configs_null_also_skips_static_configs():
    """Override `feature_configs=null` on the CLI to skip static target deltas."""
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="chat_eval",
            overrides=["source=base", "feature_configs=null"],
        )
    assert cfg.feature_configs is None


def test_chat_eval_yaml_keeps_conditioning_block_with_identity():
    """The conditioning block must still expose the identity bundle so
    auto-resolution can compute the cache_dir at parse time."""
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="chat_eval", overrides=["source=base"])
    assert "conditioning" in cfg
    assert "identity" in cfg.conditioning
    assert cfg.conditioning.identity.model_id is not None
    assert cfg.conditioning.identity.sae_id is not None
    assert cfg.conditioning.identity.layer is not None
    assert cfg.conditioning.identity.pooling is not None


def test_chat_eval_yaml_target_real_default_auto_resolves_at_parse(monkeypatch, tmp_path):
    """Opted-in YAML + parse_target_real_configs_from_cfg auto-resolves cache_dir."""
    import steering.eval.eval_cache_paths as paths_mod

    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    from steering.eval.target_real_runner import parse_target_real_configs_from_cfg

    feature_hash = "a" * 64
    with initialize_config_dir(config_dir=str(_CHAT_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="chat_eval",
            overrides=[
                "source=base",
                f"conditioning.identity.feature_selection_hash={feature_hash}",
                "target_real_configs=[{name:pooled,mode:pooled},{name:lockstep,mode:lockstep}]",
            ],
        )
    parsed = parse_target_real_configs_from_cfg(cfg, suite="chat_target")
    assert len(parsed) == 2
    for trc in parsed:
        cache_dir_str = str(trc.cache_dir)
        assert "eval_caches/chat_target" in cache_dir_str, (
            f"cache_dir must auto-resolve under eval_caches/chat_target/; got {cache_dir_str}"
        )
        assert cache_dir_str.endswith(f"/c{cfg.model_max_seq_len}/{feature_hash[:16]}")
