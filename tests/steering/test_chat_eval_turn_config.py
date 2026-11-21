"""Hydra presets cannot inherit incompatible fixed identity/selection defaults."""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from scripts.eval import chat_eval
from steering.segmentation import SegmentMode
from tests.steering.test_chat_eval_turn_cache_integration import turn_main_fixture

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/eval")


@pytest.mark.parametrize("name", ["build_eval_context_cache_turn", "validate_eval_context_cache_turn", "chat_eval_turn"])
def test_turn_presets_clear_legacy_defaults(name):
    # kills: silently inheriting threshold/Gemma-2 defaults, c1024 identity, or a 100-row enumeration cap.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=name)
    assert cfg.segment_mode == SegmentMode.turn_delimited_fixed_tokens.value
    assert cfg.model_max_seq_len == 2048
    if name == "chat_eval_turn":
        block = cfg.conditioning
        assert cfg.num_samples == 1 and cfg.temperature == 0.0
        assert cfg.max_problems is None
        assert cfg.problem_start is None
        assert cfg.feature_configs == []
        assert cfg.target_real_configs[0].mode == "pooled"
    else:
        block = cfg
        assert cfg.evals == "chat_target"
        assert cfg.chat_max_problems is None
    assert block.n_sae_features is None and block.num_buckets is None
    assert all(value is None for value in block.identity.values())
    if name == "build_eval_context_cache_turn":
        assert cfg.feature_selection is None
        assert cfg.negative_selection is None
        assert cfg.source_bos_char_len is None
        assert cfg.stats_path is None


def test_fixed_configuration_defaults_remain_unchanged():
    # kills: routing fixed evaluations into the turn preset or changing legacy selection/BOS defaults.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        builder = compose(config_name="build_eval_context_cache")
        chat = compose(config_name="chat_eval")
    assert builder.segment_mode == SegmentMode.fixed_tokens.value
    assert builder.source_bos_char_len == 5
    assert builder.feature_selection.threshold == 2048.9
    assert chat.segment_mode is None
    assert chat.conditioning.identity.model_id == "gemma-2-2b"
    assert chat.conditioning.n_sae_features == 16384
    assert chat.conditioning.num_buckets == 2
    assert chat.model_max_seq_len == 1024


def test_turn_preset_runs_real_chat_main(tmp_path, monkeypatch):
    # kills: a syntactically valid preset that cannot resolve and admit the real schema-5 fixture.
    fixture = turn_main_fixture(tmp_path, monkeypatch)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="chat_eval_turn", overrides=[
            "device_type=cpu", "dtype=float32", "max_problems=1", "max_new_tokens=3",
            f"model_tag={fixture.cfg.model_tag}", f"step={fixture.cfg.step}",
            f"target_real_configs=[{{name:pooled,mode:pooled,cache_dir:{fixture.cache_cfg.output_dir}}}]",
            "conditioning.source_segment_size=4", "conditioning.source_context_window=4096",
        ])
    chat_eval.main.__wrapped__(cfg)
    result = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/pooled.yaml").read_text())
    assert result["kind"] == "target_real_pooled"
    assert len(result["seeds"]["adhoc"]["results"]) == 6
