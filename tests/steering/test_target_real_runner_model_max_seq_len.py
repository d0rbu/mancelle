"""Tests for the model_max_seq_len kwarg plumbing through
parse_target_real_configs_from_cfg → _build_auto_resolver →
default_target_cache_dir.

Background: training_progression's checkpoints can have varying context
lengths (e.g., model.dec_config.sequence_len = 2048). The auto-resolver
must use the model's actual context length, not a default of 1024 read
from a cfg layout that doesn't carry the field.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from steering.eval.target_real_runner import parse_target_real_configs_from_cfg


@pytest.fixture
def fake_base_dir(tmp_path, monkeypatch):
    import steering.eval.eval_cache_paths as paths_mod
    monkeypatch.setattr(paths_mod, "get_base_dir", lambda: str(tmp_path))
    return str(tmp_path)


def _make_full_cfg(model_max_seq_len: int | None):
    """Top-level cfg layout — base_eval/chat_eval/dclm_eval style."""
    cfg_dict: dict = {
        "target_real_configs": [{"name": "trc1", "mode": "pooled"}],
        "conditioning": {
            "identity": {
                "model_id": "gemma-2-2b",
                "sae_id": "gemmascope-residual-l8-16k",
                "layer": 8,
                "pooling": "max_pool",
                "feature_selection_hash": "a" * 64,
            },
        },
    }
    if model_max_seq_len is not None:
        cfg_dict["model_max_seq_len"] = model_max_seq_len
    return OmegaConf.create(cfg_dict)


def _make_cond_block(model_max_seq_len: int | None):
    """training_progression layout: cfg.conditioning passed directly,
    identity at root, no model_max_seq_len at this level."""
    cfg_dict: dict = {
        "target_real_configs": [{"name": "trc1", "mode": "pooled"}],
        "identity": {
            "model_id": "gemma-2-2b",
            "sae_id": "gemmascope-residual-l8-16k",
            "layer": 8,
            "pooling": "max_pool",
            "feature_selection_hash": "a" * 64,
        },
    }
    if model_max_seq_len is not None:
        cfg_dict["model_max_seq_len"] = model_max_seq_len
    return OmegaConf.create(cfg_dict)


def test_explicit_model_max_seq_len_kwarg_overrides_cfg(fake_base_dir):
    """When kwarg=2048 and cfg.model_max_seq_len=512, resolver uses 2048."""
    cfg = _make_full_cfg(512)
    parsed = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", model_max_seq_len=2048,
    )
    assert len(parsed) == 1
    cd = str(parsed[0].cache_dir)
    assert "/c2048/" in cd, f"expected c2048 in {cd}"


def test_explicit_model_max_seq_len_used_when_cfg_missing(fake_base_dir):
    """When cfg has NO model_max_seq_len AND kwarg=2048, resolver uses 2048
    (NOT the default 1024). This is the training_progression bug fix."""
    cfg = _make_cond_block(None)
    parsed = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", model_max_seq_len=2048,
    )
    assert len(parsed) == 1
    cd = str(parsed[0].cache_dir)
    assert "/c2048/" in cd, f"expected c2048 in {cd}"


def test_kwarg_none_falls_back_to_cfg(fake_base_dir):
    """When kwarg=None and cfg.model_max_seq_len=512, resolver uses 512
    (preserves legacy base_eval/chat_eval/dclm_eval contract)."""
    cfg = _make_full_cfg(512)
    parsed = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", model_max_seq_len=None,
    )
    assert len(parsed) == 1
    cd = str(parsed[0].cache_dir)
    assert "/c512/" in cd, f"expected c512 in {cd}"


def test_kwarg_default_is_none(fake_base_dir):
    """The default for model_max_seq_len is None — no behavior change for
    callers that don't pass the kwarg."""
    cfg = _make_full_cfg(512)
    parsed_explicit = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", model_max_seq_len=None,
    )
    parsed_default = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target",
    )
    assert str(parsed_explicit[0].cache_dir) == str(parsed_default[0].cache_dir)


def test_neither_kwarg_nor_cfg_falls_back_to_1024(fake_base_dir):
    """When neither cfg.model_max_seq_len nor kwarg is set, the legacy
    default 1024 still applies (preserves prior behavior for misconfigured
    callers)."""
    cfg = _make_cond_block(None)
    parsed = parse_target_real_configs_from_cfg(cfg, suite="chat_target")
    assert len(parsed) == 1
    cd = str(parsed[0].cache_dir)
    assert "/c1024/" in cd, f"expected c1024 in {cd}"


def test_kwarg_zero_rejected_by_default_target_cache_dir(fake_base_dir):
    """model_max_seq_len=0 is non-positive — default_target_cache_dir
    raises ValueError. Pin this so the auto-resolver doesn't silently
    accept invalid context lengths."""
    cfg = _make_cond_block(None)
    with pytest.raises(ValueError, match="model_max_seq_len must be a positive"):
        parse_target_real_configs_from_cfg(
            cfg, suite="chat_target", model_max_seq_len=0,
        )


def test_kwarg_negative_rejected_by_default_target_cache_dir(fake_base_dir):
    """Negative model_max_seq_len is invalid."""
    cfg = _make_cond_block(None)
    with pytest.raises(ValueError, match="model_max_seq_len must be a positive"):
        parse_target_real_configs_from_cfg(
            cfg, suite="chat_target", model_max_seq_len=-1,
        )


def test_explicit_kwarg_with_multiple_entries_consistent(fake_base_dir):
    """When the cfg has multiple target_real_configs entries needing
    auto-resolve, all entries get the same model_max_seq_len."""
    cfg_dict: dict = {
        "target_real_configs": [
            {"name": "pooled", "mode": "pooled"},
            {"name": "lockstep", "mode": "lockstep"},
        ],
        "identity": {
            "model_id": "gemma-2-2b",
            "sae_id": "gemmascope-residual-l8-16k",
            "layer": 8,
            "pooling": "max_pool",
            "feature_selection_hash": "a" * 64,
        },
    }
    cfg = OmegaConf.create(cfg_dict)
    parsed = parse_target_real_configs_from_cfg(
        cfg, suite="core_target", model_max_seq_len=2048,
    )
    assert len(parsed) == 2
    for p in parsed:
        assert "/c2048/" in str(p.cache_dir)


def test_explicit_kwarg_does_not_affect_explicit_cache_dir_entries(
    fake_base_dir, tmp_path,
):
    """When an entry has an explicit cache_dir, the kwarg is irrelevant
    for that entry — the cache_dir flows through verbatim."""
    explicit = str(tmp_path / "my_cache")
    cfg_dict: dict = {
        "target_real_configs": [
            {"name": "explicit", "mode": "pooled", "cache_dir": explicit},
            {"name": "auto", "mode": "pooled"},  # cache_dir missing
        ],
        "identity": {
            "model_id": "gemma-2-2b",
            "sae_id": "gemmascope-residual-l8-16k",
            "layer": 8,
            "pooling": "max_pool",
            "feature_selection_hash": "a" * 64,
        },
    }
    cfg = OmegaConf.create(cfg_dict)
    parsed = parse_target_real_configs_from_cfg(
        cfg, suite="core_target", model_max_seq_len=2048,
    )
    assert len(parsed) == 2
    # Explicit cache_dir is unchanged.
    assert str(parsed[0].cache_dir) == explicit
    # Auto-resolved cache_dir uses the kwarg.
    assert "/c2048/" in str(parsed[1].cache_dir)


def test_kwarg_propagates_through_default_target_cache_dir_call(
    fake_base_dir, monkeypatch,
):
    """Verify the kwarg flows to the underlying default_target_cache_dir
    call, not silently dropped. This catches the bug where the parser
    might compute the path itself without delegating to the canonical
    builder."""

    import steering.eval.target_real_runner as runner_mod
    captured = []
    real_default = runner_mod.default_target_cache_dir
    def spy(**kwargs):
        captured.append(kwargs)
        return real_default(**kwargs)
    monkeypatch.setattr(runner_mod, "default_target_cache_dir", spy)

    cfg = _make_cond_block(None)
    parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", model_max_seq_len=2048,
    )
    assert len(captured) == 1
    assert captured[0]["model_max_seq_len"] == 2048


def test_kwarg_overrides_cfg_when_both_set_via_omegaconf(fake_base_dir):
    """When cfg.model_max_seq_len exists AND kwarg is explicit, kwarg wins.
    This is the training_progression contract — the cfg's value (which
    may be a stale top-level default) must NOT shadow the model's actual
    sequence_len."""
    cfg = _make_full_cfg(1024)
    parsed = parse_target_real_configs_from_cfg(
        cfg, suite="chat_target", model_max_seq_len=2048,
    )
    cd = str(parsed[0].cache_dir)
    # kwarg=2048 wins over cfg=1024
    assert "/c2048/" in cd
    assert "/c1024/" not in cd
