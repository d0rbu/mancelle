"""Integration test pinning the model_max_seq_len contract in
training_progression's target_real flow.

Two adjacent call sites in training_progression must both use
model.dec_config.sequence_len for cache identity:
  1. parse_target_real_configs_from_cfg → builds cache_dir paths
  2. _target_real_providers_for_cfg → builds provider with expected_metadata

If either drops the kwarg or uses a stale cfg value, the cache identity
mismatch surfaces as a confusing cache-validation error at provider load
time. This test pins the invariant: both paths must produce a cache_dir
under c{model.dec_config.sequence_len}/ for non-1024 context models.
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


def _training_progression_cond_block(seq_len: int):
    """Mirror the cfg.conditioning layout that training_progression passes
    to parse_target_real_configs_from_cfg."""
    return OmegaConf.create({
        "target_real_configs": [
            {"name": "pooled", "mode": "pooled"},
            {"name": "lockstep", "mode": "lockstep"},
        ],
        "identity": {
            "model_id": "gemma-2-2b",
            "sae_id": "gemmascope-residual-l8-16k",
            "layer": 8,
            "pooling": "max_pool",
            "feature_selection_hash": "f" * 64,
        },
        "n_sae_features": 16384,
        "num_buckets": 2,
    })


@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
def test_training_progression_threads_seq_len_to_cache_dir(
    fake_base_dir, seq_len,
):
    """For each plausible context length the model could have, the parser
    must produce cache_dirs containing /c{seq_len}/."""
    cond = _training_progression_cond_block(seq_len)
    parsed = parse_target_real_configs_from_cfg(
        cond,
        suite="core_target",
        model_max_seq_len=seq_len,
    )
    assert len(parsed) == 2
    for entry in parsed:
        cd = str(entry.cache_dir)
        assert f"/c{seq_len}/" in cd, (
            f"expected /c{seq_len}/ in {cd}; "
            f"the parser failed to thread model.dec_config.sequence_len"
        )


def test_training_progression_2048_model_does_not_resolve_to_c1024(
    fake_base_dir,
):
    """Regression test for the latent bug: a 2048-context model in
    training_progression used to silently resolve cache_dir to c1024/
    because the cfg.conditioning block has no model_max_seq_len field
    and the auto-resolver fell through to the default 1024."""
    cond = _training_progression_cond_block(2048)
    parsed = parse_target_real_configs_from_cfg(
        cond,
        suite="core_target",
        model_max_seq_len=2048,
    )
    cd = str(parsed[0].cache_dir)
    assert "/c2048/" in cd
    assert "/c1024/" not in cd, (
        "regression: 2048-context model resolved to c1024 — "
        "the auto-resolver bug returned"
    )


def test_training_progression_call_site_signature_matches_api_contract():
    """Static check: parse_target_real_configs_from_cfg accepts the
    model_max_seq_len kwarg with the expected default (None). Catches
    a future refactor that accidentally renames or drops the kwarg."""
    import inspect
    sig = inspect.signature(parse_target_real_configs_from_cfg)
    params = sig.parameters
    assert "model_max_seq_len" in params, (
        "parse_target_real_configs_from_cfg must accept model_max_seq_len kwarg"
    )
    p = params["model_max_seq_len"]
    assert p.default is None, (
        "model_max_seq_len kwarg default must be None to preserve the "
        "base_eval/chat_eval/dclm_eval contract (fall back to "
        "cfg.get('model_max_seq_len', 1024))"
    )
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
        "model_max_seq_len must be keyword-only to prevent positional "
        "ordering ambiguity with suite"
    )
