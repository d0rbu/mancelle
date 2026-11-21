import importlib
import inspect

import pytest
from omegaconf import OmegaConf


def test_benchmark_exposes_native_runner():
    mod = importlib.import_module(
        "scripts.benchmark.benchmark_activation_methods"
    )
    assert hasattr(mod, "run_method_native"), (
        "benchmark must expose run_method_native for the native backend"
    )


def test_benchmark_has_correctness_gate():
    mod = importlib.import_module(
        "scripts.benchmark.benchmark_activation_methods"
    )
    src = inspect.getsource(mod)
    assert "assert_feature_equivalent" in src or "compare_outputs" in src, (
        "benchmark must gate timing on a correctness comparison"
    )
    assert "run_method_native" in src


def test_opt_in_gpt2_benchmark_supplies_exact_sae_identity(monkeypatch):
    mod = importlib.import_module(
        "scripts.benchmark.benchmark_activation_methods"
    )
    captured: dict[str, object] = {}

    class _StopAfterLoad(RuntimeError):
        pass

    def _capture_load(**kwargs):
        captured.update(kwargs)
        raise _StopAfterLoad

    monkeypatch.setattr(mod, "load_sae_and_model", _capture_load)
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "n_warmup": 0,
            "n_iters": 0,
            "smoke": False,
            "models": [],
            "run_gpt2": True,
            "batch_size": 1,
            "seq_len": 8,
            "layer": 7,
        }
    )

    with pytest.raises(_StopAfterLoad):
        mod.main.__wrapped__(cfg)

    assert captured == {
        "model_name": "gpt2",
        "layer": 7,
        "device": "cpu",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.7.hook_resid_pre",
    }
