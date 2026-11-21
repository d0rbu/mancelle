import inspect

import torch as t

import steering.extract_features as ef


def test_find_optimal_batch_sizes_accepts_native_model():
    sig = inspect.signature(ef.find_optimal_batch_sizes)
    assert "native_model" in sig.parameters
    assert sig.parameters["native_model"].default is None


def test_find_optimal_batch_sizes_threads_native_into_warmup(monkeypatch, tmp_path):
    """When native_model is set, the tuner's warmup _get_sae_activations_batched
    must receive it (so profiling uses the right backend)."""
    seen = {}

    def fake_batched(model, sae, token_id_lists, *, layer,
                     nnterp_model=None, native_model=None,
                     hook_name=None, activation_type=ef.ActivationType.residual):
        seen["native_model"] = native_model
        seen["called"] = True
        raise RuntimeError("stop after warmup")  # abort tuner early

    monkeypatch.setattr(ef, "_get_sae_activations_batched", fake_batched)

    sentinel = object()

    class _Cfg:
        n_ctx = 64

    class _SAE:
        def __init__(self):
            self._p = t.nn.Parameter(t.zeros(1))

        def parameters(self):
            yield self._p

    class _Model:
        cfg = _Cfg()

    try:
        ef.find_optimal_batch_sizes(
            _Model(), _SAE(), layer=0, context_window=16,
            native_model=sentinel,  # ty: ignore[invalid-argument-type]
            max_gpu_batch_size=1, max_cpu_batch_size=1,
            min_gpu_batch_size=1, min_cpu_batch_size=1,
            n_trials=1, n_docs=2,
        )
    except RuntimeError:
        pass  # expected: we abort inside the stubbed warmup
    assert seen.get("called") is True
    assert seen.get("native_model") is sentinel, (
        "find_optimal_batch_sizes must thread native_model into its warmup "
        "_get_sae_activations_batched call"
    )
