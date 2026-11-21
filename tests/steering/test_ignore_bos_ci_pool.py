"""_context_independent_forward_and_pool_batch applies the per-segment mask.

Uses mean_pool so the assertion is strictly RED before the change:
acts rows = [[1,1],[2,2],[3,3]]; without the mask the CI batch pools all 3
rows → mean [2,2]; with keep_mask=(False,True,True) it pools rows 1,2 →
mean [2.5,2.5]. The pre-change code (no mask threaded) yields [2,2] → fail.
"""

import torch as t

import steering.extract_features as ef
from steering.extract_features import (
    ContextIndependentSegmentInput,
    _context_independent_forward_and_pool_batch,
)
from steering.feature_paths import PoolingMode


class _StubSAE(t.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = t.nn.Parameter(t.zeros(1))

    def encode(self, x):
        return x.clamp(min=0.0)


class _StubModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = t.nn.Parameter(t.zeros(1))


def _acts_for(model_input):
    n = len(model_input)
    return t.stack([t.tensor([float(i + 1), float(i + 1)]) for i in range(n)])


def test_ci_pool_drops_natural_bos_mean(monkeypatch):
    def fake_batched(model, sae, model_inputs, **kw):
        return [_acts_for(mi) for mi in model_inputs]

    monkeypatch.setattr(ef, "_get_sae_activations_batched", fake_batched)

    ci = ContextIndependentSegmentInput(
        doc_idx=0, seg_idx=0, model_input=[2, 10, 11],
        bos_injected=False, original_length=3,
        keep_mask=(False, True, True),
    )
    pooled = _context_independent_forward_and_pool_batch(
        model=_StubModel(), sae=_StubSAE(), batch=[ci], layer=0,  # ty: ignore[invalid-argument-type]
        pooling=PoolingMode.mean_pool, hook_name=None,
        activation_type=None, nnterp_model=None,
    )
    # rows = [[1,1],[2,2],[3,3]]; drop row0 → mean of rows 1,2 = [2.5,2.5]
    assert t.allclose(pooled, t.tensor([[2.5, 2.5]]))


def test_ci_pool_keep_mask_none_unaffected(monkeypatch):
    def fake_batched(model, sae, model_inputs, **kw):
        return [_acts_for(mi) for mi in model_inputs]

    monkeypatch.setattr(ef, "_get_sae_activations_batched", fake_batched)

    ci = ContextIndependentSegmentInput(
        doc_idx=0, seg_idx=0, model_input=[2, 10, 11],
        bos_injected=False, original_length=3, keep_mask=None,
    )
    pooled = _context_independent_forward_and_pool_batch(
        model=_StubModel(), sae=_StubSAE(), batch=[ci], layer=0,  # ty: ignore[invalid-argument-type]
        pooling=PoolingMode.mean_pool, hook_name=None,
        activation_type=None, nnterp_model=None,
    )
    # no mask → mean of all 3 rows = [2,2]
    assert t.allclose(pooled, t.tensor([[2.0, 2.0]]))
