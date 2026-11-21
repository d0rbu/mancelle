"""`_run_batched_extractor` per-item reduction: the #1240 memory bound.

Before this, `batch_size` bounded only the transient forward. Every item's full
``[seq_len, n_features]`` activation tensor was appended to the returned list and
stayed resident, so peak memory scaled with the ITEM COUNT. At a 65k-wide SAE and
~432 tokens that is ~113 MiB per item -- ~45 GiB for 400 items, to produce a
~105 MiB pooled result -- which made `feature_diffmean_encdec` unrunnable on a
24 GiB card at any batch size.

`per_item_reduce` collapses each item as its sub-batch is produced, so peak
scales with `batch_size`, which is what the parameter already implied.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

import steering.eval.axbench_base.extraction as extraction
from steering.eval.axbench_base.extraction import _run_batched_extractor
from steering.eval.axbench_base.types import AxBenchBaseError

_SEQ_LEN = 6
_N_FEATURES = 4


def _fake_acts(n_items: int) -> list[t.Tensor]:
    return [
        t.arange(_SEQ_LEN * _N_FEATURES, dtype=t.float32).reshape(_SEQ_LEN, _N_FEATURES) + float(i)
        for i in range(n_items)
    ]


@pytest.fixture
def stub_backend(monkeypatch):
    """Replace the real SAE forward; record the sub-batch sizes it was asked for."""
    calls: list[int] = []

    def _fake(*, model, sae, token_id_lists, layer, nnterp_model, hook_name, activation_type):
        del model, sae, layer, nnterp_model, hook_name, activation_type
        calls.append(len(token_id_lists))
        return _fake_acts(len(token_id_lists))

    monkeypatch.setattr(extraction, "_get_sae_activations_batched", _fake)
    return calls


def _run(n_items: int, batch_size: int | None, reduce_fn=None):
    return _run_batched_extractor(
        model=object(),
        sae=object(),
        token_id_lists=[[1, 2, 3] for _ in range(n_items)],
        layer=0,
        hook_name=None,
        batch_size=batch_size,
        per_item_reduce=reduce_fn,
    )


def test_without_reduction_behavior_is_unchanged(stub_backend) -> None:
    """The default path must be exactly what it was before #1240."""
    out = _run(5, batch_size=2)
    assert len(out) == 5
    assert all(tuple(x.shape) == (_SEQ_LEN, _N_FEATURES) for x in out)
    assert stub_backend == [2, 2, 1]


def test_native_side_channel_reaches_batched_fit_extractor(monkeypatch) -> None:
    collector = object()
    seen: list[dict[str, object]] = []

    def fake(**kwargs):
        seen.append(kwargs)
        return _fake_acts(len(kwargs["token_id_lists"]))

    monkeypatch.setattr(extraction, "_get_sae_activations_batched", fake)
    result = _run_batched_extractor(
        model=SimpleNamespace(native_model=collector),
        sae=object(),
        token_id_lists=[[1, 2, 3]],
        layer=0,
        hook_name="blocks.0.hook_resid_post",
    )
    assert seen[0]["native_model"] is collector
    assert len(result) == 1

    both = SimpleNamespace(native_model=collector, nnterp_model=object())
    _run_batched_extractor(
        model=both,
        sae=object(),
        token_id_lists=[[1, 2, 3]],
        layer=0,
        hook_name="blocks.0.hook_resid_post",
    )
    assert "native_model" not in seen[1]
    assert seen[1]["nnterp_model"] is both.nnterp_model


def test_reduction_is_applied_to_every_item(stub_backend) -> None:
    out = _run(5, batch_size=2, reduce_fn=lambda acts, i: acts.max(dim=0).values)
    assert len(out) == 5
    assert all(tuple(x.shape) == (_N_FEATURES,) for x in out), (
        "each item must be collapsed, not returned whole"
    )


def test_reduction_receives_GLOBAL_item_index(stub_backend) -> None:
    """The index must span the whole call, not restart per sub-batch.

    Callers key per-item metadata (completion offsets, input ids) off this
    index. A per-chunk index would silently pool the wrong span for every item
    after the first sub-batch -- wrong numbers, no error.
    """
    seen: list[int] = []

    def _reduce(acts: t.Tensor, index: int) -> t.Tensor:
        seen.append(index)
        return acts[0]

    _run(7, batch_size=3, reduce_fn=_reduce)
    assert seen == [0, 1, 2, 3, 4, 5, 6]


def test_reduction_result_matches_post_hoc_reduction(stub_backend) -> None:
    """Bitwise identity with the old materialize-then-reduce order."""
    raw = _run(5, batch_size=2)
    expected = [acts.max(dim=0).values for acts in raw]
    actual = _run(5, batch_size=2, reduce_fn=lambda acts, i: acts.max(dim=0).values)
    for want, got in zip(expected, actual, strict=True):
        assert t.equal(want, got)


def test_reduction_applies_when_batching_is_not_triggered(stub_backend) -> None:
    """batch_size=None and batch_size>=n_items both take the single-shot path."""
    for batch_size in (None, 10):
        out = _run(3, batch_size=batch_size, reduce_fn=lambda acts, i: acts[0])
        assert len(out) == 3
        assert all(tuple(x.shape) == (_N_FEATURES,) for x in out)


def test_peak_retained_tensors_scale_with_batch_not_item_count(monkeypatch) -> None:
    """The regression this change exists to prevent.

    Counts how many FULL activation tensors are alive at once, by tracking
    liveness with weakrefs. Without a reduction this grows with the item count;
    with one it must stay bounded by the sub-batch.
    """
    import weakref

    alive: list[weakref.ref] = []

    def _fake(*, model, sae, token_id_lists, layer, nnterp_model, hook_name, activation_type):
        del model, sae, layer, nnterp_model, hook_name, activation_type
        made = _fake_acts(len(token_id_lists))
        alive.extend(weakref.ref(x) for x in made)
        return made

    monkeypatch.setattr(extraction, "_get_sae_activations_batched", _fake)

    def live_count() -> int:
        return sum(1 for r in alive if r() is not None)

    alive.clear()
    # Hold the returned list: it IS the retention. Dropping it would free the
    # tensors before the count and make this leg pass vacuously.
    retained = _run(12, batch_size=3)  # no reduction
    assert live_count() == 12, "baseline: every raw tensor is retained"
    del retained

    alive.clear()
    out = _run(12, batch_size=3, reduce_fn=lambda acts, i: acts.max(dim=0).values)
    del out
    assert live_count() == 0, (
        "raw per-item activations must not survive the call when a reduction "
        "is supplied; peak must scale with batch_size, not item count"
    )


def test_invalid_batch_size_still_raises_its_own_message(stub_backend) -> None:
    with pytest.raises(AxBenchBaseError, match="batch_size must be >= 1 when set"):
        _run(3, batch_size=0)
