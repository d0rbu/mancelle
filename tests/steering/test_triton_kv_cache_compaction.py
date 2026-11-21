from __future__ import annotations

import copy
import importlib.util
import sys
from contextlib import contextmanager
from typing import Any

import pytest
import torch as t

import steering.eval.greedy_token_batch as greedy

compaction: Any
if sys.platform == "linux":
    import steering.eval.triton_kv_cache_compaction as compaction
else:
    compaction = None

pytestmark = pytest.mark.skipif(compaction is None, reason="Triton helper requires Linux")

Cache = greedy._ExpandableRaggedKVCache


def _original_compact(cache, keep):
    """Pre-integration method, including direct-call partial-error semantics."""
    for layer in range(cache.n_layers):
        k, v = cache.k_layers[layer], cache.v_layers[layer]
        for destination, source in enumerate(keep):
            if destination != source:
                k[destination].copy_(k[source])
                v[destination].copy_(v[source])
        cache.k_layers[layer] = k[: len(keep)]
        cache.v_layers[layer] = v[: len(keep)]
    for destination, source in enumerate(keep):
        if destination != source:
            cache.cache_seqlens[destination].copy_(cache.cache_seqlens[source])
    cache.cache_seqlens = cache.cache_seqlens[: len(keep)]
    cache.batch_size = len(keep)


def _bytes(tensor):
    storage = tensor.untyped_storage()
    return t.empty(0, dtype=t.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _cache(device=None, *, batch=5, width=1041, offset=0, layers=2):
    device = t.device("cpu") if device is None else device
    cache = Cache(batch, 1, 1, 3, width, layers, device, t.float8_e4m3fn)
    for index in range(layers * 2):
        coordinates = t.arange(batch * width + offset + 13, device=device, dtype=t.int64)
        raw = ((coordinates ^ (coordinates >> 8) ^ (coordinates >> 16)) + index * 29).to(t.uint8)
        tensor = (
            raw[offset : offset + batch * width].view(t.float8_e4m3fn).reshape(batch, 1, 1, width)
        )
        (cache.k_layers if index < layers else cache.v_layers)[index % layers] = tensor
    cache.cache_seqlens.copy_(t.arange(batch, device=device, dtype=t.int32) * 3 + 1)
    return cache


def _snapshot(cache):
    tensors = [*cache.k_layers, *cache.v_layers, cache.cache_seqlens]
    return {
        "metadata": [
            cache.batch_size,
            cache.max_seq_len,
            cache.final_seq_len,
            cache._initial_row_token_budget,
            [(tuple(x.shape), x.stride(), x.storage_offset(), x.dtype) for x in tensors],
        ],
        "bytes": [_bytes(x).cpu().clone() for x in tensors],
        "pointers": [x.untyped_storage().data_ptr() for x in tensors],
    }


def _assert_equal(left, right, *, pointers=False):
    assert left["metadata"] == right["metadata"]
    assert all(t.equal(a, b) for a, b in zip(left["bytes"], right["bytes"], strict=True))
    if pointers:
        assert left["pointers"] == right["pointers"]


@contextmanager
def _cuda_metadata(monkeypatch, *, capability=(8, 9), error=None, fake_cuda=True):
    """Run host admission with real CPU storage and an explicit launch simulator."""
    original_tensor = t.tensor
    observed: dict[str, Any] = {"queries": [], "indices": [], "launches": []}

    class Kernel:
        def __getitem__(self, grid):
            def launch(k, v, indices, rows, row_bytes, **flags):
                assert grid == ((row_bytes + 4095) // 4096,)
                assert flags == {
                    "block_bytes": 4096,
                    "num_warps": 4,
                    "num_stages": 1,
                    "enable_fp_fusion": False,
                    "enable_reflect_ftz": False,
                }
                assert k.dtype == v.dtype == t.uint8 and indices.dtype == t.int64
                observed["launches"].append((rows, row_bytes))
                if error is not None:
                    k.flatten()[0] = 231
                    raise error
                for destination, source in enumerate(indices.tolist()):
                    if destination != source:
                        k[destination].copy_(k[source])
                        v[destination].copy_(v[source])

            return launch

    def tensor(data, **kwargs):
        assert kwargs == {"dtype": t.int64, "device": t.device("cuda:0")}
        observed["indices"].append(list(data))
        return original_tensor(data, dtype=t.int64)

    def capability_query(device):
        observed["queries"].append(device)
        return capability

    with monkeypatch.context() as patch:
        if fake_cuda:
            patch.setattr(t.Tensor, "is_cuda", property(lambda self: True))
            patch.setattr(t.Tensor, "device", property(lambda self: t.device("cuda:0")))
        patch.setattr(t.cuda, "get_device_capability", capability_query)
        patch.setattr(t, "tensor", tensor)
        patch.setattr(compaction, "_compact_layer_rows_kernel", Kernel())
        yield observed


@pytest.mark.parametrize(
    "keep", [[1, 2, 3, 4], [0, 2, 4], [0, 1, 2]], ids=["overlap", "holes", "suffix"]
)
def test_admitted_host_route_preserves_full_storage(monkeypatch, keep):
    cache = _cache(offset=7)
    expected = copy.deepcopy(cache)
    before = _snapshot(cache)
    _original_compact(expected, keep)
    with t.inference_mode(), _cuda_metadata(monkeypatch) as observed:
        assert compaction.try_compact_rows(cache, keep, cache_type=Cache) is True
    _assert_equal(_snapshot(cache), _snapshot(expected))
    assert _snapshot(cache)["pointers"] == before["pointers"]
    assert observed["queries"] == [t.device("cuda:0")]
    assert observed["indices"] == [keep]
    assert len(observed["launches"]) == cache.n_layers


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "bfloat16",
        "position_dtype",
        "shape",
        "missing_metadata",
        "post_oom_stride",
        "alias_kv",
        "alias_layers",
        "outside_inference",
        "non_ada",
        "cache_subclass",
        "tensor_subclass",
    ],
)
def test_rejected_host_route_is_unmodified_then_uses_original(monkeypatch, case):
    cache = _cache()
    expected = None
    if case == "bfloat16":
        cache.k_layers[0] = cache.k_layers[0].to(t.bfloat16)
    elif case == "position_dtype":
        cache.cache_seqlens = cache.cache_seqlens.to(t.int64)
    elif case == "shape":
        cache.head_dim -= 1
    elif case == "missing_metadata":
        del cache.head_dim
    elif case == "post_oom_stride":
        cache.k_layers[0] = t.zeros((5, 3, 1, 1041), dtype=t.float8_e4m3fn)[:, :1]
    elif case == "alias_kv":
        cache.v_layers[0] = cache.k_layers[0]
    elif case == "alias_layers":
        cache.k_layers[1] = cache.k_layers[0]
    elif case == "cache_subclass":

        class Child(Cache):
            pass

        cache.__class__ = Child
    elif case == "tensor_subclass":

        class TensorChild(t.Tensor):
            pass

        expected = copy.deepcopy(cache)
        cache.k_layers[0] = cache.k_layers[0].as_subclass(TensorChild)
        expected.k_layers[0] = expected.k_layers[0].as_subclass(TensorChild)
    if expected is None:
        expected = copy.deepcopy(cache)
    before = _snapshot(cache)
    keep = [1, 3, 4]
    with (
        t.inference_mode(case != "outside_inference"),
        _cuda_metadata(
            monkeypatch, capability=(9, 0) if case == "non_ada" else (8, 9), fake_cuda=case != "cpu"
        ) as observed,
    ):
        assert compaction.try_compact_rows(cache, keep, cache_type=Cache) is False
    _assert_equal(before, _snapshot(cache), pointers=True)
    assert observed["indices"] == observed["launches"] == []
    assert len(observed["queries"]) == int(case == "non_ada")
    # Exercise the actual class-method fallback, including inherited subclasses.
    cache.compact(keep)
    _original_compact(expected, keep)
    _assert_equal(_snapshot(cache), _snapshot(expected))


@pytest.mark.parametrize("keep", [[], [0, 1, 2, 3, 4], [1, 1], [2, 0], [-1, 2], [1, 9], (1, 2)])
def test_direct_unsupported_keeps_preserve_original_errors_and_partial_state(monkeypatch, keep):
    cache, expected = _cache(), _cache()
    before = _snapshot(cache)
    with t.inference_mode(), _cuda_metadata(monkeypatch) as observed:
        assert compaction.try_compact_rows(cache, keep, cache_type=Cache) is False
    _assert_equal(before, _snapshot(cache), pointers=True)
    assert observed == {"queries": [], "indices": [], "launches": []}
    errors = []
    for function, value in ((_original_compact, expected), (lambda c, k: c.compact(k), cache)):
        try:
            function(value, keep)
        except (IndexError, ValueError) as error:
            errors.append((type(error), str(error)))
        else:
            errors.append(None)
    assert errors[0] == errors[1]
    _assert_equal(_snapshot(cache), _snapshot(expected))


def test_outer_validation_and_full_length_noop_precede_helper(monkeypatch):
    cache = _cache()
    before = _snapshot(cache)
    monkeypatch.setattr(
        greedy, "try_compact_rows", lambda *a, **k: pytest.fail("unexpected helper")
    )
    greedy._compact_kv_cache_rows(cache, [99] * 5)
    for keep, error in (
        ([], ValueError),
        ([1, 1], ValueError),
        ([2, 0], ValueError),
        ([1, 9], IndexError),
    ):
        with pytest.raises(error):
            greedy._compact_kv_cache_rows(cache, keep)
    _assert_equal(before, _snapshot(cache), pointers=True)


def test_outer_dispatch_keeps_subclass_override(monkeypatch):
    calls = []

    class CustomCache(Cache):
        def compact(self, keep_rows):
            calls.append(list(keep_rows))

    cache = _cache()
    cache.__class__ = CustomCache
    before = _snapshot(cache)
    monkeypatch.setattr(
        greedy, "try_compact_rows", lambda *a, **k: pytest.fail("override bypassed")
    )
    greedy._compact_kv_cache_rows(cache, [1, 3])
    assert calls == [[1, 3]]
    _assert_equal(before, _snapshot(cache), pointers=True)


def test_admitted_launch_failure_does_not_retry_eager(monkeypatch):
    cache = _cache()
    error = RuntimeError("injected launch failure after write")
    with t.inference_mode(), _cuda_metadata(monkeypatch, error=error) as observed:
        monkeypatch.setattr(t.Tensor, "copy_", lambda *a, **k: pytest.fail("eager retry"))
        with pytest.raises(RuntimeError) as caught:
            cache.compact([1, 2, 3, 4])
    assert caught.value is error and len(observed["launches"]) == 1
    assert cache.batch_size == 5 and _bytes(cache.k_layers[0])[0].item() == 231


def test_partial_growth_rollback_remains_eligible_for_eager_compaction(monkeypatch):
    cache = Cache(4, 1, 3, 5, 17, 2, t.device("cpu"), t.float8_e4m3fn)
    cache.compact([1, 3])
    original_empty, allocations = t.empty, []

    def fail_third(*args, **kwargs):
        allocations.append(1)
        if len(allocations) == 3:
            raise t.OutOfMemoryError("second layer")
        return original_empty(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(t, "empty", fail_third)
        with pytest.raises(greedy.GreedyBatchCapacityError, match="allocation failed"):
            cache.ensure_capacity(4)
    assert cache.max_seq_len == 3 and not cache.k_layers[0].is_contiguous()
    expected = copy.deepcopy(cache)
    with t.inference_mode(), _cuda_metadata(monkeypatch) as observed:
        assert compaction.try_compact_rows(cache, [1], cache_type=Cache) is False
    assert observed == {"queries": [], "indices": [], "launches": []}
    cache.compact([1])
    _original_compact(expected, [1])
    _assert_equal(_snapshot(cache), _snapshot(expected))


def test_missing_helper_and_nonlinux_import_keep_original_method(monkeypatch):
    cache, expected = _cache(), _cache()
    monkeypatch.setattr(greedy, "try_compact_rows", None)
    cache.compact([1, 3])
    _original_compact(expected, [1, 3])
    _assert_equal(_snapshot(cache), _snapshot(expected))
    specification = importlib.util.spec_from_file_location("_compaction_nonlinux", greedy.__file__)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "darwin")
        specification.loader.exec_module(module)
    assert module.try_compact_rows is None


@pytest.mark.slow
@pytest.mark.parametrize("case", ["overlap_chain", "offset_holes", "tile_tail"])
def test_cuda_compaction_preserves_all_backing_bytes(monkeypatch, case):
    if not t.cuda.is_available() or t.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires Ada CUDA")
    device = t.device("cuda", t.cuda.current_device())
    width = 4113 if case == "tile_tail" else 1041
    offset = 7 if case in ("offset_holes", "tile_tail") else 0
    cache = _cache(device, width=width, offset=offset)
    reference = _cache(device, width=width, offset=offset)
    initial = _snapshot(cache)
    assert set(initial["bytes"][0][:256].tolist()) == set(range(256))
    if case == "tile_tail":
        source = offset + width  # The retained row is source row 1.
        assert cache.k_layers[0].storage_offset() == 7
        # A broken second CTA copying source tile 0 must differ in its 17 tail bytes.
        assert not t.equal(
            initial["bytes"][0][source : source + 17],
            initial["bytes"][0][source + 4096 : source + 4113],
        )
    else:
        assert not t.equal(initial["bytes"][0][:17], initial["bytes"][0][1024:1041])
    keeps = (
        [[1, 2, 3, 4], [1, 2, 3], [0, 2]]
        if case == "overlap_chain"
        else [[0, 2, 4]]
        if case == "offset_holes"
        else [[1]]
    )
    original_kernel, launches = compaction._compact_layer_rows_kernel, []

    class Witness:
        def __getitem__(self, grid):
            assert grid == ((width + 4095) // 4096,)
            launch = original_kernel[grid]

            def run(*args, **kwargs):
                result = launch(*args, **kwargs)
                launches.append(grid)
                return result

            return run

    monkeypatch.setattr(compaction, "_compact_layer_rows_kernel", Witness())
    with t.inference_mode():
        for keep in keeps:
            _original_compact(reference, keep)
            greedy._compact_kv_cache_rows(cache, keep)
            _assert_equal(_snapshot(cache), _snapshot(reference))
            assert _snapshot(cache)["pointers"] == initial["pointers"]
    assert len(launches) == cache.n_layers * len(keeps)
