from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import steering.eval.greedy_token_batch as greedy_token_batch
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, Linear
from steering.eval.greedy_token_batch import generate_greedy_token_batch


class _Model:
    is_target_aware_encdec = False

    def __init__(
        self,
        *,
        sequence_len: int = 32,
        token_plan: list[list[int]] | None = None,
        non_finite_step: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            sequence_len=sequence_len,
            n_kv_head=1,
            n_embd=4,
            n_head=1,
            n_layer=1,
        )
        self.calls: list[t.Tensor] = []
        self.token_plan = token_plan or []
        self.non_finite_step = non_finite_step

    def get_device(self) -> t.device:
        return t.device("cpu")

    def forward(self, idx: t.Tensor, *, kv_cache) -> t.Tensor:
        self.calls.append(idx.clone())
        kv_cache.advance(idx.size(1))
        logits = t.full((idx.size(0), idx.size(1), 16), -1.0)
        step = len(self.calls) - 1
        planned = self.token_plan[min(step, len(self.token_plan) - 1)]
        for row, token in enumerate(planned):
            logits[row, -1, token] = 1.0
        if step == self.non_finite_step:
            logits[0, -1, 3] = t.nan
        return logits


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def encode_special(self, name: str) -> int:
        return {
            "<python_start>": 8,
            "<assistant_end>": 9,
            "<python_end>": 10,
            "<output_start>": 11,
            "<output_end>": 12,
        }[name]

    def decode(self, tokens: list[int]) -> str:
        assert tokens == [4]
        return "1+1"

    def encode(self, text: str) -> list[int]:
        assert text == "2"
        return [6]


class _RaggedTokenizer(_Tokenizer):
    def get_bos_token_id(self) -> int:
        return 63

    def encode_special(self, name: str) -> int:
        return {
            "<python_start>": 62,
            "<assistant_end>": 61,
            "<python_end>": 60,
            "<output_start>": 59,
            "<output_end>": 58,
        }[name]


def _tiny_gpt() -> GPT:
    model = GPT(
        GPTConfig(
            sequence_len=32,
            vocab_size=64,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=32,
            window_pattern="S",
        ),
        pad_vocab_size_to=64,
    ).eval()
    model.init_weights()
    return model


def test_staged_linear_forward_reuses_compute_weight_and_restores(
    monkeypatch,
) -> None:
    linear = Linear(3, 2, bias=False)
    compute_weight = linear.weight.detach().to(t.bfloat16)
    captured_weights: list[t.Tensor] = []

    def fake_linear(
        x: t.Tensor,
        weight: t.Tensor,
        bias: t.Tensor | None,
    ) -> t.Tensor:
        assert bias is None
        captured_weights.append(weight)
        return t.zeros((*x.shape[:-1], weight.size(0)), dtype=x.dtype)

    monkeypatch.setattr(greedy_token_batch.F, "linear", fake_linear)
    assert "forward" not in linear.__dict__
    with greedy_token_batch._replace_linear_forwards_with_staged_weights(
        [(linear, compute_weight)]
    ):
        linear(t.ones(1, 3, dtype=t.bfloat16))
        linear(t.ones(1, 3, dtype=t.bfloat16))

    assert "forward" not in linear.__dict__
    assert len(captured_weights) == 2
    assert all(weight.dtype == t.bfloat16 for weight in captured_weights)
    assert all(
        weight.untyped_storage().data_ptr() == compute_weight.untyped_storage().data_ptr()
        for weight in captured_weights
    )
    assert (
        captured_weights[0].untyped_storage().data_ptr()
        == captured_weights[1].untyped_storage().data_ptr()
    )


def test_staged_linear_forward_falls_back_for_unmapped_models(
    monkeypatch,
) -> None:
    linear = Linear(3, 2, bias=False)
    other_linear = Linear(3, 2, bias=False)
    compute_weight = linear.weight.detach().to(t.bfloat16)
    captured_weights: list[t.Tensor] = []

    def fake_linear(
        x: t.Tensor,
        weight: t.Tensor,
        bias: t.Tensor | None,
    ) -> t.Tensor:
        captured_weights.append(weight)
        return t.zeros((*x.shape[:-1], weight.size(0)), dtype=x.dtype)

    monkeypatch.setattr(greedy_token_batch.F, "linear", fake_linear)
    with greedy_token_batch._replace_linear_forwards_with_staged_weights(
        [(linear, compute_weight)]
    ):
        other_linear(t.ones(1, 3, dtype=t.float32))

    assert captured_weights == [other_linear.weight]


def test_staged_linear_forward_restores_class_after_error() -> None:
    linear = Linear(3, 2, bias=False)
    compute_weight = linear.weight.detach().to(t.bfloat16)
    original_forward = Linear.forward
    with pytest.raises(RuntimeError, match="probe"):
        with greedy_token_batch._replace_linear_forwards_with_staged_weights(
            [(linear, compute_weight)]
        ):
            raise RuntimeError("probe")

    assert Linear.forward is original_forward


def test_staged_weight_admission_accounts_for_ragged_source_cache(
    monkeypatch,
) -> None:
    model = _tiny_gpt()
    model.get_device = lambda: t.device("cuda")  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))
    original_dtypes = [parameter.dtype for parameter in model.parameters()]
    projected_cache_bytes = greedy_token_batch._projected_kv_cache_bytes(
        model,
        batch_size=3,
        max_total_len=7,
        ragged=True,
        dtype=t.bfloat16,
    )
    expected_cache_bytes = 2 * 2 * 4 * 7 * 2 * 8 * 2
    assert projected_cache_bytes == expected_cache_bytes

    monkeypatch.setattr(
        t.cuda,
        "mem_get_info",
        lambda _device: (
            projected_cache_bytes + greedy_token_batch._STAGED_WEIGHT_HEADROOM_BYTES,
            10**12,
        ),
    )
    with greedy_token_batch._stage_linear_weights_for_greedy_batch(
        model,
        batch_size=3,
        max_total_len=7,
        ragged=True,
        dtype=t.bfloat16,
    ) as staged:
        assert not staged

    original_forward = Linear.forward
    monkeypatch.setattr(t.cuda, "mem_get_info", lambda _device: (10**12, 10**12))
    with greedy_token_batch._stage_linear_weights_for_greedy_batch(
        model,
        batch_size=3,
        max_total_len=7,
        ragged=True,
        dtype=t.bfloat16,
    ) as staged:
        assert staged
        assert Linear.forward is not original_forward
    assert Linear.forward is original_forward
    assert [parameter.dtype for parameter in model.parameters()] == original_dtypes


def test_nested_staged_weight_context_reuses_outer_mapping(monkeypatch) -> None:
    model = _tiny_gpt()
    model.get_device = lambda: t.device("cuda")  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(t.cuda, "mem_get_info", lambda _device: (10**12, 10**12))
    original_forward = Linear.forward

    with greedy_token_batch._stage_linear_weights_for_greedy_batch(
        model,
        batch_size=3,
        max_total_len=7,
        ragged=True,
        dtype=t.bfloat16,
    ) as outer_staged:
        assert outer_staged
        outer_forward = Linear.forward
        assert outer_forward is not original_forward
        with greedy_token_batch._stage_linear_weights_for_greedy_batch(
            model,
            batch_size=2,
            max_total_len=5,
            ragged=True,
            dtype=t.bfloat16,
        ) as inner_staged:
            assert inner_staged
            assert Linear.forward is outer_forward
        assert Linear.forward is outer_forward

    assert Linear.forward is original_forward
    assert id(model) not in greedy_token_batch._ACTIVE_STAGED_LINEAR_MODEL_IDS


def test_multi_batch_staging_uses_largest_exact_cache_projection(monkeypatch) -> None:
    model = _tiny_gpt()
    captured: list[tuple[int, int, bool, t.dtype]] = []

    @contextmanager
    def fake_stage(
        _model,
        *,
        batch_size: int,
        max_total_len: int,
        ragged: bool,
        dtype: t.dtype,
    ):
        captured.append((batch_size, max_total_len, ragged, dtype))
        yield True

    monkeypatch.setattr(
        greedy_token_batch,
        "_stage_linear_weights_for_greedy_batch",
        fake_stage,
    )
    with greedy_token_batch._stage_linear_weights_for_greedy_batches(
        model,
        batch_specs=[(10, 100, True), (20, 40, True)],
        dtype=t.bfloat16,
    ) as staged:
        assert staged

    assert captured == [(10, 100, True, t.bfloat16)]


def test_categorical_token_batch_projects_only_answer_rows_and_tokens() -> None:
    class CategoricalModel(t.nn.Module):
        is_target_aware_encdec = False

        def __init__(self) -> None:
            super().__init__()
            self.lm_head = Linear(2, 8, bias=False)
            with t.no_grad():
                self.lm_head.weight[:, 0] = t.arange(8)
                self.lm_head.weight[:, 1] = 0

        def get_device(self) -> t.device:
            return t.device("cpu")

        def forward(self, token_ids: t.Tensor) -> t.Tensor:
            hidden = t.stack(
                (token_ids.to(t.float32), t.ones_like(token_ids, dtype=t.float32)),
                dim=-1,
            )
            return self.lm_head(hidden)

    model = CategoricalModel()
    original_forward = Linear.forward
    predictions = greedy_token_batch.predict_categorical_token_batch(
        model,
        prompt_token_ids=[[1, 2], [3]],
        answer_token_ids=[[1, 3], [4, 2]],
        pad_token_id=0,
    )

    assert predictions == [1, 0]
    assert Linear.forward is original_forward


def test_categorical_token_batch_restores_linear_forward_after_failure() -> None:
    class FailingModel(t.nn.Module):
        is_target_aware_encdec = False

        def __init__(self) -> None:
            super().__init__()
            self.lm_head = Linear(2, 8, bias=False)

        def get_device(self) -> t.device:
            return t.device("cpu")

        def forward(self, _token_ids: t.Tensor) -> t.Tensor:
            raise RuntimeError("injected categorical failure")

    original_forward = Linear.forward
    with pytest.raises(RuntimeError, match="injected categorical failure"):
        greedy_token_batch.predict_categorical_token_batch(
            FailingModel(),
            prompt_token_ids=[[1]],
            answer_token_ids=[[2]],
            pad_token_id=0,
        )

    assert Linear.forward is original_forward


def test_hopper_keeps_original_linear_forward_without_querying_memory(
    monkeypatch,
) -> None:
    model = _tiny_gpt()
    model.get_device = lambda: t.device("cuda")  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (9, 0))

    def unexpected_mem_get_info(_device):
        raise AssertionError("Hopper should reject staging before querying memory")

    monkeypatch.setattr(t.cuda, "mem_get_info", unexpected_mem_get_info)
    original_forward = Linear.forward
    with greedy_token_batch._stage_linear_weights_for_greedy_batch(
        model,
        batch_size=3,
        max_total_len=7,
        ragged=True,
        dtype=t.bfloat16,
    ) as staged:
        assert not staged
        assert Linear.forward is original_forward


def test_sdpa_cache_positions_stay_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", False)

    assert greedy_token_batch._cache_position_device(t.device("cuda:3")) == t.device("cpu")
    assert greedy_token_batch._cache_position_device(t.device("cpu")) == t.device("cpu")


def test_fa3_cache_positions_stay_on_the_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", True)

    assert greedy_token_batch._cache_position_device(t.device("cuda:3")) == t.device("cuda:3")


def test_ragged_sdpa_reads_only_host_bounded_cache_in_storage_dtype(monkeypatch) -> None:
    cache_dtype = greedy_token_batch.COMPUTE_DTYPE
    query_dtype = t.float32 if cache_dtype != t.float32 else t.float64
    q = t.randn(2, 1, 4, 8, dtype=query_dtype)
    k = t.randn(2, 1, 2, 8, dtype=query_dtype)
    v = t.randn(2, 1, 2, 8, dtype=query_dtype)
    k_cache = t.zeros(2, 5, 2, 8, dtype=cache_dtype)
    v_cache = t.zeros_like(k_cache)
    cache_seqlens = t.tensor([2, 3], dtype=t.int32)
    captured: dict[str, t.Tensor] = {}

    def fake_sdpa(
        query: t.Tensor,
        key: t.Tensor,
        value: t.Tensor,
        **kwargs,
    ) -> t.Tensor:
        captured.update(query=query, key=key, value=value, mask=kwargs["attn_mask"])
        return t.zeros_like(query)

    monkeypatch.setattr(greedy_token_batch.F, "scaled_dot_product_attention", fake_sdpa)
    cache_positions, batch_indices, masks = greedy_token_batch._build_ragged_sdpa_inputs(
        cache_seqlens,
        cache_end=4,
        window_sizes=[(-1, 0)],
        device=q.device,
    )

    result = greedy_token_batch._ragged_sdpa_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_positions,
        batch_indices,
        masks[(-1, 0)],
        4,
    )

    assert result.dtype == cache_dtype
    assert captured["query"].dtype == cache_dtype
    assert captured["key"].dtype == cache_dtype
    assert captured["value"].dtype == cache_dtype
    assert captured["key"].untyped_storage().data_ptr() == k_cache.untyped_storage().data_ptr()
    assert captured["value"].untyped_storage().data_ptr() == v_cache.untyped_storage().data_ptr()
    assert captured["key"].shape == (2, 2, 4, 8)
    assert captured["value"].shape == (2, 2, 4, 8)
    assert captured["mask"].shape == (2, 1, 1, 4)
    assert captured["mask"][:, 0, 0].tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_ragged_sdpa_routes_supported_cuda_shape_to_triton(monkeypatch) -> None:
    q = t.zeros(2, 1, 4, 8)
    k = t.zeros(2, 1, 2, 8)
    v = t.zeros_like(k)
    k_cache = t.zeros(2, 5, 2, 8)
    v_cache = t.zeros_like(k_cache)
    cache_positions = t.tensor([2, 3], dtype=t.long)
    batch_indices = t.arange(2)
    attn_mask = t.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    sentinel = t.full_like(q, 7)
    captured: dict[str, object] = {}

    def fake_triton(*args: object) -> t.Tensor:
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(
        greedy_token_batch,
        "_should_use_triton_ragged_attention",
        lambda _q, _k_cache: True,
    )
    monkeypatch.setattr(
        greedy_token_batch,
        "triton_ragged_attention_with_kvcache",
        fake_triton,
    )

    result = greedy_token_batch._ragged_sdpa_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_positions,
        batch_indices,
        attn_mask,
        4,
    )

    assert result is sentinel
    assert captured["args"] == (
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_positions,
        attn_mask,
        4,
    )


@pytest.mark.parametrize("cache_end", [0, 6])
def test_ragged_sdpa_rejects_cache_end_outside_storage(cache_end: int) -> None:
    q = t.zeros(1, 1, 1, 2)
    k = t.zeros_like(q)
    v = t.zeros_like(q)
    k_cache = t.zeros(1, 5, 1, 2)
    v_cache = t.zeros_like(k_cache)

    with pytest.raises(ValueError, match="cache end must be"):
        greedy_token_batch._ragged_sdpa_with_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            t.zeros(1, dtype=t.long),
            t.arange(1),
            t.zeros(1, max(cache_end, 0), dtype=t.bool),
            cache_end,
        )


def test_ragged_sdpa_inputs_share_inactive_window_mask() -> None:
    positions, batch_indices, masks = greedy_token_batch._build_ragged_sdpa_inputs(
        t.tensor([2, 4], dtype=t.int32),
        cache_end=5,
        window_sizes=[(-1, 0), (10, 0), (2, 0)],
        device=t.device("cpu"),
    )

    assert positions.dtype == t.long
    assert positions.tolist() == [2, 4]
    assert batch_indices.tolist() == [0, 1]
    assert masks[(-1, 0)] is masks[(10, 0)]
    assert masks[(-1, 0)].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert masks[(2, 0)].tolist() == [
        [True, True, True, False, False],
        [False, False, True, True, True],
    ]


def test_ragged_sdpa_inputs_reject_right_window() -> None:
    with pytest.raises(ValueError, match="right window 0"):
        greedy_token_batch._build_ragged_sdpa_inputs(
            t.tensor([2], dtype=t.int32),
            cache_end=3,
            window_sizes=[(-1, 1)],
            device=t.device("cpu"),
        )


def test_kv_cache_compaction_packs_rows_without_reallocating_storage() -> None:
    cache = KVCache(
        batch_size=4,
        num_heads=1,
        seq_len=3,
        head_dim=2,
        num_layers=2,
        device=t.device("cpu"),
        dtype=t.float32,
    )
    for row in range(4):
        cache.k_cache[:, row].fill_(10 + row)
        cache.v_cache[:, row].fill_(20 + row)
        cache.cache_seqlens[row] = 30 + row
    k_storage = cache.k_cache.untyped_storage().data_ptr()
    v_storage = cache.v_cache.untyped_storage().data_ptr()

    greedy_token_batch._compact_kv_cache_rows(cache, [1, 3])

    assert cache.batch_size == 2
    assert cache.k_cache.shape == (2, 2, 3, 1, 2)
    assert cache.v_cache.shape == (2, 2, 3, 1, 2)
    assert cache.cache_seqlens.tolist() == [31, 33]
    assert t.all(cache.k_cache[:, 0] == 11)
    assert t.all(cache.k_cache[:, 1] == 13)
    assert t.all(cache.v_cache[:, 0] == 21)
    assert t.all(cache.v_cache[:, 1] == 23)
    assert cache.k_cache.untyped_storage().data_ptr() == k_storage
    assert cache.v_cache.untyped_storage().data_ptr() == v_storage


def test_expandable_ragged_cache_compacts_then_grows_within_budget() -> None:
    cache = greedy_token_batch._ExpandableRaggedKVCache(
        batch_size=4,
        num_heads=1,
        seq_len=3,
        final_seq_len=5,
        head_dim=2,
        num_layers=2,
        device=t.device("cpu"),
        dtype=t.float32,
    )
    for layer_idx in range(2):
        k_cache, v_cache = cache.get_layer_cache(layer_idx)
        for row in range(4):
            k_cache[row].fill_(100 * layer_idx + 10 + row)
            v_cache[row].fill_(100 * layer_idx + 20 + row)
    cache.cache_seqlens.copy_(t.tensor([30, 31, 32, 33], dtype=t.int32))

    greedy_token_batch._compact_kv_cache_rows(cache, [1, 3])
    cache.ensure_capacity(4)

    assert cache.batch_size == 2
    assert cache.max_seq_len == 5
    assert cache.cache_seqlens.tolist() == [31, 33]
    for layer_idx in range(2):
        k_cache, v_cache = cache.get_layer_cache(layer_idx)
        assert k_cache.shape == (2, 5, 1, 2)
        assert v_cache.shape == (2, 5, 1, 2)
        assert t.all(k_cache[0, :3] == 100 * layer_idx + 11)
        assert t.all(k_cache[1, :3] == 100 * layer_idx + 13)
        assert t.all(v_cache[0, :3] == 100 * layer_idx + 21)
        assert t.all(v_cache[1, :3] == 100 * layer_idx + 23)


def test_expandable_ragged_cache_rejects_growth_before_mutation() -> None:
    cache = greedy_token_batch._ExpandableRaggedKVCache(
        batch_size=4,
        num_heads=1,
        seq_len=3,
        final_seq_len=5,
        head_dim=2,
        num_layers=2,
        device=t.device("cpu"),
        dtype=t.float32,
    )

    with pytest.raises(
        greedy_token_batch.GreedyBatchCapacityError,
        match="smaller batch",
    ):
        cache.ensure_capacity(4)

    assert cache.max_seq_len == 3
    assert all(layer.shape == (4, 3, 1, 2) for layer in cache.k_layers)
    assert all(layer.shape == (4, 3, 1, 2) for layer in cache.v_layers)


def test_expandable_cache_admission_reserves_shadows_and_headroom(
    monkeypatch,
) -> None:
    model = _tiny_gpt()
    model.get_device = lambda: t.device("cuda")  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", False)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))
    shadow_bytes = greedy_token_batch._staged_linear_shadow_bytes(model)
    bytes_per_token = greedy_token_batch._projected_kv_cache_bytes(
        model,
        batch_size=4,
        max_total_len=1,
        ragged=True,
        dtype=t.bfloat16,
    )
    monkeypatch.setattr(
        t.cuda,
        "mem_get_info",
        lambda _device: (
            shadow_bytes
            + greedy_token_batch._EXPANDABLE_CACHE_HEADROOM_BYTES
            + 9 * bytes_per_token,
            10**12,
        ),
    )

    assert (
        greedy_token_batch._initial_expandable_ragged_cache_len(
            model,
            batch_size=4,
            max_prompt_len=5,
            max_total_len=20,
            dtype=t.bfloat16,
        )
        == 9
    )

    monkeypatch.setattr(
        t.cuda,
        "mem_get_info",
        lambda _device: (
            shadow_bytes
            + greedy_token_batch._EXPANDABLE_CACHE_HEADROOM_BYTES
            + 5 * bytes_per_token,
            10**12,
        ),
    )
    assert (
        greedy_token_batch._initial_expandable_ragged_cache_len(
            model,
            batch_size=4,
            max_prompt_len=5,
            max_total_len=20,
            dtype=t.bfloat16,
        )
        is None
    )


def test_hopper_sdpa_keeps_fixed_width_but_ada_sdpa_compacts(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", False)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (9, 0))

    assert not greedy_token_batch._should_compact_finished_rows(t.device("cuda"))

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))

    assert greedy_token_batch._should_compact_finished_rows(t.device("cuda"))


def test_padded_ragged_prefill_dispatches_only_on_pre_hopper_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", False)
    monkeypatch.setattr(
        greedy_token_batch,
        "triton_ragged_attention_with_kvcache",
        object(),
    )
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))

    assert greedy_token_batch._should_use_padded_ragged_prefill(t.device("cuda"))
    assert greedy_token_batch._should_use_float8_ragged_cache(t.device("cuda"))
    assert not greedy_token_batch._should_use_padded_ragged_prefill(t.device("cpu"))
    assert not greedy_token_batch._should_use_float8_ragged_cache(t.device("cpu"))

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 0))
    assert greedy_token_batch._should_use_padded_ragged_prefill(t.device("cuda"))
    assert not greedy_token_batch._should_use_float8_ragged_cache(t.device("cuda"))

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (9, 0))
    assert not greedy_token_batch._should_use_padded_ragged_prefill(t.device("cuda"))
    assert not greedy_token_batch._should_use_float8_ragged_cache(t.device("cuda"))

    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))
    assert not greedy_token_batch._should_use_padded_ragged_prefill(t.device("cuda"))
    assert not greedy_token_batch._should_use_float8_ragged_cache(t.device("cuda"))


def test_triton_relu_squared_dispatches_only_on_measured_ada(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "triton_relu_squared_", object())
    tensor: Any = SimpleNamespace(
        is_cuda=True,
        is_contiguous=lambda: True,
        dtype=t.bfloat16,
        device=t.device("cuda"),
    )
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))

    assert greedy_token_batch._should_use_triton_relu_squared(tensor)

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 0))
    assert not greedy_token_batch._should_use_triton_relu_squared(tensor)

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (9, 0))
    assert not greedy_token_batch._should_use_triton_relu_squared(tensor)

    tensor.dtype = t.float32
    assert not greedy_token_batch._should_use_triton_relu_squared(tensor)
    tensor.dtype = t.bfloat16
    tensor.is_contiguous = lambda: False
    assert not greedy_token_batch._should_use_triton_relu_squared(tensor)


def test_triton_residual_blend_dispatch_preserves_unsupported_paths(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "triton_residual_blend", object())
    monkeypatch.setattr(greedy_token_batch, "COMPUTE_DTYPE", t.bfloat16)
    x0: Any = SimpleNamespace(
        is_cuda=True, dtype=t.float32, is_contiguous=lambda: True,
        device=t.device("cuda:0"),
    )
    scales: Any = SimpleNamespace(
        device=x0.device, dtype=t.float32, ndim=1, numel=lambda: 2,
    )
    for capability, expected in (((8, 9), True), ((8, 0), False), ((9, 0), False)):
        monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device, value=capability: value)
        assert greedy_token_batch._should_use_triton_residual_blend(
            x0, scales, scales, num_layers=2,
        ) is expected

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))
    unsupported = (
        (x0, "is_cuda", False),
        (x0, "dtype", t.bfloat16),
        (x0, "is_contiguous", lambda: False),
        (scales, "device", t.device("cpu")),
        (scales, "dtype", t.float16),
        (scales, "ndim", 2),
        (scales, "numel", lambda: 1),
    )
    for tensor, attribute, value in unsupported:
        with monkeypatch.context() as context:
            context.setattr(tensor, attribute, value)
            assert not greedy_token_batch._should_use_triton_residual_blend(
                x0, scales, scales, num_layers=2,
            )
    monkeypatch.setattr(greedy_token_batch, "COMPUTE_DTYPE", t.float32)
    assert not greedy_token_batch._should_use_triton_residual_blend(
        x0, scales, scales, num_layers=2,
    )
    monkeypatch.setattr(greedy_token_batch, "COMPUTE_DTYPE", t.bfloat16)
    monkeypatch.setattr(greedy_token_batch, "triton_residual_blend", None)
    assert not greedy_token_batch._should_use_triton_residual_blend(
        x0, scales, scales, num_layers=2,
    )


def test_ragged_decode_routes_live_residual_scales_once_per_layer(monkeypatch) -> None:
    t.manual_seed(197)
    model = _tiny_gpt()
    prompts = [[1, 2], [3, 4, 5]]
    next_ids = t.tensor([[6], [7]])
    calls: list[tuple[int, bool]] = []

    def blend(x, x0, resid_lambdas, x0_lambdas, layer_idx, *, output_dtype):
        assert resid_lambdas is model.resid_lambdas
        assert x0_lambdas is model.x0_lambdas
        calls.append((layer_idx, x is x0))
        return (resid_lambdas[layer_idx] * x + x0_lambdas[layer_idx] * x0).to(output_dtype)

    with t.inference_mode():
        model.resid_lambdas.copy_(t.tensor([0.73, 1.27]))
        model.x0_lambdas.copy_(t.tensor([-0.11, 0.19]))
        reference_cache, _ = greedy_token_batch._prefill_ragged_prompt_batch(
            model, prompts, max_total_len=8, dtype=t.float32,
        )
        candidate_cache, _ = greedy_token_batch._prefill_ragged_prompt_batch(
            model, prompts, max_total_len=8, dtype=t.float32,
        )
        monkeypatch.setattr(greedy_token_batch, "_should_use_triton_residual_blend", lambda *a, **kw: False)
        expected = greedy_token_batch._ragged_forward_one(
            model, next_ids, reference_cache, cache_end=4,
        )
        monkeypatch.setattr(greedy_token_batch, "_should_use_triton_residual_blend", lambda *a, **kw: True)
        monkeypatch.setattr(greedy_token_batch, "triton_residual_blend", blend)
        actual = greedy_token_batch._ragged_forward_one(
            model, next_ids, candidate_cache, cache_end=4,
        )

    assert calls == [(0, True), (1, False)]
    assert t.equal(actual, expected)
    assert isinstance(candidate_cache, KVCache)
    assert isinstance(reference_cache, KVCache)
    for row, length in enumerate((3, 4)):
        assert t.equal(candidate_cache.k_cache[:, row, :length], reference_cache.k_cache[:, row, :length])
        assert t.equal(candidate_cache.v_cache[:, row, :length], reference_cache.v_cache[:, row, :length])
    assert t.equal(candidate_cache.cache_seqlens, reference_cache.cache_seqlens)


def test_triton_qk_rope_norm_dispatches_only_on_measured_ada(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "triton_qk_rope_rms_norm", object())
    q_shape = (2, 1, 13, 128)
    q: Any = SimpleNamespace(
        is_cuda=True,
        ndim=4,
        dtype=t.bfloat16,
        device=t.device("cuda"),
        shape=q_shape,
        size=lambda dim: q_shape[dim],
        stride=lambda dim: (4992, 1664, 128, 1)[dim],
    )
    k = q
    rope_shape = (2, 1, 1, 64)
    rope: Any = SimpleNamespace(
        dtype=t.bfloat16,
        device=t.device("cuda"),
        shape=rope_shape,
        stride=lambda dim: (64, 64, 64, 1)[dim],
    )
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 9))

    assert greedy_token_batch._should_use_triton_qk_rope_norm(q, k, rope, rope)

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (8, 0))
    assert not greedy_token_batch._should_use_triton_qk_rope_norm(q, k, rope, rope)

    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (9, 0))
    assert not greedy_token_batch._should_use_triton_qk_rope_norm(q, k, rope, rope)

    q.dtype = t.float32
    assert not greedy_token_batch._should_use_triton_qk_rope_norm(q, k, rope, rope)


def test_padded_prefill_attention_matches_windowed_gqa_sdpa() -> None:
    t.manual_seed(123)
    q = t.randn(2, 4, 6, 8)
    k = t.randn(2, 2, 6, 8)
    v = t.randn_like(k)
    positions = t.arange(6)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    mask = (key_positions <= query_positions) & (query_positions - key_positions <= 2)

    expected = greedy_token_batch.F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        enable_gqa=True,
    )
    actual = greedy_token_batch._padded_prefill_attention(q, k, v, (2, 0))

    t.testing.assert_close(actual, expected)


def test_equal_length_batch_stops_rows_independently() -> None:
    model = _Model(token_plan=[[4, 5], [9, 6], [7, 0], [7, 7]])
    completions = generate_greedy_token_batch(
        model,
        _Tokenizer(),
        [[1, 2, 3], [2, 3, 4]],
        max_new_tokens=4,
    )

    assert completions == [[4], [5, 6]]
    assert model.calls[0].tolist() == [[1, 2, 3], [2, 3, 4]]
    assert [tuple(call.shape) for call in model.calls] == [
        (2, 3),
        (2, 1),
        (2, 1),
        (2, 1),
    ]


def test_finished_rows_are_removed_from_followup_sdpa_steps(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", False)
    model = _Model(token_plan=[[9, 5], [6], [9]])

    completions = generate_greedy_token_batch(
        model,
        _Tokenizer(),
        [[1], [2]],
        max_new_tokens=4,
        enable_python_tools=True,
    )

    assert completions == [[], [5, 6]]
    assert [tuple(call.shape) for call in model.calls] == [
        (2, 1),
        (1, 1),
        (1, 1),
    ]


def test_fa3_keeps_the_original_batch_width_after_rows_finish(monkeypatch) -> None:
    monkeypatch.setattr(greedy_token_batch, "HAS_FA3", True)
    model = _Model(token_plan=[[9, 5], [7, 6], [7, 9]])

    completions = generate_greedy_token_batch(
        model,
        _Tokenizer(),
        [[1], [2]],
        max_new_tokens=4,
        enable_python_tools=True,
    )

    assert completions == [[], [5, 6]]
    assert [tuple(call.shape) for call in model.calls] == [
        (2, 1),
        (2, 1),
        (2, 1),
    ]


def test_ragged_forward_matches_sequential_rows(monkeypatch) -> None:
    t.manual_seed(123)
    model = _tiny_gpt()
    prompts = [list(range(1, 19)), list(range(20, 40))]
    cache_kwargs = {
        "num_heads": 2,
        "seq_len": 32,
        "head_dim": 8,
        "num_layers": 2,
        "device": t.device("cpu"),
        "dtype": t.float32,
    }
    batch_cache = KVCache(batch_size=2, **cache_kwargs)
    next_ids = t.tensor([[9], [10]])
    expected_rows: list[t.Tensor] = []
    build_calls = 0
    original_build = greedy_token_batch._build_ragged_sdpa_inputs

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(greedy_token_batch, "_build_ragged_sdpa_inputs", counted_build)

    with t.inference_mode():
        for row, prompt in enumerate(prompts):
            reference_cache = KVCache(batch_size=1, **cache_kwargs)
            source_cache = KVCache(batch_size=1, **cache_kwargs)
            prompt_tensor = t.tensor([prompt])
            reference_prefill = model(prompt_tensor, kv_cache=reference_cache)
            source_prefill = model(prompt_tensor, kv_cache=source_cache)
            assert t.equal(reference_prefill, source_prefill)
            batch_cache.k_cache[:, row, : len(prompt)].copy_(
                source_cache.k_cache[:, 0, : len(prompt)]
            )
            batch_cache.v_cache[:, row, : len(prompt)].copy_(
                source_cache.v_cache[:, 0, : len(prompt)]
            )
            batch_cache.cache_seqlens[row] = len(prompt)
            expected_rows.append(model(next_ids[row : row + 1], kv_cache=reference_cache)[0])

        expected = t.stack(expected_rows)
        actual = greedy_token_batch._ragged_forward_one(
            model,
            next_ids,
            batch_cache,
            cache_end=21,
        )

    assert t.equal(actual, expected)
    assert build_calls == 1
    assert batch_cache.cache_seqlens.tolist() == [19, 21]


def test_mixed_prompt_decode_grows_host_bounded_cache_end(monkeypatch) -> None:
    captured_cache_ends: list[int] = []

    def fake_prefill(
        _model,
        prompt_tokens: list[list[int]],
        *,
        max_total_len: int,
        dtype: t.dtype,
        initial_cache_len: int | None,
    ) -> tuple[object, t.Tensor]:
        assert max_total_len == 8
        assert dtype == t.float32
        assert initial_cache_len is None
        return object(), t.zeros((len(prompt_tokens), 64))

    def fake_ragged_forward(
        _model,
        idx: t.Tensor,
        _cache: object,
        *,
        cache_end: int,
    ) -> t.Tensor:
        captured_cache_ends.append(cache_end)
        return t.zeros((idx.size(0), idx.size(1), 64))

    monkeypatch.setattr(
        greedy_token_batch,
        "_prefill_ragged_prompt_batch",
        fake_prefill,
    )
    monkeypatch.setattr(
        greedy_token_batch,
        "_ragged_forward_one",
        fake_ragged_forward,
    )

    completions = generate_greedy_token_batch(
        _Model(),
        _RaggedTokenizer(),
        [[1, 2], [3, 4, 5, 6]],
        max_new_tokens=4,
    )

    assert completions == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert captured_cache_ends == [5, 6, 7]


def test_ragged_prefill_batches_equal_lengths_within_source_budget() -> None:
    model = _Model(token_plan=[[1, 1], [1]])

    cache, logits = greedy_token_batch._prefill_ragged_prompt_batch(
        model,
        [[1, 2], [3, 4], [5, 6, 7]],
        max_total_len=8,
        dtype=t.float32,
    )

    assert [tuple(call.shape) for call in model.calls] == [(2, 2), (1, 3)]
    assert cache.cache_seqlens.tolist() == [2, 2, 3]
    assert logits.shape == (3, 16)


def test_ragged_prefill_splits_group_at_single_source_row_token_budget() -> None:
    model = _Model(token_plan=[[1], [1], [1]])

    cache, logits = greedy_token_batch._prefill_ragged_prompt_batch(
        model,
        [[1, 2], [3, 4], [5, 6]],
        max_total_len=3,
        dtype=t.float32,
    )

    assert [tuple(call.shape) for call in model.calls] == [(1, 2), (1, 2), (1, 2)]
    assert cache.cache_seqlens.tolist() == [2, 2, 2]
    assert logits.shape == (3, 16)


def test_padded_ragged_prefill_matches_exact_prefix_and_decode() -> None:
    t.manual_seed(123)
    model = _tiny_gpt()
    prompts = [list(range(1, 9)), list(range(10, 20))]

    with t.inference_mode():
        reference_cache, reference_logits = (
            greedy_token_batch._prefill_ragged_prompt_batch(
                model,
                prompts,
                max_total_len=16,
                dtype=t.float32,
            )
        )
        padded_cache, padded_logits = (
            greedy_token_batch._prefill_padded_ragged_prompt_batch(
                model,
                prompts,
                max_total_len=16,
                dtype=t.float32,
                initial_cache_len=12,
            )
        )

        t.testing.assert_close(padded_logits, reference_logits, rtol=1e-5, atol=1e-5)
        assert isinstance(padded_cache, greedy_token_batch._ExpandableRaggedKVCache)
        assert padded_cache.cache_seqlens.tolist() == [8, 10]
        for layer_idx in range(model.config.n_layer):
            reference_k, reference_v = reference_cache.get_layer_cache(layer_idx)
            padded_k, padded_v = padded_cache.get_layer_cache(layer_idx)
            for row, prompt in enumerate(prompts):
                prompt_len = len(prompt)
                t.testing.assert_close(
                    padded_k[row, :prompt_len],
                    reference_k[row, :prompt_len],
                    rtol=1e-5,
                    atol=1e-5,
                )
                t.testing.assert_close(
                    padded_v[row, :prompt_len],
                    reference_v[row, :prompt_len],
                    rtol=1e-5,
                    atol=1e-5,
                )

        next_ids = reference_logits.argmax(dim=-1, keepdim=True)
        reference_next = greedy_token_batch._ragged_forward_one(
            model,
            next_ids,
            reference_cache,
            cache_end=11,
        )
        padded_next = greedy_token_batch._ragged_forward_one(
            model,
            next_ids,
            padded_cache,
            cache_end=11,
        )

    t.testing.assert_close(padded_next, reference_next, rtol=1e-5, atol=1e-5)


def test_mixed_prompt_lengths_generate_without_padding() -> None:
    model = _tiny_gpt()
    with t.no_grad():
        model.lm_head.weight.zero_()

    completions = generate_greedy_token_batch(
        model,
        _RaggedTokenizer(),
        [[1, 2], [3, 4, 5]],
        max_new_tokens=4,
    )

    assert completions == [[0, 0, 0, 0], [0, 0, 0, 0]]


def test_python_start_requests_exact_sequential_restart() -> None:
    model = _Model(token_plan=[[8, 4]])
    completions = generate_greedy_token_batch(
        model,
        _Tokenizer(),
        [[1], [2]],
        max_new_tokens=4,
    )

    assert completions is None
    assert len(model.calls) == 4


def test_python_tool_batch_forces_calculator_output_tokens() -> None:
    model = _Model(
        token_plan=[
            [8, 8],
            [4, 4],
            [10, 10],
            [7, 7],
            [7, 7],
            [7, 7],
            [9, 9],
        ]
    )

    completions = generate_greedy_token_batch(
        model,
        _Tokenizer(),
        [[1], [2]],
        max_new_tokens=8,
        enable_python_tools=True,
    )

    assert completions == [[8, 4, 10, 11, 6, 12], [8, 4, 10, 11, 6, 12]]
    assert [call.tolist() for call in model.calls[1:]] == [
        [[8], [8]],
        [[4], [4]],
        [[10], [10]],
        [[11], [11]],
        [[6], [6]],
        [[12], [12]],
    ]


def test_non_finite_logits_still_fail_before_results_are_returned() -> None:
    model = _Model(token_plan=[[4]], non_finite_step=2)

    with pytest.raises(FloatingPointError, match="non-finite logits"):
        generate_greedy_token_batch(
            model,
            _Tokenizer(),
            [[1]],
            max_new_tokens=4,
        )


def test_delayed_checks_ignore_logits_after_a_terminal_token() -> None:
    model = _Model(token_plan=[[9]], non_finite_step=2)

    assert generate_greedy_token_batch(
        model,
        _Tokenizer(),
        [[1]],
        max_new_tokens=4,
    ) == [[]]


def test_zero_budget_returns_one_empty_completion_per_prompt() -> None:
    assert generate_greedy_token_batch(
        _Model(),
        _Tokenizer(),
        [[1], [2]],
        max_new_tokens=0,
    ) == [[], []]


def test_rejects_padded_batch_beyond_context_window() -> None:
    with pytest.raises(ValueError, match="exceeds context window"):
        generate_greedy_token_batch(
            _Model(sequence_len=4),
            _Tokenizer(),
            [[1, 2, 3], [2, 3, 4]],
            max_new_tokens=2,
        )


def test_rejects_target_aware_encoder_decoder() -> None:
    model = _Model()
    model.is_target_aware_encdec = True

    with pytest.raises(ValueError, match="decoder-only"):
        generate_greedy_token_batch(
            model,
            _Tokenizer(),
            [[1], [2]],
            max_new_tokens=1,
        )
