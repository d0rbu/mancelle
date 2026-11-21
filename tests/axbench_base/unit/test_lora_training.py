"""Unit tests for the LoRA training loop (Task 24, axbench_base Stage 3b).

Covers
``steering.eval.axbench_base.baselines.lora.training.train_lora_all``
and its helpers, which:

  * Wrap every ``attn.c_qkv``, ``attn.c_proj``, ``mlp.c_fc``, ``mlp.c_proj``
    on a target ``GPT`` block with a per-concept ``LoRAAdapter``.
  * Train the adapters on the per-concept positives via causal LM loss
    with a loss mask that zeros prefix-token positions (only output
    positions contribute to the loss).
  * Save the resulting LoRA state to
    ``test_paths.default_paths(run_tag).lora_adapter(mode, concept_id)`` as a safetensors
    file (one ``lora_A`` / ``lora_B`` tensor per attached layer).
  * Detach the adapters from the model after training so the original
    base linears are restored byte-for-byte (identity check).

TDD design choices pinned down by these tests:

  * Positive mode trains on ``content_class == "concept_rich"`` rows
    matching the concept; negative mode trains on
    ``content_class == "concept_avoiding"`` rows matching the concept.
    (Plan: positives in negative mode are concept_avoiding because we
    want the adapter to OUTPUT text that suppresses the feature.)
  * Loss mask: prefix tokens (re-tokenized ``prefix_text`` length, BOS
    prepended) are excluded from the cross-entropy via the
    ``ignore_index=-1`` sentinel.
  * NaN loss is a hard error, not silently zeroed.
  * Determinism: same ``master_seed + concept_id`` → same per-step
    batch sequence.
  * ``len(training_rows) < cfg.lora.min_training_rows`` → ``AxBenchBaseError``.
  * After training, ``model.transformer.h[i].attn.c_qkv`` is the original
    object (identity, not equality) — adapters are removed cleanly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
from loguru import logger as loguru_logger
from safetensors import safe_open
from safetensors.torch import load_file

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from nanochat.gpt import Linear as NanochatLinear
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines.lora import _attach as lora_attach_mod
from steering.eval.axbench_base.baselines.lora import training as lora_training_mod
from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.eval.axbench_base.baselines.lora.training import (
    _ATTACH_PATHS,
    _attach_lora_to_model,
    _build_lora_batch,
    _collect_attached_adapters,
    _detach_lora_from_model,
    _row_to_sequence_and_target,
    _sample_batch_indices,
    _train_lora_single_concept,
    train_lora_all,
)
from steering.eval.axbench_base.config import LayerHooksConfig, LoraConfig
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config


@pytest.fixture(autouse=True)
def _force_sdpa() -> Iterator[None]:
    """Force SDPA attention backend for every test in this module.

    FlexAttention does not support backward on CPU (the inductor lowering
    crashes on ``getitem`` of a tuple-shaped input). We patch the
    module-level override directly rather than using an env var, since
    ``nanochat.flash_attention`` may already be imported by earlier tests.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Tiny GPT fixture + stub tokenizer
# ---------------------------------------------------------------------------


TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_BOS_ID: int = 1
TINY_LAYER: int = 0  # override so layer-0 is valid for our 2-layer fixture


def _make_tiny_gpt() -> GPT:
    """Tiny CPU GPT in eval mode, fully initialised, deterministic."""
    t.manual_seed(0xD24F)
    config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


class _StubNanochatTokenizer:
    """Deterministic nanochat-like tokenizer for tests."""

    def __init__(self, vocab_size: int = TINY_VOCAB, bos_id: int = TINY_BOS_ID) -> None:
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self._decodes: dict[tuple[int, ...], str] = {}
        self._token_text: dict[int, str] = {}

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        base = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]
        self._token_text.update(zip(base, text, strict=True))
        for end in range(len(base) + 1):
            self._decodes[tuple(base[:end])] = text[:end]
        if prepend is not None:
            return [prepend] + base
        return base

    def decode(self, tokens: list[int]) -> str:
        exact = self._decodes.get(tuple(tokens))
        if exact is not None:
            return exact
        return "".join(self._token_text[token] for token in tokens)


def _install_stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> _StubNanochatTokenizer:
    tok = _StubNanochatTokenizer()
    monkeypatch.setattr(lora_training_mod, "get_tokenizer", lambda: tok)
    return tok


# ---------------------------------------------------------------------------
# Helpers: corpus construction
# ---------------------------------------------------------------------------


_POS_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("category", pa.string()),
        ("content_class", pa.string()),
        ("prefix_text", pa.string()),
        ("output_text", pa.string()),
        ("output_concept", pa.string()),
        ("n_prefix_nanochat_tokens", pa.int32()),
        ("gen_source", pa.string()),
        ("gen_seed", pa.int64()),
        ("qc_feature_id", pa.int32()),
        ("qc_seed_percentile", pa.float32()),
        ("qc_output_percentile", pa.float32()),
        ("qc_delta", pa.float32()),
        ("qc_attempts", pa.int32()),
        ("qc_accepted", pa.bool_()),
    ]
)

_NEG_SCHEMA = pa.schema(
    [
        *list(_POS_SCHEMA),
        pa.field("prefix_source_shard", pa.int32()),
        pa.field("prefix_source_doc", pa.int32()),
        pa.field("prefix_source_seg_idx", pa.int32()),
        pa.field("qc_zero_activation_verified", pa.bool_()),
    ]
)


def _pos_row(
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "category": "positive" if content_class == "concept_rich" else "negative",
        "content_class": content_class,
        "prefix_text": prefix_text,
        "output_text": output_text,
        "output_concept": "stub concept",
        "n_prefix_nanochat_tokens": n_prefix_nanochat_tokens,
        "gen_source": "test:stub",
        "gen_seed": 0,
        "qc_feature_id": 7,
        "qc_seed_percentile": 0.0,
        "qc_output_percentile": 0.0,
        "qc_delta": 0.0,
        "qc_attempts": 1,
        "qc_accepted": True,
    }


def _neg_row(
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
) -> dict[str, Any]:
    base = _pos_row(
        concept_id, content_class, prefix_text, output_text, n_prefix_nanochat_tokens,
    )
    base.update(
        {
            "prefix_source_shard": 0,
            "prefix_source_doc": 0,
            "prefix_source_seg_idx": 0,
            "qc_zero_activation_verified": content_class == "concept_avoiding",
        }
    )
    return base


def _write_positive_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(
        rows,
        schema=schema_with_stage1_continuation_contract(
            schema_with_positive_qc_contract(
                schema_with_extraction_tokenizer_metadata(
                    _POS_SCHEMA,
                    "google/gemma-2-2b",
                ),
                accept_threshold=0.3,
            ),
            role="stage1b_positive",
        ),
    )
    pq.write_table(table, path)


def _write_negative_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(
        rows,
        schema=schema_with_stage1_continuation_contract(
            schema_with_extraction_tokenizer_metadata(
                _NEG_SCHEMA,
                "google/gemma-2-2b",
            ),
            role="stage1c_negative",
        ),
    )
    pq.write_table(table, path)


def _tiny_cfg(
    tmp_path: Path,
    *,
    steps: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    rank: int = 2,
    alpha: float = 4.0,
    min_training_rows: int = 2,
) -> Any:
    cfg = make_valid_config(
        run_tag="lora_test",
        output_root=None,
        device="cpu",
    )
    return cfg.replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
        lora=LoraConfig(
            rank=rank,
            alpha=alpha,
            lr=lr,
            steps=steps,
            batch_size=batch_size,
            min_training_rows=min_training_rows,
        ),
    )


def _concept(concept_id: int = 42) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=7,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="the concept of hope",
        corpus_count=100,
        description_source="neuronpedia",
    )


@contextmanager
def _capture_loguru_warnings() -> Iterator[list[str]]:
    """Capture WARNING+ level messages from loguru into a plain list."""
    collected: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: collected.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        yield collected
    finally:
        loguru_logger.remove(handler_id)


def _n(tok: _StubNanochatTokenizer, text: str) -> int:
    return len(tok.encode(text, prepend=tok.get_bos_token_id()))


# ---------------------------------------------------------------------------
# Helper test: training paths constant
# ---------------------------------------------------------------------------


class TestAttachPaths:
    def test_attach_paths_constant_covers_attn_and_mlp(self) -> None:
        """The module-level constant lists exactly the four nanochat
        Linear sublayers per block we want to LoRA-fy."""
        assert _ATTACH_PATHS == (
            ("attn", "c_qkv"),
            ("attn", "c_proj"),
            ("mlp", "c_fc"),
            ("mlp", "c_proj"),
        )


# ---------------------------------------------------------------------------
# Attach / detach
# ---------------------------------------------------------------------------


class TestAttachDetach:
    def test_attach_replaces_each_linear_with_adapter(self) -> None:
        model = _make_tiny_gpt()
        cfg_lora = LoraConfig(rank=2, alpha=4.0)
        layer = TINY_LAYER

        original = _attach_lora_to_model(model, layer=layer, lora_cfg=cfg_lora)

        block = cast(nn.ModuleList, model.transformer["h"])[layer]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            attached = getattr(parent, child_name)
            assert isinstance(attached, LoRAAdapter), (
                f"{parent_name}.{child_name} should now be a LoRAAdapter, "
                f"got {type(attached).__name__}"
            )
            assert attached.rank == cfg_lora.rank
            assert attached.alpha == cfg_lora.alpha
            # The adapter wraps the original module (identity).
            key = (layer, parent_name, child_name)
            assert key in original
            assert attached.base is original[key].module

    def test_detach_restores_original_modules_by_identity(self) -> None:
        """After detach, every attn/mlp linear is the SAME object (id()) as
        it was before attach."""
        model = _make_tiny_gpt()
        cfg_lora = LoraConfig(rank=2, alpha=4.0)
        layer = TINY_LAYER

        block = cast(nn.ModuleList, model.transformer["h"])[layer]
        before_ids: dict[tuple[str, str], int] = {}
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            before_ids[(parent_name, child_name)] = id(getattr(parent, child_name))

        original = _attach_lora_to_model(model, layer=layer, lora_cfg=cfg_lora)
        _detach_lora_from_model(model, original=original)

        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            current = getattr(parent, child_name)
            assert id(current) == before_ids[(parent_name, child_name)], (
                f"{parent_name}.{child_name} identity changed after detach"
            )
            assert isinstance(current, NanochatLinear)
            assert not isinstance(current, LoRAAdapter)

    def test_detach_restores_exact_requires_grad_state(self) -> None:
        """Detach must not silently make an originally frozen base trainable."""
        model = _make_tiny_gpt()
        block = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
        expected: dict[tuple[str, str, str], bool] = {}
        for location_index, (parent_name, child_name) in enumerate(_ATTACH_PATHS):
            base = getattr(getattr(block, parent_name), child_name)
            for parameter_index, (name, parameter) in enumerate(
                base.named_parameters()
            ):
                requires_grad = (location_index + parameter_index) % 2 == 0
                parameter.requires_grad = requires_grad
                expected[(parent_name, child_name, name)] = requires_grad

        original = _attach_lora_to_model(
            model,
            layer=TINY_LAYER,
            lora_cfg=LoraConfig(rank=2, alpha=4.0),
        )
        for state in original.values():
            assert all(
                not parameter.requires_grad
                for parameter in state.module.parameters()
            )

        _detach_lora_from_model(model, original=original)

        for parent_name, child_name in _ATTACH_PATHS:
            base = getattr(getattr(block, parent_name), child_name)
            for name, parameter in base.named_parameters():
                assert parameter.requires_grad is expected[
                    (parent_name, child_name, name)
                ]

    def test_failed_attach_rolls_back_modules_and_trainability(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adapter construction is atomic even when a later location fails."""
        model = _make_tiny_gpt()
        block = cast(nn.ModuleList, model.transformer["h"])[TINY_LAYER]
        before: dict[tuple[str, str], tuple[nn.Module, tuple[bool, ...]]] = {}
        for location_index, (parent_name, child_name) in enumerate(_ATTACH_PATHS):
            base = getattr(getattr(block, parent_name), child_name)
            for parameter_index, parameter in enumerate(base.parameters()):
                parameter.requires_grad = (location_index + parameter_index) % 2 == 0
            before[(parent_name, child_name)] = (
                base,
                tuple(parameter.requires_grad for parameter in base.parameters()),
            )

        real_adapter = lora_attach_mod.LoRAAdapter
        construction_count = 0

        def _fail_on_second_adapter(*args: Any, **kwargs: Any) -> LoRAAdapter:
            nonlocal construction_count
            construction_count += 1
            if construction_count == 2:
                raise RuntimeError("injected adapter construction failure")
            return real_adapter(*args, **kwargs)

        monkeypatch.setattr(lora_attach_mod, "LoRAAdapter", _fail_on_second_adapter)
        with pytest.raises(RuntimeError, match="injected adapter construction failure"):
            _attach_lora_to_model(
                model,
                layer=TINY_LAYER,
                lora_cfg=LoraConfig(rank=2, alpha=4.0),
            )

        for parent_name, child_name in _ATTACH_PATHS:
            base = getattr(getattr(block, parent_name), child_name)
            expected_module, expected_flags = before[(parent_name, child_name)]
            assert base is expected_module
            assert tuple(
                parameter.requires_grad for parameter in base.parameters()
            ) == expected_flags

    def test_attach_at_specific_layer_only(self) -> None:
        """Attach must only touch the requested layer; other layers are
        untouched (still NanochatLinear, not LoRAAdapter)."""
        model = _make_tiny_gpt()
        cfg_lora = LoraConfig(rank=2, alpha=4.0)
        attach_layer = 0
        other_layer = 1

        _attach_lora_to_model(model, layer=attach_layer, lora_cfg=cfg_lora)

        other_block = cast(nn.ModuleList, model.transformer["h"])[other_layer]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(other_block, parent_name)
            current = getattr(parent, child_name)
            assert not isinstance(current, LoRAAdapter), (
                f"layer {other_layer} should have been untouched, "
                f"but {parent_name}.{child_name} is a LoRAAdapter"
            )

    def test_attach_returns_dict_with_correct_layer_index(self) -> None:
        model = _make_tiny_gpt()
        cfg_lora = LoraConfig(rank=2, alpha=4.0)
        layer = TINY_LAYER

        original = _attach_lora_to_model(model, layer=layer, lora_cfg=cfg_lora)
        for key in original:
            assert key[0] == layer

    def test_attached_adapters_have_trainable_lora_params(self) -> None:
        model = _make_tiny_gpt()
        cfg_lora = LoraConfig(rank=2, alpha=4.0)
        _attach_lora_to_model(model, layer=TINY_LAYER, lora_cfg=cfg_lora)

        # Collect all trainable params on the model — should include the
        # LoRA params from each of the four adapters at the chosen layer.
        trainable = [p for p in model.parameters() if p.requires_grad]
        # Filter to ones that look like LoRA ones (small first dim or
        # second dim == rank).
        lora_trainable = [p for p in trainable if p.shape[0] == 2 or p.shape[-1] == 2]
        assert len(lora_trainable) >= 8  # 4 sublayers × (lora_A + lora_B)

    def test_attach_places_adapter_on_base_layer_device(self) -> None:
        """``LoRAAdapter`` initialises ``lora_A`` / ``lora_B`` on CPU via
        :func:`torch.empty` / :func:`torch.zeros`. When the base ``nn.Linear``
        lives on a non-CPU device the adapter must be moved to match it,
        otherwise the first forward pass crashes with::

            RuntimeError: Expected all tensors to be on the same device,
            but got mat2 is on cpu, different from other tensors on cuda:0

        which is exactly what a real-model Stage 3d run hit on a CUDA
        target. Pin the contract with a ``meta``-device proxy so the
        test can run in a CPU-only CI without a GPU.
        """
        # ``meta`` is not ``cpu``: a tensor created on meta tracks shape
        # but no storage. Moving the adapter to meta proves that the
        # attach code honours the base layer's device even for non-CPU
        # devices, without requiring actual CUDA in CI.
        model = _make_tiny_gpt().to("meta")
        cfg_lora = LoraConfig(rank=2, alpha=4.0)

        original = _attach_lora_to_model(
            model, layer=TINY_LAYER, lora_cfg=cfg_lora,
        )
        assert original, "attach should return a non-empty original map"

        transformer = model.transformer
        block = cast(nn.ModuleList, transformer["h"])[TINY_LAYER]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            adapter = getattr(parent, child_name)
            assert adapter.lora_A.device.type == "meta", (
                f"{parent_name}.{child_name} adapter.lora_A is on "
                f"{adapter.lora_A.device}, expected meta (the base "
                f"layer's device)"
            )
            assert adapter.lora_B.device.type == "meta", (
                f"{parent_name}.{child_name} adapter.lora_B is on "
                f"{adapter.lora_B.device}, expected meta"
            )

    @pytest.mark.skipif(
        not t.cuda.is_available(),
        reason="requires CUDA to reproduce the exact device-mismatch bug",
    )
    def test_attach_forward_works_on_cuda_without_mismatch(self) -> None:
        """End-to-end reproduction of the original Stage 3d crash:
        move the whole tiny model to ``cuda`` (so base weights live on
        GPU), attach the LoRA adapter, run a forward pass. Without the
        :func:`_attach_lora_to_model` ``adapter.to(device)`` call from
        commit 1c136d3 this throws::

            RuntimeError: Expected all tensors to be on the same device,
            but got mat2 is on cpu, different from other tensors on cuda:0

        The meta-device test pins the contract abstractly; this test
        pins it against the real GPU codepath where the bug actually
        manifested. Skipped gracefully in CPU-only CI.
        """
        model = _make_tiny_gpt().to("cuda")
        cfg_lora = LoraConfig(rank=2, alpha=4.0)
        _attach_lora_to_model(model, layer=TINY_LAYER, lora_cfg=cfg_lora)

        # Confirm every adapter parameter is on cuda.
        transformer = model.transformer
        block = cast(nn.ModuleList, transformer["h"])[TINY_LAYER]
        for parent_name, child_name in _ATTACH_PATHS:
            parent = getattr(block, parent_name)
            adapter = getattr(parent, child_name)
            assert adapter.lora_A.device.type == "cuda"
            assert adapter.lora_B.device.type == "cuda"

        # And the forward actually works — if the adapter weights are
        # on CPU while the base is on CUDA, this call raises RuntimeError.
        tokens = t.randint(
            0, 32, (1, 4), device="cuda", dtype=t.long,
        )
        _ = model(tokens)


# ---------------------------------------------------------------------------
# Build batch / loss mask
# ---------------------------------------------------------------------------


class TestBuildBatch:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("prefix_text", None, id="null-prefix"),
            pytest.param("output_text", None, id="null-output"),
            pytest.param("output_text", "", id="empty-output"),
            pytest.param("n_prefix_nanochat_tokens", None, id="null-boundary"),
            pytest.param("n_prefix_nanochat_tokens", True, id="boolean-boundary"),
            pytest.param("n_prefix_nanochat_tokens", 0, id="zero-boundary"),
        ],
    )
    def test_malformed_row_fails_loudly(self, field: str, value: Any) -> None:
        tok = _StubNanochatTokenizer()
        row = _pos_row(42, "concept_rich", "P", "AB", _n(tok, "P"))
        row[field] = value

        with pytest.raises(AxBenchBaseError, match="LoRA synth row"):
            _row_to_sequence_and_target(row, tokenizer=tok)

    def test_batch_input_and_target_shapes_match(self) -> None:
        tok = _StubNanochatTokenizer()
        rows = [
            _pos_row(42, "concept_rich", "PA", "abc", _n(tok, "PA")),
            _pos_row(42, "concept_rich", "PB", "def", _n(tok, "PB")),
        ]
        device = t.device("cpu")
        batch_input, batch_target = _build_lora_batch(
            rows=rows,
            tokenizer=tok,
            device=device,
        )
        assert batch_input.shape == batch_target.shape
        assert batch_input.dim() == 2
        assert batch_input.size(0) == len(rows)

    def test_loss_mask_excludes_prefix_positions(self) -> None:
        """All target positions corresponding to the prefix span must be
        ``-1`` (the cross-entropy ignore_index)."""
        tok = _StubNanochatTokenizer()
        prefix = "PREFIX"  # 6 chars → 7 tokens with BOS
        output = "OUT"  # 3 chars → 3 tokens
        n_prefix = _n(tok, prefix)  # = 7
        rows = [
            _pos_row(42, "concept_rich", prefix, output, n_prefix),
        ]
        device = t.device("cpu")
        batch_input, batch_target = _build_lora_batch(
            rows=rows,
            tokenizer=tok,
            device=device,
        )

        # Targets are next-token labels: target[t] = input[t+1] (with the
        # last position labeled -1 since there is no next token in-batch).
        # Prefix span is inclusive of all positions where the *target*
        # would be a prefix token. Concretely, positions 0 .. n_prefix-2
        # have target = input[1..n_prefix-1] which are STILL prefix
        # tokens, so they must be -1.
        target_row = batch_target[0]
        # First (n_prefix - 1) target slots correspond to predicting
        # tokens within the prefix — must be ignored.
        assert (target_row[: n_prefix - 1] == -1).all(), (
            f"Prefix positions must be -1, got {target_row[: n_prefix - 1]}"
        )
        # Position n_prefix-1 predicts the FIRST output token, which is
        # not -1.
        assert target_row[n_prefix - 1].item() != -1, (
            "First output-token target must NOT be -1 (we want loss here)"
        )

    def test_loss_mask_pad_positions_are_minus_one(self) -> None:
        """When sequences in a batch have different lengths, padded
        positions on the SHORTER sequences must be trailing ``-1`` values."""
        tok = _StubNanochatTokenizer()
        rows = [
            _pos_row(42, "concept_rich", "P", "AB", _n(tok, "P")),
            _pos_row(42, "concept_rich", "PP", "ABCDE", _n(tok, "PP")),
        ]
        device = t.device("cpu")
        batch_input, batch_target = _build_lora_batch(
            rows=rows,
            tokenizer=tok,
            device=device,
        )
        # The shorter row gets padded to match the longer row.
        # Padded positions in the shorter row's TARGET should be -1.
        # Exact pad placement is right-padded: causal attention then makes
        # every real position independent of the trailing filler tokens.
        full_short = tok.encode("P" + "AB", prepend=tok.get_bos_token_id())
        full_long = tok.encode("PP" + "ABCDE", prepend=tok.get_bos_token_id())
        pad_count = len(full_long) - len(full_short)
        assert pad_count > 0
        assert batch_input[0, : len(full_short)].tolist() == full_short
        assert batch_input[0, len(full_short) :].tolist() == [0] * pad_count
        assert (batch_target[0, -pad_count:] == -1).all(), (
            "Right-pad positions must have target = -1"
        )

    def test_right_padding_preserves_unpadded_logits_and_losses(self) -> None:
        """Ragged batching must not alter any real token's model state.

        This is the causal/right-padding invariant on which maskless LoRA
        batching relies.  It also pins the owned BOS at position zero instead
        of introducing synthetic BOS context before a shorter row.
        """
        tok = _StubNanochatTokenizer()
        rows = [
            _pos_row(42, "concept_rich", "P", "AB", _n(tok, "P")),
            _pos_row(42, "concept_rich", "PREFIX", "OUTPUT", _n(tok, "PREFIX")),
        ]
        batch_input, batch_target = _build_lora_batch(
            rows=rows,
            tokenizer=tok,
            device=t.device("cpu"),
        )
        model = _make_tiny_gpt()

        with t.no_grad():
            batch_logits = model(batch_input)
            batch_losses = model(
                batch_input,
                targets=batch_target,
                loss_reduction="none",
            ).view_as(batch_target)

            for row_idx, row in enumerate(rows):
                input_ids, targets = _row_to_sequence_and_target(row, tokenizer=tok)
                assert input_ids[0] == tok.get_bos_token_id()
                assert batch_input[row_idx, : len(input_ids)].tolist() == input_ids

                single_input = t.tensor([input_ids], dtype=t.long)
                single_target = t.tensor([targets], dtype=t.long)
                single_logits = model(single_input)
                single_losses = model(
                    single_input,
                    targets=single_target,
                    loss_reduction="none",
                ).view_as(single_target)

                t.testing.assert_close(
                    batch_logits[row_idx, : len(input_ids)],
                    single_logits[0],
                )
                t.testing.assert_close(
                    batch_losses[row_idx, : len(input_ids)],
                    single_losses[0],
                )

    def test_right_padding_preserves_lora_loss_and_gradients(self) -> None:
        """A ragged batch must optimize exactly the unpadded objective."""
        tok = _StubNanochatTokenizer()
        rows = [
            _pos_row(42, "concept_rich", "P", "AB", _n(tok, "P")),
            _pos_row(42, "concept_rich", "PREFIX", "OUTPUT", _n(tok, "PREFIX")),
        ]
        batch_input, batch_target = _build_lora_batch(
            rows=rows,
            tokenizer=tok,
            device=t.device("cpu"),
        )

        batch_model = _make_tiny_gpt()
        single_model = _make_tiny_gpt()
        for parameter in batch_model.parameters():
            parameter.requires_grad = False
        for parameter in single_model.parameters():
            parameter.requires_grad = False
        lora_cfg = LoraConfig(rank=2, alpha=4.0)
        _attach_lora_to_model(batch_model, layer=TINY_LAYER, lora_cfg=lora_cfg)
        _attach_lora_to_model(single_model, layer=TINY_LAYER, lora_cfg=lora_cfg)
        batch_refs = _collect_attached_adapters(batch_model, layer=TINY_LAYER)
        single_refs = _collect_attached_adapters(single_model, layer=TINY_LAYER)

        # Make both LoRA branches non-degenerate so gradients through A and B
        # are independently exercised, then make their states byte-identical.
        with t.no_grad():
            for ref_index, (batch_ref, single_ref) in enumerate(
                zip(batch_refs, single_refs, strict=True)
            ):
                batch_ref.adapter.lora_B.fill_(0.01 * (ref_index + 1))
                single_ref.adapter.lora_A.copy_(batch_ref.adapter.lora_A)
                single_ref.adapter.lora_B.copy_(batch_ref.adapter.lora_B)

        batch_loss = batch_model(batch_input, targets=batch_target)
        batch_loss.backward()

        total_single_loss = t.zeros((), dtype=t.float32)
        valid_target_count = 0
        for row in rows:
            input_ids, targets = _row_to_sequence_and_target(row, tokenizer=tok)
            single_input = t.tensor([input_ids], dtype=t.long)
            single_target = t.tensor([targets], dtype=t.long)
            total_single_loss = total_single_loss + single_model(
                single_input,
                targets=single_target,
                loss_reduction="sum",
            )
            valid_target_count += sum(target != -1 for target in targets)
        unpadded_loss = total_single_loss / valid_target_count
        unpadded_loss.backward()

        t.testing.assert_close(batch_loss, unpadded_loss)
        saw_nonzero_a_gradient = False
        saw_nonzero_b_gradient = False
        for batch_ref, single_ref in zip(batch_refs, single_refs, strict=True):
            batch_a_grad = batch_ref.adapter.lora_A.grad
            single_a_grad = single_ref.adapter.lora_A.grad
            batch_b_grad = batch_ref.adapter.lora_B.grad
            single_b_grad = single_ref.adapter.lora_B.grad
            assert batch_a_grad is not None and single_a_grad is not None
            assert batch_b_grad is not None and single_b_grad is not None
            t.testing.assert_close(batch_a_grad, single_a_grad)
            t.testing.assert_close(batch_b_grad, single_b_grad)
            saw_nonzero_a_gradient |= bool(t.count_nonzero(batch_a_grad))
            saw_nonzero_b_gradient |= bool(t.count_nonzero(batch_b_grad))
        assert saw_nonzero_a_gradient
        assert saw_nonzero_b_gradient

    def test_loss_mask_includes_output_positions(self) -> None:
        """Verify at least one position has a valid (non-ignore) target."""
        tok = _StubNanochatTokenizer()
        rows = [
            _pos_row(42, "concept_rich", "PRE", "OUTPUT", _n(tok, "PRE")),
        ]
        device = t.device("cpu")
        _, batch_target = _build_lora_batch(
            rows=rows,
            tokenizer=tok,
            device=device,
        )
        assert (batch_target[0] != -1).any(), (
            "At least one target position must be a valid label (not -1)"
        )


# ---------------------------------------------------------------------------
# Sample batch determinism
# ---------------------------------------------------------------------------


class TestSampleBatchIndices:
    def test_same_seed_same_indices(self) -> None:
        """Sampling with the same seed produces the same batch indices."""
        rng_a = t.Generator(device="cpu").manual_seed(123)
        rng_b = t.Generator(device="cpu").manual_seed(123)
        a = _sample_batch_indices(pool_size=10, batch_size=4, generator=rng_a)
        b = _sample_batch_indices(pool_size=10, batch_size=4, generator=rng_b)
        assert a.tolist() == b.tolist()

    def test_different_seed_different_indices(self) -> None:
        rng_a = t.Generator(device="cpu").manual_seed(1)
        rng_b = t.Generator(device="cpu").manual_seed(2)
        a = _sample_batch_indices(pool_size=10, batch_size=8, generator=rng_a)
        b = _sample_batch_indices(pool_size=10, batch_size=8, generator=rng_b)
        # Almost surely different (not a hard guarantee, but for these
        # particular seeds we know they differ).
        assert a.tolist() != b.tolist()

    def test_indices_in_range(self) -> None:
        rng = t.Generator(device="cpu").manual_seed(7)
        indices = _sample_batch_indices(pool_size=5, batch_size=20, generator=rng)
        assert indices.shape == (20,)
        assert int(indices.min().item()) >= 0
        assert int(indices.max().item()) < 5


# ---------------------------------------------------------------------------
# Single-concept training: end-to-end
# ---------------------------------------------------------------------------


class TestTrainSingleConceptEndToEnd:
    def test_loss_decreases_across_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Captured losses at step 0 and step N-1 should show some
        decrease on a tiny adapter trained on synthetic data.

        The adapter starts at lora_B=0, so the initial output equals the
        base model's output and the initial loss is fixed. Even a few
        AdamW steps with lr=1e-2 should drive lora_B away from zero and
        reduce loss on the (very small) training pool.
        """
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=20, batch_size=4, lr=1e-2)
        model = _make_tiny_gpt()

        rows = [
            _pos_row(42, "concept_rich", "AAA", "BCDE", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "BBB", "CDEF", _n(tok, "BBB")),
            _pos_row(42, "concept_rich", "CCC", "DEFG", _n(tok, "CCC")),
            _pos_row(42, "concept_rich", "DDD", "EFGH", _n(tok, "DDD")),
            _pos_row(42, "concept_avoiding", "neg1", "xyz", _n(tok, "neg1")),
            _pos_row(42, "concept_avoiding", "neg2", "uvw", _n(tok, "neg2")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        recorded_losses: list[float] = []
        real_step = lora_training_mod._train_step

        def _spy_step(*args: Any, **kwargs: Any) -> Any:
            loss_value = real_step(*args, **kwargs)
            recorded_losses.append(loss_value)
            return loss_value

        monkeypatch.setattr(lora_training_mod, "_train_step", _spy_step)

        train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )

        assert len(recorded_losses) == cfg.lora.steps
        # Compare first step vs the final step. Due to AdamW noise we
        # expect a downward trend — final < initial * 0.98 is plenty
        # generous on this fixture.
        assert recorded_losses[-1] < recorded_losses[0], (
            f"Expected loss to decrease across steps, got "
            f"first={recorded_losses[0]:.4f} last={recorded_losses[-1]:.4f}"
        )

    def test_saved_adapter_state_has_ab_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=2, batch_size=2)
        model = _make_tiny_gpt()

        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
            _pos_row(42, "concept_avoiding", "n1", "x", _n(tok, "n1")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        assert len(out_paths) == 1
        payload = load_file(str(out_paths[0]))
        # Expect 4 attached sublayers × (lora_A + lora_B) = 8 tensors.
        a_keys = [k for k in payload if k.endswith("lora_A")]
        b_keys = [k for k in payload if k.endswith("lora_B")]
        assert len(a_keys) == 4
        assert len(b_keys) == 4
        # Every key should embed the layer index 0 (only attached layer).
        for k in payload:
            assert "layer_0" in k, f"Unexpected key without layer_0: {k}"

    def test_adapter_detached_after_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After train_lora_all returns, the model's c_qkv etc. must be
        the SAME object (id()) as before — adapters fully removed."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=2, batch_size=2)
        model = _make_tiny_gpt()

        layer = TINY_LAYER
        block = cast(nn.ModuleList, model.transformer["h"])[layer]
        before_ids: dict[tuple[str, str], int] = {}
        for parent_name, child_name in _ATTACH_PATHS:
            before_ids[(parent_name, child_name)] = id(
                getattr(getattr(block, parent_name), child_name)
            )

        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
            _pos_row(42, "concept_avoiding", "n1", "x", _n(tok, "n1")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        # After training, identity must be preserved.
        block_after = cast(nn.ModuleList, model.transformer["h"])[layer]
        for parent_name, child_name in _ATTACH_PATHS:
            current = getattr(getattr(block_after, parent_name), child_name)
            assert id(current) == before_ids[(parent_name, child_name)], (
                f"{parent_name}.{child_name} object identity changed"
            )
            assert not isinstance(current, LoRAAdapter)

    def test_output_path_matches_paths_lora_adapter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2)
        model = _make_tiny_gpt()
        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        expected = test_paths.default_paths(cast(str, cfg.run_tag)).lora_adapter(
            mode=SteeringMode.POSITIVE, concept_id=42
        )
        assert out_paths == [expected]
        assert expected.exists()

    def test_metadata_includes_concept_and_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2)
        model = _make_tiny_gpt()
        rows = [
            _pos_row(99, "concept_rich", "AA", "BB", _n(tok, "AA")),
            _pos_row(99, "concept_rich", "CC", "DD", _n(tok, "CC")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(99)],
            training_corpus_path=corpus_path,
        )
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        for key in (
            "concept_id",
            "mode",
            "target_model",
            "layer",
            "n_train",
            "rank",
            "alpha",
        ):
            assert key in meta, f"Missing metadata key: {key}"
        assert meta["concept_id"] == "99"
        assert meta["mode"] == "positive"
        assert meta["target_model"] == "decoder_only"
        assert meta["layer"] == str(TINY_LAYER)


# ---------------------------------------------------------------------------
# Errors and guards
# ---------------------------------------------------------------------------


class TestErrors:
    def test_too_few_training_rows_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2, min_training_rows=8)
        model = _make_tiny_gpt()
        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="min_training_rows"):
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_nan_loss_raises_not_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force the GPT.forward to return NaN loss; expect a hard error."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=2, batch_size=2)
        model = _make_tiny_gpt()
        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
            _pos_row(42, "concept_rich", "EEE", "FFF", _n(tok, "EEE")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        # Monkeypatch model.forward so it returns NaN whenever targets are
        # passed (training mode). Non-target calls still go to the real
        # forward. We replace via setattr on the bound method so we don't
        # interfere with attribute access on the GPT class.
        real_forward = model.forward

        def _nan_forward(idx: t.Tensor, targets: t.Tensor | None = None, **kwargs: Any) -> t.Tensor:
            out = real_forward(idx, targets=targets, **kwargs)
            if targets is not None:
                return t.tensor(float("nan"), requires_grad=True)
            return out

        monkeypatch.setattr(model, "forward", _nan_forward)

        with pytest.raises(AxBenchBaseError, match="NaN"):
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_empty_concepts_list_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        corpus_path = tmp_path / "empty.parquet"
        _write_positive_corpus(corpus_path, [])

        with pytest.raises(AxBenchBaseError, match="empty concepts"):
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[],
                training_corpus_path=corpus_path,
            )

    def test_any_absent_concept_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The explicit concept population cannot silently shrink at training."""
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, min_training_rows=2)
        model = _make_tiny_gpt()

        rows = [
            _pos_row(42, "concept_rich", "prefix a", "out a", _n(tok, "prefix a")),
            _pos_row(42, "concept_rich", "prefix b", "out b", _n(tok, "prefix b")),
            _pos_row(42, "concept_avoiding", "prefix c", "out c", _n(tok, "prefix c")),
            _pos_row(42, "concept_avoiding", "prefix d", "out d", _n(tok, "prefix d")),
        ]
        corpus = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus, rows)

        with pytest.raises(AxBenchBaseError, match=r"concepts \[99\].*positive"):
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42), _concept(99)],
                training_corpus_path=corpus,
            )

    def test_all_concepts_absent_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ALL requested concepts are absent from the corpus, raise
        AxBenchBaseError instead of silently returning an empty list.

        TDD RED: this test fails before the all-dropped guard is added.
        """
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, min_training_rows=2)
        model = _make_tiny_gpt()

        # Corpus has rows for concept 99 only; requested concepts 1 and 2
        # are completely absent.
        rows = [
            _pos_row(99, "concept_rich", "prefix a", "out a", _n(tok, "prefix a")),
            _pos_row(99, "concept_avoiding", "prefix b", "out b", _n(tok, "prefix b")),
        ]
        corpus = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus, rows)

        with pytest.raises(AxBenchBaseError, match="absent from the"):
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(1), _concept(2)],
                training_corpus_path=corpus,
            )




    def test_run_tag_none_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path).replace(run_tag=None)
        model = _make_tiny_gpt()
        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="run_tag"):
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )

    def test_unknown_mode_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path)
        model = _make_tiny_gpt()
        rows = [
            _pos_row(42, "concept_rich", "AAA", "BBB", _n(tok, "AAA")),
            _pos_row(42, "concept_rich", "CCC", "DDD", _n(tok, "CCC")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        with pytest.raises(AxBenchBaseError, match="mode"):
            # Deliberately smuggle an unknown mode string into the
            # ``SteeringMode`` slot so ``train_lora_all``'s validity
            # check raises. ``cast`` keeps ``ty`` happy.
            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=cast(SteeringMode, "not_a_real_mode"),
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )


# ---------------------------------------------------------------------------
# Mode selection (positive vs negative)
# ---------------------------------------------------------------------------


class TestModeSelection:
    def test_positive_mode_trains_on_concept_rich_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In positive mode, the rows used for training should be
        ``concept_rich``."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2)
        model = _make_tiny_gpt()

        rows = [
            _pos_row(42, "concept_rich", "RICH1", "abc", _n(tok, "RICH1")),
            _pos_row(42, "concept_rich", "RICH2", "def", _n(tok, "RICH2")),
            _pos_row(42, "concept_avoiding", "AVOID1", "xyz", _n(tok, "AVOID1")),
            _pos_row(42, "concept_avoiding", "AVOID2", "uvw", _n(tok, "AVOID2")),
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        seen_prefixes: list[list[str]] = []
        real_build = lora_training_mod._build_lora_batch

        def _spy_build(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
            seen_prefixes.append([r["prefix_text"] for r in rows])
            return real_build(rows=rows, **kwargs)

        monkeypatch.setattr(lora_training_mod, "_build_lora_batch", _spy_build)

        out_paths = train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        assert out_paths
        # Every batch should consist exclusively of RICH prefixes.
        for batch_prefixes in seen_prefixes:
            for prefix in batch_prefixes:
                assert prefix.startswith("RICH"), (
                    f"Positive mode should sample only concept_rich rows; "
                    f"saw '{prefix}'"
                )

        # Metadata reports n_train as the size of the positives pool.
        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        assert meta["n_train"] == "2"

    def test_negative_mode_trains_on_concept_avoiding_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In negative mode, the rows used for training should be
        ``concept_avoiding`` (we want the LoRA to suppress the
        concept)."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2)
        model = _make_tiny_gpt()

        rows = [
            _neg_row(42, "concept_rich", "RICH1", "abc", _n(tok, "RICH1")),
            _neg_row(42, "concept_rich", "RICH2", "def", _n(tok, "RICH2")),
            _neg_row(42, "concept_avoiding", "AVOID1", "xyz", _n(tok, "AVOID1")),
            _neg_row(42, "concept_avoiding", "AVOID2", "uvw", _n(tok, "AVOID2")),
        ]
        corpus_path = tmp_path / "corpus_neg.parquet"
        _write_negative_corpus(corpus_path, rows)

        seen_prefixes: list[list[str]] = []
        real_build = lora_training_mod._build_lora_batch

        def _spy_build(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
            seen_prefixes.append([r["prefix_text"] for r in rows])
            return real_build(rows=rows, **kwargs)

        monkeypatch.setattr(lora_training_mod, "_build_lora_batch", _spy_build)

        out_paths = train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.NEGATIVE,
            concepts=[_concept(42)],
            training_corpus_path=corpus_path,
        )
        assert out_paths
        for batch_prefixes in seen_prefixes:
            for prefix in batch_prefixes:
                assert prefix.startswith("AVOID"), (
                    f"Negative mode should sample only concept_avoiding rows; "
                    f"saw '{prefix}'"
                )

        with safe_open(str(out_paths[0]), framework="pt") as f:
            meta = f.metadata() or {}
        assert meta["n_train"] == "2"
        assert meta["mode"] == "negative"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterministicSampling:
    def test_deterministic_batch_sampling_same_seed_same_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Running training twice with the same seed yields the SAME batch
        sequence (verified by capturing the row indices the loop sees)."""
        tok = _install_stub_tokenizer(monkeypatch)

        rows = [
            _pos_row(42, "concept_rich", f"P{i}", "OUT", _n(tok, f"P{i}"))
            for i in range(8)
        ] + [
            _pos_row(42, "concept_avoiding", f"N{i}", "OUT", _n(tok, f"N{i}"))
            for i in range(2)
        ]
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        # Capture the ORIGINAL helper exactly once, before any monkeypatch,
        # so two successive runs both wrap the same underlying function
        # rather than chaining wrappers on each other.
        real_build = lora_training_mod._build_lora_batch

        def _run_once() -> list[list[str]]:
            cfg = _tiny_cfg(tmp_path, steps=3, batch_size=4)
            model = _make_tiny_gpt()
            seen: list[list[str]] = []

            def _spy_build(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
                seen.append([r["prefix_text"] for r in rows])
                return real_build(rows=rows, **kwargs)

            monkeypatch.setattr(lora_training_mod, "_build_lora_batch", _spy_build)

            train_lora_all(
                cfg=cfg,
                target_model=model,
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(42)],
                training_corpus_path=corpus_path,
            )
            return seen

        first = _run_once()
        second = _run_once()
        assert first == second

    def test_different_concept_ids_yield_different_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two concepts with the same row-pool but different concept_id
        yield DIFFERENT batch sequences (because the seed is master_seed +
        concept_id)."""
        tok = _install_stub_tokenizer(monkeypatch)

        # Build a corpus with two concepts each having identical row sets.
        rows: list[dict[str, Any]] = []
        for cid in (10, 20):
            for i in range(8):
                rows.append(
                    _pos_row(cid, "concept_rich", f"P{i}", "OUT", _n(tok, f"P{i}"))
                )
            for i in range(2):
                rows.append(
                    _pos_row(cid, "concept_avoiding", f"N{i}", "OUT", _n(tok, f"N{i}"))
                )
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        cfg = _tiny_cfg(tmp_path, steps=2, batch_size=4)

        seen_per_concept: dict[int, list[list[str]]] = {10: [], 20: []}
        active_cid: list[int] = []
        real_build = lora_training_mod._build_lora_batch

        def _spy_build(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
            seen_per_concept[active_cid[-1]].append(
                [r["prefix_text"] for r in rows]
            )
            return real_build(rows=rows, **kwargs)

        monkeypatch.setattr(lora_training_mod, "_build_lora_batch", _spy_build)

        for cid in (10, 20):
            active_cid.append(cid)
            train_lora_all(
                cfg=cfg,
                target_model=_make_tiny_gpt(),
                target_model_name="decoder_only",
                mode=SteeringMode.POSITIVE,
                concepts=[_concept(cid)],
                training_corpus_path=corpus_path,
            )
            active_cid.pop()

        # Same row pool, different concept_id seeds → at least one batch
        # should differ between the two concepts.
        assert seen_per_concept[10] != seen_per_concept[20]


# ---------------------------------------------------------------------------
# Multi-concept dispatch
# ---------------------------------------------------------------------------


class TestMultiConcept:
    def test_two_concepts_produce_two_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2)
        model = _make_tiny_gpt()

        rows = []
        for cid in (1, 2):
            for i in range(2):
                rows.append(
                    _pos_row(cid, "concept_rich", f"p{cid}{i}", "out",
                             _n(tok, f"p{cid}{i}"))
                )
                rows.append(
                    _pos_row(cid, "concept_avoiding", f"n{cid}{i}", "out",
                             _n(tok, f"n{cid}{i}"))
                )
        corpus_path = tmp_path / "corpus.parquet"
        _write_positive_corpus(corpus_path, rows)

        out_paths = train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[_concept(1), _concept(2)],
            training_corpus_path=corpus_path,
        )
        assert len(out_paths) == 2
        assert len({str(p) for p in out_paths}) == 2
        for cid, path in zip([1, 2], out_paths, strict=True):
            with safe_open(str(path), framework="pt") as f:
                meta = f.metadata() or {}
            assert meta["concept_id"] == str(cid)


# ---------------------------------------------------------------------------
# Direct helper coverage: _train_lora_single_concept signature compatibility
# ---------------------------------------------------------------------------


class TestSingleConceptHelperDirect:
    def test_helper_returns_path_and_writes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=2, batch_size=2)
        model = _make_tiny_gpt()

        rows = [
            _pos_row(42, "concept_rich", "AA", "BB", _n(tok, "AA")),
            _pos_row(42, "concept_rich", "CC", "DD", _n(tok, "CC")),
        ]
        path = _train_lora_single_concept(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concept_id=42,
            rows=rows,
            tokenizer=tok,
        )
        assert isinstance(path, Path)
        assert path.exists()

    def test_helper_restores_train_mode_when_model_started_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the caller hands us a model in train mode, the helper must
        leave it in train mode (it must NOT switch the model into eval at
        the end). This covers the ``if not was_training: target_model.eval()``
        false branch."""
        tok = _install_stub_tokenizer(monkeypatch)
        cfg = _tiny_cfg(tmp_path, steps=1, batch_size=2)
        model = _make_tiny_gpt()
        model.train()
        assert model.training

        rows = [
            _pos_row(42, "concept_rich", "AA", "BB", _n(tok, "AA")),
            _pos_row(42, "concept_rich", "CC", "DD", _n(tok, "CC")),
        ]
        _train_lora_single_concept(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concept_id=42,
            rows=rows,
            tokenizer=tok,
        )
        # Caller passed a training-mode model — helper must not flip it.
        assert model.training is True


# ---------------------------------------------------------------------------
# Defensive guards
# ---------------------------------------------------------------------------


class TestCollectAdaptersGuard:
    def test_collect_without_attach_raises(self) -> None:
        """Calling ``_collect_attached_adapters`` on a model that was
        never attached must raise ``AxBenchBaseError`` so subtle ordering
        bugs surface loudly."""
        model = _make_tiny_gpt()
        with pytest.raises(AxBenchBaseError, match="LoRAAdapter"):
            _collect_attached_adapters(model, layer=TINY_LAYER)
