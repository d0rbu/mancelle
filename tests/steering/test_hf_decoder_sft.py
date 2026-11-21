"""Offline Hugging Face SFT backend, tokenizer, optimizer, and export contracts."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch as t
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from torch._dynamo import config as dynamo_config
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    SmolLM3Config,
    SmolLM3ForCausalLM,
)

from nanochat.loss_eval import evaluate_bpb
from scripts.train.chat_sft import _hf_metadata
from steering.sft import hf_decoder
from steering.sft.clipping import apply_gradient_clipping, resolve_grad_clip_max_norm
from steering.sft.hf_checkpoint import (
    build_cpu_bf16_state_dict,
    publish_hf_checkpoint,
    validate_export_smoke,
)
from steering.sft.hf_contract import (
    DecoderSFTBackend,
    HFAttentionImplementation,
    HFDecoderContract,
    HFDecoderFamily,
    HFDistributedImplementation,
)
from steering.sft.hf_decoder import (
    HFDecoderModelAdapter,
    HFSFTTokenizer,
    build_hf_optimizer,
    configure_generation_contract,
    load_hf_config_and_tokenizer,
    load_hf_model_in_local_rank_slots,
    resolve_hf_output_root,
    validate_decoder_sft_config,
    validate_hf_source_config,
    validate_hf_source_tokenizer,
    validate_loaded_hf_model,
)
from steering.sft.iteration import sft_data_generator_bos_bestfit_core
from steering.sft.loss_mask import LossMaskMode
from steering.sft.pythia import (
    PYTHIA_CONTRACT,
    PYTHIA_END_OF_TEXT,
    PYTHIA_END_OF_TEXT_ID,
    PYTHIA_GEOMETRIES,
    PYTHIA_PAD,
    PYTHIA_PAD_ID,
    PythiaModelId,
)
from steering.sft.smollm3 import (
    BEGIN_OF_TEXT,
    BEGIN_OF_TEXT_ID,
    FINETUNE_PAD,
    FINETUNE_PAD_ID,
    IM_END,
    IM_END_ID,
    IM_START,
    SMOLLM3_CONTRACT,
    SOURCE_END_OF_TEXT,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
)
from steering.sft.token_bytes import compute_hf_token_bytes

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")


def _synthetic_smollm3_tokenizer() -> PreTrainedTokenizerFast:
    """Build a byte-level tokenizer with the pinned 128256-id layout offline."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate(alphabet)}
    for index in range(len(vocab), 128000):
        vocab[f"<base_{index}>"] = index
    backend = Tokenizer(BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        clean_up_tokenization_spaces=False,
    )
    controls = [
        BEGIN_OF_TEXT,
        SOURCE_END_OF_TEXT,
        "<think>",
        "</think>",
        FINETUNE_PAD,
        "<|reserved_special_token_2|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|python_tag|>",
        IM_START,
        IM_END,
        TOOL_RESPONSE_OPEN,
        TOOL_RESPONSE_CLOSE,
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
        *[f"<|reserved_special_token_{index}|>" for index in range(9, 248)],
    ]
    assert tokenizer.add_tokens(controls) == 256
    assert len(tokenizer) == 128256
    return tokenizer


@pytest.fixture(scope="module")
def sft_tokenizer() -> HFSFTTokenizer:
    return HFSFTTokenizer(_synthetic_smollm3_tokenizer(), SMOLLM3_CONTRACT)


def _synthetic_pythia_tokenizer() -> PreTrainedTokenizerFast:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {PYTHIA_END_OF_TEXT: PYTHIA_END_OF_TEXT_ID, PYTHIA_PAD: PYTHIA_PAD_ID}
    for token in alphabet:
        if token not in vocab:
            vocab[token] = len(vocab)
    # Reproduce two real Pythia ByteLevel boundary pieces. ``Ġ<`` crosses
    # from trailing assistant whitespace into a tool-call tag, while
    # ``Ġanswer`` has a trimmed leading-space offset.
    merges = [("Ġ", "<"), ("Ġ", "a")]
    merged = ["Ġ<", "Ġa"]
    for previous, character in zip(
        ("Ġa", "Ġan", "Ġans", "Ġansw", "Ġanswe"),
        "nswer",
        strict=True,
    ):
        merges.append((previous, character))
        merged.append(previous + character)
    for token in merged:
        vocab[token] = len(vocab)
    for index in range(len(vocab), PYTHIA_CONTRACT.tokenizer_vocab_size):
        vocab[f"<base_{index}>"] = index
    backend = Tokenizer(BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=PYTHIA_END_OF_TEXT,
        eos_token=PYTHIA_END_OF_TEXT,
        unk_token=PYTHIA_END_OF_TEXT,
        pad_token=PYTHIA_PAD,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.backend_tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=False,
        trim_offsets=True,
        use_regex=True,
    )
    assert len(tokenizer) == PYTHIA_CONTRACT.tokenizer_vocab_size
    return tokenizer


@pytest.fixture(scope="module")
def pythia_tokenizer() -> HFSFTTokenizer:
    return HFSFTTokenizer(_synthetic_pythia_tokenizer(), PYTHIA_CONTRACT)


def _tiny_model() -> SmolLM3ForCausalLM:
    config = SmolLM3Config(
        vocab_size=128256,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return SmolLM3ForCausalLM(config).float()


def _tiny_pythia_model() -> GPTNeoXForCausalLM:
    config = GPTNeoXConfig(
        vocab_size=PYTHIA_CONTRACT.model_vocab_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=PYTHIA_END_OF_TEXT_ID,
        eos_token_id=PYTHIA_END_OF_TEXT_ID,
        pad_token_id=PYTHIA_PAD_ID,
    )
    return GPTNeoXForCausalLM(config).float()


@pytest.mark.parametrize(
    ("model_class", "config", "parameter_count", "unique_specializations"),
    [
        (
            GPTNeoXForCausalLM,
            GPTNeoXConfig(
                vocab_size=50304,
                hidden_size=2048,
                intermediate_size=8192,
                num_hidden_layers=24,
                num_attention_heads=16,
                max_position_embeddings=2048,
                tie_word_embeddings=False,
            ),
            1_414_647_808,
            8,
        ),
        (
            GPTNeoXForCausalLM,
            GPTNeoXConfig(
                vocab_size=50304,
                hidden_size=2560,
                intermediate_size=10240,
                num_hidden_layers=32,
                num_attention_heads=32,
                max_position_embeddings=2048,
                tie_word_embeddings=False,
            ),
            2_775_208_960,
            8,
        ),
        (
            SmolLM3ForCausalLM,
            SmolLM3Config(
                vocab_size=128256,
                hidden_size=2048,
                intermediate_size=11008,
                num_hidden_layers=36,
                num_attention_heads=16,
                num_key_value_heads=4,
                max_position_embeddings=4096,
                tie_word_embeddings=True,
            ),
            3_075_098_624,
            6,
        ),
    ],
)
def test_full_architecture_adamw_shapes_fit_the_fused_kernel_compile_cache(
    model_class,
    config,
    parameter_count: int,
    unique_specializations: int,
) -> None:
    # DistMuon routes every HF parameter through nanochat's shape-specialized
    # compiled AdamW kernel. Materializing the exact architectures on meta proves
    # the production shape set fits TorchDynamo's per-code-object cache limit.
    with t.device("meta"):
        model = model_class(config)
    parameters = list(model.parameters())
    assert sum(parameter.numel() for parameter in parameters) == parameter_count
    for world_size in (1, 8, 32):
        # DistMuon passes a first-dimension shard to the kernel for parameters
        # large enough to shard evenly; Pythia sits exactly at the cache limit
        # in both the one-GPU gate and intended 8/32-rank topologies.
        signatures = set()
        for parameter in parameters:
            shape = tuple(parameter.shape)
            if world_size > 1 and parameter.numel() >= 1024 and shape[0] % world_size == 0:
                shape = (shape[0] // world_size, *shape[1:])
            signatures.add((shape, tuple(parameter.stride()), parameter.dtype))
        assert len(signatures) == unique_specializations
        assert len(signatures) <= dynamo_config.recompile_limit


def _composed(name: str) -> dict[str, Any]:
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=name)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return cast(dict[str, Any], resolved)


def test_native_and_hf_config_composition() -> None:
    native = _composed("chat_sft")
    assert native["model_backend"] == "nanochat"
    assert validate_decoder_sft_config(native) is DecoderSFTBackend.NANOCHAT
    assert native["huggingface"] is None
    assert "smollm3" not in native
    assert native["grad_clip_max_norm"] is None
    smol = _composed("chat_sft_smollm3_stage1_step_40000")
    assert smol["model_backend"] == "huggingface"
    assert validate_decoder_sft_config(smol) is DecoderSFTBackend.HUGGINGFACE
    assert smol["huggingface"]["model_id"] == "HuggingFaceTB/SmolLM3-3B-checkpoints"
    assert smol["huggingface"]["revision"] == "524d818535ee98e56a4f0d621c84dd5c8fb8d3d4"
    assert smol["device_batch_size"] == 1
    assert smol["total_batch_size"] == 524288
    assert smol["embedding_lr"] is None
    assert "smollm3" not in smol
    pythia = _composed("chat_sft_pythia_2_8b")
    assert validate_decoder_sft_config(pythia) is DecoderSFTBackend.HUGGINGFACE
    assert pythia["huggingface"]["model_id"] == "EleutherAI/pythia-2.8b"
    assert pythia["huggingface"]["revision"] == "2a259cdd96a4beb1cdf467512e3904197345f6a9"
    assert "smollm3" not in pythia
    pythia_1_4b = _composed("chat_sft_pythia_1_4b")
    assert validate_decoder_sft_config(pythia_1_4b) is DecoderSFTBackend.HUGGINGFACE
    assert pythia_1_4b["huggingface"]["model_id"] == PythiaModelId.PYTHIA_1_4B.value
    assert (
        pythia_1_4b["huggingface"]["revision"]
        == "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"
    )
    assert {
        key: value
        for key, value in pythia_1_4b.items()
        if key != "huggingface"
    } == {key: value for key, value in pythia.items() if key != "huggingface"}
    assert set(pythia_1_4b["huggingface"]) == set(pythia["huggingface"])


def test_pythia_metadata_uses_generic_backend_contract() -> None:
    cfg = OmegaConf.create(_composed("chat_sft_pythia_2_8b"))
    model = _tiny_pythia_model()
    metadata = _hf_metadata(
        cfg=cfg,
        user_config=_composed("chat_sft_pythia_2_8b"),
        completed_step=7,
        val_bpb=float("nan"),
        min_val_bpb=1.25,
        ddp_world_size=2,
        grad_accum_steps=256,
        grad_clip_max_norm=None,
        train_fingerprint={"split": "train"},
        val_fingerprint={"split": "val"},
        raw_model=model,
        contract=PYTHIA_CONTRACT,
        attention_implementation=HFAttentionImplementation.SDPA,
        distributed_implementation=HFDistributedImplementation.DIST_MUON_ADAMW,
    )
    assert metadata["format"] == "huggingface_standard_sft"
    assert metadata["format_version"] == 2
    assert metadata["backend"] == "huggingface"
    assert metadata["family"] == "pythia"
    assert metadata["source"] == {
        "model_id": "EleutherAI/pythia-2.8b",
        "revision": "2a259cdd96a4beb1cdf467512e3904197345f6a9",
        "architecture": "GPTNeoXForCausalLM",
        "model_type": "gpt_neox",
    }
    assert metadata["latest_validation_bpb"] is None
    assert metadata["minimum_validation_bpb"] == 1.25
    assert metadata["vocabulary"] == {"model": 50304, "tokenizer": 50277}
    assert metadata["tied_embeddings"] is False
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("model_backend",), "generic_hf", "model_backend"),
        (("huggingface", "model_id"), "", "model_id"),
        (("huggingface", "revision"), "ABC", "revision"),
        (("huggingface", "revision"), "A" * 40, "revision"),
        (("huggingface", "learning_rate"), 0.0, "learning_rate"),
        (("huggingface", "learning_rate"), math.inf, "learning_rate"),
        (("huggingface", "adam_eps"), 0.0, "adam_eps"),
        (("huggingface", "adam_eps"), math.inf, "adam_eps"),
        (("huggingface", "adam_betas"), [0.8], "adam_betas"),
        (("huggingface", "adam_betas"), [-0.1, 0.9], "adam_betas"),
        (("huggingface", "adam_betas"), [0.8, math.nan], "adam_betas"),
        (("huggingface", "adam_betas"), [0.8, 1.0], "adam_betas"),
        (("huggingface", "gradient_checkpointing"), 1, "boolean"),
        (("huggingface", "validate_export"), "yes", "boolean"),
        (("huggingface", "output_dir"), "", "output_dir"),
        (("dtype",), "float32", "dtype=bfloat16"),
        (("max_seq_len",), 0, "max_seq_len"),
        (("num_iterations",), 0, "num_iterations"),
        (("num_iterations",), -2, "num_iterations"),
        (("num_iterations",), True, "num_iterations"),
        (("grad_clip_max_norm",), True, "grad_clip"),
        (("grad_clip_max_norm",), 0.0, "grad_clip"),
        (("grad_clip_max_norm",), math.nan, "grad_clip"),
        (("grad_clip_max_norm",), math.inf, "grad_clip"),
        (("weight_decay",), -1.0, "weight_decay"),
        (("weight_decay",), math.inf, "weight_decay"),
        (("matrix_lr",), 0.02, "matrix_lr"),
    ],
)
def test_invalid_hf_config_values_fail(path, value, match) -> None:
    config = copy.deepcopy(_composed("chat_sft_smollm3_stage1_step_40000"))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_decoder_sft_config(config)


def test_source_context_limit_is_enforced() -> None:
    config = copy.deepcopy(_composed("chat_sft_smollm3_stage1_step_40000"))
    config["max_seq_len"] = 4097
    with pytest.raises(ValueError, match="exceeds source model limit"):
        validate_decoder_sft_config(
            config,
            source_config=SimpleNamespace(max_position_embeddings=4096),
        )


def test_native_backend_rejects_hf_block() -> None:
    config = copy.deepcopy(_composed("chat_sft"))
    config["huggingface"] = {"model_id": "unexpected"}
    with pytest.raises(ValueError, match="requires huggingface: null"):
        validate_decoder_sft_config(config)


def test_source_config_and_tokenizer_calls_share_exact_identity() -> None:
    config = SmolLM3Config(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=None,
        architectures=["SmolLM3ForCausalLM"],
    )
    tokenizer = _synthetic_smollm3_tokenizer()
    with (
        patch(
            "steering.sft.hf_decoder.AutoConfig.from_pretrained", return_value=config
        ) as auto_config,
        patch(
            "steering.sft.hf_decoder.AutoTokenizer.from_pretrained", return_value=tokenizer
        ) as auto_tokenizer,
    ):
        load_hf_config_and_tokenizer(
            model_id="org/model",
            revision="a" * 40,
        )
    expected = {"revision": "a" * 40, "trust_remote_code": False}
    auto_config.assert_called_once_with("org/model", **expected)
    auto_tokenizer.assert_called_once_with("org/model", **expected)


def test_cheap_source_config_failure_precedes_tokenizer_loading() -> None:
    bad_config = SmolLM3Config(
        vocab_size=128256,
        hidden_size=16,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=None,
        architectures=["SmolLM3ForCausalLM"],
    )
    with (
        patch("steering.sft.hf_decoder.AutoConfig.from_pretrained", return_value=bad_config),
        patch("steering.sft.hf_decoder.AutoTokenizer.from_pretrained") as auto_tokenizer,
    ):
        with pytest.raises(ValueError, match="hidden_size"):
            load_hf_config_and_tokenizer(model_id="org/model", revision="a" * 40)
    auto_tokenizer.assert_not_called()


def test_model_weight_loader_uses_the_exact_source_identity(monkeypatch) -> None:
    model = _tiny_model()
    model.eval()
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with patch(
        "steering.sft.hf_decoder.AutoModelForCausalLM.from_pretrained",
        return_value=model,
    ) as auto_model:
        loaded = load_hf_model_in_local_rank_slots(
            model_id="org/model",
            revision="a" * 40,
            contract=SMOLLM3_CONTRACT,
            device="cpu",
            local_rank=0,
            gradient_checkpointing=True,
        )
    assert loaded is model
    auto_model.assert_called_once_with(
        "org/model",
        revision="a" * 40,
        trust_remote_code=False,
        dtype=t.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    assert model.config.use_cache is False
    assert model.is_gradient_checkpointing
    assert model.training


def test_model_weight_loader_accepts_pythia_contract(monkeypatch) -> None:
    model = _tiny_pythia_model()
    model.eval()
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with patch(
        "steering.sft.hf_decoder.AutoModelForCausalLM.from_pretrained",
        return_value=model,
    ):
        loaded = load_hf_model_in_local_rank_slots(
            model_id="org/model",
            revision="a" * 40,
            contract=PYTHIA_CONTRACT,
            device="cpu",
            local_rank=0,
            gradient_checkpointing=True,
        )
    assert loaded is model
    assert model.config.use_cache is False
    assert model.is_gradient_checkpointing
    assert model.training


def test_model_weight_loader_rejects_wrong_contract(monkeypatch) -> None:
    model = _tiny_pythia_model()
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with patch(
        "steering.sft.hf_decoder.AutoModelForCausalLM.from_pretrained",
        return_value=model,
    ):
        with pytest.raises(TypeError, match="SmolLM3ForCausalLM"):
            load_hf_model_in_local_rank_slots(
                model_id="org/model",
                revision="a" * 40,
                contract=SMOLLM3_CONTRACT,
                device="cpu",
                local_rank=0,
                gradient_checkpointing=False,
            )


def test_source_preflight_rejects_dimension_token_and_tie_drift() -> None:
    config = SmolLM3Config(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=None,
        architectures=["SmolLM3ForCausalLM"],
    )
    validate_hf_source_config(config)
    config.num_hidden_layers = 35
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_hf_source_config(config)

    tokenizer = _synthetic_smollm3_tokenizer()
    validate_hf_source_tokenizer(tokenizer, SMOLLM3_CONTRACT)
    tokenizer.add_tokens(["unexpected"])
    with pytest.raises(ValueError, match="size"):
        validate_hf_source_tokenizer(tokenizer, SMOLLM3_CONTRACT)

    model = _tiny_model()
    model.lm_head.weight = t.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="must be tied"):
        validate_loaded_hf_model(model, SMOLLM3_CONTRACT)


def test_pythia_source_preflight_and_untied_embedding_contract() -> None:
    config = GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        architectures=["GPTNeoXForCausalLM"],
    )
    assert validate_hf_source_config(config).family is HFDecoderFamily.PYTHIA
    config.hidden_size = 2048
    with pytest.raises(ValueError, match="hidden_size"):
        validate_hf_source_config(config)
    tokenizer = _synthetic_pythia_tokenizer()
    validate_hf_source_tokenizer(tokenizer, PYTHIA_CONTRACT)
    model = _tiny_pythia_model()
    validate_loaded_hf_model(model, PYTHIA_CONTRACT)
    model.lm_head.weight = model.gpt_neox.embed_in.weight
    with pytest.raises(ValueError, match="must be untied"):
        validate_loaded_hf_model(model, PYTHIA_CONTRACT)


@pytest.mark.parametrize("model_id", list(PythiaModelId))
def test_pythia_source_geometry_is_selected_by_pinned_model_id(
    model_id: PythiaModelId,
) -> None:
    geometry = PYTHIA_GEOMETRIES[model_id]
    config = GPTNeoXConfig(
        vocab_size=PYTHIA_CONTRACT.model_vocab_size,
        **geometry.source_config_fields(),
        bos_token_id=PYTHIA_CONTRACT.source_bos_token_id,
        eos_token_id=PYTHIA_CONTRACT.source_eos_token_id,
        pad_token_id=PYTHIA_CONTRACT.source_pad_token_id,
        tie_word_embeddings=PYTHIA_CONTRACT.tied_embeddings,
        architectures=[PYTHIA_CONTRACT.architecture],
    )
    snapshot = config.to_dict()

    assert validate_hf_source_config(config, model_id=model_id.value) is PYTHIA_CONTRACT
    assert config.to_dict() == snapshot
    if model_id is PythiaModelId.PYTHIA_2_8B:
        assert (
            PYTHIA_CONTRACT.source_config_fields_for_model_id(model_id.value)
            is PYTHIA_CONTRACT.source_config_fields
        )


def test_pythia_source_geometry_rejects_mismatched_and_unknown_sizes() -> None:
    geometry = PYTHIA_GEOMETRIES[PythiaModelId.PYTHIA_2_8B]
    config = GPTNeoXConfig(
        vocab_size=PYTHIA_CONTRACT.model_vocab_size,
        **geometry.source_config_fields(),
        bos_token_id=PYTHIA_CONTRACT.source_bos_token_id,
        eos_token_id=PYTHIA_CONTRACT.source_eos_token_id,
        pad_token_id=PYTHIA_CONTRACT.source_pad_token_id,
        tie_word_embeddings=PYTHIA_CONTRACT.tied_embeddings,
        architectures=[PYTHIA_CONTRACT.architecture],
    )

    with pytest.raises(ValueError, match="hidden_size"):
        validate_hf_source_config(config, model_id=PythiaModelId.PYTHIA_1_4B.value)
    with pytest.raises(ValueError, match="unsupported pythia source model_id"):
        validate_hf_source_config(config, model_id="EleutherAI/pythia-6.9b")


def test_local_chained_pythia_export_infers_only_a_known_exact_geometry() -> None:
    geometry = PYTHIA_GEOMETRIES[PythiaModelId.PYTHIA_1_4B]
    config = GPTNeoXConfig(
        vocab_size=PYTHIA_CONTRACT.model_vocab_size,
        **geometry.source_config_fields(),
        bos_token_id=PYTHIA_CONTRACT.generation_bos_token_id,
        eos_token_id=PYTHIA_CONTRACT.generation_eos_token_id,
        pad_token_id=PYTHIA_CONTRACT.generation_pad_token_id,
        tie_word_embeddings=PYTHIA_CONTRACT.tied_embeddings,
        architectures=[PYTHIA_CONTRACT.architecture],
    )

    assert (
        validate_hf_source_config(
            config,
            model_id="/relocated/step_001102",
            local_source=True,
        )
        is PYTHIA_CONTRACT
    )
    config.num_attention_heads = 8
    with pytest.raises(ValueError, match="does not match exactly one"):
        validate_hf_source_config(
            config,
            model_id="/relocated/step_001102",
            local_source=True,
        )


def test_pythia_sizes_resolve_distinct_default_export_namespaces(tmp_path: Path) -> None:
    revision = "a" * 40
    roots = {
        model_id: resolve_hf_output_root(
            base_dir=tmp_path,
            model_id=model_id.value,
            revision=revision,
            output_dir=None,
        )
        for model_id in PythiaModelId
    }

    assert roots[PythiaModelId.PYTHIA_1_4B] == (
        tmp_path
        / "hf_checkpoints"
        / "EleutherAI--pythia-1.4b"
        / revision
        / "standard_sft_checkpoints"
    )
    assert roots[PythiaModelId.PYTHIA_2_8B] == (
        tmp_path
        / "hf_checkpoints"
        / "EleutherAI--pythia-2.8b"
        / revision
        / "standard_sft_checkpoints"
    )


def _family_source_config(contract: HFDecoderContract) -> PretrainedConfig:
    return AutoConfig.for_model(
        contract.model_type,
        **contract.source_config_fields,
        vocab_size=contract.model_vocab_size,
        architectures=[contract.architecture],
        tie_word_embeddings=contract.tied_embeddings,
        bos_token_id=contract.source_bos_token_id,
        eos_token_id=contract.source_eos_token_id,
        pad_token_id=contract.source_pad_token_id,
    )


@pytest.mark.parametrize("contract", list(hf_decoder._CONTRACTS_BY_MODEL_TYPE.values()))
@pytest.mark.parametrize("exported", [False, True], ids=["upstream", "sft-export"])
def test_family_source_token_identity_is_exact_and_nonmutating(
    contract: HFDecoderContract, exported: bool
) -> None:
    config = _family_source_config(contract)
    if exported:
        config.bos_token_id = contract.generation_bos_token_id
        config.eos_token_id = contract.generation_eos_token_id
        config.pad_token_id = contract.generation_pad_token_id
    snapshot = config.to_dict()
    assert validate_hf_source_config(config) is contract
    assert config.to_dict() == snapshot

    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        for foreign in (-1, 2, contract.model_vocab_size, True, False, 0.0, 1.0, "1", [1]):
            changed = copy.deepcopy(config)
            # Bypass Transformers' version-dependent setters to test our boundary.
            changed.__dict__[name] = foreign
            with pytest.raises(ValueError, match=name):
                validate_hf_source_config(changed)
    for name, foreign in (
        ("tie_word_embeddings", not contract.tied_embeddings),
        ("vocab_size", contract.model_vocab_size + 1),
        ("architectures", []),
        *((name, value + 1) for name, value in contract.source_config_fields.items()),
    ):
        changed = copy.deepcopy(config)
        changed.__dict__[name] = foreign
        with pytest.raises(ValueError, match=name):
            validate_hf_source_config(changed)


@pytest.mark.parametrize(
    ("eos_token_id", "pad_token_id"),
    [
        (SMOLLM3_CONTRACT.source_eos_token_id, FINETUNE_PAD_ID),
        (IM_END_ID, SMOLLM3_CONTRACT.source_pad_token_id),
    ],
)
def test_smollm3_source_rejects_mixed_upstream_and_export_token_ids(
    eos_token_id: int, pad_token_id: int | None
) -> None:
    config = _family_source_config(SMOLLM3_CONTRACT)
    config.eos_token_id = eos_token_id
    config.pad_token_id = pad_token_id
    with pytest.raises(ValueError, match="BOS/EOS/PAD"):
        validate_hf_source_config(config)


@pytest.mark.parametrize("contract", list(hf_decoder._CONTRACTS_BY_MODEL_TYPE.values()))
def test_export_chains_through_source_preflight_for_every_family(
    contract: HFDecoderContract,
    sft_tokenizer: HFSFTTokenizer,
    pythia_tokenizer: HFSFTTokenizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_factory, tokenizer = {
        HFDecoderFamily.PYTHIA: (_tiny_pythia_model, pythia_tokenizer),
        HFDecoderFamily.SMOLLM3: (_tiny_model, sft_tokenizer),
    }[contract.family]
    model = model_factory()
    # Only scale architecture dimensions; keep real vocabulary and token identities.
    tiny_contract = replace(
        contract,
        source_config_fields={
            name: getattr(model.config, name) for name in contract.source_config_fields
        },
    )
    monkeypatch.setitem(hf_decoder._CONTRACTS_BY_MODEL_TYPE, contract.model_type, tiny_contract)
    model.config.use_cache = False  # The training loader disables the KV cache.
    configure_generation_contract(model, contract)
    original_config = model.config.to_dict()
    for step in (1, 2):
        output = publish_hf_checkpoint(
            output_root=tmp_path,
            completed_step=step,
            model=model,
            tokenizer=tokenizer,
            metadata={"format_version": 2, "resumable": False},
        )
        bodies = {path.name: path.read_bytes() for path in output.iterdir()}
        exported = AutoConfig.from_pretrained(output, local_files_only=True)
        assert validate_hf_source_config(exported) is tiny_contract
        # Exercise the real next-stage preflight, including tokenizer reinstallation.
        loaded_config, tokenizer, loaded_contract = load_hf_config_and_tokenizer(
            model_id=str(output), revision="a" * 40
        )
        assert loaded_contract is tiny_contract
        assert loaded_config.to_dict() == exported.to_dict()
        assert {path.name: path.read_bytes() for path in output.iterdir()} == bodies
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "tie_word_embeddings",
            "vocab_size",
            "use_cache",
            *contract.source_config_fields,
        ):
            assert getattr(exported, name) == original_config[name]
        assert exported.architectures == [contract.architecture]
        assert exported.dtype == t.bfloat16
        assert "torch_dtype" not in json.loads(bodies["config.json"])
        reloaded = AutoModelForCausalLM.from_pretrained(
            output, local_files_only=True, dtype=t.bfloat16
        )
        for name, tensor in model.state_dict().items():
            assert t.equal(reloaded.state_dict()[name], tensor.to(t.bfloat16))
        assert (
            reloaded.generation_config.bos_token_id,
            reloaded.generation_config.eos_token_id,
            reloaded.generation_config.pad_token_id,
        ) == (
            contract.generation_bos_token_id,
            contract.generation_eos_token_id,
            contract.generation_pad_token_id,
        )
        if contract.family is HFDecoderFamily.SMOLLM3:
            assert reloaded.get_input_embeddings().padding_idx == FINETUNE_PAD_ID
        model = reloaded.float()
        validate_loaded_hf_model(model, tiny_contract)
        configure_generation_contract(model, tiny_contract)


def test_golden_chatml_and_completion_mask(sft_tokenizer: HFSFTTokenizer) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Hi ü"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "python", "text": "x()"},
                    {"type": "python_output", "text": "ok\n"},
                    {"type": "text", "text": "After"},
                ],
            },
        ]
    }
    ids, mask = sft_tokenizer.render_conversation(conversation)
    expected = (
        "<|begin_of_text|><|im_start|>user\nHi ü<|im_end|>\n"
        "<|im_start|>assistant\nBefore<tool_call>x()</tool_call>"
        "<tool_response>ok\n</tool_response>After<|im_end|>\n"
    )
    assert sft_tokenizer.decode(ids) == expected
    assert ids.count(BEGIN_OF_TEXT_ID) == 1
    supervised = sft_tokenizer.decode(
        [token_id for token_id, supervised in zip(ids, mask, strict=True) if supervised]
    )
    assert supervised == "Before<tool_call>x()</tool_call>After<|im_end|>\n"
    for token_id in (
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_OPEN)),
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_CLOSE)),
    ):
        assert mask[ids.index(token_id)] == 0
    assert mask[ids.index(IM_END_ID, ids.index(IM_END_ID) + 1)] == 1


def test_smollm3_completion_mask_accumulates_whole_unicode_multipart(
    sft_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Question?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before😀"},
                    {"type": "python", "text": "x()"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": " 😀"},
                ],
            },
        ]
    }
    ids, mask = sft_tokenizer.render_conversation(conversation)
    assert sft_tokenizer.decode(ids) == (
        "<|begin_of_text|><|im_start|>user\nQuestion?<|im_end|>\n"
        "<|im_start|>assistant\nBefore😀<tool_call>x()</tool_call>"
        "<tool_response>4</tool_response> 😀<|im_end|>\n"
    )
    assert (
        sft_tokenizer.decode(
            [token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled]
        )
        == "Before😀<tool_call>x()</tool_call> 😀<|im_end|>\n"
    )


def test_pythia_golden_template_and_completion_mask(
    pythia_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "python", "text": "x()"},
                    {"type": "python_output", "text": "ok"},
                    {"type": "text", "text": "After"},
                ],
            },
        ]
    }
    snapshot = copy.deepcopy(conversation)
    ids, mask = pythia_tokenizer.render_conversation(conversation)
    assert pythia_tokenizer.decode(ids) == (
        "<|endoftext|>system:\nSystem<|endoftext|>\n"
        "user:\nQuestion?<|endoftext|>\n"
        "assistant:\nBefore<tool_call>x()</tool_call>"
        "\n<tool_response>ok</tool_response>\nAfter<|endoftext|>\n"
    )
    assert (
        pythia_tokenizer.decode(
            [token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled]
        )
        == "Before<tool_call>x()</tool_call>\n\nAfter<|endoftext|>\n"
    )
    prompt = pythia_tokenizer.decode(pythia_tokenizer.render_for_completion(conversation))
    assert prompt.endswith("assistant:\n")
    assert "After" not in prompt
    assert conversation == snapshot


def test_pythia_completion_mask_preserves_leading_space_assistant(
    pythia_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": " answer"},
        ]
    }
    ids, mask = pythia_tokenizer.render_conversation(conversation)
    assert pythia_tokenizer.decode(ids) == (
        "<|endoftext|>user:\nQuestion?<|endoftext|>\nassistant:\n answer<|endoftext|>\n"
    )
    assert (
        pythia_tokenizer.decode(
            [token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled]
        )
        == " answer<|endoftext|>\n"
    )
    encoded = pythia_tokenizer.tokenizer(
        pythia_tokenizer.decode(ids),
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    assert encoded["input_ids"] == ids
    leading_space_token = pythia_tokenizer.encode(" answer")
    assert len(leading_space_token) == 1
    position = ids.index(leading_space_token[0])
    assert encoded["offset_mapping"][position][0] < encoded["offset_mapping"][position][1]
    assert mask[position] == 1


def test_pythia_completion_mask_excludes_gsm8k_tool_response_after_space(
    pythia_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "What is two plus two?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The result = "},
                    {"type": "python", "text": "2 + 2"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "So 4."},
                ],
            },
        ]
    }
    ids, mask = pythia_tokenizer.render_conversation(conversation)
    wire = pythia_tokenizer.decode(ids)
    assert (
        pythia_tokenizer.decode(
            [token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled]
        )
        == "The result = <tool_call>2 + 2</tool_call>\n\nSo 4.<|endoftext|>\n"
    )
    encoded = pythia_tokenizer.tokenizer(
        wire,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    assert encoded["input_ids"] == ids
    response_start = wire.index("<tool_response>")
    response_end = wire.index("</tool_response>", response_start) + len("</tool_response>")
    overlapping = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if start < response_end and end > response_start
    ]
    assert overlapping
    assert all(mask[index] == 0 for index in overlapping)


def test_pythia_completion_mask_resumes_whole_unicode_after_tool_response(
    pythia_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Question?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "python", "text": "x()"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": " 😀"},
                ],
            },
        ]
    }
    ids, mask = pythia_tokenizer.render_conversation(conversation)
    assert pythia_tokenizer.decode(ids) == (
        "<|endoftext|>user:\nQuestion?<|endoftext|>\nassistant:\n"
        "Before<tool_call>x()</tool_call>\n<tool_response>4</tool_response>\n"
        " 😀<|endoftext|>\n"
    )
    assert (
        pythia_tokenizer.decode(
            [token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled]
        )
        == "Before<tool_call>x()</tool_call>\n\n 😀<|endoftext|>\n"
    )


def test_pythia_offset_fix_preserves_token_ids_and_round_trips(tmp_path: Path) -> None:
    tokenizer = _synthetic_pythia_tokenizer()
    samples = [" answer", "The result = <tool_call>", "😀\n text"]
    before = [tokenizer.encode(sample, add_special_tokens=False) for sample in samples]
    adapter = HFSFTTokenizer(tokenizer, PYTHIA_CONTRACT)
    after = [tokenizer.encode(sample, add_special_tokens=False) for sample in samples]
    assert after == before
    state = json.loads(tokenizer.backend_tokenizer.post_processor.__getstate__())
    assert state == {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": True,
    }
    adapter.save(tmp_path)
    reloaded = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False),
    )
    assert [reloaded.encode(sample, add_special_tokens=False) for sample in samples] == before
    reloaded_state = json.loads(reloaded.backend_tokenizer.post_processor.__getstate__())
    assert reloaded_state == state


def test_pythia_rejects_incompatible_offset_processors() -> None:
    missing = _synthetic_pythia_tokenizer()
    missing.backend_tokenizer.post_processor = None
    with pytest.raises(ValueError, match="must use a ByteLevel post-processor"):
        HFSFTTokenizer(missing, PYTHIA_CONTRACT)

    drifted = _synthetic_pythia_tokenizer()
    drifted.backend_tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=True,
        trim_offsets=True,
        use_regex=True,
    )
    with pytest.raises(ValueError, match="post-processor contract drifted"):
        HFSFTTokenizer(drifted, PYTHIA_CONTRACT)


def test_pythia_leading_space_survives_completion_only_packing(
    pythia_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": " answer"},
        ]
    }
    ids, _mask = pythia_tokenizer.render_conversation(conversation)
    generator = sft_data_generator_bos_bestfit_core(
        dataset=[conversation],
        tokenizer=pythia_tokenizer,
        max_seq_len=len(ids) - 1,
        device_batch_size=1,
        ddp_rank=0,
        ddp_world_size=1,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=1,
        loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
    )
    _inputs, targets, progress = next(generator)
    supervised_targets = targets[0][targets[0] != -1].tolist()
    assert pythia_tokenizer.decode(supervised_targets) == " answer<|endoftext|>\n"
    assert progress.consumed == 1


def test_pythia_saved_template_round_trip(
    pythia_tokenizer: HFSFTTokenizer,
    tmp_path: Path,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "Hello 😀"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    expected = pythia_tokenizer.render_conversation(conversation)
    pythia_tokenizer.save(tmp_path)
    reloaded = HFSFTTokenizer(
        cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False),
        ),
        PYTHIA_CONTRACT,
    )
    assert reloaded.render_conversation(conversation) == expected


def test_system_role_and_generation_prompt_do_not_mutate_input(
    sft_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    snapshot = copy.deepcopy(conversation)
    ids, mask = sft_tokenizer.render_conversation(conversation)
    assert sft_tokenizer.decode(ids) == (
        "<|begin_of_text|><|im_start|>system\nsystem<|im_end|>\n"
        "<|im_start|>user\nquestion<|im_end|>\n"
        "<|im_start|>assistant\nanswer<|im_end|>\n"
    )
    exported = cast(
        dict[str, list[int]],
        sft_tokenizer.tokenizer.apply_chat_template(
            conversation["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        ),
    )
    assert exported["input_ids"] == ids
    assert exported["assistant_masks"] == mask
    prompt_ids = sft_tokenizer.render_for_completion(conversation)
    exported_prompt_ids = sft_tokenizer.tokenizer.apply_chat_template(
        conversation["messages"][:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        return_assistant_tokens_mask=False,
    )
    assert exported_prompt_ids == prompt_ids
    prompt = sft_tokenizer.decode(prompt_ids)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "answer" not in prompt
    assert conversation == snapshot


def test_multiturn_and_empty_assistant_wire_text(
    sft_tokenizer: HFSFTTokenizer,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "first\nline"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": ""},
        ]
    }
    ids, mask = sft_tokenizer.render_conversation(conversation)
    assert sft_tokenizer.decode(ids) == (
        "<|begin_of_text|><|im_start|>user\nfirst\nline<|im_end|>\n"
        "<|im_start|>assistant\none<|im_end|>\n"
        "<|im_start|>user\nsecond<|im_end|>\n"
        "<|im_start|>assistant\n<|im_end|>\n"
    )
    assert (
        sft_tokenizer.decode(
            [token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled]
        )
        == "one<|im_end|>\n<|im_end|>\n"
    )


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        ["not a message"],
        [{"role": "assistant", "content": "bad"}],
        [{"role": "system", "content": "s"}],
        [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
        [{"role": "system", "content": []}, {"role": "user", "content": "u"}],
        [{"role": "user", "content": []}],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": 3},
        ],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": ["not a part"]},
        ],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": [{"type": "shell", "text": "x"}]},
        ],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": [{"type": "text", "text": 3}]},
        ],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "again"},
        ],
    ],
)
def test_invalid_conversation_shapes_fail(
    sft_tokenizer: HFSFTTokenizer,
    messages,
) -> None:
    with pytest.raises(ValueError):
        sft_tokenizer.render_conversation({"messages": messages})


def test_saved_tokenizer_round_trip_is_identical(
    sft_tokenizer: HFSFTTokenizer,
    tmp_path: Path,
) -> None:
    conversation = {
        "messages": [
            {"role": "user", "content": "line 1\nline 2"},
            {"role": "assistant", "content": ""},
        ]
    }
    expected = sft_tokenizer.render_conversation(conversation)
    sft_tokenizer.save(tmp_path)
    reloaded = HFSFTTokenizer(
        cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False),
        ),
        SMOLLM3_CONTRACT,
    )
    assert reloaded.render_conversation(conversation) == expected
    assert len(reloaded.tokenizer) == 128256


def test_structural_controls_have_zero_bpb_bytes(
    sft_tokenizer: HFSFTTokenizer,
) -> None:
    structural: set[int] = {
        BEGIN_OF_TEXT_ID,
        FINETUNE_PAD_ID,
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(IM_START)),
        IM_END_ID,
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(TOOL_CALL_OPEN)),
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(TOOL_CALL_CLOSE)),
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_OPEN)),
        cast(int, sft_tokenizer.tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_CLOSE)),
    }
    token_bytes = compute_hf_token_bytes(
        sft_tokenizer,
        additional_zero_ids=structural,
    )
    assert all(token_bytes[token_id] == 0 for token_id in structural)
    ascii_id = sft_tokenizer.encode("A")[0]
    assert token_bytes[ascii_id] == 1
    emoji_ids = sft_tokenizer.encode("😀")
    assert sum(token_bytes[token_id] for token_id in emoji_ids) == len("😀".encode())
    assert sum(
        len(sft_tokenizer.decode([token_id]).encode("utf-8")) for token_id in emoji_ids
    ) > len("😀".encode())


def test_pythia_bytelevel_bpb_and_special_ids(
    pythia_tokenizer: HFSFTTokenizer,
) -> None:
    token_bytes = compute_hf_token_bytes(
        pythia_tokenizer,
        additional_zero_ids=PYTHIA_CONTRACT.structural_token_ids.values(),
    )
    assert token_bytes[PYTHIA_END_OF_TEXT_ID] == 0
    assert token_bytes[PYTHIA_PAD_ID] == 0
    assert token_bytes[pythia_tokenizer.encode("A")[0]] == 1
    emoji_ids = pythia_tokenizer.encode("😀")
    assert sum(token_bytes[token_id] for token_id in emoji_ids) == 4


@pytest.mark.parametrize("factory", [_tiny_model, _tiny_pythia_model])
def test_model_adapter_matches_direct_cross_entropy_and_shifts_once(factory) -> None:
    t.manual_seed(0)
    model = factory()
    adapter = HFDecoderModelAdapter(model, max_seq_len=8)
    inputs = t.tensor([[1, 2, 3, 4]], dtype=t.int32)
    targets = t.tensor([[2, 3, -1, 5]], dtype=t.int64)
    with patch.object(model, "forward", wraps=model.forward) as forward:
        logits = adapter(inputs)
    assert forward.call_args.kwargs["use_cache"] is False
    reference = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-1,
    )
    assert t.equal(adapter(inputs, targets), reference)
    per_token = adapter(inputs, targets, loss_reduction="none")
    assert per_token.shape == (4,)
    assert per_token[2].item() == 0.0


@pytest.mark.parametrize(
    "targets",
    [
        t.tensor([[2, 3, 4]], dtype=t.int64),
        t.tensor([[-1, 3, -1]], dtype=t.int64),
    ],
)
def test_existing_bpb_evaluator_matches_manual_nats_per_byte(targets: t.Tensor) -> None:
    model = _tiny_model()
    adapter = HFDecoderModelAdapter(model, max_seq_len=8)
    inputs = t.tensor([[1, 2, 3]], dtype=t.int32)
    token_bytes = t.zeros(128256, dtype=t.int32)
    token_bytes[2] = 1
    token_bytes[3] = 2
    token_bytes[4] = 4
    losses = adapter(inputs, targets, loss_reduction="none")
    valid = targets.reshape(-1) >= 0
    safe_targets = t.where(valid, targets.reshape(-1), 0)
    byte_lengths = t.where(valid, token_bytes[safe_targets], 0)
    expected = float(
        (losses * (byte_lengths > 0)).sum().item() / (math.log(2) * byte_lengths.sum().item())
    )
    actual = evaluate_bpb(adapter, [(inputs, targets)], 1, token_bytes)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("factory", [_tiny_model, _tiny_pythia_model])
def test_gradient_checkpointing_backward_reaches_embeddings(factory) -> None:
    model = factory()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter = HFDecoderModelAdapter(model, max_seq_len=8)
    loss = adapter(
        t.tensor([[1, 2, 3]], dtype=t.int32),
        t.tensor([[2, 3, 4]], dtype=t.int64),
    )
    loss.backward()
    assert t.isfinite(loss)
    input_embedding = cast(t.nn.Embedding, model.get_input_embeddings())
    assert input_embedding.weight.grad is not None
    assert t.isfinite(input_embedding.weight.grad).all()
    output_embedding = cast(t.nn.Linear, model.get_output_embeddings())
    assert output_embedding.weight.grad is not None
    assert t.isfinite(output_embedding.weight.grad).all()


@pytest.mark.parametrize("factory", [_tiny_model, _tiny_pythia_model])
def test_hf_optimizer_is_one_exact_adamw_group(factory) -> None:
    model = factory()
    adapter = HFDecoderModelAdapter(model, max_seq_len=8)
    optimizer = build_hf_optimizer(
        adapter,
        ddp=False,
        learning_rate=2e-5,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        init_lr_frac=1.0,
    )
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["kind"] == "adamw"
    assert group["lr"] == 2e-5
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == 1e-10
    assert len(group["params"]) == len({id(parameter) for parameter in model.parameters()})


def test_single_process_update_matches_plain_adamw() -> None:
    # The vendored fused AdamW step is shape-specialized. Keep this parity test
    # independent of specializations compiled by earlier tests in the process.
    t.compiler.reset()
    config = SmolLM3Config(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = SmolLM3ForCausalLM(config).float()
    reference = SmolLM3ForCausalLM(config).float()
    reference.load_state_dict(model.state_dict())
    adapter = HFDecoderModelAdapter(model, max_seq_len=8)
    optimizer = build_hf_optimizer(
        adapter,
        ddp=False,
        learning_rate=2e-5,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        init_lr_frac=1.0,
    )
    reference_optimizer = t.optim.AdamW(
        reference.parameters(),
        lr=2e-5,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
    )
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        gradient = t.linspace(-0.5, 0.5, parameter.numel()).reshape_as(parameter)
        parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
    optimizer.step()
    reference_optimizer.step()
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert t.allclose(parameter, reference_parameter, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_clipping_values_fail(bad) -> None:
    with pytest.raises(ValueError):
        resolve_grad_clip_max_norm(bad)


def test_disabled_and_nonfinite_clipping_contract() -> None:
    parameter = t.nn.Parameter(t.ones(2))
    parameter.grad = t.tensor([3.0, 4.0])
    snapshot = parameter.grad.clone()
    assert apply_gradient_clipping([parameter], None) is None
    assert t.equal(parameter.grad, snapshot)
    parameter.grad[0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        apply_gradient_clipping([parameter], 1.0)


def test_bf16_export_is_atomic_reloadable_and_nonmutating(
    sft_tokenizer: HFSFTTokenizer,
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    configure_generation_contract(model, SMOLLM3_CONTRACT)
    config_snapshot = model.config.to_dict()
    generation_config_snapshot = model.generation_config.to_dict()
    snapshots = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cpu_state = build_cpu_bf16_state_dict(model)
    assert cpu_state["model.embed_tokens.weight"] is cpu_state["lm_head.weight"]
    output = publish_hf_checkpoint(
        output_root=tmp_path,
        completed_step=3,
        model=model,
        tokenizer=sft_tokenizer,
        metadata={"format_version": 2, "resumable": False},
        max_shard_size="20MB",
    )
    assert output.name == "step_000003"
    assert list(tmp_path.glob(".*.tmp-*")) == []
    assert list(tmp_path.glob(".*.lock")) == []
    for name, value in model.state_dict().items():
        assert t.equal(value, snapshots[name])
        assert value.dtype == t.float32
    assert model.config.to_dict() == config_snapshot
    assert model.generation_config.to_dict() == generation_config_snapshot
    reloaded = AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=False,
        dtype=t.bfloat16,
    )
    assert reloaded.config.dtype == t.bfloat16
    assert {p.dtype for p in reloaded.parameters() if p.is_floating_point()} == {t.bfloat16}
    assert reloaded.get_input_embeddings().weight is reloaded.get_output_embeddings().weight
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(output, trust_remote_code=False),
    )
    assert tokenizer.chat_template
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (
        BEGIN_OF_TEXT_ID,
        IM_END_ID,
        FINETUNE_PAD_ID,
    )
    validate_export_smoke(
        output,
        contract=SMOLLM3_CONTRACT,
        device="cpu",
        max_new_tokens=4,
    )
    with pytest.raises(FileExistsError):
        publish_hf_checkpoint(
            output_root=tmp_path,
            completed_step=3,
            model=model,
            tokenizer=sft_tokenizer,
            metadata={"format_version": 2, "resumable": False},
        )


def test_pythia_bf16_export_preserves_untied_embeddings(
    pythia_tokenizer: HFSFTTokenizer,
    tmp_path: Path,
) -> None:
    model = _tiny_pythia_model()
    configure_generation_contract(model, PYTHIA_CONTRACT)
    with t.no_grad():
        model.lm_head.weight.zero_()
    snapshots = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cpu_state = build_cpu_bf16_state_dict(model)
    assert cpu_state["gpt_neox.embed_in.weight"] is not cpu_state["lm_head.weight"]
    output = publish_hf_checkpoint(
        output_root=tmp_path,
        completed_step=2,
        model=model,
        tokenizer=pythia_tokenizer,
        metadata={"format_version": 2, "resumable": False},
        max_shard_size="20MB",
    )
    for name, value in model.state_dict().items():
        assert t.equal(value, snapshots[name])
        assert value.dtype == t.float32
    reloaded = AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=False,
        dtype=t.bfloat16,
    )
    assert isinstance(reloaded, GPTNeoXForCausalLM)
    assert reloaded.get_input_embeddings().weight is not reloaded.get_output_embeddings().weight
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(output, trust_remote_code=False),
    )
    assert len(tokenizer) == PYTHIA_CONTRACT.tokenizer_vocab_size
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (
        PYTHIA_END_OF_TEXT_ID,
        PYTHIA_END_OF_TEXT_ID,
        PYTHIA_PAD_ID,
    )
    validate_export_smoke(
        output,
        contract=PYTHIA_CONTRACT,
        device="cpu",
        max_new_tokens=4,
    )


def test_export_failure_cleans_writer_owned_temp_lock_and_final(
    sft_tokenizer: HFSFTTokenizer,
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    configure_generation_contract(model, SMOLLM3_CONTRACT)

    with patch.object(model, "save_pretrained", side_effect=RuntimeError("injected save")):
        with pytest.raises(RuntimeError, match="injected save"):
            publish_hf_checkpoint(
                output_root=tmp_path,
                completed_step=4,
                model=model,
                tokenizer=sft_tokenizer,
                metadata={"format_version": 2, "resumable": False},
            )
    assert not (tmp_path / "step_000004").exists()
    assert list(tmp_path.glob(".step_000004.*")) == []

    real_rename = Path.rename

    def rename_then_fail(source: Path, target: Path) -> Path:
        real_rename(source, target)
        raise RuntimeError("injected publication")

    with patch.object(Path, "rename", rename_then_fail):
        with pytest.raises(RuntimeError, match="injected publication"):
            publish_hf_checkpoint(
                output_root=tmp_path,
                completed_step=5,
                model=model,
                tokenizer=sft_tokenizer,
                metadata={"format_version": 2, "resumable": False},
            )
    assert not (tmp_path / "step_000005").exists()
    assert list(tmp_path.glob(".step_000005.*")) == []
