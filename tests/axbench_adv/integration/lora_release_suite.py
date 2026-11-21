"""Vendored-environment suite: a loaded LoRA adapter must not outlive its concept."""

from __future__ import annotations

from pathlib import Path

import torch
from axbench.models.lora import LoRA
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


def _tiny_model() -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(config).eval()


def _save_adapter(dump_dir: Path, concept_id: int, seed: int) -> None:
    """Write a non-zero adapter where LoRA.load looks for this concept."""
    model = _tiny_model()
    torch.manual_seed(seed)
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["dense"],
        layers_to_transform=[0, 1],
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    get_peft_model(model, config).save_pretrained(dump_dir / "lora" / str(concept_id))


def _has_lora(model: torch.nn.Module) -> bool:
    return any(isinstance(module, LoraLayer) for module in model.modules())


def _logits(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.arange(8).unsqueeze(0)).logits


def test_release_restores_the_shared_model_between_concepts(tmp_path: Path) -> None:
    for concept_id, seed in ((0, 1), (1, 2)):
        _save_adapter(tmp_path, concept_id, seed)
    reference = {}
    for concept_id in (0, 1):
        alone = PeftModel.from_pretrained(_tiny_model(), tmp_path / "lora" / str(concept_id))
        reference[concept_id] = _logits(alone)

    shared = _tiny_model()
    base = _logits(shared)
    for concept_id in (0, 1):
        lora = LoRA(shared, None, layer=0, device="cpu")
        lora.load(dump_dir=tmp_path, concept_id=concept_id)
        # The adapter lives in the shared instance, so anything generated with it now is steered.
        assert _has_lora(shared)
        assert torch.allclose(_logits(lora.ax_model), reference[concept_id])
        lora.release()
        assert not _has_lora(shared)
        assert torch.equal(_logits(shared), base)


def test_release_without_a_loaded_adapter_is_a_no_op() -> None:
    shared = _tiny_model()
    base = _logits(shared)
    LoRA(shared, None, layer=0, device="cpu").release()
    assert not _has_lora(shared)
    assert torch.equal(_logits(shared), base)
