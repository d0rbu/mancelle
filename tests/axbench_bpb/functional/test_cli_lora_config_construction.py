"""Lock the lora YAML schema against the LoraConfig dataclass.

The lora: block in configs/eval/axbench_bpb/axbench_bpb_eval.yaml must
produce a valid LoraConfig via ``LoraConfig(**lora_dict)`` (the splat
performed by scripts/axbench_bpb_eval.py:353). These tests pin both
ends of the contract: silent schema drift between the hand-edited YAML
and the dataclass would otherwise crash deep at runtime when the
intervention dispatch first instantiates a LoRA-kind method. CI fails
immediately if either side adds, renames, or removes a field that the
other does not match.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from steering.eval.axbench_base.config import LoraConfig

# tests/axbench_bpb/functional/test_cli_lora_config_construction.py ->
#   .parent (functional) -> .parent (axbench_bpb) -> .parent (tests)
#   -> .parent (repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_YAML_PATH = (
    _REPO_ROOT / "configs" / "eval" / "axbench_bpb" / "axbench_bpb_eval.yaml"
)


def _load_lora_block() -> dict:
    raw = yaml.safe_load(_YAML_PATH.read_text())
    return raw["lora"]


def test_default_yaml_lora_block_constructs_valid_loraconfig() -> None:
    """Round-trip: YAML lora block -> ``LoraConfig(**block)`` does not raise.

    Mirrors the splat in scripts/axbench_bpb_eval.py:353. Verifies the
    canonical fields actually propagated rather than being silently
    swallowed by ``**kwargs``-style absorption.
    """
    lora_block = _load_lora_block()
    cfg = LoraConfig(**lora_block)
    assert cfg.rank == lora_block["rank"]
    assert cfg.alpha == lora_block["alpha"]


def test_loraconfig_no_required_fields_missing_from_yaml() -> None:
    """Required LoraConfig fields must be covered by the YAML.

    A required field (no default, no default_factory) that the YAML does
    not provide would make ``LoraConfig(**lora_block)`` raise TypeError
    at runtime as soon as a LoRA-kind intervention is built. Catch it at
    test time instead.
    """
    lora_block = _load_lora_block()
    yaml_keys = set(lora_block.keys())
    required_fields = {
        f.name
        for f in dataclasses.fields(LoraConfig)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }
    missing = required_fields - yaml_keys
    assert not missing, (
        f"LoraConfig has required fields not in axbench_bpb_eval.yaml's "
        f"lora block: {sorted(missing)}. Either add them to the YAML or "
        f"give them a default in LoraConfig."
    )


def test_yaml_lora_block_no_unknown_fields() -> None:
    """Every YAML lora field must correspond to a LoraConfig dataclass field.

    A YAML key that LoraConfig does not declare would make the
    ``LoraConfig(**lora_block)`` splat raise ``TypeError`` at runtime.
    Lock it here so the failure surfaces in CI instead of mid-eval.
    """
    lora_block = _load_lora_block()
    yaml_keys = set(lora_block.keys())
    valid_fields = {f.name for f in dataclasses.fields(LoraConfig)}
    extras = yaml_keys - valid_fields
    assert not extras, (
        f"YAML lora block has fields not on LoraConfig: {sorted(extras)}. "
        f"Either remove them from the YAML or add them to the dataclass."
    )
