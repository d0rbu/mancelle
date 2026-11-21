"""Generate committed AxBench encdec numeric-parity fixtures once."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import torch as t
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
# Dev-only exception: the isolated AxBench project is not installed as a package.
sys.path.insert(0, str(REPO_ROOT / "vendor" / "axbench"))

from axbench.utils.model_utils import set_decoder_norm_to_unit_norm  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "axbench_encdec"
VENDORED_COMMIT = "41c8332543e5a631f9a8c0a9df38799893ace758"

ACTIVATION_SEED = 2026082501
LORA_BASE_SEED = 2026082503
LORA_INIT_SEED = 1234
LORA_TRAINED_WEIGHT_SEED = 2026082504
LORA_INPUT_SEED = 2026082505

WIDTH = 64
ROW_LENGTHS = (3, 4, 5, 6, 7, 8, 9, 3)


class _DirectionProxy(nn.Module):
    """Minimal shape expected by the real vendored direction helpers."""

    def __init__(self, width: int, *, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 1, bias=bias)


class _LoRAFixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 16, bias=False)

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return self.linear(inputs)


def _activation_rows() -> tuple[list[t.Tensor], list[t.Tensor]]:
    t.manual_seed(ACTIVATION_SEED)
    positive = [0.1 * t.randn(length, WIDTH) + 0.02 for length in ROW_LENGTHS]
    negative = [0.1 * t.randn(length, WIDTH) - 0.01 for length in ROW_LENGTHS]
    return positive, negative


def _diffmean_fixture(
    positive: list[t.Tensor],
    negative: list[t.Tensor],
) -> dict[str, t.Tensor]:
    raw = t.cat(positive).mean(dim=0) - t.cat(negative).mean(dim=0)
    proxy = _DirectionProxy(WIDTH, bias=False)
    with t.no_grad():
        proxy.proj.weight.copy_(raw.unsqueeze(0))
    set_decoder_norm_to_unit_norm(proxy)
    tensors = {f"positive_row_{index}": row for index, row in enumerate(positive)}
    tensors.update({f"negative_row_{index}": row for index, row in enumerate(negative)})
    tensors["raw"] = raw
    tensors["direction"] = proxy.proj.weight.detach().squeeze(0).clone()
    return tensors


def _lora_fixture() -> dict[str, t.Tensor]:
    t.manual_seed(LORA_BASE_SEED)
    base = _LoRAFixtureModel()
    base_weight = base.linear.weight.detach().clone()

    t.manual_seed(LORA_INIT_SEED)
    wrapped = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=32,
            use_rslora=True,
            lora_dropout=0.05,
            target_modules=["linear"],
            bias="none",
        ),
    )
    layer = wrapped.base_model.model.linear
    lora_a = layer.lora_A["default"].weight
    lora_b = layer.lora_B["default"].weight
    initial_a = lora_a.detach().clone()

    t.manual_seed(LORA_TRAINED_WEIGHT_SEED)
    with t.no_grad():
        lora_a.copy_(t.randn_like(lora_a))
        lora_b.copy_(t.randn_like(lora_b))
    t.manual_seed(LORA_INPUT_SEED)
    inputs = t.randn(5, 32)
    wrapped.eval()
    with t.no_grad():
        output = wrapped(inputs)
    return {
        "base_weight": base_weight,
        "lora_A_initial": initial_a,
        "lora_A": lora_a.detach().clone(),
        "lora_B": lora_b.detach().clone(),
        "input": inputs,
        "output": output,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    positive, negative = _activation_rows()
    fixtures_to_write = {
        "diffmean_fixture.safetensors": _diffmean_fixture(positive, negative),
        "lora_fixture.safetensors": _lora_fixture(),
    }
    for filename, tensors in fixtures_to_write.items():
        save_file(tensors, FIXTURE_DIR / filename)

    fixture_filenames = [
        "diffmean_fixture.safetensors",
        "lora_fixture.safetensors",
    ]
    missing = [filename for filename in fixture_filenames if not (FIXTURE_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"missing fixture files after generation: {missing}")

    versions = {
        package: importlib.metadata.version(package)
        for package in ("peft", "safetensors", "torch", "transformers")
    }
    manifest = {
        "format_version": 1,
        "generator": "dev/generate_axbench_encdec_fixtures.py",
        "generator_args": [],
        "generator_command": (
            "uv run --project vendor/axbench --frozen python "
            "dev/generate_axbench_encdec_fixtures.py"
        ),
        "seeds": {
            "activation_rows": ACTIVATION_SEED,
            "lora_base_weight": LORA_BASE_SEED,
            "lora_peft_A_initialization": LORA_INIT_SEED,
            "lora_fixture_A_B": LORA_TRAINED_WEIGHT_SEED,
            "lora_input": LORA_INPUT_SEED,
        },
        "library_versions": versions,
        "vendored_commit": VENDORED_COMMIT,
        "fixture_parameters": {
            "dtype": "float32",
            "device": "cpu",
            "diffmean": {
                "class_rows": len(ROW_LENGTHS),
                "row_lengths": list(ROW_LENGTHS),
                "width": WIDTH,
            },
            "lora": {
                "in_features": 32,
                "out_features": 16,
                "rank": 4,
                "alpha": 32,
                "scaling": 32 / math.sqrt(4),
                "dropout": 0.05,
                "eval_mode": True,
            },
        },
        "files": {
            filename: {"sha256": _sha256(FIXTURE_DIR / filename)} for filename in fixture_filenames
        },
    }
    (FIXTURE_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
