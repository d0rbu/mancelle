"""Execute the vendor transport adapter with CPU doubles for external model classes."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, list]:
    calls = []

    class FakeMethod:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.ax = SimpleNamespace(proj=t.nn.Linear(3, 1))
            self.ax.to = lambda **kwargs: self.ax

        def make_model(self, **kwargs: Any) -> None:
            calls.append(("make", kwargs))

        def train(self, frame: pd.DataFrame, **kwargs: Any) -> None:
            calls.append(("train", frame.copy(), kwargs))

        def predict_latent(self, frame: pd.DataFrame, **kwargs: Any) -> dict:
            calls.append(("latent", kwargs))
            return {"max_act": [2.0] * len(frame)}

        def save(self, path: Path) -> None:
            (path / "DiffMean_weight.pt").write_bytes(b"weights")

        def load(self, path: Path, **kwargs: Any) -> None:
            calls.append(("load", kwargs))

        def predict_steer(self, frame: pd.DataFrame, **kwargs: Any) -> dict:
            calls.append(("generate", frame.copy(), kwargs))
            return {"steered_generation": ["response"]}

    for name in ("axbench", "axbench.models", "axbench.models.mean", "axbench.models.prompt"):
        module: Any = ModuleType(name)
        module.DiffMean = FakeMethod
        module.PromptSteering = FakeMethod
        monkeypatch.setitem(sys.modules, name, module)
    module = runpy.run_path(str(ROOT / "scripts/eval/campaign2_axbench_worker.py"))
    return module, calls


class FakeTokenizer:
    chat_template = "test template"
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 1

    def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
        return "BOS " + " | ".join(message["content"] for message in messages)

    def __call__(self, text: str, **kwargs: Any) -> dict:
        return {"input_ids": text.split()}


@pytest.mark.parametrize("variant", ["pythia_diffmean", "pythia_prompt"])
def test_worker_passes_rendered_chat_to_vendor_and_preserves_original_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker: tuple[dict, list], variant: str,
) -> None:
    module, calls = worker
    tokenizer = FakeTokenizer()
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048,
        bos_token_id=0, eos_token_id=0, pad_token_id=1), eval=lambda: None)
    model.to = lambda *args: model
    monkeypatch.setattr(module["AutoTokenizer"], "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(module["AutoModelForCausalLM"], "from_pretrained", lambda *args, **kwargs: model)
    (tmp_path / "sft_metadata.json").write_text(json.dumps({
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}))
    request = {"variant": variant, "checkpoint_dir": str(tmp_path), "output_dir": str(tmp_path),
        "training_pairs": [{"prompt": "training question", "safe_response": "safe", "unsafe_response": "unsafe"}],
        "instruction": "Safety instruction", "device": "cpu", "layer": 25, "factor": 0.5,
        "fit_batch_size": 4, "seed": 42, "temperature": 0.7, "top_k": 50, "max_new_tokens": 512,
        "rows": [{"index": 0, "seed": 43, "row": {"prompt": "eval question", "pool": "harmful", "category": "test"}}]}
    module["run"](request)
    output = json.loads((tmp_path / "records/000000.json").read_text())
    assert output["prompt"] == "eval question" and output["response"] == "response"
    generations = [call for call in calls if call[0] == "generate"]
    text = generations[0][1].iloc[0]["steered_input"]
    assert text.startswith("BOS ")
    assert ("Safety instruction" in text) == (variant == "pythia_prompt")
    assert generations[0][2]["compute_perplexity"] is False
    if variant == "pythia_diffmean":
        train = next(call for call in calls if call[0] == "train")
        assert train[1]["labels"].tolist() == [1, 0]
        assert train[2] == {"prefix_length": 1}
        assert json.loads((tmp_path / "diffmean/fit.json").read_text())["max_act"] == 2.0
        # Resume reuses the checkpoint-specific fit without appending another direction.
        module["fit_direction"](request, model, tokenizer, tmp_path)
        assert len([call for call in calls if call[0] == "train"]) == 1


def test_worker_rejects_truncation(worker: tuple[dict, list]) -> None:
    module, _ = worker
    with pytest.raises(ValueError, match="refusing truncation"):
        module["render"](FakeTokenizer(), [{"role": "user", "content": "too long"}], generation=True, limit=1)


def test_worker_reduces_transformers_seed_deterministically(worker: tuple[dict, list]) -> None:
    module, _ = worker
    modulus = module["TRANSFORMERS_SEED_MODULUS"]
    assert module["transformers_seed"](0) == 0
    assert module["transformers_seed"](modulus + 7) == 7
    with pytest.raises(ValueError, match="non-negative"):
        module["transformers_seed"](-1)
