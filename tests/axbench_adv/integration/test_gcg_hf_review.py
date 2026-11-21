"""Root-Python and vendored-Python CPU review oracles for Hugging Face GCG."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch as t
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scripts.eval.axbench_adv.merge_slices import _validate_gcg_calibration
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS
from steering.eval.axbench_adv.worker_client import _FramedProcess, _load_tensor_sidecar

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_SUITE = Path(__file__).with_name("gcg_hf_review_suite.py")


def _load_worker_with_fakes() -> ModuleType:
    """Load the worker math without importing the vendored AxBench package."""

    module_name = "_independent_gcg_hf_review_worker"
    cached = sys.modules.get(module_name)
    if isinstance(cached, ModuleType):
        return cached
    axbench = ModuleType("axbench")
    utils = ModuleType("axbench.utils")
    model_utils = ModuleType("axbench.utils.model_utils")
    cast(Any, model_utils).decoder_layers = lambda model: model.layers
    pyvene = ModuleType("pyvene")
    cast(Any, pyvene).CollectIntervention = object
    cast(Any, pyvene).IntervenableConfig = object
    cast(Any, pyvene).IntervenableModel = object

    class _FakeDirection(t.nn.Module):
        def __init__(self, weight: t.Tensor) -> None:
            super().__init__()
            self.proj = t.nn.Linear(weight.shape[1], weight.shape[0])
            self.proj.weight.data.copy_(weight)

    class _FakeDiffMean:
        def __init__(self, model: t.nn.Module, tokenizer: object, **_kwargs: object) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.max_activations: dict[int, float] = {}

        def load(self, dump_dir: Path, **_kwargs: object) -> None:
            weight = t.load(Path(dump_dir) / "DiffMean_weight.pt", weights_only=True)
            self.ax = _FakeDirection(weight)

        def to(self, device: str) -> None:
            self.ax.to(device)

        def pre_compute_mean_activations(self, dump_dir: Path) -> None:
            frame = pd.read_parquet(Path(dump_dir) / "latent_data.parquet")
            self.max_activations = {
                int(concept_id): float(raw) if float(raw) > 0.0 else 50.0
                for concept_id, raw in frame.groupby("concept_id")["DiffMean_max_act"].max().items()
            }

    cast(Any, axbench).DiffMean = _FakeDiffMean
    names = ("axbench", "axbench.utils", "axbench.utils.model_utils", "pyvene")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(
        {
            "axbench": axbench,
            "axbench.utils": utils,
            "axbench.utils.model_utils": model_utils,
            "pyvene": pyvene,
        }
    )
    path = REPO_ROOT / "scripts/eval/axbench_adv/vendored/gcg_worker.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return module


class _ByteLevelAdapter:
    all_special_ids: tuple[int, ...] = ()
    added_tokens_decoder: dict[int, object] = {}
    eos_token_id: int | None = None
    padding_side = "right"

    def __init__(self) -> None:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            ["alpha hello world", "beta hello world", "gamma hello again"],
            trainers.BpeTrainer(vocab_size=128, special_tokens=["[UNK]"]),
        )
        self._tokenizer = tokenizer

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": self._tokenizer.encode(text, add_special_tokens=False).ids}

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)


def test_real_bytelevel_outer_space_candidates_are_filtered() -> None:
    module = _load_worker_with_fakes()
    tokenizer = _ByteLevelAdapter()
    leading_ids = [
        token_id
        for token_id in range(tokenizer._tokenizer.get_vocab_size())
        if tokenizer.decode([token_id]).startswith(" ") and tokenizer.decode([token_id]).strip()
    ]
    whitespace_ids = [
        token_id
        for token_id in range(tokenizer._tokenizer.get_vocab_size())
        if tokenizer.decode([token_id]) and not tokenizer.decode([token_id]).strip()
    ]
    assert leading_ids and whitespace_ids
    session = module.GCGSession(
        SimpleNamespace(
            tokenizer=tokenizer,
            benchmark_model=SimpleNamespace(model=t.nn.Linear(1, 1)),
        )
    )
    session.objective = "response_margin"
    session.seed_token_ids = (leading_ids[0],)
    session._allowed_token_id_set = frozenset((leading_ids[0], whitespace_ids[0]))
    session.layouts = ()
    rows = session.filter({"candidate_token_ids": [[leading_ids[0]], [whitespace_ids[0]]]})[
        "candidates"
    ]
    assert [row["valid"] for row in rows] == [False, False]
    assert all("outer-spaced" in row["error"] for row in rows)


@pytest.mark.parametrize("factor", ADV_FULL_FACTORS)
@pytest.mark.parametrize("concept_id", [0, 1])
@pytest.mark.parametrize(
    ("model_type", "model_name", "revision"),
    [
        (None, "Qwen/Qwen2.5-3B-Instruct", None),
        ("qwen2", "Qwen/Qwen2.5-3B-Instruct", None),
        ("smollm3", "HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1"),
        ("gpt_neox", "/local/pythia-sft", None),
    ],
)
@pytest.mark.parametrize(
    ("values", "expected_raw", "expected_resolved", "expected_fallback"),
    [
        ([3.25, 2.0], 3.25, 3.25, False),
        ([0.0, -4.0], 0.0, 50.0, True),
        ([float("nan"), float("nan")], None, 50.0, True),
    ],
)
def test_calibration_records_raw_resolved_fallback_and_exact_float32_offset(
    tmp_path: Path,
    factor: float,
    concept_id: int,
    model_type: str | None,
    model_name: str,
    revision: str | None,
    values: list[float],
    expected_raw: float | None,
    expected_resolved: float,
    expected_fallback: bool,
) -> None:
    module = _load_worker_with_fakes()
    train_dir = tmp_path / "train"
    inference_dir = tmp_path / "inference"
    train_dir.mkdir()
    inference_dir.mkdir()
    direction = t.tensor(
        [[1.1171875, -0.62109375, 0.30859375], [-0.476, 1.121, 0.783]], dtype=t.float32
    )
    t.save(direction, train_dir / "DiffMean_weight.pt")
    pd.DataFrame(
        {
            "concept_id": [0] * len(values) + [1] * len(values),
            "DiffMean_max_act": [*values, *(value * 1.4 for value in values)],
        }
    ).to_parquet(inference_dir / "latent_data.parquet", index=False)
    if concept_id == 1:
        expected_raw = None if expected_raw is None else expected_raw * 1.4
        if not expected_fallback:
            expected_resolved *= 1.4
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=t.nn.Linear(1, 1)),
        concept_id=concept_id,
        device="cpu",
        factor=factor,
        layer=0,
        max_concepts=2,
        method="PromptSteering",
        tokenizer=SimpleNamespace(padding_side="left"),
        _gcg_main_dump_dir=tmp_path,
    )
    if model_type is not None:
        runtime.benchmark_model.model.config = SimpleNamespace(model_type=model_type)
    session = module.GCGSession(runtime)
    calibration = session._prepare_activation()
    assert calibration["raw_max_activation"] == expected_raw
    assert calibration["max_activation"] == expected_resolved
    assert calibration["resolved_scale"] == expected_resolved * factor
    assert calibration["fallback_used"] is expected_fallback
    assert calibration["factor"] == factor
    expected_offset = (
        t.tensor([expected_resolved], dtype=t.float32).unsqueeze(-1)
        * t.tensor([factor], dtype=t.float32).unsqueeze(-1)
        * direction[concept_id].to(t.bfloat16).unsqueeze(0)
    )[0]
    assert session._direction_offset is not None
    assert session._direction_offset.dtype is t.float32
    assert t.equal(session._direction_offset, expected_offset)
    identity = {"model_name": model_name, "model_revision": revision, "factor": factor}
    _validate_gcg_calibration(
        calibration,
        objective="activation_l2",
        concept_id=concept_id,
        model_identity=identity,
        relative=Path("search.json"),
    )
    wrong_site = (
        "model.layers[0].output_after_actual_intervention"
        if model_type == "gpt_neox"
        else "block_output.after_actual_intervention"
    )
    with pytest.raises(AxBenchAdvError, match="GCG HF calibration"):
        _validate_gcg_calibration(
            {**calibration, "site": wrong_site},
            objective="activation_l2",
            concept_id=concept_id,
            model_identity=identity,
            relative=Path("search.json"),
        )


def test_preserve_runtime_keeps_existing_grad_object_identity() -> None:
    module = _load_worker_with_fakes()
    model = t.nn.Linear(3, 2)
    parameter = model.weight
    sentinel = t.full_like(parameter, 17.0)
    parameter.grad = sentinel
    runtime = SimpleNamespace(
        benchmark_model=SimpleNamespace(model=model),
        tokenizer=SimpleNamespace(padding_side="left"),
    )
    with module._preserve_runtime(runtime):
        pass
    assert parameter.grad is sentinel
    assert t.equal(parameter.grad, t.full_like(parameter, 17.0))


def test_worker_timeout_removes_root_owned_embedding_sidecar_directory(tmp_path: Path) -> None:
    sidecar = tempfile.TemporaryDirectory(prefix="gcg-timeout-", dir=tmp_path)
    sidecar_path = Path(sidecar.name)
    (sidecar_path / "input_embeddings.safetensors").write_bytes(b"snapshot")
    (sidecar_path / "gradients.safetensors").write_bytes(b"snapshot")
    process = _FramedProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=REPO_ROOT,
        timeout_seconds=0.1,
        stderr_path=tmp_path / "timeout.log",
        cleanup_callback=sidecar.cleanup,
    )
    with pytest.raises(AxBenchAdvError, match="TimeoutError"):
        process.request("gcg_gradient", {})
    assert process._closed is True
    assert not sidecar_path.exists()


def test_tensor_sidecar_rejects_symlink_to_private_file_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    save_file({"embedding_weight": t.tensor([[1.25, -2.5]], dtype=t.float32)}, outside)
    outside.chmod(0o400)
    owned = tmp_path / "root-owned"
    owned.mkdir()
    expected_path = owned / "input_embeddings.safetensors"
    expected_path.symlink_to(outside)
    descriptor = {
        "shape": [1, 2],
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    with pytest.raises(AxBenchAdvError, match="regular file"):
        _load_tensor_sidecar(
            descriptor,
            expected_path=expected_path,
            expected_key="embedding_weight",
            expected_rank=2,
        )


def test_abort_cleans_sidecar_even_if_sigkill_wait_also_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubbornProcess:
        pid = 12345

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> None:
            raise subprocess.TimeoutExpired("stubborn-worker", timeout)

    class _Stream:
        closed = False

        def close(self) -> None:
            self.closed = True

    cleaned: list[bool] = []
    subject = object.__new__(_FramedProcess)
    raw = cast(Any, subject)
    raw._closed = False
    raw._process = _StubbornProcess()
    raw._stdin = _Stream()
    raw._stdout = _Stream()
    raw._temporary = None
    raw._cleanup_callback = lambda: cleaned.append(True)
    raw.stderr_path = tmp_path / "missing-stderr.log"
    monkeypatch.setattr(os, "killpg", lambda _pid, _signal: None)
    subject._abort()
    assert cleaned == [True]
    assert subject._closed is True
    assert raw._stdin.closed is True and raw._stdout.closed is True
    assert "cleanup_errors=TimeoutExpired" in subject._last_context


def test_vendored_gcg_hf_review_suite() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "vendor/axbench"), str(REPO_ROOT)))
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(REPO_ROOT / "vendor/axbench"),
            "--frozen",
            "--offline",
            "--with",
            "pytest==9.0.2",
            "python",
            "-m",
            "pytest",
            "--confcutdir",
            str(REPO_ROOT / "tests/axbench_adv"),
            str(VENDOR_SUITE),
            "-q",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
