"""Root launchers and actual framed IPC for the real vendored soft-prompt runtime."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import load_file, save_file

from scripts.eval.axbench_adv.soft_prompt_tensors import write_soft_embeddings
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.worker_client import AxBenchAdvWorkerClient, WorkerExample

_ROOT = Path(__file__).resolve().parents[3]
_SUITE = Path(__file__).with_name("soft_prompt_hf_suite.py")
_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
_ROWS = [
    {
        "pair_id": 0,
        "positive": {"sample_id": 0, "prompt": "alpha", "response": "yes yes"},
        "negative": {"sample_id": 1, "prompt": "beta gamma delta", "response": "no"},
    },
    {
        "pair_id": 1,
        "positive": {"sample_id": 2, "prompt": "gamma beta", "response": "alpha yes beta"},
        "negative": {"sample_id": 3, "prompt": "delta", "response": "no gamma"},
    },
]


def _vendor_command() -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(_ROOT / "vendor/axbench"),
        "--frozen",
        "--offline",
        "--with",
        "pytest==9.0.2",
        "python",
    ]


def test_vendored_soft_prompt_hf_suite() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(_ROOT / "vendor/axbench"), str(_ROOT))),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            *_vendor_command(),
            "-m",
            "pytest",
            "--confcutdir",
            str(_ROOT / "tests/axbench_adv"),
            str(_SUITE),
            "-q",
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe", "LoRA", "PromptSteering"])
def test_real_framed_soft_runtime_and_retained_tensor_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(_ROOT / "vendor/axbench"), str(_ROOT))))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(
        "steering.eval.axbench_adv.worker_client.build_worker_command",
        lambda: [*_vendor_command(), str(_SUITE), "--worker"],
    )
    examples = [
        WorkerExample(
            input_id=0,
            input=f"<|im_start|>user\nalpha{_TAIL}",
            steered_input=f"<|im_start|>user\nsteer\n\nQuestion: alpha{_TAIL}",
        )
    ]
    client = AxBenchAdvWorkerClient(
        init_payload={
            "method": method,
            "concept_id": 0,
            "factor": 0.6,
            "fixture_dir": str(tmp_path / "fixture"),
            "host": "qwen2",
        },
        examples=examples,
        master_seed=123,
        stderr_path=tmp_path / "worker.log",
    )
    sidecar_dir = Path(client._gcg_sidecar_temp.name)
    with client:
        for objective in ("response_margin", "activation_l2"):
            kwargs: dict[str, Any] = {
                "objective": objective,
                "alpha": 0.01,
                "rows": _ROWS,
                "suffix_token_count": 3,
                "device_batch_size": 3,
            }
            ordinary = client.prepare_gcg(**kwargs)
            expected_score = client.gcg_score(
                [ordinary.seed_token_ids], [1, 0], candidate_eval_device_batch_size=3
            )
            expected_gradient = client.gcg_gradient(ordinary.seed_token_ids, [1, 0])
            prepared = client.prepare_soft_prompt(**kwargs)
            seed = prepared.embedding_weight[list(prepared.seed_token_ids)].clone()
            assert prepared.identity["generation_conditioning"] == "ordinary_prompt"
            assert client.soft_score(seed, [1, 0]) == expected_score
            assert all(
                t.equal(actual, expected)
                for actual, expected in zip(
                    client.soft_gradient(seed, [1, 0]), expected_gradient, strict=True
                )
            )
            assert client.evaluate_soft(seed, [0]) == client.evaluate("! ! !", [0])
            perturbed = seed + t.linspace(-0.7, 0.9, seed.numel()).reshape(seed.shape)
            retained_path = tmp_path / f"retained-{objective}.safetensors"
            save_file({"embeddings": perturbed}, retained_path)
            retained = load_file(retained_path)["embeddings"]
            assert t.equal(retained, perturbed)
            assert client.soft_score(retained, [1, 0]) == client.soft_score(perturbed, [1, 0])
            assert client.evaluate_soft(retained, [0]) == client.evaluate_soft(perturbed, [0])
            assert not (sidecar_dir / "soft_embeddings.safetensors").exists()

        # The real worker rejects a malformed tensor before any model call,
        # then the framed session aborts and releases its private directory.
        def corrupt_shape(path: Path, embeddings: t.Tensor) -> None:
            write_soft_embeddings(path, embeddings[:, :-1])

        monkeypatch.setattr(
            "steering.eval.axbench_adv.worker_client.write_soft_embeddings", corrupt_shape
        )
        with pytest.raises(AxBenchAdvError, match="finite float32 with shape"):
            client.soft_score(seed, [0])
        assert client._soft_prepared is None
        assert not sidecar_dir.exists()
        with pytest.raises(AxBenchAdvError, match="requires a successful soft prepare"):
            client.soft_score(seed, [0])
    assert not sidecar_dir.exists()


def test_real_worker_rejects_soft_calls_before_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(_ROOT / "vendor/axbench"), str(_ROOT))))
    monkeypatch.setattr(
        "steering.eval.axbench_adv.worker_client.build_worker_command",
        lambda: [*_vendor_command(), str(_SUITE), "--worker"],
    )
    with AxBenchAdvWorkerClient(
        init_payload={
            "method": "PromptSteering",
            "concept_id": 0,
            "factor": 0.6,
            "fixture_dir": str(tmp_path / "fixture"),
            "host": "qwen2",
        },
        examples=[WorkerExample(0, f"source{_TAIL}", f"steered{_TAIL}")],
        master_seed=123,
        stderr_path=tmp_path / "worker.log",
    ) as client:
        sidecar_dir = Path(client._gcg_sidecar_temp.name)
        with pytest.raises(AxBenchAdvError, match="soft_prepare must succeed"):
            client._process.request("soft_gradient", {})
    assert not sidecar_dir.exists()
