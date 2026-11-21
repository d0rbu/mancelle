"""Strict private-tensor and root-side soft worker reply validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from scripts.eval.axbench_adv.soft_prompt_tensors import (
    load_soft_embeddings,
    write_soft_embeddings,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.worker_client import AxBenchAdvWorkerClient, GcgPrepared


def test_private_tensor_atomic_replacement_preserves_prior_mapping(tmp_path: Path) -> None:
    path = tmp_path / "soft_embeddings.safetensors"
    value = t.arange(6, dtype=t.float32).reshape(2, 3)
    write_soft_embeddings(path, value)
    first_inode = path.stat().st_ino
    first = load_soft_embeddings(path=path, shape=(2, 3))
    write_soft_embeddings(path, -value)
    assert path.stat().st_ino != first_inode
    assert path.stat().st_mode & 0o077 == 0
    assert t.equal(first, value)
    assert t.equal(load_soft_embeddings(path=path, shape=(2, 3)), -value)
    path.unlink()
    assert t.equal(first, value)
    with pytest.raises(ValueError, match="unavailable"):
        load_soft_embeddings(path=path, shape=(2, 3))


@pytest.mark.parametrize(
    "mutation",
    ["key", "rank", "shape", "dtype", "nan", "inf", "public", "symlink"],
)
def test_reject_malformed_or_substituted_incoming_tensor(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "soft_embeddings.safetensors"
    value = t.ones((2, 3), dtype=t.float32)
    key = "embeddings"
    if mutation == "key":
        key = "other"
    elif mutation == "rank":
        value = value.flatten()
    elif mutation == "shape":
        value = value.transpose(0, 1).contiguous()
    elif mutation == "dtype":
        value = value.to(t.bfloat16)
    elif mutation in {"nan", "inf"}:
        value[0, 0] = float(mutation)
    save_file({key: value}, path)
    path.chmod(0o400)
    if mutation == "public":
        path.chmod(0o644)
    elif mutation == "symlink":
        outside = tmp_path / "outside.safetensors"
        path.rename(outside)
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        load_soft_embeddings(path=path, shape=(2, 3))


def _client(
    tmp_path: Path, result: dict[str, Any], *, write_gradient: bool = True
) -> AxBenchAdvWorkerClient:
    class Process:
        @staticmethod
        def request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert set(payload) == {"member_indices"}
            tensor = load_soft_embeddings(
                path=tmp_path / "soft_embeddings.safetensors",
                shape=(2, 3),
            )
            assert t.equal(tensor, t.ones((2, 3)))
            if operation == "soft_gradient":
                path = tmp_path / "gradients.safetensors"
                assert not path.exists()
                if write_gradient:
                    save_file({"gradients": t.ones((2, 2, 3))}, path)
                    path.chmod(0o400)
                    result["gradient_sidecar"] = {
                        "shape": [2, 2, 3],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
            return result

    client = object.__new__(AxBenchAdvWorkerClient)
    raw = cast(Any, client)
    raw._closed = False
    raw._process = Process()
    raw._gcg_sidecar_temp = SimpleNamespace(name=str(tmp_path))
    raw._soft_prepared = GcgPrepared(
        embedding_weight=t.zeros((4, 3)),
        allowed_token_ids=(1, 2),
        seed_token_ids=(1, 2),
        member_count=2,
        identity={},
        layouts_sha256="a" * 64,
        direction_calibration=None,
    )
    return client


@pytest.mark.parametrize(
    "reply",
    [
        {"losses": [0.1, 0.2], "member_indices": [1, 0]},
        {"losses": [0.1], "member_indices": [0, 1]},
        {"losses": [float("nan"), 0.2], "member_indices": [0, 1]},
        {"losses": [True, 0.2], "member_indices": [0, 1]},
        {"losses": [0.1, 0.2], "member_indices": [False, 1]},
        {"losses": [0.1, 0.2], "member_indices": [0, 1], "extra": None},
    ],
)
def test_score_echo_cardinality_and_values_fail_closed(
    tmp_path: Path, reply: dict[str, Any]
) -> None:
    client = _client(tmp_path, reply)
    with pytest.raises(AxBenchAdvError):
        client.soft_score(t.ones((2, 3)), [0, 1])
    assert not (tmp_path / "soft_embeddings.safetensors").exists()


def test_gradient_rejects_reordered_members_and_stale_output(tmp_path: Path) -> None:
    client = _client(tmp_path, {"member_indices": [1, 0]})
    with pytest.raises(AxBenchAdvError, match="member order"):
        client.soft_gradient(t.ones((2, 3)), [0, 1])
    path = tmp_path / "gradients.safetensors"
    descriptor = {"shape": [2, 2, 3], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    stale = _client(
        tmp_path, {"member_indices": [0, 1], "gradient_sidecar": descriptor}, write_gradient=False
    )
    with pytest.raises(AxBenchAdvError, match="unavailable"):
        stale.soft_gradient(t.ones((2, 3)), [0, 1])
    assert not path.exists()
    assert not (tmp_path / "soft_embeddings.safetensors").exists()


@pytest.mark.parametrize("indices", [[], [0, 0], [True], [-1], [2]])
def test_soft_calls_reject_bad_indices_before_writing(tmp_path: Path, indices: list[int]) -> None:
    client = _client(tmp_path, {})
    with pytest.raises(AxBenchAdvError, match="indices are invalid"):
        client.soft_score(t.ones((2, 3)), indices)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "value", [t.ones((1, 3)), t.ones((2, 3), dtype=t.bfloat16), t.full((2, 3), float("inf"))]
)
def test_soft_calls_reject_bad_tensor_before_writing(tmp_path: Path, value: t.Tensor) -> None:
    client = _client(tmp_path, {})
    with pytest.raises(AxBenchAdvError, match="finite float32"):
        client.soft_gradient(value, [0, 1])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("phase", ["write", "request"])
def test_failed_soft_request_invalidates_prepared_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    client = _client(tmp_path, {})

    def fail_write(tensors: dict[str, t.Tensor], filename: str) -> None:
        save_file(tensors, filename)
        raise RuntimeError("injected write failure")

    def fail_request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert operation == "soft_score"
        assert (tmp_path / "soft_embeddings.safetensors").exists()
        raise RuntimeError("injected request failure")

    if phase == "write":
        monkeypatch.setattr("scripts.eval.axbench_adv.soft_prompt_tensors.save_file", fail_write)
    else:
        monkeypatch.setattr(client._process, "request", fail_request)
    with pytest.raises(RuntimeError, match=f"injected {phase} failure"):
        client.soft_score(t.ones((2, 3)), [0, 1])
    assert client._soft_prepared is None
    assert not client._closed
    assert not list(tmp_path.iterdir())
    with pytest.raises(AxBenchAdvError, match="requires a successful soft prepare"):
        client.soft_score(t.ones((2, 3)), [0, 1])


@pytest.mark.parametrize("exceptional_exit", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_close_releases_prepared_embedding_even_when_cleanup_fails(
    tmp_path: Path, exceptional_exit: bool, cleanup_fails: bool
) -> None:
    client = _client(tmp_path, {})

    class Closing:
        @staticmethod
        def close() -> None:
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

        _abort = close

    cast(Any, client)._process = Closing()

    def close() -> None:
        if exceptional_exit:
            client.__exit__(ValueError, ValueError("failure"), None)
        else:
            client.close()

    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            close()
    else:
        close()
    assert client._closed and client._soft_prepared is None
