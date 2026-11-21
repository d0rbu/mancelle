"""Exercise the actual tensor boundary across more than one soft-gradient RPC."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from scripts.eval.axbench_adv.soft_prompt_tensors import load_soft_embeddings
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.worker_client import AxBenchAdvWorkerClient, GcgPrepared


@pytest.mark.parametrize("fail_second_chunk", [False, True])
def test_soft_gradient_chunk_boundary_preserves_order_and_cleans_later_failure(
    tmp_path: Path, fail_second_chunk: bool
) -> None:
    # Each member is 20 MiB, so three members cross the existing 40 MiB RPC
    # boundary. Keep real tensors, publication, hashes, and sidecar validation.
    k, width = 1280, 4096
    soft = t.linspace(-1.0, 1.0, k * width).reshape(k, width)
    calls: list[tuple[int, ...]] = []
    input_path = tmp_path / "soft_embeddings.safetensors"
    output_path = tmp_path / "gradients.safetensors"

    class Process:
        @staticmethod
        def request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert operation == "soft_gradient" and set(payload) == {"member_indices"}
            indices = tuple(payload["member_indices"])
            calls.append(indices)
            assert not output_path.exists(), "each RPC must remove the prior output"
            loaded = load_soft_embeddings(path=input_path, shape=(k, width))
            input_path.unlink()
            assert t.equal(loaded, soft), "every chunk must receive the same FP32 master"
            if fail_second_chunk and len(calls) == 2:
                raise RuntimeError("injected second chunk failure")
            values = t.tensor(indices, dtype=t.float32)[:, None, None]
            gradients = values.expand(-1, k, width).contiguous()
            save_file({"gradients": gradients}, output_path)
            output_path.chmod(0o400)
            with output_path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            return {
                "member_indices": list(indices),
                "gradient_sidecar": {"shape": list(gradients.shape), "sha256": digest},
            }

    client = object.__new__(AxBenchAdvWorkerClient)
    raw = cast(Any, client)
    raw._closed = False
    raw._process = Process()
    raw._gcg_sidecar_temp = SimpleNamespace(name=str(tmp_path))
    raw._soft_prepared = GcgPrepared(
        embedding_weight=t.zeros((1, width)),
        allowed_token_ids=(0,),
        seed_token_ids=(0,) * k,
        member_count=3,
        identity={},
        layouts_sha256="a" * 64,
        direction_calibration=None,
    )

    if fail_second_chunk:
        with pytest.raises(RuntimeError, match="second chunk failure"):
            client.soft_gradient(soft, [2, 0, 1])
        assert client._soft_prepared is None
        assert not list(tmp_path.iterdir())
        with pytest.raises(AxBenchAdvError, match="requires a successful soft prepare"):
            client.soft_gradient(soft, [0])
    else:
        gradients = client.soft_gradient(soft, [2, 0, 1])
        assert len(gradients) == 3
        output_path.unlink()
        for gradient, expected in zip(gradients, [2.0, 0.0, 1.0], strict=True):
            assert gradient.shape == (k, width) and gradient.dtype is t.float32
            # The first mapped outputs must survive deletion before chunk 2,
            # and all outputs must survive final deletion by the caller.
            assert bool(gradient.eq(expected).all())
        assert not list(tmp_path.iterdir())
    assert calls == [(2, 0), (1,)]
