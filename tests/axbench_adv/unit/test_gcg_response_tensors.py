"""Fixed response tensors are reused only within their prepared GCG cell."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch as t
from safetensors.torch import load_file

from tests.axbench_adv.gcg_response_reference import fresh_response_side_batch
from tests.axbench_adv.unit.test_gcg_vendor_worker import (
    GCGSession,
    _runtime,
    _write_activation_artifacts,
    _write_manifest,
)

_WORKER = sys.modules[GCGSession.__module__]


def _payload(*, changed: bool = False, objective: str = "response_margin") -> dict[str, Any]:
    responses = ("XYZ", "UV", "RSTU", "W") if changed else ("A !C", "D !", "FGHI", "J")
    return {
        "alpha": 0.25,
        "objective": objective,
        "rows": [
            {
                "pair_id": index,
                "positive": {
                    "sample_id": 2 * index,
                    "prompt": "P" * (index + 1),
                    "response": responses[2 * index],
                },
                "negative": {
                    "sample_id": 2 * index + 1,
                    "prompt": "N" * (index + 1),
                    "response": responses[2 * index + 1],
                },
            }
            for index in range(2)
        ],
        "separator": "\n\n",
        "suffix_token_count": 1,
        "device_batch_size": 3,
    }


def _score_payload() -> dict[str, Any]:
    return {
        "candidate_eval_device_batch_size": 3,
        "candidate_token_ids": [[ord("?")], [ord("!")]],
        "member_indices": [1, 0],
    }


def _fresh_side(
    session: Any, items: Any, *, gradient: bool, soft_embeddings: t.Tensor | None = None
) -> Any:
    assert soft_embeddings is None
    return fresh_response_side_batch(_WORKER, session, items, gradient=gradient)


def _session(tmp_path: Path) -> tuple[Any, Any]:
    _write_manifest(tmp_path)
    runtime = _runtime(tmp_path)
    session = GCGSession(runtime)
    session.prepare(_payload())
    return runtime, session


@pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0])
def test_response_tensor_reuse_preserves_exact_scores_and_gradients(
    tmp_path: Path, alpha: float
) -> None:
    runtime, session = _session(tmp_path)
    session.alpha = alpha
    assert session._response_tensors == {}
    actual_score = session.score(_score_payload())
    constants = dict(session._response_tensors)
    assert len(constants) == 4
    for pair in session.layouts:
        for layout in (pair.positive, pair.negative):
            targets, weights = constants[(layout.sample_id, t.device("cpu"))]
            assert targets.dtype == t.long and weights.dtype == t.float32
            assert not targets.requires_grad and not weights.requires_grad
            assert t.equal(targets, t.tensor(layout.response_ids))
            assert t.equal(weights, t.tensor(layout.response_byte_weights, dtype=t.float32))
    with patch.object(GCGSession, "_side_batch", _fresh_side):
        assert actual_score == session.score(_score_payload())
    assert actual_score == session.score(_score_payload())
    request = {"token_ids": [ord("?")], "member_indices": [0, 1]}
    actual_descriptor = session.gradient(request)
    path = runtime._gcg_sidecar_dir / "gradients.safetensors"
    actual = load_file(path)["gradients"].clone()
    with patch.object(GCGSession, "_side_batch", _fresh_side):
        assert actual_descriptor == session.gradient(request)
    assert t.equal(actual, load_file(path)["gradients"])
    assert constants.keys() == session._response_tensors.keys()
    for key, tensors in constants.items():
        assert all(a is b for a, b in zip(tensors, session._response_tensors[key], strict=True))
    session.close()
    assert session._response_tensors == {}


def test_inference_scoring_keeps_response_tensors_usable_by_later_gradients(
    tmp_path: Path,
) -> None:
    runtime, session = _session(tmp_path)
    with t.inference_mode():
        actual_score = session.score(_score_payload())
        assert t.is_inference_mode_enabled()
        assert not t.is_grad_enabled()
        with patch.object(GCGSession, "_side_batch", _fresh_side):
            assert actual_score == session.score(_score_payload())
    assert all(
        not value.is_inference()
        for tensors in session._response_tensors.values()
        for value in tensors
    )
    request = {"token_ids": [ord("?")], "member_indices": [0, 1]}
    descriptor = session.gradient(request)
    path = runtime._gcg_sidecar_dir / "gradients.safetensors"
    actual = load_file(path)["gradients"].clone()
    with patch.object(GCGSession, "_side_batch", _fresh_side):
        assert descriptor == session.gradient(request)
    assert t.equal(actual, load_file(path)["gradients"])
    session.close()


def test_response_reprepare_replaces_constants_for_reused_sample_ids(tmp_path: Path) -> None:
    _runtime_value, session = _session(tmp_path)
    session.score(_score_payload())
    previous = dict(session._response_tensors)
    session.prepare(_payload(changed=True))
    assert session._response_tensors == {}
    actual = session.score(_score_payload())
    with patch.object(GCGSession, "_side_batch", _fresh_side):
        assert actual == session.score(_score_payload())
    assert session._response_tensors.keys() == previous.keys()
    for key, (targets, _weights) in session._response_tensors.items():
        assert not t.equal(targets, previous[key][0])
    session.close()


def test_failed_reprepare_clears_response_constants(tmp_path: Path) -> None:
    _runtime_value, session = _session(tmp_path)
    session.score(_score_payload())
    payload = _payload()
    payload["separator"] = "invalid"
    with pytest.raises(ValueError, match="separator must be exactly"):
        session.prepare(payload)
    assert session._response_tensors == {}
    with pytest.raises(ValueError, match="gcg_prepare must succeed"):
        session.score(_score_payload())


def test_malformed_envelope_keeps_the_existing_preflight_contract(tmp_path: Path) -> None:
    _runtime_value, session = _session(tmp_path)
    before = session.score(_score_payload())
    constants = dict(session._response_tensors)
    with pytest.raises(ValueError, match="fields must be exactly"):
        session.prepare({})
    assert session.score(_score_payload()) == before
    assert constants.keys() == session._response_tensors.keys()
    for key, tensors in constants.items():
        assert all(a is b for a, b in zip(tensors, session._response_tensors[key], strict=True))
    session.close()


def test_activation_l2_does_not_retain_or_build_response_tensors(tmp_path: Path) -> None:
    _runtime_value, session = _session(tmp_path)
    session.score(_score_payload())
    _write_activation_artifacts(tmp_path)
    session.prepare(_payload(objective="activation_l2"))
    assert session._response_tensors == {}
    session.score(_score_payload())
    session.gradient({"token_ids": [ord("?")], "member_indices": [0, 1]})
    assert session._response_tensors == {}
    session.close()
