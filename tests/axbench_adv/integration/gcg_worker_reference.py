"""Test-worker bootstrap: verify validation, activations, and GCG gradients/scores."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch as t
from safetensors import safe_open

from scripts.eval.axbench_adv.vendored import gcg_worker
from scripts.eval.axbench_adv.vendored.worker import WorkerRuntime
from tests.axbench_adv.gcg_activation_reference import late_clean_transfers
from tests.axbench_adv.gcg_factor_reference import l2_with_expected_offset, tensor_digest
from tests.axbench_adv.gcg_filter_reference import scalar_candidate
from tests.axbench_adv.gcg_gradient_reference import full_sequence_logits, rowwise_vjp
from tests.axbench_adv.gcg_layer_reference import full_decoder_depth
from tests.axbench_adv.gcg_pool_reference import scalar_allowed_token_ids
from tests.axbench_adv.gcg_response_reference import fresh_response_tensors
from tests.axbench_adv.gcg_state_reference import (
    preserve_gcg_model_state as reference_model_state,
)


@contextmanager
def device_collector_dimensions(session: Any) -> Iterator[None]:
    """Restore the parent collector's model-device slice bound independently."""
    with ExitStack() as stack:
        if session._activation_model is not None:
            for intervention in session._activation_model.interventions.values():
                if isinstance(intervention, gcg_worker.CollectIntervention):
                    dimension = t.tensor(
                        int(session.runtime.benchmark_model.model.config.hidden_size),
                        dtype=t.long,
                        device=session.runtime.device,
                    )
                    stack.enter_context(patch.object(intervention, "interchange_dim", dimension))
        yield


def verify_gcg_numerics(capture_path: str) -> None:
    capture = Path(capture_path)
    original_candidate = gcg_worker.GCGSession._candidate
    original_prepare = WorkerRuntime.gcg_prepare
    original_score = WorkerRuntime.gcg_score
    original_gradient = WorkerRuntime.gcg_gradient
    expected_offsets: dict[int, t.Tensor] = {}

    @contextmanager
    def reference_state(runtime: WorkerRuntime) -> Iterator[None]:
        session = runtime._gcg_session
        assert session is not None

        def independent_loss(attacked: t.Tensor, clean: t.Tensor, observed: t.Tensor) -> t.Tensor:
            return l2_with_expected_offset(
                attacked, clean, observed, expected=expected_offsets[id(session)]
            )

        with (
            full_sequence_logits(runtime.benchmark_model.model),
            full_decoder_depth(gcg_worker),
            patch.object(gcg_worker, "preserve_gcg_model_state", reference_model_state),
            fresh_response_tensors(gcg_worker),
            device_collector_dimensions(session),
            late_clean_transfers(gcg_worker),
            patch.object(gcg_worker, "activation_l2_row", independent_loss),
        ):
            yield

    def checked_candidate(
        self: gcg_worker.GCGSession, token_ids: tuple[int, ...]
    ) -> tuple[str, str | None]:
        actual = original_candidate(self, token_ids)
        assert actual == scalar_candidate(self, token_ids)
        return actual

    def checked_prepare(self: WorkerRuntime, payload: Any) -> dict[str, Any]:
        result = original_prepare(self, payload)
        session = self._gcg_session
        assert session is not None
        assert tuple(result["allowed_token_ids"]) == scalar_allowed_token_ids(
            gcg_worker,
            self,
            self.benchmark_model.model.get_input_embeddings().weight.shape[0],
        )
        if session.objective == "activation_l2":
            calibration = result["direction_calibration"]
            # Read DiffMean even for LinearProbe; never reuse the prepared offset.
            saved = t.load(
                Path(calibration["direction_path"]), map_location="cpu", weights_only=True
            )[self.concept_id].to(t.bfloat16)
            expected = (
                t.tensor([calibration["max_activation"]], dtype=t.float32).unsqueeze(-1)
                * t.tensor([self.factor], dtype=t.float32).unsqueeze(-1)
                * saved.unsqueeze(0)
            )[0].to(self.device)
            assert calibration["factor"] == self.factor
            assert calibration["resolved_scale"] == calibration["max_activation"] * self.factor
            assert session._direction_offset is not None
            assert session._direction_offset.dtype is t.float32
            assert t.equal(session._direction_offset, expected)
            expected_offsets[id(session)] = expected
            if self.method in {"DiffMean", "LinearProbe"}:
                assert session._activation_model is not None
                collectors = [
                    item
                    for item in session._activation_model.interventions.values()
                    if isinstance(item, gcg_worker.CollectIntervention)
                ]
                assert len(collectors) == 1
                bound = collectors[0].interchange_dim
                assert bound.device.type == "cpu"
                assert bound.ndim == 0 and bound.dtype == t.long
                assert int(bound) == int(self.benchmark_model.model.config.hidden_size)
            actual = dict(session._clean_hidden)
            with (
                full_sequence_logits(self.benchmark_model.model),
                full_decoder_depth(gcg_worker),
                patch.object(gcg_worker, "preserve_gcg_model_state", reference_model_state),
                device_collector_dimensions(session),
            ):
                session._compute_clean_hidden()
            assert actual.keys() == session._clean_hidden.keys()
            for sample_id, hidden in actual.items():
                t.testing.assert_close(hidden, session._clean_hidden[sample_id], rtol=0, atol=0)
            with capture.open("a", encoding="utf-8") as stream:
                json.dump(
                    {
                        "k": len(session.seed_token_ids),
                        "offset": tensor_digest(session._direction_offset),
                        "clean": {str(key): tensor_digest(value) for key, value in actual.items()},
                    },
                    stream,
                    sort_keys=True,
                )
                stream.write("\n")
        return result

    def checked_score(self: WorkerRuntime, payload: Any) -> dict[str, Any]:
        actual = original_score(self, payload)
        with reference_state(self), patch.object(gcg_worker, "fused_scoring_mlps", nullcontext):
            reference = original_score(self, payload)
        assert actual == reference
        return actual

    def checked(self: WorkerRuntime, payload: Any) -> dict[str, Any]:
        actual_result = original_gradient(self, payload)
        assert self._gcg_sidecar_dir is not None
        path = self._gcg_sidecar_dir / "gradients.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual = handle.get_tensor("gradients").clone()
        with reference_state(self), rowwise_vjp():
            reference_result = original_gradient(self, payload)
        with safe_open(path, framework="pt", device="cpu") as handle:
            reference = handle.get_tensor("gradients")
        t.testing.assert_close(actual, reference, rtol=0, atol=0)
        # The reference call replaces the sidecar. Its tensor and descriptor
        # must be identical, so the caller still consumes the optimized result.
        assert reference_result == actual_result
        return actual_result

    gcg_worker.GCGSession._candidate = checked_candidate
    WorkerRuntime.gcg_prepare = checked_prepare
    WorkerRuntime.gcg_score = checked_score
    WorkerRuntime.gcg_gradient = checked
