"""Functional contracts for GCG root orchestration and worker decoding."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from steering.conditional.feature_gcg.edit_primitives import TokenReplacement
from steering.conditional.feature_gcg.edit_space import ValidatedGroup
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, parse_gcg_config
from steering.eval.axbench_adv.gcg_driver import (
    AppendSuffixEditSpace,
    GcgRuntime,
    _gradient_callback,
    _score_callback,
    _training_rows,
    execute_gcg_cell,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.prompt_surgery import (
    PYTHIA_CONTROL_TOKENS,
    PYTHIA_TERMINAL_MARKER,
    QWEN_CONTROL_TOKENS,
    QWEN_TERMINAL_MARKER,
    SMOLLM3_CONTROL_TOKENS,
    SMOLLM3_NO_THINK_TERMINAL_MARKER,
)
from steering.eval.axbench_adv.worker_client import (
    AxBenchAdvWorkerClient,
    GcgFilteredCandidate,
    GcgPrepared,
    WorkerExample,
    _load_tensor_sidecar,
)
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_upstream.data import ConceptTrainingPairs


class _FilterProcess:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates

    def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert operation == "gcg_filter"
        assert len(payload["candidate_token_ids"]) == len(self.candidates)
        return {"candidates": self.candidates}


def _bare_client(process: object) -> AxBenchAdvWorkerClient:
    client = object.__new__(AxBenchAdvWorkerClient)
    client._process = process  # ty: ignore[invalid-assignment]
    return client


def test_worker_filter_requires_exact_echo_for_valid_and_invalid_candidates() -> None:
    invalid = {"token_ids": [], "text": "", "valid": False, "error": "rejected"}
    valid = {"token_ids": [7], "text": "x", "valid": True, "error": None}
    client = _bare_client(_FilterProcess([invalid, valid]))
    with pytest.raises(AxBenchAdvError, match="fields are invalid"):
        client.gcg_filter([[0], [7]])

    exact_invalid = {**invalid, "token_ids": [0]}
    result = _bare_client(_FilterProcess([exact_invalid])).gcg_filter([[0]])
    assert result[0] == GcgFilteredCandidate((0,), "", False, "rejected")

    changed = _bare_client(_FilterProcess([{**valid, "token_ids": [8]}]))
    with pytest.raises(AxBenchAdvError, match="fields are invalid"):
        changed.gcg_filter([[7]])


def _write_sidecar(path: Path, key: str, tensor: t.Tensor) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp")
    save_file({key: tensor}, temporary)
    temporary.chmod(0o400)
    temporary.replace(path)
    return {"shape": list(tensor.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _gradient_client(process: object, sidecar_dir: Path) -> AxBenchAdvWorkerClient:
    client = _bare_client(process)
    client._gcg_embedding_dim = 4
    client._gcg_member_count = 3
    cast(Any, client)._gcg_sidecar_temp = SimpleNamespace(name=str(sidecar_dir))
    return client


@pytest.mark.parametrize(
    ("marker", "forbidden"),
    (
        (QWEN_TERMINAL_MARKER, QWEN_CONTROL_TOKENS),
        (PYTHIA_TERMINAL_MARKER, PYTHIA_CONTROL_TOKENS),
        (SMOLLM3_NO_THINK_TERMINAL_MARKER, SMOLLM3_CONTROL_TOKENS),
    ),
)
@pytest.mark.parametrize("method", ["LoRA", "PromptSteering"])
def test_worker_prepare_unlinks_stale_sidecar_and_preserves_prior_embedding(
    tmp_path: Path, marker: str, forbidden: tuple[str, ...], method: str
) -> None:
    path = tmp_path / "input_embeddings.safetensors"
    _write_sidecar(path, "embedding_weight", t.full((3, 2), -1.0))

    class PrepareProcess:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, operation: str, _payload: dict[str, Any]) -> dict[str, Any]:
            assert operation == "gcg_prepare"
            assert not path.exists()
            self.calls += 1
            embedding = t.full((3, 2), float(self.calls))
            return {
                "allowed_token_ids": [1, 2],
                "direction_calibration": None,
                "embedding_sidecar": _write_sidecar(path, "embedding_weight", embedding),
                "identity": {"vocab_size": 3, "embedding_dim": 2},
                "layouts_sha256": "a" * 64,
                "member_count": 1,
                "seed_token_ids": [1],
            }

    process = PrepareProcess()
    client = _bare_client(process)
    client._method = method
    client._examples = {
        0: WorkerExample(
            input_id=0,
            input="source" + (marker if method == "LoRA" else PYTHIA_TERMINAL_MARKER),
            steered_input="source"
            + (marker if method == "PromptSteering" else QWEN_TERMINAL_MARKER),
        )
    }
    cast(Any, client)._gcg_sidecar_temp = SimpleNamespace(name=str(tmp_path))
    first = client.prepare_gcg(
        objective="response_margin",
        alpha=0.01,
        rows=({"pair_id": 0},),
        suffix_token_count=1,
        device_batch_size=1,
    )
    assert first.forbidden_control_tokens == forbidden
    first_embedding = first.embedding_weight
    first_row = first_embedding[0]
    first_copy = first_embedding.clone()
    first_inode = path.stat().st_ino
    second = client.prepare_gcg(
        objective="response_margin",
        alpha=0.01,
        rows=({"pair_id": 0},),
        suffix_token_count=1,
        device_batch_size=1,
    )
    assert process.calls == 2
    assert path.stat().st_ino != first_inode
    assert t.equal(first_embedding, first_copy)
    assert t.equal(first_row, first_copy[0])
    assert t.equal(second.embedding_weight, t.full((3, 2), 2.0))
    path.unlink()
    assert t.equal(first_embedding, first_copy)


@pytest.mark.parametrize(
    ("objective", "direction_calibration", "match"),
    [
        ("activation_l2", None, "requires direction calibration"),
        ("response_margin", {"layer": 1}, "must not return direction calibration"),
    ],
)
def test_worker_prepare_rejects_calibration_presence_mismatch(
    tmp_path: Path,
    objective: str,
    direction_calibration: dict[str, int] | None,
    match: str,
) -> None:
    path = tmp_path / "input_embeddings.safetensors"

    class PrepareProcess:
        @staticmethod
        def request(operation: str, _payload: dict[str, Any]) -> dict[str, Any]:
            assert operation == "gcg_prepare"
            return {
                "allowed_token_ids": [1, 2],
                "direction_calibration": direction_calibration,
                "embedding_sidecar": _write_sidecar(
                    path, "embedding_weight", t.zeros((3, 2), dtype=t.float32)
                ),
                "identity": {"vocab_size": 3, "embedding_dim": 2},
                "layouts_sha256": "a" * 64,
                "member_count": 1,
                "seed_token_ids": [1],
            }

    client = _bare_client(PrepareProcess())
    cast(Any, client)._gcg_sidecar_temp = SimpleNamespace(name=str(tmp_path))
    with pytest.raises(AxBenchAdvError, match=match):
        client.prepare_gcg(
            objective=objective,
            alpha=0.01,
            rows=({"pair_id": 0},),
            suffix_token_count=1,
            device_batch_size=1,
        )


def test_worker_gradient_uses_one_ordered_sidecar_and_preserves_prior_views(
    tmp_path: Path,
) -> None:
    class GradientProcess:
        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert operation == "gcg_gradient"
            indices = tuple(payload["member_indices"])
            self.calls.append(indices)
            offset = 0.0 if len(self.calls) == 1 else 100.0
            tensor = t.tensor(indices, dtype=t.float32).reshape(-1, 1, 1).expand(-1, 1, 4)
            tensor = (tensor + offset).contiguous()
            descriptor = _write_sidecar(tmp_path / "gradients.safetensors", "gradients", tensor)
            return {"gradient_sidecar": descriptor, "member_indices": list(indices)}

    process = GradientProcess()
    client = _gradient_client(process, tmp_path)
    first = client.gcg_gradient([1], [2, 0, 1])
    first_values = [value.clone() for value in first]
    first_inode = (tmp_path / "gradients.safetensors").stat().st_ino
    second = client.gcg_gradient([1], [2, 0, 1])
    assert process.calls == [(2, 0, 1), (2, 0, 1)]
    assert (tmp_path / "gradients.safetensors").stat().st_ino != first_inode
    assert all(
        t.equal(actual, expected) for actual, expected in zip(first, first_values, strict=True)
    )
    assert [float(value[0, 0]) for value in first] == [2.0, 0.0, 1.0]
    assert [float(value[0, 0]) for value in second] == [102.0, 100.0, 101.0]
    (tmp_path / "gradients.safetensors").unlink()
    assert all(
        t.equal(actual, expected) for actual, expected in zip(first, first_values, strict=True)
    )


def test_worker_gradient_preserves_historical_chunk_boundary_and_first_views(
    tmp_path: Path,
) -> None:
    class GradientProcess:
        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def request(self, _operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            indices = tuple(payload["member_indices"])
            self.calls.append(indices)
            tensor = t.tensor(indices, dtype=t.float32).reshape(-1, 1, 1)
            tensor = tensor.expand(-1, 256, 3_072).contiguous()
            descriptor = _write_sidecar(tmp_path / "gradients.safetensors", "gradients", tensor)
            return {"gradient_sidecar": descriptor, "member_indices": list(indices)}

    process = GradientProcess()
    client = _gradient_client(process, tmp_path)
    client._gcg_embedding_dim = 3_072
    client._gcg_member_count = 16
    gradients = client.gcg_gradient([1] * 256, list(range(16)))
    assert process.calls == [tuple(range(13)), tuple(range(13, 16))]
    assert len(gradients) == 16
    assert [float(gradient[0, 0]) for gradient in gradients] == list(map(float, range(16)))
    assert all(gradient.shape == (256, 3_072) for gradient in gradients)


@pytest.mark.parametrize(
    ("echoed", "shape", "nonfinite", "match"),
    [
        ([1, 0], (2, 1, 4), False, "member order"),
        ([0, 0], (2, 1, 4), False, "member order"),
        ([0, 1], (1, 1, 4), False, "tensor shape"),
        ([0, 1], (2, 2, 4), False, "tensor shape"),
        ([0, 1], (2, 1, 3), False, "tensor shape"),
        ([0, 1], (2, 1, 4), True, "non-finite"),
    ],
)
def test_worker_gradient_rejects_member_order_shape_and_nonfinite_values(
    tmp_path: Path,
    echoed: list[int],
    shape: tuple[int, ...],
    nonfinite: bool,
    match: str,
) -> None:
    class GradientProcess:
        @staticmethod
        def request(_operation: str, _payload: dict[str, Any]) -> dict[str, Any]:
            tensor = t.zeros(shape, dtype=t.float32)
            if nonfinite:
                tensor[0, 0, 0] = float("nan")
            return {
                "gradient_sidecar": _write_sidecar(
                    tmp_path / "gradients.safetensors", "gradients", tensor
                ),
                "member_indices": echoed,
            }

    with pytest.raises(AxBenchAdvError, match=match):
        _gradient_client(GradientProcess(), tmp_path).gcg_gradient([1], [0, 1])


@pytest.mark.parametrize(
    "failure",
    [
        "descriptor_keys",
        "descriptor_shape",
        "sha",
        "permissions",
        "corrupt",
        "truncated",
        "key",
        "extra_key",
        "dtype",
        "rank",
        "tensor_shape",
    ],
)
def test_tensor_sidecar_rejects_malformed_or_unsafe_files(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "input_embeddings.safetensors"
    tensor = t.tensor([[1.0, 2.0]], dtype=t.float32)
    key = "embedding_weight"
    if failure == "dtype":
        tensor = tensor.double()
    elif failure == "rank":
        tensor = tensor.reshape(1, 1, 2)
    descriptor = _write_sidecar(path, "wrong" if failure == "key" else key, tensor)
    if failure == "descriptor_keys":
        descriptor["extra"] = True
    elif failure == "descriptor_shape":
        descriptor["shape"] = [0, 2]
    elif failure == "sha":
        descriptor["sha256"] = "0" * 64
    elif failure == "permissions":
        path.chmod(0o644)
    elif failure in {"corrupt", "truncated"}:
        payload = b"not safetensors" if failure == "corrupt" else path.read_bytes()[:-1]
        path.chmod(0o600)
        path.write_bytes(payload)
        path.chmod(0o400)
        descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif failure == "extra_key":
        save_file({key: tensor, "extra": tensor.clone()}, path)
        path.chmod(0o400)
        descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif failure == "tensor_shape":
        descriptor["shape"] = [2, 1]
    elif failure == "rank":
        descriptor["shape"] = [1, 2]
    with pytest.raises(AxBenchAdvError):
        _load_tensor_sidecar(
            descriptor,
            expected_path=path,
            expected_key=key,
            expected_rank=2,
        )


def test_worker_gradient_unlinks_stale_sidecar_before_request(tmp_path: Path) -> None:
    path = tmp_path / "gradients.safetensors"
    stale = _write_sidecar(path, "gradients", t.ones((1, 1, 4), dtype=t.float32))

    class StaleProcess:
        @staticmethod
        def request(_operation: str, _payload: dict[str, Any]) -> dict[str, Any]:
            assert not path.exists()
            return {"gradient_sidecar": stale, "member_indices": [0]}

    with pytest.raises(AxBenchAdvError, match="unavailable"):
        _gradient_client(StaleProcess(), tmp_path).gcg_gradient([1], [0])


class _ExploratoryFilterRuntime:
    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts

    def gcg_filter(self, candidates: list[list[int]]) -> tuple[GcgFilteredCandidate, ...]:
        return tuple(
            GcgFilteredCandidate(tuple(ids), self.texts[ids[0]], True, None) for ids in candidates
        )


def _exploratory_space(texts: dict[int, str]) -> AppendSuffixEditSpace:
    prepared = GcgPrepared(
        embedding_weight=t.zeros((4, 2)),
        allowed_token_ids=(1, 2, 3),
        seed_token_ids=(3,),
        member_count=1,
        identity={"rank_sentinel_id": 0},
        layouts_sha256="a" * 64,
        direction_calibration=None,
        forbidden_control_tokens=QWEN_CONTROL_TOKENS,
    )
    return AppendSuffixEditSpace(cast(GcgRuntime, _ExploratoryFilterRuntime(texts)), prepared)


def test_exploratory_filter_skips_unsafe_text_without_hiding_sampler_exhaustion() -> None:
    space = _exploratory_space({1: QWEN_CONTROL_TOKENS[0], 2: "safe"})
    proposals = {
        1: (
            TokenReplacement(position=1, token_id=1, score=0.0),
            TokenReplacement(position=1, token_id=2, score=0.0),
        )
    }
    current = [3]
    visited = {space.digest(current)}
    assert space.sample_candidates(
        proposals,
        current,
        batch_size=2,
        temperature=1.0,
        rng=random.Random(0),
        visited=visited,
    ) == [[2]]
    assert len(visited) == 3
    assert space.full_batch_pool_exhausted(proposals, current, visited)


def test_exploratory_filter_still_rejects_noncanonical_whitespace() -> None:
    space = _exploratory_space({1: " unsafe whitespace "})
    proposals = {1: (TokenReplacement(position=1, token_id=1, score=0.0),)}
    with pytest.raises(AxBenchAdvError, match="changed a valid sampled candidate"):
        space.sample_candidates(
            proposals,
            [3],
            batch_size=1,
            temperature=1.0,
            rng=random.Random(0),
            visited={space.digest([3])},
        )


@pytest.mark.parametrize(
    ("target", "filtered"),
    [
        ([1], ()),
        ([1], (GcgFilteredCandidate((1,), "safe", False, "invalid"),)),
        ([1], (GcgFilteredCandidate((2,), "safe", True, None),)),
        ([1, 2], (GcgFilteredCandidate((1, 2), "safe", True, None),)),
        ([1], (GcgFilteredCandidate((1,), "", True, None),)),
        ([1], (GcgFilteredCandidate((1,), " unsafe ", True, None),)),
        ([1], (GcgFilteredCandidate((1,), QWEN_CONTROL_TOKENS[0], True, None),)),
    ],
)
def test_retained_suffix_validation_remains_strict_in_finalize(
    target: list[int], filtered: tuple[GcgFilteredCandidate, ...]
) -> None:
    space = _exploratory_space({})
    space.runtime = cast(GcgRuntime, SimpleNamespace(gcg_filter=lambda _tokens: filtered))
    with pytest.raises(AxBenchAdvError):
        space.finalize(
            ValidatedGroup("final", [0], None),
            target,
            objective=0.0,
            accepted_steps=0,
            visited_count=1,
        )


class _DriverRuntime:
    def __init__(self) -> None:
        self.evaluated: list[tuple[str, tuple[int, ...]]] = []
        self.prepare_calls = 0
        self.filter_calls = 0

    def prepare_gcg(self, **kwargs: Any) -> GcgPrepared:
        self.prepare_calls += 1
        assert kwargs["separator"] == "\n\n"
        assert kwargs["suffix_token_count"] == 1
        return GcgPrepared(
            embedding_weight=t.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
            allowed_token_ids=(1, 2),
            seed_token_ids=(1,),
            member_count=1,
            identity={"rank_sentinel_id": 0, "vocab_size": 3},
            layouts_sha256="a" * 64,
            direction_calibration=None,
        )

    def gcg_filter(self, candidates: list[list[int]]) -> tuple[GcgFilteredCandidate, ...]:
        self.filter_calls += 1
        return tuple(
            GcgFilteredCandidate(tuple(ids), "!" if tuple(ids) == (1,) else "x", True, None)
            for ids in candidates
        )

    def gcg_score(
        self,
        candidates: list[tuple[int, ...]],
        members: list[int],
        *,
        candidate_eval_device_batch_size: int,
    ) -> list[float]:
        assert members == [0]
        assert candidate_eval_device_batch_size == 2
        return [float(tokens[0]) for tokens in candidates]

    def gcg_gradient(self, token_ids: tuple[int, ...], members: list[int]) -> list[t.Tensor]:
        assert token_ids == (1,)
        assert members == [0]
        return [t.tensor([[1.0, 0.0]])]

    def evaluate(self, suffix: str, input_ids: list[int]) -> tuple[RawEvaluation, ...]:
        self.evaluated.append((suffix, tuple(input_ids)))
        return tuple(RawEvaluation(f"g{i}", 1.0, 1.0, "{}", 0) for i in input_ids)


def test_training_rows_rejects_evaluation_prompt_overlap() -> None:
    training = ConceptTrainingPairs(
        concept_id=0,
        concept="topic",
        genre="genre",
        positive=(("overlap", "yes"),),
        negative=(("train negative", "no"),),
    )
    with pytest.raises(AxBenchAdvError, match="overlap final evaluation prompts"):
        _training_rows(training, master_seed=17, evaluation_prompts=("overlap",))


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        ((), ()),
        ((("positive", "yes"),), ()),
        ((("positive", "yes"),), (("negative 1", "no"), ("negative 2", "no"))),
    ],
)
def test_training_rows_requires_equal_nonempty_class_populations(
    positive: tuple[tuple[str, str], ...],
    negative: tuple[tuple[str, str], ...],
) -> None:
    training = ConceptTrainingPairs(0, "topic", "genre", positive, negative)
    with pytest.raises(AxBenchAdvError, match="equal nonempty positive and negative populations"):
        _training_rows(training, master_seed=17, evaluation_prompts=())


def test_score_callback_rejects_nonrectangular_member_groups() -> None:
    runtime = cast(GcgRuntime, SimpleNamespace(gcg_score=lambda *_args, **_kwargs: []))
    with pytest.raises(AxBenchAdvError, match="candidate-major rectangular"):
        _score_callback(
            runtime,
            ((0, (1,)), (1, (1,)), (0, (2,))),
            device_batch_size=2,
        )


def test_gradient_callback_rejects_mixed_suffixes() -> None:
    runtime = cast(GcgRuntime, SimpleNamespace(gcg_gradient=lambda *_args: []))
    with pytest.raises(AxBenchAdvError, match="share one current suffix"):
        _gradient_callback(runtime, ((0, (1,)), (1, (2,))))


def test_execute_gcg_cell_writes_control_and_search_receipts_without_search_judging(
    tmp_path: Path,
) -> None:
    runtime = _DriverRuntime()
    config = AxBenchAdvGcgConfig(
        device_batch_size=2,
        search=replace(
            FasterGcgConfig(),
            optimized_token_counts=(0, 1),
            max_steps=1,
            candidate_batch_size=1,
            top_k=1,
            universal_minibatch_size=1,
            master_seed=17,
        ),
    )
    training = ConceptTrainingPairs(
        concept_id=0,
        concept="topic",
        genre="genre",
        positive=(("train positive", "yes"),),
        negative=(("train negative", "no"),),
    )
    totals = {
        "actual_generation_calls": 0,
        "actual_evaluator_calls": 0,
        "paid_judge_cache_misses": 0,
    }
    result = execute_gcg_cell(
        config=config,
        runtime=runtime,  # ty: ignore[invalid-argument-type]
        method="DiffMean",
        concept_id=0,
        concept="topic",
        factor=1.0,
        training_pairs=training,
        evaluation_prompts=("held out",),
        evaluation_input_ids=(0, 1),
        output_dir=tmp_path,
        totals=totals,
    )
    assert set(result) == {1}
    assert runtime.prepare_calls == 1
    assert runtime.filter_calls == 2  # Exploratory proposal and retained suffix.
    assert runtime.evaluated == [("!", (0, 1))]
    assert totals == {
        "actual_generation_calls": 2,
        "actual_evaluator_calls": 2,
        "paid_judge_cache_misses": 0,
    }
    zero = json.loads((tmp_path / "gcg/DiffMean/c0/k0/search.json").read_text())
    retained = json.loads((tmp_path / "gcg/DiffMean/c0/k1/search.json").read_text())
    assert zero["suffix_token_ids"] == [] and zero["diagnostics"]["control"] is True
    assert retained["suffix"] == "!" and retained["diagnostics"]["control"] is False


def test_parse_gcg_config_wraps_beartype_domain_failures() -> None:
    raw = {
        "objective": "response_margin",
        "alpha": 0.01,
        "device_batch_size": 2,
        "search": {"optimized_token_counts": [0, 1], "top_k": "invalid"},
    }
    with pytest.raises(AxBenchAdvError, match="invalid gcg.search config"):
        parse_gcg_config(raw)
