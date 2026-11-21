"""Tests for Stage 0c cross-prefix verification wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from steering.eval.axbench_base import _stage0c_internals as _si
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import Concept


class _CandidateIndex:
    def __init__(self, candidates: list[_si._PrefixCandidate]) -> None:
        self._candidates = list(candidates)

    def candidates_for_feature(self, feature_id: int) -> list[_si._PrefixCandidate]:
        del feature_id
        return list(self._candidates)


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return ",".join(str(tid) for tid in tokens)


def _cfg(**overrides: Any) -> AxBenchBaseConfig:
    values: dict[str, Any] = {
        "validation_shards": [0],
        "generation": SimpleNamespace(
            master_seed=1000,
            temperature=0.7,
            top_k_sampling=50,
        ),
        "n_verified_prefixes_per_concept_target": 3,
        "n_verification_seeds": 2,
        "min_consistency_passes": 1,
        "verification_percentile_threshold": 0.5,
        "verification_length": 4,
        "verification_candidate_attempt_limit": None,
        "verification_zero_hit_bail_after": None,
        "verification_cross_prefix_batch_size": 1,
        "verification_api_prefix_wave_size": 100,
        "verification_prefill_mode": "shared",
        "extraction_tokenizer_id": "google/gemma-2-2b",
    }
    values.update(overrides)
    return cast(AxBenchBaseConfig, SimpleNamespace(**values))


def _concept() -> Concept:
    return Concept(
        concept_id=7,
        feature_index=123,
        layer=8,
        sae_source="gemmascope",
        description="test concept",
        corpus_count=10,
        description_source="test",
    )


def _candidates() -> list[_si._PrefixCandidate]:
    return [
        _si._PrefixCandidate(
            shard_idx=0,
            doc_idx=doc_idx,
            seg_idx=0,
            prefix_text=f"prefix-{doc_idx}",
            prefix_gemma_tokens=[doc_idx, doc_idx + 1],
        )
        for doc_idx in range(1, 7)
    ]


def _install_common_patches(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: _Tokenizer(),
    )
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda _tokenizer: set())
    monkeypatch.setattr(
        _si,
        "compute_fc_percentiles",
        lambda *args, **kwargs: (50.0, 80.0),
    )


def test_verify_concept_cross_prefix_matches_serial_outputs(monkeypatch: Any):
    """K>1 consumes the same verified rows in the same order as K=1."""
    _install_common_patches(monkeypatch)

    def fake_batched(
        _base_model: object,
        token_ids: list[int],
        *,
        seeds: list[int],
        **_kwargs: Any,
    ) -> list[list[int]]:
        return [[token_ids[0], seed % 997] for seed in seeds]

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        return [
            [[token_ids[0], seed % 997] for seed in seeds]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(_si, "batched_generate_seeds", fake_batched)
    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)
    candidates = _candidates()

    serial_rows = _si._verify_concept(
        _cfg(verification_cross_prefix_batch_size=1),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(candidates),
    )
    cross_rows = _si._verify_concept(
        _cfg(verification_cross_prefix_batch_size=2),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(candidates),
    )

    assert cross_rows == serial_rows


def test_verify_concept_cross_prefix_uses_seed_rows_and_chunk_size(
    monkeypatch: Any,
) -> None:
    """The batch path flattens K prefixes x N seeds without changing seeds."""
    _install_common_patches(monkeypatch)
    chunks: list[tuple[list[list[int]], list[list[int]]]] = []
    prefill_modes: list[object] = []

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        chunks.append((token_id_rows, seed_rows))
        prefill_modes.append(_kwargs["prefill_mode"])
        return [
            [[token_ids[0], seed % 997] for seed in seeds]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)
    rows = _si._verify_concept(
        _cfg(
            n_verified_prefixes_per_concept_target=5,
            verification_cross_prefix_batch_size=4,
            verification_prefill_mode="repeated",
        ),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )

    assert len(rows) == 5
    assert chunks
    assert prefill_modes and set(prefill_modes) == {"repeated"}
    first_token_rows, first_seed_rows = chunks[0]
    assert len(first_token_rows) == 4
    assert len(first_seed_rows) == 4
    for token_ids, seeds in zip(first_token_rows, first_seed_rows, strict=True):
        doc_idx = token_ids[0]
        expected = [
            1000
            + _si._VERIFY_SEED_OFFSET
            + 7 * _si._CONCEPT_STRIDE
            + vseed_idx * _si._VSEED_INCREMENT
            + doc_idx
            for vseed_idx in range(2)
        ]
        assert seeds == expected


def test_verify_concept_cross_prefix_defers_per_seed_cleanup(
    monkeypatch: Any,
) -> None:
    """Cross-prefix Stage 0c scores continuations without per-seed cleanup."""
    _install_common_patches(monkeypatch)
    release_flags: list[bool] = []

    def fake_compute(
        _prefix_text: str,
        _continuation_text: str,
        _f_c: int,
        _base_model: object,
        _base_sae: object,
        _percentile_table: t.Tensor,
        *,
        sae_layer: int,
        tokenizer_id: str,
        release_memory: bool = True,
    ) -> tuple[float, float]:
        del sae_layer, tokenizer_id
        release_flags.append(release_memory)
        return 50.0, 80.0

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        return [
            [[token_ids[0], seed % 997] for seed in seeds]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(_si, "compute_fc_percentiles", fake_compute)
    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)

    rows = _si._verify_concept(
        _cfg(
            n_verified_prefixes_per_concept_target=4,
            n_verification_seeds=2,
            verification_cross_prefix_batch_size=4,
        ),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )

    assert len(rows) == 4
    assert release_flags == [False] * 8


def test_prefill_mode_token_difference_changes_acceptance_without_reordering(
    monkeypatch: Any,
) -> None:
    _install_common_patches(monkeypatch)
    generated_by_mode: dict[str, list[int]] = {"shared": [], "repeated": []}

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        prefill_mode: str,
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        generated_by_mode[prefill_mode].extend(row[0] for row in token_id_rows)
        generated_token = 1 if prefill_mode == "shared" else 2
        return [[[generated_token] for _seed in seeds] for seeds in seed_rows]

    def fake_compute(
        _prefix_text: str,
        continuation_text: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[float, float]:
        return 50.0, 80.0 if continuation_text == "1" else 0.0

    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)
    monkeypatch.setattr(_si, "compute_fc_percentiles", fake_compute)

    common = {
        "n_verified_prefixes_per_concept_target": 2,
        "n_verification_seeds": 1,
        "verification_cross_prefix_batch_size": 2,
    }
    shared_rows = _si._verify_concept(
        _cfg(**common, verification_prefill_mode="shared"),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )
    repeated_rows = _si._verify_concept(
        _cfg(**common, verification_prefill_mode="repeated"),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )

    assert len(shared_rows) == 2
    assert repeated_rows == []
    assert [row.prefix_text for row in shared_rows] == [
        f"prefix-{doc_idx}" for doc_idx in generated_by_mode["shared"][:2]
    ]
    assert len(generated_by_mode["shared"]) == 2
    assert len(generated_by_mode["repeated"]) == len(_candidates())


def test_verify_concept_cross_prefix_generation_exception_reraises(
    monkeypatch: Any,
) -> None:
    _install_common_patches(monkeypatch)

    def fake_cross(*_args: Any, **_kwargs: Any) -> list[list[list[int]]]:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)

    with pytest.raises(RuntimeError, match="generation failed"):
        _si._verify_concept(
            _cfg(verification_cross_prefix_batch_size=2),
            _concept(),
            parquets=[],
            feature_shards=[],
            base_model=object(),
            base_sae=object(),
            percentile_table=t.empty(0),
            sae_layer=8,
            base_context_window=128,
            candidate_index=_CandidateIndex(_candidates()),
        )


def test_verify_concept_cross_prefix_scoring_exception_reraises(
    monkeypatch: Any,
) -> None:
    _install_common_patches(monkeypatch)

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        return [
            [[token_ids[0], seed % 997] for seed in seeds]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    def fake_compute(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)
    monkeypatch.setattr(_si, "compute_fc_percentiles", fake_compute)

    with pytest.raises(RuntimeError, match="scoring failed"):
        _si._verify_concept(
            _cfg(verification_cross_prefix_batch_size=2),
            _concept(),
            parquets=[],
            feature_shards=[],
            base_model=object(),
            base_sae=object(),
            percentile_table=t.empty(0),
            sae_layer=8,
            base_context_window=128,
            candidate_index=_CandidateIndex(_candidates()),
        )


def test_verify_concept_serial_path_keeps_per_seed_cleanup_behavior(
    monkeypatch: Any,
) -> None:
    """The serial path still lets compute_fc_percentiles own cleanup."""
    _install_common_patches(monkeypatch)
    release_flags: list[bool] = []

    def fake_batched(
        _base_model: object,
        token_ids: list[int],
        *,
        seeds: list[int],
        **_kwargs: Any,
    ) -> list[list[int]]:
        return [[token_ids[0], seed % 997] for seed in seeds]

    def fake_compute(
        _prefix_text: str,
        _continuation_text: str,
        _f_c: int,
        _base_model: object,
        _base_sae: object,
        _percentile_table: t.Tensor,
        *,
        sae_layer: int,
        tokenizer_id: str,
        release_memory: bool = True,
    ) -> tuple[float, float]:
        del sae_layer, tokenizer_id
        release_flags.append(release_memory)
        return 50.0, 80.0

    monkeypatch.setattr(_si, "batched_generate_seeds", fake_batched)
    monkeypatch.setattr(_si, "compute_fc_percentiles", fake_compute)

    rows = _si._verify_concept(
        _cfg(
            n_verified_prefixes_per_concept_target=1,
            n_verification_seeds=2,
            verification_cross_prefix_batch_size=1,
            verification_prefill_mode="repeated",
        ),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )

    assert len(rows) == 1
    assert release_flags == [True, True]


def test_verify_concept_cross_prefix_target_short_circuit_is_visible_serial(
    monkeypatch: Any,
) -> None:
    """The batch path must not score candidates past the target boundary."""
    _install_common_patches(monkeypatch)
    generated_chunk_sizes: list[int] = []
    scored_prefixes: list[str] = []

    def fake_compute(
        prefix_text: str, *_args: Any, **_kwargs: Any,
    ) -> tuple[float, float]:
        scored_prefixes.append(prefix_text)
        if len(scored_prefixes) > 1:
            raise AssertionError("scored after target was already reachable")
        return 50.0, 80.0

    monkeypatch.setattr(_si, "compute_fc_percentiles", fake_compute)

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        generated_chunk_sizes.append(len(token_id_rows))
        return [
            [[token_ids[0], seed % 997] for seed in seeds]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)
    rows = _si._verify_concept(
        _cfg(
            verification_cross_prefix_batch_size=4,
            n_verified_prefixes_per_concept_target=1,
            n_verification_seeds=1,
        ),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )

    assert len(rows) == 1
    assert generated_chunk_sizes == [1]
    assert len(scored_prefixes) == 1


def test_verify_concept_cross_prefix_zero_hit_bailout_consumes_in_order(
    monkeypatch: Any,
) -> None:
    """The batch path must not score candidates past the bailout boundary."""
    monkeypatch.setattr(
        _si,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: _Tokenizer(),
    )
    monkeypatch.setattr(_si, "_tokenizer_stop_token_ids", lambda _tokenizer: set())
    generated_chunk_sizes: list[int] = []
    scored_prefixes: list[str] = []

    def fake_compute(
        prefix_text: str, *_args: Any, **_kwargs: Any,
    ) -> tuple[float, float]:
        scored_prefixes.append(prefix_text)
        if len(scored_prefixes) > 2:
            raise AssertionError("scored after zero-hit bailout was reachable")
        return 50.0, 0.0

    monkeypatch.setattr(_si, "compute_fc_percentiles", fake_compute)
    counters: list[tuple[str, int, dict[str, Any]]] = []
    monkeypatch.setattr(
        _si,
        "record_counter",
        lambda name, value, **kwargs: counters.append((name, value, kwargs)),
    )

    def fake_cross(
        _base_model: object,
        token_id_rows: list[list[int]],
        *,
        seed_rows: list[list[int]],
        **_kwargs: Any,
    ) -> list[list[list[int]]]:
        generated_chunk_sizes.append(len(token_id_rows))
        return [
            [[token_ids[0], seed % 997] for seed in seeds]
            for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True)
        ]

    monkeypatch.setattr(_si, "cross_prefix_batched_generate_seeds", fake_cross)
    rows = _si._verify_concept(
        _cfg(
            verification_cross_prefix_batch_size=4,
            verification_zero_hit_bail_after=2,
            n_verification_seeds=1,
        ),
        _concept(),
        parquets=[],
        feature_shards=[],
        base_model=object(),
        base_sae=object(),
        percentile_table=t.empty(0),
        sae_layer=8,
        base_context_window=128,
        candidate_index=_CandidateIndex(_candidates()),
    )

    assert rows == []
    assert generated_chunk_sizes == [2]
    assert len(scored_prefixes) == 2
    assert any(
        name == "stage0c.verification_zero_hit_bailouts"
        and kwargs["concept_id"] == 7
        for name, _value, kwargs in counters
    )
