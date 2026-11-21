"""Cluster-aware Stage 1c threading, identity, and fused-read regressions."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.config import SynthConfig
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.synth import negative_mode, positive_mode
from steering.eval.axbench_base.synth.batching import SynthResult
from steering.eval.axbench_base.synth.negative_mode import (
    _synthesize_all_concept_negatives_parallel,
    _synthesize_concept_avoiding_row,
    _synthesize_concept_negatives,
)
from steering.eval.axbench_base.synth.qc import AvoidContinuationOutcome
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    TargetModelMetadata,
)
from steering.eval.cluster_contract_identity import QC_TARGET_REDUCTION_VERSION
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureTarget,
)
from tests.axbench_base.stage1_protocol_helpers import (
    TEST_STAGE1_TEMPLATE,
    frame_stage1_response,
)
from tests.axbench_base.unit.test_config import make_valid_config


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 1

    def encode(
        self,
        text: str,
        prepend: int | None = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        tokens = [ord(char) for char in text]
        return ([prepend] if prepend is not None else []) + tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _Client:
    def __init__(self, response: str = "safe continuation") -> None:
        self.response = response

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del cache_key, system_prompt
        return frame_stage1_response(prompt, self.response)


def _cfg(tmp_path: Path, **overrides: Any):
    stats_path = tmp_path / "feature_statistics.safetensors"
    if not stats_path.exists():
        save_file(
            {"_placeholder": t.zeros(1, dtype=t.int64)},
            str(stats_path),
            metadata={"n_sae_features": "16"},
        )
    cfg = make_valid_config(
        run_tag="clustered_negative_mode",
        output_root=None,
        feature_statistics_path=str(stats_path),
        max_retries=0,
        qc_fallback_on_exhaust="error",
        output_length=16,
    )
    return cfg.replace(**overrides) if overrides else cfg


def _concept() -> Concept:
    return Concept(
        concept_id=3,
        feature_index=2,
        layer=8,
        sae_source="test/sae",
        description="target concept",
        corpus_count=100,
        description_source="test",
    )


def _prefix_row() -> dict[str, Any]:
    return {
        "concept_id": 3,
        "prefix_text": "prefix",
        "prefix_source_shard": 0,
        "prefix_source_doc": 7,
        "prefix_activating_seg_idx": 1,
        "verification_continuations": ["rich continuation"],
        "verification_seeds": [11],
        "verification_percentiles": [91.0],
    }


def _target(fingerprint: str = "c" * 64) -> ClusterFeatureTarget:
    return ClusterFeatureTarget(
        representative_id=2,
        allowed_member_ids=(2, 5),
        projecting_source_ids=(2, 5),
        feature_contract_fingerprint=fingerprint,
    )


def _contract(target: ClusterFeatureTarget) -> ClusterFeatureContract:
    def _target_for_representative(feature_id: int) -> ClusterFeatureTarget:
        if feature_id != target.representative_id:
            pytest.fail(f"unexpected representative {feature_id}")
        return target

    return cast(
        ClusterFeatureContract,
        SimpleNamespace(
            feature_contract_fingerprint=target.feature_contract_fingerprint,
            semantic_dict=lambda: {
                "algorithms": {
                    "qc_target_reduction": QC_TARGET_REDUCTION_VERSION,
                }
            },
            target_for_representative=_target_for_representative,
        ),
    )


def _meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _positive_training_row(gen_seed: int = 1) -> dict[str, Any]:
    return {
        "concept_id": 3,
        "category": "positive",
        "content_class": "concept_rich",
        "prefix_text": "prefix",
        "output_text": f"rich continuation {gen_seed}",
        "output_concept": "target concept",
        "n_prefix_nanochat_tokens": 7,
        "gen_source": "test",
        "gen_seed": gen_seed,
        "qc_feature_id": 2,
        "qc_seed_percentile": 10.0,
        "qc_output_percentile": 90.0,
        "qc_delta": 80.0,
        "qc_attempts": 1,
        "qc_accepted": True,
        "qc_zero_activation_verified": False,
    }


def _negative_training_row(gen_seed: int = 1) -> dict[str, Any]:
    return {
        **_positive_training_row(gen_seed),
        "prefix_source_shard": 0,
        "prefix_source_doc": 7,
        "prefix_source_seg_idx": 1,
    }


def _negative_training_pair(gen_seed: int = 1) -> list[dict[str, Any]]:
    rich = _negative_training_row(gen_seed)
    avoiding = {
        **rich,
        "category": "negative",
        "content_class": "concept_avoiding",
        "output_text": f"avoiding continuation {gen_seed}",
        "gen_seed": gen_seed + 100,
        "qc_seed_percentile": float("nan"),
        "qc_output_percentile": float("nan"),
        "qc_delta": float("nan"),
        "qc_zero_activation_verified": True,
    }
    return [rich, avoiding]


def test_negative_avoiding_row_threads_cluster_target_and_cache_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(negative_mode, "get_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(
        negative_mode,
        "parse_stage1_continuation_or_none",
        lambda **_kwargs: "safe continuation",
    )

    def _cache_key(*, model: str, prompt: str, tag: str) -> str:
        captured["cache"] = (model, prompt, tag)
        return "cache-key"

    def _classify(**kwargs: Any) -> AvoidContinuationOutcome:
        captured["cluster_target"] = kwargs["cluster_target"]
        return AvoidContinuationOutcome.INACTIVE

    monkeypatch.setattr(negative_mode, "make_cache_key", _cache_key)
    monkeypatch.setattr(negative_mode, "classify_avoid_continuation", _classify)

    row = _synthesize_concept_avoiding_row(
        _concept(),
        _prefix_row(),
        _cfg(tmp_path),
        base_model=object(),
        base_sae=object(),
        llm_client=_Client(),
        tokenizer=_Tokenizer(),
        template=TEST_STAGE1_TEMPLATE,
        sae_layer=8,
        cluster_target=target,
    )

    assert row is not None
    assert captured["cluster_target"] is target
    assert captured["cache"][2].endswith(
        f"_fc{target.feature_contract_fingerprint}"
    )


def test_negative_serial_path_threads_cluster_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    captured: list[ClusterFeatureTarget | None] = []
    monkeypatch.setattr(
        negative_mode,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: _Tokenizer(),
    )
    monkeypatch.setattr(negative_mode, "get_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(negative_mode, "load_template", lambda _name: "template")
    monkeypatch.setattr(
        negative_mode,
        "_copy_concept_rich_rows",
        lambda *_args: [],
    )

    def _avoid(*args: Any, cluster_target=None, **kwargs: Any):
        del args, kwargs
        captured.append(cluster_target)
        return {"content_class": "concept_avoiding"}

    monkeypatch.setattr(negative_mode, "_synthesize_concept_avoiding_row", _avoid)

    rows = _synthesize_concept_negatives(
        _cfg(tmp_path),
        _concept(),
        [_prefix_row()],
        base_model=object(),
        base_sae=object(),
        llm_client=_Client(),
        target_model_meta=_meta(),
        cluster_target=target,
    )

    assert captured == [target]
    assert rows == [{"content_class": "concept_avoiding"}]


def test_negative_parallel_chain_threads_target_and_cache_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    captured_targets: list[ClusterFeatureTarget | None] = []
    captured_tags: list[str] = []
    monkeypatch.setattr(
        negative_mode,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: _Tokenizer(),
    )
    monkeypatch.setattr(negative_mode, "get_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(
        negative_mode,
        "load_template",
        lambda _name: TEST_STAGE1_TEMPLATE,
    )
    monkeypatch.setattr(
        negative_mode,
        "_copy_concept_rich_rows",
        lambda *_args: [],
    )

    def _cache_key(*, model: str, prompt: str, tag: str) -> str:
        del model, prompt
        captured_tags.append(tag)
        return tag

    def _complete(_client, requests, *, handle_results, **_kwargs):
        pending = list(requests)
        while pending:
            pairs = [
                (
                    request,
                    SynthResult(
                        index=request.index,
                        response_text=frame_stage1_response(
                            request.prompt,
                            "safe continuation",
                        ),
                    ),
                )
                for request in pending
            ]
            pending = [
                request
                for request in handle_results(pairs)
                if request is not None
            ]

    def _classify(inputs, *_args, **_kwargs):
        captured_targets.extend(item.cluster_target for item in inputs)
        return [AvoidContinuationOutcome.INACTIVE] * len(inputs)

    monkeypatch.setattr(negative_mode, "make_cache_key", _cache_key)
    monkeypatch.setattr(negative_mode, "complete_synth_request_chains", _complete)
    monkeypatch.setattr(
        negative_mode,
        "classify_avoid_continuations",
        _classify,
    )
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(
            target_for_representative=lambda feature_id: (
                target
                if feature_id == target.representative_id
                else pytest.fail(f"unexpected representative {feature_id}")
            )
        ),
    )

    rows = _synthesize_all_concept_negatives_parallel(
        _cfg(
            tmp_path,
            synth=SynthConfig(
                requests_per_minute=0,
                max_concurrent=2,
            ),
        ),
        [_concept()],
        {_concept().concept_id: [_prefix_row()]},
        base_model=object(),
        base_sae=object(),
        llm_client=_Client(),
        target_model_meta=_meta(),
        cluster_contract=contract,
    )

    assert captured_targets == [target]
    assert captured_tags == [f"neg_c3_d7_a0_fc{target.feature_contract_fingerprint}"]
    assert [row["content_class"] for row in rows] == ["concept_avoiding"]


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _prefix_table(fingerprint: str | None, value: int) -> pa.Table:
    metadata = {}
    if fingerprint is not None:
        metadata[b"feature_contract_fingerprint"] = fingerprint.encode("utf-8")
    return pa.table({"concept_id": [value]}).replace_schema_metadata(metadata)


@pytest.mark.parametrize("fingerprint", [None, "d" * 64])
def test_clustered_prefix_reader_rejects_missing_or_mismatched_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fingerprint: str | None,
) -> None:
    path = tmp_path / "negative_mode_prefixes.parquet"
    pq.write_table(_prefix_table(fingerprint, 3), path)
    monkeypatch.setattr(
        negative_mode,
        "validate_negative_prefixes_extractor_identity_metadata",
        lambda _path, *, cfg, consumer, schema: None,
    )
    monkeypatch.setattr(
        negative_mode,
        "validate_negative_prefixes_tokenizer_metadata",
        lambda _path, *, expected_tokenizer_id, consumer, schema: None,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="absent or mismatched feature-contract fingerprint",
    ):
        negative_mode._read_cluster_pinned_negative_prefixes(
            _cfg(tmp_path),
            path=path,
            expected_fingerprint=_target().feature_contract_fingerprint,
        )


def test_clustered_prefix_reader_validates_and_returns_the_same_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "negative_mode_prefixes.parquet"
    target = _target()
    pq.write_table(_prefix_table(target.feature_contract_fingerprint, 3), path)
    schemas: list[pa.Schema] = []
    monkeypatch.setattr(
        negative_mode,
        "validate_negative_prefixes_extractor_identity_metadata",
        lambda _path, *, cfg, consumer, schema: schemas.append(schema),
    )
    monkeypatch.setattr(
        negative_mode,
        "validate_negative_prefixes_tokenizer_metadata",
        lambda _path, *, expected_tokenizer_id, consumer, schema: schemas.append(
            schema
        ),
    )

    rows = negative_mode._read_cluster_pinned_negative_prefixes(
        _cfg(tmp_path),
        path=path,
        expected_fingerprint=target.feature_contract_fingerprint,
    )

    assert rows == [{"concept_id": 3}]
    assert len(schemas) == 2
    assert schemas[0].equals(schemas[1], check_metadata=True)


def test_clustered_prefix_reader_rejects_byte_length_mismatch_with_matching_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "negative_mode_prefixes.parquet"
    target = _target()
    pq.write_table(_prefix_table(target.feature_contract_fingerprint, 3), path)
    digest = negative_mode.stream_hash_file_stable(path)
    monkeypatch.setattr(
        negative_mode,
        "stream_hash_file_stable",
        lambda _path: SimpleNamespace(
            path=digest.path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            byte_length=digest.byte_length + 1,
        ),
    )
    monkeypatch.setattr(
        negative_mode,
        "validate_negative_prefixes_extractor_identity_metadata",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        negative_mode,
        "validate_negative_prefixes_tokenizer_metadata",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="changed between stable hashing and immutable read",
    ):
        negative_mode._read_cluster_pinned_negative_prefixes(
            _cfg(tmp_path),
            path=path,
            expected_fingerprint=target.feature_contract_fingerprint,
        )


def test_clustered_prefix_reader_wraps_stable_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "negative_mode_prefixes.parquet"
    path.write_bytes(b"unreadable")

    def _fail_stable_read(_path: Path):
        raise OSError("synthetic stable-read failure")

    monkeypatch.setattr(
        negative_mode,
        "stream_hash_file_stable",
        _fail_stable_read,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Could not read a stable clustered Stage 0c negative-prefix artifact",
    ):
        negative_mode._read_cluster_pinned_negative_prefixes(
            _cfg(tmp_path),
            path=path,
            expected_fingerprint=_target().feature_contract_fingerprint,
        )


def test_clustered_prefix_reader_wraps_parse_failure(tmp_path: Path) -> None:
    path = tmp_path / "negative_mode_prefixes.parquet"
    path.write_bytes(b"not a parquet payload")

    with pytest.raises(
        AxBenchBaseError,
        match="Could not parse clustered Stage 0c negative prefixes",
    ):
        negative_mode._read_cluster_pinned_negative_prefixes(
            _cfg(tmp_path),
            path=path,
            expected_fingerprint=_target().feature_contract_fingerprint,
        )


def test_clustered_prefix_reader_rejects_a_to_b_to_a_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "negative_mode_prefixes.parquet"
    target = _target()
    bytes_a = _parquet_bytes(
        _prefix_table(target.feature_contract_fingerprint, 3)
    )
    bytes_b = _parquet_bytes(
        _prefix_table(target.feature_contract_fingerprint, 999)
    )
    path.write_bytes(bytes_a)
    digest_a = negative_mode.stream_hash_file_stable(path)

    class _RoundTripRead:
        def read_bytes(self) -> bytes:
            path.write_bytes(bytes_b)
            observed = path.read_bytes()
            path.write_bytes(bytes_a)
            return observed

    monkeypatch.setattr(
        negative_mode,
        "stream_hash_file_stable",
        lambda _path: SimpleNamespace(
            path=_RoundTripRead(),
            sha256=digest_a.sha256,
            byte_length=digest_a.byte_length,
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="changed between stable hashing and immutable read",
    ):
        negative_mode._read_cluster_pinned_negative_prefixes(
            _cfg(tmp_path),
            path=path,
            expected_fingerprint=target.feature_contract_fingerprint,
        )
    assert path.read_bytes() == bytes_a


def test_clustered_positive_writer_requires_concept_population(tmp_path: Path) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="clustered synth writer requires its concept population",
    ):
        positive_mode._write_training_parquet(
            [],
            tmp_path / "positive.parquet",
            tokenizer_id="test-tokenizer",
            cfg=_cfg(tmp_path),
            cluster_contract=_contract(_target()),
        )


def test_clustered_positive_writer_metadata_and_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    contract = _contract(target)
    prefix_text = (
        "A complete, realistic clustered-synthesis prefix with enough "
        "decoded characters to satisfy the production Stage 1b contract. "
    )
    positive_row: dict[str, Any] = {
        **_positive_training_row(),
        "prefix_text": prefix_text,
    }
    avoiding_row: dict[str, Any] = {
        **positive_row,
        "category": "negative",
        "content_class": "concept_avoiding",
        "output_text": "avoiding continuation",
        "gen_seed": 101,
        "qc_seed_percentile": float("nan"),
        "qc_output_percentile": float("nan"),
        "qc_delta": float("nan"),
        "qc_zero_activation_verified": True,
    }
    rows = [positive_row, avoiding_row]
    cfg = _cfg(
        tmp_path,
        conditioning={"feature_clusters": {"path": "/clusters.yaml"}},
        n_positives_per_concept=1,
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        positive_mode,
        "load_cluster_candidate_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        positive_mode,
        "_infer_n_sae_features",
        lambda _base_sae: 16,
    )
    monkeypatch.setattr(
        positive_mode,
        "build_percentile_table_from_stats",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        positive_mode,
        "PrefixPercentileCache",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        positive_mode,
        "SynthResponseCache",
        lambda _path: nullcontext(object()),
    )
    monkeypatch.setattr(
        positive_mode,
        "_build_synth_client",
        lambda *_args, **_kwargs: object(),
    )
    trackers = [object()]
    monkeypatch.setattr(
        positive_mode,
        "_prepare_positive_target_trackers",
        lambda *_args, **_kwargs: trackers,
    )
    monkeypatch.setattr(
        positive_mode,
        "_run_positive_target_refill_engine",
        lambda _cfg, actual_trackers, *_args, **_kwargs: (
            {3: list(rows)}
            if actual_trackers is trackers
            else pytest.fail("unexpected Stage 1b tracker collection")
        ),
    )
    monkeypatch.setattr(
        positive_mode,
        "release_cuda_iteration_memory",
        lambda: None,
    )
    monkeypatch.setattr(positive_mode, "get_tokenizer", lambda: _Tokenizer())
    for row in rows:
        row["n_prefix_nanochat_tokens"] = positive_mode._exact_output_start(
            prefix_text,
            row["output_text"],
        )
    seed = SeedRow(
        shard_idx=0,
        doc_idx=7,
        source_nanochat_token_count=2,
        prefix_source_token_start=0,
        prefix_source_token_end=1,
        prefix_text=prefix_text,
        prefix_content_nanochat_tokens=[1],
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="continuation",
        continuation_content_nanochat_tokens=[2],
    )

    out = positive_mode.synthesize_positive_mode(
        cfg,
        [_concept()],
        [seed],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_meta(),
        seed_rows_sha256="a" * 64,
        cluster_contract=contract,
    )
    original_bytes = out.read_bytes()
    metadata = pq.read_schema(out).metadata or {}
    assert metadata[b"feature_contract_fingerprint"] == (
        target.feature_contract_fingerprint.encode("utf-8")
    )
    assert metadata[b"cluster_qc_target_algorithm"] == (
        QC_TARGET_REDUCTION_VERSION.encode("utf-8")
    )
    assert json.loads(metadata[b"cluster_target_members_json"]) == {"2": [2, 5]}
    contract_metadata = validate_positive_qc_contract(
        out,
        consumer="clustered writer test",
        expected_accept_threshold=cfg.accept_threshold,
    )
    assert contract_metadata.accept_threshold_points == 30.0

    replay = positive_mode.synthesize_positive_mode(
        cfg,
        [_concept()],
        [seed],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_meta(),
        seed_rows_sha256="a" * 64,
        cluster_contract=contract,
    )
    assert replay == out
    assert replay.read_bytes() == original_bytes

    replacement_row: dict[str, Any] = {
        **_positive_training_row(gen_seed=2),
        "prefix_text": prefix_text,
    }
    replacement_avoiding_row: dict[str, Any] = {
        **replacement_row,
        "category": "negative",
        "content_class": "concept_avoiding",
        "output_text": "replacement avoiding continuation",
        "gen_seed": 102,
        "qc_seed_percentile": float("nan"),
        "qc_output_percentile": float("nan"),
        "qc_delta": float("nan"),
        "qc_zero_activation_verified": True,
    }
    replacement_rows = [replacement_row, replacement_avoiding_row]
    for row in replacement_rows:
        row["n_prefix_nanochat_tokens"] = positive_mode._exact_output_start(
            prefix_text,
            row["output_text"],
        )
    rows[:] = replacement_rows
    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        positive_mode.synthesize_positive_mode(
            cfg,
            [_concept()],
            [seed],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_meta(),
            seed_rows_sha256="a" * 64,
            cluster_contract=contract,
        )
    assert out.read_bytes() == original_bytes


def test_clustered_negative_writer_metadata_and_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    contract = _contract(target)
    rows = _negative_training_pair()
    cfg = _cfg(
        tmp_path,
        conditioning={"feature_clusters": {"path": "/clusters.yaml"}},
        synth=SynthConfig(requests_per_minute=0, max_concurrent=1),
    )
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        negative_mode,
        "load_cluster_candidate_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        negative_mode,
        "_load_negative_prefixes",
        lambda *_args, **_kwargs: [_prefix_row()],
    )
    monkeypatch.setattr(
        negative_mode,
        "SynthResponseCache",
        lambda _path: nullcontext(object()),
    )
    monkeypatch.setattr(
        negative_mode,
        "OpenRouterSynthClient",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        negative_mode,
        "_synthesize_concept_negatives",
        lambda *_args, **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        negative_mode,
        "release_cuda_iteration_memory",
        lambda: None,
    )

    out = negative_mode.synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_meta(),
        cluster_contract=contract,
    )
    original_bytes = out.read_bytes()
    metadata = pq.read_schema(out).metadata or {}
    assert metadata[b"feature_contract_fingerprint"] == (
        target.feature_contract_fingerprint.encode("utf-8")
    )
    assert metadata[b"cluster_qc_target_algorithm"] == (
        QC_TARGET_REDUCTION_VERSION.encode("utf-8")
    )
    assert json.loads(metadata[b"cluster_target_members_json"]) == {"2": [2, 5]}

    replay = negative_mode.synthesize_negative_mode(
        cfg,
        [_concept()],
        base_model=object(),
        base_sae=object(),
        target_model_meta=_meta(),
        cluster_contract=contract,
    )
    assert replay == out
    assert replay.read_bytes() == original_bytes

    rows[:] = _negative_training_pair(gen_seed=2)
    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        negative_mode.synthesize_negative_mode(
            cfg,
            [_concept()],
            base_model=object(),
            base_sae=object(),
            target_model_meta=_meta(),
            cluster_contract=contract,
        )
    assert out.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("entrypoint", "clustered_cfg", "with_contract"),
    [
        ("positive", True, False),
        ("positive", False, True),
        ("negative", True, False),
        ("negative", False, True),
    ],
)
def test_stage1_entrypoints_reject_cluster_config_contract_mismatch(
    tmp_path: Path,
    entrypoint: str,
    clustered_cfg: bool,
    with_contract: bool,
) -> None:
    conditioning = (
        {"feature_clusters": {"path": "/clusters.yaml"}}
        if clustered_cfg
        else None
    )
    cfg = _cfg(tmp_path, conditioning=conditioning)
    contract = cast(ClusterFeatureContract, object()) if with_contract else None
    concept = _concept()

    with pytest.raises(
        AxBenchBaseError,
        match="requires config and validated feature-contract state",
    ):
        if entrypoint == "positive":
            positive_mode.synthesize_positive_mode(
                cfg,
                [concept],
                [
                    SeedRow(
                        shard_idx=0,
                        doc_idx=0,
                        source_nanochat_token_count=2,
                        prefix_source_token_start=0,
                        prefix_source_token_end=1,
                        prefix_text="prefix",
                        prefix_content_nanochat_tokens=[1],
                        sae_feature_ids=[],
                        sae_feature_mags=[],
                        continuation_text="continuation",
                        continuation_content_nanochat_tokens=[2],
                    )
                ],
                base_model=object(),
                base_sae=object(),
                target_model_meta=_meta(),
                seed_rows_sha256="a" * 64,
                cluster_contract=contract,
            )
        else:
            negative_mode.synthesize_negative_mode(
                cfg,
                [concept],
                base_model=object(),
                base_sae=object(),
                target_model_meta=_meta(),
                cluster_contract=contract,
            )
