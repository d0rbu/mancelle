"""Correctness-first Stage 0b candidate and extraction-orchestration tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from sae_lens import SAE
from transformer_lens import HookedTransformer

from steering.eval.axbench_base import seed_pools
from steering.eval.axbench_base.seed_contract import SeedPoolExpectation
from steering.eval.axbench_base.seed_pools import (
    SeedCandidate,
    _build_structural_candidate,
    _candidate_digest,
    _candidate_shards,
    _canonical_feature_digest,
    _extract_candidate_shard,
    _feature_sidecar_path,
    _forward_model,
    _merge_candidate_features,
    _process_or_resume_feature_shard,
    _stream_candidates,
    _write_or_validate_candidate_shards,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.extract_features import process_shard
from steering.feature_io import feature_path, load_features, save_features
from steering.feature_paths import parse_feature_dir
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from tests.axbench_base.unit.test_config import make_valid_config
from tests.runtime_attribute_helpers import set_runtime_attribute
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)


class _RawTokenizer:
    def __init__(
        self,
        source_tokens: dict[str, list[int]],
        *,
        bos_id: int = 99,
        empty_tokens: set[int] | None = None,
    ) -> None:
        self.source_tokens = source_tokens
        self.bos_id = bos_id
        self.empty_tokens = empty_tokens or set()
        self.encode_calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encode_calls.append(text)
        if text in self.source_tokens:
            return list(self.source_tokens[text])
        # Deliberately differs from the source tokenization. Tests use this to
        # prove Stage 0b never reconstructs stored slices through re-encoding.
        return [10_000 + ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            "" if token in self.empty_tokens else f"<{token}>" for token in tokens
        )

    def get_bos_token_id(self) -> int:
        return self.bos_id


class _ExtractionTokenizer:
    def __init__(self, ids: list[int] | None = None, *, bos_id: int = 2) -> None:
        self.ids = ids
        self.bos_token_id = bos_id

    def __call__(self, text: str, *, padding: bool) -> dict[str, list[int]]:
        assert padding is False
        ids = self.ids if self.ids is not None else [self.bos_token_id, *range(len(text))]
        return {"input_ids": list(ids)}


def _stage0b_wave_worker(
    rank: int,
    world_size: int,
    root_value: str,
    candidate_count: int,
    fail_rank: int | None,
) -> dict[str, Any]:
    """Exercise real gloo wave rendezvous with rank-local fake extraction."""

    candidate_paths = [
        Path(root_value) / f"candidate_{index:05d}.parquet"
        for index in range(candidate_count)
    ]
    seen: list[int] = []
    original = seed_pools._process_or_resume_feature_shard

    def fake_process(**kwargs: Any) -> dict[str, Any]:
        index = candidate_paths.index(kwargs["candidate_path"])
        if fail_rank == rank:
            raise RuntimeError(f"injected rank {rank} failure")
        seen.append(index)
        return {"canonical_output_sha256": f"{index:064x}"}

    set_runtime_attribute(
        seed_pools,
        "_process_or_resume_feature_shard",
        fake_process,
    )
    try:
        try:
            seed_pools._collect_feature_shards(
                cfg=make_valid_config(),
                candidate_paths=candidate_paths,
                root=Path(root_value),
                rank=rank,
                world_size=world_size,
                base_model=object(),
                base_sae=object(),
                extraction_fingerprint="e" * 64,
                runtime_identity={"backend": "test"},
            )
        except AxBenchBaseError as exc:
            return {"seen": seen, "error": str(exc)}
        return {"seen": seen, "error": None}
    finally:
        seed_pools._process_or_resume_feature_shard = original


def _stage0b_rank_agreement_worker(
    rank: int,
    world_size: int,
) -> str | None:
    try:
        seed_pools._require_rank_agreement(
            "injected identity",
            f"rank-{rank}",
            world_size=world_size,
        )
    except AxBenchBaseError as exc:
        return str(exc)
    return None


def _stage0b_collective_failure_worker(
    rank: int,
    world_size: int,
    fail_rank: int,
) -> str | None:
    def operation() -> int:
        if rank == fail_rank:
            raise RuntimeError(f"merge failed on rank {rank}")
        return rank

    try:
        seed_pools._collective_call(
            "injected merge",
            operation,
            rank=rank,
            world_size=world_size,
        )
    except AxBenchBaseError as exc:
        return str(exc)
    return None


def _stage0b_resume_worker(
    rank: int,
    world_size: int,
    root_value: str,
    candidate_names: list[str],
) -> list[str]:
    root = Path(root_value)
    candidate_paths = [root / "candidates" / name for name in candidate_names]
    seen: list[str] = []
    original = seed_pools._process_or_resume_feature_shard

    def tracking_process(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["candidate_path"].name)
        return original(**kwargs)

    set_runtime_attribute(
        seed_pools,
        "_process_or_resume_feature_shard",
        tracking_process,
    )
    try:
        seed_pools._collect_feature_shards(
            cfg=make_valid_config(),
            candidate_paths=candidate_paths,
            root=root,
            rank=rank,
            world_size=world_size,
            base_model=object(),
            base_sae=object(),
            extraction_fingerprint="e" * 64,
            runtime_identity={"backend": "test"},
        )
        return seen
    finally:
        seed_pools._process_or_resume_feature_shard = original


def _candidate(
    source_tokens: list[int],
    *,
    continuation_length: int = 3,
    prefix_cap: int = 99,
    extraction_tokenizer: Any | None = None,
    empty_tokens: set[int] | None = None,
    ignore_bos_features: bool = False,
) -> tuple[SeedCandidate | None, str | None, _RawTokenizer]:
    tokenizer = _RawTokenizer({"source": source_tokens}, empty_tokens=empty_tokens)
    candidate, reason = _build_structural_candidate(
        nanochat_tokenizer=tokenizer,
        extraction_tokenizer=extraction_tokenizer or _ExtractionTokenizer(),
        text="source",
        shard_idx=4,
        doc_idx=7,
        candidate_index=11,
        continuation_length=continuation_length,
        prefix_cap=prefix_cap,
        extraction_limit=128,
        ignore_bos_features=ignore_bos_features,
    )
    return candidate, reason, tokenizer


@pytest.mark.parametrize(
    ("source_length", "expected_reason", "expected_prefix", "expected_continuation"),
    [
        (5, "source_too_short", None, None),
        (6, None, [0, 1, 2], [3, 4, 5]),
        (7, None, [0, 1, 2, 3], [4, 5, 6]),
    ],
)
def test_raw_slice_threshold_equations(
    source_length: int,
    expected_reason: str | None,
    expected_prefix: list[int] | None,
    expected_continuation: list[int] | None,
) -> None:
    candidate, reason, _tokenizer = _candidate(list(range(source_length)))
    assert reason == expected_reason
    if candidate is None:
        assert expected_prefix is None
        return
    assert candidate.prefix_content_nanochat_tokens == expected_prefix
    assert candidate.continuation_content_nanochat_tokens == expected_continuation
    assert candidate.prefix_source_token_end == source_length - 3
    assert candidate.prefix_source_token_start == 0


@pytest.mark.parametrize(
    ("cap", "expected_start", "expected_prefix"),
    [
        (3, 9, [9, 10, 11]),
        (6, 6, [6, 7, 8, 9, 10, 11]),
        (20, 0, list(range(12))),
    ],
)
def test_prefix_cap_selects_immediately_preceding_source_tokens(
    cap: int,
    expected_start: int,
    expected_prefix: list[int],
) -> None:
    candidate, reason, _tokenizer = _candidate(
        list(range(15)), prefix_cap=cap
    )
    assert reason is None and candidate is not None
    assert candidate.prefix_source_token_start == expected_start
    assert candidate.prefix_source_token_end == 12
    assert candidate.prefix_content_nanochat_tokens == expected_prefix
    assert candidate.continuation_content_nanochat_tokens == [12, 13, 14]


def test_literal_bos_token_ids_remain_source_payload() -> None:
    candidate, reason, _tokenizer = _candidate([10, 99, 11, 99, 12, 13])
    assert reason is None and candidate is not None
    assert candidate.prefix_content_nanochat_tokens == [10, 99, 11]
    assert candidate.continuation_content_nanochat_tokens == [99, 12, 13]


def test_source_slices_are_not_decode_reencoded() -> None:
    candidate, reason, tokenizer = _candidate([2, 4, 6, 8, 10, 12])
    assert reason is None and candidate is not None
    assert candidate.prefix_content_nanochat_tokens == [2, 4, 6]
    assert candidate.continuation_content_nanochat_tokens == [8, 10, 12]
    assert tokenizer.encode_calls == ["source"]
    assert tokenizer.encode(candidate.text) != candidate.prefix_content_nanochat_tokens


def test_zero_width_decoded_tokens_do_not_change_source_bounds() -> None:
    candidate, reason, _tokenizer = _candidate(
        [1, 2, 3, 4, 5, 6], empty_tokens={2}
    )
    assert reason is None and candidate is not None
    assert candidate.prefix_content_nanochat_tokens == [1, 2, 3]
    assert candidate.prefix_source_token_end - candidate.prefix_source_token_start == 3
    assert candidate.text == "<1><3>"


def test_entirely_empty_decoded_prefix_is_skipped() -> None:
    candidate, reason, _tokenizer = _candidate(
        [1, 2, 3, 4, 5, 6], empty_tokens={1, 2, 3}
    )
    assert candidate is None
    assert reason == "empty_prefix"


def test_empty_decoded_continuation_preserves_authoritative_tokens() -> None:
    candidate, reason, _tokenizer = _candidate(
        [1, 2, 3, 4, 5, 6], empty_tokens={4, 5, 6}
    )
    assert reason is None and candidate is not None
    assert candidate.continuation_text == ""
    assert candidate.continuation_content_nanochat_tokens == [4, 5, 6]
    assert seed_pools._candidate_from_row(candidate.parquet_row()) == candidate


def test_extraction_preflight_rejects_empty_over_limit_and_unpoolable() -> None:
    candidate, reason, _ = _candidate(
        list(range(6)), extraction_tokenizer=_ExtractionTokenizer([])
    )
    assert candidate is None and reason == "extraction_empty"

    tokenizer = _RawTokenizer({"source": list(range(6))})
    candidate, reason = _build_structural_candidate(
        nanochat_tokenizer=tokenizer,
        extraction_tokenizer=_ExtractionTokenizer(list(range(9))),
        text="source",
        shard_idx=0,
        doc_idx=0,
        candidate_index=0,
        continuation_length=3,
        prefix_cap=3,
        extraction_limit=8,
        ignore_bos_features=False,
    )
    assert candidate is None and reason == "extraction_over_limit"

    candidate, reason, _ = _candidate(
        list(range(6)),
        extraction_tokenizer=_ExtractionTokenizer([2, 2, 2], bos_id=2),
        ignore_bos_features=True,
    )
    assert candidate is None and reason == "extraction_entirely_unpoolable"


def test_streaming_order_assigns_indices_only_to_accepted_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "shard_00000.parquet"
    pq.write_table(
        pa.table({"text": ["short", "first", "second"]}),
        parquet_path,
    )
    tokenizer = _RawTokenizer(
        {
            "short": [1, 2, 3, 4, 5],
            "first": [10, 11, 12, 13, 14, 15],
            "second": [20, 21, 22, 23, 24, 25, 26],
        }
    )
    cfg = make_valid_config(
        validation_shards=[0],
        seed_continuation_nanochat_tokens=3,
        seed_prefix_max_nanochat_tokens=6,
    )
    monkeypatch.setattr(
        seed_pools, "_list_parquets_for_feature_dir", lambda _cfg: [parquet_path]
    )
    candidates, skips, hashes = _stream_candidates(
        cfg=cfg,
        nanochat_tokenizer=tokenizer,
        extraction_tokenizer=_ExtractionTokenizer(),
        extraction_limit=128,
    )
    assert [candidate.candidate_index for candidate in candidates] == [0, 1]
    assert [(candidate.shard_idx, candidate.doc_idx) for candidate in candidates] == [
        (0, 1),
        (0, 2),
    ]
    assert skips["source_too_short"] == 1
    assert sum(skips.values()) == 1
    assert list(hashes) == [str(parquet_path.resolve())]


def test_candidate_digest_is_order_and_skip_sensitive() -> None:
    one, _, _ = _candidate(list(range(6)))
    two, _, _ = _candidate(list(range(7)))
    assert one is not None and two is not None
    skips = dict.fromkeys(seed_pools._SKIP_REASONS, 0)
    source_hashes = {"source": "a" * 64}
    baseline = _candidate_digest([one, two], skips, source_hashes)
    assert baseline == _candidate_digest([one, two], skips, source_hashes)
    assert baseline != _candidate_digest([two, one], skips, source_hashes)
    changed_skips = {**skips, "source_too_short": 1}
    assert baseline != _candidate_digest([one, two], changed_skips, source_hashes)


def test_candidate_shards_and_committed_schema_are_rank_independent(
    tmp_path: Path,
) -> None:
    candidates: list[SeedCandidate] = []
    for index in range(5):
        candidate, reason, _ = _candidate(list(range(6 + index)))
        assert reason is None and candidate is not None
        candidates.append(
            replace(candidate, candidate_index=index)
        )
    shards = _candidate_shards(candidates, shard_size=2)
    assert [len(shard) for shard in shards] == [2, 2, 1]
    paths = _write_or_validate_candidate_shards(
        root=tmp_path,
        candidate_shards=shards,
        candidate_digest="d" * 64,
        skip_counts=dict.fromkeys(seed_pools._SKIP_REASONS, 0),
        source_hashes={"source": "a" * 64},
    )
    assert [path.name for path in paths] == [
        "candidate_00000.parquet",
        "candidate_00001.parquet",
        "candidate_00002.parquet",
    ]
    for path in paths:
        assert pq.read_table(path).schema.equals(seed_pools._CANDIDATE_SCHEMA)
    assert _write_or_validate_candidate_shards(
        root=tmp_path,
        candidate_shards=shards,
        candidate_digest="d" * 64,
        skip_counts=dict.fromkeys(seed_pools._SKIP_REASONS, 0),
        source_hashes={"source": "a" * 64},
    ) == paths


def test_candidate_commit_mismatch_fails_loudly(tmp_path: Path) -> None:
    candidate, reason, _ = _candidate(list(range(6)))
    assert reason is None and candidate is not None
    _write_or_validate_candidate_shards(
        root=tmp_path,
        candidate_shards=[[candidate]],
        candidate_digest="d" * 64,
        skip_counts=dict.fromkeys(seed_pools._SKIP_REASONS, 0),
        source_hashes={"source": "a" * 64},
    )
    changed = replace(candidate, prefix_content_nanochat_tokens=[7, 8, 9])
    with pytest.raises(AxBenchBaseError, match="does not match"):
        _write_or_validate_candidate_shards(
            root=tmp_path,
            candidate_shards=[[changed]],
            candidate_digest="d" * 64,
            skip_counts=dict.fromkeys(seed_pools._SKIP_REASONS, 0),
            source_hashes={"source": "a" * 64},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("text", None, "non-empty string", id="null-text"),
        pytest.param(
            "continuation_text", None, "must be a string", id="null-continuation"
        ),
        pytest.param("candidate_index", True, "non-negative integer", id="bool-index"),
        pytest.param(
            "prefix_content_nanochat_tokens",
            [0, 1, True],
            "integer token ids",
            id="bool-token",
        ),
        pytest.param("prefix_source_token_end", 4, "prefix token length", id="bounds"),
        pytest.param(
            "source_nanochat_token_count",
            7,
            "continuation token length",
            id="continuation-length",
        ),
    ],
)
def test_candidate_reload_rejects_malformed_values_without_coercion(
    field: str,
    value: Any,
    message: str,
) -> None:
    candidate, reason, _ = _candidate(list(range(6)))
    assert reason is None and candidate is not None
    row = candidate.parquet_row()
    row[field] = value

    with pytest.raises(AxBenchBaseError, match=message):
        seed_pools._candidate_from_row(row)


def test_native_backend_hashes_actual_forward_module() -> None:
    forward = t.nn.Linear(2, 2)
    wrapper = SimpleNamespace(native_model=SimpleNamespace(_base=forward))
    resolved, backend = _forward_model(wrapper)
    assert resolved is forward
    assert backend == "native"


def test_native_backend_without_forward_state_fails() -> None:
    wrapper = SimpleNamespace(native_model=SimpleNamespace())
    with pytest.raises(AxBenchBaseError, match="actual forward module"):
        _forward_model(wrapper)


@pytest.mark.parametrize(
    ("activation", "pooling", "suffix"),
    [
        ("residual", "max_pool", ""),
        ("mlp", "mean_pool", "/no_bos"),
        ("attention", "last_token", "/context_independent/no_bos"),
    ],
)
def test_extract_candidate_shard_threads_parsed_production_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
    pooling: str,
    suffix: str,
) -> None:
    activation_label = {"residual": "res", "mlp": "mlp", "attention": "att"}[
        activation
    ]
    feature_dir = (
        f"out/data/features/s128/l8/a_{activation_label}/p_{pooling}/"
        f"f16k/m_gemma-2-2b/base{suffix}"
    )
    cfg = make_valid_config(
        feature_dir=feature_dir,
        extraction_max_tokens=900,
        seed_extraction_gpu_batch_size=7,
        seed_extraction_cpu_batch_size=19,
    )
    nnterp_model = object()
    native_model = object()
    base_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=700),
        nnterp_model=nnterp_model,
        native_model=native_model,
    )
    base_sae = SimpleNamespace(
        cfg=SimpleNamespace(hook_name="resolved.hook.name")
    )
    candidate_path = tmp_path / "candidate_00000.parquet"
    candidate_path.touch()
    output_dir = tmp_path / "features"
    expected_output = feature_path(output_dir, candidate_path.stem)
    calls: dict[str, Any] = {}

    def fake_process_shard(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(output_path=str(expected_output))

    monkeypatch.setattr(seed_pools, "process_shard", fake_process_shard)

    result = _extract_candidate_shard(
        cfg=cfg,
        candidate_path=candidate_path,
        output_dir=output_dir,
        base_model=base_model,
        base_sae=base_sae,
    )

    parsed = parse_feature_dir(feature_dir)
    assert result == expected_output
    assert calls["args"][:3] == (str(candidate_path), base_model, base_sae)
    kwargs = calls["kwargs"]
    assert kwargs["layer"] == 8
    assert kwargs["context_window"] == 700
    assert kwargs["pooling"] is parsed.pooling
    assert kwargs["gpu_batch_size"] == 7
    assert kwargs["cpu_batch_size"] == 19
    assert kwargs["nnterp_model"] is nnterp_model
    assert kwargs["native_model"] is native_model
    assert kwargs["segment_mode"] is SegmentMode.fixed_tokens
    assert kwargs["segment_size"] == 128
    assert kwargs["hook_name"] == "resolved.hook.name"
    assert kwargs["activation_type"] is parsed.activation_type
    assert kwargs["dataset_phase"] == "base"
    assert kwargs["context_independent_features"] is (
        "context_independent" in suffix
    )
    assert kwargs["ignore_bos_features"] is ("no_bos" in suffix)


def test_extract_candidate_shard_rejects_turn_delimited_feature_directory(
    tmp_path: Path,
) -> None:
    cfg = make_valid_config(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/"
            "m_gemma-2-2b/base/turn_delimited"
        )
    )
    with pytest.raises(
        AxBenchBaseError,
        match="AxBench seed-pool extraction supports fixed_tokens feature directories only",
    ):
        _extract_candidate_shard(
            cfg=cfg,
            candidate_path=tmp_path / "candidate.parquet",
            output_dir=tmp_path / "features",
            base_model=object(),
            base_sae=object(),
        )


def test_extract_candidate_shard_reads_saelens_6_metadata_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 0b accepts the cfg layout exposed by the real GemmaScope SAE."""

    cfg = make_valid_config()
    base_model = SimpleNamespace(
        cfg=SimpleNamespace(n_ctx=1024),
        nnterp_model=None,
        native_model=None,
    )
    base_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            metadata={"hook_name": "blocks.8.hook_resid_post"},
        )
    )
    candidate_path = tmp_path / "candidate_00000.parquet"
    candidate_path.touch()
    expected_output = feature_path(tmp_path / "features", candidate_path.stem)
    captured: dict[str, Any] = {}

    def fake_process_shard(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args
        captured.update(kwargs)
        return SimpleNamespace(output_path=str(expected_output))

    monkeypatch.setattr(seed_pools, "process_shard", fake_process_shard)

    _extract_candidate_shard(
        cfg=cfg,
        candidate_path=candidate_path,
        output_dir=tmp_path / "features",
        base_model=base_model,
        base_sae=base_sae,
    )

    assert captured["hook_name"] == "blocks.8.hook_resid_post"


@pytest.mark.parametrize(
    ("pooling", "suffix"),
    [
        ("max_pool", ""),
        ("mean_pool", "/no_bos"),
        ("max_pool", "/context_independent"),
        ("mean_pool", "/context_independent/no_bos"),
    ],
)
@pytest.mark.parametrize(
    ("activation_label", "hook_name"),
    [
        ("res", "blocks.8.hook_resid_post"),
        ("mlp", "blocks.8.hook_mlp_out"),
        ("att", "blocks.8.attn.hook_z"),
    ],
)
def test_stage0b_extraction_matches_direct_production_process_shard(
    tmp_path: Path,
    pooling: str,
    suffix: str,
    activation_label: str,
    hook_name: str,
) -> None:
    t.manual_seed(1234)
    model = TinyHookedTransformer(d_model=16, vocab_size=512, device="cpu")
    sae = TinySAE(d_model=16, n_features=64, device="cpu")
    set_runtime_attribute(sae.cfg, "hook_name", hook_name)
    feature_dir = (
        f"out/data/features/s128/l8/a_{activation_label}/"
        f"p_{pooling}/f16k/m_gemma-2-2b/base{suffix}"
    )
    cfg = make_valid_config(
        feature_dir=feature_dir,
        extraction_max_tokens=512,
        seed_extraction_gpu_batch_size=2,
        seed_extraction_cpu_batch_size=3,
    )
    text = "independent prefix extraction " * 10
    extraction_count = len(model.tokenizer(text, padding=False)["input_ids"])
    candidate = SeedCandidate(
        candidate_index=0,
        text=text,
        shard_idx=0,
        doc_idx=0,
        source_nanochat_token_count=6,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_content_nanochat_tokens=[1, 2, 3],
        continuation_text="continuation",
        continuation_content_nanochat_tokens=[4, 5, 6],
        extraction_token_count=extraction_count,
    )
    candidate_path = tmp_path / "candidate_00000.parquet"
    pq.write_table(
        pa.Table.from_pylist([candidate.parquet_row()], schema=seed_pools._CANDIDATE_SCHEMA),
        candidate_path,
    )
    actual_path = _extract_candidate_shard(
        cfg=cfg,
        candidate_path=candidate_path,
        output_dir=tmp_path / "actual",
        base_model=model,
        base_sae=sae,
    )
    parsed = parse_feature_dir(feature_dir)
    oracle = process_shard(
        str(candidate_path),
        cast(HookedTransformer, model),
        cast(SAE, sae),
        layer=parsed.layer,
        output_dir=str(tmp_path / "oracle"),
        context_window=min(cfg.extraction_max_tokens, model.cfg.n_ctx),
        pooling=parsed.pooling,
        gpu_batch_size=cfg.seed_extraction_gpu_batch_size,
        cpu_batch_size=cfg.seed_extraction_cpu_batch_size,
        nnterp_model=model.nnterp_model,
        native_model=None,
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=parsed.segment_size,
        hook_name=hook_name,
        activation_type=parsed.activation_type,
        dataset_phase=parsed.dataset_phase.value,
        context_independent_features=parsed.context_independent_features,
        ignore_bos_features=parsed.ignore_bos_features,
        run_l0_tier=None,
    )
    actual = load_features(actual_path)
    expected = load_features(oracle.output_path)
    assert set(actual) == set(expected)
    for key in actual:
        if isinstance(actual[key], t.Tensor):
            assert t.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


class _UnicodeSliceTokenizer(_RawTokenizer):
    """Lossless token-to-codepoint tokenizer for the audited non-head slice."""

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _IdentitySAE:
    """Expose each tiny-model one-hot activation as its matching feature ID."""

    def __init__(self, width: int, hook_name: str) -> None:
        self.cfg = SimpleNamespace(d_sae=width, hook_name=hook_name)
        self._parameter = t.nn.Parameter(t.zeros(1))

    def parameters(self):
        yield self._parameter

    def encode(self, activations: t.Tensor) -> t.Tensor:
        return t.relu(activations)


def _feature_id_set(path: Path) -> set[int]:
    return {int(value) for value in load_features(path)["feature_ids"].tolist()}


def test_audited_529_1041_prefix_is_reextracted_not_document_head(
    tmp_path: Path,
) -> None:
    """Pin the exact non-head source slice that exposed Issue #728's feature bug."""

    source_tokens = list(range(17, 1314))
    tokenizer = _UnicodeSliceTokenizer({"source": source_tokens})
    extraction_width = 1400
    model = TinyHookedTransformer(
        d_model=extraction_width,
        vocab_size=extraction_width,
        device="cpu",
    )
    model._embedding = t.eye(extraction_width)
    hook_name = "blocks.8.hook_resid_post"
    sae = _IdentitySAE(extraction_width, hook_name)
    extraction_limit = 1024
    candidate, reason = _build_structural_candidate(
        nanochat_tokenizer=tokenizer,
        extraction_tokenizer=model.tokenizer,
        text="source",
        shard_idx=0,
        doc_idx=0,
        candidate_index=0,
        continuation_length=256,
        prefix_cap=512,
        extraction_limit=extraction_limit,
        ignore_bos_features=True,
    )
    assert reason is None and candidate is not None
    assert candidate.prefix_source_token_start == 529
    assert candidate.prefix_source_token_end == 1041
    assert candidate.prefix_content_nanochat_tokens == source_tokens[529:1041]
    assert candidate.text == tokenizer.decode(source_tokens[529:1041])

    # Text outside the selected source slice cannot affect the candidate that
    # enters process_shard: mutate both the document head and continuation while
    # retaining positions [529:1041], then require the exact same candidate.
    changed_source = [1200] * 529 + source_tokens[529:1041] + [1300] * 256
    changed_tokenizer = _UnicodeSliceTokenizer({"changed": changed_source})
    changed, changed_reason = _build_structural_candidate(
        nanochat_tokenizer=changed_tokenizer,
        extraction_tokenizer=model.tokenizer,
        text="changed",
        shard_idx=0,
        doc_idx=1,
        candidate_index=1,
        continuation_length=256,
        prefix_cap=512,
        extraction_limit=extraction_limit,
        ignore_bos_features=True,
    )
    assert changed_reason is None and changed is not None
    assert changed.prefix_content_nanochat_tokens == (
        candidate.prefix_content_nanochat_tokens
    )
    assert changed.text == candidate.text

    cfg = make_valid_config(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/"
            "m_gemma-2-2b/base/no_bos"
        ),
        extraction_max_tokens=extraction_limit,
        seed_extraction_gpu_batch_size=2,
        seed_extraction_cpu_batch_size=2,
    )
    candidate_path = tmp_path / "candidate_00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [candidate.parquet_row()], schema=seed_pools._CANDIDATE_SCHEMA
        ),
        candidate_path,
    )
    extracted_path = _extract_candidate_shard(
        cfg=cfg,
        candidate_path=candidate_path,
        output_dir=tmp_path / "selected",
        base_model=model,
        base_sae=sae,
    )

    head = SeedCandidate(
        candidate_index=0,
        text=tokenizer.decode(source_tokens[:512]),
        shard_idx=0,
        doc_idx=0,
        source_nanochat_token_count=len(source_tokens),
        prefix_source_token_start=0,
        prefix_source_token_end=512,
        prefix_content_nanochat_tokens=source_tokens[:512],
        continuation_text=tokenizer.decode(source_tokens[512:768]),
        continuation_content_nanochat_tokens=source_tokens[512:768],
        extraction_token_count=513,
    )
    head_path = tmp_path / "head_00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [head.parquet_row()], schema=seed_pools._CANDIDATE_SCHEMA
        ),
        head_path,
    )
    head_extracted_path = _extract_candidate_shard(
        cfg=cfg,
        candidate_path=head_path,
        output_dir=tmp_path / "head",
        base_model=model,
        base_sae=sae,
    )
    selected_features = _feature_id_set(extracted_path)
    head_features = _feature_id_set(head_extracted_path)
    assert selected_features == set(source_tokens[529:1041])
    assert head_features == set(source_tokens[:512])
    assert selected_features.isdisjoint(head_features)


def _write_merge_fixture(
    tmp_path: Path,
) -> tuple[
    list[Path],
    Path,
    SeedPoolExpectation,
    _RawTokenizer,
    str,
    dict[str, Any],
]:
    tokenizer = _RawTokenizer({})
    candidates: list[SeedCandidate] = []
    for index, source in enumerate(([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11])):
        prefix = list(source[:3])
        continuation = list(source[3:])
        candidates.append(
            SeedCandidate(
                candidate_index=index,
                text=tokenizer.decode(prefix),
                shard_idx=0,
                doc_idx=index,
                source_nanochat_token_count=6,
                prefix_source_token_start=0,
                prefix_source_token_end=3,
                prefix_content_nanochat_tokens=prefix,
                continuation_text=tokenizer.decode(continuation),
                continuation_content_nanochat_tokens=continuation,
                extraction_token_count=10,
            )
        )
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    candidate_path = candidate_root / "candidate_00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [candidate.parquet_row() for candidate in candidates],
            schema=seed_pools._CANDIDATE_SCHEMA,
        ),
        candidate_path,
    )
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    output_path = save_features(
        {
            "offsets": t.tensor([0, 2, 4, 6], dtype=t.int64),
            "feature_ids": t.tensor([5, 1, 1, 7, 3, 5], dtype=t.int32),
            "feature_mags": t.tensor(
                [0.2, 0.4, 0.9, 0.3, 0.5, 0.6], dtype=t.float32
            ),
            "doc_ids": t.tensor([0, 0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 5, 0], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 5, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([5, 10, 10], dtype=t.int32),
            "doc_token_counts": t.tensor([10, 10], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 3,
            "layer": 8,
            "context_window": 32,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 5,
        },
        feature_path(feature_root, candidate_path.stem),
    )
    digest, _n_docs, _doc_counts = _canonical_feature_digest(output_path)
    extraction_fingerprint = "e" * 64
    runtime_identity = {"backend": "test"}
    _feature_sidecar_path(output_path).write_text(
        seed_pools.canonical_json(
            {
                "schema_version": seed_pools.SEED_SCHEMA_VERSION,
                "candidate_parquet_sha256": seed_pools.sha256_file(candidate_path),
                "extraction_fingerprint": extraction_fingerprint,
                "document_count": 2,
                "canonical_output_sha256": digest,
                "runtime_identity": runtime_identity,
            }
        ),
        encoding="utf-8",
    )
    expectation = SeedPoolExpectation(
        continuation_length=3,
        prefix_cap=3,
        n_sae_features=16,
        nanochat_tokenizer_checksum=compute_tokenizer_checksum(tokenizer),
        config_fingerprint="c" * 64,
    )
    return (
        [candidate_path],
        feature_root,
        expectation,
        tokenizer,
        extraction_fingerprint,
        runtime_identity,
    )


def test_merge_aggregates_max_per_feature_and_sorts_ids(tmp_path: Path) -> None:
    (
        candidate_paths,
        feature_root,
        expectation,
        tokenizer,
        extraction_fingerprint,
        runtime_identity,
    ) = _write_merge_fixture(tmp_path)

    rows = _merge_candidate_features(
        candidate_paths=candidate_paths,
        feature_root=feature_root,
        expectation=expectation,
        nanochat_tokenizer=tokenizer,
        extraction_fingerprint=extraction_fingerprint,
        runtime_identity=runtime_identity,
    )

    assert [(row.shard_idx, row.doc_idx) for row in rows] == [(0, 0), (0, 1)]
    assert rows[0].sae_feature_ids == [1, 5, 7]
    assert rows[0].sae_feature_mags == pytest.approx([0.9, 0.2, 0.3], abs=1e-3)
    assert rows[1].sae_feature_ids == [3, 5]
    assert rows[1].sae_feature_mags == pytest.approx([0.5, 0.6], abs=1e-3)


@pytest.mark.parametrize(
    ("field", "replacement", "error_match"),
    [
        (
            "doc_ids",
            t.tensor([0, 0, 0], dtype=t.int32),
            "does not cover every candidate",
        ),
        (
            "segment_indices",
            t.tensor([0, 2, 0], dtype=t.int32),
            "missing, duplicate, or reordered segments",
        ),
    ],
)
def test_merge_rejects_missing_document_or_segment_coverage(
    tmp_path: Path,
    field: str,
    replacement: t.Tensor,
    error_match: str,
) -> None:
    (
        candidate_paths,
        feature_root,
        expectation,
        tokenizer,
        extraction_fingerprint,
        runtime_identity,
    ) = _write_merge_fixture(tmp_path)
    output_path = feature_path(feature_root, candidate_paths[0].stem)
    features = load_features(output_path)
    features[field] = replacement
    save_features(features, output_path)
    digest, _n_docs, _doc_counts = _canonical_feature_digest(output_path)
    sidecar_path = _feature_sidecar_path(output_path)
    sidecar = seed_pools.json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["canonical_output_sha256"] = digest
    sidecar_path.write_text(seed_pools.canonical_json(sidecar), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match=error_match):
        _merge_candidate_features(
            candidate_paths=candidate_paths,
            feature_root=feature_root,
            expectation=expectation,
            nanochat_tokenizer=tokenizer,
            extraction_fingerprint=extraction_fingerprint,
            runtime_identity=runtime_identity,
        )


def test_feature_shard_resume_requires_matching_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        candidate_paths,
        feature_root,
        _expectation,
        _tokenizer,
        extraction_fingerprint,
        runtime_identity,
    ) = _write_merge_fixture(tmp_path)
    candidate_path = candidate_paths[0]
    cfg = make_valid_config()
    monkeypatch.setattr(
        seed_pools,
        "_extract_candidate_shard",
        lambda **_kwargs: pytest.fail("validated shard should have resumed"),
    )

    record = _process_or_resume_feature_shard(
        cfg=cfg,
        candidate_path=candidate_path,
        output_dir=feature_root,
        base_model=object(),
        base_sae=object(),
        extraction_fingerprint=extraction_fingerprint,
        runtime_identity=runtime_identity,
    )
    assert record["document_count"] == 2
    assert record["doc_token_counts"] == [10, 10]

    sidecar_path = _feature_sidecar_path(
        feature_path(feature_root, candidate_path.stem)
    )
    sidecar = seed_pools.json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["extraction_fingerprint"] = "f" * 64
    sidecar_path.write_text(seed_pools.json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="sidecar validation"):
        _process_or_resume_feature_shard(
            cfg=cfg,
            candidate_path=candidate_path,
            output_dir=feature_root,
            base_model=object(),
            base_sae=object(),
            extraction_fingerprint=extraction_fingerprint,
            runtime_identity=runtime_identity,
        )


def test_feature_shard_resume_recomputes_body_without_commit_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        candidate_paths,
        feature_root,
        _expectation,
        _tokenizer,
        extraction_fingerprint,
        runtime_identity,
    ) = _write_merge_fixture(tmp_path)
    candidate_path = candidate_paths[0]
    output_path = feature_path(feature_root, candidate_path.stem)
    output_bytes = output_path.read_bytes()
    _feature_sidecar_path(output_path).unlink()
    extraction_calls = 0

    def fake_extract(**_kwargs: Any) -> Path:
        nonlocal extraction_calls
        extraction_calls += 1
        assert not output_path.exists()
        output_path.write_bytes(output_bytes)
        return output_path

    monkeypatch.setattr(seed_pools, "_extract_candidate_shard", fake_extract)

    record = _process_or_resume_feature_shard(
        cfg=make_valid_config(),
        candidate_path=candidate_path,
        output_dir=feature_root,
        base_model=object(),
        base_sae=object(),
        extraction_fingerprint=extraction_fingerprint,
        runtime_identity=runtime_identity,
    )

    assert extraction_calls == 1
    assert record["document_count"] == 2
    assert record["doc_token_counts"] == [10, 10]
    assert _feature_sidecar_path(output_path).is_file()


def test_feature_shard_resume_rejects_sidecar_without_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        candidate_paths,
        feature_root,
        _expectation,
        _tokenizer,
        extraction_fingerprint,
        runtime_identity,
    ) = _write_merge_fixture(tmp_path)
    candidate_path = candidate_paths[0]
    feature_path(feature_root, candidate_path.stem).unlink()
    monkeypatch.setattr(
        seed_pools,
        "_extract_candidate_shard",
        lambda **_kwargs: pytest.fail("invalid committed state must fail loud"),
    )

    with pytest.raises(AxBenchBaseError, match="Incomplete committed"):
        _process_or_resume_feature_shard(
            cfg=make_valid_config(),
            candidate_path=candidate_path,
            output_dir=feature_root,
            base_model=object(),
            base_sae=object(),
            extraction_fingerprint=extraction_fingerprint,
            runtime_identity=runtime_identity,
        )


@pytest.mark.parametrize(
    ("replacement_index", "error_match"),
    [
        (0, "Duplicate Stage 0b candidate index 0"),
        (2, "candidate indices are missing or non-contiguous"),
    ],
)
def test_merge_rejects_duplicate_or_noncontiguous_candidate_indices(
    tmp_path: Path,
    replacement_index: int,
    error_match: str,
) -> None:
    (
        candidate_paths,
        feature_root,
        expectation,
        tokenizer,
        extraction_fingerprint,
        runtime_identity,
    ) = _write_merge_fixture(tmp_path)
    candidate_path = candidate_paths[0]
    table = pq.read_table(candidate_path)
    records = table.to_pylist()
    records[1]["candidate_index"] = replacement_index
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), candidate_path)
    output_path = feature_path(feature_root, candidate_path.stem)
    sidecar_path = _feature_sidecar_path(output_path)
    sidecar = seed_pools.json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["candidate_parquet_sha256"] = seed_pools.sha256_file(candidate_path)
    sidecar_path.write_text(seed_pools.canonical_json(sidecar), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match=error_match):
        _merge_candidate_features(
            candidate_paths=candidate_paths,
            feature_root=feature_root,
            expectation=expectation,
            nanochat_tokenizer=tokenizer,
            extraction_fingerprint=extraction_fingerprint,
            runtime_identity=runtime_identity,
        )


def test_single_rank_wave_owns_every_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_paths = [
        tmp_path / f"candidate_{index:05d}.parquet" for index in range(4)
    ]
    seen: list[int] = []

    def fake_process(**kwargs: Any) -> dict[str, Any]:
        index = candidate_paths.index(kwargs["candidate_path"])
        seen.append(index)
        return {"canonical_output_sha256": f"{index:064x}"}

    monkeypatch.setattr(
        seed_pools,
        "_process_or_resume_feature_shard",
        fake_process,
    )
    seed_pools._collect_feature_shards(
        cfg=make_valid_config(),
        candidate_paths=candidate_paths,
        root=tmp_path,
        rank=0,
        world_size=1,
        base_model=object(),
        base_sae=object(),
        extraction_fingerprint="e" * 64,
        runtime_identity={"backend": "test"},
    )
    assert seen == [0, 1, 2, 3]


def test_two_rank_collective_waves_partition_rank_independent_shards(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(
        2,
        _stage0b_wave_worker,
        str(tmp_path),
        5,
        None,
    )
    assert results == [
        {"seen": [0, 2, 4], "error": None},
        {"seen": [1, 3], "error": None},
    ]


def test_two_rank_wave_failure_is_reported_to_every_peer(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(
        2,
        _stage0b_wave_worker,
        str(tmp_path),
        4,
        1,
    )
    assert results[0]["seen"] == [0]
    assert results[1]["seen"] == []
    assert all(
        "injected rank 1 failure" in result["error"] for result in results
    )


def test_two_rank_identity_disagreement_fails_on_every_rank(
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(2, _stage0b_rank_agreement_worker)
    assert all("ranks disagree on injected identity" in result for result in results)


def test_rank_zero_merge_failure_is_broadcast_to_peers(
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(2, _stage0b_collective_failure_worker, 0)
    assert all("merge failed on rank 0" in result for result in results)


def test_committed_shards_resume_after_world_size_change(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    candidate_paths, feature_root, *_unused = _write_merge_fixture(tmp_path)
    first_candidate = candidate_paths[0]
    second_candidate = first_candidate.with_name("candidate_00001.parquet")
    second_candidate.write_bytes(first_candidate.read_bytes())
    first_feature = feature_path(feature_root, first_candidate.stem)
    second_feature = feature_path(feature_root, second_candidate.stem)
    second_feature.write_bytes(first_feature.read_bytes())
    first_sidecar = _feature_sidecar_path(first_feature)
    second_sidecar = _feature_sidecar_path(second_feature)
    second_sidecar.write_bytes(first_sidecar.read_bytes())

    results = gloo_multiprocess(
        2,
        _stage0b_resume_worker,
        str(tmp_path),
        [first_candidate.name, second_candidate.name],
    )

    assert results == [[first_candidate.name], [second_candidate.name]]
