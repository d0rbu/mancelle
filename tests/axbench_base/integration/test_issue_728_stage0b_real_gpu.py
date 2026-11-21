"""Very-slow real-artifact GPU regression for Issue #728 Stages 0b-4.

Run manually with the real feature/data roots visible::

    AXBENCH_REAL_FEATURE_DIR=/abs/path/to/out/data/features/\
        s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    uv run pytest \
        tests/axbench_base/integration/test_issue_728_stage0b_real_gpu.py \
        -v -m 'very_slow and requires_feature_data'

The test uses the audited shard-369 document 6801, executes production
``process_shard`` twice through the real Gemma-2-2B + GemmaScope SAE pair,
converts those features through the production context-cache path, and runs
real d12 decoder-only and encoder-decoder generation. It deliberately does not
use the legacy feature shard as an oracle: the old artifact describes the
document head, while the corrected contract extracts the exact non-head prefix
independently.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from loguru import logger
from safetensors import safe_open

from nanochat.tokenizer import get_tokenizer
from steering.conditional.feature_pipeline import (
    build_feature_pipeline_from_cfg_block,
)
from steering.eval.axbench_base import seed_pools
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
    validate_config,
)
from steering.eval.axbench_base.generation_helpers import (
    _generate_with_nanochat_tokens,
)
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.seed_pools import (
    SeedCandidate,
    _build_structural_candidate,
    _canonical_feature_digest,
    _extract_candidate_shard,
    _flat_token_ids,
    _list_parquets_for_feature_dir,
    _resolve_parquet,
)
from steering.eval.axbench_base.stage4_prompts import build_stage4_prompt_tokens
from steering.eval.context_cache_build import (
    build_cache_entries_from_extracted_feature_shards,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.target_conditioning import SingleEntryContextProvider
from steering.feature_io import load_features
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.stats_io import load_statistics

_AUDITED_SHARD = 369
_AUDITED_DOC = 6801
_EXPECTED_SOURCE_TOKEN_COUNT = 1297
_EXPECTED_PREFIX_START = 529
_EXPECTED_PREFIX_END = 1041
_EXPECTED_EXTRACTION_TOKEN_COUNT = 498
_EXPECTED_SEGMENT_POSITIONS = [0, 128, 256, 384]
_EXPECTED_SEGMENT_CHAR_STARTS = [0, 515, 1125, 1641]
_EXPECTED_SEGMENT_CHAR_ENDS = [515, 1125, 1641, 2163]
_EXPECTED_CACHE_SEGMENT_CHAR_STARTS = [0, 515, 1125, 1642]
_EXPECTED_CACHE_SEGMENT_CHAR_ENDS = [515, 1125, 1642, 2163]

_LEGACY_ROW_COUNT = 31_990
_LEGACY_SYNTHETIC_BOS_COUNT = 10_817
_LEGACY_RAW_511_ONLY_COUNT = 52
_CORRECTED_CANDIDATE_COUNT = 31_938
_CORRECTED_TRAIN_COUNT = 23_953
_CORRECTED_EVAL_COUNT = 7_985
_CORRECTED_AFFECTED_SURVIVOR_COUNT = 10_765
_CORRECTED_UNAFFECTED_SURVIVOR_COUNT = 21_173
_REPRESENTATIVE_RAW_LENGTH_DOCS = {
    511: 122,
    512: 612,
    513: 115,
    767: 4090,
    768: 1446,
    769: 3258,
    1297: _AUDITED_DOC,
}


def _real_feature_dir() -> Path:
    configured = os.environ.get("AXBENCH_REAL_FEATURE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[3]
        / "out/data/features/s128/l8/a_res/p_max_pool/f16k/"
        "m_gemma-2-2b/base"
    )


def _real_nanochat_base_dir() -> Path:
    configured = os.environ.get("AXBENCH_REAL_NANOCHAT_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    configured = os.environ.get("NANOCHAT_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache/nanochat").resolve()


def _real_legacy_run_dir(nanochat_base: Path) -> Path:
    configured = os.environ.get("AXBENCH_REAL_LEGACY_RUN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return nanochat_base / "eval/axbench_base/issue657-l12-mini"


def _load_legacy_seed_columns(path: Path) -> dict[str, object]:
    """Load only legacy token/text/provenance tensors, never the huge SAE CSR."""

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            "shard_idx": handle.get_tensor("shard_idx"),
            "doc_idx": handle.get_tensor("doc_idx"),
            "prefix_tokens": handle.get_tensor("prefix_tokens"),
            "prefix_token_offsets": handle.get_tensor("prefix_token_offsets"),
            "continuation_tokens": handle.get_tensor("continuation_tokens"),
            "continuation_token_offsets": handle.get_tensor(
                "continuation_token_offsets"
            ),
            "prefix_text": handle.get_tensor("prefix_text_bytes").numpy().tobytes(),
            "prefix_text_offsets": handle.get_tensor("prefix_text_offsets"),
            "continuation_text": (
                handle.get_tensor("continuation_text_bytes").numpy().tobytes()
            ),
            "continuation_text_offsets": handle.get_tensor(
                "continuation_text_offsets"
            ),
        }


def _legacy_tensor_slice(
    columns: dict[str, object], values_name: str, offsets_name: str, row: int,
) -> list[int]:
    values = columns[values_name]
    offsets = columns[offsets_name]
    assert isinstance(values, t.Tensor)
    assert isinstance(offsets, t.Tensor)
    start = int(offsets[row].item())
    end = int(offsets[row + 1].item())
    return [int(value) for value in values[start:end].tolist()]


def _legacy_text_slice(
    columns: dict[str, object], values_name: str, offsets_name: str, row: int,
) -> str:
    values = columns[values_name]
    offsets = columns[offsets_name]
    assert isinstance(values, bytes)
    assert isinstance(offsets, t.Tensor)
    start = int(offsets[row].item())
    end = int(offsets[row + 1].item())
    return values[start:end].decode("utf-8", errors="strict")


def _text_at(path: Path, doc_index: int) -> str:
    parquet = pq.ParquetFile(path)
    remaining = doc_index
    for row_group in range(parquet.num_row_groups):
        texts = parquet.read_row_group(row_group, columns=["text"])["text"]
        if remaining < len(texts):
            value = texts[remaining].as_py()
            if not isinstance(value, str):
                pytest.fail(
                    f"Expected string text at {path} row {doc_index}, "
                    f"got {type(value).__name__}"
                )
            return value
        remaining -= len(texts)
    pytest.fail(f"Document index {doc_index} is out of bounds for {path}")


def _feature_ids_for_doc(features: dict[str, object], doc_id: int) -> set[int]:
    doc_ids = features["doc_ids"]
    offsets = features["offsets"]
    feature_ids = features["feature_ids"]
    assert isinstance(doc_ids, t.Tensor)
    assert isinstance(offsets, t.Tensor)
    assert isinstance(feature_ids, t.Tensor)
    result: set[int] = set()
    for segment_row in t.nonzero(doc_ids == doc_id, as_tuple=False).flatten().tolist():
        start = int(offsets[segment_row].item())
        end = int(offsets[segment_row + 1].item())
        result.update(int(value) for value in feature_ids[start:end].tolist())
    return result


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
def test_issue_728_full_legacy_seed_parity_on_audited_real_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit every legacy seed against its authoritative raw source tokens.

    This deliberately opens the unversioned seed files without the production
    v2 loader: production must reject them, while this regression preserves the
    one-time forensic oracle needed to prove the exact ownership repair.
    """

    nanochat_base = _real_nanochat_base_dir()
    legacy_run = _real_legacy_run_dir(nanochat_base)
    source_path = nanochat_base / f"base_data/shard_{_AUDITED_SHARD:05d}.parquet"
    required = [
        legacy_run / "seeds_train.safetensors",
        legacy_run / "seeds_eval.safetensors",
        source_path,
        nanochat_base / "tokenizer/tokenizer.pkl",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        explicitly_configured = bool(
            os.environ.get("AXBENCH_REAL_LEGACY_RUN_DIR")
            or os.environ.get("AXBENCH_REAL_NANOCHAT_BASE_DIR")
            or os.environ.get("NANOCHAT_BASE_DIR")
        )
        if explicitly_configured:
            pytest.fail(f"Audited Issue #728 artifacts are missing: {missing}")
        pytest.skip(
            "set AXBENCH_REAL_NANOCHAT_BASE_DIR and "
            "AXBENCH_REAL_LEGACY_RUN_DIR to run the full parity audit"
        )

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(nanochat_base))
    tokenizer = get_tokenizer()
    bos_id = int(tokenizer.get_bos_token_id())
    bos_text = tokenizer.decode([bos_id])

    legacy_by_doc: dict[int, tuple[dict[str, object], int]] = {}
    legacy_row_count = 0
    for name in ("seeds_train.safetensors", "seeds_eval.safetensors"):
        columns = _load_legacy_seed_columns(legacy_run / name)
        doc_indices = columns["doc_idx"]
        shard_indices = columns["shard_idx"]
        assert isinstance(doc_indices, t.Tensor)
        assert isinstance(shard_indices, t.Tensor)
        assert doc_indices.shape == shard_indices.shape
        legacy_row_count += int(doc_indices.numel())
        for row in range(int(doc_indices.numel())):
            shard_idx = int(shard_indices[row].item())
            doc_idx = int(doc_indices[row].item())
            assert shard_idx == _AUDITED_SHARD
            assert doc_idx not in legacy_by_doc, (
                f"duplicate legacy provenance ({shard_idx}, {doc_idx})"
            )
            legacy_by_doc[doc_idx] = (columns, row)

    assert legacy_row_count == len(legacy_by_doc) == _LEGACY_ROW_COUNT

    continuation_length = 256
    prefix_cap = 512
    source_doc_count = 0
    corrected_candidate_count = 0
    legacy_synthetic_bos_count = 0
    legacy_text_bos_count = 0
    legacy_raw_511_only_count = 0
    corrected_affected_survivors = 0
    corrected_unaffected_survivors = 0
    audited_legacy_docs: set[int] = set()
    representative_docs: dict[int, int] = {}

    parquet = pq.ParquetFile(source_path)
    for batch in parquet.iter_batches(columns=["text"], batch_size=256):
        for source_text in batch.column(0).to_pylist():
            assert isinstance(source_text, str)
            doc_idx = source_doc_count
            source_doc_count += 1
            source_tokens = [int(token) for token in tokenizer.encode(source_text)]
            source_count = len(source_tokens)
            if source_count in _REPRESENTATIVE_RAW_LENGTH_DOCS:
                representative_docs.setdefault(source_count, doc_idx)
            if source_count >= 2 * continuation_length:
                corrected_candidate_count += 1

            legacy = legacy_by_doc.get(doc_idx)
            if legacy is None:
                continue
            columns, row = legacy
            audited_legacy_docs.add(doc_idx)
            assert source_count >= 2 * continuation_length - 1

            old_prefix_tokens = _legacy_tensor_slice(
                columns, "prefix_tokens", "prefix_token_offsets", row
            )
            old_continuation_tokens = _legacy_tensor_slice(
                columns,
                "continuation_tokens",
                "continuation_token_offsets",
                row,
            )
            old_prefix_text = _legacy_text_slice(
                columns, "prefix_text", "prefix_text_offsets", row
            )
            old_continuation_text = _legacy_text_slice(
                columns,
                "continuation_text",
                "continuation_text_offsets",
                row,
            )

            prefix_end = source_count - continuation_length
            prefix_start = max(0, prefix_end - prefix_cap)
            corrected_prefix_tokens = source_tokens[prefix_start:prefix_end]
            corrected_continuation_tokens = source_tokens[-continuation_length:]
            assert old_continuation_tokens == corrected_continuation_tokens, doc_idx
            assert tokenizer.decode(old_continuation_tokens) == old_continuation_text
            assert tokenizer.decode(old_prefix_tokens) == old_prefix_text

            has_legacy_owned_bos = bool(
                old_prefix_tokens and old_prefix_tokens[0] == bos_id
            )
            if has_legacy_owned_bos:
                legacy_synthetic_bos_count += 1
                assert old_prefix_text.startswith(bos_text), doc_idx
                assert old_prefix_tokens[1:] == corrected_prefix_tokens, doc_idx
                if source_count >= 2 * continuation_length:
                    corrected_affected_survivors += 1
            else:
                assert old_prefix_tokens == corrected_prefix_tokens, doc_idx
                if source_count >= 2 * continuation_length:
                    corrected_unaffected_survivors += 1
            if old_prefix_text.startswith(bos_text):
                legacy_text_bos_count += 1
            if source_count == 2 * continuation_length - 1:
                legacy_raw_511_only_count += 1
                assert has_legacy_owned_bos

    assert source_doc_count == 54_272
    assert audited_legacy_docs == set(legacy_by_doc)
    assert representative_docs == _REPRESENTATIVE_RAW_LENGTH_DOCS
    assert corrected_candidate_count == _CORRECTED_CANDIDATE_COUNT
    assert legacy_synthetic_bos_count == _LEGACY_SYNTHETIC_BOS_COUNT
    assert legacy_text_bos_count == _LEGACY_SYNTHETIC_BOS_COUNT
    assert legacy_raw_511_only_count == _LEGACY_RAW_511_ONLY_COUNT
    assert corrected_affected_survivors == _CORRECTED_AFFECTED_SURVIVOR_COUNT
    assert corrected_unaffected_survivors == _CORRECTED_UNAFFECTED_SURVIVOR_COUNT
    assert corrected_affected_survivors + corrected_unaffected_survivors == (
        corrected_candidate_count
    )
    corrected_train_count = int(corrected_candidate_count * 0.75)
    assert corrected_train_count == _CORRECTED_TRAIN_COUNT
    assert corrected_candidate_count - corrected_train_count == _CORRECTED_EVAL_COUNT


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_issue_728_exact_non_head_prefix_is_repeatable_on_real_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real extraction, cache alignment, and generation preserve ownership."""

    feature_dir = _real_feature_dir()
    statistics_path = feature_dir / "feature_statistics.safetensors"
    if not feature_dir.is_dir() or not statistics_path.is_file():
        if "AXBENCH_REAL_FEATURE_DIR" not in os.environ:
            pytest.skip(
                "set AXBENCH_REAL_FEATURE_DIR to run the real Issue #728 GPU gate"
            )
        pytest.fail(
            "Real Issue #728 GPU gate requires the canonical feature directory "
            f"and statistics file; got feature_dir={feature_dir}. Set "
            "AXBENCH_REAL_FEATURE_DIR to the absolute production path."
        )
    nanochat_base = _real_nanochat_base_dir()
    if not nanochat_base.is_dir():
        if (
            "AXBENCH_REAL_NANOCHAT_BASE_DIR" not in os.environ
            and "NANOCHAT_BASE_DIR" not in os.environ
        ):
            pytest.skip(
                "set AXBENCH_REAL_NANOCHAT_BASE_DIR to run the real Issue #728 GPU gate"
            )
        pytest.fail(
            f"Real nanochat cache is missing at {nanochat_base}; set "
            "AXBENCH_REAL_NANOCHAT_BASE_DIR."
        )
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(nanochat_base))

    cfg = AxBenchBaseConfig(
        feature_dir=str(feature_dir),
        feature_statistics_path=str(statistics_path),
        decoder_only={"source": "base", "model_tag": "d12_base", "step": 54600},
        encdec={"source": "encdec", "model_tag": "d12_baseline", "step": 54600},
        device="cuda",
        output_root=str(tmp_path / "unused-eval-root"),
        run_tag="issue-728-real-gpu",
        validation_shards=[_AUDITED_SHARD],
    )
    validate_config(cfg)

    source_path = _resolve_parquet(
        _list_parquets_for_feature_dir(cfg),
        _AUDITED_SHARD,
    )
    source_text = _text_at(source_path, _AUDITED_DOC)
    state = RuntimeState(cfg=cfg)
    try:
        nanochat_tokenizer = state.nanochat_tokenizer()
        extraction_tokenizer = state.extraction_tokenizer()
        candidate, skip_reason = _build_structural_candidate(
            nanochat_tokenizer=nanochat_tokenizer,
            extraction_tokenizer=extraction_tokenizer,
            text=source_text,
            shard_idx=_AUDITED_SHARD,
            doc_idx=_AUDITED_DOC,
            candidate_index=0,
            continuation_length=cfg.seed_continuation_nanochat_tokens,
            prefix_cap=cfg.seed_prefix_max_nanochat_tokens,
            extraction_limit=cfg.extraction_max_tokens,
            ignore_bos_features=False,
        )
        assert skip_reason is None
        assert candidate is not None
        assert candidate.source_nanochat_token_count == _EXPECTED_SOURCE_TOKEN_COUNT
        assert candidate.prefix_source_token_start == _EXPECTED_PREFIX_START
        assert candidate.prefix_source_token_end == _EXPECTED_PREFIX_END
        assert len(candidate.prefix_content_nanochat_tokens) == 512
        assert len(candidate.continuation_content_nanochat_tokens) == 256
        assert candidate.extraction_token_count == _EXPECTED_EXTRACTION_TOKEN_COUNT
        assert (
            nanochat_tokenizer.decode(candidate.prefix_content_nanochat_tokens)
            == candidate.text
        )
        assert (
            nanochat_tokenizer.decode(candidate.continuation_content_nanochat_tokens)
            == candidate.continuation_text
        )

        extraction_input_ids = _flat_token_ids(
            extraction_tokenizer(candidate.text, padding=False)
        )
        extraction_bos_id = int(extraction_tokenizer.bos_token_id)
        assert extraction_input_ids[0] == extraction_bos_id
        assert extraction_bos_id not in extraction_input_ids[1:]
        extraction_bos_text = extraction_tokenizer.decode([extraction_bos_id])
        extraction_decoded = extraction_tokenizer.decode(extraction_input_ids)
        assert extraction_bos_text == "<bos>"
        assert extraction_decoded == extraction_bos_text + candidate.text

        model_prompt = build_stage4_prompt_tokens(
            tokenizer=nanochat_tokenizer,
            seed_content_tokens=candidate.prefix_content_nanochat_tokens,
        )
        bos_id = int(nanochat_tokenizer.get_bos_token_id())
        assert model_prompt[0] == bos_id
        assert model_prompt[1:] == candidate.prefix_content_nanochat_tokens
        assert model_prompt[1] != bos_id
        assert model_prompt.count(bos_id) == 1
        assert nanochat_tokenizer.decode(model_prompt) == (
            extraction_bos_text + candidate.text
        )

        prompting_prefix = "Related to Irish history:\n\n"
        prompting_prefix_tokens = [
            int(token) for token in nanochat_tokenizer.encode(prompting_prefix)
        ]
        prompting_model_prompt = build_stage4_prompt_tokens(
            tokenizer=nanochat_tokenizer,
            seed_content_tokens=candidate.prefix_content_nanochat_tokens,
            prompting_prefix=prompting_prefix,
        )
        assert prompting_model_prompt == [
            bos_id,
            *prompting_prefix_tokens,
            *candidate.prefix_content_nanochat_tokens,
        ]
        assert prompting_model_prompt.count(bos_id) == 1

        source_tokens = [int(token) for token in nanochat_tokenizer.encode(source_text)]
        head_tokens = source_tokens[:512]
        head_text = nanochat_tokenizer.decode(head_tokens)
        head_extraction_tokens = _flat_token_ids(
            extraction_tokenizer(head_text, padding=False)
        )
        head_candidate = SeedCandidate(
            candidate_index=1,
            text=head_text,
            shard_idx=_AUDITED_SHARD,
            doc_idx=_AUDITED_DOC,
            source_nanochat_token_count=len(source_tokens),
            prefix_source_token_start=0,
            prefix_source_token_end=512,
            prefix_content_nanochat_tokens=head_tokens,
            continuation_text=candidate.continuation_text,
            continuation_content_nanochat_tokens=(
                candidate.continuation_content_nanochat_tokens
            ),
            extraction_token_count=len(head_extraction_tokens),
        )

        candidate_path = tmp_path / "candidate_00000.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [candidate.parquet_row(), head_candidate.parquet_row()],
                schema=seed_pools._CANDIDATE_SCHEMA,
            ),
            candidate_path,
        )

        with state.base_model() as base_model, state.base_sae() as base_sae:
            assert next(base_sae.parameters()).device.type == "cuda"
            first_path = _extract_candidate_shard(
                cfg=cfg,
                candidate_path=candidate_path,
                output_dir=tmp_path / "first",
                base_model=base_model,
                base_sae=base_sae,
            )
            second_path = _extract_candidate_shard(
                cfg=cfg,
                candidate_path=candidate_path,
                output_dir=tmp_path / "second",
                base_model=base_model,
                base_sae=base_sae,
            )

        first_digest, first_n_docs, first_counts = _canonical_feature_digest(
            first_path
        )
        second_digest, second_n_docs, second_counts = _canonical_feature_digest(
            second_path
        )
        assert first_digest == second_digest
        assert first_n_docs == second_n_docs == 2
        assert first_counts == second_counts == [
            candidate.extraction_token_count,
            head_candidate.extraction_token_count,
        ]

        features = load_features(first_path)
        doc_ids = features["doc_ids"]
        segment_indices = features["segment_indices"]
        segment_positions = features["segment_positions"]
        char_starts = features["segment_char_starts"]
        char_ends = features["segment_char_ends"]
        feature_ids = features["feature_ids"]
        feature_mags = features["feature_mags"]
        assert isinstance(doc_ids, t.Tensor)
        selected_rows = t.nonzero(doc_ids == 0, as_tuple=False).flatten()
        assert segment_indices[selected_rows].tolist() == [0, 1, 2, 3]
        assert segment_positions[selected_rows].tolist() == _EXPECTED_SEGMENT_POSITIONS
        assert char_starts[selected_rows].tolist() == _EXPECTED_SEGMENT_CHAR_STARTS
        assert char_ends[selected_rows].tolist() == _EXPECTED_SEGMENT_CHAR_ENDS
        assert int(char_ends[selected_rows[-1]].item()) == len(extraction_decoded)
        assert len(extraction_decoded) - len(candidate.text) == len(
            extraction_bos_text
        )
        assert char_starts[selected_rows][1:].tolist() == char_ends[selected_rows][
            :-1
        ].tolist()
        assert isinstance(feature_ids, t.Tensor)
        assert isinstance(feature_mags, t.Tensor)
        assert feature_ids.numel() == feature_mags.numel() > 0
        assert bool(t.isfinite(feature_mags).all())
        assert bool((feature_mags > 0).all())

        selected_features = _feature_ids_for_doc(features, 0)
        head_features = _feature_ids_for_doc(features, 1)
        assert selected_features - head_features
        assert head_features - selected_features

        head_prompt = build_stage4_prompt_tokens(
            tokenizer=nanochat_tokenizer,
            seed_content_tokens=head_tokens,
        )
        records = [
            EvalContextRecord(
                suite="axbench_base",
                token_ids=tuple(model_prompt),
                context_text=candidate.text,
            ),
            EvalContextRecord(
                suite="axbench_base",
                token_ids=tuple(head_prompt),
                context_text=head_text,
            ),
        ]
        encdec_meta = state.encdec_meta()
        assert encdec_meta.feature_selection is not None
        raw_n_sae_features = encdec_meta.n_sae_features
        raw_num_buckets = encdec_meta.num_buckets
        assert raw_n_sae_features is not None
        assert raw_num_buckets is not None
        n_sae_features = PositiveInt.parse(raw_n_sae_features)
        num_buckets = PositiveInt.parse(raw_num_buckets)
        statistics = load_statistics(statistics_path)
        pipeline_cfg = {
            "feature_selection": encdec_meta.feature_selection,
            "negative_selection": encdec_meta.negative_selection,
            "feature_mask": encdec_meta.feature_mask,
        }

        def _build_entries():
            pipeline = build_feature_pipeline_from_cfg_block(
                pipeline_cfg,
                per_feature_stats=statistics.per_feature_stats,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
            )
            return build_cache_entries_from_extracted_feature_shards(
                context_records=records,
                feature_shard_paths=[first_path],
                pipeline=pipeline,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
                tokenizer=nanochat_tokenizer,
                source_bos_char_len=NonNegativeInt.parse(len(extraction_bos_text)),
            )

        context_entries = _build_entries()
        assert context_entries == _build_entries()
        assert len(context_entries) == 2
        selected_entry = context_entries[0]
        assert selected_entry.token_ids == tuple(model_prompt)
        assert len(selected_entry.segments) == len(_EXPECTED_SEGMENT_POSITIONS)
        assert [segment.char_start for segment in selected_entry.segments] == (
            _EXPECTED_CACHE_SEGMENT_CHAR_STARTS
        )
        assert [segment.char_end for segment in selected_entry.segments] == (
            _EXPECTED_CACHE_SEGMENT_CHAR_ENDS
        )
        assert selected_entry.segments[-1].char_end == len(
            nanochat_tokenizer.decode(model_prompt)
        )
        for segment in selected_entry.segments:
            assert segment.char_end > segment.char_start
            assert segment.feature_ids
            assert len(segment.feature_ids) == len(segment.buckets)
            assert len(set(segment.feature_ids)) == len(segment.feature_ids)
            assert all(
                0 <= feature_id < n_sae_features
                for feature_id in segment.feature_ids
            )
            assert all(
                0 <= bucket < num_buckets
                for bucket in segment.buckets
            )

        generation_cfg = GenerationConfig(
            max_new_tokens=4,
            temperature=0.0,
            top_k_sampling=50,
        )

        def _generate_twice(
            *,
            model,
            prompt_tokens: list[int],
            encoder_input: dict[int, int] | None,
            context_provider: SingleEntryContextProvider | None,
        ):
            kwargs = {
                "model": model,
                "tokenizer": nanochat_tokenizer,
                "device": cfg.device,
                "nanochat_tokens": prompt_tokens,
                "sampling_seed": 1728,
                "gen_cfg": generation_cfg,
                "encoder_input": encoder_input,
                "context_provider": context_provider,
            }
            first = _generate_with_nanochat_tokens(**kwargs)
            second = _generate_with_nanochat_tokens(**kwargs)
            assert first == second
            assert first.token_count > 0
            assert first.stop_reason in {"bos_token", "max_tokens"}
            assert "<bos>" not in first.text
            return first

        with state.decoder_only_model() as decoder_only_model:
            decoder_result = _generate_twice(
                model=decoder_only_model,
                prompt_tokens=model_prompt,
                encoder_input=None,
                context_provider=None,
            )
            prompting_result = _generate_twice(
                model=decoder_only_model,
                prompt_tokens=prompting_model_prompt,
                encoder_input=None,
                context_provider=None,
            )

        provider = SingleEntryContextProvider(
            selected_entry.token_ids,
            selected_entry.segments,
        )
        assert provider.get(model_prompt) == selected_entry.segments
        assert provider.get([*model_prompt, 0]) is None
        final_segment = selected_entry.segments[-1]
        target_feature = {final_segment.feature_ids[0]: final_segment.buckets[0]}
        with state.encdec_model() as encdec_model:
            encdec_control_result = _generate_twice(
                model=encdec_model,
                prompt_tokens=model_prompt,
                encoder_input=None,
                context_provider=provider,
            )
            encdec_target_result = _generate_twice(
                model=encdec_model,
                prompt_tokens=model_prompt,
                encoder_input=target_feature,
                context_provider=provider,
            )

        assert decoder_result.text
        assert prompting_result.text
        assert encdec_control_result.text
        assert encdec_target_result.text
        logger.info(
            "Issue #728 real-GPU diagnostics: extraction_digest={digest}, "
            "raw_segment_ranges={raw_ranges}, cache_segment_ranges={cache_ranges}, "
            "selected_feature_counts={feature_counts}, decoder={decoder!r}, "
            "prompting={prompting!r}, encdec_control={control!r}, "
            "encdec_target={target!r}",
            digest=first_digest,
            raw_ranges=list(
                zip(
                    _EXPECTED_SEGMENT_CHAR_STARTS,
                    _EXPECTED_SEGMENT_CHAR_ENDS,
                    strict=True,
                )
            ),
            cache_ranges=[
                (segment.char_start, segment.char_end)
                for segment in selected_entry.segments
            ],
            feature_counts=[
                len(segment.feature_ids) for segment in selected_entry.segments
            ],
            decoder=decoder_result.text,
            prompting=prompting_result.text,
            control=encdec_control_result.text,
            target=encdec_target_result.text,
        )
    finally:
        state.close()
        t.cuda.empty_cache()
