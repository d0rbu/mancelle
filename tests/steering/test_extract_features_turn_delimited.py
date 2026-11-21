import itertools
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import OmegaConf

import steering.extract_features as ef
from steering.extract_features import (
    _assert_existing_output_dir_compatible,
    _precompute_batch_cpu_work,
    _tokenize_and_segment_chunk,
    _validate_segment_mode_dataset_phase,
    extract_document_features,
    find_optimal_batch_sizes,
    prefetch_shard_data,
    process_shard,
    run_extraction_loop,
)
from steering.feature_io import glob_features, load_features, save_features
from steering.refined_types import NonNegativeInt
from steering.segmentation import (
    SegmentMode,
    segment_turn_delimited,
    token_to_char_offsets,
    turn_token_starts,
)
from steering.sft.mixture_fingerprint import SFT_MIXTURE_FINGERPRINT_METADATA_KEY
from steering.shard_validation import validate_shard_data
from steering.stats_io import save_statistics
from steering.validate_features import validate_shard_against_source
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE
from tests.steering.test_extract_features_loop import _staging, tiny_params
from tests.steering.test_feature_dir_verification import _make_stats

_REPO_ROOT = Path(__file__).resolve().parents[2]

TURN_MODE = SegmentMode.turn_delimited_fixed_tokens


class _StraddlingTinyHookedTransformer(TinyHookedTransformer):
    """Merge ``de`` so one source token crosses the boundary after ``d``."""

    _MERGED_DE = 900

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer.decode = self._decode

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        encoded = [0]
        index = 0
        while index < len(text):
            if text.startswith("de", index):
                encoded.append(self._MERGED_DE)
                index += 2
            else:
                encoded.append(max(1, ord(text[index]) % self.vocab_size))
                index += 1
        return encoded

    def _decode(self, token_ids: list[int]) -> str:
        return "".join(
            "de"
            if token_id == self._MERGED_DE
            else chr(token_id % 128)
            if token_id < 128
            else f"<{token_id}>"
            for token_id in token_ids
        )


def _parquet(
    path: Path,
    texts: list[str],
    starts: list[list[int]] | None,
    *,
    row_group_size: int | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {"text": texts}
    if starts is not None:
        columns["turn_char_starts"] = starts
    pq.write_table(pa.table(columns), path, row_group_size=row_group_size)
    return str(path)


def _run_loop(params, path, tmp_path):
    return run_extraction_loop(
        params,
        [path],
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "staging", 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "profile"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )


def _only_shard(path: Path) -> dict:
    shards = list(glob_features(str(path)))
    assert len(shards) == 1
    return load_features(shards[0])


def _write_resume_shard(
    output_dir: Path,
    stem: str,
    segment_mode: SegmentMode | None,
    fingerprint: str | None = None,
) -> None:
    data = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([1], dtype=t.int32),
        "feature_mags": t.tensor([1.0], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "context_window": 1,
        "pooling": "max_pool",
        "segment_size": 4,
    }
    if segment_mode is not None:
        data["segment_mode"] = segment_mode.value
    if fingerprint is not None:
        data[SFT_MIXTURE_FINGERPRINT_METADATA_KEY] = fingerprint
    save_features(data, output_dir / stem)


def _assert_tensor_payload_equal(left: dict, right: dict) -> None:
    tensor_keys = {key for key, value in left.items() if isinstance(value, t.Tensor)}
    assert tensor_keys == {key for key, value in right.items() if isinstance(value, t.Tensor)}
    for key in tensor_keys:
        assert t.equal(left[key], right[key]), key


def _assert_payload_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    _assert_tensor_payload_equal(left, right)
    for key, value in left.items():
        if not isinstance(value, t.Tensor):
            assert value == right[key], key


def _expected_geometry(model, texts, boundaries, segment_size):
    expected = {
        "segment_positions": [],
        "segment_char_starts": [],
        "segment_char_ends": [],
        "doc_ids": [],
    }
    for doc_id, (text, doc_boundaries) in enumerate(
        zip(texts, boundaries, strict=True)
    ):
        kept_ids = model._encode(text)
        char_offsets = token_to_char_offsets(model.tokenizer, kept_ids)
        starts = turn_token_starts(char_offsets, doc_boundaries, bos_char_len=1)
        spans = segment_turn_delimited(starts, len(kept_ids), segment_size)
        expected["segment_positions"].extend(start for start, _ in spans)
        expected["segment_char_starts"].extend(char_offsets[start] for start, _ in spans)
        expected["segment_char_ends"].extend(char_offsets[end] for _, end in spans)
        expected["doc_ids"].extend([doc_id] * len(spans))
    return expected


def test_end_to_end_turn_extraction_geometry_and_loading(tmp_path):
    texts = ["abcdefghij", "abcdefghijklmnop"]
    boundaries = [[0, 4, 8], [0, 5, 11]]
    parquet = _parquet(tmp_path / "data" / "shard_00000.parquet", texts, boundaries)
    model, sae = TinyHookedTransformer(), TinySAE()
    output = tmp_path / "out"
    _run_loop(
        tiny_params(
            output,
            model=model,
            sae=sae,
            segment_mode=TURN_MODE,
            segment_size=4,
            dataset_phase="standard_sft",
        ),
        parquet,
        tmp_path,
    )
    shard = _only_shard(output)

    expected_positions = [0, 4, 5, 9, 0, 4, 6, 10, 12, 16]
    expected_char_starts = [0, 4, 5, 9, 0, 4, 6, 10, 12, 16]
    expected_char_ends = [4, 5, 9, 11, 4, 6, 10, 12, 16, 17]

    assert shard["segment_positions"].tolist() == expected_positions
    assert shard["segment_char_starts"].tolist() == expected_char_starts
    assert shard["segment_char_ends"].tolist() == expected_char_ends
    assert shard["segment_mode"] == TURN_MODE.value
    result = validate_shard_data(shard)
    assert result.valid, result.violations
    assert load_features(list(glob_features(str(output)))[0])["n_segments"] == len(
        expected_positions
    )


def test_straddling_source_token_stays_in_previous_turn(tmp_path: Path) -> None:
    text = "abcdefghij"
    boundaries = [0, 4, 8]
    parquet = _parquet(tmp_path / "straddle.parquet", [text], [boundaries])
    model, sae = _StraddlingTinyHookedTransformer(), TinySAE()
    output = tmp_path / "out"

    _run_loop(
        tiny_params(
            output,
            model=model,
            sae=sae,
            segment_mode=TURN_MODE,
            segment_size=4,
            dataset_phase="standard_sft",
        ),
        parquet,
        tmp_path,
    )
    shard = _only_shard(output)
    char_offsets = token_to_char_offsets(model.tokenizer, model._encode(text))
    straddled_boundary = boundaries[1] + 1

    assert straddled_boundary not in char_offsets
    assert shard["segment_positions"].tolist() == [0, 4, 5, 8]
    assert shard["segment_char_starts"].tolist() == [0, 4, 6, 9]
    assert shard["segment_char_ends"].tolist() == [4, 6, 9, 11]
    result = validate_shard_data(shard)
    assert result.valid, result.violations


def test_multi_chunk_multi_row_group_turn_geometry(tmp_path):
    texts = [
        "abcdefghijkl",
        "bcdefghijklm",
        "cdefghijklmn",
        "defghijklmno",
        "efghijklmnop",
        "fghijklmnopq",
    ]
    boundaries = [[0, 3], [0, 4], [0, 5], [0, 6], [0, 3, 7], [0, 4, 8]]
    parquet = _parquet(
        tmp_path / "multi.parquet",
        texts,
        boundaries,
        row_group_size=4,
    )
    model, sae = TinyHookedTransformer(), TinySAE()
    common = {
        "model": model,
        "sae": sae,
        "segment_mode": TURN_MODE,
        "segment_size": 4,
        "dataset_phase": "standard_sft",
        "cpu_batch_size": 2,
    }
    prefetched = prefetch_shard_data(
        parquet,
        2,
        dataset_phase="standard_sft",
        segment_mode=TURN_MODE,
    )
    shards = []
    for name, pipeline_post, prefetched_data in (
        ("inline", False, None),
        ("post", True, None),
        ("prefetched", True, prefetched),
    ):
        out = tmp_path / name
        process_shard(
            parquet,
            output_dir=str(out),
            pipeline_post=pipeline_post,
            prefetched=prefetched_data,
            **common,
        )
        shard = _only_shard(out)
        expected = _expected_geometry(model, texts, boundaries, 4)
        for key, value in expected.items():
            assert shard[key].tolist() == value, key
        shards.append(shard)
    for other in shards[1:]:
        _assert_payload_equal(shards[0], other)


@pytest.mark.parametrize("cpu_batch_size", [1, 3, 64])
@pytest.mark.parametrize("pipeline_post", [False, True])
@pytest.mark.parametrize("use_prefetch", [False, True])
def test_turn_boundaries_stay_aligned_across_row_groups_and_cpu_chunks(
    tmp_path,
    cpu_batch_size,
    pipeline_post,
    use_prefetch,
):
    texts = ["abcdefghi" + chr(106 + i) * i for i in range(6)]
    boundaries = [[0, 2 + i, 6 + i] for i in range(4)] + [[0, 3], [0, 5]]
    grouped = _parquet(
        tmp_path / "grouped.parquet",
        texts,
        boundaries,
        row_group_size=3,
    )
    reference = _parquet(tmp_path / "reference.parquet", texts, boundaries)
    model, sae = TinyHookedTransformer(), TinySAE()
    reference_out = tmp_path / "reference"
    process_shard(
        reference,
        model,
        sae,
        output_dir=str(reference_out),
        segment_mode=TURN_MODE,
        segment_size=4,
        dataset_phase="standard_sft",
        cpu_batch_size=64,
        pipeline_post=False,
    )
    prefetched = (
        prefetch_shard_data(
            grouped,
            cpu_batch_size,
            dataset_phase="standard_sft",
            segment_mode=TURN_MODE,
        )
        if use_prefetch
        else None
    )
    output = tmp_path / "actual"
    process_shard(
        grouped,
        model,
        sae,
        output_dir=str(output),
        segment_mode=TURN_MODE,
        segment_size=4,
        dataset_phase="standard_sft",
        cpu_batch_size=cpu_batch_size,
        pipeline_post=pipeline_post,
        prefetched=prefetched,
    )
    actual = _only_shard(output)
    expected = _only_shard(reference_out)
    for key in (
        "segment_positions",
        "segment_char_starts",
        "segment_char_ends",
        "segment_indices",
    ):
        assert t.equal(actual[key], expected[key]), key
    result = validate_shard_against_source(
        actual,
        grouped,
        model.tokenizer,
        context_window=1024,
        segment_size=4,
    )
    assert result.valid, result.violations


def test_staged_input_prefetch_turn_mode_matches_direct(tmp_path):
    texts = ["abcdefghij", "abcdefghijklmnop"]
    boundaries = [[0, 4, 8], [0, 5, 11]]
    paths = [
        _parquet(
            tmp_path / "data" / f"shard_{i:05d}.parquet",
            texts,
            boundaries,
        )
        for i in range(2)
    ]
    model, sae = TinyHookedTransformer(), TinySAE()
    outputs = []
    for input_depth in (0, 1):
        output = tmp_path / f"out{input_depth}"
        run_extraction_loop(
            tiny_params(
                output,
                model=model,
                sae=sae,
                segment_mode=TURN_MODE,
                segment_size=4,
                dataset_phase="standard_sft",
            ),
            paths,
            rank=0,
            world_size=1,
            local_rank=0,
            shard_start=NonNegativeInt.parse(0),
            shard_end=NonNegativeInt.parse(2),
            staging_cfg=_staging(tmp_path / f"stg{input_depth}", 0, input_depth),
            profile_enabled=False,
            profile_output=str(tmp_path / f"p{input_depth}"),
            flops_per_token=1.0,
            gpu_peak_flops=1.0,
        )
        outputs.append([load_features(path) for path in glob_features(output)])
    assert len(outputs[0]) == len(outputs[1]) == 2
    for direct, staged in zip(*outputs, strict=True):
        _assert_payload_equal(direct, staged)


@pytest.mark.parametrize("pooling", [ef.PoolingMode.max_pool, ef.PoolingMode.mean_pool])
def test_single_turn_tensors_match_fixed_mode(tmp_path, pooling):
    text = "abcdefghijklmnop"
    parquet = _parquet(tmp_path / "one.parquet", [text], [[0]])
    model, sae = TinyHookedTransformer(), TinySAE()
    fixed_out, turn_out = tmp_path / "fixed", tmp_path / "turn"
    process_shard(
        parquet,
        model,
        sae,
        output_dir=str(fixed_out),
        segment_size=4,
        segment_mode=SegmentMode.fixed_tokens,
        pooling=pooling,
    )
    process_shard(
        parquet,
        model,
        sae,
        output_dir=str(turn_out),
        segment_size=4,
        segment_mode=TURN_MODE,
        dataset_phase="standard_sft",
        pooling=pooling,
    )
    fixed, turn = _only_shard(fixed_out), _only_shard(turn_out)
    _assert_tensor_payload_equal(fixed, turn)
    assert fixed["segment_mode"] == "fixed_tokens"
    assert turn["segment_mode"] == TURN_MODE.value


def test_boundary_beyond_head_truncation_is_dropped():
    model = TinyHookedTransformer()
    token_info, precomputed = _tokenize_and_segment_chunk(
        model.tokenizer,
        ["abcdefghijklmnop"],
        context_window=7,
        segment_mode=TURN_MODE,
        segment_size=4,
        sae_max_context=7,
        turn_char_starts=[[0, 3, 12]],
    )
    assert len(token_info[0][0]) == 7
    assert precomputed[0][0] == [(0, 4), (4, 7)]


class _WideTokenTokenizer:
    bos_token_id = 0

    def decode(self, ids):
        return "".join("<bos>" if token == 0 else "abc" for token in ids)


def test_turn_that_receives_zero_tokens_raises():
    tokenizer = _WideTokenTokenizer()
    with pytest.raises(
        ValueError,
        match="turn 2 received zero tokens.*document 'abcabc'",
    ):
        _precompute_batch_cpu_work(
            [([0, 1, 2], 3)],
            tokenizer,
            TURN_MODE,
            4,
            texts=["abcabc"],
            turn_char_starts=[[0, 1, 2]],
        )


def test_missing_turn_column_and_base_phase_fail(tmp_path):
    parquet = _parquet(tmp_path / "missing.parquet", ["abc"], None)
    with pytest.raises(ValueError, match="re-run scripts/preprocess/convert_sft_to_parquet"):
        prefetch_shard_data(parquet, 2, dataset_phase="standard_sft", segment_mode=TURN_MODE)
    with pytest.raises(ValueError, match="requires an SFT dataset_phase"):
        _validate_segment_mode_dataset_phase(TURN_MODE, "base")


def test_context_independent_preserves_turn_spans():
    model, sae = TinyHookedTransformer(), TinySAE()
    kwargs = {
        "model": model,
        "sae": sae,
        "text": "abcdefghijklmnop",
        "segment_mode": TURN_MODE,
        "segment_size": 4,
        "turn_char_starts": [0, 5, 11],
    }
    dependent = extract_document_features(**kwargs)
    independent = extract_document_features(**kwargs, context_independent_features=True)
    assert dependent[2:5] == independent[2:5]


def test_pipeline_prefetch_and_post_modes_produce_identical_shards(tmp_path):
    texts, boundaries = ["abcdefghij", "klmnopqrst"], [[0, 4], [0, 6]]
    parquet = _parquet(tmp_path / "data.parquet", texts, boundaries)
    model, sae = TinyHookedTransformer(), TinySAE()
    prefetched = prefetch_shard_data(
        parquet,
        64,
        dataset_phase="standard_sft",
        segment_mode=TURN_MODE,
    )
    shards = []
    for pipeline_post, use_prefetch in itertools.product((False, True), repeat=2):
        output = tmp_path / f"out_{pipeline_post}_{use_prefetch}"
        process_shard(
            parquet,
            model,
            sae,
            output_dir=str(output),
            segment_mode=TURN_MODE,
            segment_size=4,
            dataset_phase="standard_sft",
            pipeline_post=pipeline_post,
            prefetched=prefetched if use_prefetch else None,
        )
        shards.append(_only_shard(output))
    for other in shards[1:]:
        _assert_payload_equal(shards[0], other)


class _TailDriftTokenizer:
    bos_token_id = 0

    def __init__(self, text, *, mismatch):
        self.text = text
        self.mismatch = mismatch

    def decode(self, ids):
        decoded = "<bos>" + self.text[: max(0, len(ids) - 1)]
        if len(ids) == len(self.text) + 1:
            index = 5 + self.mismatch
            decoded = decoded[:index] + "!" + decoded[index + 1 :]
        return decoded


class _DigramMergeTransformer(TinyHookedTransformer):
    """Tiny model whose ``><`` token can straddle a recorded turn boundary."""

    _DIGRAM_ID = 900

    def __init__(self):
        super().__init__()
        self.tokenizer.encode = self._encode
        self.tokenizer.decode = self._decode
        self.tokenizer.side_effect = self._tokenizer_call

    def _encode(self, text):
        ids = [0]
        index = 0
        while index < len(text):
            if text[index : index + 2] == "><":
                ids.append(self._DIGRAM_ID)
                index += 2
            else:
                ids.append(max(1, ord(text[index]) % self.vocab_size))
                index += 1
        return ids

    def _decode(self, ids):
        return "".join(
            "<bos>"
            if token == 0
            else "><"
            if token == self._DIGRAM_ID
            else chr(token % 128)
            for token in ids
        )

    def _tokenizer_call(self, texts_or_text, **_kwargs):
        if isinstance(texts_or_text, str):
            return {"input_ids": self._encode(texts_or_text)}
        return {"input_ids": [self._encode(text) for text in texts_or_text]}


def test_tail_roundtrip_drift_records_counter(monkeypatch):
    text = "abcdefghij"
    tokenizer = _TailDriftTokenizer(text, mismatch=7)
    calls = []
    monkeypatch.setattr(ef, "record_counter", lambda *args, **kwargs: calls.append((args, kwargs)))
    _precompute_batch_cpu_work(
        [([0, *range(1, len(text) + 1)], len(text) + 1)],
        tokenizer,
        TURN_MODE,
        4,
        texts=[text],
        turn_char_starts=[[0, 4]],
    )
    assert calls == [
        (("segmentation.turn_boundary.tail_drift", 1), {"delta": 3, "direction": "tail"})
    ]


def test_roundtrip_drift_before_boundary_names_document():
    text = "named-document"
    tokenizer = _TailDriftTokenizer(text, mismatch=2)
    with pytest.raises(ValueError, match="named-document.*character index 7"):
        _precompute_batch_cpu_work(
            [([0, *range(1, len(text) + 1)], len(text) + 1)],
            tokenizer,
            TURN_MODE,
            8,
            texts=[text],
            turn_char_starts=[[0, 4]],
        )


def test_far_from_tail_drift_after_boundary_rejects():
    text = "abcdefghij" * 4
    tokenizer = _TailDriftTokenizer(text, mismatch=10)
    with pytest.raises(ValueError, match="tail_delta=30, window=20"):
        _precompute_batch_cpu_work(
            [([0, *range(1, len(text) + 1)], len(text) + 1)],
            tokenizer,
            TURN_MODE,
            8,
            texts=[text],
            turn_char_starts=[[0, 4]],
        )


def test_turn_boundary_mapping_handles_straddling_digram_and_exact_boundary():
    model = _DigramMergeTransformer()
    text = "a><bc"
    ids = model._encode(text)
    result = _precompute_batch_cpu_work(
        [(ids, len(ids))],
        model.tokenizer,
        TURN_MODE,
        8,
        texts=[text],
        turn_char_starts=[[0, 2, 4, 5]],
    )
    assert result[0][0] == [(0, 3), (3, 4), (4, 5)]


def test_auto_tuner_always_probes_fixed_windows(monkeypatch, tmp_path):
    """The text-only tuning parquet can only drive fixed-token extraction."""
    sae = TinySAE()
    model = TinyHookedTransformer()
    seen = []
    monkeypatch.setattr(ef, "_get_sae_activations_batched", lambda *args, **kwargs: [t.zeros(1)])
    monkeypatch.setattr(
        ef,
        "process_shard",
        lambda *args, **kwargs: seen.append(kwargs["segment_mode"]),
    )
    assert find_optimal_batch_sizes(
        model,
        sae,
        layer=1,
        context_window=8,
        min_gpu_batch_size=1,
        max_gpu_batch_size=1,
        min_cpu_batch_size=1,
        max_cpu_batch_size=1,
        n_trials=1,
        n_docs=1,
        segment_size=4,
        parquet_paths=[_parquet(tmp_path / "source.parquet", ["abc"], [[0]])],
    ) == (1, 1)
    assert seen == [SegmentMode.fixed_tokens]


def test_main_reports_turn_auto_tuning_memory_caveat(
    monkeypatch,
    tmp_path,
):
    cfg = OmegaConf.load(_REPO_ROOT / "configs" / "extraction" / "extract_features.yaml")
    cfg.output_dir = str(tmp_path / "out")
    cfg.data_dir = str(tmp_path / "data")
    cfg.dataset_phase = "standard_sft"
    cfg.segment_mode = TURN_MODE.value
    cfg.segment_size = 4
    cfg.gpu_batch_size = "auto"
    cfg.cpu_batch_size = "auto"
    cfg.device = "cpu"
    cfg.model_dtype = "float32"
    model, sae = TinyHookedTransformer(), TinySAE()
    resolved = Mock(hook_name="blocks.1.hook_resid_pre", release="test", sae_id="test")
    messages = []

    monkeypatch.setattr(ef.logger, "info", lambda message: messages.append(str(message)))
    monkeypatch.setattr(ef, "_assert_context_window_within_sae_max", lambda *_args: None)
    monkeypatch.setattr(ef, "resolve_data_dir", lambda *_args: tmp_path / "data")
    monkeypatch.setattr(ef, "check_sft_data_ready", lambda *_args: (True, "OK"))
    monkeypatch.setattr(ef, "_load_sft_conversion_fingerprint_json", lambda *_args: None)
    monkeypatch.setattr(ef, "list_parquet_files", lambda *_args: ["shard_00000.parquet"])
    monkeypatch.setattr(ef, "_resolve_extraction_sae", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(ef, "load_sae_and_model", lambda *_args, **_kwargs: (model, sae, None))
    monkeypatch.setattr(ef, "l0_tier_for_resolved", lambda *_args: None)
    monkeypatch.setattr(ef, "assert_l0_tier_compatible", lambda *_args: None)
    monkeypatch.setattr(ef, "_build_accelerated_backend", lambda *_args: (None, None))
    monkeypatch.setattr(ef, "_run_compile_gate", lambda *_args: None)
    monkeypatch.setattr(ef, "_run_rowpack_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ef, "estimate_extraction_flops_per_token", lambda *_args: 1.0)
    monkeypatch.setattr(ef, "find_optimal_batch_sizes", lambda *_args, **_kwargs: (1, 1))
    monkeypatch.setattr(ef, "run_extraction_loop", lambda *_args, **_kwargs: [])

    ef.main.__wrapped__(cfg)

    # #1619 vectorized turn pooling, so the throughput caveat is gone; what
    # survives is the memory caveat, which applies to low-precision mean_pool
    # only (the probe's fixed windows match max_pool's peak exactly).
    assert any(
        "turn pooling now uses the same segmented reduction" in message
        and "float32" in message
        for message in messages
    ), messages
    assert not any("upper bound" in message for message in messages), messages


def test_fixed_mode_rejects_turn_boundaries():
    model = TinyHookedTransformer()
    with pytest.raises(ValueError, match="fixed_tokens.*turn_char_starts"):
        _precompute_batch_cpu_work(
            [(model._encode("abc"), 4)],
            model.tokenizer,
            SegmentMode.fixed_tokens,
            4,
            turn_char_starts=[[0]],
        )


def test_fixed_mode_rejects_texts():
    model = TinyHookedTransformer()
    with pytest.raises(ValueError, match="fixed_tokens.*texts"):
        _precompute_batch_cpu_work(
            [(model._encode("abc"), 4)],
            model.tokenizer,
            SegmentMode.fixed_tokens,
            4,
            texts=["abc"],
        )


def test_turn_mode_precondition_failures():
    model = TinyHookedTransformer()
    ids = model._encode("abc")
    with pytest.raises(ValueError, match="requires texts and turn_char_starts"):
        _precompute_batch_cpu_work(
            [(ids, 4)], model.tokenizer, TURN_MODE, 4, texts=["abc"]
        )
    with pytest.raises(
        ValueError,
        match=r"turn-delimited batch/text/boundary length mismatch: 1, 2, 1",
    ):
        _precompute_batch_cpu_work(
            [(ids, 4)],
            model.tokenizer,
            TURN_MODE,
            4,
            texts=["abc", "def"],
            turn_char_starts=[[0]],
        )
    with pytest.raises(
        ValueError,
        match="assumes a leading BOS token.*expected 0, got 7",
    ):
        _precompute_batch_cpu_work(
            [([7, *ids[1:]], 4)],
            model.tokenizer,
            TURN_MODE,
            4,
            texts=["abc"],
            turn_char_starts=[[0]],
        )
    with pytest.raises(ValueError, match="requires texts and turn_char_starts"):
        extract_document_features(
            model,
            TinySAE(),
            text="abcdefgh",
            segment_mode=TURN_MODE,
            segment_size=4,
        )


@pytest.mark.parametrize("pipeline_post", [False, True])
@pytest.mark.parametrize("requested_mode", [SegmentMode.fixed_tokens, TURN_MODE])
def test_prefetched_shard_mode_must_match_process_shard_mode(
    tmp_path,
    pipeline_post,
    requested_mode,
):
    texts, boundaries = ["abcdefghij", "klmnopqrst"], [[0, 4], [0, 6]]
    parquet = _parquet(tmp_path / "data.parquet", texts, boundaries)
    prefetched_mode = (
        TURN_MODE if requested_mode is SegmentMode.fixed_tokens else SegmentMode.fixed_tokens
    )
    prefetched = prefetch_shard_data(
        parquet,
        64,
        dataset_phase="standard_sft" if prefetched_mode is TURN_MODE else None,
        segment_mode=prefetched_mode,
    )
    match = (
        "fixed_tokens segmentation does not accept turn_char_starts"
        if requested_mode is SegmentMode.fixed_tokens
        else "requires texts and turn_char_starts"
    )
    with pytest.raises(ValueError, match=match):
        process_shard(
            parquet,
            TinyHookedTransformer(),
            TinySAE(),
            output_dir=str(tmp_path / "out"),
            segment_mode=requested_mode,
            segment_size=4,
            dataset_phase=(
                "standard_sft" if requested_mode is TURN_MODE else None
            ),
            pipeline_post=pipeline_post,
            prefetched=prefetched,
        )


def test_resume_rejects_all_existing_fixed_shards_for_turn_request(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(output_dir, "shard_00000_features", SegmentMode.fixed_tokens)
    _write_resume_shard(output_dir, "shard_00001_features", SegmentMode.fixed_tokens)
    with pytest.raises(ValueError, match="fixed_tokens.*turn_delimited_fixed_tokens"):
        _assert_existing_output_dir_compatible(output_dir, TURN_MODE, None)


def test_partial_resume_mode_mismatch_raises_before_extraction(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(output_dir, "shard_00000_features", SegmentMode.fixed_tokens)
    data_dir = tmp_path / "data"
    _parquet(data_dir / "shard_00000.parquet", ["abc"], [[0]])
    _parquet(data_dir / "shard_00001.parquet", ["def"], [[0]])

    cfg = OmegaConf.load(_REPO_ROOT / "configs" / "extraction" / "extract_features.yaml")
    cfg.output_dir = str(output_dir)
    cfg.data_dir = str(data_dir)
    cfg.dataset_phase = "standard_sft"
    cfg.segment_mode = TURN_MODE.value
    extraction_started = Mock(side_effect=AssertionError("extraction started"))
    monkeypatch.setattr(ef, "_assert_context_window_within_sae_max", lambda *_args: None)
    monkeypatch.setattr(ef, "check_sft_data_ready", lambda *_args: (True, "OK"))
    monkeypatch.setattr(ef, "_resolve_extraction_sae", extraction_started)

    with pytest.raises(ValueError, match="fixed_tokens.*turn_delimited_fixed_tokens"):
        ef.main.__wrapped__(cfg)
    extraction_started.assert_not_called()


def test_resume_accepts_legacy_shard_for_fixed_request(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(output_dir, "shard_00000_features", None)
    _assert_existing_output_dir_compatible(output_dir, SegmentMode.fixed_tokens, None)


def test_resume_accepts_turn_shards_for_turn_request(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(output_dir, "shard_00000_features", TURN_MODE)
    _assert_existing_output_dir_compatible(output_dir, TURN_MODE, None)


def test_resume_rejects_statistics_mode_mismatch(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(output_dir, "shard_00000_features", TURN_MODE)
    save_statistics(
        _make_stats(segment_mode=SegmentMode.fixed_tokens.value),
        output_dir / "feature_statistics",
    )
    with pytest.raises(ValueError, match="statistics.*fixed_tokens.*turn_delimited"):
        _assert_existing_output_dir_compatible(output_dir, TURN_MODE, None)


def test_resume_mode_mismatch_precedes_earlier_fingerprint_mismatch(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(
        output_dir,
        "shard_00000_features",
        TURN_MODE,
        fingerprint="wrong",
    )
    _write_resume_shard(
        output_dir,
        "shard_00001_features",
        SegmentMode.fixed_tokens,
        fingerprint="wanted",
    )
    with pytest.raises(ValueError, match="segment_mode"):
        _assert_existing_output_dir_compatible(output_dir, TURN_MODE, "wanted")


def test_resume_mode_mismatch_precedes_corrupt_later_header(tmp_path):
    """Headers are read in shard order and checked as they are read."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_resume_shard(output_dir, "shard_00000_features", SegmentMode.fixed_tokens)
    _write_resume_shard(output_dir, "shard_00001_features", TURN_MODE)
    corrupt = next(path for path in output_dir.iterdir() if "shard_00001" in path.name)
    corrupt.write_bytes(b"not a feature shard")
    with pytest.raises(
        ValueError,
        match=r"shard_00000.*recorded segment_mode='fixed_tokens', requested "
        r"segment_mode='turn_delimited_fixed_tokens'",
    ):
        _assert_existing_output_dir_compatible(output_dir, TURN_MODE, None)


def test_variable_segment_max_pool_rejects_negative_activations():
    with pytest.raises(AssertionError, match="max_pool assumes non-negative"):
        ef.pool_segments_batched_gpu(
            t.tensor([[-1.0], [-2.0]]),
            [(0, 1), (1, 2)],
            ef.PoolingMode.max_pool,
            segment_mode=TURN_MODE,
            keep_mask=t.tensor([True, True]),
        )
