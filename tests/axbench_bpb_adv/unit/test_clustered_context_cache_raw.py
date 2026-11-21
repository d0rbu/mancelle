"""Contracts for the raw phase of the clustered BPB-Adv context cache."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from safetensors.torch import save_file

import steering.eval.axbench_bpb_adv.clustered_context_cache as cache_module
from steering._common.strict_json import canonical_json
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.segmentation import CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS, SegmentMode

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_PACKED_SELECTION_ALGORITHM = "packed-selection-v1"
_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)


class _SourceBosTokenizer:
    """Minimal extraction tokenizer with Gemma's visible five-char BOS."""

    bos_token_id = 0

    def __call__(self, text: str, *, padding: bool = False) -> dict[str, list[int]]:
        del padding
        return {"input_ids": [self.bos_token_id, *(ord(char) for char in text)]}

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            "<bos>" if int(token_id) == self.bos_token_id else chr(int(token_id))
            for token_id in token_ids
        )


def _source_model() -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=_SourceBosTokenizer(),
        cfg=SimpleNamespace(n_ctx=4096),
    )


def _contract(
    *,
    n_sae_features: int = 8,
    segment_size: int = 4,
    num_buckets: int = 2,
    pooling: str = "max_pool",
    packed_selection_algorithm: object = _PACKED_SELECTION_ALGORITHM,
) -> Any:
    algorithms = {"packed_percentile_selection": packed_selection_algorithm}
    return SimpleNamespace(
        n_sae_features=n_sae_features,
        feature_contract_fingerprint=_SHA_A,
        checkpoint=SimpleNamespace(
            model_file=SimpleNamespace(sha256=_SHA_B),
        ),
        statistics=SimpleNamespace(
            file_sha256=_SHA_C,
            percentile_table_checksum=_SHA_D,
        ),
        selection=SimpleNamespace(
            selection_hash=_SHA_A,
            selection_manifest_sha256=_SHA_B,
            selection_shards_sha256=_SHA_C,
            packed_row_fingerprint=_SHA_D,
            preselection_seed=19,
            positive_selection_json=json.dumps(
                {
                    "threshold": 0.5,
                    "threshold_type": "feature_percentile",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            negative_selection_json=json.dumps(
                {
                    "distribution": "normal",
                    "max_k": 2,
                    "mean": 1.0,
                    "min_k": 1,
                    "std": 0.01,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            num_buckets=num_buckets,
            effective_positive_buckets=1,
        ),
        sae=SimpleNamespace(
            layer=8,
            hook_name="blocks.8.hook_resid_post",
        ),
        feature_directory=SimpleNamespace(
            segment_size=segment_size,
            segment_mode=SegmentMode.fixed_tokens.value,
            pooling=pooling,
            context_independent_features=False,
            ignore_bos_features=False,
        ),
        semantic_dict=lambda: {"algorithms": algorithms},
    )


def _sae(
    contract: Any,
    *,
    d_sae: object | None = None,
    hook_name: object | None = None,
    decoder: object | None = None,
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        d_sae=contract.n_sae_features if d_sae is None else d_sae,
        hook_name=contract.sae.hook_name if hook_name is None else hook_name,
    )
    return SimpleNamespace(
        cfg=cfg,
        W_dec=(t.zeros((contract.n_sae_features, 3)) if decoder is None else decoder),
    )


def _decoded_source_len(prefix_text: str) -> int:
    model = _source_model()
    token_ids = model.tokenizer(prefix_text, padding=False)["input_ids"]
    return len(model.tokenizer.decode(token_ids))


def _patch_extraction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    feature_ids: list[t.Tensor] | None = None,
    feature_magnitudes: list[t.Tensor] | None = None,
    char_starts: list[int | float] | None = None,
    char_ends: list[int | float] | None = None,
    n_tokens: int = 7,
) -> None:
    ids = [t.tensor([1])] if feature_ids is None else feature_ids
    magnitudes = [t.tensor([4.0])] if feature_magnitudes is None else feature_magnitudes
    starts = [0] if char_starts is None else char_starts
    ends = [11] if char_ends is None else char_ends

    def extract_document_features(**_kwargs: Any):
        return ids, magnitudes, list(range(len(ids))), starts, ends, n_tokens

    monkeypatch.setattr(
        cache_module,
        "extract_document_features",
        extract_document_features,
    )


def _extract(
    *,
    prefix_text: str,
    contract: Any,
    base_sae: Any,
) -> cache_module.RawClusteredPrefix:
    return cache_module.extract_raw_clustered_prefix(
        prefix_text=prefix_text,
        base_model=_source_model(),
        base_sae=base_sae,
        contract=contract,
        segment_size=contract.feature_directory.segment_size,
        extraction_max_tokens=128,
    )


def _raw_prefix() -> cache_module.RawClusteredPrefix:
    return cache_module.RawClusteredPrefix(
        segments=(
            cache_module.RawClusteredFeatureSegment(
                feature_ids=t.tensor([1, 3], dtype=t.int64),
                feature_magnitudes=t.tensor([0.5, 2.0], dtype=t.float32),
                char_start=0,
                char_end=6,
            ),
            cache_module.RawClusteredFeatureSegment(
                feature_ids=t.empty(0, dtype=t.int64),
                feature_magnitudes=t.empty(0, dtype=t.float32),
                char_start=6,
                char_end=11,
            ),
        ),
        n_tokens=7,
    )


def _raw_metadata(
    contract: Any,
    *,
    n_tokens: int = 7,
) -> dict[str, str]:
    return cache_module._raw_metadata(
        contract=contract,
        prefix_hash=make_prefix_hash("prefix"),
        prefix_text_sha256=cache_module._sha256_text("prefix"),
        extraction_identity_sha256=_SHA_D,
        n_tokens=n_tokens,
    )


def _expected_metadata(metadata: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in metadata.items() if key != "n_tokens"}


def _load_raw_prefix(
    path: Path,
    *,
    expected_metadata_without_n_tokens: dict[str, str],
    n_sae_features: int,
) -> cache_module.RawClusteredPrefix:
    raw, _sha256 = cache_module._load_raw_prefix_snapshot(
        path,
        expected_metadata_without_n_tokens=expected_metadata_without_n_tokens,
        n_sae_features=n_sae_features,
    )
    return raw


def _write_safetensors(
    path: Path,
    tensors: dict[str, t.Tensor],
    *,
    metadata: dict[str, str] | None,
) -> None:
    save_file(tensors, str(path), metadata=metadata)


def _valid_raw_tensors() -> dict[str, t.Tensor]:
    return {
        "n_segments": t.tensor([2], dtype=t.int64),
        "seg_0_feature_ids": t.tensor([1, 3], dtype=t.int64),
        "seg_0_feature_magnitudes": t.tensor([0.5, 2.0], dtype=t.float32),
        "seg_0_char_bounds": t.tensor([0, 6], dtype=t.int64),
        "seg_1_feature_ids": t.empty(0, dtype=t.int64),
        "seg_1_feature_magnitudes": t.empty(0, dtype=t.float32),
        "seg_1_char_bounds": t.tensor([6, 11], dtype=t.int64),
    }


def _assert_raw_prefix_equal(
    actual: cache_module.RawClusteredPrefix,
    expected: cache_module.RawClusteredPrefix,
) -> None:
    assert actual.n_tokens == expected.n_tokens
    assert len(actual.segments) == len(expected.segments)
    for actual_segment, expected_segment in zip(
        actual.segments,
        expected.segments,
        strict=True,
    ):
        assert t.equal(actual_segment.feature_ids, expected_segment.feature_ids)
        assert t.equal(
            actual_segment.feature_magnitudes,
            expected_segment.feature_magnitudes,
        )
        assert actual_segment.char_start == expected_segment.char_start
        assert actual_segment.char_end == expected_segment.char_end


def _runtime_cfg(
    *,
    segment_size: object = 4,
    num_buckets: object = 2,
    extraction_threshold: object = 0.0,
    extraction_max_tokens: object = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        context_cache=SimpleNamespace(
            segment_size=segment_size,
            num_buckets=num_buckets,
        ),
        extraction_threshold=extraction_threshold,
        extraction_max_tokens=extraction_max_tokens,
    )


def test_clustered_path_helpers_use_raw_subtree_two_hex_shard_and_marker(
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    prefix_hash = "abcdef0123456789"

    assert paths.clustered_context_cache_raw_dir() == paths.context_cache_dir() / "raw"
    assert paths.clustered_context_cache_raw_file(prefix_hash) == (
        paths.context_cache_dir() / "raw" / "ab" / f"{prefix_hash}.safetensors"
    )
    assert paths.clustered_context_cache_partial_marker() == (
        paths.context_cache_dir() / "clustered_partial.json"
    )


def test_canonical_json_and_hash_helpers_are_compact_sorted_and_utf8() -> None:
    value = {"z": "é", "a": [2, 1]}
    canonical = '{"a":[2,1],"z":"é"}'

    assert canonical_json(value) == canonical
    assert (
        cache_module._sha256_bytes(canonical.encode())
        == hashlib.sha256(canonical.encode()).hexdigest()
    )
    assert cache_module._sha256_json(value) == hashlib.sha256(canonical.encode()).hexdigest()
    assert cache_module._sha256_text("é") == hashlib.sha256("é".encode()).hexdigest()


def test_raw_phase_constants_and_metadata_are_exact() -> None:
    contract = _contract()

    assert cache_module.CLUSTERED_CONTEXT_CACHE_SCHEMA_VERSION == 1
    assert cache_module.CLUSTERED_CONTEXT_CACHE_KIND == ("axbench_bpb_adv_clustered_context_cache")
    assert cache_module.RAW_PREFIX_SCHEMA_VERSION == 1
    assert cache_module.CLUSTERED_CONTEXT_CHAR_ALIGNMENT_VERSION == (
        "source_bos_inclusive_end_marker_reconcile_v2"
    )
    assert _raw_metadata(contract) == {
        "schema_version": "1",
        "kind": "axbench_bpb_adv_clustered_context_cache_raw_prefix",
        "feature_contract_fingerprint": _SHA_A,
        "prefix_hash": make_prefix_hash("prefix"),
        "prefix_text_sha256": hashlib.sha256(b"prefix").hexdigest(),
        "extraction_identity_sha256": _SHA_D,
        "n_tokens": "7",
    }


def test_deterministic_safetensors_failure_is_branded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_serializer(
        _tensors: dict[str, t.Tensor],
        *,
        metadata: dict[str, str],
    ) -> bytes:
        del metadata
        raise ValueError("malformed serializer output")

    monkeypatch.setattr(
        cache_module,
        "deterministic_safetensors_bytes",
        fail_serializer,
    )

    with pytest.raises(AxBenchBpbAdvError, match="malformed serializer output"):
        cache_module._deterministic_safetensors_bytes(
            {"value": t.tensor([1])},
            metadata={"kind": "test"},
        )


def test_population_entries_validate_hashes_and_sort_by_prefix_hash() -> None:
    prefixes = {
        "beta": make_prefix_hash("beta"),
        "alpha": make_prefix_hash("alpha"),
    }

    entries = cache_module._population_entries(prefixes)

    assert entries == sorted(entries, key=lambda entry: entry["prefix_hash"])
    assert entries == [
        {
            "prefix_hash": prefix_hash,
            "prefix_text_sha256": hashlib.sha256(prefix_text.encode()).hexdigest(),
        }
        for prefix_text, prefix_hash in sorted(
            prefixes.items(),
            key=lambda item: item[1],
        )
    ]


def test_population_entries_reject_prefix_hash_mismatch() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="does not match its prefix text"):
        cache_module._population_entries({"alpha": "wrong"})


def test_population_entries_reject_distinct_text_hash_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "make_prefix_hash", lambda _text: "collision")

    with pytest.raises(AxBenchBpbAdvError, match="distinct prefix texts collide"):
        cache_module._population_entries(
            {
                "alpha": "collision",
                "beta": "collision",
            }
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("non_cpu", "resident on CPU"),
        ("non_1d", "one-dimensional"),
        ("misaligned", "misaligned"),
        ("ids_dtype", "IDs must use int64"),
        ("magnitudes_dtype", "magnitudes must use float32"),
        ("nan", "non-finite"),
        ("inf", "non-finite"),
        ("zero", "non-positive sparse magnitudes"),
        ("negative_magnitude", "non-positive sparse magnitudes"),
        ("negative_id", "out-of-range"),
        ("too_large_id", "out-of-range"),
        ("duplicate_id", "duplicate"),
    ],
)
def test_validate_raw_segment_rejects_every_invalid_domain(
    case: str,
    match: str,
) -> None:
    ids = t.tensor([1, 2], dtype=t.int64)
    magnitudes = t.tensor([1.0, 2.0], dtype=t.float32)
    if case == "non_cpu":
        ids = t.empty(2, dtype=t.int64, device="meta")
    elif case == "non_1d":
        ids = ids.reshape(1, 2)
        magnitudes = magnitudes.reshape(1, 2)
    elif case == "misaligned":
        magnitudes = magnitudes[:1]
    elif case == "ids_dtype":
        ids = ids.to(t.int32)
    elif case == "magnitudes_dtype":
        magnitudes = magnitudes.to(t.float64)
    elif case == "nan":
        magnitudes[0] = float("nan")
    elif case == "inf":
        magnitudes[0] = float("inf")
    elif case == "zero":
        magnitudes[0] = 0.0
    elif case == "negative_magnitude":
        magnitudes[0] = -1.0
    elif case == "negative_id":
        ids[0] = -1
    elif case == "too_large_id":
        ids[0] = 8
    elif case == "duplicate_id":
        ids[1] = ids[0]
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(f"unknown raw-segment case {case}")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cache_module._validate_raw_segment(
            ids,
            magnitudes,
            n_sae_features=8,
            surface="test segment",
        )


def test_validate_raw_segment_accepts_documented_empty_selected_domain() -> None:
    cache_module._validate_raw_segment(
        t.empty(0, dtype=t.int64),
        t.empty(0, dtype=t.float32),
        n_sae_features=8,
        surface="empty segment",
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("width", "SAE width does not match"),
        ("hook", "SAE hook does not match"),
        ("missing_decoder", "missing a rank-2 W_dec"),
        ("rank_one_decoder", "missing a rank-2 W_dec"),
        ("transposed_decoder", "decoder feature axis"),
        ("pooling", "unsupported contract pooling mode"),
    ],
)
def test_extract_raw_clustered_prefix_rejects_runtime_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    match: str,
) -> None:
    del tmp_path
    contract = _contract(pooling="invalid" if case == "pooling" else "max_pool")
    sae = _sae(contract)
    if case == "width":
        sae.cfg.d_sae = 7
    elif case == "hook":
        sae.cfg.hook_name = "blocks.7.hook_resid_post"
    elif case == "missing_decoder":
        del sae.W_dec
    elif case == "rank_one_decoder":
        sae.W_dec = t.zeros(8)
    elif case == "transposed_decoder":
        sae.W_dec = t.zeros((3, 8))
    monkeypatch.setattr(
        cache_module,
        "extract_document_features",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime mismatch must fail before extraction")
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _extract(
            prefix_text="abcdef",
            contract=contract,
            base_sae=sae,
        )


def test_extract_raw_clustered_prefix_accepts_metadata_hook_and_calls_11_arg_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    sae = _sae(contract)
    del sae.cfg.hook_name
    sae.cfg.metadata = {"hook_name": contract.sae.hook_name}
    captured: dict[str, Any] = {}
    decoded_len = _decoded_source_len("abcdef")

    def extract_document_features(**kwargs: Any):
        captured.update(kwargs)
        return (
            [t.tensor([1])],
            [t.tensor([4.0])],
            [0],
            [0],
            [decoded_len],
            7,
        )

    monkeypatch.setattr(
        cache_module,
        "extract_document_features",
        extract_document_features,
    )

    raw = _extract(
        prefix_text="abcdef",
        contract=contract,
        base_sae=sae,
    )

    assert raw.n_tokens == 7
    assert set(captured) == {
        "model",
        "sae",
        "text",
        "layer",
        "context_window",
        "pooling",
        "segment_mode",
        "segment_size",
        "hook_name",
        "context_independent_features",
        "ignore_bos_features",
    }
    assert captured["hook_name"] == contract.sae.hook_name
    assert [(segment.char_start, segment.char_end) for segment in raw.segments] == [
        (0, decoded_len)
    ]


def test_extract_raw_clustered_prefix_rejects_tokenizer_replay_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    decoded_len = _decoded_source_len("abcdef")
    _patch_extraction(
        monkeypatch,
        char_ends=[decoded_len],
        n_tokens=6,
    )

    with pytest.raises(AxBenchBpbAdvError, match="token count disagrees"):
        _extract(
            prefix_text="abcdef",
            contract=contract,
            base_sae=_sae(contract),
        )


def test_extract_raw_clustered_prefix_requires_model_tokenizer_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    decoded_len = _decoded_source_len("abcdef")
    _patch_extraction(monkeypatch, char_ends=[decoded_len])

    with pytest.raises(AxBenchBpbAdvError, match="missing tokenizer/cfg.n_ctx"):
        cache_module.extract_raw_clustered_prefix(
            prefix_text="abcdef",
            base_model=object(),
            base_sae=_sae(contract),
            contract=contract,
            segment_size=4,
            extraction_max_tokens=128,
        )


def test_extract_raw_clustered_prefix_brands_tokenizer_replay_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    decoded_len = _decoded_source_len("abcdef")
    _patch_extraction(monkeypatch, char_ends=[decoded_len])
    monkeypatch.setattr(
        cache_module,
        "truncate_to_training_window",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("tokenizer failure")),
    )

    with pytest.raises(AxBenchBpbAdvError, match="could not independently reconstruct"):
        _extract(
            prefix_text="abcdef",
            contract=contract,
            base_sae=_sae(contract),
        )


@pytest.mark.parametrize("case", ["no_ids", "zero_width"])
def test_extract_raw_clustered_prefix_requires_positive_width_decoded_source(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    contract = _contract()
    _patch_extraction(
        monkeypatch,
        feature_ids=[],
        feature_magnitudes=[],
        char_starts=[],
        char_ends=[],
    )
    if case == "no_ids":
        monkeypatch.setattr(
            cache_module,
            "truncate_to_training_window",
            lambda *_args: ([], 7),
        )
    else:
        monkeypatch.setattr(
            cache_module,
            "truncate_to_training_window",
            lambda *_args: ([0], 7),
        )
        monkeypatch.setattr(
            cache_module,
            "token_to_char_offsets",
            lambda *_args: [0, 0],
        )

    with pytest.raises(AxBenchBpbAdvError, match="no positive-width decoded source"):
        _extract(
            prefix_text="abcdef",
            contract=contract,
            base_sae=_sae(contract),
        )


def test_extract_raw_clustered_prefix_rejects_nonpositive_extraction_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    _patch_extraction(
        monkeypatch,
        char_ends=[5],
        n_tokens=0,
    )
    monkeypatch.setattr(
        cache_module,
        "truncate_to_training_window",
        lambda *_args: ([0], 0),
    )

    with pytest.raises(AxBenchBpbAdvError, match="invalid token count"):
        _extract(
            prefix_text="abcdef",
            contract=contract,
            base_sae=_sae(contract),
        )


@pytest.mark.parametrize(
    ("char_starts", "char_ends", "match"),
    [
        ([-1], [11], "could not reconcile"),
        ([12], [12], "could not reconcile"),
        ([0.0], [11.0], "could not reconcile"),
    ],
)
def test_extract_raw_clustered_prefix_rejects_invalid_character_bounds(
    monkeypatch: pytest.MonkeyPatch,
    char_starts: list[int | float],
    char_ends: list[int | float],
    match: str,
) -> None:
    contract = _contract()
    assert _decoded_source_len("abcdef") == 11
    _patch_extraction(
        monkeypatch,
        char_starts=char_starts,
        char_ends=char_ends,
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _extract(
            prefix_text="abcdef",
            contract=contract,
            base_sae=_sae(contract),
        )


def test_extract_raw_clustered_prefix_forward_extends_with_shared_default_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    prefix_text = "x" * (CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS * 2 + 5)
    decoded_len = _decoded_source_len(prefix_text)
    reported_end = decoded_len - CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    _patch_extraction(
        monkeypatch,
        char_ends=[reported_end],
        n_tokens=len(prefix_text) + 1,
    )
    real_align = cache_module.align_char_ranges_to_decoded_prefix

    def align_without_private_window(**kwargs: Any):
        assert "end_marker_window" not in kwargs
        return real_align(**kwargs)

    monkeypatch.setattr(
        cache_module,
        "align_char_ranges_to_decoded_prefix",
        align_without_private_window,
    )

    raw = _extract(
        prefix_text=prefix_text,
        contract=contract,
        base_sae=_sae(contract),
    )

    assert decoded_len - reported_end == CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    assert raw.segments[0].char_end == decoded_len


def test_extract_raw_clustered_prefix_rejects_forward_drift_beyond_shared_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    prefix_text = "x" * (CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS * 2 + 5)
    decoded_len = _decoded_source_len(prefix_text)
    reported_end = decoded_len - CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS - 1
    _patch_extraction(
        monkeypatch,
        char_ends=[reported_end],
        n_tokens=len(prefix_text) + 1,
    )

    with pytest.raises(AxBenchBpbAdvError, match="delta exceeds end-marker window"):
        _extract(
            prefix_text=prefix_text,
            contract=contract,
            base_sae=_sae(contract),
        )


def test_extract_raw_clustered_prefix_reverse_filters_and_clips_in_lockstep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    prefix_text = "x" * (CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS * 2 + 5)
    decoded_len = _decoded_source_len(prefix_text)
    coverage_end = decoded_len + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    _patch_extraction(
        monkeypatch,
        feature_ids=[
            t.tensor([1]),
            t.tensor([3]),
            t.tensor([6]),
        ],
        feature_magnitudes=[
            t.tensor([1.0]),
            t.tensor([3.0]),
            t.tensor([6.0]),
        ],
        char_starts=[0, decoded_len - 2, decoded_len + 1],
        char_ends=[decoded_len - 2, decoded_len + 1, coverage_end],
        n_tokens=len(prefix_text) + 1,
    )

    raw = _extract(
        prefix_text=prefix_text,
        contract=contract,
        base_sae=_sae(contract),
    )

    assert coverage_end - decoded_len == CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    assert [(segment.char_start, segment.char_end) for segment in raw.segments] == [
        (0, decoded_len - 2),
        (decoded_len - 2, decoded_len),
    ]
    assert [segment.feature_ids.tolist() for segment in raw.segments] == [[1], [3]]


def test_extract_raw_clustered_prefix_rejects_reverse_drift_beyond_shared_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    prefix_text = "x" * 5
    decoded_len = _decoded_source_len(prefix_text)
    _patch_extraction(
        monkeypatch,
        char_ends=[decoded_len + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1],
        n_tokens=len(prefix_text) + 1,
    )

    with pytest.raises(AxBenchBpbAdvError, match="delta exceeds end-marker window"):
        _extract(
            prefix_text=prefix_text,
            contract=contract,
            base_sae=_sae(contract),
        )


def test_raw_prefix_round_trip_reproduces_exact_segments(tmp_path: Path) -> None:
    contract = _contract()
    raw = _raw_prefix()
    metadata = _raw_metadata(contract)
    path = tmp_path / "raw.safetensors"
    path.write_bytes(cache_module._serialize_raw_prefix(raw, metadata=metadata))

    loaded = _load_raw_prefix(
        path,
        expected_metadata_without_n_tokens=_expected_metadata(metadata),
        n_sae_features=contract.n_sae_features,
    )

    _assert_raw_prefix_equal(loaded, raw)


def test_raw_prefix_serialization_is_deterministic_and_install_is_immutable(
    tmp_path: Path,
) -> None:
    contract = _contract()
    raw = _raw_prefix()
    metadata = _raw_metadata(contract)
    reordered_metadata = dict(reversed(tuple(metadata.items())))

    first = cache_module._serialize_raw_prefix(raw, metadata=metadata)
    second = cache_module._serialize_raw_prefix(
        raw,
        metadata=reordered_metadata,
    )

    assert first == second
    path = tmp_path / "raw" / "prefix.safetensors"
    cache_module._install_once(first, path, surface="raw prefix")
    cache_module._install_once(second, path, surface="raw prefix")
    assert path.read_bytes() == first

    changed_raw = cache_module.RawClusteredPrefix(
        segments=(
            cache_module.RawClusteredFeatureSegment(
                feature_ids=t.tensor([1, 3], dtype=t.int64),
                feature_magnitudes=t.tensor([0.5, 3.0], dtype=t.float32),
                char_start=0,
                char_end=6,
            ),
            raw.segments[1],
        ),
        n_tokens=raw.n_tokens,
    )
    changed = cache_module._serialize_raw_prefix(
        changed_raw,
        metadata=metadata,
    )
    assert changed != first
    with pytest.raises(AxBenchBpbAdvError, match="refusing to overwrite"):
        cache_module._install_once(changed, path, surface="raw prefix")


@pytest.mark.parametrize("direction", ["missing", "extra"])
def test_load_raw_prefix_rejects_metadata_key_set_mismatch(
    tmp_path: Path,
    direction: str,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    if direction == "missing":
        metadata.pop("prefix_text_sha256")
    else:
        metadata["unexpected"] = "value"
    path = tmp_path / f"{direction}.safetensors"
    _write_safetensors(
        path,
        _valid_raw_tensors(),
        metadata=metadata,
    )

    expected = _expected_metadata(_raw_metadata(contract))
    with pytest.raises(AxBenchBpbAdvError, match="metadata keys mismatch"):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=expected,
            n_sae_features=contract.n_sae_features,
        )


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "kind",
        "feature_contract_fingerprint",
        "prefix_hash",
        "prefix_text_sha256",
        "extraction_identity_sha256",
    ],
)
def test_load_raw_prefix_rejects_each_metadata_value_mismatch(
    tmp_path: Path,
    key: str,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    path = tmp_path / f"{key}.safetensors"
    _write_safetensors(
        path,
        _valid_raw_tensors(),
        metadata=metadata,
    )
    expected = _expected_metadata(metadata)
    expected[key] = "wrong"

    with pytest.raises(AxBenchBpbAdvError, match=rf"mismatch .* for {key}"):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=expected,
            n_sae_features=contract.n_sae_features,
        )


@pytest.mark.parametrize(
    ("n_tokens", "match"),
    [
        ("not-an-int", "invalid n_tokens"),
        ("0", "non-positive n_tokens"),
        ("-1", "non-positive n_tokens"),
    ],
)
def test_load_raw_prefix_rejects_invalid_n_tokens_metadata(
    tmp_path: Path,
    n_tokens: str,
    match: str,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    metadata["n_tokens"] = n_tokens
    path = tmp_path / f"n_tokens_{n_tokens}.safetensors"
    _write_safetensors(
        path,
        _valid_raw_tensors(),
        metadata=metadata,
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=_expected_metadata(metadata),
            n_sae_features=contract.n_sae_features,
        )


@pytest.mark.parametrize(
    ("n_segments", "match"),
    [
        (t.tensor(1, dtype=t.int64), "malformed n_segments"),
        (t.tensor([0, 0], dtype=t.int64), "malformed n_segments"),
        (t.tensor([0], dtype=t.int32), "malformed n_segments"),
        (t.tensor([-1], dtype=t.int64), "negative n_segments"),
    ],
)
def test_load_raw_prefix_rejects_malformed_or_negative_segment_count(
    tmp_path: Path,
    n_segments: t.Tensor,
    match: str,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    path = tmp_path / f"segments_{len(tuple(n_segments.shape))}_{n_segments.dtype}.safetensors"
    _write_safetensors(
        path,
        {"n_segments": n_segments},
        metadata=metadata,
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=_expected_metadata(metadata),
            n_sae_features=contract.n_sae_features,
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "seg_0_feature_ids",
        "seg_0_feature_magnitudes",
        "seg_0_char_bounds",
    ],
)
def test_load_raw_prefix_rejects_each_missing_segment_tensor(
    tmp_path: Path,
    missing_key: str,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    tensors = _valid_raw_tensors()
    tensors["n_segments"] = t.tensor([1], dtype=t.int64)
    for key in tuple(tensors):
        if key.startswith("seg_1_"):
            tensors.pop(key)
    tensors.pop(missing_key)
    path = tmp_path / f"missing_{missing_key}.safetensors"
    _write_safetensors(path, tensors, metadata=metadata)

    with pytest.raises(AxBenchBpbAdvError, match=rf"missing tensor.*{missing_key}"):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=_expected_metadata(metadata),
            n_sae_features=contract.n_sae_features,
        )


def test_load_raw_prefix_rejects_extra_tensor(tmp_path: Path) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    tensors = _valid_raw_tensors()
    tensors["unexpected"] = t.tensor([1], dtype=t.int64)
    path = tmp_path / "extra.safetensors"
    _write_safetensors(path, tensors, metadata=metadata)

    with pytest.raises(AxBenchBpbAdvError, match="tensor keys mismatch"):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=_expected_metadata(metadata),
            n_sae_features=contract.n_sae_features,
        )


def test_load_raw_prefix_rejects_missing_segment_count_tensor(
    tmp_path: Path,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    path = tmp_path / "missing_n_segments.safetensors"
    _write_safetensors(
        path,
        {"unexpected": t.tensor([0], dtype=t.int64)},
        metadata=metadata,
    )

    with pytest.raises(AxBenchBpbAdvError, match="could not load raw prefix tensors"):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=_expected_metadata(metadata),
            n_sae_features=contract.n_sae_features,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("negative", "invalid or unordered bounds"),
        ("reversed", "invalid or unordered bounds"),
        ("overlap", "invalid or unordered bounds"),
        ("wrong_shape", "malformed char bounds"),
        ("wrong_dtype", "malformed char bounds"),
    ],
)
def test_load_raw_prefix_rejects_invalid_unordered_or_malformed_bounds(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    tensors = _valid_raw_tensors()
    if case == "negative":
        tensors["seg_0_char_bounds"] = t.tensor([-1, 6], dtype=t.int64)
    elif case == "reversed":
        tensors["seg_0_char_bounds"] = t.tensor([7, 6], dtype=t.int64)
    elif case == "overlap":
        tensors["seg_1_char_bounds"] = t.tensor([5, 11], dtype=t.int64)
    elif case == "wrong_shape":
        tensors["seg_0_char_bounds"] = t.tensor([0, 6, 7], dtype=t.int64)
    elif case == "wrong_dtype":
        tensors["seg_0_char_bounds"] = t.tensor([0, 6], dtype=t.int32)
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(f"unknown bounds case {case}")
    path = tmp_path / f"{case}.safetensors"
    _write_safetensors(path, tensors, metadata=metadata)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _load_raw_prefix(
            path,
            expected_metadata_without_n_tokens=_expected_metadata(metadata),
            n_sae_features=contract.n_sae_features,
        )


def test_load_raw_prefix_accepts_zero_segments(tmp_path: Path) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    path = tmp_path / "zero_segments.safetensors"
    _write_safetensors(
        path,
        {"n_segments": t.tensor([0], dtype=t.int64)},
        metadata=metadata,
    )

    loaded = _load_raw_prefix(
        path,
        expected_metadata_without_n_tokens=_expected_metadata(metadata),
        n_sae_features=contract.n_sae_features,
    )

    assert loaded == cache_module.RawClusteredPrefix(segments=(), n_tokens=7)


def test_load_raw_prefix_rejects_missing_corrupt_and_metadata_free_files(
    tmp_path: Path,
) -> None:
    contract = _contract()
    metadata = _raw_metadata(contract)
    expected = _expected_metadata(metadata)
    with pytest.raises(AxBenchBpbAdvError, match="artifact is missing"):
        _load_raw_prefix(
            tmp_path / "missing.safetensors",
            expected_metadata_without_n_tokens=expected,
            n_sae_features=contract.n_sae_features,
        )

    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"not safetensors")
    with pytest.raises(AxBenchBpbAdvError, match="could not read safetensors metadata"):
        _load_raw_prefix(
            corrupt,
            expected_metadata_without_n_tokens=expected,
            n_sae_features=contract.n_sae_features,
        )

    metadata_free = tmp_path / "metadata_free.safetensors"
    _write_safetensors(
        metadata_free,
        {"n_segments": t.tensor([0], dtype=t.int64)},
        metadata=None,
    )
    with pytest.raises(AxBenchBpbAdvError, match="missing metadata"):
        _load_raw_prefix(
            metadata_free,
            expected_metadata_without_n_tokens=expected,
            n_sae_features=contract.n_sae_features,
        )


def test_expected_extraction_identity_has_exact_raw_phase_contract() -> None:
    contract = _contract()

    identity = cache_module._expected_extraction_identity(
        contract=contract,
        segment_size=4,
        extraction_max_tokens=32,
    )

    assert set(identity) == {
        "feature_contract_fingerprint",
        "checkpoint_model_sha256",
        "statistics_file_sha256",
        "percentile_table_checksum",
        "selection_hash",
        "selection_manifest_sha256",
        "selection_shards_sha256",
        "packed_row_fingerprint",
        "preselection_seed",
        "positive_selection",
        "negative_selection",
        "num_buckets",
        "effective_positive_buckets",
        "packed_selection_algorithm",
        "char_alignment_algorithm",
        "segment_size",
        "extraction_max_tokens",
        "sae_layer",
        "sae_hook",
        "pooling",
        "context_independent_features",
        "ignore_bos_features",
    }
    assert identity["packed_selection_algorithm"] == _PACKED_SELECTION_ALGORITHM
    assert identity["char_alignment_algorithm"] == (
        cache_module.CLUSTERED_CONTEXT_CHAR_ALIGNMENT_VERSION
    )


@_PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("segment_size", "extraction_max_tokens")),
    value=st.one_of(
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        st.none(),
        st.text(max_size=32),
    ),
)
def test_expected_extraction_identity_scalar_domains_match_positive_integer_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "contract": _contract(),
        "segment_size": 4,
        "extraction_max_tokens": 32,
    }
    kwargs[field] = value
    valid = isinstance(value, int) and not isinstance(value, bool) and value > 0

    if valid:
        assert cache_module._expected_extraction_identity(**kwargs)[field] == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            cache_module._expected_extraction_identity(**kwargs)


@pytest.mark.parametrize("packed_algorithm", [None, "", 7])
def test_expected_extraction_identity_requires_packed_algorithm(
    packed_algorithm: object,
) -> None:
    contract = _contract(packed_selection_algorithm=packed_algorithm)

    with pytest.raises(AxBenchBpbAdvError, match="missing the packed selection"):
        cache_module._expected_extraction_identity(
            contract=contract,
            segment_size=4,
            extraction_max_tokens=32,
        )


@pytest.mark.parametrize("direction", ["missing", "extra"])
def test_validate_extraction_identity_enforces_exact_key_set_both_ways(
    direction: str,
) -> None:
    contract = _contract()
    identity = cache_module._expected_extraction_identity(
        contract=contract,
        segment_size=4,
        extraction_max_tokens=32,
    )
    if direction == "missing":
        identity.pop("selection_hash")
    else:
        identity["unexpected"] = "value"

    with pytest.raises(
        AxBenchBpbAdvError,
        match="extraction identity fields mismatch",
    ):
        cache_module._validate_extraction_identity_against_contract(
            identity,
            contract=contract,
        )


@pytest.mark.parametrize(
    "key",
    [
        "feature_contract_fingerprint",
        "checkpoint_model_sha256",
        "statistics_file_sha256",
        "percentile_table_checksum",
        "selection_hash",
        "selection_manifest_sha256",
        "selection_shards_sha256",
        "packed_row_fingerprint",
        "preselection_seed",
        "positive_selection",
        "negative_selection",
        "num_buckets",
        "effective_positive_buckets",
        "packed_selection_algorithm",
        "char_alignment_algorithm",
        "segment_size",
        "sae_layer",
        "sae_hook",
        "pooling",
        "context_independent_features",
        "ignore_bos_features",
    ],
)
def test_validate_extraction_identity_rejects_every_contract_mismatch(
    key: str,
) -> None:
    contract = _contract()
    identity = cache_module._expected_extraction_identity(
        contract=contract,
        segment_size=4,
        extraction_max_tokens=32,
    )
    identity[key] = "wrong"

    with pytest.raises(
        AxBenchBpbAdvError,
        match="extraction identity disagrees with contract",
    ):
        cache_module._validate_extraction_identity_against_contract(
            identity,
            contract=contract,
        )


@pytest.mark.parametrize("extraction_max_tokens", [True, 0, -1, 1.5, "32"])
def test_validate_extraction_identity_rejects_invalid_extraction_max_tokens(
    extraction_max_tokens: object,
) -> None:
    contract = _contract()
    identity = cache_module._expected_extraction_identity(
        contract=contract,
        segment_size=4,
        extraction_max_tokens=32,
    )
    identity["extraction_max_tokens"] = extraction_max_tokens

    with pytest.raises(AxBenchBpbAdvError, match="invalid extraction_max_tokens"):
        cache_module._validate_extraction_identity_against_contract(
            identity,
            contract=contract,
        )


def test_validate_extraction_identity_accepts_exact_contract() -> None:
    contract = _contract()
    identity = cache_module._expected_extraction_identity(
        contract=contract,
        segment_size=4,
        extraction_max_tokens=32,
    )

    cache_module._validate_extraction_identity_against_contract(
        identity,
        contract=contract,
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("segment_size", "segment size does not match"),
        ("bucket_mismatch", "bucket count does not match"),
        ("bucket_not_two", "requires checkpoint num_buckets == 2"),
        ("threshold", "does not support magnitude thresholding"),
        ("zero_max_tokens", "extraction_max_tokens must be positive"),
        ("negative_max_tokens", "extraction_max_tokens must be positive"),
        ("sae_width", "SAE width exceeds"),
    ],
)
def test_validate_runtime_configuration_rejects_every_guard(
    case: str,
    match: str,
) -> None:
    contract = _contract()
    cfg = _runtime_cfg()
    if case == "segment_size":
        cfg.context_cache.segment_size = 5
    elif case == "bucket_mismatch":
        cfg.context_cache.num_buckets = 3
    elif case == "bucket_not_two":
        contract.selection.num_buckets = 3
        cfg.context_cache.num_buckets = 3
    elif case == "threshold":
        cfg.extraction_threshold = 0.1
    elif case == "zero_max_tokens":
        cfg.extraction_max_tokens = 0
    elif case == "negative_max_tokens":
        cfg.extraction_max_tokens = -1
    elif case == "sae_width":
        contract.n_sae_features = 65_537
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(f"unknown runtime-config case {case}")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cache_module._validate_runtime_configuration(
            cfg=cfg,
            contract=contract,
        )


def test_validate_runtime_configuration_rejects_unknown_contract_segment_mode() -> None:
    # kills: letting an unparseable contract segment mode escape as a bare
    # ValueError instead of the branded AxBench-BPB-Adv contract error.
    contract = _contract()
    contract.feature_directory.segment_mode = "sliding_window"

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"clustered feature contract has unknown segment mode 'sliding_window'",
    ):
        cache_module._validate_runtime_configuration(
            cfg=_runtime_cfg(),
            contract=contract,
        )


def test_validate_runtime_configuration_accepts_uint16_width_boundary() -> None:
    contract = _contract(n_sae_features=65_536)

    assert cache_module._validate_runtime_configuration(
        cfg=_runtime_cfg(),
        contract=contract,
    ) == (4, 32)


@_PROPERTY_SETTINGS
@given(
    value=st.one_of(
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        st.none(),
        st.text(max_size=32),
    )
)
def test_runtime_extraction_max_tokens_matches_strict_positive_integer_oracle(
    value: object,
) -> None:
    cfg = _runtime_cfg()
    cfg.extraction_max_tokens = value
    valid = isinstance(value, int) and not isinstance(value, bool) and value > 0

    if valid:
        assert cache_module._validate_runtime_configuration(
            cfg=cfg,
            contract=_contract(),
        ) == (4, value)
    else:
        with pytest.raises(AxBenchBpbAdvError, match="extraction_max_tokens must be positive"):
            cache_module._validate_runtime_configuration(
                cfg=cfg,
                contract=_contract(),
            )


@_PROPERTY_SETTINGS
@given(
    value=st.one_of(
        st.integers(min_value=-100, max_value=100),
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        st.booleans(),
        st.none(),
        st.text(max_size=32),
    )
)
def test_runtime_extraction_threshold_matches_exact_zero_number_oracle(
    value: object,
) -> None:
    cfg = _runtime_cfg()
    cfg.extraction_threshold = value
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0.0

    if valid:
        assert cache_module._validate_runtime_configuration(
            cfg=cfg,
            contract=_contract(),
        ) == (4, 32)
    else:
        with pytest.raises(AxBenchBpbAdvError, match="magnitude thresholding"):
            cache_module._validate_runtime_configuration(
                cfg=cfg,
                contract=_contract(),
            )


def test_d1_extraction_rejects_overlapping_bounds_the_loader_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G3a-D1: the WRITER must be at least as strict as the READER.

    ``_load_raw_prefix`` enforces ``char_start < previous_end``, but extraction
    validated each interval in isolation. So ``[(0, 10), (5, 11)]`` was accepted
    here, serialized, and installed as IMMUTABLE evidence -- and only then
    rejected by the loader. Under the frozen protocol the install precedes the
    verifying load, so the bad bytes are already committed and recovery needs a
    fresh run tag.
    """

    contract = _contract()
    prefix_text = "x" * 40
    _patch_extraction(
        monkeypatch,
        feature_ids=[t.tensor([1]), t.tensor([2])],
        feature_magnitudes=[t.tensor([4.0]), t.tensor([5.0])],
        char_starts=[0, 5],
        char_ends=[10, _decoded_source_len(prefix_text)],
        n_tokens=len(prefix_text) + 1,
    )

    with pytest.raises(AxBenchBpbAdvError, match="before the previous segment ended"):
        _extract(prefix_text=prefix_text, contract=contract, base_sae=_sae(contract))


def test_d1_extraction_still_accepts_touching_and_zero_width_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against over-correction: contiguous tiling must stay legal.

    Segments abut (``previous_end == char_start``) in the normal case, and a
    zero-width segment is legitimate -- neither may be rejected by D1.
    """

    contract = _contract()
    prefix_text = "x" * 40
    end = _decoded_source_len(prefix_text)
    _patch_extraction(
        monkeypatch,
        feature_ids=[t.tensor([1]), t.tensor([], dtype=t.int64), t.tensor([2])],
        feature_magnitudes=[
            t.tensor([4.0]),
            t.tensor([], dtype=t.float32),
            t.tensor([5.0]),
        ],
        char_starts=[0, 10, 10],
        char_ends=[10, 10, end],
        n_tokens=len(prefix_text) + 1,
    )

    raw = _extract(prefix_text=prefix_text, contract=contract, base_sae=_sae(contract))

    assert [(s.char_start, s.char_end) for s in raw.segments] == [
        (0, 10),
        (10, 10),
        (10, end),
    ]


@pytest.mark.parametrize("dtype", [t.float32, t.float64, t.float16, t.bfloat16, t.bool])
def test_d2_non_integral_feature_ids_raise_before_truncation(
    monkeypatch: pytest.MonkeyPatch, dtype: t.dtype
) -> None:
    """G3a-D2: the int64 cast happened BEFORE the dtype guard could see it.

    ``_validate_raw_segment`` therefore always observed int64 and could never
    detect a malformed extractor result: a float32 ``[1.9]`` was silently
    TRUNCATED to int64 ``[1]`` and persisted as immutable evidence with no
    error. Real upstream emits int32, so the intended int32 -> int64 widening
    stays lossless; only never-integral inputs are refused.
    """

    contract = _contract()
    prefix_text = "x" * 40
    _patch_extraction(
        monkeypatch,
        feature_ids=[t.tensor([1.9]).to(dtype)],
        feature_magnitudes=[t.tensor([4.0])],
        char_starts=[0],
        char_ends=[_decoded_source_len(prefix_text)],
        n_tokens=len(prefix_text) + 1,
    )

    with pytest.raises(AxBenchBpbAdvError, match="must be an integer tensor"):
        _extract(prefix_text=prefix_text, contract=contract, base_sae=_sae(contract))


def test_d3_serializer_binds_metadata_token_count_to_the_prefix() -> None:
    """G3a-D3: the loader treats metadata n_tokens as authoritative.

    Accepting it independently of ``raw.n_tokens`` meant a caller mismatch
    round-tripped silently (prefix 7, metadata "8" -> loaded 8), disproving
    total fidelity for a helper whose whole job is immutable evidence.
    """

    contract = _contract()
    raw = _raw_prefix()
    mismatched = dict(_raw_metadata(contract, n_tokens=raw.n_tokens))
    mismatched["n_tokens"] = str(raw.n_tokens + 1)

    with pytest.raises(AxBenchBpbAdvError, match="does not describe the prefix"):
        cache_module._serialize_raw_prefix(raw, metadata=mismatched)

    # The honest path is unaffected.
    matching = dict(_raw_metadata(contract, n_tokens=raw.n_tokens))
    matching["n_tokens"] = str(raw.n_tokens)
    assert cache_module._serialize_raw_prefix(raw, metadata=matching)


@pytest.mark.slow
def test_raw_prefix_serialization_is_byte_stable_ACROSS_PROCESSES() -> None:
    """The determinism guarantee only means anything across processes.

    ``deterministic_safetensors_bytes`` exists because the Rust serializer emits
    ``__metadata__`` through a hash map whose JSON key order varies PER PROCESS.
    Comparing two serializations inside one interpreter cannot exercise that --
    both reviewers of this PR flagged the in-repo check as structurally weaker
    than the property it guards, and one empirically reproduced the raw
    serializer producing eight different digests across eight interpreters.

    DDP owners write this artifact independently on different ranks, so
    identical input MUST give identical bytes in unrelated processes or the
    immutable-evidence design fails.
    """

    script = textwrap.dedent(
        """
        import hashlib, sys
        import torch as t
        from steering.eval.axbench_bpb_adv import clustered_context_cache as m

        raw = m.RawClusteredPrefix(
            segments=(
                m.RawClusteredFeatureSegment(
                    feature_ids=t.tensor([3, 1, 2], dtype=t.int64),
                    feature_magnitudes=t.tensor([0.5, 2.0, 1.0], dtype=t.float32),
                    char_start=0,
                    char_end=6,
                ),
            ),
            n_tokens=7,
        )
        # Many keys, inserted in an order that differs from sorted order, so a
        # non-canonical serializer has ample room to vary.
        metadata = {f"k{i:02d}": f"v{i}" for i in range(20, 0, -1)}
        metadata["n_tokens"] = "7"
        payload = m._serialize_raw_prefix(raw, metadata=metadata)
        sys.stdout.write(hashlib.sha256(payload).hexdigest())
        """
    )

    digests = set()
    for seed in ("0", "1", "42", "1337", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        digests.add(result.stdout.strip())

    assert len(digests) == 1, (
        "serialized bytes differ across processes; canonicalization is not "
        f"holding: {sorted(digests)}"
    )


def test_g3b_d3_raw_prefix_metadata_and_tensors_come_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A->B->A: the loader must parse the bytes it read, not reopen the path.

    `_load_raw_prefix` used to read the pathname twice -- `safe_open` for
    metadata, then `load_file` for tensors. A replacement between the two reads
    paired A's metadata, INCLUDING the n_tokens it returns, with B's segments,
    certifying a header that does not describe the tensors actually loaded.

    Held at the exact observable schedule of that swap: B sits at the pathname
    while the single read returns A. Everything must therefore describe A.
    """
    contract = _contract()
    metadata = _raw_metadata(contract)
    expected = _expected_metadata(metadata)

    a_tensors = _valid_raw_tensors()
    b_tensors = {
        "n_segments": t.tensor([1], dtype=t.int64),
        "seg_0_feature_ids": t.tensor([2], dtype=t.int64),
        "seg_0_feature_magnitudes": t.tensor([9.0], dtype=t.float32),
        "seg_0_char_bounds": t.tensor([0, 3], dtype=t.int64),
    }

    a_path = tmp_path / "a.safetensors"
    path = tmp_path / "raw.safetensors"
    _write_safetensors(a_path, a_tensors, metadata=metadata)
    _write_safetensors(path, b_tensors, metadata=metadata)
    a_bytes = a_path.read_bytes()
    assert a_bytes != path.read_bytes()

    real_read_bytes = Path.read_bytes

    def _swapped(self: Path) -> bytes:
        if self == path:
            return a_bytes
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _swapped)

    loaded = _load_raw_prefix(
        path,
        expected_metadata_without_n_tokens=expected,
        n_sae_features=contract.n_sae_features,
    )

    # A has two segments, B has one. Parsing the reopened pathname would yield
    # B's single segment beside A's metadata.
    assert len(loaded.segments) == 2
    assert t.equal(loaded.segments[0].feature_ids, t.tensor([1, 3], dtype=t.int64))
    assert loaded.n_tokens == int(metadata["n_tokens"])
