from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.eval.autointerp_filter.source_index import segment_token_bounds
from steering.eval.autointerp_filter.token_examples import _segment_token_bounds
from steering.eval.autointerp_filter.types import AutoInterpDataError
from steering.extract_features import process_shard
from steering.feature_io import glob_features, load_features
from steering.feature_shard_splice import (
    AlignmentError,
    load_source_segments,
    splice_feature_shard,
)
from steering.segmentation import SegmentMode
from steering.validate_features import (
    validate_shard_activations,
    validate_shard_against_source,
    validate_shard_segments,
)
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE

TURN_MODE = SegmentMode.turn_delimited_fixed_tokens


def _make_turn_shard(tmp_path):
    parquet = tmp_path / "shard_00000.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["abcdefghijklmnop"],
                "turn_char_starts": [[0, 5, 11]],
            }
        ),
        parquet,
    )
    model, sae = TinyHookedTransformer(), TinySAE()
    output = tmp_path / "features"
    process_shard(
        str(parquet),
        model,
        sae,
        layer=1,
        output_dir=str(output),
        context_window=64,
        segment_size=4,
        segment_mode=TURN_MODE,
        dataset_phase="standard_sft",
    )
    shard = load_features(glob_features(output)[0])
    return shard, parquet, model, sae


def test_source_validator_passes_turn_shard(tmp_path):
    shard, parquet, model, _ = _make_turn_shard(tmp_path)
    result = validate_shard_against_source(
        shard,
        parquet,
        model.tokenizer,
        context_window=64,
        segment_size=4,
    )
    assert result.valid, result.violations


@pytest.mark.parametrize(
    "key,match",
    [
        ("segment_char_starts", "segment_char_starts"),
        ("segment_positions", "segment_positions"),
    ],
)
def test_source_validator_reports_tampered_turn_geometry(tmp_path, key, match):
    shard, parquet, model, _ = _make_turn_shard(tmp_path)
    shard[key] = shard[key].clone()
    shard[key][1] += 1
    result = validate_shard_against_source(
        shard,
        parquet,
        model.tokenizer,
        context_window=64,
        segment_size=4,
    )
    assert not result.valid
    assert any(match in violation for violation in result.violations)


def test_source_validator_reports_missing_turn_column(tmp_path):
    shard, _, model, _ = _make_turn_shard(tmp_path)
    parquet = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text": ["abcdefghijklmnop"]}), parquet)
    result = validate_shard_against_source(
        shard,
        parquet,
        model.tokenizer,
        context_window=64,
        segment_size=4,
    )
    assert not result.valid
    assert result.violations == [
        "Parquet missing 'turn_char_starts' column for turn-delimited shard"
    ]


def test_activation_validator_reports_missing_turn_column(tmp_path):
    shard, _, model, sae = _make_turn_shard(tmp_path)
    parquet = tmp_path / "missing_activation.parquet"
    pq.write_table(pa.table({"text": ["abcdefghijklmnop"]}), parquet)
    result = validate_shard_activations(
        shard,
        parquet,
        model,
        sae,
        layer=1,
        context_window=64,
        segment_size=4,
        pooling="mean_pool",
        n_spot_check=1,
    )
    assert result.valid is False
    assert result.violations == [
        "Parquet missing 'turn_char_starts' column for turn-delimited shard"
    ]


def test_activation_validator_reextracts_turn_mode(tmp_path):
    shard, parquet, model, sae = _make_turn_shard(tmp_path)
    result = validate_shard_activations(
        shard,
        parquet,
        model,
        sae,
        layer=1,
        context_window=64,
        segment_size=4,
        pooling="mean_pool",
        n_spot_check=1,
    )
    assert result.valid, result.violations


def test_splicer_rejects_turn_base_shards():
    with pytest.raises(
        AlignmentError,
        match="feature-shard splicing supports fixed_tokens base shards only",
    ):
        load_source_segments(
            {"segment_mode": TURN_MODE.value},
            [],
            object(),
            segment_size=4,
            context_window=8,
            sae_max_context=None,
            shard_name="turn",
        )


def test_splice_feature_shard_rejects_turn_base_shard(tmp_path):
    _, parquet, model, _ = _make_turn_shard(tmp_path)
    shard_path = glob_features(tmp_path / "features")[0]
    with pytest.raises(
        AlignmentError,
        match="feature-shard splicing supports fixed_tokens base shards only",
    ):
        splice_feature_shard(
            shard_path,
            parquet,
            model.tokenizer,
            model="tiny",
            cut_fraction=0.5,
            seed=0,
            shard_index=0,
            shard_name="shard_00000",
            segment_size=4,
            context_window=64,
            sae_max_context=None,
        )


def _overlong_turn_shard(segment_mode=TURN_MODE.value):
    return {
        "offsets": t.tensor([0, 1, 2, 3], dtype=t.int64),
        "feature_ids": t.tensor([1, 2, 3], dtype=t.int32),
        "feature_mags": t.tensor([1.0, 1.0, 1.0]),
        "n_segments": 3,
        "n_docs": 1,
        "doc_ids": t.tensor([0, 0, 0], dtype=t.int32),
        "segment_indices": t.tensor([0, 1, 2], dtype=t.int32),
        "segment_positions": t.tensor([0, 40, 80], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 100, 200], dtype=t.int32),
        "segment_char_ends": t.tensor([100, 200, 300], dtype=t.int32),
        "doc_token_counts": t.tensor([120], dtype=t.int32),
        "segment_size": 4,
        "context_window": 1024,
        "segment_mode": segment_mode,
    }


def test_layer1_rejects_turn_segment_longer_than_segment_size():
    result = validate_shard_segments(_overlong_turn_shard())
    assert result.valid is False
    assert (
        "Doc 0: segment 0 at position 0 has length 40; expected 0 < length <= "
        "segment_size=4"
    ) in result.violations


def test_layer1_rejects_unknown_segment_mode(tmp_path):
    shard, _, _, _ = _make_turn_shard(tmp_path)
    shard["segment_mode"] = "bogus"
    result = validate_shard_segments(shard)
    assert result.valid is False
    assert result.violations == [
        "feature shard has unknown segment_mode='bogus'; must be one of "
        "['fixed_tokens', 'turn_delimited_fixed_tokens']"
    ]


def test_layer1_accepts_real_turn_shard(tmp_path):
    shard, _, _, _ = _make_turn_shard(tmp_path)
    result = validate_shard_segments(shard)
    assert result.valid is True, result.violations


def test_layer1_fixed_mode_reports_spacing_and_length_violations():
    result = validate_shard_segments(_overlong_turn_shard("fixed_tokens"))
    assert result.valid is False
    assert any("expected fixed_tokens spacing" in item for item in result.violations)
    assert any("segment 0 at position 0 has length 40" in item for item in result.violations)


@pytest.mark.parametrize(
    "validator",
    [validate_shard_against_source, validate_shard_activations],
)
def test_validators_report_unknown_segment_mode(tmp_path, validator):
    shard, parquet, model, sae = _make_turn_shard(tmp_path)
    shard["segment_mode"] = "bogus"
    if validator is validate_shard_against_source:
        result = validator(shard, parquet, model.tokenizer, 64, 4)
    else:
        result = validator(
            shard,
            parquet,
            model,
            sae,
            layer=1,
            context_window=64,
            segment_size=4,
            pooling="mean_pool",
            n_spot_check=1,
        )
    assert result.valid is False
    assert result.violations == [
        f"feature shard for {parquet} has unknown segment_mode='bogus'; must be one of "
        "['fixed_tokens', 'turn_delimited_fixed_tokens']"
    ]


def test_source_validator_collects_leading_bos_violation(tmp_path):
    shard, parquet, model, _ = _make_turn_shard(tmp_path)
    original = model.tokenizer.side_effect

    def _without_leading_bos(texts_or_text, **kwargs):
        result = original(texts_or_text, **kwargs)
        ids = result["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = [[7, *row[1:]] for row in ids]
        elif ids:
            ids = [7, *ids[1:]]
        return {"input_ids": ids}

    model.tokenizer.side_effect = _without_leading_bos
    result = validate_shard_against_source(
        shard,
        parquet,
        model.tokenizer,
        context_window=64,
        segment_size=4,
    )
    assert result.valid is False
    assert any(
        "turn segmentation failed" in item
        and "leading BOS" in item
        and "abcdefghijklmnop" in item
        for item in result.violations
    )


def test_source_validator_reports_turn_segmentation_failure(tmp_path):
    shard, _, model, _ = _make_turn_shard(tmp_path)
    parquet = tmp_path / "bad_turns.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["abcdefghijklmnop"],
                "turn_char_starts": [[0, 5, 4]],
            }
        ),
        parquet,
    )
    result = validate_shard_against_source(shard, parquet, model.tokenizer, 64, 4)
    assert result.valid is False
    assert any(
        "turn segmentation failed" in item
        and "turn_char_starts must be strictly increasing" in item
        for item in result.violations
    )


def test_source_validator_uses_default_tail_drift_window(tmp_path):
    text = "abcdefghij" * 4
    parquet = tmp_path / "far_drift.parquet"
    pq.write_table(
        pa.table({"text": [text], "turn_char_starts": [[0, 4]]}),
        parquet,
    )
    model, sae = TinyHookedTransformer(), TinySAE()
    output = tmp_path / "far_drift_features"
    process_shard(
        str(parquet),
        model,
        sae,
        layer=1,
        output_dir=str(output),
        context_window=64,
        segment_size=8,
        segment_mode=TURN_MODE,
        dataset_phase="standard_sft",
    )
    shard = load_features(glob_features(output)[0])
    original_decode = model.tokenizer.decode

    def _drifting_decode(ids):
        decoded = original_decode(ids)
        if len(ids) == len(text) + 1:
            decoded = decoded[:11] + "!" + decoded[12:]
        return decoded

    model.tokenizer.decode = _drifting_decode
    result = validate_shard_against_source(shard, parquet, model.tokenizer, 64, 8)
    assert result.valid is False
    assert any("tail_delta=30, window=20" in item for item in result.violations)


def test_source_validator_rejects_shifted_source_boundary(tmp_path):
    shard, _, model, _ = _make_turn_shard(tmp_path)
    parquet = tmp_path / "shifted_turns.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["abcdefghijklmnop"],
                "turn_char_starts": [[0, 6, 11]],
            }
        ),
        parquet,
    )
    result = validate_shard_against_source(shard, parquet, model.tokenizer, 64, 4)
    assert result.valid is False
    assert any("segment_positions" in item for item in result.violations)


@pytest.mark.parametrize("bounds", [_segment_token_bounds, segment_token_bounds])
def test_autointerp_positions_fallback_rejects_turn_mode(bounds):
    data = {
        "segment_size": 4,
        "segment_positions": t.tensor([0, 3], dtype=t.int32),
        "segment_mode": TURN_MODE.value,
    }
    with pytest.raises(
        AutoInterpDataError,
        match="has no explicit segment token provenance",
    ):
        bounds(data, shard_path=Path("turn.safetensors.zst"), n_rows=2)
