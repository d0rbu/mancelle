"""Real paired and packed fixtures; each guard documents its killed mutation."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from scripts.utils import record_fixed_cadence as recorder
from steering.conditional.encoder_decoder.fixed_cadence import (
    FixedCadenceArtifactError,
    evenly_spaced_rows,
)
from steering.conditional.encoder_decoder.fixed_cadence import (
    measure_fixed_self_cond_segment_size_rustbpe as measure,
)
from steering.sft.segment_mode_identity import (
    FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY as PROVENANCE,
)
from steering.sft.segment_mode_identity import (
    FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY as FIXED,
)
from steering.sft.segment_mode_identity import (
    resolve_fixed_self_cond_segment_size_rustbpe,
)
from tests.steering.fixed_cadence_helpers import CadenceFixture, make_cadence_fixture


@pytest.fixture
def artifacts(tmp_path: Path) -> CadenceFixture:
    return make_cadence_fixture(tmp_path)


def test_measure_full_clipped_and_heldout_windows(artifacts: CadenceFixture) -> None:
    # include held-out documents, use a document-average rather than window weighting.
    result = measure(**artifacts.arguments())
    assert result == measure(**artifacts.arguments())
    assert result.minimum == 4
    assert result.maximum == 7
    assert result.exclusions["clipped_windows"] > 0
    assert result.provenance["held_out_parquet"].endswith("shard_00002.parquet")
    assert result.sampled_documents == 4
    assert result.provenance["source_bos_char_lengths"] == {"shard_00000": 0, "shard_00001": 0}
    assert result.mean == Fraction(41, 7)
    assert result.windows == 7
    assert result.cadence == 6


@pytest.mark.parametrize("documents,expected", [(((2,), (4,)), 2), (((4,), (6,)), 4)])
def test_exact_ties_to_even_and_padding(tmp_path: Path, documents: Any, expected: int) -> None:
    fixture = make_cadence_fixture(tmp_path, documents=documents, context_len=32, n_shards=2)
    result = measure(**fixture.arguments())
    assert result.windows == 2
    assert result.mean == Fraction(2 * expected + (1 if expected == 2 else -1), 2)
    assert result.cadence == expected
    assert result.exclusions["padding_input_positions"] > 0
    assert result.exclusions["prediction_only_positions"] == result.sampled_rows
    assert result.quantiles["p05"] == result.minimum
    assert result.quantiles["p95"] == result.maximum


def test_spliced_source_positions_exclude_synthetic_bos_and_tail(tmp_path: Path) -> None:
    fixture = make_cadence_fixture(tmp_path, context_len=64, documents=((10, 14, 8),),
                                    source_lengths=((129, 128, 40),), spliced=True, n_shards=2)
    result = measure(**fixture.arguments())
    assert result.windows == 1
    assert result.mean == 7
    assert result.exclusions["oversize_source_windows"] == 1
    assert result.exclusions["incomplete_source_tails"] == 1
    assert len(result.provenance["inventories"]["splice"]) == 2


@pytest.mark.parametrize("total,limit,expected", [(1, 4096, (0,)), (4, 4, (0, 1, 2, 3)), (10, 3, (1, 5, 8)), (9, 2, (2, 6))])
def test_even_spacing(total: int, limit: int, expected: tuple[int, ...]) -> None:
    assert evenly_spaced_rows(total, limit) == expected


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    fixture = make_cadence_fixture(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    (original / "meta_250000.json").write_text(json.dumps(fixture.meta))
    (original / "model_250000.pt").write_bytes(b"unchanged model weights")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.pkl").write_bytes(b"fixture decoder tokenizer bytes")
    # Only replace tokenizer deserialization; the measurement and publication
    # consume real native feature, overlay, parquet and packed files.
    monkeypatch.setattr(recorder.RustBPETokenizer, "from_directory", lambda _: fixture.tokenizer)
    return OmegaConf.create({"checkpoint_meta_path": str(original / "meta_250000.json"),
        "feature_dir": str(fixture.feature_dir), "parquet_dir": str(fixture.parquet_dir),
        "tokenizer_dir": str(tokenizer_dir), "max_rows": 4096,
        "output_dir": str(tmp_path / "eval"), "fixed_segment_size_assertion": None})


def test_publish_unchanged_weights_and_production_reresolve(cfg: DictConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path(cfg.checkpoint_meta_path)
    original_bytes = original.read_bytes()
    result = recorder.run(cfg)
    meta = json.loads(result.read_text())
    assert resolve_fixed_self_cond_segment_size_rustbpe(meta, checkpoint_dir=str(result.parent)) == 6
    assert original.read_bytes() == original_bytes
    assert (result.parent / "model_250000.pt").is_symlink()
    assert (result.parent / "model_250000.pt").resolve() == original.with_name("model_250000.pt")
    assert meta[PROVENANCE]["original_checkpoint"][0]["path"] == str(original)
    assert len(meta[PROVENANCE]["decoder_tokenizer_inventory"]) == 1
    before = result.read_bytes()
    assert recorder.run(cfg) == result
    assert result.read_bytes() == before
    with pytest.raises(ValueError, match="conflicts"):
        resolve_fixed_self_cond_segment_size_rustbpe(meta, checkpoint_dir=str(result.parent), segment_size_assertion=7)
    meta[FIXED] += 1
    with pytest.raises(ValueError, match="contradicts measurement"):
        resolve_fixed_self_cond_segment_size_rustbpe(meta, checkpoint_dir=str(result.parent))


@pytest.mark.parametrize("change", ["cadence", "provenance", "weights", "incomplete"])
def test_contradictory_existing_leaf_never_overwritten(cfg: DictConfig, change: str) -> None:
    result = recorder.run(cfg)
    if change in {"cadence", "provenance"}:
        meta = json.loads(result.read_text())
        meta[FIXED if change == "cadence" else PROVENANCE] = 1
        result.write_text(json.dumps(meta))
    elif change == "weights":
        model = result.parent / "model_250000.pt"
        model.unlink()
        model.write_bytes(b"other weights")
    else:
        result.unlink()
    before = result.read_bytes() if result.exists() else None
    with pytest.raises(FixedCadenceArtifactError, match="contradictory|unchanged"):
        recorder.run(cfg)
    assert (result.read_bytes() if result.exists() else None) == before
