"""Real splicer -> packing -> cooldown junction-mask and resume contracts."""

import json
from pathlib import Path

import pytest
import torch as t

import steering.conditional.encoder_decoder.conditioning_splice as conditioning_splice
from nanochat.dataset import DatasetSplit
from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from steering.conditional.encoder_decoder.conditioning_splice import ConditioningSpliceIndex
from steering.conditional.encoder_decoder.dataloader import (
    _document_batches_with_features,
    encoder_decoder_data_loader_with_state,
    precomputed_data_loader,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import glob_overlay_shards, load_overlay_shard, save_overlay_shard
from steering.packed_batch_io import load_packed_row_shard
from steering.refined_types import PositiveInt
from tests.steering._splice_cf_fixtures import build_derived_env
from tests.steering.test_live_dataloader_splice_cf import _sampling_pipeline, _threshold_pipeline
from tests.steering.test_splice_counterfactual_preselected import _write_token_boundary_overlay


@pytest.fixture
def spliced_packed(tmp_path):
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    tok_checksum = _write_token_boundary_overlay(
        feature_dir=Path(env["feature_dir"]),
        data_dir=env["data_dir"],
        tokenizer=env["tokenizer"],
        segment_size=env["seg_size"],
    )
    feature_index = FeatureIndex(env["feature_dir"], tokenizer_checksum=tok_checksum)
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    precompute_packed_rows(
        parquet_paths=[str(path) for path in sorted(Path(env["data_dir"]).glob("*.parquet"))],
        tokenizer=env["tokenizer"],
        feature_index=feature_index,
        context_len=16,
        output_dir=packed_dir,
        buffer_size=2,
        rows_per_shard=2,
    )
    paths = glob_overlay_shards(packed_dir)
    (packed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": "test-cooldown-provenance",
                "metadata": {
                    "base_shard_stems": list(feature_index.shard_stems),
                    "provenance_shards": len(paths),
                },
            }
        )
    )
    return env, packed_dir, paths


def _expected_flags(feature_dir, packed_dir, shard_name, base, row_idx):
    manifest = json.loads((packed_dir / "manifest.json").read_text())
    stems = manifest["metadata"]["base_shard_stems"]
    prov = load_overlay_shard(provenance_dir(packed_dir) / shard_name)
    start, end = base.row_seg_offsets[row_idx : row_idx + 2].tolist()
    expected = []
    for segment in range(start, end):
        stem = stems[int(prov["base_shard_idx"][segment])]
        sidecar = json.loads(
            (Path(feature_dir) / "_splice_provenance" / f"{stem}.json").read_text()
        )
        doc = sidecar["docs"][int(prov["doc_idx"][segment])]
        expected.append(
            int(prov["seg_ordinal"][segment]) in {item["pos"] for item in doc["splices"]}
        )
    return expected


def _loader(env, packed_dir, *, enabled=True, resume=None):
    return precomputed_data_loader(
        packed_dir,
        B=4,
        feature_pipeline=_threshold_pipeline(mode=None),
        device="cpu",
        effective_buckets=4,
        splice_feature_dir=env["feature_dir"],
        include_splice_mask=enabled,
        resume_state_dict=resume,
    )


def test_real_packed_junction_join_and_exact_resume(spliced_packed):
    env, packed_dir, paths = spliced_packed
    direct = ConditioningSpliceIndex(
        packed_dir, Path(env["feature_dir"]), packed_shard_names=[p.name for p in paths]
    )
    all_flags = []
    for path in paths:
        base = load_packed_row_shard(path)
        for row in range(base.n_rows):
            actual = direct.row_flags(path.name, base, row)
            assert actual == _expected_flags(env["feature_dir"], packed_dir, path.name, base, row)
            all_flags.extend(actual)
    assert any(all_flags) and not all(all_flags)

    enabled = _loader(env, packed_dir)
    legacy = _loader(env, packed_dir, enabled=False)
    crossed_shards = False
    crossed_epochs = False
    for _ in range(8):
        batch = next(enabled)
        reference = next(legacy)
        for actual, expected in zip(batch[:9], reference[:9], strict=True):
            assert actual is None if expected is None else t.equal(actual, expected)
        state = batch[9]
        reduced_state = {k: v for k, v in state.items() if not k.startswith("conditioning_splice_")}
        assert reduced_state == reference[9]
        crossed_shards |= len({row[1] for row in state["batch_rows"]}) > 1
        crossed_epochs |= len({row[0] for row in state["batch_rows"]}) > 1
        for row, (_epoch, shard_name, row_idx) in enumerate(state["batch_rows"]):
            base = load_packed_row_shard(packed_dir / shard_name)
            expected = _expected_flags(env["feature_dir"], packed_dir, shard_name, base, row_idx)
            expected += [False] * (batch[4].shape[1] - len(expected))
            assert state["conditioning_splice_mask"][row] == expected
    assert crossed_shards and crossed_epochs
    restored = next(_loader(env, packed_dir, resume=json.loads(json.dumps(batch[9]))))
    for actual, expected in zip(restored, batch, strict=True):
        assert t.equal(actual, expected) if isinstance(expected, t.Tensor) else actual == expected


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("base_shard_idx", "bounds"),
        ("doc_idx", "bounds"),
        ("seg_ordinal", "prefix"),
        ("doc_idx", "coherence"),
        ("seg_ordinal", "length"),
        ("seg_ordinal", "dtype"),
    ],
)
def test_packed_join_rejects_malformed_provenance(spliced_packed, field, mutation):
    env, packed_dir, paths = spliced_packed
    path = paths[0]
    base = load_packed_row_shard(path)
    prov_path = provenance_dir(packed_dir) / path.name
    prov = load_overlay_shard(prov_path)
    if mutation == "bounds":
        prov[field][0] = 1_000_000
    elif mutation == "prefix":
        prov[field][0] = 1
    elif mutation == "coherence":
        assert base.seg_doc_id[0] == base.seg_doc_id[1]
        prov[field][1] = prov[field][0] + 1
    elif mutation == "length":
        prov[field] = prov[field][:-1]
    else:
        prov[field] = prov[field].to(t.int64)
    save_overlay_shard(prov, prov_path)
    index = ConditioningSpliceIndex(
        packed_dir, Path(env["feature_dir"]), packed_shard_names=[p.name for p in paths]
    )
    with pytest.raises(SpliceProvenanceError):
        index.row_flags(path.name, base, 0)


@pytest.mark.parametrize("artifact", ["manifest", "provenance", "sidecar"])
def test_resume_binds_cooldown_provenance_bytes(spliced_packed, artifact):
    env, packed_dir, paths = spliced_packed
    state = next(_loader(env, packed_dir))[9]
    if artifact == "manifest":
        path = packed_dir / "manifest.json"
        path.write_text(path.read_text() + "\n")
    elif artifact == "sidecar":
        path = sorted((Path(env["feature_dir"]) / "_splice_provenance").glob("*.json"))[0]
        path.write_text(path.read_text() + "\n")
    else:
        path = provenance_dir(packed_dir) / paths[0].name
        prov = load_overlay_shard(path)
        prov["seg_ordinal"][0] += 1
        save_overlay_shard(prov, path)
    with pytest.raises(SpliceProvenanceError, match="changed since"):
        next(_loader(env, packed_dir, resume=state))


def test_constructor_rejects_manifest_rewrite_restored_before_final_hash(
    spliced_packed, monkeypatch
):
    env, packed_dir, paths = spliced_packed
    manifest_path = packed_dir / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    rewritten = json.loads(original_bytes)
    original_stems = list(rewritten["metadata"]["base_shard_stems"])
    assert len(original_stems) > 1
    rewritten["metadata"]["base_shard_stems"] = original_stems[::-1]
    original_read = conditioning_splice.read_manifest

    def read_rewritten_manifest(directory):
        manifest_path.write_text(json.dumps(rewritten))
        try:
            observed = original_read(directory)
            assert observed is not None
            assert observed["metadata"]["base_shard_stems"] != original_stems
            return observed
        finally:
            # Both content hashes now see identical bytes, but the mapping read
            # between them came from a different file version.
            manifest_path.write_bytes(original_bytes)

    monkeypatch.setattr(conditioning_splice, "read_manifest", read_rewritten_manifest)
    with pytest.raises(SpliceProvenanceError, match="manifest changed"):
        ConditioningSpliceIndex(
            packed_dir,
            Path(env["feature_dir"]),
            packed_shard_names=[path.name for path in paths],
        )
    assert manifest_path.read_bytes() == original_bytes


def test_missing_packed_provenance_fails_and_disabled_remains_legacy(spliced_packed):
    env, packed_dir, paths = spliced_packed
    (provenance_dir(packed_dir) / paths[0].name).unlink()
    with pytest.raises(SpliceProvenanceError, match="one provenance shard"):
        next(_loader(env, packed_dir))
    assert "conditioning_splice_mask" not in next(_loader(env, packed_dir, enabled=False))[9]


def test_known_spliced_directory_requires_sidecars_in_both_loaders(spliced_packed):
    env, packed_dir, _paths = spliced_packed
    sidecar_dir = Path(env["feature_dir"]) / "_splice_provenance"
    sidecar_dir.rename(sidecar_dir.with_name("sidecars_removed"))
    with pytest.raises(SpliceProvenanceError, match="provenance dir not found"):
        next(_loader(env, packed_dir))
    with pytest.raises(SpliceProvenanceError, match="provenance dir not found"):
        next(
            _document_batches_with_features(
                DatasetSplit.TRAIN,
                None,
                128,
                env["tokenizer"],
                FeatureIndex(env["feature_dir"]),
                _threshold_pipeline(mode=None),
                PositiveInt.parse(env["seg_size"]),
                data_dir=env["data_dir"],
                include_splice_mask=True,
            )
        )
    assert "conditioning_splice_mask" not in next(_loader(env, packed_dir, enabled=False))[9]


def test_packed_manifest_requires_source_shard_mapping(spliced_packed):
    env, packed_dir, _paths = spliced_packed
    path = packed_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["metadata"]["base_shard_stems"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(SpliceProvenanceError, match="base_shard_stems"):
        next(_loader(env, packed_dir))


def test_live_real_sidecars_mark_only_junctions_and_preserve_payloads(tmp_path):
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    documents, state = next(
        _document_batches_with_features(
            DatasetSplit.TRAIN,
            None,
            128,
            env["tokenizer"],
            FeatureIndex(env["feature_dir"]),
            _threshold_pipeline(mode=None),
            PositiveInt.parse(env["seg_size"]),
            data_dir=env["data_dir"],
            include_splice_mask=True,
        )
    )
    sidecar = json.loads(
        (Path(env["feature_dir"]) / "_splice_provenance/shard_00000.json").read_text()
    )
    assert len(documents) == len(sidecar["docs"])
    for (_tokens, segments), record in zip(documents, sidecar["docs"], strict=True):
        assert segments is not None
        assert [s.is_splice for s in segments] == [
            i in {s["pos"] for s in record["splices"]} for i in range(record["n_segments"])
        ]
    assert state["conditioning_splice_identity"]["sidecars"]

    def loader(enabled, resume=None):
        return encoder_decoder_data_loader_with_state(
            env["tokenizer"],
            B=2,
            T=32,
            split=DatasetSplit.TRAIN,
            feature_index=FeatureIndex(env["feature_dir"]),
            pipeline=_sampling_pipeline(mode=None, with_negatives=True),
            segment_size=env["seg_size"],
            device="cpu",
            buffer_size=2,
            data_dir=env["data_dir"],
            include_splice_mask=enabled,
            resume_state_dict=resume,
        )

    enabled, legacy = loader(True), loader(False)
    for _ in range(3):
        batch, expected = next(enabled), next(legacy)
        for actual_tensor, expected_tensor in zip(batch[:9], expected[:9], strict=True):
            assert (
                actual_tensor is None
                if expected_tensor is None
                else t.equal(actual_tensor, expected_tensor)
            )
        assert {
            k: v for k, v in batch[9].items() if not k.startswith("conditioning_splice_")
        } == expected[9]
        assert t.tensor(batch[9]["conditioning_splice_mask"]).shape == batch[7].shape
    path = Path(env["feature_dir"]) / "_splice_provenance/shard_00000.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(SpliceProvenanceError, match="changed since"):
        next(loader(True, batch[9]))
