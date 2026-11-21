"""Conversion resume must prove input coverage before publishing progress."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.preprocess.convert_sft_to_parquet as converter
from errors import DataLoadError
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.sft_schema import SFTSplit
from tasks.common import TaskMixture
from tests.steering.test_sft_features_failfast import _write_feature_shard
from tests.tasks.test_convert_sft_to_parquet import (
    FakeTask,
    _convert_with_fake_tokenizer,
    _fake_conversations,
    _make_fake_mixture,
)


def _snapshot(root):
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


@pytest.mark.parametrize(
    ("first_cap", "next_cap", "first_width", "next_width"),
    [
        (4, 8, 10, 10),
        (4, 12, 10, 10),
        (8, 4, 10, 10),
        (4, None, 10, 10),
        (None, 12, 10, 10),
        (12, 12, 10, 6),
    ],
)
def test_changed_cap_or_shard_width_refuses_without_writes(
    tmp_path, first_cap, next_cap, first_width, next_width
):
    mixture = _make_fake_mixture(12)
    _convert_with_fake_tokenizer(
        mixture, tmp_path, max_conversations=first_cap, conversations_per_shard=first_width
    )
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="resume identity mismatch"):
        _convert_with_fake_tokenizer(
            mixture, tmp_path, max_conversations=next_cap, conversations_per_shard=next_width
        )
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("field", ["max_conversations", "conversations_per_shard"])
def test_existing_legacy_rows_cannot_acquire_guessed_conversion_identity(tmp_path, field):
    mixture = _make_fake_mixture(4)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "conversion_meta.json"
    meta = json.loads(path.read_text())
    del meta[field]
    path.write_text(json.dumps(meta))
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match=f"identity mismatch for {field}"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("indices", [[0, 1], [0, 1, 1, 3], [0, 2, 1, 3], [0, 1, 2, 4]])
def test_same_schema_shard_with_missing_duplicate_reordered_or_wrong_rows_is_rejected(
    tmp_path, indices
):
    mixture = _make_fake_mixture(4)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "shard_00000.parquet"
    table = pq.read_table(path).slice(0, len(indices))
    table = table.set_column(1, "original_index", pa.array(indices))
    pq.write_table(table, path)
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="row coverage mismatch"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before


def test_no_metadata_cannot_adopt_existing_full_format_shards(tmp_path):
    mixture = _make_fake_mixture(4)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    (tmp_path / "conversion_meta.json").unlink()
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="no conversion_meta.json identity"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before


def test_extra_shard_cannot_survive_a_successful_conversion(tmp_path):
    mixture = _make_fake_mixture(4)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    (tmp_path / "shard_00001.parquet").write_bytes((tmp_path / "shard_00000.parquet").read_bytes())
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="Unexpected conversion shards"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("crash_shard", [0, 1])
def test_crash_between_shard_and_metadata_publication_resumes_with_exact_counts(
    tmp_path, monkeypatch, crash_shard, caplog
):
    conversations = _fake_conversations(8)
    mixture = TaskMixture([FakeTask(conversations)])
    # One legitimate skip per shard; the input-prefix count remains eight.
    mixture[1]["messages"][-1]["content"] = ""
    mixture[5]["messages"][-1]["content"] = ""
    atomic_write = converter.atomic_write_parquet

    def crash_after_write(table, path):
        atomic_write(table, path)
        if path.name == f"shard_{crash_shard:05d}.parquet":
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(converter, "atomic_write_parquet", crash_after_write)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=4)
    meta = json.loads((tmp_path / "conversion_meta.json").read_text())
    assert meta["completed_shards"] == crash_shard
    assert meta["written_conversations"] == 3 * crash_shard
    assert meta["skipped_conversations"] == crash_shard
    shards = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.glob("*.parquet")}
    monkeypatch.setattr(converter, "atomic_write_parquet", atomic_write)
    _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=4)
    for path, identity in shards.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == identity
    meta = json.loads((tmp_path / "conversion_meta.json").read_text())
    assert meta["completed_shards"] == 2
    assert meta["total_conversations"] == 8
    assert meta["written_conversations"] == 6
    assert meta["skipped_conversations"] == 2
    assert "6 conversations written, 2 skipped from 8 input conversations" in caplog.text
    _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=4)
    assert json.loads((tmp_path / "conversion_meta.json").read_text()) == meta


def test_skip_rate_failure_is_not_bypassed_by_resuming_completed_shards(tmp_path):
    conversations = _fake_conversations(4)
    for row in conversations[:3]:
        row["messages"][-1]["content"] = ""
    mixture = TaskMixture([FakeTask(conversations)])
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_feature_shard(feature_dir)
    for _ in range(2):
        with pytest.raises(DataLoadError, match="Skipped 3/4.*More than 50%"):
            _convert_with_fake_tokenizer(mixture, tmp_path)
        # Check fresh failure before retry can revoke a premature completion
        # marker and hide the publication-order regression.
        meta = json.loads((tmp_path / "conversion_meta.json").read_text())
        assert meta["completed_shards"] == 0
        assert meta["written_conversations"] == 0
        assert meta["skipped_conversations"] == 0
        with pytest.raises(DataLoadError, match="conversion .* is incomplete"):
            load_sft_features(str(feature_dir), str(tmp_path))


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("text", ["wrong payload"] * 4),
        ("text", [None] * 4),
        ("turn_char_starts", [[1]] * 4),
        ("turn_char_starts", "float_coercion"),
        ("turn_char_starts", [None] * 4),
        ("turn_char_starts", [[False]] * 4),
    ],
)
def test_resume_rejects_payload_corruption_even_with_exact_indices(tmp_path, column, values):
    mixture = _make_fake_mixture(4)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "shard_00000.parquet"
    table = pq.read_table(path)
    if values == "float_coercion":
        values = [[float(start) for start in row] for row in table[column].to_pylist()]
    table = table.set_column(table.schema.get_field_index(column), column, pa.array(values))
    pq.write_table(table, path)
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="row content mismatch"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before


def test_real_tokenizer_resume_detects_source_drift_and_preserves_unicode_payload(
    tmp_path, monkeypatch
):
    # The old constant-output FakeTokenizer could never catch this regression.
    # Train a tiny real RustBPE tokenizer locally, without cache or downloads.
    tokenizer = RustBPETokenizer.train_from_iterator(
        iter(["café 日本語 🎉 Question Answer"]), vocab_size=256 + len(SPECIAL_TOKENS)
    )
    monkeypatch.setattr(converter, "get_tokenizer", lambda: tokenizer)
    mixture = _make_fake_mixture(4)
    mixture[0]["messages"][0]["content"] = "café 日本語 🎉"
    converter.convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=2)
    stored = pq.read_table(tmp_path / "shard_00000.parquet").to_pylist()[0]
    ids, _ = tokenizer.render_conversation(mixture[0])
    assert stored == {
        "original_index": 0,
        "text": tokenizer.decode(ids[1:]),
        "turn_char_starts": [0, tokenizer.decode(ids[1:]).index("<assistant_start>")],
    }
    shards_before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.glob("*.parquet")
    }
    converter.convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=2)
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.glob("*.parquet")
    } == shards_before
    mixture[0]["messages"][0]["content"] = "changed source with the same index map"
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="row content mismatch"):
        converter.convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=2)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("indices", [[False, True], [0.0, 1.0], [None, 1], ["0", "1"]])
def test_original_index_requires_integers_not_equal_coercions(tmp_path, indices):
    mixture = _make_fake_mixture(2)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "shard_00000.parquet"
    table = pq.read_table(path).set_column(1, "original_index", pa.array(indices))
    pq.write_table(table, path)
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="row coverage mismatch"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("completed_shards", -1, "outside the requested shard range"),
        ("completed_shards", 3, "outside the requested shard range"),
        ("max_conversations", True, "identity mismatch for max_conversations"),
        ("max_conversations", 2.0, "identity mismatch for max_conversations"),
        ("conversations_per_shard", 40000.0, "identity mismatch for conversations_per_shard"),
        ("mixture_fingerprint", {}, "fields mismatch"),
    ],
)
def test_malformed_resume_identity_fails_without_mutating_artifacts(tmp_path, field, value, error):
    mixture = _make_fake_mixture(2)
    _convert_with_fake_tokenizer(mixture, tmp_path, max_conversations=2)
    path = tmp_path / "conversion_meta.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match=error):
        _convert_with_fake_tokenizer(mixture, tmp_path, max_conversations=2)
    assert _snapshot(tmp_path) == before


def test_empty_legacy_progress_initializes_missing_identity_then_resumes(tmp_path):
    path = tmp_path / "conversion_meta.json"
    path.write_text(
        json.dumps(
            {
                "mixture": "unknown",
                "format": "full_nanochat_text",
                "completed_shards": 0,
                "total_shards": 1,
                "total_conversations": 2,
            }
        )
    )
    mixture = _make_fake_mixture(2)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    meta = json.loads(path.read_text())
    assert meta["completed_shards"] == 1
    assert meta["max_conversations"] is None
    assert meta["written_conversations"] == 2
    _convert_with_fake_tokenizer(mixture, tmp_path)
    assert json.loads(path.read_text()) == meta


@pytest.mark.parametrize("width", [1, 3, 20])
@pytest.mark.parametrize("publication", ["identity", "shard", "progress", "completion"])
@pytest.mark.parametrize("after_publish", [False, True])
def test_interrupted_conversion_matches_clean_artifacts(
    tmp_path, monkeypatch, width, publication, after_publish
):
    # Cover empty shards, a legitimate no-assistant skip, and an uneven final
    # range. Independent expected indices catch errors shared by both runs.
    mixture = _make_fake_mixture(7)
    mixture[0]["messages"] = [{"role": "user", "content": "no answer"}]
    mixture[2]["messages"][-1]["content"] = ""
    clean = tmp_path / "clean"
    resumed = tmp_path / "resumed"
    _convert_with_fake_tokenizer(mixture, clean, conversations_per_shard=width)
    real_json = converter.atomic_json_dump
    real_parquet = converter.atomic_write_parquet
    interrupted = False

    def publish_json(meta, path):
        nonlocal interrupted
        phase = (
            "identity"
            if meta["completed_shards"] == 0
            else "completion"
            if meta["completed_shards"] == meta["total_shards"]
            else "progress"
        )
        # A one-shard run has no intermediate progress publication.
        target = "completion" if width == 20 and publication == "progress" else publication
        fail = not interrupted and phase == target
        if not fail or after_publish:
            real_json(meta, path)
        if fail:
            interrupted = True
            raise RuntimeError("injected publication failure")

    def publish_parquet(table, path):
        nonlocal interrupted
        fail = not interrupted and publication == "shard"
        if not fail or after_publish:
            real_parquet(table, path)
        if fail:
            interrupted = True
            raise RuntimeError("injected publication failure")

    monkeypatch.setattr(converter, "atomic_json_dump", publish_json)
    monkeypatch.setattr(converter, "atomic_write_parquet", publish_parquet)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        _convert_with_fake_tokenizer(mixture, resumed, conversations_per_shard=width)
    monkeypatch.setattr(converter, "atomic_json_dump", real_json)
    monkeypatch.setattr(converter, "atomic_write_parquet", real_parquet)
    _convert_with_fake_tokenizer(mixture, resumed, conversations_per_shard=width)
    assert _snapshot(resumed) == _snapshot(clean)
    actual = [
        index
        for shard in sorted(resumed.glob("*.parquet"))
        for index in pq.read_table(shard)["original_index"].to_pylist()
    ]
    assert actual == [1, 3, 4, 5, 6]
    meta = json.loads((resumed / "conversion_meta.json").read_text())
    assert (meta["written_conversations"], meta["skipped_conversations"]) == (5, 2)


def test_empty_mixture_has_an_idempotent_zero_shard_conversion(tmp_path):
    mixture = _make_fake_mixture(0)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    before = _snapshot(tmp_path)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob("*.parquet"))
    meta = json.loads((tmp_path / "conversion_meta.json").read_text())
    assert meta["completed_shards"] == meta["total_shards"] == 0
    assert meta["written_conversations"] == meta["skipped_conversations"] == 0


def test_split_absent_metadata_cannot_be_adopted_as_validation(tmp_path):
    mixture = _make_fake_mixture(2)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "conversion_meta.json"
    meta = json.loads(path.read_text())
    del meta["split"]
    path.write_text(json.dumps(meta))
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="has no split metadata"):
        _convert_with_fake_tokenizer(mixture, tmp_path, split=SFTSplit.VAL)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("failed_name", ["conversion_meta.json", "shard_00000.parquet"])
def test_atomic_rename_failure_preserves_final_files_and_cleans_temporary_files(
    tmp_path, monkeypatch, failed_name
):
    mixture = _make_fake_mixture(2)
    clean = tmp_path / "clean"
    resumed = tmp_path / "resumed"
    _convert_with_fake_tokenizer(mixture, clean)
    real_replace = converter.os.replace
    failed = False

    def fail_replace(source, destination):
        nonlocal failed
        if not failed and str(destination).endswith(failed_name):
            failed = True
            raise OSError("injected atomic rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(converter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic rename failure"):
        _convert_with_fake_tokenizer(mixture, resumed)
    assert failed
    assert not list(resumed.glob("*.tmp.*"))
    assert not (resumed / "shard_00000.parquet").exists()
    if failed_name.endswith("parquet"):
        # Identity exists, but a failed rename never claims a committed shard.
        meta = json.loads((resumed / "conversion_meta.json").read_text())
        assert meta["completed_shards"] == 0
    else:
        assert not (resumed / "conversion_meta.json").exists()
    monkeypatch.setattr(converter.os, "replace", real_replace)
    _convert_with_fake_tokenizer(mixture, resumed)
    assert _snapshot(resumed) == _snapshot(clean)


@pytest.mark.parametrize("width", [1, 3, 10])
def test_exactly_half_skipped_is_accepted_including_empty_shards_on_resume(tmp_path, width):
    mixture = _make_fake_mixture(4)
    for index in (0, 1):
        mixture[index]["messages"][-1]["content"] = ""
    _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=width)
    before = _snapshot(tmp_path)
    _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=width)
    assert _snapshot(tmp_path) == before
    meta = json.loads((tmp_path / "conversion_meta.json").read_text())
    assert meta["completed_shards"] == meta["total_shards"]
    assert meta["written_conversations"] == meta["skipped_conversations"] == 2


def test_resume_reconstructs_derived_counts_from_audited_rows(tmp_path):
    mixture = _make_fake_mixture(3)
    mixture[0]["messages"][-1]["content"] = ""
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "conversion_meta.json"
    expected = json.loads(path.read_text())
    damaged = {**expected, "written_conversations": "unknown", "skipped_conversations": -100}
    path.write_text(json.dumps(damaged))
    _convert_with_fake_tokenizer(mixture, tmp_path)
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize("width", [2, 10])
def test_rejected_resume_revokes_completion_published_by_old_converter(tmp_path, width):
    mixture = _make_fake_mixture(4)
    for index in (0, 1, 2):
        mixture[index]["messages"][-1]["content"] = ""
    with pytest.raises(DataLoadError, match="Skipped 3/4"):
        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=width)
    path = tmp_path / "conversion_meta.json"
    meta = json.loads(path.read_text())
    # Reproduce the pre-review converter's marker ordering exactly.
    meta.update(
        completed_shards=meta["total_shards"], written_conversations=1, skipped_conversations=3
    )
    path.write_text(json.dumps(meta))
    shards_before = {p.name: p.read_bytes() for p in tmp_path.glob("*.parquet")}
    with pytest.raises(DataLoadError, match="Skipped 3/4"):
        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=width)
    meta = json.loads(path.read_text())
    assert meta["completed_shards"] == meta["total_shards"] - 1
    assert meta["written_conversations"] == 0
    assert meta["skipped_conversations"] == (2 if width == 2 else 0)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.parquet")} == shards_before
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _write_feature_shard(feature_dir)
    with pytest.raises(DataLoadError, match="conversion .* is incomplete"):
        load_sft_features(str(feature_dir), str(tmp_path))


def test_late_shard_corruption_is_rejected_before_republishing_any_progress(tmp_path):
    mixture = _make_fake_mixture(4)
    _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=2)
    path = tmp_path / "shard_00001.parquet"
    table = pq.read_table(path).set_column(0, "text", pa.array(["stale", "stale"]))
    pq.write_table(table, path)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    with pytest.raises(DataLoadError, match="row content mismatch.*shard_00001"):
        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=2)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("artifact_format", ["prompt_response", "structured_nanochat_rollout_v1"])
def test_legacy_resume_cannot_relabel_another_artifact_format(tmp_path, artifact_format):
    mixture = _make_fake_mixture(2)
    _convert_with_fake_tokenizer(mixture, tmp_path)
    path = tmp_path / "conversion_meta.json"
    meta = json.loads(path.read_text())
    meta["format"] = artifact_format
    path.write_text(json.dumps(meta))
    before = _snapshot(tmp_path)
    with pytest.raises(DataLoadError, match="SFT conversion requires 'full_nanochat_text'"):
        _convert_with_fake_tokenizer(mixture, tmp_path)
    assert _snapshot(tmp_path) == before
