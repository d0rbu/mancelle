"""CPU invariants for frozen pairs, pinned populations and campaign admission."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from steering.eval import campaign2 as contracts
from steering.eval import campaign2_training_pairs as pairs
from steering.eval.harmfulness_safety_targets import (
    SAFETY_EVALUATION_SOURCE_NAMES,
    SAFETY_SOURCE_SPECS,
    SafetySourceName,
)
from tests.scripts.campaign2_fixtures import campaign_config as config
from tests.scripts.campaign2_fixtures import export_fixture


def training_rows(n: int = 6) -> list[dict]:
    return [{"prompt": f"Prompt {i}", "response": f"{'Safe' if safe else 'Unsafe'} {i}",
             "is_safe": safe} for i in range(n) for safe in (True, False)]


def source_metadata(training: list, evaluation: list) -> dict:
    return {SafetySourceName.TRAINING: pairs.TRAINING_SPEC.metadata(len(training)),
            **{name: SAFETY_SOURCE_SPECS[name].metadata(len(evaluation))
               for name in SAFETY_EVALUATION_SOURCE_NAMES}}


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    training, evaluation = training_rows(), [{"prompt": "held out"}]
    sources = source_metadata(training, evaluation)
    monkeypatch.setattr(pairs, "load_evaluation", lambda: (evaluation, {
        key: value for key, value in sources.items() if key != SafetySourceName.TRAINING}))
    pairs.write_pairs(tmp_path, training, evaluation, sources)
    return tmp_path / pairs.PAIR_JSONL


def edit_manifest(path: Path, key: str, value: object) -> None:
    sidecar = pairs.manifest_path(path)
    data = json.loads(sidecar.read_text())
    data[key] = value
    sidecar.write_text(json.dumps(data))


def test_first_native_responses_and_seed42_prompt_order() -> None:
    training = training_rows()
    training.insert(1, {"prompt": "Prompt 0", "response": "Later safe", "is_safe": True})
    selected, counts = pairs.pair_rows(training, [{"prompt": "held out"}])
    assert [row["prompt"] for row in selected] == [f"Prompt {i}" for i in (3, 1, 2, 4, 0, 5)]
    assert next(row for row in selected if row["prompt"] == "Prompt 0")["safe_response"] == "Safe 0"
    assert counts["rows_seen"] == 13 and counts["prompts_seen"] == counts["paired"] == 6
    assert pairs.pair_rows(training, [{"prompt": "held out"}]) == (selected, counts)


def test_normalized_exclusion_precedes_pairing_and_cap() -> None:
    training = training_rows()
    # All three normalization operations matter; exclude even unpaired prompts.
    training.append({"prompt": "ＨＥＬＤ\tOUT", "response": "A", "is_safe": True})
    selected, counts = pairs.pair_rows(training, [{"prompt": "  ＰＲＯＭＰＴ\n0 "},
                                                {"prompt": "held out"}], cap=2)
    assert counts["excluded_eval"] == 2 and counts["unpaired"] == 0
    assert counts["paired"] == 5 and counts["selected"] == 2 and counts["capped"] == 3
    assert len(selected) == 2 and all(row["prompt"] != "Prompt 0" for row in selected)


def test_dedupe_keeps_first_eligible_exact_prompt_without_cross_pairing() -> None:
    training = training_rows(1) + [
        {"prompt": " ＰＲＯＭＰＴ  0", "response": "Other safe", "is_safe": True},
        {"prompt": " ＰＲＯＭＰＴ  0", "response": "Other unsafe", "is_safe": False},
        {"prompt": "Variant", "response": "A", "is_safe": True},
        {"prompt": "VARIANT", "response": "B", "is_safe": False},
    ]
    selected, counts = pairs.pair_rows(training, [{"prompt": "held out"}])
    assert selected == [{"prompt": "Prompt 0", "safe_response": "Safe 0", "unsafe_response": "Unsafe 0"}]
    assert counts["duplicate_prompts"] == 1 and counts["unpaired"] == 2


def test_identical_first_responses_are_excluded_without_substitution() -> None:
    training = training_rows(1) + [
        {"prompt": "Contradiction", "response": "Same", "is_safe": True},
        {"prompt": "Contradiction", "response": "Same", "is_safe": False},
        {"prompt": "Contradiction", "response": "Later different", "is_safe": False},
    ]
    selected, counts = pairs.pair_rows(training, [{"prompt": "held out"}])
    assert len(selected) == 1 and counts["identical_responses"] == 1


@pytest.mark.parametrize("cap", [0, -1, 129, True, 1.5, "128", None])
def test_invalid_cap_refused_before_io(cap: object) -> None:
    with pytest.raises(ValueError, match="cap"):
        pairs.pair_rows([], [], cap=cap)


def test_cap_is_exact_and_counters_reconcile() -> None:
    selected, counts = pairs.pair_rows(training_rows(130), [{"prompt": "held out"}])
    assert len(selected) == 128 and counts["capped"] == 2 and counts["paired"] == 130
    assert counts["prompts_seen"] == 130 and counts["rows_seen"] == 260


@pytest.mark.parametrize("training,evaluation", [
    ([], [{"prompt": "Q"}]),
    ([{"prompt": "Q", "response": "A", "is_safe": True}], [{"prompt": "Other"}]),
    (training_rows(1), [{"prompt": "Prompt 0"}]),
])
def test_empty_selection_refused(training: list, evaluation: list) -> None:
    with pytest.raises(ValueError, match="1..128"):
        pairs.pair_rows(training, evaluation)


@pytest.mark.parametrize("evaluation", [[], [{}], [{"prompt": " \n"}], [{"prompt": 42}]])
def test_bad_evaluation_refused(evaluation: list) -> None:
    with pytest.raises(ValueError, match="evaluation"):
        pairs.pair_rows(training_rows(1), evaluation)


@pytest.mark.parametrize("field,value", [
    ("is_safe", None), ("is_safe", 1), ("is_safe", "True"),
    ("prompt", " "), ("prompt", None), ("response", ""), ("response", 3),
])
def test_bad_training_row_refused(field: str, value: object) -> None:
    training = training_rows(1)
    training[0][field] = value
    with pytest.raises(ValueError, match="training row"):
        pairs.pair_rows(training, [{"prompt": "held out"}])


@pytest.mark.parametrize("rows", [
    [], [{}], [None], [{"prompt": "Q", "safe_response": "A"}],
    [{"prompt": "Q", "safe_response": "A", "unsafe_response": "B", "extra": "no"}],
    [{"prompt": "Q", "safe_response": "A", "unsafe_response": "A"}],
    [{"prompt": "Q", "safe_response": "A", "unsafe_response": " A "}],
    [{"prompt": "Q", "safe_response": " \n", "unsafe_response": "B"}],
    [{"prompt": "", "safe_response": "A", "unsafe_response": "B"}],
    [{"prompt": "Q", "safe_response": "A", "unsafe_response": None}],
    [{"prompt": "Q", "safe_response": "A", "unsafe_response": "B"},
     {"prompt": " Ｑ ", "safe_response": "C", "unsafe_response": "D"}],
])
def test_jsonl_schema_refusals(tmp_path: Path, rows: list) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        pairs.safety_pairs(path)


def test_blank_jsonl_line_and_oversize_refused(frozen: Path) -> None:
    with pytest.raises(ValueError):
        pairs.decode_pairs(frozen.read_bytes() + b"\n")
    rows = [{"prompt": str(i), "safe_response": "A", "unsafe_response": "B"} for i in range(129)]
    with pytest.raises(ValueError, match="1..128"):
        pairs.validate_pairs(rows)


def test_manifest_binds_exact_bytes_and_all_counts(frozen: Path) -> None:
    identity = pairs.validate_training_pairs(frozen)
    assert identity["safety_training_pairs"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert identity["safety_training_pairs_manifest"] == hashlib.sha256(pairs.manifest_path(frozen).read_bytes()).hexdigest()
    frozen.write_bytes(frozen.read_bytes().replace(b"Safe 0", b"Safe X"))
    with pytest.raises(ValueError, match="hash mismatch"):
        pairs.validate_training_pairs(frozen)


@pytest.mark.parametrize("key,value,message", [
    ("format", "other", "contract"), ("rule", "other", "contract"),
    ("seed", 41, "contract"), ("seed", 42.0, "contract"),
    ("row_count", 7, "row count"), ("row_count", 6.0, "row count"),
    ("cap", 1, "row count"), ("counts", {}, "counts"),
    ("sources", {}, "provenance"), ("evaluation_overlap", 1, "overlap"),
    ("evaluation_overlap", False, "overlap"), ("evaluation_prompt_count", 2, "overlap"),
])
def test_manifest_tampering_refused(frozen: Path, key: str, value: object, message: str) -> None:
    edit_manifest(frozen, key, value)
    with pytest.raises(ValueError, match=message):
        pairs.validate_training_pairs(frozen)


def test_overlap_recomputed_even_when_hash_and_manifest_claim_valid(frozen: Path) -> None:
    frozen.write_bytes(frozen.read_bytes().replace(b"Prompt 0", b"  HELD  OUT "))
    edit_manifest(frozen, "jsonl_sha256", hashlib.sha256(frozen.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="overlap"):
        pairs.validate_training_pairs(frozen)


def test_missing_manifest_refused(frozen: Path) -> None:
    pairs.manifest_path(frozen).unlink()
    with pytest.raises(FileNotFoundError):
        pairs.validate_training_pairs(frozen)


def test_preflight_accepts_and_binds_frozen_pairs(frozen: Path) -> None:
    checkpoint = export_fixture(frozen.parent)
    cfg = config(frozen.parent, variant="pythia_diffmean", checkpoint_dir=str(checkpoint),
                 device="cpu", safety={"training_pairs": str(frozen)})
    identity = contracts.preflight(cfg)
    expected = pairs.validate_training_pairs(frozen)
    assert all(identity["checkpoint_files"][key] == value for key, value in expected.items())


@pytest.mark.parametrize("failure", ["schema", "count", "hash", "overlap", "missing"])
def test_preflight_rejects_invalid_artifact(frozen: Path, failure: str) -> None:
    checkpoint = export_fixture(frozen.parent)
    cfg = config(frozen.parent, variant="pythia_diffmean", checkpoint_dir=str(checkpoint),
                 device="cpu", safety={"training_pairs": str(frozen)})
    if failure == "schema":
        frozen.write_text('{}\n')
    elif failure == "count":
        edit_manifest(frozen, "row_count", 999)
    elif failure == "hash":
        frozen.write_bytes(frozen.read_bytes().replace(b"Safe 0", b"Other safe"))
    elif failure == "overlap":
        frozen.write_bytes(frozen.read_bytes().replace(b"Prompt 0", b"held out"))
        edit_manifest(frozen, "jsonl_sha256", hashlib.sha256(frozen.read_bytes()).hexdigest())
    else:
        pairs.manifest_path(frozen).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        contracts.preflight(cfg)


def test_builder_config_and_idempotence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training, evaluation = training_rows(), [{"prompt": "held out"}]
    sources = source_metadata(training, evaluation)
    monkeypatch.setattr(pairs, "load_source", lambda spec: training)
    monkeypatch.setattr(pairs, "load_evaluation", lambda: (evaluation, {
        key: value for key, value in sources.items() if key != SafetySourceName.TRAINING}))
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs/eval"), version_base=None):
        cfg = compose(config_name="campaign2_training_pairs", overrides=[f"output_dir={tmp_path}", "cap=2"])
    pairs.main.__wrapped__(cfg)
    path = tmp_path / pairs.PAIR_JSONL
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    pairs.main.__wrapped__(cfg)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert len(pairs.safety_pairs(path)) == 2
    cfg.cap = 3
    with pytest.raises(ValueError, match="differs"):
        pairs.main.__wrapped__(cfg)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_pinned_dataset_revision_checked_not_just_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def load(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(cache_files=[{"filename": "/cache/wrong/data.arrow"}])

    monkeypatch.setattr(pairs, "load_dataset", load)
    with pytest.raises(ValueError, match="attest requested revision"):
        pairs.load_source(pairs.TRAINING_SPEC)
    assert calls == [((pairs.TRAINING_SPEC.dataset, pairs.TRAINING_SPEC.config),
                      {"split": "330k_train", "revision": pairs.TRAINING_SPEC.revision})]

@pytest.mark.parametrize("payload", [
    b'{"prompt":"Q","prompt":"Hidden","safe_response":"A","unsafe_response":"B"}\n',
    b'{"prompt":"Q","safe_response":"A","unsafe_response":NaN}\n',
    b'not JSON\n',
])
def test_strict_jsonl_refuses_duplicate_keys_and_nonstandard_values(payload: bytes) -> None:
    with pytest.raises(ValueError):
        pairs.decode_pairs(payload)


@pytest.mark.parametrize("field,value", [("paired", -1), ("selected", True), ("capped", 99),
                                         ("prompts_seen", 1), ("rows_seen", 1)])
def test_manifest_count_invariants(frozen: Path, field: str, value: object) -> None:
    counts = json.loads(pairs.manifest_path(frozen).read_text())["counts"]
    counts[field] = value
    edit_manifest(frozen, "counts", counts)
    with pytest.raises(ValueError, match="count"):
        pairs.validate_training_pairs(frozen)
