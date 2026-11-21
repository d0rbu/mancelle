from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    evaluation_prefixes_for_scoring,
    read_evaluation_population,
    require_table_in_evaluation_population,
    write_evaluation_population,
)
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA


def _prefix(idx: int, *, concept_id: int = 1) -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id=f"p{idx}",
        concept_id=concept_id,
        mode=SteeringMode.POSITIVE,
        prefix_text=f"prefix {idx}",
        prefix_hash=f"h{idx}",
        prefix_tokens=[idx],
        concept_rich_sample_ids=[f"r{idx}"],
        concept_avoiding_sample_ids=[f"a{idx}"],
        concept_rich_output_tokens=[[10 + idx]],
        concept_avoiding_output_tokens=[[20 + idx]],
    )


def _candidate(
    *,
    core_prefix_id: str,
    rich_digest: str,
    avoid_digest: str,
    method_name: str = "diffmean_decoder_only",
) -> dict[str, Any]:
    return {
        "candidate_id": f"cand-{method_name}-{core_prefix_id}-{rich_digest}-{avoid_digest}",
        "attack_row_id": f"attack-{core_prefix_id}-{rich_digest}-{avoid_digest}",
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": f"rich-{rich_digest}",
        "concept_avoiding_sample_id": f"avoid-{avoid_digest}",
        "concept_id": 1,
        "mode": "positive",
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 0.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "prefix_text": core_prefix_id,
        "prefix_hash": f"h-{core_prefix_id}",
        "source_prompt_tokens": [1, 2],
        "source_prompt_token_digest": "src",
        "attacked_prompt_text": core_prefix_id,
        "attacked_prompt_tokens": [1, 2],
        "attacked_prompt_token_digest": "src",
        "concept_rich_output_tokens": [10],
        "concept_avoiding_output_tokens": [20],
        "concept_rich_output_digest": rich_digest,
        "concept_avoiding_output_digest": avoid_digest,
        "editable_token_start": 2,
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": None,
        "candidate_metadata_json": '{"candidate":"no_adversary_control"}',
    }


def test_evaluation_prefixes_for_scoring_uses_held_out_split() -> None:
    prefixes = [_prefix(i) for i in range(5)]
    scored = evaluation_prefixes_for_scoring(
        prefixes,
        modes={SteeringMode.POSITIVE},
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    _optimize, expected_held_out = split_prefixes_for_concept(
        prefixes, ratio=0.6, seed=0
    )

    assert len(scored) == 2
    assert {prefix.core_prefix_id for prefix in scored} == {
        prefix.core_prefix_id for prefix in expected_held_out
    }


def test_evaluation_prefixes_for_scoring_reports_single_prefix_groups() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="Offending groups"):
        evaluation_prefixes_for_scoring(
            [_prefix(1)],
            modes={SteeringMode.POSITIVE},
            ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
        )


def test_evaluation_population_requires_exact_pairs_not_just_prefixes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation_population.json"
    scored_row = _candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")
    write_evaluation_population(
        path,
        [scored_row],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    population = read_evaluation_population(path)
    table = pa.Table.from_pylist(
        [
            scored_row,
            _candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a2"),
        ],
        schema=CANDIDATES_SCHEMA,
    )

    with pytest.raises(AxBenchBpbAdvError, match="absent from the exact held-out pair keys"):
        require_table_in_evaluation_population(
            table,
            population=population,
            label="unit-test candidates",
        )

    covered = require_table_in_evaluation_population(
        pa.Table.from_pylist([scored_row], schema=CANDIDATES_SCHEMA),
        population=population,
        label="unit-test candidates",
    )
    assert covered.num_rows == 1


def test_missing_evaluation_population_raises(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="missing universal evaluation-population"):
        read_evaluation_population(tmp_path / "missing.json")


def test_unclustered_evaluation_population_preserves_main_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "feature_contract_fingerprint" not in payload
    assert len(path.read_bytes()) == 445
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "d16c6d82923c62c6f8ac227550fe675d9c5d209058fd75e7173ed0a46063e46e"
    )


def test_clustered_evaluation_population_is_owned_and_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation_population.json"
    row = _candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")
    fingerprint = "a" * 64
    ours = OursGcgConfig(split_ratio=0.6, split_seed=0)
    write_evaluation_population(
        path,
        [row],
        ours=ours,
        feature_contract_fingerprint=fingerprint,
    )
    original = path.read_bytes()
    first_stat = path.stat()

    read_evaluation_population(
        path,
        expected_feature_contract_fingerprint=fingerprint,
    )
    with pytest.raises(AxBenchBpbAdvError, match="unclustered reader rejects"):
        read_evaluation_population(path)
    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        read_evaluation_population(
            path,
            expected_feature_contract_fingerprint="b" * 64,
        )

    write_evaluation_population(
        path,
        [row],
        ours=ours,
        feature_contract_fingerprint=fingerprint,
    )
    second_stat = path.stat()
    assert path.read_bytes() == original
    assert (
        second_stat.st_dev,
        second_stat.st_ino,
        second_stat.st_size,
        second_stat.st_mtime_ns,
        second_stat.st_ctime_ns,
    ) == (
        first_stat.st_dev,
        first_stat.st_ino,
        first_stat.st_size,
        first_stat.st_mtime_ns,
        first_stat.st_ctime_ns,
    )

    changed = _candidate(
        core_prefix_id="p-held",
        rich_digest="r1",
        avoid_digest="a2",
    )
    with pytest.raises(AxBenchBpbAdvError, match="content changed on resume"):
        write_evaluation_population(
            path,
            [changed],
            ours=ours,
            feature_contract_fingerprint=fingerprint,
        )
    assert path.read_bytes() == original


def test_clustered_reader_rejects_unstamped_evaluation_population(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        read_evaluation_population(
            path,
            expected_feature_contract_fingerprint="a" * 64,
        )


def test_evaluation_population_rejects_prefix_pair_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["groups"][0]["scored_core_prefix_ids"].append("not-in-pairs")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="mismatched scored_core_prefix_ids"):
        read_evaluation_population(path)


def test_evaluation_population_rejects_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["n_pairs"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="declares n_pairs=2"):
        read_evaluation_population(path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", True),
        ("n_groups", -1),
        ("n_prefixes", "1"),
        ("n_pairs", 1.0),
    ],
)
def test_evaluation_population_rejects_invalid_integer_domains(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = invalid_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="corrupt or unreadable"):
        read_evaluation_population(path)


def test_evaluation_population_brands_bad_group_fields(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["groups"][0]["mode"] = "positve"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="corrupt or unreadable"):
        read_evaluation_population(path)


def test_evaluation_population_brands_missing_pair_fields(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_population.json"
    write_evaluation_population(
        path,
        [_candidate(core_prefix_id="p-held", rich_digest="r1", avoid_digest="a1")],
        ours=OursGcgConfig(split_ratio=0.6, split_seed=0),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["groups"][0]["scored_pairs"][0]["concept_avoiding_output_digest"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="corrupt or unreadable"):
        read_evaluation_population(path)
