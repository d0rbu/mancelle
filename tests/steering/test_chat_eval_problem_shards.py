"""Problem shards retain canonical inputs, real scoring, and complete cache admission."""

import hashlib
import json
from copy import deepcopy

import pytest
import torch as t
import yaml
from hydra import compose, initialize_config_dir

from scripts.eval import chat_eval
from scripts.eval import merge_chat_eval_target_real as merger
from steering.eval.context_cache_io import ContextCacheEntry, save_context_cache
from steering.eval.context_enumeration import TargetRealRole
from steering.eval.target_real_runner import build_context_feature_provider
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from tests.steering.test_chat_eval_turn_cache_integration import turn_main_fixture
from tests.steering.test_chat_eval_turn_config import CONFIG_DIR


@pytest.fixture(scope="module")
def shard_results_store():
    return []


@pytest.fixture
def shard_results(tmp_path_factory, shard_results_store):
    if shard_results_store:
        return shard_results_store[0]
    directory = tmp_path_factory.mktemp("problem-shards")
    with pytest.MonkeyPatch.context() as monkeypatch:
        fixture = turn_main_fixture(directory, monkeypatch, gsm_problems=3, cache_max_problems=3)
        tokenizer = fixture.tokenizer
        gsm = fixture.tasks["GSM8K"]
        prompts = [tuple(chat_eval.render_chat_completion(tokenizer, gsm[index]).token_ids) for index in range(3)]
        assert len(set(prompts)) == 3
        correct = gsm[0]["messages"][-1]["content"][-1]["text"]
        # Unequal shards with different accuracies distinguish weighted counts
        # from an average of shard accuracies. All logits still run the real model.
        scripts = {prompt: tokenizer.encode(correct if index == 0 else "#### -999") +
                   [tokenizer.encode_special("<assistant_end>")] for index, prompt in enumerate(prompts)}
        current_prompt = ()
        forwards = []
        original_forward = fixture.model.forward

        def capture_forward(ids, *args, **kwargs):
            nonlocal current_prompt
            cache = kwargs.get("kv_cache")
            if cache is not None and cache.get_pos() == 0:
                current_prompt = tuple(ids[0].tolist())
            logits = original_forward(ids, *args, **kwargs)
            assert t.isfinite(logits).all()
            forwards.append((ids.tolist(), hashlib.sha256(logits.detach().contiguous().numpy().tobytes()).hexdigest()))
            if cache is None or current_prompt not in scripts:
                return logits
            scripted = t.full_like(logits, -20.0)
            scripted[:, -1, scripts[current_prompt][cache.get_pos() - len(current_prompt)]] = 20.0
            return scripted

        monkeypatch.setattr(fixture.model, "forward", capture_forward)
        paths, payloads, traces = [], [], []
        for label, tasks, start, count in [
            ("first", "GSM8K", 0, 1), ("second", "GSM8K", 1, 2),
            ("other", "HumanEval", None, None), ("whole", "GSM8K|HumanEval", None, None),
            ("suffix", "GSM8K", 1, None),
            ("remaining", "ARC-Easy|ARC-Challenge|MMLU|SpellingBee", None, 1),
        ]:
            cfg = deepcopy(fixture.cfg)
            cfg.task_name, cfg.problem_start, cfg.max_problems = tasks, start, count
            cfg.max_new_tokens = 16
            cfg.target_real_output_dir = str(directory / label)
            forwards.clear()
            chat_eval.main.__wrapped__(cfg)
            path = directory / label / "target_real/target.yaml"
            paths.append(path)
            payloads.append(yaml.safe_load(path.read_text()))
            traces.append(list(forwards))
    shard_results_store.append((paths, payloads, traces))
    return paths, payloads, traces


def test_problem_shards_merge_equal_unsharded_yaml(tmp_path, shard_results):
    # kills: ignore problem_start (including a precheck-only implementation), unweighted merging.
    paths, payloads, traces = shard_results
    assert traces[0] + traces[1] + traces[2] == traces[3]
    assert traces[1] == traces[4]
    assert payloads[1] == payloads[4]
    assert payloads[0]["seeds"]["adhoc"]["results"]["GSM8K"] == 1.0
    assert payloads[1]["seeds"]["adhoc"]["results"]["GSM8K"] == 0.0
    assert payloads[3]["seeds"]["adhoc"]["results"]["GSM8K"] == 1 / 3
    output = tmp_path / "merged.yaml"
    result = merger.merge_target_real_results([paths[2], paths[1], paths[0]], output,
                                              task_totals={"GSM8K": 3, "HumanEval": 1})
    assert result == yaml.safe_load(output.read_text()) == payloads[3]
    assert result["seeds"]["adhoc"]["task_details"]["GSM8K"] == {
        "problem_start": 0, "max_problems": 3, "task_size": 3, "num_passed": 1, "total": 3,
    }
    assert "chatcore_metric" not in result["seeds"]["adhoc"]


@pytest.mark.parametrize("field", ["contexts_fingerprint", "shards", "input_provenance"])
def test_problem_shards_require_equal_cache_content_identity(tmp_path, shard_results, field):
    paths, payloads, _ = shard_results
    changed = deepcopy(payloads[1])
    cache = changed["seeds"]["adhoc"]["run_identity"]["cache"]
    if field == "contexts_fingerprint":
        cache[field] = "1" * 64
    elif field == "shards":
        cache[field][0]["sha256"] = "2" * 64
    else:
        cache[field]["feature_files"][0]["sha256"] = "3" * 64
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(changed))
    # All three cache fields legitimately share the one run-identity comparison; the
    # message names the block, not the field, so it cannot distinguish them further.
    with pytest.raises(ValueError, match="checkpoint identity / target config / conditioning / generation settings disagree"):
        merger.merge_target_real_results([paths[0], altered], tmp_path / "merged.yaml", task_totals={"GSM8K": 3})
    assert not (tmp_path / "merged.yaml").exists()


def test_shard_merge_hydra_and_then_across_tasks(tmp_path, shard_results):
    paths, payloads, _ = shard_results
    output = tmp_path / "merged.yaml"
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="merge_chat_eval_target_real", overrides=[
            f"input_files={json.dumps([str(path) for path in paths[:2]])}",
            "task_totals={GSM8K:3}", f"output_file={output}",
        ])
    merger.main.__wrapped__(cfg)
    result = merger.merge_target_real_results([output, paths[2]], tmp_path / "combined.yaml",
                                              task_totals={"GSM8K": 3, "HumanEval": 1})
    assert result == payloads[3]
    complete = merger.merge_target_real_results([tmp_path / "combined.yaml", paths[5]], tmp_path / "all.yaml")
    body = complete["seeds"]["adhoc"]
    assert set(body["results"]) == set(chat_eval.ALL_CHAT_TASKS)
    assert body["results"]["GSM8K"] == 1 / 3
    assert body["chatcore_metric"] == chat_eval.compute_chatcore_metric(body["results"])


@pytest.mark.parametrize("case,message", [
    # `overlap` and `gap` share the interval-walk guard and both first disagree at
    # problem 1; `missing-first` reaches the same guard at problem 0.
    ("overlap", "GSM8K: overlapping or missing problem ranges at 1"),
    ("gap", "GSM8K: overlapping or missing problem ranges at 1"),
    ("missing-first", "GSM8K: overlapping or missing problem ranges at 0"),
    ("missing-last", r"GSM8K: ranges cover \[0, 1\), expected \[0, 3\)"),
    ("missing-task", r"cannot compute full ChatCORE; missing tasks: \['HumanEval'\]"),
    ("wrong-total", r"GSM8K: ranges cover \[0, 3\), expected \[0, 2\)"),
    ("extra-task", "input tasks disagree with declared task_totals"),
    ("undeclared", r"duplicate tasks: \['GSM8K'\]"),
])
def test_merge_rejects_invalid_shard_coverage(tmp_path, shard_results, case, message):
    paths, payloads, _ = shard_results
    inputs = list(paths[:3])
    totals = {"GSM8K": 3, "HumanEval": 1}
    if case == "overlap":
        # Keep sum(total)==declared N: only the interval check detects overlap.
        changed = deepcopy(payloads[1])
        changed["seeds"]["adhoc"]["task_details"]["GSM8K"]["problem_start"] = 0
        path = tmp_path / "overlap.yaml"
        path.write_text(yaml.safe_dump(changed))
        inputs[1] = path
    elif case == "gap":
        changed = deepcopy(payloads[1])
        row = changed["seeds"]["adhoc"]["task_details"]["GSM8K"]
        row.update(problem_start=2, max_problems=1, total=1)
        path = tmp_path / "gap.yaml"
        path.write_text(yaml.safe_dump(changed))
        inputs[1] = path
    elif case == "missing-first":
        inputs.pop(0)
    elif case == "missing-last":
        inputs.pop(1)
    elif case == "missing-task":
        inputs.pop(2)
    elif case == "wrong-total":
        totals["GSM8K"] = 2
    elif case == "extra-task":
        del totals["HumanEval"]
    else:
        totals = None
    output = tmp_path / "merged.yaml"
    output.write_bytes(b"previous result\n")
    with pytest.raises(ValueError, match=message):
        merger.merge_target_real_results(inputs, output, task_totals=totals)
    assert output.read_bytes() == b"previous result\n"


@pytest.mark.parametrize("field,value,message", [
    # In-domain values that break the range arithmetic share the consistency guard;
    # bools/floats/negatives share the domain guard ahead of it.
    ("num_passed", 3, "GSM8K: invalid problem range/counts"),
    ("num_passed", True, "GSM8K: range/counts require nonnegative integers"),
    ("total", 1, "GSM8K: invalid problem range/counts"),
    ("total", 2.0, "GSM8K: range/counts require nonnegative integers"),
    ("max_problems", 0, "GSM8K: invalid problem range/counts"),
    ("problem_start", -1, "GSM8K: range/counts require nonnegative integers"),
    ("problem_start", 2, "GSM8K: invalid problem range/counts"),
    ("task_size", 4, "GSM8K: task sizes disagree"),
    ("num_passed", 1, "GSM8K: accuracy disagrees with num_passed / total"),
])
def test_merge_rejects_inconsistent_range_counts(tmp_path, shard_results, field, value, message):
    paths, payloads, _ = shard_results
    changed = deepcopy(payloads[1])
    changed["seeds"]["adhoc"]["task_details"]["GSM8K"][field] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match=message):
        merger.merge_target_real_results([paths[0], path, paths[2]], tmp_path / "merged.yaml",
                                         task_totals={"GSM8K": 3, "HumanEval": 1})


@pytest.mark.parametrize("section,key,value", [
    ("checkpoint", "model_sha256", "0" * 64), ("generation", "max_new_tokens", 17),
    ("conditioning", "turn_context_contract_sha256", "0" * 64),
])
def test_merge_rejects_shard_identity_mismatch(tmp_path, shard_results, section, key, value):
    paths, payloads, _ = shard_results
    changed = deepcopy(payloads[1])
    changed["seeds"]["adhoc"]["run_identity"][section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    # All three sections legitimately share the one run-identity comparison.
    with pytest.raises(ValueError, match="checkpoint identity / target config / conditioning / generation settings disagree"):
        merger.merge_target_real_results([paths[0], path, paths[2]], tmp_path / "merged.yaml",
                                         task_totals={"GSM8K": 3, "HumanEval": 1})


@pytest.mark.parametrize("totals", [{}, {"GSM8K": 0}, {"GSM8K": True}, {"GSM8K": 3.0}, {"unknown": 3}])
def test_merge_rejects_invalid_task_totals(tmp_path, shard_results, totals):
    paths, _, _ = shard_results
    # Empty, zero, bool, float and unknown-task totals share the one declaration guard.
    with pytest.raises(ValueError, match="task_totals must declare known tasks with positive integer totals"):
        merger.merge_target_real_results(paths[:3], tmp_path / "merged.yaml", task_totals=totals)


def test_merge_requires_counted_task_receipt(tmp_path, shard_results):
    paths, payloads, _ = shard_results
    changed = deepcopy(payloads[1])
    del changed["seeds"]["adhoc"]["task_details"]
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match=r"task_details requires exactly these fields: \['GSM8K'\]"):
        merger.merge_target_real_results([paths[0], path, paths[2]], tmp_path / "merged.yaml",
                                         task_totals={"GSM8K": 3, "HumanEval": 1})


@pytest.mark.parametrize("role", list(TargetRealRole))
@pytest.mark.parametrize("later_config", [False, True])
def test_cache_coverage_fails_before_any_forward(tmp_path, monkeypatch, role, later_config):
    fixture = turn_main_fixture(tmp_path, monkeypatch, gsm_problems=3, cache_max_problems=3)
    provider = build_context_feature_provider(fixture.cfg, cache_dir=fixture.cache_cfg.output_dir,
                                             turn_contract=fixture.contract)
    entries = []
    for record in fixture.records:
        if record.task == "GSM8K" and record.example_index == 2 and record.metadata["target_real_role"] == role:
            continue
        segments = provider.get(record.token_ids)
        assert segments is not None
        entries.append(ContextCacheEntry(record.token_ids, segments,
                                        SegmentMode.turn_delimited_fixed_tokens))
    missing_cache = tmp_path / "missing" / "c2048" / fixture.cache.name
    metadata = json.loads((fixture.cache / "manifest.json").read_text())["metadata"]
    save_context_cache(missing_cache, entries, identity=provider.identity,
        n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4), metadata=metadata,
        source_segment_mode=SegmentMode.turn_delimited_fixed_tokens, turn_contract=fixture.contract)
    if later_config:
        fixture.cfg.target_real_configs.append({"name": "incomplete", "mode": "pooled", "cache_dir": str(missing_cache)})
    else:
        fixture.cfg.target_real_configs[0].cache_dir = str(missing_cache)
    # HumanEval is complete; GSM8K's first selected problem is also complete.
    # Either lazy per-task or lazy per-problem checking would already forward.
    fixture.cfg.task_name = "HumanEval|GSM8K"
    fixture.cfg.max_problems = None
    forwards = []
    original_forward = fixture.model.forward

    def capture_forward(*args, **kwargs):
        forwards.append(args)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(fixture.model, "forward", capture_forward)
    with pytest.raises(RuntimeError, match=f"problem 2 missing {role.value} cache row before evaluation"):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert forwards == []
    assert not (fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").exists()


@pytest.mark.parametrize("start,count,message", [
    # Negative/bool/float starts share the problem_start domain guard, and the four
    # invalid counts share the max_problems domain guard; the three in-domain ranges
    # must each name their own interval in the task-size guard.
    (-1, 1, "problem_start must be null or a nonnegative integer"),
    (True, 1, "problem_start must be null or a nonnegative integer"),
    (0.5, 1, "problem_start must be null or a nonnegative integer"),
    (3, 1, r"GSM8K: problem range \[3, 4\) exceeds task size 3"),
    (1, 3, r"GSM8K: problem range \[1, 4\) exceeds task size 3"),
    (None, 4, r"GSM8K: problem range \[0, 4\) exceeds task size 3"),
    (0, 0, "max_problems must be null or a positive integer"),
    (0, -1, "max_problems must be null or a positive integer"),
    (0, True, "max_problems must be null or a positive integer"),
    (0, 1.5, "max_problems must be null or a positive integer"),
])
def test_turn_rejects_invalid_problem_ranges_before_forward(tmp_path, monkeypatch, start, count, message):
    fixture = turn_main_fixture(tmp_path, monkeypatch, gsm_problems=3, cache_max_problems=3)
    fixture.cfg.task_name = "GSM8K"
    fixture.cfg.problem_start, fixture.cfg.max_problems = start, count
    monkeypatch.setattr(fixture.model, "forward", lambda *a, **kw: pytest.fail("invalid range reached forward"))
    # Ranges are validated when the selection is built in main(), before any provider or forward.
    with pytest.raises(ValueError, match=message):
        chat_eval.main.__wrapped__(fixture.cfg)


@pytest.mark.parametrize("start", [0, 1, -1])
def test_fixed_path_rejects_problem_start_before_model_load(tmp_path, monkeypatch, start):
    fixture = turn_main_fixture(tmp_path, monkeypatch)
    metadata = {"sft_segment_mode": SegmentMode.fixed_tokens.value}
    fixture.checkpoint_path.write_text(json.dumps(metadata))
    fixture.cfg.problem_start = start
    monkeypatch.setattr(chat_eval, "load_model", lambda *a, **kw: pytest.fail("fixed range reached model load"))
    with pytest.raises(ValueError, match="problem_start is only supported on the turn path"):
        chat_eval.main.__wrapped__(fixture.cfg)
