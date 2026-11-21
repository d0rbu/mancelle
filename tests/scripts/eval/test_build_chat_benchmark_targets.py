"""Native preparation boundaries and independent numerical fitting oracles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from scripts.eval import build_chat_benchmark_targets as builder
from scripts.eval import chat_eval
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import MissingPercentileResolver
from steering.eval.chat_generation import render_chat_completion
from steering.eval.dense_feature_targets import dense_diffmean_from_means, dense_mean_percentiles
from steering.eval.target_conditioning import token_context_key
from tasks import arc, gsm8k, humaneval, mmlu, smoltalk
from tasks.spellingbee import SpellingBee
from tests.scripts.eval.test_axbench_project_methods import _consolidation
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.test_chat_eval_self_conditioned import fixed_fixture, selfcond_turn_fixture
from tests.steering.test_chat_eval_turn_cache_integration import local_chat_tasks


def conversation(prompt, answer="reference"):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


class SourceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


def example(name, split, index, *, key=None, answer="reference"):
    prompt = f"{name} {split} distinct input {index}"
    return builder.TrainingExample(f"{name}/{split}/{index}", key or "question:" + prompt.casefold(),
                                   answer, conversation(prompt, answer))


@pytest.fixture
def populations(tmp_path, monkeypatch):
    tasks = local_chat_tasks(tmp_path, monkeypatch, gsm_problems=2)
    tasks["HumanEval"].ds = Dataset.from_list([{**tasks["HumanEval"].ds[0],
        "prompt": 'def f():\n    """Return the fixed integer two."""\n'}])
    fit, calibration = {}, {}
    for name in builder.ALL_CHAT_TASKS:
        fit[name] = [example(name, "fit", i) for i in range(40)]
        calibration[name] = [example(name, "heldout", i) for i in range(3)]
    background = [conversation(f"Unrelated background input {i}") for i in range(5)]
    return tasks, fit, calibration, background


def prepare(populations, **kwargs):
    return builder.prepare_populations(*populations, tokenizer=ByteChatTokenizer(),
        source_tokenizer=SourceTokenizer(), window=4096, positive_cap=3, background_count=2, seed=42, **kwargs)


def test_full_test_population_and_calibration_are_independent_of_caps(populations):
    positive, negative, prepared, population = prepare(populations)
    assert all(len(rows) == 3 for rows in positive.values())
    assert len(negative) == 2
    for name, task in populations[0].items():
        assert population[name] == [token_context_key(render_chat_completion(ByteChatTokenizer(), task[i]).token_ids)
                                    for i in range(len(task))]
    accepted = prepared["report"]["accepted"]
    heldout = {r["input_key"] for name in builder.ALL_CHAT_TASKS for r in accepted[f"{name}/calibration"]}
    fit = {r["input_key"] for name in builder.ALL_CHAT_TASKS for r in accepted[f"{name}/fit"]}
    assert not heldout & fit
    assert len(accepted["GSM8K/calibration"]) > 3
    assert prepare(populations) == (positive, negative, prepared, population)


def test_full_cross_task_test_and_heldout_exclusion_precede_cap(populations):
    tasks, fit, calibration, _ = populations
    forbidden = builder.task_input_keys(tasks["GSM8K"], 1)[0]
    fit["MMLU"][:0] = [example("MMLU", "leak", 0, key=forbidden),
        example("MMLU", "leak", 1, key=calibration["ARC-Easy"][0].input_key)]
    _, _, prepared, _ = prepare(populations)
    reasons = {r["source_id"]: r["reason"] for r in prepared["report"]["excluded"]}
    assert reasons.get("MMLU/leak/0") == "evaluation_overlap"
    assert reasons.get("MMLU/leak/1") == "calibration_overlap"
    assert [r["source_id"] for r in prepared["report"]["accepted"]["MMLU/fit"]] == [f"MMLU/fit/{i}" for i in range(3)]


def test_conflicting_answers_reject_entire_input_group_but_word_group_allows_different_questions(populations):
    _, fit, _, _ = populations
    bad = example("MMLU", "conflict", 0)
    changed = builder.TrainingExample("MMLU/conflict/1", bad.input_key, "wrong", conversation(bad.conversation["messages"][0]["content"], "wrong"))
    fit["MMLU"][:0] = [bad, changed]
    a = example("ARC-Easy", "same-word", 0, key="word:shared")
    b = example("ARC-Easy", "same-word", 1, key="word:shared", answer="different letter count")
    fit["ARC-Easy"][:0] = [a, b]
    _, _, prepared, _ = prepare(populations)
    reasons = {r["source_id"]: r["reason"] for r in prepared["report"]["excluded"]}
    assert reasons[bad.source_id] == reasons[changed.source_id] == "conflicting_references"
    assert a.source_id not in reasons
    assert reasons[b.source_id] == "duplicate_input"


def test_background_excludes_uncapped_candidates_reviewed_coding_and_both_turns(populations):
    _, fit, _, background = populations
    candidate = fit["MMLU"][-1].input_key.split(":", 1)[1]
    background[:0] = [conversation("clean user", "Quoted: " + candidate), conversation("Reviewed coding duplicate"),
                      conversation("Unrelated background input 0")]
    _, negatives, prepared, _ = prepare(populations, excluded_coding_inputs=("code:reviewed coding duplicate",))
    reasons = {r["source_id"]: r["reason"] for r in prepared["report"]["excluded"]}
    assert reasons["background/train/0"] == reasons["background/train/1"] == "source_overlap"
    assert reasons["background/train/3"] == "duplicate_input"
    assert all(candidate not in text for text in negatives)


def test_background_uses_only_first_complete_exchange_and_preserves_system(populations):
    background = populations[3]
    first = [{"role": "system", "content": "System contract"}, *background[0]["messages"]]
    background[0] = {"messages": [*first, {"role": "user", "content": "LATER USER"},
                                  {"role": "assistant", "content": "LATER ANSWER"}]}
    _, negatives, _, _ = prepare(populations)
    assert "System contract" in negatives[0]
    assert "LATER" not in negatives[0]


def test_insufficient_background_and_empty_class_fail(populations):
    populations[3].clear()
    with pytest.raises(ValueError, match="insufficient eligible background"):
        prepare(populations)
    populations[1]["MMLU"].clear()
    with pytest.raises(ValueError, match="no eligible fit documents for MMLU"):
        prepare(populations)


def test_native_render_preserves_complete_reference_and_exact_bos_inclusive_window():
    tokenizer = ByteChatTokenizer()
    sample = conversation("Unicode café", "Z" * 4096)
    text = builder.fitting_text(sample, tokenizer, SourceTokenizer(), 10000)
    assert text is not None
    assert text.startswith("<user_start>")
    assert text.endswith("<assistant_end>")
    assert "Z" * 4096 in text
    assert "<|bos|>" not in text
    size = len(text.encode()) + 1
    assert builder.fitting_text(sample, tokenizer, SourceTokenizer(), size) == text
    assert builder.fitting_text(sample, tokenizer, SourceTokenizer(), size - 1) is None


@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "unfinished"}],
                                      [{"role": "assistant", "content": ""}]])
def test_incomplete_reference_rejected(messages):
    with pytest.raises(ValueError, match="complete nonempty"):
        builder.fitting_text({"messages": messages}, ByteChatTokenizer(), SourceTokenizer(), 4096)


def test_unicode_normalization_and_word_partition():
    assert builder.normalize_input("  ＡB\nCafé\t") == "ab café"
    values = [builder.calibration_group("SpellingBee", f"word:word-{i}", 42) for i in range(1000)]
    assert 150 < sum(values) < 250
    assert values == [builder.calibration_group("SpellingBee", f"word:word-{i}", 42) for i in range(1000)]
    assert values != [builder.calibration_group("SpellingBee", f"word:word-{i}", 43) for i in range(1000)]


@pytest.mark.parametrize("module,args", [(mmlu, ("all", "dev")), (arc, ("ARC-Easy", "train")),
    (gsm8k, ("main", "train")), (humaneval, ()), (smoltalk, ("train",))])
def test_optional_revision_is_forwarded_without_changing_default_load_call(monkeypatch, module, args):
    calls = []
    monkeypatch.setattr(module, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or Dataset.from_dict({"dummy": [0]}))
    cls = {mmlu: mmlu.MMLU, arc: arc.ARC, gsm8k: gsm8k.GSM8K, humaneval: humaneval.HumanEval, smoltalk: smoltalk.SmolTalk}[module]
    cls(*args)
    cls(*args, revision="b" * 40)
    assert "revision" not in calls[0][1]
    assert calls[1] == (calls[0][0], {**calls[0][1], "revision": "b" * 40})


@pytest.mark.parametrize("files", [[], [{"filename": "/cache/" + "a" * 40 + "/data.arrow"}],
    [{"filename": "/cache/" + "b" * 40 + "/data.arrow"}, {"filename": "/cache/" + "a" * 40 + "/indices.arrow"}]])
def test_offline_revision_fallback_cannot_be_mislabeled(files):
    with pytest.raises(ValueError, match="requested revision"):
        builder.validate_dataset_revision(SimpleNamespace(cache_files=files), "b" * 40)


def test_dataset_and_shuffle_cache_attest_same_revision():
    builder.validate_dataset_revision(SimpleNamespace(cache_files=[
        {"filename": "/cache/" + "b" * 40 + "/data.arrow"},
        {"filename": "/cache/" + "b" * 40 + "/indices.arrow"}]), "b" * 40)


def test_input_exclusion_never_reads_humaneval_solutions_or_tests(populations):
    task = populations[0]["HumanEval"]
    before = builder.task_input_keys(task, 0)
    task.ds = Dataset.from_list([{**task.ds[0], "canonical_solution": "poison solution", "test": "poison test"}])
    assert builder.task_input_keys(task, 0) == before
    assert all("poison" not in value for value in before)


def test_spelling_identity_uses_word_before_any_augmentation(populations):
    task = populations[0]["SpellingBee"]
    assert isinstance(task, SpellingBee)
    for index in range(len(task)):
        _, word = task._example_rng_and_word(index)
        assert builder.task_input_keys(task, index) == ("word:" + builder.normalize_input(word),)
        assert word in task[index]["messages"][0]["content"]


def test_mbpp_reference_is_executed_and_only_prompt_code_enter_fit(monkeypatch):
    row = {"task_id": 900, "prompt": "Compute an integer", "code": "def f(): return 7",
           "test_imports": ["import math"], "test_list": ["assert f() == 7"]}
    calls = []
    monkeypatch.setattr(builder, "execute_code", lambda program: calls.append(program) or SimpleNamespace(success=True))
    examples, excluded = builder.mbpp_examples([row], "train", {})
    assert calls == ["import math\ndef f(): return 7\nassert f() == 7"]
    assert examples[0].conversation == conversation(row["prompt"], row["code"])
    assert not excluded
    skipped, excluded = builder.mbpp_examples([row], "train", {"900": "reviewed overlap"})
    assert not skipped and len(calls) == 1
    assert excluded == [{"source_id": "mbpp/sanitized/train/900", "input_key": "code:compute an integer", "reason": "reviewed overlap"}]
    monkeypatch.setattr(builder, "execute_code", lambda program: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="reference failed qualification"):
        builder.mbpp_examples([row], "train", {})


def test_native_training_adapters_preserve_conversations_and_correct_choice_text(populations):
    for name, task in populations[0].items():
        rows = builder.training_examples(task, name)
        assert len(rows) == len(task)
        for index, row in enumerate(rows):
            assert row.conversation == task[index]
            assert row.source_id == f"{name}/{index}"
        if name == "MMLU":
            assert rows[0].answer_key == "one"
            assert rows[0].conversation["messages"][-1]["content"] == "A"
        if name.startswith("ARC"):
            assert rows[0].answer_key == "no"
            assert rows[0].conversation["messages"][-1]["content"] == "B"


def test_all_source_adapters_use_only_approved_splits_and_pins(populations, monkeypatch):
    revisions = dict.fromkeys(("mmlu", "arc", "gsm8k", "humaneval", "mbpp", "background"), "a" * 40)
    calls, checked = [], []
    for module in (mmlu, arc, gsm8k, humaneval):
        original = module.load_dataset
        def load(*args, _original=original, **kwargs):
            calls.append((args, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, "load_dataset", load)
    def background(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset.from_list([conversation("background")])
    monkeypatch.setattr(smoltalk, "load_dataset", background)
    def mbpp(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset.from_list([{"task_id": 1, "prompt": "constant seven", "code": "def f(): return 7",
                                  "test_imports": [], "test_list": ["assert f() == 7"]}])
    monkeypatch.setattr(builder, "load_dataset", mbpp)
    monkeypatch.setattr(builder, "validate_dataset_revision", lambda ds, pin: checked.append(pin))
    evaluation, fit, calibration, bg, excluded = builder.load_sources(revisions, {})
    assert set(evaluation) == set(fit) == set(calibration) == set(builder.ALL_CHAT_TASKS)
    assert len(fit["SpellingBee"]) == 10_000 and len(evaluation["SpellingBee"]) == 256
    assert calibration["SpellingBee"] == calibration["GSM8K"] == []
    assert len(checked) == len(calls) == 15
    assert all(kwargs["revision"] == "a" * 40 for _, kwargs in calls)
    splits = {}
    for args, kwargs in calls:
        splits.setdefault(args[0], []).append(kwargs.get("split", "test"))
    assert splits == {"allenai/ai2_arc": ["test", "test", "train", "validation", "train", "validation"],
        "cais/mmlu": ["test", "dev", "validation"], "openai/gsm8k": ["test", "train"],
        "openai/openai_humaneval": ["test"], "google-research-datasets/mbpp": ["train", "validation"],
        "HuggingFaceTB/smol-smoltalk": ["train"]}
    assert not excluded and len(bg) == 1


@pytest.mark.parametrize("clustered", [False, True])
def test_dense_reduction_real_percentile_table_independent_scalar_oracle(clustered):
    # Rows intentionally alternate sibling dominance: max(mean) is the wrong operation.
    values = t.tensor([[0, 10, 99, 90, 40], [0, 80, 99, 20, 60]], dtype=t.float32)
    table = t.arange(1, 100, dtype=t.float64).repeat(5, 1)
    allowed = t.tensor([1, 3, 4])
    result = dense_mean_percentiles(values, percentile_table=table, allowed_ids=allowed,
        consolidation=_consolidation() if clustered else None, missing=MissingPercentileResolver.error())
    # Linear grid: every in-range activation is its own percentile rank.
    from_scalar = []
    for row in values.tolist():
        ranks = [0.5 if value < 1 else 99.5 if value >= 99 else value for value in row]
        from_scalar.append([0, max(ranks[1], ranks[3]), 0, 0, ranks[4]] if clustered
                           else [0, ranks[1], 0, ranks[3], ranks[4]])
    expected = t.tensor(from_scalar, dtype=t.float64).mean(0)
    assert t.equal(result, expected)


def test_shared_background_extracted_once_and_targets_gather_exact_head_order(monkeypatch):
    background = ["background-a", "background-b"]
    positives = {name: [name + "-a", name + "-b"] for name in builder.ALL_CHAT_TASKS}
    calls = []
    matrices = {tuple(background): t.tensor([[10., 20., 30., 40.], [20., 30., 40., 50.]])}
    for offset, texts in enumerate(positives.values(), 1):
        matrices[tuple(texts)] = matrices[tuple(background)] + offset
    def extract(texts, *args, **kwargs):
        calls.append(tuple(texts))
        assert kwargs["batch_size"] == 8
        return matrices[tuple(texts)]
    monkeypatch.setattr(builder, "_batched_dense_sae_max_pool", extract)
    result = builder.fit_targets(positives, background, model=object(), sae=object(), sae_model="gemma-3-4b-pt",
        sae_layer=22, percentile_table=t.arange(1, 100, dtype=t.float64).repeat(4, 1), allowed_ids=t.tensor([1, 3]),
        consolidation=None, missing=MissingPercentileResolver.error(),
        feature_vocab=FeatureVocab(n_sae_features=4, original_ids=(1, 3)), batch_size=8)
    assert calls == [tuple(background), *map(tuple, positives.values())]
    for offset, name in enumerate(builder.ALL_CHAT_TASKS, 1):
        assert t.equal(result[name], t.full((2,), offset / 100, dtype=t.float32))
        assert result[name].device.type == "cpu" and result[name].is_contiguous()


@pytest.mark.parametrize("bad", [t.empty(0, 5), t.zeros(3), t.zeros(2, 4), t.full((2, 5), float("nan")), t.full((2, 5), float("inf"))])
@pytest.mark.parametrize("clustered", [False, True])
def test_dense_reduction_rejects_invalid_extractor_output(bad, clustered):
    with pytest.raises(ValueError):
        dense_mean_percentiles(bad, percentile_table=t.ones(5, 99), allowed_ids=t.arange(5),
            consolidation=_consolidation() if clustered else None, missing=MissingPercentileResolver.error())


@pytest.mark.parametrize("positive,negative", [(t.ones(2), t.ones(3)), (t.tensor([float("nan")]), t.zeros(1)),
    (t.tensor([100.]), t.zeros(1)), (t.tensor([-100.]), t.zeros(1)), (t.empty(0), t.empty(0))])
def test_final_target_rejects_invalid_shape_nonfinite_and_closed_endpoints(positive, negative):
    with pytest.raises(ValueError):
        dense_diffmean_from_means(positive, negative, feature_vocab=None, label="test")


class ExplicitBosSource:
    """Give the existing source fixture the production encode keyword contract."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def encode(self, text, *, add_special_tokens=True):
        ids = self.source.encode(text)
        return ids if add_special_tokens else ids[1:]


@pytest.mark.parametrize("family,clustered,raw_pretrain", [("fixed", False, False), ("fixed", True, False),
    ("fixed", False, True), ("fixed", True, True), ("turn", False, False), ("turn", True, False)])
def test_offline_builder_dry_run_fit_bundle_then_real_evaluator(tmp_path, monkeypatch, family, clustered, raw_pretrain):
    fixture = (fixed_fixture if family == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch, clustered=clustered)
    if raw_pretrain:
        # record_fixed_cadence preserves raw pretraining metadata; model loading
        # has not yet promoted these fields when the offline builder reads it.
        meta = json.loads(fixture.checkpoint_path.read_text())
        meta.pop("num_buckets")
        meta.pop("n_sae_features")
        fixture.checkpoint_path.write_text(json.dumps(meta))
    tasks = {name: chat_eval._chat_task(name) for name in chat_eval.ALL_CHAT_TASKS}
    tasks["HumanEval"].ds = Dataset.from_list([{**tasks["HumanEval"].ds[0],
        "prompt": 'def f():\n    """Return two."""\n'}])
    fit = {name: [example(name, "fit", i) for i in range(40)] for name in tasks}
    heldout = {name: [example(name, "heldout", i) for i in range(3)] for name in tasks}
    backgrounds = [conversation(f"Unrelated background {i}") for i in range(3)]
    monkeypatch.setattr(builder, "load_sources", lambda *a: (tasks, fit, heldout, backgrounds, []))
    monkeypatch.setattr(builder, "get_tokenizer", lambda: fixture.tokenizer)
    source = SourceTokenizer() if family == "fixed" else ExplicitBosSource(fixture.source)
    monkeypatch.setattr(builder, "get_extraction_tokenizer", lambda *a: source)
    cfg = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_chat_benchmark_targets.yaml")
    assert isinstance(cfg, DictConfig)
    cfg.checkpoint_meta_path = str(fixture.checkpoint_path)
    cfg.output_dir = str(tmp_path / "builder")
    cfg.positive_cap, cfg.background_count, cfg.device = 3, 2, "cpu"
    cfg.stats_path = fixture.cfg.stats_path
    monkeypatch.setattr(builder, "load_dense_extraction_stack", lambda **kw: pytest.fail("dry run loaded model"))
    preparation = builder.run(cfg)
    assert preparation.name == "preparation.json"
    before = preparation.read_bytes()
    corpus = json.loads((preparation.parent / "corpus.json").read_text())
    assert len(corpus["background"]) == 2 and all(len(rows) == 3 for rows in corpus["fit"].values())
    assert not (preparation.parent / "targets.safetensors").exists()
    source_model = SimpleNamespace(cfg=SimpleNamespace(n_ctx=4096), eval=lambda: None)
    sae = t.nn.Linear(1, 1)
    monkeypatch.setattr(builder, "load_dense_extraction_stack", lambda **kw: (source_model, sae))
    def extract(texts, *args, **kwargs):
        return t.tensor([[value / 256 for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts])
    monkeypatch.setattr(builder, "_batched_dense_sae_max_pool", extract)
    cfg.dry_run = False
    bundle = builder.run(cfg)
    assert preparation.read_bytes() == before
    assert bundle.is_file()
    assert builder.run(cfg).read_bytes() == bundle.read_bytes()
    fixture.cfg.self_conditioned.benchmark_steering = {"targets": str(bundle), "strength": 2.5}
    fixture.cfg.task_name = "MMLU"
    chat_eval.main.__wrapped__(fixture.cfg)
    assert (Path(fixture.cfg.self_conditioned_output_dir) / "self_conditioned/result.yaml").exists()


@pytest.mark.parametrize("key,value", [("positive_cap", 0), ("positive_cap", True), ("background_count", -1),
    ("extraction_batch_size", 1.5), ("partition_seed", True), ("dry_run", "yes"),
    ("revisions", []), ("revisions", {}), ("revisions.gsm8k", "main"),
    ("coding_exclusions", []), ("coding_exclusions", {"9": ""})])
def test_invalid_builder_configuration_fails_before_data_or_models(monkeypatch, key, value):
    cfg = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_chat_benchmark_targets.yaml")
    assert isinstance(cfg, DictConfig)
    OmegaConf.update(cfg, key, value, merge=False)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: pytest.fail("invalid config reached tokenizer"))
    with pytest.raises(ValueError):
        builder.run(cfg)
