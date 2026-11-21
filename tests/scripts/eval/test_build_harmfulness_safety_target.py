"""Held-out safe-reference preparation and immutable dense target production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from scripts.eval import build_chat_benchmark_targets as shared
from scripts.eval import build_harmfulness_safety_target as builder
from steering.eval import harmfulness_safety_targets as safety_targets
from steering.eval.chat_generation import render_chat_completion
from steering.eval.harmfulness_safety_targets import (
    BACKGROUND_REVISION,
    SAFETY_RENDER_REDUCTION,
    SAFETY_SOURCE_SPECS,
    UNRECORDED_SOURCE_WEIGHT_REVISION,
    ExclusionReason,
    PromptHarmLabel,
    ResponseRefusalLabel,
    SafeResponseLabel,
    SafetySourceName,
    SafetySubpopulation,
    load_safety_targets,
    safety_fit_selection,
)
from steering.eval.target_conditioning import token_context_key
from steering.immutable_io import ImmutableArtifactError
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import SegmentMode
from tests.scripts.eval.test_build_chat_benchmark_targets import ExplicitBosSource, SourceTokenizer
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.test_chat_eval_self_conditioned import fixed_fixture, selfcond_turn_fixture


def safe_row(prompt="Unique safe training prompt", response="A safe reference response.", label=SafeResponseLabel.SAFE,
             prompt_label=PromptHarmLabel.HARMFUL, refusal_label=ResponseRefusalLabel.REFUSAL):
    return {"prompt": prompt, "response": response, "response_harm_label": label,
            "prompt_harm_label": prompt_label, "response_refusal_label": refusal_label}


@pytest.mark.parametrize("prompt_label,response_label,refusal_label,expected", [
    (PromptHarmLabel.HARMFUL, SafeResponseLabel.SAFE, ResponseRefusalLabel.REFUSAL, SafetySubpopulation.SAFE_HARMFUL),
    (PromptHarmLabel.HARMFUL, SafeResponseLabel.SAFE, ResponseRefusalLabel.NON_REFUSAL, SafetySubpopulation.SAFE_HARMFUL),
    (PromptHarmLabel.HARMFUL, SafeResponseLabel.SAFE, None, SafetySubpopulation.SAFE_HARMFUL),
    (PromptHarmLabel.BENIGN, SafeResponseLabel.SAFE, ResponseRefusalLabel.NON_REFUSAL, SafetySubpopulation.BENIGN_NON_REFUSAL),
    (PromptHarmLabel.BENIGN, SafeResponseLabel.SAFE, ResponseRefusalLabel.REFUSAL, ExclusionReason.BENIGN_REFUSAL),
    (PromptHarmLabel.HARMFUL, SafeResponseLabel.UNSAFE, ResponseRefusalLabel.NON_REFUSAL, ExclusionReason.UNSAFE),
    (PromptHarmLabel.BENIGN, SafeResponseLabel.UNSAFE, ResponseRefusalLabel.NON_REFUSAL, ExclusionReason.UNSAFE),
    (None, SafeResponseLabel.SAFE, ResponseRefusalLabel.NON_REFUSAL, ExclusionReason.UNLABELLED_PROMPT),
    (PromptHarmLabel.BENIGN, SafeResponseLabel.SAFE, None, ExclusionReason.UNLABELLED_REFUSAL),
    (PromptHarmLabel.HARMFUL, None, None, ExclusionReason.UNLABELLED),
])
def test_v2_label_selection(prompt_label, response_label, refusal_label, expected):
    row = safe_row(prompt_label=prompt_label, label=response_label, refusal_label=refusal_label)
    assert safety_fit_selection(row) is expected


@pytest.mark.parametrize("prompt_label", [*PromptHarmLabel, None])
@pytest.mark.parametrize("response_label", [*SafeResponseLabel, None])
@pytest.mark.parametrize("refusal_label", [*ResponseRefusalLabel, None])
def test_complete_annotation_product_matches_independent_eligibility_rule(prompt_label, response_label, refusal_label):
    eligible = response_label == SafeResponseLabel.SAFE and (
        prompt_label == PromptHarmLabel.HARMFUL or
        (prompt_label == PromptHarmLabel.BENIGN and refusal_label == ResponseRefusalLabel.NON_REFUSAL)
    )
    selected = safety_fit_selection(safe_row(prompt_label=prompt_label, label=response_label, refusal_label=refusal_label))
    assert isinstance(selected, SafetySubpopulation) is eligible


@pytest.mark.parametrize("field", ["prompt_harm_label", "response_harm_label", "response_refusal_label"])
@pytest.mark.parametrize("value", ["unknown", False, 42, ""])
def test_unknown_labels_fail_even_for_excluded_rows(field, value):
    training, evaluation, background = populations()
    row = safe_row(evaluation[0]["prompt"], label=SafeResponseLabel.UNSAFE)
    row[field] = value
    training.append(row)
    with pytest.raises(ValueError, match=f"unknown {field}"):
        prepare((training, evaluation, background), positive_cap=1)


@pytest.mark.parametrize("field", ["prompt_harm_label", "response_harm_label", "response_refusal_label"])
def test_missing_label_field_fails(field):
    row = safe_row()
    del row[field]
    with pytest.raises(ValueError, match=f"missing {field}"):
        safety_fit_selection(row)


def test_natural_seed42_order_and_subpopulation_counts():
    _, evaluation, background = populations()
    training = [safe_row(f"Training prompt {i}", prompt_label=PromptHarmLabel.BENIGN if i % 2 else PromptHarmLabel.HARMFUL,
                         refusal_label=ResponseRefusalLabel.NON_REFUSAL) for i in range(16)]
    training.append(safe_row("Benign refusal", prompt_label=PromptHarmLabel.BENIGN))
    _, report, _ = prepare((training, evaluation, background), positive_cap=8)
    # Freeze the seed-42 corpus order independently of the producer's RNG and
    # seed constant. Original row 16 is a benign refusal, so it cannot use a slot.
    expected = [12, 11, 6, 10, 9, 7, 3, 0]
    assert [r["source_id"] for r in report["accepted"]["safety"]] == [f"wildguardtrain/train/{i}" for i in expected]
    assert report["subpopulation_counts"] == {
        SafetySubpopulation.SAFE_HARMFUL: sum(i % 2 == 0 for i in expected),
        SafetySubpopulation.BENIGN_NON_REFUSAL: sum(i % 2 == 1 for i in expected)}
    assert report["exclusion_counts"][ExclusionReason.BENIGN_REFUSAL] == 1


def populations():
    return ([safe_row()], [{"prompt": "Held out evaluation prompt"}, {"prompt": "Held out evaluation prompt"}],
            [builder.conversation(f"Independent background topic {index}", "A helpful background answer.")
             for index in range(4)])


def source_provenance():
    return {name.value: spec.metadata(4 if name is SafetySourceName.BACKGROUND else 1)
            for name, spec in SAFETY_SOURCE_SPECS.items()}


def fitting_provenance(contract):
    """Complete synthetic extraction receipt for a checkpoint-backed test bundle."""
    source = contract["source"]
    turn = contract["segment_mode"] == SegmentMode.turn_delimited_fixed_tokens.value
    config = ({field: source["source_lineage"][field]
               for field in ("model", "release", "activation_type", "layer", "width", "l0")}
              if turn else source["sae"])
    release = SAERelease(config["release"])
    resolved = resolve_sae(model=config["model"], release=release, activation_type=ActivationType(config["activation_type"]),
        layer=config["layer"], width=SAEWidth(config["width"]), l0=coerce_config_l0(release, config["l0"]))
    return {"checkpoint_metadata_sha256": "c" * 64,
            "extraction_tokenizer_checksum": source["extraction_tokenizer_checksum"] if turn else "d" * 64,
            "sae": config, "source_weight_revision": UNRECORDED_SOURCE_WEIGHT_REVISION,
            "extraction_batch_size": 1, "source_window": source["source_context_window"] if turn else 4096,
            "render_reduction": SAFETY_RENDER_REDUCTION,
            "corpus_sha256": "e" * 64,
            "code": {"runtime_version": "fixture", "git_commit": "f" * 40, "source_sha256": "a" * 64},
            "backend": "transformerlens", "device": "cpu",
            "resolved_sae": {"release": resolved.release, "sae_id": resolved.sae_id, "hook_name": resolved.hook_name}}


def prepare(data, **overrides):
    options: dict[str, Any] = {"tokenizer": ByteChatTokenizer(), "source_tokenizer": SourceTokenizer(),
               "window": 4096, "positive_cap": 2, "background_count": 2}
    return builder.prepare_populations(*data, **(options | overrides))


@pytest.mark.parametrize("duplicate", ["ＡＢＣ Café", "abc CAFE\u0301", "  ABC\tCafé\n"])
def test_normalized_evaluation_exclusion_precedes_selection(duplicate):
    training, _, background = populations()
    training.insert(0, safe_row(duplicate))
    corpus, report, _ = prepare((training, [{"prompt": "abc café"}], background), positive_cap=1)
    assert report["exclusion_counts"] == {"evaluation_overlap": 1}
    assert report["evaluation_input_keys"] == ["abc café"]
    assert report["accepted"]["safety"][0]["source_id"] == "wildguardtrain/train/1"
    assert len(corpus["fit"]["safety"]) == 1


def test_entire_evaluation_population_and_label_audit_survive_fit_cap():
    training, evaluation, background = populations()
    evaluation += [{"prompt": "  FINAL held-out prompt  "}]
    training += [safe_row("Capped training candidate"), safe_row("final HELD-OUT prompt", label="harmful"),
                 safe_row("Unsafe training candidate", label="harmful"), safe_row("Unlabelled candidate", label=None)]
    _, report, population = prepare((training, evaluation, background), positive_cap=1)
    assert report["evaluation_rows"] == 3
    assert report["evaluation_input_keys"] == ["final held-out prompt", "held out evaluation prompt"]
    assert report["exclusion_counts"] == {"positive_cap": 1, "evaluation_overlap": 1,
                                          "unsafe_response": 1, "unlabelled_response": 1}
    tokenizer = ByteChatTokenizer()
    expected = {token_context_key(render_chat_completion(tokenizer, builder.conversation(text, "")).token_ids)
                for row in evaluation for text in (row["prompt"], row["prompt"].strip())}
    assert population == {"safety": sorted(expected)}


def test_only_safe_nonempty_responses_and_unique_normalized_inputs_enter_fit():
    _, evaluation, background = populations()
    training = [safe_row("Unsafe candidate", label="harmful"), safe_row("Missing label", label=None),
                safe_row("Empty response", response=""), safe_row("Whitespace response", response=" \t"),
                safe_row("Absent response", response=None), safe_row("Ａccepted prompt"),
                safe_row(" accepted\tPROMPT ", response="A second safe answer."), safe_row("Another allowed prompt")]
    corpus, report, _ = prepare((training, evaluation, background))
    assert {row["input_key"] for row in report["accepted"]["safety"]} == {"accepted prompt", "another allowed prompt"}
    assert report["exclusion_counts"] == {"unsafe_response": 1, "unlabelled_response": 1,
                                          "empty_response": 3, "duplicate_input": 1}
    assert len(corpus["fit"]["safety"]) == 2
    assert prepare((training, evaluation, background)) == (corpus, report, _)


def test_fit_text_preserves_real_chat_template_and_hashes_prompt_plus_response():
    training, evaluation, background = populations()
    corpus, report, _ = prepare((training, evaluation, background))
    text = corpus["fit"]["safety"][0]
    expected = "<user_start>Unique safe training prompt<user_end><assistant_start>A safe reference response.<assistant_end>"
    assert text == expected
    accepted = report["accepted"]["safety"][0]
    assert accepted["text_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert accepted["response_sha256"] == hashlib.sha256(training[0]["response"].encode()).hexdigest()
    assert accepted["response_harm_label"] == "unharmful"


def test_overlength_reference_is_excluded_without_truncation_or_consuming_cap():
    training, evaluation, background = populations()
    training.append(safe_row("Very long reference", response="x" * 300))
    corpus, report, _ = prepare((training, evaluation, background), window=160, positive_cap=1)
    assert report["exclusion_counts"] == {"source_overlength": 1}
    assert report["accepted"]["safety"][0]["source_id"] == "wildguardtrain/train/0"
    assert "x" * 300 not in corpus["fit"]["safety"][0]


def test_shared_background_excludes_all_candidates_even_capped_and_unsafe():
    training, evaluation, background = populations()
    training += [safe_row("Candidate beyond safety cap"), safe_row("Unsafe excluded candidate", label="harmful")]
    background[:0] = [builder.conversation("Clean background question", "candidate BEYOND safety cap"),
                      builder.conversation("Other background question", "Unsafe excluded candidate"),
                      builder.conversation("Held out evaluation prompt", "Background answer.")]
    corpus, report, _ = prepare((training, evaluation, background), positive_cap=1)
    assert report["background_exclusion_counts"] == {"source_overlap": 3}
    assert len(corpus["background"]) == 2
    assert builder.prepare_background is shared.prepare_background
    assert builder.fit_targets is shared.fit_targets
    reference = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_chat_benchmark_targets.yaml")
    assert BACKGROUND_REVISION == reference.revisions.background


def test_background_exact_message_exclusion_does_not_reject_single_letters_or_quoted_fragments():
    training, evaluation, background = populations()
    training += [safe_row("i", label="harmful"), safe_row("Candidate beyond cap")]
    background.insert(0, builder.conversation("Unrelated question", "Quoted fragment: Candidate beyond cap"))
    corpus, report, _ = prepare((training, evaluation, background), positive_cap=1)
    assert len(corpus["background"]) == 2
    assert "Quoted fragment: Candidate beyond cap" in corpus["background"][0]
    assert report["background_exclusion_counts"] == {}
    assert report["background_overlap_policy"] == shared.BackgroundOverlapPolicy.EXACT_MESSAGE.value


def test_shared_background_rejects_unknown_overlap_policy():
    with pytest.raises(ValueError, match="unknown background overlap policy"):
        shared.prepare_background(populations()[2], forbidden=(), tokenizer=ByteChatTokenizer(),
            source_tokenizer=SourceTokenizer(), window=4096, count=1,
            overlap_policy=cast(shared.BackgroundOverlapPolicy, "unrecognized"))


@pytest.mark.parametrize("training,evaluation", [([], [{"prompt": "Held out"}]),
    ([safe_row(label="harmful")], [{"prompt": "Held out"}]),
    ([safe_row("Held out")], [{"prompt": "Held out"}]), ([safe_row()], [])])
def test_empty_or_fully_overlapping_fit_population_fails(training, evaluation):
    with pytest.raises(ValueError, match="eligible safe|exclusion population"):
        prepare((training, evaluation, populations()[2]))


@pytest.mark.parametrize("field,value", [("prompt", 3),
    ("response", 3), ("response_harm_label", "safe"), ("response_harm_label", False)])
def test_malformed_training_rows_fail_loud(field, value):
    training, evaluation, background = populations()
    training[0][field] = value
    with pytest.raises(ValueError):
        prepare((training, evaluation, background))


@pytest.mark.parametrize("prompt", [None, "", " \t"])
def test_missing_training_prompts_are_explicitly_audited(prompt):
    training, evaluation, background = populations()
    training.insert(0, safe_row(prompt))
    _, report, _ = prepare((training, evaluation, background))
    assert report["exclusion_counts"] == {"empty_prompt": 1}
    assert report["excluded"] == [{"source_id": "wildguardtrain/train/0", "reason": "empty_prompt"}]
    assert report["accepted"]["safety"][0]["source_id"] == "wildguardtrain/train/1"


@pytest.mark.parametrize("prompt", [None, "", " \t"])
@pytest.mark.parametrize("response", [3, False, [], {}])
def test_blank_prompt_cannot_hide_malformed_response(prompt, response):
    training, evaluation, background = populations()
    training.insert(0, safe_row(prompt, response=response))
    with pytest.raises(ValueError, match="response must be text or absent"):
        prepare((training, evaluation, background))


@pytest.mark.parametrize("prompt", ["", " \n", None, 3])
def test_malformed_evaluation_prompt_fails(prompt):
    with pytest.raises(ValueError, match="evaluation prompt"):
        prepare((populations()[0], [{"prompt": prompt}], populations()[2]))


def test_insufficient_clean_background_fails():
    with pytest.raises(ValueError, match="insufficient eligible background"):
        prepare((populations()[0], populations()[1], []))


@pytest.mark.parametrize("key", ["window", "positive_cap", "background_count"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_preparation_limits_require_positive_integers(key, value):
    with pytest.raises(ValueError, match=key):
        prepare(populations(), **{key: value})


class AttestedRows(list):
    def __init__(self, rows, revision):
        super().__init__(rows)
        self.cache_files = [{"filename": f"/cached/config/0.0.0/{revision}/data.arrow"}]


@pytest.mark.parametrize("stale_source", [None, "training", "beaver", "wildguard", "background"])
def test_source_splits_and_revision_attestation_include_all_four_populations(monkeypatch, stale_source):
    calls = []

    def load_dataset(dataset, *args, split, revision):
        name = ("beaver" if dataset == "PKU-Alignment/BeaverTails-Evaluation" else
                "training" if args == ("wildguardtrain",) else "wildguard")
        calls.append((dataset, args, split, revision))
        rows = [safe_row()] if name == "training" else [{"prompt": f"Evaluation prompt {name}"}]
        return AttestedRows(rows, "b" * 40 if name == stale_source else revision)

    def background(split, *, revision):
        calls.append(("HuggingFaceTB/smol-smoltalk", (), split, revision))
        return SimpleNamespace(ds=AttestedRows(populations()[2], "b" * 40 if stale_source == "background" else revision))

    monkeypatch.setattr(safety_targets, "load_dataset", load_dataset)
    monkeypatch.setattr(safety_targets, "SmolTalk", background)
    if stale_source:
        with pytest.raises(ValueError, match="attest requested revision"):
            builder.load_target_populations()
        return
    training, evaluation, _, provenance = builder.load_target_populations()
    assert len(training) == 1 and len(evaluation) == 2
    assert calls == [(spec.dataset,
                      () if name in (SafetySourceName.EVALUATION_BEAVERTAILS, SafetySourceName.BACKGROUND) else (spec.config,),
                      spec.split, spec.revision) for name, spec in SAFETY_SOURCE_SPECS.items()]
    assert set(provenance) == {"training", "evaluation_beavertails", "evaluation_wildguard", "background"}
    assert all(len(source["revision"]) == 40 for source in provenance.values())


@pytest.mark.parametrize("family,clustered", [("fixed", False), ("turn", False), ("turn", True)])
def test_real_checkpoint_contract_dry_run_fit_and_immutable_replay(tmp_path, monkeypatch, family, clustered):
    fixture = (fixed_fixture if family == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch, clustered=clustered)
    data = populations()
    monkeypatch.setattr(builder, "load_target_populations", lambda: (*data, source_provenance()))
    monkeypatch.setattr(shared, "get_tokenizer", lambda: fixture.tokenizer)
    source = SourceTokenizer() if family == "fixed" else ExplicitBosSource(fixture.source)
    monkeypatch.setattr(shared, "get_extraction_tokenizer", lambda *a: source)
    cfg = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_harmfulness_safety_target.yaml")
    assert isinstance(cfg, DictConfig)
    assert cfg.extraction_batch_size == 1
    cfg.checkpoint_meta_path, cfg.stats_path = str(fixture.checkpoint_path), fixture.cfg.stats_path
    cfg.output_dir, cfg.device, cfg.positive_cap, cfg.background_count = str(tmp_path / "builder"), "cpu", 1, 2
    monkeypatch.setattr(builder, "load_dense_extraction_stack", lambda **kw: pytest.fail("dry run loaded a model"))
    preparation = builder.run(cfg)
    corpus_path = preparation.parent / "corpus.json"
    before = (preparation.read_bytes(), corpus_path.read_bytes())
    assert preparation.name == "preparation.json" and corpus_path.is_file()
    assert not (preparation.parent / "targets.safetensors").exists()
    manifest = json.loads(before[0])
    summary = json.loads((preparation.parent / builder.TargetArtifact.AUDIT_SUMMARY).read_text())
    assert summary["subpopulation_counts"] == manifest["preparation"]["subpopulation_counts"]
    assert summary["accepted_counts"] == {"safety": 1, "background": 2}
    assert manifest["fit_contract"]["statistics_sha256"]
    mode = SegmentMode.fixed_tokens if family == "fixed" else SegmentMode.turn_delimited_fixed_tokens
    assert manifest["fit_contract"]["segment_mode"] == mode.value
    nnterp_backend = t.nn.Dropout() if clustered else None
    source_model = SimpleNamespace(cfg=SimpleNamespace(n_ctx=4096), eval=lambda: None,
        nnterp_model=SimpleNamespace(model=nnterp_backend) if nnterp_backend is not None else None)
    monkeypatch.setattr(builder, "load_dense_extraction_stack", lambda **kw: (source_model, t.nn.Linear(1, 1)))

    def extract(texts, *args, **kwargs):
        assert kwargs["batch_size"] == 1
        return t.tensor([[value / 256 for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts])

    monkeypatch.setattr(shared, "_batched_dense_sae_max_pool", extract)
    cfg.dry_run = False
    target_path = builder.run(cfg)
    assert (preparation.read_bytes(), corpus_path.read_bytes()) == before
    tensors = load_file(target_path)
    assert set(tensors) == {"feature_ids", "target.safety"}
    bundle = load_safety_targets(target_path, expected_contract=manifest["fit_contract"], expected_feature_ids=tensors["feature_ids"],
        tokenizer=fixture.tokenizer)
    if nnterp_backend is not None:
        assert not nnterp_backend.training
    assert bundle.sha256 == hashlib.sha256(target_path.read_bytes()).hexdigest()
    assert builder.run(cfg).read_bytes() == target_path.read_bytes()
    data[0][0]["response"] += " Changed reference."
    with pytest.raises(ImmutableArtifactError):
        builder.run(cfg)


@pytest.mark.parametrize("key,value", [("positive_cap", 0), ("positive_cap", True),
    ("background_count", -1), ("extraction_batch_size", 1.5), ("dry_run", "yes")])
def test_invalid_builder_config_fails_before_environment_or_datasets(monkeypatch, key, value):
    cfg = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_harmfulness_safety_target.yaml")
    assert isinstance(cfg, DictConfig)
    OmegaConf.update(cfg, key, value)
    monkeypatch.setattr(builder, "resolve_fitting_environment", lambda *a: pytest.fail("invalid config reached environment"))
    monkeypatch.setattr(
        builder,
        "load_target_populations",
        lambda: pytest.fail("invalid config reached datasets"),
    )
    with pytest.raises(ValueError, match=key):
        builder.run(cfg)


def test_source_code_mutation_during_preparation_refuses_publication(tmp_path, monkeypatch):
    cfg = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_harmfulness_safety_target.yaml")
    assert isinstance(cfg, DictConfig)
    cfg.output_dir, cfg.positive_cap, cfg.background_count = str(tmp_path / "builder"), 1, 2
    identities = iter([{"source_sha256": "a" * 64}, {"source_sha256": "b" * 64}])
    monkeypatch.setattr(builder, "self_conditioned_code_identity", lambda: next(identities))
    monkeypatch.setattr(builder, "resolve_fitting_environment", lambda cfg: SimpleNamespace(
        tokenizer=ByteChatTokenizer(), source_tokenizer=SourceTokenizer(), window=4096,
        meta_bytes=b"{}", fit_contract={"statistics_sha256": "c" * 64}, sae_config={"source": "fixture"}))
    monkeypatch.setattr(
        builder, "load_target_populations", lambda: (*populations(), source_provenance())
    )
    monkeypatch.setattr(builder, "load_dense_extraction_stack", lambda **kw: pytest.fail("changed source reached GPU"))
    with pytest.raises(ValueError, match="source code changed during safety target preparation"):
        builder.run(cfg)
    assert not (Path(cfg.output_dir) / "targets.safetensors").exists()


@pytest.mark.parametrize("failure", ["source_context", "changed_code"])
def test_fitting_failure_cannot_publish_target(tmp_path, monkeypatch, failure):
    fixture = fixed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        builder, "load_target_populations", lambda: (*populations(), source_provenance())
    )
    monkeypatch.setattr(shared, "get_tokenizer", lambda: fixture.tokenizer)
    monkeypatch.setattr(shared, "get_extraction_tokenizer", lambda *a: SourceTokenizer())
    cfg = OmegaConf.load(Path(__file__).parents[3] / "configs/eval/build_harmfulness_safety_target.yaml")
    assert isinstance(cfg, DictConfig)
    cfg.checkpoint_meta_path, cfg.stats_path = str(fixture.checkpoint_path), fixture.cfg.stats_path
    cfg.output_dir, cfg.device, cfg.positive_cap, cfg.background_count = str(tmp_path / "builder"), "cpu", 1, 2
    cfg.dry_run = False
    code = {"runtime_version": "fixture", "git_commit": "a" * 40, "source_sha256": "b" * 64}
    identities = iter([code, code, code | {"source_sha256": "c" * 64}])
    monkeypatch.setattr(builder, "self_conditioned_code_identity", lambda: next(identities))
    model = SimpleNamespace(cfg=SimpleNamespace(n_ctx=1 if failure == "source_context" else 4096), eval=lambda: None)
    monkeypatch.setattr(builder, "load_dense_extraction_stack", lambda **kw: (model, t.nn.Linear(1, 1)))

    def fit(*args, **kwargs):
        assert failure == "changed_code", "invalid source context reached extraction"
        return {"safety": t.zeros(8)}

    monkeypatch.setattr(builder, "fit_targets", fit)
    with pytest.raises(ValueError, match="source window exceeds" if failure == "source_context" else "code changed.*fitting"):
        builder.run(cfg)
    assert (Path(cfg.output_dir) / builder.TargetArtifact.PREPARATION).is_file()
    assert not (Path(cfg.output_dir) / builder.TargetArtifact.TARGETS).exists()


def test_hydra_entrypoint_invokes_builder_and_reports_artifact(monkeypatch, caplog, tmp_path):
    cfg = OmegaConf.create({"dry_run": True})
    target = tmp_path / builder.TargetArtifact.PREPARATION
    calls = []

    def run(actual):
        calls.append(actual)
        return target

    monkeypatch.setattr(builder, "run", run)
    builder.main.__wrapped__(cfg)
    assert calls == [cfg]
    assert str(target) in caplog.text
