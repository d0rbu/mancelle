"""Campaign safety CPU admission using real s32 caches, targets and tiny checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors.torch import load, save

from nanochat import checkpoint_manager
from steering.conditional.checkpoint_conditioning import read_checkpoint_conditioning
from steering.conditional.encoder_decoder import self_conditioned
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
)
from steering.conditional.sampling import parse_feature_selection, parse_negative_selection
from steering.eval import campaign2 as campaign
from steering.eval import campaign2_safety as binding
from steering.eval.axbench_upstream.methods import build_self_conditioning_config
from steering.eval.campaign2 import cell_output_dir, strength_leaf
from steering.eval.chat_benchmark_targets import TARGET_METADATA_KEY, normalize_input
from steering.eval.harmfulness_safety_targets import (
    SAFETY_EVALUATION_SOURCE_NAMES,
    SAFETY_TARGET_NAME,
    SafetySourceName,
    SafetySteeringSettings,
    safety_evaluation_prompt_keys,
)
from steering.eval.harmfulness_self_conditioning import harmfulness_prompt_seeds
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.refined_types import UnitInterval
from steering.safetensors_io import read_safetensors_metadata_from_bytes
from steering.segmentation import SegmentMode
from tests.steering import test_chat_eval_turn_cache_integration as integration
from tests.steering.test_harmfulness_safety_runtime import _runtime_fixture


def target_contents(path):
    payload = path.read_bytes()
    metadata = read_safetensors_metadata_from_bytes(payload)
    return load(payload), json.loads(metadata[TARGET_METADATA_KEY])


def replace_target(path, tensors, manifest):
    path.write_bytes(save(tensors, metadata={TARGET_METADATA_KEY: canonical_turn_payload(manifest).decode()}))


def safety_fixture(tmp_path, monkeypatch):
    # Change the source width before producing every shard/contract, rather than
    # relabelling an existing s4 fixture as a Campaign-compatible s32 cache.
    builder = integration.build_fixture
    monkeypatch.setattr(integration, "build_fixture", lambda *args, **kwargs: builder(*args, **kwargs, size=32))
    fixture, runtime_cfg, arguments, prompt, target_path = _runtime_fixture(
        tmp_path, monkeypatch, "turn", compacted=True,
    )
    tensors, manifest = target_contents(target_path)
    count = len(manifest["preparation"]["accepted"][SAFETY_TARGET_NAME])
    manifest["preparation"]["positive_cap"] = count
    replace_target(target_path, tensors, manifest)
    monkeypatch.setattr(binding, "SAFETY_POSITIVE_COUNT", count)
    meta = json.loads(fixture.checkpoint_path.read_bytes())
    cadence = meta["turn_self_cond_segment_size_rustbpe"]
    meta["turn_self_cond_cadence_provenance"] = {
        "tool": "tiny-training-measurement", "documents": 4, "windows": 12,
        "mean": f"{cadence}/1", "minimum": cadence, "maximum": cadence,
    }
    fixture.checkpoint_path.write_text(json.dumps(meta))
    conditioning = read_checkpoint_conditioning(meta)
    features = arguments["feature_contract"]
    setup = SimpleNamespace(
        target_meta_path=fixture.checkpoint_path, meta=meta,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens, prompt_segment_size=32,
        turn_self_cond_segment_size_rustbpe=cadence,
        n_sae_features=fixture.model.encoder.config.n_sae_features,
        feature_mask=features.feature_mask, consolidation=features.consolidation,
        feature_vocab=fixture.model.feature_vocab,
        output_selection=parse_feature_selection(dict(conditioning.positive_selection)),
        output_negative_selection=parse_negative_selection(conditioning.negative_selection),
        percentile_table=t.arange(8 * 99, dtype=t.float64).reshape(8, 99),
    )
    rows = [{"prompt": prompt, "pool": "harmful", "category": "test"}]
    cfg = OmegaConf.create({
        "variant": "encdec_safety", "step": 1116, "seed": 42, "device": "cpu",
        "num_harmful": 1, "num_benign": 0, "max_new_tokens": 5,
        "statistics": fixture.contract.provenance["statistics_path"],
        "conditioning": runtime_cfg["conditioning"],
        "safety": {"targets": str(target_path), "strength": 3},
        "output_dir": str(tmp_path / "out"),
    })
    (tmp_path / "staged_tokenizer" / "tokenizer.pkl").write_bytes(b"tiny fixture tokenizer bytes")
    # The real tiny-checkpoint loader reads and binds actual tensor bytes on CPU.
    loader = checkpoint_manager.build_encdec_model

    def load_cpu(*args, **kwargs):
        assert args[2] == t.device("cpu")
        return loader(*args, **kwargs)

    monkeypatch.setattr(binding, "build_encdec_model", load_cpu)
    monkeypatch.setattr(binding, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=rows))
    evaluation = [{"prompt": value} for value in manifest["preparation"]["evaluation_prompts"]]
    sources = {name.value: manifest["preparation"]["sources"][name.value] for name in (
        SafetySourceName.EVALUATION_BEAVERTAILS, SafetySourceName.EVALUATION_WILDGUARD,
    )}
    def load_populations(names):
        assert tuple(names) == SAFETY_EVALUATION_SOURCE_NAMES
        boundary = sources[SafetySourceName.EVALUATION_BEAVERTAILS]["rows"]
        return {
            SafetySourceName.EVALUATION_BEAVERTAILS: evaluation[:boundary],
            SafetySourceName.EVALUATION_WILDGUARD: evaluation[boundary:],
        }, sources

    monkeypatch.setattr(binding, "load_populations", load_populations)
    monkeypatch.setattr(binding, "self_conditioned_code_identity", lambda: {
        "runtime_version": "test", "git_commit": "a" * 40, "source_sha256": "b" * 64,
    })
    monkeypatch.setattr(type(fixture.model), "forward", lambda *args, **kwargs: pytest.fail("forward during CPU admission"))
    return SimpleNamespace(fixture=fixture, cfg=cfg, setup=setup, rows=rows,
        path=target_path, manifest=manifest, evaluation=evaluation, sources=sources)


def test_real_cpu_admission_binds_target_rows_source_geometry_cache_and_prompt_seeds(tmp_path, monkeypatch):
    case = safety_fixture(tmp_path, monkeypatch)
    cell = binding.prepare_safety_cell(case.cfg, case.setup)
    identity = cell.identity
    assert json.loads(json.dumps(identity, allow_nan=False)) == identity
    assert cell.runtime.device == t.device("cpu")
    assert identity["target"]["sha256"] == hashlib.sha256(case.path.read_bytes()).hexdigest()
    assert identity["target"]["tensor"] == "target.safety"
    assert identity["target"]["feature_ids"] == list(case.fixture.model.feature_vocab.original_ids)
    contract = identity["target"]["fit_contract"]
    assert contract["source"]["source_segment_size"] == 32
    assert contract["source"]["source_segment_mode"] == SegmentMode.turn_delimited_fixed_tokens
    assert contract["source"]["source_bos_char_len"] > 0
    assert contract["statistics_sha256"] == case.fixture.contract.lineage.statistics_sha256
    assert contract["feature_vocab"] == case.fixture.model.feature_vocab.metadata_dict()
    assert contract["feature_mask"] and contract["feature_clusters"]
    assert identity["target"]["response_provenance"] == case.manifest["preparation"]["accepted"][SAFETY_TARGET_NAME]
    assert identity["target"]["sources"] == case.manifest["preparation"]["sources"]
    assert identity["cache"]["schema_version"] == 5
    assert identity["cache"]["manifest_sha256"] == hashlib.sha256((case.fixture.cache / "manifest.json").read_bytes()).hexdigest()
    assert identity["completion"]["safety_steering"]["strength"] == 3
    assert identity["prompt_seeds"] == [list(harmfulness_prompt_seeds(base_seed=42,
        stable_pool_index=0, **case.rows[0]))]
    cell.assert_stable()


@pytest.mark.parametrize("strength", [0, 3, 6, 2.5])
def test_strength_binds_completion_policy_and_separate_output_directory(tmp_path, monkeypatch, strength):
    case = safety_fixture(tmp_path, monkeypatch)
    case.cfg.safety.strength = strength
    cell = binding.prepare_safety_cell(case.cfg, case.setup)
    assert cell.identity["completion"]["strength"] == strength
    assert float(cell.runtime.policy.score_policy.weight) == strength / 6
    assert cell_output_dir(case.cfg).name == strength_leaf(strength)
    assert cell_output_dir(case.cfg).parent == Path(case.cfg.output_dir)
    assert strength_leaf(float(strength)) == strength_leaf(strength)


@pytest.mark.parametrize("fault", ["appended", "different", "source_revision"])
def test_complete_pinned_exclusion_population_required(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    if fault == "appended":
        case.evaluation.append({"prompt": "Excluded full source prompt missing from prefix bundle"})
    elif fault == "different":
        case.evaluation[0] = {"prompt": "Changed evaluation source prompt"}
    else:
        case.sources[SafetySourceName.EVALUATION_WILDGUARD]["revision"] = "f" * 40
    with pytest.raises(ValueError, match="complete pinned evaluation sources"):
        binding.prepare_safety_cell(case.cfg, case.setup)


@pytest.mark.parametrize("fault", ["cap", "accepted"])
def test_campaign_requires_exact_positive_population(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    if fault == "accepted":
        monkeypatch.setattr(binding, "SAFETY_POSITIVE_COUNT", 128)
    else:
        tensors, manifest = target_contents(case.path)
        manifest["preparation"]["positive_cap"] += 1
        replace_target(case.path, tensors, manifest)
    with pytest.raises(ValueError, match="exactly .* safe training positives"):
        binding.prepare_safety_cell(case.cfg, case.setup)


@pytest.mark.parametrize("fault", ["row_order", "statistics", "source_bos", "missing_percentile", "nan", "tensor_name", "invalid_bytes"])
def test_changed_target_contract_refused_before_any_forward(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    tensors, manifest = target_contents(case.path)
    if fault == "row_order":
        tensors["feature_ids"] = tensors["feature_ids"].flip(0)
    elif fault == "statistics":
        manifest["fit_contract"]["statistics_sha256"] = "f" * 64
    elif fault == "source_bos":
        manifest["fit_contract"]["source"]["source_bos_char_len"] += 1
    elif fault == "missing_percentile":
        manifest["fit_contract"]["missing_percentile_policy"] = "lowest"
    elif fault == "nan":
        tensors["target.safety"][0] = float("nan")
    elif fault == "tensor_name":
        tensors["target.other"] = tensors.pop("target.safety")
    else:
        case.path.write_bytes(b"corrupt bundle")
    if fault != "invalid_bytes":
        replace_target(case.path, tensors, manifest)
    with pytest.raises((ValueError, RuntimeError)):
        binding.prepare_safety_cell(case.cfg, case.setup)


def test_changed_valid_target_bytes_create_different_cell_identity(tmp_path, monkeypatch):
    case = safety_fixture(tmp_path, monkeypatch)
    first = binding.prepare_safety_cell(case.cfg, case.setup)
    tensors, manifest = target_contents(case.path)
    tensors["target.safety"] = tensors["target.safety"] * 0.5
    replace_target(case.path, tensors, manifest)
    second = binding.prepare_safety_cell(case.cfg, case.setup)
    assert first.identity != second.identity
    assert first.identity["target"]["sha256"] != second.identity["target"]["sha256"]
    with pytest.raises(ValueError, match="target bytes changed"):
        first.assert_stable()


@pytest.mark.parametrize("fault", ["metadata", "model", "target", "cache_manifest", "cache_shard",
    "statistics", "tokenizer", "cadence_receipt"])
def test_live_cell_fences_changed_artifact_bytes(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    cadence_receipt = case.fixture.checkpoint_path.parent / "campaign2_cadence.json"
    cadence_receipt.write_text(json.dumps(case.setup.meta["turn_self_cond_cadence_provenance"]))
    cell = binding.prepare_safety_cell(case.cfg, case.setup)
    paths = {
        "metadata": case.fixture.checkpoint_path,
        "model": cell.checkpoint_snapshot.model_path,
        "target": case.path,
        "cache_manifest": case.fixture.cache / "manifest.json",
        "cache_shard": case.fixture.cache / cell.runtime.context_provider.manifest["shards"][0]["file"],
        "statistics": Path(case.cfg.statistics),
        "tokenizer": tmp_path / "staged_tokenizer" / "tokenizer.pkl",
        "cadence_receipt": cadence_receipt,
    }
    path = paths[fault]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        cell.assert_stable()


def test_loaded_code_inventory_is_fenced_without_editing_repository_files(tmp_path, monkeypatch):
    case = safety_fixture(tmp_path, monkeypatch)
    cell = binding.prepare_safety_cell(case.cfg, case.setup)
    code_path = Path(binding.__file__)
    assert any(path == code_path for path, _ in cell.input_signatures)
    signature = binding.file_signature
    monkeypatch.setattr(binding, "file_signature", lambda path: (*signature(path), 1) if path == code_path else signature(path))
    with pytest.raises(ValueError, match="code changed"):
        cell.assert_stable()


@pytest.mark.parametrize("fault", ["checkpoint_before_load", "loader_cadence", "loader_vocab", "loader_architecture"])
def test_checkpoint_contract_mutation_refused_before_admission(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    if fault == "checkpoint_before_load":
        meta = json.loads(case.fixture.checkpoint_path.read_bytes())
        meta["turn_self_cond_segment_size_rustbpe"] += 1
        case.fixture.checkpoint_path.write_text(json.dumps(meta))
    else:
        loader = binding.build_encdec_model

        def changed_loader(*args, **kwargs):
            model, tokenizer, metadata = loader(*args, **kwargs)
            metadata = copy.deepcopy(metadata)
            if fault == "loader_cadence":
                metadata["turn_self_cond_segment_size_rustbpe"] += 1
            elif fault == "loader_vocab":
                metadata["feature_vocab"]["sha256"] = "f" * 64
            else:
                metadata["model_config"]["sequence_len"] += 1
            return model, tokenizer, metadata

        monkeypatch.setattr(binding, "build_encdec_model", changed_loader)
    with pytest.raises(ValueError, match="metadata (changed|differs)"):
        binding.prepare_safety_cell(case.cfg, case.setup)


def test_resolved_feature_head_layer_changes_identity_with_unchanged_leaf_and_weights(tmp_path, monkeypatch):
    case = safety_fixture(tmp_path, monkeypatch)
    first = binding.prepare_safety_cell(case.cfg, case.setup)
    loader = binding.build_encdec_model

    def inherited_layer_loader(*args, **kwargs):
        model, tokenizer, metadata = loader(*args, **kwargs)
        # Ancestor recovery can construct the same-shaped head at another
        # decoder layer without altering the selected leaf or weight bytes.
        recovered = copy.deepcopy(model)
        assert recovered.feature_pred_config.layer != 0
        recovered.feature_pred_config = replace(recovered.feature_pred_config, layer=0)
        recovered._feature_pred_layer_idx = 0
        return recovered, tokenizer, metadata

    monkeypatch.setattr(binding, "build_encdec_model", inherited_layer_loader)
    second = binding.prepare_safety_cell(case.cfg, case.setup)
    before, after = first.identity["checkpoint"], second.identity["checkpoint"]
    for key in ("metadata_sha256", "model_sha256", "architecture_sha256"):
        assert before[key] == after[key]
    assert first.identity["target"] == second.identity["target"]
    assert before["feature_prediction"]["layer"] == -1
    assert after["feature_prediction"]["layer"] == 0
    assert first.identity != second.identity


@pytest.mark.parametrize("fault", ["cache_path", "cache_geometry", "prompt_budget", "absent_cached_prompt"])
def test_missing_or_incompatible_real_cache_refused_on_cpu(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    if fault == "cache_path":
        case.cfg.conditioning.context_cache_dir = str(tmp_path / "missing")
    elif fault == "cache_geometry":
        path = case.fixture.cache / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["turn_contract"]["source_segment_size"] += 1
        path.write_text(json.dumps(manifest))
    elif fault == "prompt_budget":
        case.cfg.max_new_tokens = case.fixture.model.dec_config.sequence_len
    else:
        # The second full-evaluation prompt is excluded from the target fit but
        # was never produced by the real prompt-cache builder.
        tensors, manifest = target_contents(case.path)
        unknown = "An excluded evaluation prompt without a cached row"
        manifest["preparation"]["evaluation_prompts"][1] = unknown
        prompts = manifest["preparation"]["evaluation_prompts"]
        manifest["preparation"]["evaluation_input_keys"] = sorted({normalize_input(value) for value in prompts})
        manifest["evaluation_prompt_keys"][SAFETY_TARGET_NAME] = safety_evaluation_prompt_keys(prompts, case.fixture.tokenizer)
        replace_target(case.path, tensors, manifest)
        case.evaluation[:] = [{"prompt": value} for value in prompts]
        case.rows[0]["prompt"] = unknown
    expected = {"cache_path": "manifest", "cache_geometry": "turn context contract",
        "prompt_budget": "generation budget", "absent_cached_prompt": "no real context cache row"}
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError), match=expected[fault]):
        binding.prepare_safety_cell(case.cfg, case.setup)


def test_execution_device_changes_identity_while_both_admissions_remain_cpu(tmp_path, monkeypatch):
    case = safety_fixture(tmp_path, monkeypatch)
    cpu = binding.prepare_safety_cell(case.cfg, case.setup)
    case.cfg.device = "cuda"
    cuda = binding.prepare_safety_cell(case.cfg, case.setup)
    assert cpu.runtime.device == cuda.runtime.device == t.device("cpu")
    assert cpu.identity["execution_device_type"] == "cpu"
    assert cuda.identity["execution_device_type"] == "cuda"
    assert cpu.identity != cuda.identity


@pytest.mark.parametrize("fault", ["fixed", "s4"])
def test_campaign_source_geometry_refused_before_loading_checkpoint(tmp_path, monkeypatch, fault):
    case = safety_fixture(tmp_path, monkeypatch)
    if fault == "fixed":
        case.setup.segment_mode = SegmentMode.fixed_tokens
    else:
        case.setup.prompt_segment_size = 4
    monkeypatch.setattr(binding, "build_encdec_model", lambda *args, **kwargs: pytest.fail("loaded incompatible checkpoint"))
    with pytest.raises(ValueError, match="turn-delimited source s32"):
        binding.prepare_safety_cell(case.cfg, case.setup)


@pytest.mark.parametrize("value", [True, False, None, "3", -1, 7, float("nan"), float("inf"), -float("inf")])
def test_invalid_strength_fails_before_model_loading(tmp_path, monkeypatch, value):
    with pytest.raises((ValueError, TypeError)):
        SafetySteeringSettings.parse({"targets": str(tmp_path / "target"), "strength": value})


@pytest.mark.parametrize("value", [None, "", "   ", 3, False])
def test_invalid_target_path_fails_loud(value):
    with pytest.raises((ValueError, TypeError)):
        SafetySteeringSettings.parse({"targets": value, "strength": 0})


@pytest.mark.parametrize("witness_kind", ["embedded", "receipt", "both"])
def test_recorded_cadence_accepts_existing_and_campaign_witnesses(tmp_path, witness_kind):
    witness = {"documents": 3257, "windows": 40263, "mean": "448701/13421", "minimum": 17, "maximum": 209}
    setup = SimpleNamespace(target_meta_path=tmp_path / "meta_001125.json",
        meta={}, turn_self_cond_segment_size_rustbpe=33)
    if witness_kind in ("embedded", "both"):
        setup.meta["turn_self_cond_cadence_provenance"] = witness
    if witness_kind in ("receipt", "both"):
        (tmp_path / "campaign2_cadence.json").write_text(json.dumps({**witness, "cadence": 33}))
    evidence = binding.cadence_evidence(setup)
    assert evidence["cadence"] == 33
    assert len(evidence["witnesses"]) == (2 if witness_kind == "both" else 1)


@pytest.mark.parametrize("fault", ["missing", "not_object", "documents", "windows", "minimum", "maximum", "bad_mean", "infinite", "wrong_cadence", "wrong_receipt"])
def test_missing_or_inconsistent_measured_cadence_refused(tmp_path, fault):
    witness = {"documents": 4, "windows": 12, "mean": "33/1", "minimum": 30, "maximum": 40}
    if fault in ("documents", "windows", "minimum", "maximum"):
        witness[fault] = True
    elif fault == "bad_mean":
        witness["mean"] = "1/0"
    elif fault == "infinite":
        witness["mean"] = "Infinity"
    elif fault == "wrong_cadence":
        witness["mean"] = "35/1"
    elif fault == "wrong_receipt":
        witness["cadence"] = 34
    setup = SimpleNamespace(target_meta_path=tmp_path / "meta_001125.json",
        meta={} if fault == "missing" else {"turn_self_cond_cadence_provenance": [] if fault == "not_object" else witness},
        turn_self_cond_segment_size_rustbpe=33)
    with pytest.raises(ValueError, match="cadence"):
        binding.cadence_evidence(setup)


def test_strength_zero_effective_config_and_raw_logit_selection_match_unsteered(tmp_path, monkeypatch):
    case = safety_fixture(tmp_path, monkeypatch)
    case.cfg.safety.strength = 0
    cell = binding.prepare_safety_cell(case.cfg, case.setup)
    token_seed, selection_seed = cell.identity["prompt_seeds"][0]
    assert token_seed != selection_seed
    width = case.fixture.model.feature_vocab.size
    policy = SelfCondScoreInterpolation(target_scores=t.zeros(width),
        negative_selection=case.setup.output_negative_selection, weight=UnitInterval.parse(0))
    baseline = build_self_conditioning_config(case.setup, case.fixture.model, policy, selection_seed=selection_seed)
    steered = cell.runtime.make_config(selection_seed=selection_seed)
    for name in ("self_cond_segment_size", "pool_mode", "selection_producer", "bucket_producer",
                 "num_buckets", "n_sae_features", "segment_mode", "feature_mask"):
        assert getattr(baseline, name) == getattr(steered, name)
    assert baseline.cluster_consolidation.resolved.definition == steered.cluster_consolidation.resolved.definition
    assert baseline.cluster_consolidation.resolved.descriptor == steered.cluster_consolidation.resolved.descriptor
    assert asdict(baseline.selection.selection) == asdict(steered.selection.selection)
    assert asdict(baseline.score_policy.negative_selection) == asdict(steered.score_policy.negative_selection)
    assert baseline.score_policy.weight == steered.score_policy.weight == 0
    assert t.equal(baseline.selection.rng.get_state(), steered.selection.rng.get_state())
    assert not t.equal(baseline.score_policy.target_scores, steered.score_policy.target_scores)
    # Saturated logits distinguish raw-logit ranking from signed-sigmoid
    # ranking; both configs must retain the exact same counts, rows and stream.
    logits = t.linspace(100, 200, width).reshape(1, 1, -1)
    calls = []
    select = self_conditioned.select_topk_bottoml

    def capture(scores, *args, **kwargs):
        calls.append(scores.clone())
        return select(scores, *args, **kwargs)

    monkeypatch.setattr(self_conditioned, "select_topk_bottoml", capture)
    monkeypatch.setattr(SelfConditionedFeatureProvider, "_finalize_two_sided_selection",
        lambda self, state, positive, negative, **kwargs: (positive.tolist(), negative.tolist()))
    providers = [SelfConditionedFeatureProvider(case.fixture.model, config, t.device("cpu")) for config in (baseline, steered)]
    for _ in range(5):
        first = providers[0]._step_score_interpolation(None, logits, owner="test")
        second = providers[1]._step_score_interpolation(None, logits, owner="test")
        assert first == second
        assert t.equal(calls[-2], calls[-1])
        assert t.equal(calls[-1], logits[:, 0])
        assert t.equal(baseline.selection.rng.get_state(), steered.selection.rng.get_state())


def test_existing_baseline_identity_fields_remain_serialized(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs/eval/campaign2.yaml")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    cfg.checkpoint_dir, cfg.output_dir = str(checkpoint), str(tmp_path / "output")
    cfg.step, cfg.statistics = 7, str(tmp_path / "statistics")
    cfg.safety.targets, cfg.safety.strength = str(tmp_path / "unused-target"), 6
    cfg.conditioning.context_cache_dir = str(tmp_path / "unused-cache")
    metadata = {"step": 7, "dataset": "standard_steered", "sft_loss_mask": "completion_only"}
    (checkpoint / "meta_000007.json").write_text(json.dumps(metadata))
    (checkpoint / "model_000007.pt").write_bytes(b"checkpoint")
    (checkpoint / "campaign2_cadence.json").write_bytes(b"legacy witness")
    Path(cfg.statistics).write_bytes(b"statistics")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.pkl").write_bytes(b"tokenizer")
    monkeypatch.setattr(campaign, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(campaign, "load_method_setup", lambda *args: SimpleNamespace(prompt_segment_size=32))
    monkeypatch.setattr(campaign, "prepare_safety_cell", lambda *args: pytest.fail("baseline admitted safety runtime"))
    identity = campaign.preflight(cfg)
    persisted = identity["config"]
    assert "batch_size" not in persisted
    assert not {"target_source", "target_scores", "interpolation_weight"} & persisted["safety"].keys()
    assert "safety_binding" not in identity
    cfg.safety.strength, cfg.safety.targets = 0, None
    cfg.conditioning.context_cache_dir = None
    assert campaign.preflight(cfg) == identity
    assert cell_output_dir(cfg) == Path(cfg.output_dir)
