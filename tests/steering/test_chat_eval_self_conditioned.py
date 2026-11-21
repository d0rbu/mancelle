"""Both ChatCORE families through main, actual local task objects and real FP/KV forwards."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
import yaml
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from errors import FeatureClustersError, FeatureVocabError
from scripts.eval import chat_eval
from scripts.eval import merge_chat_eval_target_real as merger
from steering.conditional.checkpoint_feature_contract import resolve_checkpoint_feature_contract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig, FeaturePredLossMode
from steering.conditional.encoder_decoder.fixed_cadence import (
    measure_fixed_self_cond_segment_size_rustbpe,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfConditionedFeatureProvider
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval import chat_self_conditioned_cache as fixed_cache
from steering.eval import chat_self_conditioning as runtime
from steering.eval.chat_self_conditioned_cache import resolve_fixed_chat_cache_contract
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.context_enumeration import enumerate_chat_generative_contexts
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from steering.sae_resolver import SAEWidth
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY
from tests.encdec_test_helpers import build_tiny_model
from tests.steering import fixed_cadence_helpers as cadence_helpers
from tests.steering import test_chat_eval_turn_cache_integration as integration
from tests.steering import test_turn_context_cache_build as turn_build
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.chatcore_checkpoint_helpers import PRODUCTION_MODEL_TAG, stage_checkpoint
from tests.steering.test_turn_context_contract import checkpoint_fixture, cluster_checkpoint

ROOT = Path(__file__).resolve().parents[2]


class MeasurementChatTokenizer(ByteChatTokenizer):
    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return [self.encode(item, **kwargs) for item in text]
        return super().encode(text, **kwargs)


def tiny_model(*, vocab=None):
    t.manual_seed(554)
    width = 8 if vocab is None else vocab.n_sae_features
    model = build_tiny_model(n_sae_features=PositiveInt.parse(width), num_buckets=PositiveInt.parse(2),
        feature_pred_config=FeaturePredictionConfig(loss_mode=FeaturePredLossMode.MAX_POOL),
        seq_len=2048, n_layer=2, xattn_frequency=1, vocab_size=512, feature_vocab=vocab)
    model.eval()
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if "cross_attn" in name and parameter.ndim >= 2:
                parameter.normal_(std=0.04)
    return model


def selfcond_turn_fixture(tmp_path, monkeypatch, *, gsm_problems=1, clustered=False, production=False):
    if not production:
        original = turn_build.checkpoint_fixture
        monkeypatch.setattr(turn_build, "checkpoint_fixture", lambda *args, **kwargs: original(*args, **kwargs, buckets=2))
    monkeypatch.setattr(integration, "turn_model", lambda _mp, contract=None: tiny_model(vocab=None if contract is None else contract.feature_vocab))
    fixture = integration.turn_main_fixture(tmp_path, monkeypatch, gsm_problems=gsm_problems,
        cache_max_problems=gsm_problems, clustered=clustered, production=production)
    # turn_main_fixture converts with build_cache_from_config and deep-validates
    # through both CLI validators. No manifest metadata is patched for this consumer.
    manifest = json.loads((fixture.cache / "manifest.json").read_text())
    assert "feature_mask_canonical" not in manifest["metadata"]
    meta = json.loads(fixture.checkpoint_path.read_text())
    lineage = meta[ONLINE_SAE_LINEAGE_META_KEY]
    meta.setdefault("turn_self_cond_segment_size_rustbpe", 2)
    meta["turn_self_cond_cadence_measurement"] = {
        "windows": 12, "mean": f"{meta['turn_self_cond_segment_size_rustbpe']}/1", "source": "tiny_train",
    }
    meta["pretrain_conditioning"].update(positive_selection={**lineage["feature_selection"], "mode": "top_k_percentile"}, negative_selection=lineage["negative_selection"])
    fixture.checkpoint_path.write_text(json.dumps(meta))
    fixture.meta.update(meta)
    cfg = fixture.cfg
    cfg.batch_size = 1
    cfg.evaluation_mode = "self_conditioned"
    cfg.target_real_configs = []
    cfg.self_conditioned.cache_dir = fixture.cache_cfg.output_dir
    cfg.self_conditioned_output_dir = str(tmp_path / "result")
    cfg.next_segment_conditioning_offset = False
    return fixture


def fixed_fixture(tmp_path, monkeypatch, *, gsm_problems=1, clustered=False, masked=False):
    tasks = integration.local_chat_tasks(tmp_path, monkeypatch, gsm_problems=gsm_problems)
    tokenizer = MeasurementChatTokenizer()
    # Scale only the external SAE registry width; inference and selection stay real.
    monkeypatch.setitem(fixed_cache.SAE_WIDTH_TO_NUM_FEATURES, SAEWidth("65k"), 8)
    monkeypatch.setattr(cadence_helpers, "PairTokenizer", MeasurementChatTokenizer)
    measured = cadence_helpers.make_cadence_fixture(tmp_path / "measured", documents=((2, 2, 2), (2, 2)), source_bos=5)
    base_meta, _, stats_path = checkpoint_fixture(tmp_path / "stats", size=128, buckets=2)
    stats_path = tmp_path / "stats/statistics.safetensors"
    with safe_open(stats_path, framework="pt") as sf:
        tensors, header = {key: sf.get_tensor(key) for key in sf.keys()}, sf.metadata()
    header.update(segment_mode="fixed_tokens", percentile_checksum="a" * 64)
    save_file(tensors, measured.feature_dir / "feature_statistics.safetensors", metadata=header)
    measurement = measure_fixed_self_cond_segment_size_rustbpe(**measured.arguments())
    meta = measured.meta
    canonical = tmp_path / "s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos"
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(measured.feature_dir, target_is_directory=True)
    lineage = base_meta[ONLINE_SAE_LINEAGE_META_KEY]
    meta.update(step=1116, n_sae_features=8, num_buckets=2, feature_vocab=None, model_config={"sequence_len": 2048},
        fixed_self_cond_segment_size_rustbpe=int(measurement.cadence),
        fixed_self_cond_cadence_provenance={**measurement.provenance, "measurement": measurement.metadata_dict()})
    meta["user_config"].update(feature_dir=str(canonical), seed=42, num_buckets=2,
        sae={key: lineage[key] for key in ("model", "release", "activation_type", "layer", "width", "l0")},
        feature_selection={**lineage["feature_selection"], "mode": "top_k_percentile"}, negative_selection=lineage["negative_selection"],
        next_segment_conditioning_offset=False)
    vocab = None
    if clustered:
        consolidation = cluster_checkpoint(base_meta, tmp_path)
        for key in ("feature_mask_canonical", "feature_clusters_canonical", "feature_clusters_descriptor", "feature_vocab"):
            meta[key] = base_meta[key]
        vocab = effective_feature_vocab(consolidation, 8)
    if masked:
        assert not clustered
        vocab = FeatureVocab(PositiveInt.parse(8), (1, 2, 3, 4))
        meta["feature_mask_canonical"] = {"mode": "whitelist", "ids": list(vocab.original_ids)}
        meta["feature_vocab"] = vocab.metadata_dict()
    model = tiny_model(vocab=vocab)
    meta_path, loaded = stage_checkpoint(tmp_path, monkeypatch, meta, model, tokenizer)
    feature_contract = resolve_checkpoint_feature_contract(meta, n_sae_features=PositiveInt.parse(8), surface="test", source_path=str(meta_path))
    contract = resolve_fixed_chat_cache_contract(meta, tokenizer, feature_contract)
    records = [row for name, task in tasks.items() for row in enumerate_chat_generative_contexts(
        tokenizer=tokenizer, task_object=task, max_problems=gsm_problems, task_name=name)]
    entries = [ContextCacheEntry(row.token_ids, (ContextFeatureSegment((1, 2) if vocab is None else tuple(vocab.original_ids[:2]), (0, 1), 0,
        len(tokenizer.decode(list(row.token_ids)))),)) for row in records]
    cache = tmp_path / "cache/c2048" / contract.identity.feature_selection_hash[:16]
    save_context_cache(cache, entries, identity=contract.identity, n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(2),
        metadata={**contract.metadata, "population": {name: len(task) for name, task in tasks.items()},
            "preparation_provenance": {"producer": "measured CPU fixture"}})
    cfg = OmegaConf.load(ROOT / "configs/eval/chat_eval.yaml")
    cfg.source, cfg.model_tag, cfg.step = "encdec", PRODUCTION_MODEL_TAG, 1116
    cfg.device_type, cfg.dtype = "cpu", "float32"
    cfg.evaluation_mode, cfg.batch_size = "self_conditioned", 1
    cfg.max_problems, cfg.max_new_tokens, cfg.model_max_seq_len = 1, 5, 2048
    cfg.self_conditioned.cache_dir = str(tmp_path / "cache")
    cfg.self_conditioned_output_dir = str(tmp_path / "result")
    monkeypatch.setattr(chat_eval, "compute_init", lambda _: (False, 0, 0, 1, t.device("cpu")))
    monkeypatch.setattr(chat_eval, "compute_cleanup", lambda: None)
    return SimpleNamespace(cfg=cfg, model=model, tokenizer=tokenizer, tasks=tasks, records=records,
        cache=cache, contract=contract, checkpoint_path=meta_path, meta=meta, loaded=loaded, source=None)


@pytest.fixture(params=["fixed", "turn"])
def family(request, tmp_path, monkeypatch):
    fixture = (fixed_fixture if request.param == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch)
    if request.param == "turn":
        moved = tmp_path / "relocated.safetensors"
        Path(fixture.contract.provenance["statistics_path"]).rename(moved)
        fixture.cfg.stats_path = str(moved)
    return fixture


def _trace(fixture, monkeypatch):
    rows, slots = [], []
    forward = fixture.model.forward
    append = SelfConditionedFeatureProvider._finalize_two_sided_selection

    def capture(*args, **kwargs):
        result = forward(*args, **kwargs)
        logits = result[0] if isinstance(result, tuple) else result
        rows.append((args[0].tolist(), kwargs["seg_map"].tolist(), logits.detach().clone()))
        return result

    def selected(self, *args, **kwargs):
        value = append(self, *args, **kwargs)
        slots.append((args[1].tolist(), args[2].tolist()))
        return value

    monkeypatch.setattr(fixture.model, "forward", capture)
    monkeypatch.setattr(SelfConditionedFeatureProvider, "_finalize_two_sided_selection", selected)
    return rows, slots


def _result(fixture):
    return Path(fixture.cfg.self_conditioned_output_dir) / "self_conditioned/result.yaml"


@pytest.mark.parametrize("family_name", ["fixed", "turn"])
def test_all_six_tasks_both_families_main(family_name, tmp_path, monkeypatch, request):
    # Match the golden's single-threaded float32 forwards, including model setup.
    previous_threads = t.get_num_threads()
    request.addfinalizer(lambda: t.set_num_threads(previous_threads))
    t.set_num_threads(1)
    monkeypatch.setenv("NANOCHAT_DTYPE", "float32")
    # nanochat.common resolves this env var at import time; these modules copy it.
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    family = (fixed_fixture if family_name == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch)
    if family_name == "turn":
        moved = tmp_path / "relocated.safetensors"
        Path(family.contract.provenance["statistics_path"]).rename(moved)
        family.cfg.stats_path = str(moved)
    predictions = {}
    factory = chat_eval._chat_task
    def task_factory(name):
        task = factory(name)
        evaluate = task.evaluate
        def capture(conversation, prediction):
            predictions.setdefault(name, []).append(prediction)
            return evaluate(conversation, prediction)
        monkeypatch.setattr(task, "evaluate", capture)
        return task
    monkeypatch.setattr(chat_eval, "_chat_task", task_factory)
    forwards, selected = _trace(family, monkeypatch)
    chat_eval.main.__wrapped__(family.cfg)
    result = yaml.safe_load(_result(family).read_text())
    body = result["seeds"]["adhoc"]
    assert result["kind"] == "self_conditioned"
    assert set(body["results"]) == set(chat_eval.ALL_CHAT_TASKS)
    assert all(row["total"] == 1 for row in body["task_details"].values())
    identity = body["run_identity"]
    assert identity["conditioning"]["recorded_k"] == family.meta.get(
        "fixed_self_cond_segment_size_rustbpe", family.meta.get("turn_self_cond_segment_size_rustbpe"))
    digest = hashlib.sha256()
    for ids, slots, logits in forwards:
        digest.update(json.dumps([ids, slots]).encode())
        digest.update(logits.float().numpy().tobytes())
    observed = {"predictions": predictions, "results": body["results"],
        "forwards_sha256": digest.hexdigest(),
        "selection_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest()}
    mode = identity["conditioning"]["segment_mode"]
    golden = json.loads((ROOT / "tests/steering/golden/chat_self_conditioned.json").read_text())
    assert observed == golden[mode]


@pytest.mark.parametrize("fault", ["missing", "hash", "schema", "header", "overlength", "no_fp", "fp_mode", "buckets", "cadence", "witness", "stats"])
def test_failure_before_forward_or_publication(family, monkeypatch, fault):
    meta = json.loads(family.checkpoint_path.read_text())
    manifest_path = family.cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if fault == "stats":
        bad = family.cache.parent / "wrong-statistics.safetensors"
        bad.write_bytes(b"wrong statistics")
        family.cfg.stats_path = str(bad)
    elif fault == "missing":
        monkeypatch.setattr(SafetensorsContextFeatureProvider, "get", lambda *a: None)
    elif fault == "hash":
        (family.cache / manifest["shards"][0]["file"]).write_bytes(b"corrupt")
    elif fault in ("schema", "header"):
        manifest["schema_version"] = 3 if fault == "schema" else (5 if manifest["schema_version"] == 4 else 4)
        manifest_path.write_text(json.dumps(manifest))
    elif fault == "overlength":
        monkeypatch.setattr(runtime, "render_chat_completion", lambda *a: SimpleNamespace(token_ids=[0] * 2049))
    elif fault == "no_fp":
        family.model.feature_pred_head = None
    elif fault == "fp_mode":
        family.model.feature_pred_config = replace(family.model.feature_pred_config, loss_mode=FeaturePredLossMode.BEST_TOKEN)
    elif fault == "buckets":
        family.model.encoder.config.num_buckets = 3
    elif fault == "cadence":
        meta.pop("fixed_self_cond_segment_size_rustbpe", None)
        meta.pop("turn_self_cond_segment_size_rustbpe", None)
        family.checkpoint_path.write_text(json.dumps(meta))
    elif fault == "witness":
        meta["selfcond_data"] = {"sft_segment_mode": "turn_delimited_fixed_tokens" if "fixed_self_cond_segment_size_rustbpe" in meta else "fixed_tokens"}
        family.checkpoint_path.write_text(json.dumps(meta))
    monkeypatch.setattr(family.model, "forward", lambda *a, **k: pytest.fail("scored before admission"))
    with pytest.raises((ValueError, RuntimeError)):
        chat_eval.main.__wrapped__(family.cfg)
    assert not _result(family).exists()


def _assert_same_trace(first, second):
    assert len(first) == len(second)
    for (ids_a, slots_a, logits_a), (ids_b, slots_b, logits_b) in zip(first, second, strict=True):
        assert (ids_a, slots_a) == (ids_b, slots_b)
        assert t.equal(logits_a, logits_b)


@pytest.mark.parametrize("family_name", ["fixed", "turn"])
def test_reordered_problem_shards_preserve_per_example_state(tmp_path, monkeypatch, family_name):
    fixture = (fixed_fixture if family_name == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch, gsm_problems=3)
    fixture.cfg.task_name, fixture.cfg.max_problems = "GSM8K", 3
    traces, selected = _trace(fixture, monkeypatch)
    chat_eval.main.__wrapped__(fixture.cfg)
    unsharded = yaml.safe_load(_result(fixture).read_text())
    unsharded_trace, unsharded_selected = list(traces), copy.deepcopy(selected)
    shards, trace_by_start, selection_by_start = [], {}, {}
    for start, count in ((2, 1), (0, 2)):
        traces.clear()
        selected.clear()
        fixture.cfg.problem_start, fixture.cfg.max_problems = start, count
        fixture.cfg.self_conditioned_output_dir = str(tmp_path / f"shard-{start}")
        chat_eval.main.__wrapped__(fixture.cfg)
        shards.append(_result(fixture))
        trace_by_start[start], selection_by_start[start] = list(traces), copy.deepcopy(selected)
    _assert_same_trace(unsharded_trace, trace_by_start[0] + trace_by_start[2])
    assert unsharded_selected == selection_by_start[0] + selection_by_start[2]
    merged = merger.merge_target_real_results(shards, tmp_path / "merged.yaml", task_totals={"GSM8K": 3})
    assert merged["seeds"]["adhoc"] == unsharded["seeds"]["adhoc"]


@pytest.fixture
def ranged_result(tmp_path, monkeypatch):
    fixture = fixed_fixture(tmp_path, monkeypatch, gsm_problems=3)
    fixture.cfg.task_name, fixture.cfg.max_problems = "GSM8K", 3
    chat_eval.main.__wrapped__(fixture.cfg)
    return yaml.safe_load(_result(fixture).read_text())


def _write_shards(tmp_path, result):
    paths = []
    for start, count, passed in ((0, 2, 1), (2, 1, 1)):
        shard = copy.deepcopy(result)
        body = shard["seeds"]["adhoc"]
        body["task_details"]["GSM8K"].update(problem_start=start, max_problems=count, total=count, num_passed=passed)
        body["results"]["GSM8K"] = passed / count
        path = tmp_path / f"shard-{start}.yaml"
        path.write_text(yaml.safe_dump(shard))
        paths.append(path)
    return paths


def test_merger_adds_counts_instead_of_averaging(tmp_path, ranged_result):
    paths = _write_shards(tmp_path, ranged_result)
    merged = merger.merge_target_real_results(paths[::-1], tmp_path / "merged.yaml", task_totals={"GSM8K": 3})
    body = merged["seeds"]["adhoc"]
    assert body["results"]["GSM8K"] == 2 / 3
    assert body["task_details"]["GSM8K"]["num_passed"] == 2
    assert body["task_details"]["GSM8K"]["total"] == 3


@pytest.mark.parametrize("field", ["recorded_k", "selection_policy", "base_seed", "measurement_provenance_sha256",
    "cache", "checkpoint", "missing_k", "duplicate", "missing", "size", "forged", "kind"])
def test_merger_rejects_changed_semantics_ranges_and_forgery(tmp_path, ranged_result, field):
    paths = _write_shards(tmp_path, ranged_result)
    shard = yaml.safe_load(paths[1].read_text())
    body = shard["seeds"]["adhoc"]
    identity = body["run_identity"]
    if field == "missing_k":
        del identity["conditioning"]["recorded_k"]
    elif field == "cache":
        identity["cache"]["manifest_sha256"] = "f" * 64
    elif field == "checkpoint":
        identity["checkpoint"]["model_sha256"] = "f" * 64
    elif field in ("recorded_k", "base_seed"):
        identity["conditioning"][field] += 1
    elif field in ("selection_policy", "measurement_provenance_sha256"):
        identity["conditioning"][field] = "changed"
    elif field in ("duplicate", "missing"):
        body["task_details"]["GSM8K"]["problem_start"] += -1 if field == "duplicate" else 1
    elif field == "size":
        body["task_details"]["GSM8K"]["task_size"] += 1
    elif field == "forged":
        body["results"]["GSM8K"] = 0
    else:
        shard["kind"] = "target_real_lockstep"
    paths[1].write_text(yaml.safe_dump(shard))
    with pytest.raises(ValueError):
        merger.merge_target_real_results(paths, tmp_path / "merged.yaml", task_totals={"GSM8K": 3})
    assert not (tmp_path / "merged.yaml").exists()


@pytest.mark.parametrize("family_name", ["fixed", "turn"])
@pytest.mark.parametrize("fault", ["mask", "clusters", "vocab"])
def test_compact_feature_contract_mismatch_precedes_forward(tmp_path, monkeypatch, fault, family_name):
    fixture = (fixed_fixture if family_name == "fixed" else selfcond_turn_fixture)(tmp_path, monkeypatch, clustered=True)
    meta = json.loads(fixture.checkpoint_path.read_text())
    if fault == "mask":
        meta["feature_mask_canonical"] = {"mode": "whitelist", "ids": [0, 1]}
    elif fault == "clusters":
        meta.pop("feature_clusters_canonical")
    else:
        meta["feature_vocab"]["sha256"] = "f" * 64
    fixture.checkpoint_path.write_text(json.dumps(meta))
    monkeypatch.setattr(fixture.model, "forward", lambda *a, **k: pytest.fail("forward with incompatible compact contract"))
    with pytest.raises((ValueError, RuntimeError, FeatureClustersError, FeatureVocabError)):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert not _result(fixture).exists()


def test_fixed_compact_checkpoint_main(tmp_path, monkeypatch):
    fixture = fixed_fixture(tmp_path, monkeypatch, clustered=True)
    chat_eval.main.__wrapped__(fixture.cfg)
    assert _result(fixture).exists()


def test_clustered_turn_production_writer_main(tmp_path, monkeypatch):
    fixture = selfcond_turn_fixture(tmp_path, monkeypatch, clustered=True, production=True)
    fixture.cfg.task_name = "MMLU"
    manifest = json.loads((fixture.cache / "manifest.json").read_text())
    assert manifest["schema_version"] == 5
    assert manifest["turn_contract"] == fixture.contract.semantic
    assert fixture.model.encoder.config.n_sae_features == 65536
    assert fixture.model.feature_vocab_size == 844
    forwards, _ = _trace(fixture, monkeypatch)
    chat_eval.main.__wrapped__(fixture.cfg)
    assert len(forwards) == 2  # Real prefix and categorical marker forwards.
    receipt = yaml.safe_load(_result(fixture).read_text())["seeds"]["adhoc"]
    assert receipt["task_details"]["MMLU"]["total"] == 1
    assert receipt["run_identity"]["cache"]["schema_version"] == 5


def test_turn_cache_semantic_mismatch_precedes_forward(tmp_path, monkeypatch):
    fixture = selfcond_turn_fixture(tmp_path, monkeypatch, clustered=True, production=True)
    fixture.cfg.task_name = "MMLU"
    path = fixture.cache / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["turn_contract"]["feature_vocab"]["sha256"] = "f" * 64
    # Leave identity, mirrored metadata and shards intact to isolate the semantic check.
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fixture.model, "forward", lambda *a, **k: pytest.fail("forward with mismatched turn contract"))
    with pytest.raises(ContextCacheValidationError, match="turn context contract disagrees"):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert not _result(fixture).exists()
