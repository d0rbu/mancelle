"""Task-split publication matches an actual all-task CPU ChatCORE invocation."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nanochat.common import COMPUTE_DTYPE, DeviceType
from nanochat.gpt import GPTConfig
from scripts.eval import build_eval_context_cache as cache_builder
from scripts.eval import chat_eval
from scripts.eval import merge_chat_eval_target_real as merger
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.eval.chat_eval_result_identity import (
    ChatEvalIdentityVersion,
    effective_architecture_sha256,
)
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.turn_context_contract import turn_context_contract_hash
from steering.feature_io import load_features, save_features
from tests.steering.test_chat_eval_turn_cache_integration import turn_main_fixture
from tests.steering.test_turn_context_cache_build import synthetic_shards


@pytest.fixture(scope="module")
def recorded_results_store():
    return []


@pytest.fixture
def recorded_task_results(tmp_path_factory, recorded_results_store):
    # Build under the function-scoped CPU reference fixtures, then reuse only
    # immutable file receipts for the rejection tests.
    if recorded_results_store:
        return recorded_results_store[0]
    directory = tmp_path_factory.mktemp("merge-chatcore")
    with pytest.MonkeyPatch.context() as monkeypatch:
        fixture = turn_main_fixture(directory, monkeypatch, clustered=True)
        outputs = []
        for label, tasks in (("first", chat_eval.ALL_CHAT_TASKS[::2]),
                             ("second", chat_eval.ALL_CHAT_TASKS[1::2]),
                             ("all", None)):
            cfg = deepcopy(fixture.cfg)
            cfg.task_name = "|".join(tasks) if tasks is not None else None
            cfg.target_real_output_dir = str(directory / label)
            chat_eval.main.__wrapped__(cfg)
            outputs.append(directory / label / "target_real/target.yaml")
        receipts = [yaml.safe_load(path.read_text()) for path in outputs]
        identity = receipts[-1]["seeds"]["adhoc"]["run_identity"]
        assert identity["version"] == ChatEvalIdentityVersion.PROBLEM_RANGES.value
        assert identity["checkpoint"]["step"] == 1116
        assert identity["checkpoint"]["metadata_sha256"] == hashlib.sha256(fixture.checkpoint_path.read_bytes()).hexdigest()
        assert identity["checkpoint"]["model_sha256"] == hashlib.sha256(
            fixture.checkpoint_path.with_name("model_001116.pt").read_bytes()).hexdigest()
        assert identity["conditioning"]["turn_context_contract_sha256"] == turn_context_contract_hash(fixture.contract)
        manifest = json.loads((fixture.cache / "manifest.json").read_text())
        assert identity["cache"]["resolved_path"] == str(fixture.cache.resolve())
        assert identity["cache"]["contexts_fingerprint"] == manifest["metadata"]["contexts_fingerprint"]
        assert identity["cache"]["shards"] == [{"file": row["file"], "sha256": row["sha256"]}
                                               for row in manifest["shards"]]
        assert identity["cache"]["input_provenance"] == {key: manifest["metadata"]["input_provenance"][key]
            for key in ("jsonl_sha256", "parquet_files", "feature_files")}
        assert identity["generation"] == {
            "dtype": "float32", "compute_dtype": str(COMPUTE_DTYPE), "device_type": "cpu", "num_samples": 1,
            "max_new_tokens": 3, "temperature": 0.0, "top_k": 50,
            "batch_size": 8, "greedy_microbatch_size": 1, "greedy_python_tools": False,
        }
        for payload in receipts[:-1]:
            assert "chatcore_metric" not in payload["seeds"]["adhoc"]
        assert not (fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").exists()
    recorded_results_store.append((outputs, receipts))
    return outputs, receipts


def test_merge_disjoint_task_jobs_matches_all_task_yaml(tmp_path, recorded_task_results):
    paths, payloads = recorded_task_results
    original_bytes = [path.read_bytes() for path in paths]
    output = tmp_path / "merged.yaml"
    result = merger.merge_target_real_results(paths[:2][::-1], output)
    assert result == yaml.safe_load(output.read_text()) == payloads[-1]
    assert list(result["seeds"]["adhoc"]["results"]) == chat_eval.ALL_CHAT_TASKS
    assert [path.read_bytes() for path in paths] == original_bytes


def test_native_cuda_device_type_produces_safe_mergeable_yaml(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    fixture.cfg.device_type = DeviceType.CUDA
    fixture.cfg.target_real_output_dir = str(tmp_path / "cuda")
    chat_eval.main.__wrapped__(fixture.cfg)

    source = tmp_path / "cuda/target_real/target.yaml"
    rendered = source.read_text()
    assert "python/object" not in rendered
    payload = yaml.safe_load(rendered)
    assert payload["seeds"]["adhoc"]["run_identity"]["generation"]["device_type"] == "cuda"

    output = tmp_path / "merged.yaml"
    assert merger.merge_target_real_results([source], output) == yaml.safe_load(output.read_text())


@pytest.mark.parametrize("field,value", [
    (("checkpoint", "metadata_sha256"), "1" * 64),
    (("checkpoint", "model_sha256"), "2" * 64),
    (("checkpoint", "step"), 1117),
    (("checkpoint", "step"), 1116.0),
    (("checkpoint", "resolved_tag"), "another-checkpoint"),
    (("checkpoint", "architecture_sha256"), "4" * 64),
    (("conditioning", "turn_context_contract_sha256"), "3" * 64),
    (("generation", "dtype"), "bfloat16"),
    (("generation", "compute_dtype"), "torch.float16"),
    (("generation", "device_type"), "cuda"),
    (("generation", "num_samples"), 2),
    (("generation", "max_new_tokens"), 512),
    (("generation", "temperature"), 0.5),
    (("generation", "top_k"), 10),
    (("generation", "batch_size"), 1),
    (("generation", "greedy_microbatch_size"), 2),
    (("generation", "greedy_python_tools"), True),
])
def test_merge_rejects_execution_identity_disagreement(tmp_path, recorded_task_results, field, value):
    paths, payloads = recorded_task_results
    changed = deepcopy(payloads[1])
    changed["seeds"]["adhoc"]["run_identity"][field[0]][field[1]] = value
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(changed))
    output = tmp_path / "merged.yaml"
    output.write_bytes(b"existing result\n")
    with pytest.raises(ValueError, match="disagree|nonnegative step"):
        merger.merge_target_real_results([paths[0], altered], output)
    assert output.read_bytes() == b"existing result\n"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [
    ("target_conditioning_mode", TargetConditioningMode.NEXT.value),
    ("pooling_op", PoolingOp.BUCKET_MEAN.value),
    ("cache_dir", "/different-cache"),
])
def test_merge_rejects_target_config_disagreement(tmp_path, recorded_task_results, field, value):
    paths, payloads = recorded_task_results
    changed = deepcopy(payloads[1])
    changed[field] = value
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="target config.*disagree"):
        merger.merge_target_real_results([paths[0], altered], tmp_path / "merged.yaml")


def test_architecture_digest_separates_recovered_cross_attention_settings():
    # kills: digesting only the leaf metadata or only the decoder config, which
    # merges ancestor-recovered shape-compatible cross-attention settings.
    base = SimpleNamespace(dec_config=GPTConfig(), enc_config=EncoderConfig(),
                           xattn_config=CrossAttentionConfig(n_head=2))
    same = SimpleNamespace(dec_config=GPTConfig(), enc_config=EncoderConfig(),
                           xattn_config=CrossAttentionConfig(n_head=2))
    heads = SimpleNamespace(dec_config=GPTConfig(), enc_config=EncoderConfig(),
                            xattn_config=CrossAttentionConfig(n_head=4))
    encoder = SimpleNamespace(dec_config=GPTConfig(), enc_config=EncoderConfig(depth=1),
                              xattn_config=CrossAttentionConfig(n_head=2))
    digests = {label: effective_architecture_sha256(model)
               for label, model in (("base", base), ("same", same), ("heads", heads), ("encoder", encoder))}
    assert digests["base"] == digests["same"]
    assert len({digests["base"], digests["heads"], digests["encoder"]}) == 3


def test_run_identity_binds_effective_architecture_and_compute_dtype(recorded_task_results):
    # kills: hashing only the leaf metadata bytes / recording the requested dtype only.
    _, payloads = recorded_task_results
    identity = payloads[-1]["seeds"]["adhoc"]["run_identity"]
    assert len(identity["checkpoint"]["architecture_sha256"]) == 64
    assert identity["generation"]["compute_dtype"] == str(COMPUTE_DTYPE)


def test_merge_rejects_duplicate_tasks(tmp_path, recorded_task_results):
    paths, _ = recorded_task_results
    with pytest.raises(ValueError, match="duplicate tasks"):
        merger.merge_target_real_results([paths[0], paths[-1]], tmp_path / "merged.yaml")


@pytest.mark.parametrize("section", [None, "checkpoint", "conditioning", "generation", "cache"])
def test_merge_rejects_missing_identity(tmp_path, recorded_task_results, section):
    _, payloads = recorded_task_results
    changed = deepcopy(payloads[-1])
    body = changed["seeds"]["adhoc"]
    if section is None:
        del body["run_identity"]
    else:
        del body["run_identity"][section]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="run_identity"):
        merger.merge_target_real_results([path], tmp_path / "merged.yaml")


@pytest.mark.parametrize("score", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_merge_rejects_invalid_scores(tmp_path, recorded_task_results, score):
    _, payloads = recorded_task_results
    changed = deepcopy(payloads[-1])
    changed["seeds"]["adhoc"]["results"]["MMLU"] = score
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="invalid accuracy"):
        merger.merge_target_real_results([path], tmp_path / "merged.yaml")


def test_merge_recomputes_metric_and_rejects_incomplete_or_unknown_tasks(tmp_path, recorded_task_results):
    paths, payloads = recorded_task_results
    output = tmp_path / "merged.yaml"
    with pytest.raises(ValueError, match="missing tasks"):
        merger.merge_target_real_results(paths[:1], output)
    with pytest.raises(ValueError, match="at least one"):
        merger.merge_target_real_results([], output)
    changed = deepcopy(payloads[-1])
    changed["seeds"]["adhoc"]["chatcore_metric"] = -123.0
    path = tmp_path / "stale-metric.yaml"
    path.write_text(yaml.safe_dump(changed))
    assert merger.merge_target_real_results([path], output) == payloads[-1]
    changed["seeds"]["adhoc"]["results"]["unknown"] = 0.5
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="unknown ChatCORE task"):
        merger.merge_target_real_results([path], output)


def test_merge_rejects_duplicate_yaml_keys(tmp_path, recorded_task_results):
    _, payloads = recorded_task_results
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(payloads[0]) + "kind: target_real_pooled\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        merger.merge_target_real_results([path], tmp_path / "merged.yaml")


@pytest.mark.parametrize("change", ["population", "extracted-shards"])
def test_different_valid_cache_contents_at_same_path_cannot_merge(tmp_path, monkeypatch, change):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    monkeypatch.chdir(tmp_path)
    fixture.cfg.target_real_configs[0].cache_dir = "cache"
    fixture.cfg.task_name = "|".join(chat_eval.ALL_CHAT_TASKS[::2])
    fixture.cfg.target_real_output_dir = str(tmp_path / "first")
    chat_eval.main.__wrapped__(fixture.cfg)
    first = tmp_path / "first/target_real/target.yaml"
    if change == "population":
        records = list(reversed(fixture.records))
        cache_builder.write_context_records(fixture.cache_cfg.contexts_file, records)
        cache_builder.write_context_parquet(fixture.cache_cfg.context_parquet_dir, records)
        synthetic_shards(tmp_path / "features", records, fixture.source)
    else:
        for path in fixture.paths:
            data = load_features(path)
            data["feature_mags"] *= 0.9
            save_features(data, path)
    fixture.cache_cfg.overwrite = True
    cache_builder.build_cache_from_config(fixture.cache_cfg)
    cache_builder.validate_cache_from_config(fixture.cache_cfg)
    fixture.cfg.task_name = "|".join(chat_eval.ALL_CHAT_TASKS[1::2])
    fixture.cfg.target_real_output_dir = str(tmp_path / "second")
    chat_eval.main.__wrapped__(fixture.cfg)
    second = tmp_path / "second/target_real/target.yaml"
    one, two = [yaml.safe_load(path.read_text()) for path in (first, second)]
    assert one["cache_dir"] == two["cache_dir"] == "cache"
    left, right = [payload["seeds"]["adhoc"]["run_identity"] for payload in (one, two)]
    assert left["checkpoint"] == right["checkpoint"]
    assert left["conditioning"] == right["conditioning"]
    assert left["cache"]["resolved_path"] == right["cache"]["resolved_path"]
    if change == "population":
        assert left["cache"]["contexts_fingerprint"] != right["cache"]["contexts_fingerprint"]
    else:
        assert left["cache"]["contexts_fingerprint"] == right["cache"]["contexts_fingerprint"]
        assert left["cache"]["input_provenance"]["feature_files"] != right["cache"]["input_provenance"]["feature_files"]
    with pytest.raises(ValueError, match="disagree"):
        merger.merge_target_real_results([first, second], tmp_path / "merged.yaml")
    assert not (tmp_path / "merged.yaml").exists()


def test_merge_legacy_receipts_keep_previous_contract(tmp_path, recorded_task_results):
    _, payloads = recorded_task_results
    paths = []
    legacy = []
    for index, payload in enumerate(payloads):
        changed = deepcopy(payload)
        body = changed["seeds"]["adhoc"]
        del body["task_details"]
        body["run_identity"]["version"] = ChatEvalIdentityVersion.LEGACY.value
        body["run_identity"]["generation"]["max_problems"] = 1
        path = tmp_path / f"legacy-{index}.yaml"
        path.write_text(yaml.safe_dump(changed))
        paths.append(path)
        legacy.append(changed)
    assert merger.merge_target_real_results(paths[:2], tmp_path / "merged.yaml") == legacy[-1]
    current = tmp_path / "current.yaml"
    current.write_text(yaml.safe_dump(payloads[1]))
    with pytest.raises(ValueError, match="disagree"):
        merger.merge_target_real_results([paths[0], current], tmp_path / "merged.yaml")
    with pytest.raises(ValueError, match="range/counts receipts"):
        merger.merge_target_real_results(paths[:2], tmp_path / "merged.yaml", task_totals={"GSM8K": 1})
    legacy[1]["seeds"]["adhoc"]["run_identity"]["generation"]["max_problems"] = 2
    paths[1].write_text(yaml.safe_dump(legacy[1]))
    with pytest.raises(ValueError, match="generation settings disagree"):
        merger.merge_target_real_results(paths[:2], tmp_path / "merged.yaml")


@pytest.mark.parametrize("field", ["metadata", "model", "conditioning", "population", "cache-shard", "input-shard"])
def test_merge_rejects_identical_malformed_digests(tmp_path, recorded_task_results, field):
    _, payloads = recorded_task_results
    paths = []
    for index, payload in enumerate(payloads[:2]):
        changed = deepcopy(payload)
        identity = changed["seeds"]["adhoc"]["run_identity"]
        if field in ("metadata", "model"):
            identity["checkpoint"][f"{field}_sha256"] = "not-a-sha256"
        elif field == "conditioning":
            identity["conditioning"]["turn_context_contract_sha256"] = "not-a-sha256"
        elif field == "population":
            identity["cache"]["contexts_fingerprint"] = "not-a-sha256"
        elif field == "cache-shard":
            identity["cache"]["shards"][0]["sha256"] = "not-a-sha256"
        else:
            identity["cache"]["input_provenance"]["feature_files"][0]["sha256"] = "not-a-sha256"
        path = tmp_path / f"invalid-{index}.yaml"
        path.write_text(yaml.safe_dump(changed))
        paths.append(path)
    with pytest.raises(ValueError, match="requires SHA-256 digests"):
        merger.merge_target_real_results(paths, tmp_path / "merged.yaml")
    assert not (tmp_path / "merged.yaml").exists()


def test_merge_rejects_consistent_boolean_receipt_version(tmp_path, recorded_task_results):
    _, payloads = recorded_task_results
    paths = []
    for index, payload in enumerate(payloads[:2]):
        changed = deepcopy(payload)
        body = changed["seeds"]["adhoc"]
        # True compares equal to legacy version 1, so use otherwise valid v1 bodies.
        del body["task_details"]
        body["run_identity"]["version"] = True
        body["run_identity"]["generation"]["max_problems"] = 1
        path = tmp_path / f"boolean-{index}.yaml"
        path.write_text(yaml.safe_dump(changed))
        paths.append(path)
    with pytest.raises(ValueError, match="unsupported run_identity version"):
        merger.merge_target_real_results(paths, tmp_path / "merged.yaml")
    assert not (tmp_path / "merged.yaml").exists()


def test_merge_rejects_consistent_zero_num_samples(tmp_path, recorded_task_results):
    _, payloads = recorded_task_results
    paths = []
    for index, payload in enumerate(payloads[:2]):
        changed = deepcopy(payload)
        changed["seeds"]["adhoc"]["run_identity"]["generation"]["num_samples"] = 0
        path = tmp_path / f"zero-{index}.yaml"
        path.write_text(yaml.safe_dump(changed))
        paths.append(path)
    with pytest.raises(ValueError, match="positive integer num_samples"):
        merger.merge_target_real_results(paths, tmp_path / "merged.yaml")
    assert not (tmp_path / "merged.yaml").exists()


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace", None])
def test_merge_publishes_atomically(tmp_path, monkeypatch, recorded_task_results, failure):
    paths, payloads = recorded_task_results
    output = tmp_path / "merged.yaml"
    output.write_bytes(b"previous complete result\n")
    original_dump, original_replace = merger.yaml.safe_dump, merger.os.replace
    original_fsync = merger.os.fsync
    replacements = []
    syncs = []

    def dump(payload, stream, **kwargs):
        assert Path(stream.name) != output
        assert Path(stream.name).parent == output.parent
        assert output.read_bytes() == b"previous complete result\n"
        if failure == "serialize":
            stream.write("partial")
            raise OSError("injected serialization failure")
        return original_dump(payload, stream, **kwargs)

    def fsync(fd):
        stat = merger.os.fstat(fd)
        assert yaml.safe_load(merger.os.pread(fd, stat.st_size, 0)) == payloads[-1]
        assert output.read_bytes() == b"previous complete result\n"
        syncs.append(stat.st_ino)
        if failure == "fsync":
            raise OSError("injected synchronization failure")
        original_fsync(fd)

    def replace(source, destination):
        assert syncs == [Path(source).stat().st_ino], "complete temporary file must be fsynced before replacement"
        assert yaml.safe_load(Path(source).read_text()) == payloads[-1]
        assert output.read_bytes() == b"previous complete result\n"
        replacements.append((source, destination))
        if failure == "replace":
            raise OSError("injected replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(merger.yaml, "safe_dump", dump)
    monkeypatch.setattr(merger.os, "replace", replace)
    monkeypatch.setattr(merger.os, "fsync", fsync)
    if failure is None:
        merger.merge_target_real_results(paths[:2], output)
        assert len(replacements) == 1
        assert yaml.safe_load(output.read_text()) == payloads[-1]
    else:
        with pytest.raises(OSError, match="injected"):
            merger.merge_target_real_results(paths[:2], output)
        assert output.read_bytes() == b"previous complete result\n"
    assert list(tmp_path.iterdir()) == [output]
    assert len(syncs) == (0 if failure == "serialize" else 1)


def test_merge_hydra_entrypoint_and_input_preservation(tmp_path, recorded_task_results):
    paths, payloads = recorded_task_results
    output = tmp_path / "merged.yaml"
    config_dir = Path(__file__).resolve().parents[2] / "configs/eval"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="merge_chat_eval_target_real", overrides=[
            f"input_files={json.dumps([str(path) for path in paths[:2]])}",
            f"output_file={output}",
        ])
    merger.main.__wrapped__(cfg)
    assert yaml.safe_load(output.read_text()) == payloads[-1]
    with pytest.raises(ValueError, match="differ from every input"):
        merger.merge_target_real_results(paths[:2], paths[0])
    with pytest.raises(ValueError, match="must be a list"):
        merger.main.__wrapped__(OmegaConf.create({"input_files": str(paths[0]), "output_file": str(output)}))
