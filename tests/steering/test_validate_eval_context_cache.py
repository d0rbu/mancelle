from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch as t
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from safetensors.torch import save_file

import scripts.eval.validate_eval_context_cache as validator
import steering.eval.harmfulness_judge as harmfulness_judge
from errors import FeatureClustersError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.stats_io import content_sha256
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
    MockTokenizer,
)


def _cfg(tmp_path: Path, **overrides):
    data = {
        "cache_dir": str(tmp_path / "cache"),
        "contexts_file": None,
        "summary_file": None,
        "n_sae_features": None,
        "num_buckets": None,
        "enumerate_evals": False,
        "evals": "core,chat,harmfulness",
        "model_max_seq_len": 1024,
        "num_seeds": 2,
        "base_seed": 10,
        "max_per_task": 1,
        "chat_tasks": None,
        "chat_batch_size": 2,
        "chat_max_problems": 2,
        "harm_num_benign": 1,
        "harm_num_harmful": 1,
        "target_feature_configs": None,
        "require_nonempty_context_features": True,
        "require_nonempty_context_segments": False,
        "fail_on_extra_cached_contexts": False,
        "identity": {
            "model_id": "test-model",
            "sae_id": "test-sae",
            "layer": 0,
            "pooling": "mean",
            "feature_selection_hash": "0" * 64,
        },
    }
    data.update(overrides)
    return OmegaConf.create(data)


def _record(tokens=(0, 99, 100, 101), suite="chat", task="toy", metadata=None):
    return EvalContextRecord(
        suite=suite,
        task=task,
        token_ids=tuple(tokens),
        context_text="abc",
        metadata=metadata if metadata is not None else {"answer_position": 2},
    )


def _segments():
    return (ContextFeatureSegment((1,), (1,), 0, 3),)


def _write_contexts(path: Path, records: list[EvalContextRecord]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "suite": record.suite,
                    "task": record.task,
                    "token_ids": list(record.token_ids),
                    "context_text": record.context_text,
                    "metadata": record.metadata,
                }
            )
            for record in records
        )
        + "\n"
    )


def _write_cache(
    path: Path,
    records: list[EvalContextRecord],
    *,
    model_max_seq_len: int = 1024,
) -> None:
    save_context_cache(
        context_length_cache_dir(
            path,
            model_max_seq_len,
            hash_leaf="0" * 16,
        ),
        [ContextCacheEntry.from_sequences(record.token_ids, _segments()) for record in records],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={"model_max_seq_len": model_max_seq_len},
        identity=TEST_CACHE_IDENTITY,
    )


def test_validate_cache_from_config_checks_contexts_file_and_writes_summary(
    tmp_path,
    monkeypatch,
):
    record = _record()
    contexts_file = tmp_path / "contexts.jsonl"
    summary_file = tmp_path / "summary.json"
    _write_contexts(contexts_file, [record])
    _write_cache(tmp_path / "cache", [record])
    monkeypatch.setattr(validator, "get_tokenizer", lambda: MockTokenizer())

    summary = validator.validate_cache_from_config(
        _cfg(
            tmp_path,
            contexts_file=str(contexts_file),
            summary_file=str(summary_file),
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            target_feature_configs={"name": "target", "features": {2: 1}},
        )
    )

    assert summary["unique_contexts"] == 1
    assert summary["total_context_features"] == 1
    assert json.loads(summary_file.read_text())["cached_contexts"] == 1


def test_validate_cache_from_config_reads_matching_context_length_subdir(
    tmp_path,
    monkeypatch,
):
    record_2048 = _record(tokens=(0, 10, 11, 12))
    stale_1024 = _record(tokens=(0, 20, 21, 22))
    contexts_file = tmp_path / "contexts.jsonl"
    _write_contexts(contexts_file, [record_2048])
    _write_cache(tmp_path / "cache", [stale_1024], model_max_seq_len=1024)
    _write_cache(tmp_path / "cache", [record_2048], model_max_seq_len=2048)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: MockTokenizer())

    summary = validator.validate_cache_from_config(
        _cfg(
            tmp_path,
            contexts_file=str(contexts_file),
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            model_max_seq_len=2048,
        )
    )

    assert summary["unique_contexts"] == 1
    assert summary["cached_contexts"] == 1


def test_validate_cache_from_config_rejects_context_length_metadata_mismatch(
    tmp_path,
    monkeypatch,
):
    record = _record()
    contexts_file = tmp_path / "contexts.jsonl"
    _write_contexts(contexts_file, [record])
    cache_dir = context_length_cache_dir(
        tmp_path / "cache",
        1024,
        hash_leaf="0" * 16,
    )
    save_context_cache(
        cache_dir,
        [ContextCacheEntry.from_sequences(record.token_ids, _segments())],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={"model_max_seq_len": 1024},
        identity=TEST_CACHE_IDENTITY,
    )
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["model_max_seq_len"] = 2048
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(validator, "get_tokenizer", lambda: MockTokenizer())

    with pytest.raises(ContextCacheValidationError, match="layout mismatch|model_max_seq_len"):
        validator.validate_cache_from_config(
            _cfg(tmp_path, contexts_file=str(contexts_file), model_max_seq_len=1024)
        )


def test_validate_cache_from_config_compares_context_file_to_eval_enumeration(
    tmp_path,
    monkeypatch,
):
    eval_record = _record(tokens=(0, 1, 2))
    file_record = _record(tokens=(0, 9, 9))
    contexts_file = tmp_path / "contexts.jsonl"
    _write_contexts(contexts_file, [file_record])
    _write_cache(tmp_path / "cache", [eval_record])
    monkeypatch.setattr(validator, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        validator,
        "enumerate_expected_eval_contexts",
        lambda cfg, tokenizer: [eval_record],
    )

    with pytest.raises(ContextCacheValidationError, match="missing"):
        validator.validate_cache_from_config(
            _cfg(
                tmp_path,
                contexts_file=str(contexts_file),
                enumerate_evals=True,
                evals="chat",
            )
        )


def test_validate_cache_from_config_uses_hydra_missing_cache_dir():
    with pytest.raises(MissingMandatoryValue, match="cache_dir"):
        validator.validate_cache_from_config(
            OmegaConf.create(
                {
                    "cache_dir": "???",
                    "model_max_seq_len": 1024,
                    "contexts_file": None,
                    "enumerate_evals": False,
                }
            )
        )


def test_standalone_validator_resolves_clusters_before_provider_or_output(
    tmp_path: Path,
):
    contexts_file = tmp_path / "contexts.jsonl"
    _write_contexts(contexts_file, [_record()])
    cfg = _cfg(
        tmp_path,
        contexts_file=str(contexts_file),
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        stats_source_sha256="a" * 64,
        feature_clusters={"path": str(tmp_path / "clusters.yaml")},
        summary_file=str(tmp_path / "summary.json"),
    )

    with pytest.raises(
        FeatureClustersError,
        match="path not found or inaccessible",
    ):
        validator.validate_cache_from_config(cfg)

    assert not (tmp_path / "summary.json").exists()


def test_standalone_validator_loads_complete_clustered_cache(
    tmp_path: Path,
    monkeypatch,
):
    record = _record()
    contexts_file = tmp_path / "contexts.jsonl"
    _write_contexts(contexts_file, [record])
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 3]]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(cluster_path)}},
        8,
    )
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    vocab = effective_feature_vocab(clusters, 8)
    assert vocab is not None
    stats_path = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(stats_path), metadata={"k": "v"})
    stats_sha = content_sha256(stats_path)
    cache_dir = context_length_cache_dir(
        tmp_path / "cache",
        1024,
        hash_leaf="0" * 16,
    )
    save_context_cache(
        cache_dir,
        [ContextCacheEntry.from_sequences(record.token_ids, _segments())],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={
            "model_max_seq_len": 1024,
            **clustered_context_cache_metadata(
                feature_mask=None,
                feature_clusters=clusters,
                feature_vocab=vocab,
                stats_source_sha256=stats_sha,
            ),
        },
        identity=TEST_CACHE_IDENTITY,
    )
    monkeypatch.setattr(validator, "get_tokenizer", lambda: MockTokenizer())

    summary = validator.validate_cache_from_config(
        _cfg(
            tmp_path,
            contexts_file=str(contexts_file),
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=TEST_CACHE_NUM_BUCKETS,
            feature_clusters={"path": str(cluster_path)},
            stats_path=str(stats_path),
            stats_source_sha256=stats_sha,
        )
    )

    assert summary["cached_contexts"] == 1
    assert summary["total_context_features"] == 1


def test_validate_cache_from_config_rejects_no_context_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "get_tokenizer", lambda: MockTokenizer())

    with pytest.raises(ValueError, match="No contexts"):
        validator.validate_cache_from_config(_cfg(tmp_path))


def test_main_invokes_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validator,
        "validate_cache_from_config",
        lambda cfg: calls.append(cfg.stage),
    )

    validator.main.__wrapped__(OmegaConf.create({"stage": "validate"}))

    assert calls == ["validate"]


def test_enumerate_expected_eval_contexts_dispatches_selected_suites(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validator,
        "_enumerate_core_contexts",
        lambda cfg, tokenizer, seed: calls.append(("core", seed)) or [],
    )
    monkeypatch.setattr(
        validator,
        "_enumerate_chat_contexts",
        lambda cfg, tokenizer: calls.append(("chat", None)) or [],
    )
    monkeypatch.setattr(
        validator,
        "_enumerate_harmfulness_contexts",
        lambda cfg, tokenizer, seed: calls.append(("harm", seed)) or [],
    )

    validator.enumerate_expected_eval_contexts(
        _cfg(Path("/tmp"), enumerate_evals=True, evals="core,chat,harmfulness"),
        MockTokenizer(),
    )

    assert calls == [
        ("core", 10),
        ("core", 1010),
        ("chat", None),
        ("harm", 10),
        ("harm", 1010),
    ]


def test_enumerate_expected_eval_contexts_rejects_unknown_suite():
    with pytest.raises(ValueError, match="Unsupported eval suite"):
        validator.enumerate_expected_eval_contexts(
            _cfg(Path("/tmp"), evals="bogus"),
            MockTokenizer(),
        )


def test_enumerate_chat_contexts_matches_task_eval_types(monkeypatch):
    class CatTask:
        eval_type = "categorical"

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"text": "choose A", "letters": ["A", "B"]}

    class GenTask:
        eval_type = "generative"

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"text": "solve"}

    monkeypatch.setattr(
        validator,
        "CHAT_TASK_FACTORIES",
        {"Cat": CatTask, "Gen": GenTask},
    )

    records = validator._enumerate_chat_contexts(
        _cfg(Path("/tmp"), chat_tasks="Cat,Gen"),
        MockTokenizer(),
    )

    assert [(record.task, record.metadata["eval_type"]) for record in records] == [
        ("Cat", "categorical"),
        ("Gen", "generative"),
    ]


def test_enumerate_chat_contexts_rejects_unknown_eval_type(monkeypatch):
    class BadTask:
        eval_type = "other"

    monkeypatch.setattr(validator, "CHAT_TASK_FACTORIES", {"Bad": BadTask})

    with pytest.raises(ValueError, match="Unsupported chat task"):
        validator._enumerate_chat_contexts(
            _cfg(Path("/tmp"), chat_tasks="Bad"),
            MockTokenizer(),
        )


def test_enumerate_core_contexts_reads_eval_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "eval_bundle"
    data_dir = bundle / "eval_data"
    data_dir.mkdir(parents=True)
    (bundle / "core.yaml").write_text(
        yaml.safe_dump(
            {
                "icl_tasks": [
                    {
                        "label": "ToyMC",
                        "icl_task_type": "multiple_choice",
                        "num_fewshot": [0],
                        "continuation_delimiter": " ",
                        "dataset_uri": "toy.jsonl",
                    }
                ]
            }
        )
    )
    (bundle / "eval_meta_data.csv").write_text("Eval Task,Random baseline\nToyMC,50\n")
    (data_dir / "toy.jsonl").write_text(
        json.dumps({"query": "Q?", "choices": ["A", "B"], "gold": 0}) + "\n"
    )
    monkeypatch.setattr(validator, "get_base_dir", lambda: str(tmp_path))

    records = validator._enumerate_core_contexts(_cfg(tmp_path), MockTokenizer(), seed=123)

    assert len(records) == 2
    assert {record.task for record in records} == {"ToyMC"}
    assert all(record.suite == "core" for record in records)


def test_enumerate_core_contexts_requires_model_max_seq_len(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "get_base_dir", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="model_max_seq_len is required"):
        validator._enumerate_core_contexts(
            _cfg(tmp_path, model_max_seq_len=None),
            MockTokenizer(),
            seed=123,
        )

    with pytest.raises(ValueError, match="model_max_seq_len is required"):
        validator._required_core_model_max_seq_len(OmegaConf.create({"model_max_seq_len": "???"}))
    with pytest.raises(ValueError, match="must be positive"):
        validator._required_core_model_max_seq_len(OmegaConf.create({"model_max_seq_len": 0}))


def test_enumerate_core_contexts_rejects_missing_or_bad_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "get_base_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="CORE eval bundle"):
        validator._enumerate_core_contexts(_cfg(tmp_path), MockTokenizer(), seed=1)

    bundle = tmp_path / "eval_bundle"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        validator._enumerate_core_contexts(_cfg(tmp_path), MockTokenizer(), seed=1)

    (bundle / "core.yaml").write_text(yaml.safe_dump({"icl_tasks": []}))
    metadata = bundle / "eval_meta_data.csv"
    metadata.write_text("Wrong\n")
    with pytest.raises(ValueError, match="Eval Task"):
        validator._load_core_metadata(metadata)

    metadata.write_text("Eval Task,Random baseline\n")
    with pytest.raises(ValueError, match="no rows"):
        validator._load_core_metadata(metadata)


def test_enumerate_harmfulness_contexts_uses_prompt_loaders(monkeypatch):
    monkeypatch.setattr(
        harmfulness_judge,
        "load_harmful_prompts",
        lambda: [{"prompt": "harm 0"}, {"prompt": "harm 1"}],
    )
    monkeypatch.setattr(
        harmfulness_judge,
        "load_benign_prompts",
        lambda n, seed: [{"prompt": f"benign {seed}"} for _ in range(n)],
    )

    records = validator._enumerate_harmfulness_contexts(
        _cfg(Path("/tmp"), harm_num_harmful=1, harm_num_benign=2),
        MockTokenizer(),
        seed=7,
    )

    assert len(records) == 3
    assert [record.metadata["pool"] for record in records] == [
        "harmful",
        "benign",
        "benign",
    ]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"harm_num_harmful": 1.5}, "max_harmful must be an integer or None"),
        ({"harm_num_benign": False}, "num_benign must be an integer"),
    ],
)
def test_enumerate_harmfulness_contexts_rejects_raw_invalid_controls_before_io(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    message: str,
) -> None:
    load_harmful = MagicMock()
    load_benign = MagicMock()
    monkeypatch.setattr(
        harmfulness_judge,
        "load_harmful_prompts",
        load_harmful,
    )
    monkeypatch.setattr(
        harmfulness_judge,
        "load_benign_prompts",
        load_benign,
    )

    with pytest.raises(TypeError, match=message):
        validator._enumerate_harmfulness_contexts(
            _cfg(Path("/tmp"), **override),
            MockTokenizer(),
            seed=7,
        )

    load_harmful.assert_not_called()
    load_benign.assert_not_called()
