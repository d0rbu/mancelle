"""Regression tests for chat target_real progression result shape.

The standalone chat_eval target_real path writes SuiteResult YAML payloads,
but training_progression stores one seed body directly under results.json.
These tests pin that the shared chat helper returns the flat body and that
progression fails loudly if the wrapped YAML schema leaks across that boundary.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

import scripts.eval.chat_eval as chat_mod
import scripts.train.training_progression as tp_mod
from scripts.train.training_progression import (
    CheckpointSpec,
    _evaluate_single_checkpoint_impl,
    _run_chat_target_real_eval,
)
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    PooledTargetRealConfig,
)
from steering.eval.target_real_runner import build_target_real_payload
from steering.segmentation import token_to_char_offsets
from tests.conftest import build_tiny_encdec, build_tiny_tokenizer


class _AlwaysCategoricalTask:
    eval_type = "categorical"

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "messages": [
                {"role": "user", "content": f"q{idx}"},
                {"role": "assistant", "content": ""},
            ],
            "letters": ["x", "y"],
            "answer": "x",
        }

    def evaluate(self, conversation, predicted_letter):
        return predicted_letter == conversation["answer"]


class _SingleEntryProvider:
    def __init__(self, segments):
        self._segments = segments
        self.call_count = 0

    def get(self, token_ids):
        self.call_count += 1
        return self._segments


def _segments_for_full_conv(tokenizer, conv):
    full_ids, _mask = tokenizer.render_conversation(conv)
    offsets = token_to_char_offsets(tokenizer, full_ids)
    return (
        ContextFeatureSegment(
            feature_ids=(2,),
            buckets=(1,),
            char_start=0,
            char_end=offsets[-1],
        ),
    )


def _patch_all_chat_tasks(monkeypatch: pytest.MonkeyPatch) -> _AlwaysCategoricalTask:
    task = _AlwaysCategoricalTask()
    monkeypatch.setattr(chat_mod, "HumanEval", lambda: task)
    monkeypatch.setattr(chat_mod, "MMLU", lambda subset, split: task)
    monkeypatch.setattr(chat_mod, "ARC", lambda subset, split: task)
    monkeypatch.setattr(chat_mod, "GSM8K", lambda subset, split: task)
    monkeypatch.setattr(chat_mod, "SpellingBee", lambda size, split: task)
    return task


def _progression_cfg(*, target_real_configs=None, chat_tasks=None):
    return OmegaConf.create({
        "model_tag": "test",
        "checkpoint_source": "local",
        "checkpoints_dir": None,
        "checkpoint_root": None,
        "checkpoint_prefix": "",
        "model_family": "encoder_decoder",
        "results_file": None,
        "fig_dir": None,
        "evals": "chat",
        "max_per_task": 1,
        "chat_tasks": chat_tasks,
        "chat_batch_size": 1,
        "chat_max_new_tokens": 8,
        "chat_temperature": 0.0,
        "chat_top_k": 50,
        "chat_num_samples": 1,
        "chat_max_problems": 1,
        "harm_temperature": 0.7,
        "harm_top_k": 50,
        "harm_max_new_tokens": 8,
        "harm_num_benign": 1,
        "harm_num_harmful": 1,
        "harm_judge_model": "gpt-5-nano",
        "device_type": "cpu",
        "dtype": "float32",
        "num_gpus": 1,
        "num_seeds": 1,
        "base_seed": 1234,
        "phases": [{"phase": "sft", "subdir": "", "steps": [0]}],
        "cleanup_after_eval": False,
        "source": "encdec",
        "conditioning": {
            "identity": {
                "model_id": "test-model",
                "sae_id": "test-sae",
                "layer": 0,
                "pooling": "mean",
                "feature_selection_hash": "h",
            },
            "n_sae_features": 64,
            "num_buckets": 4,
            "context_cache_dir": None,
            "conditioned_configs": {},
            "target_real_configs": target_real_configs or [],
        },
    })


def _expected_chatcore(seed_body: dict) -> float:
    centered_sum = 0.0
    for task_name in chat_mod.ALL_CHAT_TASKS:
        acc = seed_body[task_name]
        baseline_acc = chat_mod.BASELINE_ACCURACIES.get(task_name, 0.0)
        centered_sum += (acc - baseline_acc) / (1.0 - baseline_acc)
    return centered_sum / len(chat_mod.ALL_CHAT_TASKS)


def test_run_chat_target_real_eval_consumes_real_flat_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_encdec()
    task = _patch_all_chat_tasks(monkeypatch)
    provider = _SingleEntryProvider(_segments_for_full_conv(tokenizer, task[0]))
    trc = PooledTargetRealConfig(
        name="pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )

    result = _run_chat_target_real_eval(
        model=model,
        tokenizer=tokenizer,
        cfg=_progression_cfg(),
        trc=trc,
        provider=cast(Any, provider),
    )

    assert set(chat_mod.ALL_CHAT_TASKS).issubset(result.keys())
    assert "chatcore_metric" in result
    assert "results" not in result
    assert "seeds" not in result
    assert result["chatcore_metric"] == pytest.approx(_expected_chatcore(result))
    assert provider.call_count == 2 * len(chat_mod.ALL_CHAT_TASKS)


def test_run_chat_target_real_eval_subset_task_omits_chatcore_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_encdec()
    task = _patch_all_chat_tasks(monkeypatch)
    provider = _SingleEntryProvider(_segments_for_full_conv(tokenizer, task[0]))
    trc = PooledTargetRealConfig(
        name="pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )

    result = _run_chat_target_real_eval(
        model=model,
        tokenizer=tokenizer,
        cfg=_progression_cfg(chat_tasks="ARC-Easy"),
        trc=trc,
        provider=cast(Any, provider),
    )

    assert set(result.keys()) == {"ARC-Easy"}
    assert isinstance(result["ARC-Easy"], float)
    assert provider.call_count == 2


def test_run_chat_target_real_eval_rejects_wrapped_suite_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    trc = PooledTargetRealConfig(
        name="pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    wrapped = build_target_real_payload(
        trc,
        {"results": {"ARC-Easy": 0.5}, "chatcore_metric": 0.1},
    )
    monkeypatch.setattr(
        tp_mod,
        "_chat_eval_target_real_per_config",
        lambda **kwargs: wrapped,
    )

    with pytest.raises(ValueError, match="SuiteResult-style payload"):
        _run_chat_target_real_eval(
            model=object(),
            tokenizer=object(),
            cfg=_progression_cfg(),
            trc=trc,
            provider=cast(Any, object()),
        )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "mapping"),
        ({"results": {}}, "non-empty top-level 'results'"),
        ({"results": {1: 0.5}}, "string task names"),
        ({"results": {"ARC-Easy": "bad"}}, "numeric accuracy"),
        ({"results": {"ARC-Easy": True}}, "numeric accuracy"),
        ({"results": {"ARC-Easy": 0.5}, "chatcore_metric": "bad"}, "numeric"),
        ({"results": {"ARC-Easy": 0.5}, "chatcore_metric": False}, "numeric"),
    ],
)
def test_run_chat_target_real_eval_rejects_malformed_flat_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    body,
    message: str,
):
    trc = PooledTargetRealConfig(
        name="pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    monkeypatch.setattr(
        tp_mod,
        "_chat_eval_target_real_per_config",
        lambda **kwargs: body,
    )

    with pytest.raises(ValueError, match=message):
        _run_chat_target_real_eval(
            model=object(),
            tokenizer=object(),
            cfg=_progression_cfg(),
            trc=trc,
            provider=cast(Any, object()),
        )


def test_evaluate_single_checkpoint_chat_target_real_seed_is_non_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_encdec()
    task = _patch_all_chat_tasks(monkeypatch)
    provider = _SingleEntryProvider(_segments_for_full_conv(tokenizer, task[0]))
    cfg = _progression_cfg(
        target_real_configs=[
            {
                "name": "pooled",
                "mode": "pooled",
                "cache_dir": str(tmp_path / "cache"),
                "pooling_op": "bucket_max",
            },
        ],
    )
    spec = CheckpointSpec(step=0, phase="sft", subdir="")

    with (
        patch.object(tp_mod, "resolve_local_checkpoint_dir", return_value=str(tmp_path)),
        patch.object(
            tp_mod,
            "load_progression_model",
            return_value=(
                model,
                tokenizer,
                {"model_type": "encoder_decoder"},
                CheckpointFeatureContract(None, None),
            ),
        ),
        patch.object(
            tp_mod,
            "_target_real_providers_for_cfg",
            return_value={"pooled": provider},
        ) as providers_for_cfg,
    ):
        evals = _evaluate_single_checkpoint_impl(
            spec,
            torch.device("cpu"),
            torch.float32,
            ["chat"],
            [1234],
            cfg,
            existing_entry=None,
            work_keys=None,
        )

    providers_for_cfg.assert_called_once()
    seed_body = evals["chat"]["target_real"]["configs"]["pooled"]["seeds"]["1234"]
    assert seed_body
    assert set(chat_mod.ALL_CHAT_TASKS).issubset(seed_body.keys())
    assert "chatcore_metric" in seed_body
    assert "results" not in seed_body
    assert "seeds" not in seed_body
    assert seed_body["chatcore_metric"] == pytest.approx(_expected_chatcore(seed_body))
    assert provider.call_count == 2 * len(chat_mod.ALL_CHAT_TASKS)
