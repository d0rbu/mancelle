"""Offline HTTP-to-bank-to-both-consumers integration, with exact SAE span reduction.

The provider, activation forward and target text generation are deterministic
fixtures. Separate QC/target tests exercise real TransformerLens and native KV
engines; this test verifies their orchestration and persisted cross-module data.
"""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import tiktoken
import torch as t
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from steering.eval.axbench_base import extraction
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_negative import qc
from steering.eval.axbench_negative.api import ConversationQualityJudge
from steering.eval.axbench_negative.artifacts import load_negative_source
from steering.eval.axbench_negative.build import create_chat_api, run_negative_build
from steering.eval.axbench_negative.consumer import EvaluationConfig, evaluate_negative_source
from steering.eval.axbench_negative.evaluate import PRESENCE_RUBRIC, ConceptPresenceJudge
from steering.eval.axbench_negative.producer import ProducerConfig, build_negative_source
from steering.eval.axbench_negative.qc import ChatScorer
from steering.eval.axbench_negative.source import (
    SOURCE_FORMAT as ORIGINAL_SOURCE_FORMAT,
)
from steering.eval.axbench_negative.source import (
    FeatureTarget,
    canonical_bytes,
    file_sha256,
    load_source_candidates,
)
from steering.eval.chat_generation import ChatGenerationResult


class OffsetTokenizer:
    bos_token_id = 0

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class FixtureTarget:
    def __init__(self, family):
        self.family = family
        self.methods = ("DiffMean",) if family == "native" else ("SelfConditioning",)
        self.identity = {"family": family, "fixture": "deterministic-text"}
        self.calls = []

    def generate(self, group, *, method, **kwargs):
        self.calls.append((group.messages, method, kwargs))
        text = "Rich useful answer" if method == "Control" else "clear useful answer"
        return ChatGenerationResult(text, tuple(map(ord, text)), False)


class TokenIdentityModel(t.nn.Module):
    """Loaded-weight stand-in; retains the real batched extractor and SAE path."""

    def __init__(self, tokenizer):
        super().__init__()
        self.anchor = t.nn.Parameter(t.zeros(1))
        self.cfg = SimpleNamespace(n_ctx=512)
        self.tokenizer = tokenizer

    def run_with_cache(self, tokens, names_filter, **_kwargs):
        return None, {names_filter[0]: tokens.unsqueeze(-1).float()}


class ExactCharacterSAE(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = t.nn.Parameter(t.zeros(1))
        self.cfg = SimpleNamespace(d_sae=16384, hook_name="blocks.20.hook_resid_post")

    def encode(self, activations):
        values = t.zeros((len(activations), 16384), device=activations.device)
        values[:, 0] = (activations[:, 0] == ord("R") + 2).float()
        return values


def test_original_messages_to_verified_bank_and_both_family_reports(tmp_path, monkeypatch):
    messages = [
        {"role": "system", "content": "Original system policy."},
        {"role": "user", "content": "Remember the equipment context."},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Give me useful options.  \n"},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "source_id": "source0",
                "group_id": "family0",
                "split": "test",
                "source_row": 0,
                "messages": messages,
            }
        )
        + "\n"
    )
    candidate = load_source_candidates(source)[-1]
    renderer = RustBPETokenizer(
        tiktoken.Encoding(
            name="integration-byte-renderer",
            pat_str=r"(?s).",
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
        ),
        "<bos>",
    )
    scorer = ChatScorer(
        model=SimpleNamespace(cfg=SimpleNamespace(n_ctx=1024), tokenizer=OffsetTokenizer()),
        sae=SimpleNamespace(cfg=SimpleNamespace(d_sae=1, hook_name="blocks.0.hook_resid_post")),
        render_tokenizer=renderer,
        layer=0,
        context_limit=1024,
        identity={"fixture": "token-activation"},
    )

    def activation_forward(*, token_id_lists, per_item_reduce, **_kwargs):
        # All prompt R activations remain present in the forward, proving that
        # the completion mask, rather than prefix removal, excludes them.
        return [
            per_item_reduce((t.tensor(ids) == ord("R")).float().unsqueeze(1), index)
            for index, ids in enumerate(token_id_lists)
        ]

    monkeypatch.setattr(extraction, "_run_batched_extractor", activation_forward)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-integration-key")
    calls: list[dict[str, Any]] = []
    neutral_responses = ["Rich first answer", "Rich second answer", "clear ordinary answer"]

    def post(_session, _url, *, json, **_kwargs):
        calls.append(json)
        if json["model"] == "neutral":
            assert json["messages"] == messages
            body = neutral_responses.pop(0)
        elif json["model"] == "avoiding":
            body = "clear useful avoiding answer"
        elif json["messages"][0]["content"] == PRESENCE_RUBRIC:
            data = json_load(json["messages"][1]["content"])
            assert data["conversation"] == messages
            body = json_dump({"concept_present": "R" in data["response"], "reason": "fixture"})
        else:
            data = json_load(json["messages"][1]["content"])
            assert data["conversation"] == messages
            body = json_dump(
                {
                    "useful": True,
                    "refusal": False,
                    "coherent": True,
                    "substantive": True,
                    "task_fulfilled": True,
                    "reason": "fixture",
                }
            )
        raw = {
            "model": json["model"],
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": body}}
            ],
        }
        return SimpleNamespace(status_code=200, json=lambda: raw)

    json_load, json_dump = json.loads, json.dumps
    monkeypatch.setattr(requests.Session, "post", post)
    with SynthResponseCache(tmp_path / "responses.sqlite") as cache:

        def api(model):
            return create_chat_api(
                {
                    "model": model,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 64,
                    "seed": None,
                    "max_retries": 0,
                    "requests_per_minute": 0,
                    "transport_retry_budget_seconds": 0.0,
                },
                cache,
            )

        judge_api = api("judge")
        quality = ConversationQualityJudge(judge_api)
        bank = build_negative_source(
            candidates=[candidate],
            targets=[FeatureTarget(0, (0,), "R concept")],
            scorer=scorer,
            neutral_api=api("neutral"),
            avoiding_api=api("avoiding"),
            quality_judge=quality,
            config=ProducerConfig(
                n_verification_seeds=3,
                min_consistency_passes=2,
                max_concurrent=2,
                avoid_max_attempts=1,
            ),
            output_root=tmp_path / "bank",
            semantic_identity={"fixture": "full-pipeline"},
        )
        assert bank.counts["neutral_samples"] == 3 and len(bank.groups) == 1
        assert bank.groups[0].messages == tuple(messages)
        assert (
            scorer.assess_completion(
                candidate, bank.groups[0].avoiding_response, FeatureTarget(0, (0,), "R concept")
            ).max_activation
            == 0
        )
        bank = load_negative_source(bank.root)
        for family in ("native", "encdec"):
            target = FixtureTarget(family)
            config = EvaluationConfig(
                output_dir=tmp_path / family, magnitudes=(1.0,), max_new_tokens=64
            )
            result = evaluate_negative_source(
                bank, target, scorer, quality, ConceptPresenceJudge(judge_api), config
            )
            assert result["complete"] and result["summary"]["declared_cells"] == 2
            suppressor = next(
                arm for arm in result["summary"]["arms"] if arm["method"] != "Control"
            )
            assert suppressor["paired_active_change"] == {"n": 1, "mean": -1.0}
            assert suppressor["paired_useful_change"] == {"n": 1, "mean": 0.0}
            assert suppressor["paired_concept_presence_change"] == {"n": 1, "mean": -1.0}
            assert all(call[0] == tuple(messages) for call in target.calls)
            assert target.calls[0][2]["seed"] == target.calls[1][2]["seed"]
            before = len(calls)
            replay = evaluate_negative_source(
                bank, target, scorer, quality, ConceptPresenceJudge(judge_api), config
            )
            assert replay == result and len(target.calls) == 2 and len(calls) == before


def test_resolved_build_cli_wires_source_manifest_admission_qc_http_and_resume(
    tmp_path, monkeypatch
):
    """Run the production launcher; only downloaded weights and HTTP are fixtures."""
    messages = [
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "Relevant prior answer"},
        {"role": "user", "content": "give useful options"},
        {"role": "assistant", "content": "future answer must never reach neutral API"},
    ]
    source = tmp_path / "source.jsonl"
    source.write_bytes(
        canonical_bytes(
            {
                "source_id": "local/original-row0",
                "group_id": "family0",
                "source_row": 0,
                "split": "test",
                "messages": messages,
            }
        )
    )
    candidates = load_source_candidates(source)
    assert len(candidates) == 2
    manifest = tmp_path / "source_manifest.json"
    manifest_document = {
        "format": ORIGINAL_SOURCE_FORMAT,
        "complete": True,
        "source": {"path": source.name, "sha256": file_sha256(source), "rows": 1},
    }
    manifest.write_bytes(canonical_bytes(manifest_document))
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_bytes(
        canonical_bytes(
            {
                "concept_id": 0,
                "concept": "R concept",
                "ref": "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/0",
            }
        )
    )
    native_tokenizer_path = tmp_path / "tokenizer" / "tokenizer.pkl"
    native_tokenizer_path.parent.mkdir()
    native_encoding = tiktoken.Encoding(
        name="build-cli-byte-tokenizer",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
    )
    with native_tokenizer_path.open("wb") as handle:
        pickle.dump(native_encoding, handle)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    target_meta, target_model = checkpoint / "meta_000001.json", checkpoint / "model_000001.pt"
    target_meta.write_text(json.dumps({"step": 1, "model_config": {
                                          "sequence_len": 1024, "n_layer": 14, "n_embd": 896},
                                      "sft_loss_mask": "assistant",
                                      "user_config": {"mixture": "fixture"}}))
    target_model.write_bytes(b"Target weights are hashed, never loaded by source construction.")

    backend = Tokenizer(
        models.WordLevel(
            vocab={"<bos>": 0, "<unk>": 1, **{chr(i): i + 2 for i in range(128)}},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", unk_token="<unk>"
    )
    weight_loads = []

    def load_weights(**kwargs):
        weight_loads.append(kwargs)
        return TokenIdentityModel(tokenizer), ExactCharacterSAE()

    monkeypatch.setattr(qc, "load_dense_extraction_stack", load_weights)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-cli-integration-key")
    calls = []
    neutral = iter(["Rich first answer", "Rich second answer", "clear ordinary answer"])

    def post(_session, _url, *, json, **_kwargs):
        calls.append(json)
        if json["model"] == "neutral":
            assert json["messages"] == messages[:3]
            content = next(neutral)
        elif json["model"] == "avoiding":
            content = "clear useful answer"
        else:
            assert json["model"] == "quality"
            data = json_load(json["messages"][1]["content"])
            assert data["conversation"] == messages[:3]
            content = json_dump(
                {
                    "useful": True,
                    "refusal": False,
                    "coherent": True,
                    "substantive": True,
                    "task_fulfilled": True,
                    "reason": "fixture",
                }
            )
        raw = {
            "id": f"response-{len(calls)}",
            "model": json["model"],
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": content}}
            ],
        }
        return SimpleNamespace(status_code=200, json=lambda: raw)

    json_load, json_dump = json.loads, json.dumps
    monkeypatch.setattr(requests.Session, "post", post)
    api = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "seed": None,
        "requests_per_minute": 0,
        "max_retries": 0,
        "transport_retry_budget_seconds": 0.0,
    }
    cfg = OmegaConf.create(
        {
            "source_path": str(source),
            "source_sha256": file_sha256(source),
            "source_manifest_path": str(manifest),
            "metadata_path": str(metadata),
            "metadata_sha256": file_sha256(metadata),
            "concept_ids": [0],
            "render_tokenizer_path": str(native_tokenizer_path),
            "render_tokenizer_sha256": file_sha256(native_tokenizer_path),
            "reserved_extraction_tokens": 32,
            "response_cache": str(tmp_path / "responses.sqlite"),
            "output_root": str(tmp_path / "bank"),
            "target_admission": [
                {
                    "family": "native",
                    "meta_path": str(target_meta),
                    "tokenizer_path": str(native_tokenizer_path),
                    "max_new_tokens": 24,
                    "feature_statistics_path": None,
                    "pretrain_meta_path": None,
                    "expected_meta_sha256": file_sha256(target_meta),
                    "expected_model_sha256": file_sha256(target_model),
                    "expected_tokenizer_sha256": file_sha256(native_tokenizer_path),
                }
            ],
            "extractor": {
                "model": "gemma-2-2b",
                "sae_release": "gemmascope",
                "sae_width": "16k",
                "sae_l0": None,
                "layer": 20,
                "activation_type": "residual",
                "device": "cpu",
                "batch_size": 1,
                "context_limit": 512,
            },
            "producer": {
                "n_verification_seeds": 3,
                "min_consistency_passes": 2,
                "max_concurrent": 1,
                "avoid_max_attempts": 1,
                "max_candidates_per_feature": 1,
            },
            "neutral": {**api, "model": "neutral"},
            "avoiding": {**api, "model": "avoiding"},
            "quality": {**api, "model": "quality"},
        }
    )
    # The preparation manifest must authenticate the source before model/API work.
    manifest.write_bytes(canonical_bytes({**manifest_document, "source": {"sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="manifest does not bind"):
        run_negative_build(cfg)
    assert calls == weight_loads == []
    manifest.write_bytes(canonical_bytes(manifest_document))
    bank = run_negative_build(cfg)
    assert bank.counts["candidates"] == 2 and bank.counts["screened_pairs"] == 1
    assert bank.counts["neutral_samples"] == 3 and bank.counts["neutral_admitted_pairs"] == 1
    assert len(bank.groups) == 1 and bank.groups[0].candidate_id == candidates[1].candidate_id
    assert bank.groups[0].messages == tuple(messages[:3])
    assert len([call for call in calls if call["model"] == "neutral"]) == 3
    assert "future answer must never reach neutral API" not in json_dump(calls)
    admission = bank.identity["context_admission"]
    assert admission == {
        "candidate_count": 2,
        "admitted_count": 2,
        "rejections": [],
        "extraction_rejections": [],
        "reserved_extraction_tokens": 32,
    }
    assert bank.identity["source_preparation"]["sha256"] == file_sha256(manifest)
    assert bank.identity["targets"][0]["max_new_tokens"] == 24
    assert bank.identity["targets"][0]["model_sha256"] == file_sha256(target_model)
    assert bank.identity["scorer"]["context_limit"] == 512
    assert weight_loads[0]["sae_model"] == "gemma-2-2b"
    before = len(calls)
    assert run_negative_build(cfg) == bank
    assert len(calls) == before
