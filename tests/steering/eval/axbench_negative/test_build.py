"""Admission and configuration must fail before any paid sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from omegaconf import OmegaConf

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_negative import build
from steering.eval.axbench_negative.qc import ChatScorer
from steering.eval.axbench_negative.source import load_source_candidates
from steering.eval.axbench_upstream.native_decoder import checkpoint_identity


class Tokenizer:
    def render_conversation(self, conversation, max_tokens=2048):
        assert max_tokens is None
        ids = [0]
        for message in conversation["messages"]:
            ids.extend([5, *[ord(char) for char in message["content"]], 6])
        return ids, [0] * len(ids)

    def encode_special(self, value):
        return {"<assistant_start>": 7, "<assistant_end>": 8}[value]


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path, text):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"source_id": "row0", "group_id": "family0", "split": "test",
                                "source_row": 0, "messages": [{"role": "user", "content": text}]}) + "\n")
    return load_source_candidates(path, _hash(path))


def _target(tmp_path, family, sequence_len):
    root = tmp_path / family
    root.mkdir()
    meta = {"step": 1, "model_config": {"sequence_len": sequence_len}}
    if family == "encdec":
        meta["encoder_config"] = {}
    else:
        meta["model_config"].update(n_layer=12, n_embd=768)
        meta.update(sft_loss_mask="assistant", user_config={"mixture": "fixture"})
    path = root / "meta_000001.json"
    path.write_text(json.dumps(meta))
    weights = root / "model_000001.pt"
    weights.write_bytes(b"test checkpoint identity")
    tokenizer = root / "tokenizer.pkl"
    tokenizer.write_bytes(b"test tokenizer identity")
    return {"family": family, "meta_path": str(path), "tokenizer_path": str(tokenizer),
            "max_new_tokens": 4, "expected_meta_sha256": _hash(path),
            "expected_model_sha256": _hash(weights), "expected_tokenizer_sha256": _hash(tokenizer),
            "feature_statistics_path": "fixture_stats" if family == "encdec" else None,
            "pretrain_meta_path": None}


def test_common_admission_uses_both_targets_without_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: Tokenizer())
    monkeypatch.setattr(build, "load_conditioning_admission", lambda *_: SimpleNamespace(
        identity={"context_window": 100, "extraction_tokenizer_sha256": "fixture"},
        prompt_token_count=lambda *_: 20))
    candidates = _source(tmp_path, "abcdef")  # 10 prompt tokens including assistant opener.
    native = _target(tmp_path, "native", 1024)
    encdec = _target(tmp_path, "encdec", 14)
    accepted, contracts, rejected = build.admit_target_contexts(candidates, [native, encdec])
    assert accepted == candidates and rejected == []
    assert contracts[1]["model_sha256"] == encdec["expected_model_sha256"]
    encdec["max_new_tokens"] = 5
    accepted, _, rejected = build.admit_target_contexts(candidates, [native, encdec])
    assert accepted == []
    assert rejected == [{"candidate_id": candidates[0].candidate_id, "reasons": [
        {"family": "encdec", "prompt_tokens": 10, "reason": "target_context_overflow"}]}]
    assert candidates[0].messages[0]["content"] == "abcdef"


@pytest.mark.parametrize("field", ["expected_meta_sha256", "expected_model_sha256", "expected_tokenizer_sha256"])
def test_target_identity_mismatch_fails_before_render(tmp_path, monkeypatch, field):
    target = _target(tmp_path, "native", 1024)
    target[field] = "0" * 64
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="content hash differs"):
        build.admit_target_contexts(_source(tmp_path, "abc"), [target])


def test_target_family_and_duplicate_family_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: Tokenizer())
    target = _target(tmp_path, "native", 1024)
    with pytest.raises(ValueError, match="unique"):
        build.admit_target_contexts([], [target, target])
    target["family"] = "encdec"
    with pytest.raises(ValueError, match="family differs"):
        build.admit_target_contexts([], [target])


@pytest.mark.parametrize("fault,expected", [
    ("missing_sft_mask", "explicit SFT checkpoint step"),
    ("wrong_step", "explicit SFT checkpoint step"),
    ("cross_attention", "decoder-only model_config"),
    ("wrong_backend", "native SFT mixture provenance"),
    ("missing_mixture", "native SFT mixture provenance"),
])
def test_native_ineligible_checkpoint_rejected_before_weights_and_tokenizer(
    tmp_path, monkeypatch, fault, expected,
):
    target = _target(tmp_path, "native", 1024)
    path = Path(target["meta_path"])
    metadata = json.loads(path.read_text())
    if fault == "missing_sft_mask":
        del metadata["sft_loss_mask"]
    elif fault == "wrong_step":
        metadata["step"] = 2
    elif fault == "cross_attention":
        metadata["cross_attention_config"] = {}
    elif fault == "wrong_backend":
        metadata["user_config"]["model_backend"] = "huggingface"
    else:
        del metadata["user_config"]["mixture"]
    path.write_text(json.dumps(metadata))
    target["expected_meta_sha256"] = _hash(path)
    path.with_name("model_000001.pt").unlink()
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: pytest.fail("not eligible"))
    with pytest.raises(ValueError, match=expected):
        build.admit_target_contexts([], [target])


@pytest.mark.parametrize("n_layer,n_embd", [(12, 768), (14, 896)])
def test_native_admission_identity_matches_existing_checkpoint_identity(
    tmp_path, monkeypatch, n_layer, n_embd,
):
    target = _target(tmp_path, "native", 1024)
    metadata = Path(target["meta_path"])
    document = json.loads(metadata.read_text())
    document["model_config"].update(n_layer=n_layer, n_embd=n_embd)
    metadata.write_text(json.dumps(document))
    target["expected_meta_sha256"] = _hash(metadata)
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: Tokenizer())
    _accepted, contracts, _rejected = build.admit_target_contexts([], [target])
    existing = checkpoint_identity(metadata.parent, 1, Path(target["tokenizer_path"]))
    assert existing == {
        "checkpoint_dir": str(metadata.parent.resolve()), "step": 1,
        "tokenizer_path": str(Path(target["tokenizer_path"]).resolve()),
        "model_sha256": target["expected_model_sha256"],
        "meta_sha256": target["expected_meta_sha256"],
        "tokenizer_sha256": target["expected_tokenizer_sha256"],
    }
    for key in ("model_sha256", "meta_sha256", "tokenizer_sha256"):
        assert contracts[0][key] == existing[key]


@pytest.mark.parametrize("geometry", [
    {"n_layer": 13}, {"n_embd": 512}, {"sequence_len": 2048}, {"n_layer": None},
])
def test_build_rejects_unreviewed_native_geometry_before_scorer_or_api(
    tmp_path, monkeypatch, geometry,
):
    target = _target(tmp_path, "native", 1024)
    metadata = Path(target["meta_path"])
    document = json.loads(metadata.read_text())
    document["model_config"].update(geometry)
    metadata.write_text(json.dumps(document))
    target["expected_meta_sha256"] = _hash(metadata)

    def forbidden(*_args, **_kwargs):
        pytest.fail("unsupported native geometry reached tokenizer, scorer or API")

    monkeypatch.setattr(build, "load_source_candidates", lambda *_: [])
    monkeypatch.setattr(build, "load_feature_targets", lambda *_: ([], "fixture"))
    monkeypatch.setattr(build, "load_render_tokenizer", forbidden)
    monkeypatch.setattr(build, "build_chat_scorer", forbidden)
    monkeypatch.setattr(build, "create_chat_api", forbidden)
    cfg = OmegaConf.create({
        "source_path": "unused", "source_sha256": "unused", "source_manifest_path": None,
        "metadata_path": "unused", "metadata_sha256": "unused", "concept_ids": [0],
        "extractor": {}, "target_admission": [target],
    })
    with pytest.raises(AxBenchAdvError, match="geometry is not reviewed"):
        build.run_negative_build(cfg)


def test_extraction_headroom_is_separate_from_provider_token_budget(tmp_path):
    candidates = _source(tmp_path, "abcdef")
    scorer = cast(ChatScorer, SimpleNamespace(context_limit=12, prepare=lambda _: SimpleNamespace(token_ids=tuple(range(8)))))
    accepted, rejected = build.admit_extraction_contexts(candidates, scorer, 4)
    assert accepted == candidates and rejected == []
    accepted, rejected = build.admit_extraction_contexts(candidates, scorer, 5)
    assert accepted == []
    assert rejected[0]["reason"] == "extraction_reserved_context_overflow"
    assert rejected[0]["prompt_tokens"] == 8
    with pytest.raises(ValueError, match="reserved_extraction_tokens"):
        build.admit_extraction_contexts(candidates, scorer, True)


def _extractor():
    return {"model": "gemma-2-2b", "sae_release": "gemmascope", "sae_l0": None,
            "activation_type": "residual", "layer": 20, "sae_width": "16k"}


def _metadata(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text(json.dumps({"concept_id": 0, "concept": "renting equipment",
                                "ref": "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/16038"}) + "\n")
    return path


def test_concept_mapping_binds_namespace_and_exact_description(tmp_path):
    path = _metadata(tmp_path)
    targets, prefix = build.load_feature_targets(path, _hash(path), [0], _extractor())
    assert targets[0].to_dict() == {"feature_id": 16038, "member_ids": [16038], "description": "renting equipment"}
    assert prefix == "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k"
    with pytest.raises(ValueError, match="namespace differ"):
        build.load_feature_targets(path, _hash(path), [0], {**_extractor(), "layer": 12})
    with pytest.raises(ValueError, match="canonical GemmaScope"):
        build.load_feature_targets(path, _hash(path), [0], {**_extractor(), "sae_l0": 71})


@pytest.mark.parametrize("ids", [[], [True], [-1], [0, 0], [1]])
def test_invalid_concept_selection_fails(tmp_path, ids):
    path = _metadata(tmp_path)
    with pytest.raises(ValueError):
        build.load_feature_targets(path, _hash(path), ids, _extractor())


def _api_config():
    return {"model": "test/model", "temperature": 1.0, "top_p": 1.0, "max_tokens": 32,
            "seed": None, "requests_per_minute": 0, "max_retries": 2,
            "transport_retry_budget_seconds": 0.0}


@pytest.mark.parametrize(("key", "value"), [
    ("max_retries", True), ("requests_per_minute", -1),
    ("transport_retry_budget_seconds", float("nan")), ("temperature", -0.1), ("max_tokens", 0),
    ("provider", "other"), ("provider", None), ("endpoint", "https://other.invalid"),
])
def test_api_validation_happens_before_transport_construction(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(build, "OpenRouterLLMClient", lambda *_a, **_kw: pytest.fail("invalid config constructed transport"))
    with SynthResponseCache(tmp_path / "cache.sqlite") as cache, pytest.raises(ValueError):
        build.create_chat_api({**_api_config(), key: value}, cache)


@pytest.mark.parametrize("provider", [None, "openrouter", "openai"])
def test_api_configuration_preserves_explicit_sampling_and_retry_limits(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-live-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-live-openai-key")
    config = _api_config() if provider is None else {**_api_config(), "provider": provider}
    with SynthResponseCache(tmp_path / "cache.sqlite") as cache:
        api = build.create_chat_api(config, cache)
        assert api.identity["provider"] == (provider or "openrouter")
        assert api.identity["sampling"] == {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32}
        assert api.identity["transport_max_retries"] == 2
        assert api.identity["transport_retry_budget_seconds"] == 0.0


def test_new_hydra_profiles_default_to_direct_openai_and_avoidance_inherits_neutral():
    root = Path(__file__).resolve().parents[4] / "configs/eval/axbench_negative"
    cfg = OmegaConf.load(root / "build.yaml")
    assert cfg.neutral.provider == cfg.avoiding.provider == cfg.quality.provider == "openai"
    cfg.neutral.provider = "openrouter"
    assert cfg.avoiding.provider == "openrouter"
    assert cfg.quality.provider == "openai"
    assert OmegaConf.load(root / "eval.yaml").judge.provider == "openai"
