from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.eval import harmfulness_eval
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
)


class _Tokenizer:
    def render_for_completion(self, conversation: dict) -> list[int]:
        prompt = conversation["messages"][0]["content"]
        return [0, *(ord(char) for char in prompt), 2]

    def encode_special(self, token: str) -> int:
        assert token == "<assistant_end>"
        return 3

    def decode(self, token_ids) -> str:
        return "".join(chr(int(token)) for token in token_ids if int(token) > 3)


class _EncDecStub(EncoderDecoderGPT):
    def __init__(self) -> None:
        t.nn.Module.__init__(self)
        self.encoder = SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=TEST_CACHE_N_SAE_FEATURES,
                num_buckets=TEST_CACHE_NUM_BUCKETS,
            ),
            feature_vocab=None,
        )
        self.dec_config = SimpleNamespace(sequence_len=64)


def _cfg():
    return OmegaConf.create(
        {
            "source": "encdec",
            "model_tag": "d-test",
            "step": 1,
            "judge_only": False,
            "responses_file": None,
            "feature_configs": None,
            "output_file": None,
            "num_harmful": None,
            "num_benign": 1,
            "seed": 17,
            "batch_size": 8,
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_k": 1,
            "judge_model": "test",
            "cache_path": "",
            "requests_per_minute": 1,
        }
    )


def _prompt_pairs() -> list[dict[str, str]]:
    return [
        {"prompt": "harmful", "category": "Violence", "pool": "harmful"},
        {"prompt": "benign", "category": "benign", "pool": "benign"},
    ]


def _provider(
    cache_dir: Path,
    tokenizer: _Tokenizer,
    pairs: list[dict[str, str]],
) -> SafetensorsContextFeatureProvider:
    entries = []
    segments = (ContextFeatureSegment((1,), (1,), 0, 1),)
    for pair in pairs:
        _prompt_text, token_ids = harmfulness_eval._render_harmfulness_prompt_for_coverage(
            tokenizer,
            pair["prompt"],
        )
        entries.append(ContextCacheEntry.from_sequences(token_ids, segments))
    save_context_cache(
        cache_dir,
        entries,
        identity=TEST_CACHE_IDENTITY,
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
    )
    return SafetensorsContextFeatureProvider(
        cache_dir,
        identity=TEST_CACHE_IDENTITY,
    )


def _engine_for_provider(
    provider: SafetensorsContextFeatureProvider,
    tokenizer: _Tokenizer,
    *,
    fail_after_lookup: bool,
) -> MagicMock:
    wrapper = TargetAwareEncoderDecoderEvalWrapper.__new__(TargetAwareEncoderDecoderEvalWrapper)
    t.nn.Module.__init__(wrapper)
    wrapper.context_provider = provider

    def generate_batch(prompts: list[str], **_kwargs) -> list[str]:
        for prompt in prompts:
            _prompt_text, token_ids = harmfulness_eval._render_harmfulness_prompt_for_coverage(
                tokenizer,
                prompt,
            )
            wrapper._lookup_context_segments(token_ids)
        if fail_after_lookup:
            raise AssertionError("generation must not start")
        return [f"response to {prompt}" for prompt in prompts]

    return MagicMock(generate_batch=MagicMock(side_effect=generate_batch))


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    model,
    tokenizer: _Tokenizer,
    provider: SafetensorsContextFeatureProvider | None,
    engine: MagicMock,
) -> None:
    cfg = _cfg()
    is_encdec = isinstance(model, EncoderDecoderGPT)
    cfg.feature_configs = None
    cfg.self_cond_segment_size = 256 if is_encdec else None
    cfg.batch_size = 1 if is_encdec else 8
    monkeypatch.setattr(
        harmfulness_eval,
        "initialize_config_dir",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(harmfulness_eval, "compose", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        harmfulness_eval,
        "_load_model_for_harmfulness",
        lambda _cfg, _device: (
            model,
            tokenizer,
            {},
            object() if is_encdec else None,
            {
                "format": "nanochat_checkpoint",
                "backend": "nanochat",
                "architecture": type(model).__name__,
                "model_type": "encoder_decoder" if is_encdec else "decoder_only",
                "step": 1,
                "checkpoint_sha256": "1" * 64,
                "tokenizer_sha256": "2" * 64,
            },
        ),
    )
    monkeypatch.setattr(harmfulness_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        harmfulness_eval,
        "_is_encdec_root",
        lambda source: source == "encdec" and is_encdec,
    )
    monkeypatch.setattr(
        harmfulness_eval,
        "_load_expected_prompt_pairs",
        lambda _cfg: _prompt_pairs(),
    )
    monkeypatch.setattr(harmfulness_eval, "InferenceEngine", lambda *_args: engine)
    setup = MagicMock(context_provider=provider, segment_size=256, safety_strength=0.0, safety_steering=None, assert_target_unchanged=MagicMock())
    monkeypatch.setattr(
        harmfulness_eval,
        "prepare_harmfulness_self_conditioning",
        lambda _cfg, **_kwargs: setup,
    )

    def generate_self_conditioned(*, prompt: str, **_kwargs) -> str:
        return engine.generate_batch([prompt])[0]

    monkeypatch.setattr(
        harmfulness_eval,
        "generate_harmfulness_response_self_conditioned",
        generate_self_conditioned,
    )

    def get_responses(
        cfg,
        condition_name,
        response_generator,
        *,
        generation_conditioning,
        generation_provenance,
    ):
        assert response_generator is not None
        assert generation_conditioning is not None
        assert generation_provenance is not None
        pairs = harmfulness_eval._generate_responses(
            cfg,
            str(tmp_path / "responses.yaml"),
            condition_name=condition_name,
            response_generator=response_generator,
            conditioning=generation_conditioning,
            generation=generation_provenance,
        )
        return (
            condition_name,
            str(tmp_path / "results.yaml"),
            pairs,
            generation_conditioning,
            generation_provenance,
        )

    monkeypatch.setattr(harmfulness_eval, "_get_responses", get_responses)
    monkeypatch.setattr(
        harmfulness_eval,
        "current_harmfulness_payload",
        lambda *_args, **_kwargs: ({"current": True}, "current"),
    )
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(harmfulness_eval.sys, "argv", ["harmfulness_eval.py"])

    harmfulness_eval.main()


def test_missing_cache_rows_fail_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    provider = _provider(tmp_path / "cache", tokenizer, _prompt_pairs()[:1])
    engine = _engine_for_provider(
        provider,
        tokenizer,
        fail_after_lookup=True,
    )

    with pytest.raises(ContextCacheValidationError, match="1 of 2"):
        _run_main(
            monkeypatch,
            tmp_path,
            model=_EncDecStub(),
            tokenizer=tokenizer,
            provider=provider,
            engine=engine,
        )

    engine.generate_batch.assert_not_called()


def test_complete_cache_proceeds_to_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    provider = _provider(tmp_path / "cache", tokenizer, _prompt_pairs())
    engine = _engine_for_provider(
        provider,
        tokenizer,
        fail_after_lookup=False,
    )

    _run_main(
        monkeypatch,
        tmp_path,
        model=_EncDecStub(),
        tokenizer=tokenizer,
        provider=provider,
        engine=engine,
    )

    assert engine.generate_batch.call_count == 2


def test_decoder_only_generation_skips_prompt_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    engine = MagicMock()
    engine.generate_batch.return_value = ["response to harmful", "response to benign"]

    _run_main(
        monkeypatch,
        tmp_path,
        model=t.nn.Module(),
        tokenizer=tokenizer,
        provider=None,
        engine=engine,
    )

    engine.generate_batch.assert_called_once()
