"""Real-model execution tests for the nanochat-token generation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch as t
import yaml

from steering.eval.axbench_base.config import GenerationConfig
from steering.eval.axbench_base.generation_helpers import (
    GenerationResult,
    _generate_samples_with_nanochat_tokens,
    _generate_with_nanochat_tokens,
)
from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.inference_engine import build_engine_for_features
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer


class _FullPrefixContextProvider:
    """Return one real full-prefix context segment for tiny encdec tests."""

    def __init__(self, tokenizer: MockTokenizer) -> None:
        self._tokenizer = tokenizer

    def get(
        self,
        token_ids: Sequence[int],
    ) -> tuple[ContextFeatureSegment, ...] | None:
        normalized = [int(token) for token in token_ids]
        if not normalized:
            return None
        char_end = token_to_char_offsets(self._tokenizer, normalized)[-1]
        if char_end <= 0:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


def _prompt(tokenizer: MockTokenizer) -> list[int]:
    encoded = tokenizer.encode("shared prompt", prepend=tokenizer.get_bos_token_id())
    assert isinstance(encoded[0], int)
    return cast(list[int], encoded)


def _expected_results(
    tokenizer: MockTokenizer,
    rows: list[list[int]],
    *,
    max_new_tokens: int,
) -> list[GenerationResult]:
    return [
        GenerationResult(
            text=tokenizer.decode(row),
            token_count=len(row),
            stop_reason=("max_tokens" if len(row) >= max_new_tokens else "bos_token"),
        )
        for row in rows
    ]


def _counter_totals(profile_dir: Path) -> dict[str, float]:
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    return {str(row["name"]): float(row["total"]) for row in summary["counters"]}


def _region_names(profile_dir: Path) -> set[str]:
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    return {str(row["name"]) for row in summary["regions"]}


@pytest.mark.parametrize(
    ("seed", "expected_count", "expected_reason"),
    [
        pytest.param(0, 3, "max_tokens", id="budget-boundary"),
        pytest.param(146, 0, "bos_token", id="bos-stop"),
    ],
)
def test_sequential_helper_matches_real_engine_and_profiles_contract(
    gpt_model,
    mock_tokenizer: MockTokenizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seed: int,
    expected_count: int,
    expected_reason: str,
) -> None:
    # Uniform real-GPT logits make the two seeded stop boundaries stable while
    # retaining the actual attention/KV-cache forward path.
    for parameter in gpt_model.parameters():
        parameter.data.zero_()
    prompt = [2, 3]
    gen_cfg = GenerationConfig(
        max_new_tokens=3,
        temperature=1.0,
        top_k_sampling=128,
    )
    engine = build_engine_for_features(
        gpt_model,
        mock_tokenizer,
        t.device("cpu"),
        features=None,
        context_provider=None,
    )
    expected_tokens = list(
        engine.generate_stream(
            token_ids=prompt,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k_sampling,
            seed=seed,
        )
    )
    decoded_rows: list[list[int]] = []
    original_decode = mock_tokenizer.decode

    def recording_decode(tokens: list[int]) -> str:
        decoded_rows.append(list(tokens))
        return original_decode(tokens)

    monkeypatch.setattr(mock_tokenizer, "decode", recording_decode)

    profile_dir = tmp_path / f"sequential-helper-{seed}"
    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = _generate_with_nanochat_tokens(
            model=gpt_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            nanochat_tokens=prompt,
            sampling_seed=seed,
            gen_cfg=gen_cfg,
            encoder_input=None,
            context_provider=None,
        )

    assert actual.token_count == len(expected_tokens) == expected_count
    assert decoded_rows == [expected_tokens]
    assert actual.text == original_decode(expected_tokens)
    assert actual.stop_reason == expected_reason
    counters = _counter_totals(profile_dir)
    assert counters["axbench_base.generation.requests"] == 1
    assert counters["axbench_base.generation.prompt_tokens"] == len(prompt)
    assert counters["axbench_base.generation.engine_constructions"] == 1
    assert counters["axbench_base.generation.generated_tokens"] == expected_count
    regions = _region_names(profile_dir)
    assert "stage4.generate_stream" in regions
    assert "axbench_base.generation.prefill" in regions
    assert "axbench_base.generation.decode" in regions


def test_decoder_only_results_match_direct_batched_engine(
    gpt_model,
    mock_tokenizer: MockTokenizer,
) -> None:
    prompt = _prompt(mock_tokenizer)
    seed = 7
    n_samples = 3
    gen_cfg = GenerationConfig(
        max_new_tokens=4,
        temperature=0.7,
        top_k_sampling=32,
    )
    engine = build_engine_for_features(
        gpt_model,
        mock_tokenizer,
        t.device("cpu"),
        features=None,
        context_provider=None,
    )
    expected_rows = engine.generate_samples(
        token_ids=prompt,
        seed=seed,
        n_samples=n_samples,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_k=gen_cfg.top_k_sampling,
    )
    expected = _expected_results(
        mock_tokenizer,
        expected_rows,
        max_new_tokens=gen_cfg.max_new_tokens,
    )

    actual = _generate_samples_with_nanochat_tokens(
        model=gpt_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        nanochat_tokens=prompt,
        sampling_seed=seed,
        n_samples=n_samples,
        gen_cfg=gen_cfg,
        encoder_input=None,
        context_provider=None,
    )

    assert actual == expected
    assert [result.text for result in actual] == [
        mock_tokenizer.decode(row) for row in expected_rows
    ]


def test_stop_reason_maps_budget_early_stop_and_empty_rows(
    gpt_model,
    mock_tokenizer: MockTokenizer,
) -> None:
    for parameter in gpt_model.parameters():
        parameter.data.zero_()
    gen_cfg = GenerationConfig(
        max_new_tokens=3,
        temperature=1.0,
        top_k_sampling=128,
    )

    actual = _generate_samples_with_nanochat_tokens(
        model=gpt_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        nanochat_tokens=[2, 3],
        sampling_seed=790,
        n_samples=3,
        gen_cfg=gen_cfg,
        encoder_input=None,
        context_provider=None,
    )

    assert [result.token_count for result in actual] == [3, 2, 0]
    assert [result.stop_reason for result in actual] == [
        "max_tokens",
        "bos_token",
        "bos_token",
    ]
    assert actual[-1].text == ""


def test_wrapped_encdec_results_match_direct_batched_engine(
    encdec_model,
    mock_tokenizer: MockTokenizer,
) -> None:
    prompt = _prompt(mock_tokenizer)
    seed = 7
    n_samples = 3
    encoder_input = {5: 2}
    context_provider = _FullPrefixContextProvider(mock_tokenizer)
    gen_cfg = GenerationConfig(
        max_new_tokens=4,
        temperature=0.7,
        top_k_sampling=32,
    )
    engine = build_engine_for_features(
        encdec_model,
        mock_tokenizer,
        t.device("cpu"),
        features=encoder_input,
        context_provider=context_provider,
    )
    expected_rows = engine.generate_samples(
        token_ids=prompt,
        seed=seed,
        n_samples=n_samples,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_k=gen_cfg.top_k_sampling,
    )

    actual = _generate_samples_with_nanochat_tokens(
        model=encdec_model,
        tokenizer=mock_tokenizer,
        device="cpu",
        nanochat_tokens=prompt,
        sampling_seed=seed,
        n_samples=n_samples,
        gen_cfg=gen_cfg,
        encoder_input=encoder_input,
        context_provider=context_provider,
    )

    assert actual == _expected_results(
        mock_tokenizer,
        expected_rows,
        max_new_tokens=gen_cfg.max_new_tokens,
    )


def test_batched_helper_profiler_accounts_for_one_physical_prefill(
    gpt_model,
    mock_tokenizer: MockTokenizer,
    tmp_path: Path,
) -> None:
    prompt = _prompt(mock_tokenizer)
    seed = 7
    n_samples = 3
    gen_cfg = GenerationConfig(
        max_new_tokens=3,
        temperature=0.7,
        top_k_sampling=32,
    )
    profile_dir = tmp_path / "profile"

    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        actual = _generate_samples_with_nanochat_tokens(
            model=gpt_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            nanochat_tokens=prompt,
            sampling_seed=seed,
            n_samples=n_samples,
            gen_cfg=gen_cfg,
            encoder_input=None,
            context_provider=None,
        )

    counters = _counter_totals(profile_dir)
    regions = _region_names(profile_dir)
    assert counters["axbench_base.generation.requests"] == n_samples
    assert counters["axbench_base.generation.prompt_tokens"] == len(prompt)
    assert counters["axbench_base.generation.engine_constructions"] == 1
    assert counters["axbench_base.generation.generated_tokens"] == sum(
        result.token_count for result in actual
    )
    assert "stage4.generate_samples" in regions
    assert "stage4.generate_stream" not in regions
