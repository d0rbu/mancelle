"""Contract tests for the lazy, deterministic prompt extraction service."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch as t

from scripts.chat_tui import extraction as extraction_module
from scripts.chat_tui.extraction import GemmaExtractionService
from scripts.chat_tui.setup import load_chat_setup, resolve_conditioning_seed
from steering.conditional.sampling import SamplingBundle
from steering.eval.axbench_upstream.methods import MethodSetup, PreparedPromptContext
from tests.test_chat_tui.test_setup_loader import (
    FIXTURES,
    REAL_STATISTICS,
    _fixture_meta,
    _nested_lineage,
)


class _Tokenizer:
    def encode_special(self, text: str) -> int:
        if text != "<assistant_end>":
            raise ValueError(f"unexpected special token: {text}")
        return 91


@pytest.fixture(scope="module")
def leaf_setup(tmp_path_factory: pytest.TempPathFactory) -> MethodSetup:
    if not REAL_STATISTICS.is_file():
        pytest.skip(f"real feature statistics are absent: {REAL_STATISTICS}")
    leaf_path, _ancestor_path = _nested_lineage(tmp_path_factory.mktemp("extraction-lineage"))
    return load_chat_setup(leaf_path, REAL_STATISTICS)


@pytest.fixture
def fake_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    stack_calls: list[tuple[Any, ...]] = []
    extraction_calls: list[tuple[Any, ...]] = []
    base_model = object()
    sae = object()

    def fake_load(setup: MethodSetup, device: str) -> tuple[Any, Any]:
        stack_calls.append((setup, device))
        return base_model, sae

    def fake_prepare(
        setup: MethodSetup,
        actual_base_model: Any,
        actual_sae: Any,
        tokenizer: Any,
        *,
        token_ids: tuple[int, ...],
        assistant_end_id: int,
    ) -> PreparedPromptContext:
        rng = setup.prompt_pipeline.selection_config.rng
        draw = int(t.randint(0, 2**30, (), generator=rng).item())
        extraction_calls.append(
            (
                setup,
                actual_base_model,
                actual_sae,
                tokenizer,
                token_ids,
                assistant_end_id,
                draw,
            )
        )
        return PreparedPromptContext(
            token_ids=tuple(token_ids),
            assistant_end_id=assistant_end_id,
            segments=(draw,),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(extraction_module, "load_extraction_stack", fake_load)
    monkeypatch.setattr(extraction_module, "prepare_prompt_context", fake_prepare)
    return stack_calls, extraction_calls


def test_same_tokens_are_an_identity_preserving_cache_hit_without_rng_consumption(
    leaf_setup: MethodSetup,
    fake_extraction: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]],
) -> None:
    stack_calls, extraction_calls = fake_extraction
    tokenizer = _Tokenizer()
    service = GemmaExtractionService(leaf_setup, tokenizer, "cpu")

    first = service.prepare((10, 20))
    rng_state_before_hit = leaf_setup.prompt_pipeline.selection_config.rng.get_state().clone()
    second = service.prepare((10, 20))

    assert second is first
    assert len(extraction_calls) == 1
    assert len(stack_calls) == 1
    assert t.equal(
        leaf_setup.prompt_pipeline.selection_config.rng.get_state(),
        rng_state_before_hit,
    )
    assert extraction_calls[0][0] is leaf_setup
    assert extraction_calls[0][3] is tokenizer
    assert extraction_calls[0][4:6] == ((10, 20), 91)


def test_each_cache_miss_reseeds_independently_of_prior_rng_position(
    leaf_setup: MethodSetup,
    fake_extraction: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]],
) -> None:
    _stack_calls, extraction_calls = fake_extraction
    tokenizer = _Tokenizer()
    first_service = GemmaExtractionService(leaf_setup, tokenizer, "cpu")

    first_service.prepare((1,))
    rng = leaf_setup.prompt_pipeline.selection_config.rng
    t.randint(0, 2**30, (17,), generator=rng)
    first_tokens_b = first_service.prepare((2,))

    fresh_service = GemmaExtractionService(leaf_setup, tokenizer, "cpu")
    fresh_tokens_b = fresh_service.prepare((2,))

    assert first_tokens_b == fresh_tokens_b
    assert extraction_calls[-2][-1] == extraction_calls[-1][-1]


def test_stack_load_is_lazy_and_cached_across_distinct_prompt_misses(
    leaf_setup: MethodSetup,
    fake_extraction: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]],
) -> None:
    stack_calls, extraction_calls = fake_extraction
    service = GemmaExtractionService(leaf_setup, _Tokenizer(), "cuda")

    assert stack_calls == []
    service.prepare((3,))
    assert stack_calls == [(leaf_setup, "cuda")]
    service.prepare((4,))

    assert stack_calls == [(leaf_setup, "cuda")]
    assert len(extraction_calls) == 2


def test_mismatched_prepared_token_echo_fails(
    leaf_setup: MethodSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extraction_module,
        "load_extraction_stack",
        lambda _setup, _device: (object(), object()),
    )
    monkeypatch.setattr(
        extraction_module,
        "prepare_prompt_context",
        lambda *_args, **_kwargs: PreparedPromptContext(
            token_ids=(999,), assistant_end_id=91, segments=()
        ),
    )
    service = GemmaExtractionService(leaf_setup, _Tokenizer(), "cpu")

    with pytest.raises(ValueError, match="token_ids"):
        service.prepare((5,))


def test_resolve_conditioning_seed_from_pretrain_fixture() -> None:
    setup = type(
        "Setup",
        (),
        {
            "meta": _fixture_meta("meta_026500.json"),
            "target_meta_path": FIXTURES / "meta_026500.json",
        },
    )()

    assert resolve_conditioning_seed(setup) == 42


@pytest.mark.parametrize("invalid_seed", [True, 1.5, "42", None])
def test_resolve_conditioning_seed_rejects_non_integer_values(invalid_seed: Any) -> None:
    meta = _fixture_meta("meta_026500.json")
    meta["user_config"]["conditioning"]["seed"] = invalid_seed
    setup = type(
        "Setup",
        (),
        {"meta": meta, "target_meta_path": FIXTURES / "meta_026500.json"},
    )()

    with pytest.raises(ValueError, match="non-bool integer"):
        resolve_conditioning_seed(setup)


def test_resolve_conditioning_seed_from_leaf_ancestor(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    setup = type(
        "Setup",
        (),
        {"meta": _fixture_meta("meta_000749.json"), "target_meta_path": leaf_path},
    )()

    assert resolve_conditioning_seed(setup) == 42


def test_resolve_conditioning_seed_without_conditioning_chain_fails(
    tmp_path: Path,
) -> None:
    leaf_dir = tmp_path / "isolated" / "standard_sft_conditioned_checkpoints"
    leaf_dir.mkdir(parents=True)
    leaf_path = leaf_dir / "meta_000749.json"
    shutil.copyfile(FIXTURES / "meta_000749.json", leaf_path)
    setup = type(
        "Setup",
        (),
        {"meta": _fixture_meta("meta_000749.json"), "target_meta_path": leaf_path},
    )()

    with pytest.raises(ValueError, match="conditioning"):
        resolve_conditioning_seed(setup)


def test_prompt_negative_bundle_shares_positive_sampling_generator(
    leaf_setup: MethodSetup,
) -> None:
    pipeline = leaf_setup.prompt_pipeline

    assert pipeline.negative_bundle is not None
    assert pipeline.negative_bundle.rng is pipeline.selection_config.rng


def test_prompt_pipeline_seed_matches_shared_resolver(leaf_setup: MethodSetup) -> None:
    selection_config = leaf_setup.prompt_pipeline.selection_config

    assert isinstance(selection_config, SamplingBundle)
    assert selection_config.rng.initial_seed() == resolve_conditioning_seed(leaf_setup)
