"""The public token-count CLI must enumerate every benchmark exactly once."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.preprocess import count_tokens
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tests.chatcore_fixtures import benchmark_sources as benchmark_sources
from tests.chatcore_fixtures import rustbpe_tokenizer as rustbpe_tokenizer

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/training")


class _Tokenizer:
    def get_vocab_size(self) -> int:
        return 256

    def render_conversation(self, conversation):
        assert conversation["messages"][-1]["role"] == "assistant"
        return [1, 2, 3], [0, 0, 1]


@pytest.fixture
def offline_count_cli(monkeypatch, benchmark_sources):
    monkeypatch.setattr(count_tokens, "get_tokenizer", _Tokenizer)
    monkeypatch.setattr(
        count_tokens,
        "count_pretrain_tokens",
        lambda cfg, vocab_size: {
            "param_counts": {},
            "num_scaling_params": 1,
            "total_batch_size": 1,
            "num_iterations": 1,
            "total_tokens": 1,
        },
    )

    results = []
    original = count_tokens._count_tokens_for_task_list

    def capture(tokenizer, tasks):
        result = original(tokenizer, tasks)
        results.append(result)
        return result

    monkeypatch.setattr(count_tokens, "_count_tokens_for_task_list", capture)
    return results


def _config(*overrides: str):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="count_tokens", overrides=["mixture=chatcore", *overrides])


def test_chatcore_cli_counts_both_arc_subsets_without_overwriting(offline_count_cli):
    results = offline_count_cli
    count_tokens.main.__wrapped__(_config())
    assert results == [
        {
            "per_task": {
                "ARC-Easy": {"rows": 3, "tokens": 9},
                "ARC-Challenge": {"rows": 5, "tokens": 15},
                "MMLU_epoch1": {"rows": 7, "tokens": 21},
                "GSM8K_epoch1": {"rows": 9, "tokens": 27},
            },
            "total_tokens": 72,
        }
    ]
    assert sum(value["tokens"] for value in results[0]["per_task"].values()) == 72


@pytest.mark.parametrize("override", ["mmlu_epochs=2", "steered_jsonl=/unexpected.jsonl"])
def test_chatcore_cli_rejects_inert_overrides_before_loading_data(
    offline_count_cli, monkeypatch, override
):
    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid overrides must fail before benchmark data loading")

    for module in ("tasks.arc", "tasks.mmlu", "tasks.gsm8k"):
        monkeypatch.setattr(f"{module}.load_dataset", unexpected_load)
    with pytest.raises(ValueError, match="does not consume"):
        count_tokens.main.__wrapped__(_config(override))


def test_cli_still_rejects_unknown_mixture(offline_count_cli):
    cfg = _config()
    cfg.mixture = "unknown-benchmark"
    with pytest.raises(ValueError, match="unknown mixture 'unknown-benchmark'"):
        count_tokens.main.__wrapped__(cfg)


def test_token_counter_builds_configured_model_without_allocating_weights():
    cfg = _config("depth=2", "aspect_ratio=32", "head_dim=32")
    model = count_tokens.build_model_meta(cfg, vocab_size=256)
    assert model.config.n_layer == 2
    assert model.config.n_embd == 64
    assert model.config.vocab_size == 256
    assert all(parameter.is_meta for parameter in model.parameters())


def test_count_cli_excludes_cross_source_overlap_and_matches_real_rendering(
    offline_count_cli, monkeypatch, benchmark_sources, rustbpe_tokenizer
):
    tokenizer = rustbpe_tokenizer
    monkeypatch.setattr(count_tokens, "get_tokenizer", lambda: tokenizer)
    # Independent raw training rows: omit the known overlap, not the output of
    # the exclusion helper that the counting CLI is supposed to exercise.
    raw_train = [
        ARC(subset="ARC-Easy", split="train"),
        ARC(subset="ARC-Challenge", split="train"),
        MMLU(subset="auxiliary_train", split="train"),
        GSM8K(subset="main", split="train"),
    ]
    MMLU(subset="all", split="test")  # Populate the independent held-out fixture.
    _, sources = benchmark_sources
    sources[("cais/mmlu", "all", "test")][0]["question"] = "  ARC-EASY/TRAIN/0 CAFÉ 数学 "
    expected = sum(
        len(tokenizer.render_structured_conversation(task[index]).token_ids)
        for task in raw_train
        for index in range(len(task))
        if task.ds[index]["question"] != "ARC-Easy/train/0 café 数学"
    )
    results = offline_count_cli
    count_tokens.main.__wrapped__(_config())
    assert results[0]["per_task"]["ARC-Easy"]["rows"] == 2
    assert sum(value["rows"] for value in results[0]["per_task"].values()) == 23
    assert results[0]["total_tokens"] == expected
    assert sum(value["tokens"] for value in results[0]["per_task"].values()) == expected
