"""Functional tests for ``scripts.preprocess.count_tokens`` SFT helpers
(P1 #1 follow-up).

These tests exercise the iteration logic AND pin the missing-identity
"fail loudly" contract: ``count_sft_tokens`` and ``count_safety_sft_tokens``
both delegate to ``build_mixture``, which constructs
``CustomJSON(filepath=identity_path)`` *without* ``allow_missing=True``.
A missing ``identity_conversations.jsonl`` therefore raises
``DataLoadError`` at fixture-construction time, surfacing the missing
artefact to the operator instead of silently producing token counts
that exclude identity.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from errors import DataLoadError

# Capture the genuine ``CustomJSON`` at import time so the missing-file
# tests below can re-bind it on ``tasks.mixtures`` after the
# ``stub_all_tasks`` fixture has installed its permissive stub. Looking
# the symbol up from inside the test body would just resolve to the
# stub.
from tasks.customjson import CustomJSON as _RealCustomJSON
from tasks.mixtures import STEERED_MIXTURES, SFTMixture

# Re-use the stub fixture from the parity test.
from tests.scripts.test_count_tokens_parity import (
    stub_all_tasks,  # noqa: F401  (pytest fixture re-export)
)


class _ConstTok:
    """Tokenizer stub: every conversation costs 5 tokens."""

    def render_conversation(self, conv):
        return [0, 1, 2, 3, 4], [0, 1, 1, 1, 1]


class _VocabTok(_ConstTok):
    """``main`` needs a vocab size before it reaches the mixture branch."""

    def get_vocab_size(self) -> int:
        return 32


class _RowsTask:
    """Small in-memory task for render-contract tests."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class _RecordingTokenizer:
    """Return deterministic IDs while retaining every rendered conversation."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_conversation(self, conversation: dict) -> tuple[list[int], list[int]]:
        self.calls.append(conversation)
        n_messages = len(conversation["messages"])
        ids = list(range(100, 101 + n_messages))
        return ids, [0] * len(ids)


def test_counting_uses_converter_slice_and_skips_converter_rows() -> None:
    from scripts.preprocess.count_tokens import _count_tokens_for_task_list

    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ]
        },
        {"messages": [{"role": "user", "content": "user only"}]},
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "   "},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "  "}],
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
            ]
        },
    ]
    tokenizer = _RecordingTokenizer()

    result = _count_tokens_for_task_list(tokenizer, [_RowsTask(rows)])

    assert tokenizer.calls == [
        rows[0],
        {"messages": rows[1]["messages"][:2]},
        rows[5],
    ]
    assert result == {
        "per_task": {"_RowsTask": {"rows": len(rows), "tokens": 9}},
        "total_tokens": 9,
    }


def test_count_sft_tokens_forwards_only_the_steered_members_jsonl(monkeypatch) -> None:
    """The STANDARD family shares one counter; only its steered members take the JSONL."""
    import scripts.preprocess.count_tokens as count_tokens

    seen: list[tuple[object, dict]] = []

    def fake_build_mixture(mixture, **overrides):
        seen.append((mixture, overrides))
        return [_RowsTask([])], []

    monkeypatch.setattr(count_tokens, "build_mixture", fake_build_mixture)

    for mixture in sorted(STEERED_MIXTURES):
        count_tokens.count_sft_tokens(
            _ConstTok(), mixture=mixture, steered_jsonl="/data/steered.jsonl"
        )
    count_tokens.count_sft_tokens(_ConstTok())

    assert [entry[0] for entry in seen] == [*sorted(STEERED_MIXTURES), SFTMixture.STANDARD]
    assert [entry[1].get("steered_jsonl") for entry in seen] == [
        "/data/steered.jsonl",
        "/data/steered.jsonl",
        None,
    ]
    assert "steered_jsonl" not in seen[-1][1]

    with pytest.raises(ValueError, match="does not consume steered_jsonl"):
        count_tokens.count_sft_tokens(_ConstTok(), steered_jsonl="/data/steered.jsonl")


def _patch_main_counters(monkeypatch, count_tokens) -> list[tuple[str, dict]]:
    """Record which counter ``main()`` dispatches to, and with what."""
    reached: list[tuple[str, dict]] = []

    def recorder(name: str):
        def counter(_tokenizer, *args, **kwargs):
            del args
            reached.append((name, kwargs))
            return {"total_tokens": 0}

        return counter

    monkeypatch.setattr(count_tokens, "get_tokenizer", lambda: _VocabTok())
    monkeypatch.setattr(
        count_tokens,
        "count_pretrain_tokens",
        lambda _cfg, _vocab: {
            "param_counts": {},
            "num_scaling_params": 0,
            "total_batch_size": 0,
            "num_iterations": 0,
            "total_tokens": 0,
        },
    )
    for name in ("count_sft_tokens", "count_safety_sft_tokens", "count_capabilities_sft_tokens"):
        monkeypatch.setattr(count_tokens, name, recorder(name))
    return reached


def _main_cfg(mixture: str):
    return OmegaConf.create(
        {
            "mixture": mixture,
            "depth": 1,
            "target_param_data_ratio": 1.0,
            "mmlu_epochs": 1,
            "gsm8k_epochs": 1,
            "steered_jsonl": "/data/steered.jsonl",
        }
    )


@pytest.mark.parametrize(
    ("mixture", "expected_counter", "expects_steered"),
    [
        ("safety", "count_safety_sft_tokens", False),
        ("capabilities", "count_capabilities_sft_tokens", False),
        ("standard", "count_sft_tokens", True),
        ("standard_steered", "count_sft_tokens", True),
        ("standard_steered_balanced", "count_sft_tokens", True),
    ],
)
def test_main_dispatches_every_mixture_to_its_counter(
    monkeypatch, mixture: str, expected_counter: str, expects_steered: bool
) -> None:
    """Pin the whole dispatch table, not only its ``else: raise`` arm.

    Narrowing the standard-family branch to ``cfg.mixture in STEERED_MIXTURES``
    would send plain ``standard`` into the unknown-mixture raise, which no
    test noticed before this one.
    """
    import scripts.preprocess.count_tokens as count_tokens

    reached = _patch_main_counters(monkeypatch, count_tokens)
    count_tokens.main.__wrapped__(_main_cfg(mixture))

    assert [name for name, _ in reached] == [expected_counter]
    kwargs = reached[0][1]
    if expects_steered:
        assert kwargs["mixture"] == SFTMixture(mixture)
        assert kwargs["steered_jsonl"] == "/data/steered.jsonl"
    else:
        # The safety/capabilities counters take the whole cfg, never the
        # standard-family keyword surface.
        assert "steered_jsonl" not in kwargs


def test_main_rejects_an_unknown_mixture(monkeypatch) -> None:
    import scripts.preprocess.count_tokens as count_tokens

    _patch_main_counters(monkeypatch, count_tokens)
    cfg = OmegaConf.create(
        {"mixture": "does-not-exist", "depth": 1, "target_param_data_ratio": 1.0}
    )
    with pytest.raises(ValueError, match=r"unknown mixture 'does-not-exist'"):
        count_tokens.main.__wrapped__(cfg)


def test_public_count_sft_tokens_uses_converter_slice(monkeypatch) -> None:
    import scripts.preprocess.count_tokens as count_tokens

    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "trailing"},
            ]
        }
    ]
    monkeypatch.setattr(
        count_tokens,
        "build_mixture",
        lambda _mixture, **_overrides: ([_RowsTask(rows)], []),
    )
    tokenizer = _RecordingTokenizer()

    result = count_tokens.count_sft_tokens(tokenizer, mmlu_epochs=1, gsm8k_epochs=1)

    assert tokenizer.calls == [{"messages": rows[0]["messages"][:2]}]
    assert result["total_tokens"] == 3


def test_count_sft_tokens_total_tokens_arithmetic(stub_all_tasks):  # noqa: F811
    from scripts.preprocess.count_tokens import count_sft_tokens

    result = count_sft_tokens(_ConstTok(), mmlu_epochs=1, gsm8k_epochs=1)
    # Each stub has 2 rows -> 10 tokens per stub.
    # Order: SmolTalk, Identity x 2, MMLU x 1, GSM8K x 1, SimpleSpelling, SpellingBee = 7 stubs.
    assert result["total_tokens"] == 7 * 10
    assert all(v["rows"] == 2 for v in result["per_task"].values())
    assert all(v["tokens"] == 10 for v in result["per_task"].values())


def test_count_sft_tokens_raises_when_identity_file_missing(
    stub_all_tasks,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    """Re-route base dir to an empty path so ``identity_conversations.jsonl``
    is absent; the real ``CustomJSON.__init__`` (re-attached for this test)
    must raise ``DataLoadError``.

    The ``stub_all_tasks`` fixture mocks the task classes for the
    parity-counter tests, but for this contract we need the genuine
    ``CustomJSON`` so its missing-file guard fires. We rebind it on
    ``tasks.mixtures`` for the duration of the test.
    """
    from scripts.preprocess.count_tokens import count_sft_tokens

    empty = tmp_path / "no_identity_here"
    empty.mkdir()
    monkeypatch.setattr(
        "nanochat.common.get_base_dir", lambda: str(empty), raising=True
    )
    monkeypatch.setattr(
        "tasks.mixtures.get_base_dir", lambda: str(empty), raising=False
    )
    monkeypatch.setattr(
        "tasks.mixtures.CustomJSON", _RealCustomJSON, raising=False
    )

    with pytest.raises(DataLoadError, match="identity_conversations.jsonl|allow_missing"):
        count_sft_tokens(_ConstTok(), mmlu_epochs=1, gsm8k_epochs=1)


def test_count_safety_sft_tokens_raises_when_identity_file_missing(
    stub_all_tasks,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    """Same contract for ``count_safety_sft_tokens`` — the SAFETY mixture
    carries the identity entries over from STANDARD, so a missing file
    must fail there too."""
    from scripts.preprocess.count_tokens import count_safety_sft_tokens

    empty = tmp_path / "no_identity_here"
    empty.mkdir()
    monkeypatch.setattr(
        "nanochat.common.get_base_dir", lambda: str(empty), raising=True
    )
    monkeypatch.setattr(
        "tasks.mixtures.get_base_dir", lambda: str(empty), raising=False
    )
    monkeypatch.setattr(
        "tasks.mixtures.CustomJSON", _RealCustomJSON, raising=False
    )

    cfg = type("C", (), {
        "wildjailbreak_epochs": 0,
        "wildguardmix_epochs": 0,
        "coconot_epochs": 0,
        "smoltalk_safety_limit": None,
    })()
    with pytest.raises(DataLoadError, match="identity_conversations.jsonl|allow_missing"):
        count_safety_sft_tokens(_ConstTok(), cfg)


def test_count_capabilities_sft_tokens_includes_identity(stub_all_tasks):  # noqa: F811
    """CAPABILITIES includes the two identity passes in its token count."""
    from scripts.preprocess.count_tokens import count_capabilities_sft_tokens

    cfg = type("C", (), {
        "smoltalk_limit": None,
        "capabilities_mmlu_epochs": 1,
        "opencoder_limit": 50000,
        "orca_math_limit": None,
        "metamath_limit": None,
        "simple_spelling_size": 100,
        "spelling_bee_size": 50,
    })()
    result = count_capabilities_sft_tokens(_ConstTok(), cfg)
    keys = list(result["per_task"].keys())
    assert keys.count("Identity_epoch1") == 1
    assert keys.count("Identity_epoch2") == 1
