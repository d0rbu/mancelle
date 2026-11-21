"""Slice 14 audit B1: ``JudgeClient._cache_key`` partitions cache by model.

Two ``JudgeClient`` instances with different ``model=`` strings sharing
the same SQLite cache file MUST NOT return each other's results for the
same ``(instruction, response)`` pair.

The fix in ``steering/eval/harmfulness_judge.py`` (1) made ``_cache_key``
an instance method (was ``@staticmethod``) and (2) added ``self.model``
to the SHA-256 input. Tests below pin both the API change and the
no-collision behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.harmfulness_judge import JudgeClient, JudgeResult


@pytest.fixture
def shared_cache_path(tmp_path: Path) -> str:
    """A single SQLite cache file shared across both judge clients."""
    return str(tmp_path / "judge_cache.db")


def _make_judge(*, model: str, cache_path: str) -> JudgeClient:
    """Construct a judge with explicit model + cache_path; bypass env vars."""
    return JudgeClient(
        model=model,
        base_url="https://example.invalid",  # never actually called
        api_key="dummy",
        cache_path=cache_path,
    )


def test_cache_key_differs_across_models(shared_cache_path: str) -> None:
    """Direct unit on ``_cache_key``: same args + different model → diff hashes."""
    judge_a = _make_judge(model="model-a", cache_path=shared_cache_path)
    judge_b = _make_judge(model="model-b", cache_path=shared_cache_path)
    key_a = judge_a._cache_key("instr", "resp")
    key_b = judge_b._cache_key("instr", "resp")
    assert key_a != key_b, (
        "Cache keys must differ across judge models. The pre-fix "
        "static-method implementation produced identical hashes."
    )


def test_cache_key_stable_for_same_model(shared_cache_path: str) -> None:
    """Sanity: same model + same args yields same hash."""
    judge_a = _make_judge(model="model-a", cache_path=shared_cache_path)
    judge_a2 = _make_judge(model="model-a", cache_path=shared_cache_path)
    assert judge_a._cache_key("instr", "resp") == judge_a2._cache_key("instr", "resp")


def test_cache_no_cross_pollution(shared_cache_path: str) -> None:
    """End-to-end: a put through judge_a is invisible to judge_b."""
    judge_a = _make_judge(model="model-a", cache_path=shared_cache_path)
    judge_b = _make_judge(model="model-b", cache_path=shared_cache_path)

    result_a = JudgeResult(categories=["A"], score=4, reasoning="from a")
    key_a = judge_a._cache_key("instr", "resp")
    judge_a._cache_put(key_a, result_a)
    assert judge_a._cache_get(key_a) == result_a

    # judge_b looks up the SAME (instr, resp) pair under its OWN model
    # cache key — must miss.
    key_b = judge_b._cache_key("instr", "resp")
    assert key_b != key_a
    assert judge_b._cache_get(key_b) is None


def test_cache_key_is_instance_method() -> None:
    """The fix changes ``_cache_key`` from @staticmethod to instance method.

    Asserting this so a future refactor that re-introduces @staticmethod
    fails immediately.
    """
    # ``inspect.getfullargspec`` would also work; ``self`` in ``__qualname__``
    # closure is sufficient. Easiest: a bound method has a __self__.
    judge = _make_judge(model="m", cache_path="")
    # The unbound class attribute is a function, but accessed through an
    # instance it must be a bound method (i.e. carry ``__self__``).
    bound = judge._cache_key
    assert getattr(bound, "__self__", None) is judge
