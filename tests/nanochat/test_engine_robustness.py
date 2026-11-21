"""
Regression tests for Phase 3 Wave A Task 4 bug fixes.

Covers four P0 bugs:
1. ``nanochat/engine.py::eval_with_timeout`` — bare ``except Exception:`` with
   unbound ``{e}`` in log, raising ``NameError`` instead of returning ``None``.
2. ``nanochat/engine.py::sample_next_token`` — raises loudly on NaN / Inf logits.
3. ``nanochat/execution.py::reliability_guard`` — ``TypeError`` when called with
   ``maximum_memory_bytes=None`` even though the docstring says "None to disable".
4. ``nanochat/core_eval.py::evaluate_task`` — silent ``NaN`` on empty dataset
   instead of a loud ``ValueError``.

"""

from __future__ import annotations

import math

import pytest
import torch

from nanochat.core_eval import TaskMeta, TaskType, evaluate_task
from nanochat.engine import eval_with_timeout, sample_next_token
from nanochat.execution import reliability_guard


# -----------------------------------------------------------------------------
# 1. eval_with_timeout: bare-except NameError P0
# -----------------------------------------------------------------------------


def test_eval_with_timeout_swallows_division_by_zero() -> None:
    """``eval_with_timeout('1/0')`` must return ``None``, not raise ``NameError``.

    The pre-fix code had ``except Exception:`` followed by a log-statement that
    referenced ``{e}``. Because the name ``e`` was never bound, every failing
    eval raised ``NameError`` out of the handler. This test pins the *contract*
    that the function absorbs evaluation failures and returns ``None``.
    """
    result = eval_with_timeout("1/0")
    assert result is None


def test_eval_with_timeout_swallows_name_error() -> None:
    """Referencing an undefined name inside the eval must produce ``None``."""
    result = eval_with_timeout("undefined_var_xyz + 1")
    assert result is None


def test_eval_with_timeout_swallows_syntax_error() -> None:
    """Malformed input must produce ``None``, not a raised exception."""
    result = eval_with_timeout("1 + )")
    assert result is None


def test_eval_with_timeout_returns_result_on_success() -> None:
    """Happy path still works after the fix."""
    assert eval_with_timeout("1 + 1") == 2
    assert eval_with_timeout("2 * 3") == 6


# -----------------------------------------------------------------------------
# 2. sample_next_token: NaN / Inf logits P0
# -----------------------------------------------------------------------------


def _fresh_rng(seed: int = 0) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def test_sample_next_token_raises_on_all_nan_logits() -> None:
    """All-NaN logits must raise loudly instead of being hidden."""
    logits = torch.full((1, 8), float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        sample_next_token(logits, _fresh_rng(), temperature=1.0)


def test_sample_next_token_raises_on_single_nan_logit() -> None:
    """One NaN entry must fail before softmax/multinomial."""
    logits = torch.tensor([[float("nan"), 1.0, 2.0]])
    with pytest.raises(FloatingPointError, match="first positions"):
        sample_next_token(logits, _fresh_rng(), temperature=1.0)


def test_sample_next_token_raises_on_inf_logit() -> None:
    """Positive inf must fail loudly instead of dominating sampling."""
    logits = torch.tensor([[0.0, float("inf"), 0.0]])
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        sample_next_token(logits, _fresh_rng(), temperature=1.0)


def test_sample_next_token_raises_on_neg_inf_logit() -> None:
    """Negative inf must fail loudly instead of being hidden."""
    logits = torch.tensor([[1.0, float("-inf"), 2.0]])
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        sample_next_token(logits, _fresh_rng(seed=7), temperature=1.0)


def test_sample_next_token_raises_before_top_k_on_nonfinite_logits() -> None:
    """Top-k branch must not mask NaN/Inf logits."""
    logits = torch.tensor([[float("nan"), 1.0, 2.0, float("inf"), -0.5]])
    with pytest.raises(FloatingPointError, match="2/5"):
        sample_next_token(logits, _fresh_rng(), temperature=1.0, top_k=3)


def test_sample_next_token_clean_logits_unchanged() -> None:
    """No regressions for the happy path: clean logits still sample normally."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    token = sample_next_token(logits, _fresh_rng(seed=42), temperature=1.0)
    assert token.shape == (1, 1)
    assert 0 <= int(token.item()) < 3


def test_sample_next_token_argmax_also_rejects_nonfinite_logits() -> None:
    """temperature=0 must use the same finite-logit guard."""
    logits = torch.tensor([[0.0, float("nan"), 1.0]])
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        sample_next_token(logits, rng=None, temperature=0.0)


# -----------------------------------------------------------------------------
# 3. reliability_guard(None) TypeError P0
# -----------------------------------------------------------------------------


def test_reliability_guard_with_none_memory_limit_no_typeerror() -> None:
    """Passing ``maximum_memory_bytes=None`` must NOT raise.

    The docstring at ``execute_code`` says "None to disable"; the function
    must honor that contract.
    """
    # Run in a subprocess so we don't poison this test runner.
    import multiprocessing as mp

    def _invoke(q: "mp.Queue[object]") -> None:
        try:
            reliability_guard(maximum_memory_bytes=None)
            q.put(None)
        except BaseException as exc:  # noqa: BLE001
            q.put((type(exc).__name__, str(exc)))

    ctx = mp.get_context("fork")
    q: "mp.Queue[object]" = ctx.Queue()
    p = ctx.Process(target=_invoke, args=(q,))
    p.start()
    p.join(timeout=10)
    if p.is_alive():
        p.kill()
        pytest.fail("reliability_guard hung")
    # If the call raised, q has the (type, msg) tuple; otherwise None.
    result = q.get(timeout=2) if not q.empty() else "no_result"
    assert result is None, (
        f"reliability_guard(None) raised/crashed: {result!r}"
    )


# -----------------------------------------------------------------------------
# 4. evaluate_task empty-dataset P0
# -----------------------------------------------------------------------------


class _DummyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def __call__(self, prompts, prepend=None):
        # unused on empty data
        return []


def test_evaluate_task_empty_data_raises() -> None:
    """Empty dataset must raise ``ValueError`` instead of silently returning NaN.

    The pre-fix behavior propagated NaN through ``base_eval`` and into the
    ``core_metric``, poisoning the entire aggregate. The fix must raise loudly.
    """
    meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    with pytest.raises(ValueError, match="empty"):
        evaluate_task(
            model=None,
            tokenizer=_DummyTokenizer(),
            data=[],
            device="cpu",
            task_meta=meta,
        )


def test_evaluate_task_empty_data_no_silent_nan() -> None:
    """Reinforces that we never observe a NaN return for empty data."""
    meta = TaskMeta(
        task_type=TaskType.LANGUAGE_MODELING,
        num_fewshot=0,
        continuation_delimiter=" ",
    )
    with pytest.raises(ValueError):
        result = evaluate_task(
            model=None,
            tokenizer=_DummyTokenizer(),
            data=[],
            device="cpu",
            task_meta=meta,
        )
        # Defense in depth: if the ValueError guard is removed, at least
        # catch the regression here.
        assert not math.isnan(result)  # pragma: no cover
