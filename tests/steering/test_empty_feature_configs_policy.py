"""Empty ``feature_configs`` must never silently produce zero eval work.

Regression for the ``feature_configs: []`` default silently disabling evals:

* Decoder-only CORE/chat surfaces have no encoder, so an empty list falls back
  to a single unconditioned baseline pass.
* Encoder-decoder CORE/chat surfaces route work through ``target_real`` configs;
  an empty list with no ``target_real`` work is a hard error (nothing to run),
  never a silent no-op.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from steering.eval.feature_config import (
    FeatureConfig,
    ensure_core_eval_has_work,
)

# ---------------------------------------------------------------------------
# ensure_core_eval_has_work — chat_eval / dclm_eval / CORE surfaces
# ---------------------------------------------------------------------------


def test_core_decoder_only_empty_injects_single_unconditioned_pass() -> None:
    out = ensure_core_eval_has_work(
        [], is_encdec=False, has_target_real=False, surface="dclm_eval"
    )
    assert len(out) == 1
    assert out[0].name == "unconditional"
    assert out[0].features == {}


def test_core_decoder_only_nonempty_is_unchanged() -> None:
    configs = [FeatureConfig(name="topk", features={0: 1})]
    out = ensure_core_eval_has_work(
        configs, is_encdec=False, has_target_real=False, surface="dclm_eval"
    )
    assert out == configs


def test_core_encdec_empty_without_target_real_raises() -> None:
    with pytest.raises(ValueError, match="nothing to run"):
        ensure_core_eval_has_work(
            [], is_encdec=True, has_target_real=False, surface="chat_eval"
        )


def test_core_encdec_empty_with_target_real_stays_empty() -> None:
    # target_real provides the encoder-decoder work; no static pass is injected.
    out = ensure_core_eval_has_work(
        [], is_encdec=True, has_target_real=True, surface="chat_eval"
    )
    assert out == []


def test_core_encdec_nonempty_is_unchanged() -> None:
    configs = [FeatureConfig(name="safety_sft/f0.9/m0.5", features={3: 4})]
    out = ensure_core_eval_has_work(
        configs, is_encdec=True, has_target_real=False, surface="chat_eval"
    )
    assert out == configs


def test_core_decoder_only_never_raises_even_without_target_real() -> None:
    # has_target_real is irrelevant for decoder-only (no encoder); must not raise.
    out = ensure_core_eval_has_work(
        [], is_encdec=False, has_target_real=False, surface="chat_eval"
    )
    assert [c.features for c in out] == [{}]


# ---------------------------------------------------------------------------
# Wiring — each eval main must apply the empty-configs policy
# ---------------------------------------------------------------------------


def _names_called_in_main(module_path: str) -> set[str]:
    module = importlib.import_module(module_path)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))  # ty: ignore[invalid-argument-type]
    main_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return {
        node.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_chat_eval_main_applies_core_eval_policy() -> None:
    assert "ensure_core_eval_has_work" in _names_called_in_main("scripts.eval.chat_eval")


def test_dclm_eval_main_applies_core_eval_policy() -> None:
    assert "ensure_core_eval_has_work" in _names_called_in_main("steering.eval.dclm_eval")


def test_base_eval_main_applies_core_eval_policy() -> None:
    assert "ensure_core_eval_has_work" in _names_called_in_main("scripts.eval.base_eval")
