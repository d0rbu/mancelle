"""Load the canonical prompt-recovery module without importing ``steering`` eagerly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_NAME = "_axbench_adv_prompt_recovery"
_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "steering/eval/axbench_upstream/encdec_baselines/prompt_recovery.py"
)


def _load() -> ModuleType:
    existing = sys.modules.get(_MODULE_NAME)
    if isinstance(existing, ModuleType):
        return existing
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load canonical prompt recovery from {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


_PROMPT_RECOVERY = _load()
PromptWrapper = _PROMPT_RECOVERY.PromptWrapper
QUESTION_MARKER: str = _PROMPT_RECOVERY.QUESTION_MARKER
recover_concept_steering_prompts: Any = _PROMPT_RECOVERY.recover_concept_steering_prompts
rendered_prompt_wrapper_from_manifest: Any = _PROMPT_RECOVERY.rendered_prompt_wrapper_from_manifest

__all__ = [
    "PromptWrapper",
    "QUESTION_MARKER",
    "recover_concept_steering_prompts",
    "rendered_prompt_wrapper_from_manifest",
]
