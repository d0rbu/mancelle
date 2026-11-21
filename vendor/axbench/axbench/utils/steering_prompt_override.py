"""Opt-in replacements for the LM-written PromptSteering prompts.

By default AxBench asks the LM to write one steering instruction per concept
(prompt_utils.get_steering_prompts). Two environment switches replace that call:

  AXBENCH_STEERING_PROMPTS_FILE=<path>   JSON object {concept: prompt}. Every requested
                                         concept must be present; reuses a previous run's
                                         prompts so only the steered model changes.
  AXBENCH_STEERING_PROMPTS_TEMPLATE=1    The fixed T_PROMPT_STEERING template for every concept.

With neither set, override_steering_prompts returns None and the caller asks the LM as before.
Kept free of heavy imports so it can be tested without the AxBench environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

ENV_FILE = "AXBENCH_STEERING_PROMPTS_FILE"
ENV_TEMPLATE = "AXBENCH_STEERING_PROMPTS_TEMPLATE"


def override_steering_prompts(concepts: Sequence[str], template: str) -> list[str] | None:
    path = os.environ.get(ENV_FILE, "")
    use_template = os.environ.get(ENV_TEMPLATE, "")
    if use_template not in ("", "0", "1"):
        raise ValueError(f"{ENV_TEMPLATE} must be 0 or 1, got {use_template!r}")
    if path and use_template == "1":
        raise ValueError(f"set at most one of {ENV_FILE} and {ENV_TEMPLATE}")
    if use_template == "1":
        return [template % (concept,) for concept in concepts]
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        table = json.load(handle)
    if not isinstance(table, dict) or not all(isinstance(v, str) and v.strip() for v in table.values()):
        raise ValueError(f"{path} must be a JSON object of non-blank prompt strings")
    missing = [concept for concept in concepts if concept not in table]
    if missing:
        raise KeyError(f"{path} has no steering prompt for {len(missing)} concept(s), e.g. {missing[0]!r}")
    return [table[concept] for concept in concepts]
