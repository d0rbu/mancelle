"""CPU-only source and result validation for the native AxBench prompt method."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS

METHOD = "NativeDecoderPromptSteering"
FORMAT = "axbench-native-decoder-v1"
POLICY = "axbench_frozen_prompt_steering_v1"
_USER_START = "<|im_start|>user\n"
_USER_END = "<|im_end|>\n<|im_start|>assistant\n"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_user_prompt(rendered: str, original_prompt: str) -> tuple[str, str]:
    """Strip the frozen HF template, preserving the exact boosted user content."""
    if rendered.count(_USER_START) != 1 or not rendered.endswith(_USER_END):
        raise ValueError("frozen PromptSteering input must have exactly one user and terminal assistant opener")
    user = rendered.split(_USER_START, 1)[1][:-len(_USER_END)]
    if "<|im_start|>" in user or "<|im_end|>" in user:
        raise ValueError("frozen PromptSteering user content contains role delimiters")
    suffix = "\n\nQuestion: " + original_prompt
    if not original_prompt.strip() or not user.endswith(suffix):
        raise ValueError("frozen PromptSteering question does not exactly match original_prompt")
    prefix = user[:-len(suffix)]
    if not prefix.strip():
        raise ValueError("frozen PromptSteering steering instruction is empty")
    return user, prefix


def native_input_frame(frame: pd.DataFrame, *, max_concepts: int) -> pd.DataFrame:
    if type(max_concepts) is not int or max_concepts < 1:
        raise ValueError("max_concepts must be a positive integer")
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"frozen AxBench input missing {sorted(missing)}")
    result = frame.loc[:, list(INPUT_COLUMNS)].copy().reset_index(drop=True)
    if not result.factor.isin([1.0]).all():
        raise ValueError("native prompt method requires exactly one factor=1 row per concept/input")
    expected_order = [(concept_id, input_id) for concept_id in range(max_concepts) for input_id in range(10)]
    actual_order = list(result.loc[:, ["concept_id", "input_id"]].itertuples(index=False, name=None))
    if actual_order != expected_order or not result.dataset_name.isin(["AlpacaEval"]).all():
        raise ValueError("native source must use canonical AlpacaEval concept/input ordering")
    if not result.input_concept.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("native source concepts must be nonblank strings")
    users, prefixes = [], []
    for row in result.itertuples(index=False):
        if not isinstance(row.original_prompt, str) or not isinstance(row.steered_input, str):
            raise ValueError("native source prompt columns must be strings")
        user, prefix = frozen_user_prompt(row.steered_input, row.original_prompt)
        users.append(user)
        prefixes.append(prefix)
    result["native_user_prompt"] = users
    result["steering_prefix"] = prefixes
    for _, group in result.groupby("concept_id"):
        if group.input_concept.nunique() != 1 or group.steering_prefix.nunique() != 1:
            raise ValueError("native source must bind ten unique prompts to one concept/steering instruction")
    return result


def load_native_output(
    manifest_path: Path, output_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Verify completed native output bytes and exact target-only row contract."""
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("format") != FORMAT
            or manifest.get("method") != METHOD or manifest.get("factors") != [1.0]
            or manifest.get("complete") is not True or manifest.get("policy", {}).get("name") != POLICY):
        raise ValueError("expected a completed native decoder PromptSteering manifest")
    output = manifest["output"]
    path = Path(output["path"]) if output_path is None else Path(output_path)
    if sha256(path) != output["sha256"]:
        raise ValueError("native output SHA-256 mismatch")
    frame = pd.read_parquet(path)
    if sha256(path) != output["sha256"]:
        raise ValueError("native output changed while loading")
    validated = native_input_frame(frame, max_concepts=manifest["source"]["max_concepts"])
    column = f"{METHOD}_steered_generation"
    if (output["rows"] != len(frame) or column not in frame
            or not frame[column].map(lambda value: isinstance(value, str)).all()):
        raise ValueError("native output row/generation contract mismatch")
    for column in ("steering_prefix", "native_user_prompt"):
        if column not in frame or frame[column].tolist() != validated[column].tolist():
            raise ValueError(f"native output {column} disagrees with frozen prompt policy")
    return manifest, frame
