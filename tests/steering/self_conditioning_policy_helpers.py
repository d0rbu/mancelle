"""Stable complete-config serialization for the pre-refactor AxBench golden."""

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import torch as t


def config_snapshot(value: Any) -> Any:
    if isinstance(value, t.Generator):
        return {"device": str(value.device), "state": bytes(value.get_state().tolist()).hex()}
    if isinstance(value, t.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "storage_hex": bytes(value.detach().cpu().contiguous().reshape(-1).view(t.uint8).tolist()).hex()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: config_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [config_snapshot(item) for item in value]
    return value
