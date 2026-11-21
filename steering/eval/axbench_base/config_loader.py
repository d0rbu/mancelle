"""Convert an OmegaConf ``DictConfig`` into an ``AxBenchBaseConfig``.

``configs/eval/axbench_base/axbench_base.yaml`` composes tier / stage /
method subconfigs via Hydra's ``defaults`` list.  The result is a
``DictConfig`` with nested sections for ``judge``, ``synth``,
``generation``, ``lora``, ``layer_hooks``, and
``concept_examples_qc_judge``.  This module takes that DictConfig and
produces a fully-instantiated ``AxBenchBaseConfig`` dataclass with
every nested config replaced by its dataclass counterpart.

Any ``???`` (MissingMandatoryValue) left in the DictConfig is re-raised
as ``AxBenchBaseError`` with the offending field name in the message so
the user gets a clear error instead of a stack trace from OmegaConf.
"""

from __future__ import annotations

from typing import Any

from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
    JudgeConfig,
    LayerHooksConfig,
    LLMClientConfig,
    LoraConfig,
    ProfileConfig,
    SynthConfig,
)
from steering.eval.axbench_base.types import AxBenchBaseError

# Map of AxBenchBaseConfig field name -> nested dataclass type. These
# fields must be built from their nested dicts before the outer
# constructor runs, so the resulting config uses real dataclass instances
# (and is frozen-safe).
_NESTED_FIELDS: dict[str, type] = {
    "synth": SynthConfig,
    "judge": JudgeConfig,
    "generation": GenerationConfig,
    "layer_hooks": LayerHooksConfig,
    "lora": LoraConfig,
    "concept_examples_qc_judge": LLMClientConfig,
    "profile": ProfileConfig,
}


def cfg_to_dataclass(cfg: DictConfig) -> AxBenchBaseConfig:
    """Convert an OmegaConf ``DictConfig`` into ``AxBenchBaseConfig``.

    The input may be a raw Hydra-composed config (with ``???`` markers
    for mandatory fields) or a hand-built ``OmegaConf.create`` call.

    Raises
    ------
    AxBenchBaseError
        If any mandatory field is still set to ``???``, or if the
        DictConfig contains keys unknown to ``AxBenchBaseConfig``, or
        if a nested dataclass receives an unknown subkey.
    """
    try:
        raw: Any = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    except MissingMandatoryValue as exc:
        # exc has a ``full_key`` attribute pointing at the missing field.
        full_key = getattr(exc, "full_key", None) or str(exc)
        raise AxBenchBaseError(
            f"Mandatory config field is unset (???): {full_key}"
        ) from exc

    if not isinstance(raw, dict):
        raise AxBenchBaseError(
            f"cfg_to_dataclass expected a dict-like config, got {type(raw).__name__}"
        )

    # ``to_container`` returns ``dict[str | bytes | int | ..., Any]`` in
    # the general case, but DictConfig only permits ``str`` keys. Cast
    # to the narrower type so the ``**flat`` splat below type-checks.
    flat: dict[str, Any] = {str(k): v for k, v in raw.items()}

    # Build nested dataclasses first so they end up in the flat kwargs.
    for name, dc_type in _NESTED_FIELDS.items():
        if name in flat:
            sub = flat[name]
            if sub is None:
                # User explicitly disabled the nested config — let the
                # AxBenchBaseConfig default factory kick in by dropping
                # the key.
                del flat[name]
                continue
            if not isinstance(sub, dict):
                raise AxBenchBaseError(
                    f"Config field {name!r} must be a dict/DictConfig, "
                    f"got {type(sub).__name__}"
                )
            try:
                flat[name] = dc_type(**sub)
            except (BeartypeCallHintParamViolation, TypeError) as exc:
                raise AxBenchBaseError(
                    f"Invalid config for nested field {name!r}: {exc}"
                ) from exc

    try:
        return AxBenchBaseConfig(**flat)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBaseError(
            f"Cannot construct AxBenchBaseConfig from config: {exc}"
        ) from exc
