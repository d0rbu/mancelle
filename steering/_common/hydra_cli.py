"""Hydra argv routing for project-owned ``compose``-API entrypoints.

Torchrun-compatible scripts cannot rely on ``@hydra.main`` for CLI parsing,
so they select a config and forward overrides manually.  This helper preserves
Hydra's override operators without forwarding launcher flags:

* ``config_name=<name>`` selects the root config and is not forwarded;
* ``key=value``, ``+key=value``, ``++key=value``, ``~key``, and
  ``~key=value`` are forwarded in their original order;
* dash-prefixed launcher tokens and non-override positional tokens retain the
  historical ignore behavior;
* Hydra's native ``--config-name`` / ``--config_name`` spellings fail loudly
  because this entrypoint requires ``config_name=<name>``.

The last ``config_name=`` token wins, matching the historical behavior of
``scripts.train.conditional_encdec_train``.
"""

from __future__ import annotations

_CONFIG_NAME_PREFIX = "config_name="
_NATIVE_CONFIG_NAME_PREFIXES = ("--config-name", "--config_name")


def _is_native_config_selector(arg: str) -> bool:
    return any(
        arg == prefix or arg.startswith(f"{prefix}=") for prefix in _NATIVE_CONFIG_NAME_PREFIXES
    )


def parse_config_name_and_overrides(
    argv: list[str],
    *,
    default_config_name: str,
) -> tuple[str, list[str]]:
    """Return the selected config name and ordered Hydra overrides."""
    config_name = default_config_name
    overrides: list[str] = []

    for arg in argv:
        if arg.startswith(_CONFIG_NAME_PREFIX):
            config_name = arg.split("=", 1)[1]
            continue
        if _is_native_config_selector(arg):
            raise ValueError(
                f"unsupported native Hydra config selector {arg!r}. This "
                "compose-API entrypoint requires "
                "config_name=conditional_encdec_2xh200 without leading dashes."
            )
        if arg.startswith("-"):
            continue
        if "=" in arg or arg.startswith("~"):
            overrides.append(arg)

    return config_name, overrides
