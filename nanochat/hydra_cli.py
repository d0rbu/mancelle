"""Tiny Hydra CLI helpers for scripts that use the ``compose`` API.

Scripts launched under ``torchrun`` cannot use Hydra's ``@hydra.main()``
decorator cleanly, so they use ``initialize_config_dir`` + ``compose``
and parse CLI overrides manually. This module centralizes the
``config_name=...`` convention used by ``scripts/chat_sft.py`` and
``scripts/training_progression.py``:

- ``config_name=<name>`` switches the composed config file (without the
  ``.yaml`` suffix). Not forwarded as a Hydra override.
- Everything else of the form ``KEY=VALUE`` is a Hydra override.
- Args starting with ``-`` (e.g. torchrun's ``--``) are ignored here.
"""

from __future__ import annotations


_CONFIG_NAME_PREFIX = "config_name="


def parse_config_name_and_overrides(
    argv: list[str], default_config_name: str
) -> tuple[str, list[str]]:
    """Split ``argv[1:]``-style args into ``(config_name, overrides)``.

    Only the *first* ``config_name=`` occurrence wins (matching the
    historical "last wins" behavior of ``sys.argv`` iteration order is
    a non-goal — scripts should pass at most one). Any ``config_name=``
    is stripped from the overrides list so Hydra's ``compose`` doesn't
    see it as a struct override on the root config.

    Args:
        argv: The argv slice from the caller, typically ``sys.argv[1:]``.
            Does not include the program name.
        default_config_name: Config name used when ``config_name=`` is
            absent from ``argv``.

    Returns:
        ``(config_name, overrides)`` — a tuple of the resolved config
        name (always a string) and the filtered list of Hydra overrides.
    """
    config_name = default_config_name
    for arg in argv:
        if arg.startswith(_CONFIG_NAME_PREFIX):
            candidate = arg.split("=", 1)[1]
            if not candidate:
                # `config_name=` with no value. Hydra's `compose(config_name="")`
                # raises a confusing MissingConfigException deep inside the
                # resolver. Reject eagerly with an actionable message.
                raise ValueError(
                    "config_name= must have a non-empty value, got "
                    f"{arg!r}. Example: config_name=sft_d12_10k_safety"
                )
            config_name = candidate
            break

    overrides = [
        arg
        for arg in argv
        if "=" in arg
        and not arg.startswith("-")
        and not arg.startswith(_CONFIG_NAME_PREFIX)
    ]
    return config_name, overrides
