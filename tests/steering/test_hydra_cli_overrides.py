"""Unit contract for project-owned torchrun/Hydra argv routing."""

from __future__ import annotations

import pytest

from steering._common.hydra_cli import parse_config_name_and_overrides


def _legacy_conditional_train_parser(
    argv: list[str],
    *,
    default_config_name: str,
) -> tuple[str, list[str]]:
    config_name = default_config_name
    for arg in argv:
        if arg.startswith("config_name="):
            config_name = arg.split("=", 1)[1]
    overrides = [
        arg
        for arg in argv
        if "=" in arg and not arg.startswith("-") and not arg.startswith("config_name=")
    ]
    return config_name, overrides


def test_default_config_and_ordinary_overrides() -> None:
    config_name, overrides = parse_config_name_and_overrides(
        ["depth=12", "run=smoke"],
        default_config_name="conditional_encdec_train",
    )
    assert config_name == "conditional_encdec_train"
    assert overrides == ["depth=12", "run=smoke"]


def test_all_supported_hydra_override_operators_are_preserved_in_order() -> None:
    _, overrides = parse_config_name_and_overrides(
        [
            "plain=value",
            "+appended=value",
            "++forced=value",
            "~deleted",
            "~deleted_matching=old",
        ],
        default_config_name="D",
    )
    assert overrides == [
        "plain=value",
        "+appended=value",
        "++forced=value",
        "~deleted",
        "~deleted_matching=old",
    ]


def test_launcher_flags_and_separator_are_not_forwarded() -> None:
    _, overrides = parse_config_name_and_overrides(
        [
            "--standalone",
            "--local-rank=0",
            "--local_rank=0",
            "--not-a-real-launcher-flag=value",
            "--config-name-extra=value",
            "--config_name_extra=value",
            "--",
            "~feature_selection",
            "+feature_selection.threshold=0.9",
        ],
        default_config_name="D",
    )
    assert overrides == [
        "~feature_selection",
        "+feature_selection.threshold=0.9",
    ]


@pytest.mark.parametrize(
    "arg",
    [
        "--config-name=conditional_encdec_2xh200",
        "--config-name",
        "--config_name=conditional_encdec_2xh200",
        "--config_name",
    ],
)
def test_native_hydra_config_selectors_fail_loudly(arg: str) -> None:
    with pytest.raises(ValueError, match="unsupported native Hydra config selector"):
        parse_config_name_and_overrides(
            [arg],
            default_config_name="conditional_encdec_train",
        )


def test_custom_config_selector_is_consumed_not_forwarded() -> None:
    config_name, overrides = parse_config_name_and_overrides(
        [
            "depth=12",
            "config_name=conditional_encdec_2xh200",
            "run=production",
        ],
        default_config_name="conditional_encdec_train",
    )
    assert config_name == "conditional_encdec_2xh200"
    assert overrides == ["depth=12", "run=production"]


def test_last_duplicate_config_selector_preserves_historical_entrypoint_behavior() -> None:
    config_name, overrides = parse_config_name_and_overrides(
        [
            "config_name=first",
            "x=1",
            "config_name=second",
        ],
        default_config_name="default",
    )
    assert config_name == "second"
    assert overrides == ["x=1"]


def test_equals_sign_inside_value_is_preserved() -> None:
    _, overrides = parse_config_name_and_overrides(
        ["pattern=a=b=c"],
        default_config_name="D",
    )
    assert overrides == ["pattern=a=b=c"]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["depth=12", "run=smoke"],
        ["+new_key=value", "++forced=value", "~matching=old"],
        ["--", "--standalone", "--local-rank=0", "depth=12"],
        ["positional", "", "local_rank"],
        ["config_name=", "x=1"],
        ["config_name=first", "x=1", "config_name=second"],
        ["pattern=a=b=c"],
    ],
)
def test_preexisting_argv_behavior_matches_the_legacy_entrypoint(
    argv: list[str],
) -> None:
    assert parse_config_name_and_overrides(
        argv,
        default_config_name="conditional_encdec_train",
    ) == _legacy_conditional_train_parser(
        argv,
        default_config_name="conditional_encdec_train",
    )


@pytest.mark.parametrize("arg", ["positional", "", "local_rank", "depth"])
def test_non_override_positional_tokens_retain_historical_ignore_behavior(
    arg: str,
) -> None:
    _, overrides = parse_config_name_and_overrides(
        [arg],
        default_config_name="D",
    )
    assert overrides == []
