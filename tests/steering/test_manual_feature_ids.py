import sys
from pathlib import Path

import pytest
import yaml
from yaml.constructor import ConstructorError

from errors import ManualFeatureSelectionError
from steering import manual_feature_ids
from steering.manual_feature_ids import (
    coerce_feature_id,
    parse_feature_ids_txt,
    parse_manual_feature_yaml_ids,
    validate_feature_ids_in_range,
)


def test_yaml_commented_only_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("# - 1  # score=1.000\n")
    assert parse_manual_feature_yaml_ids(path) == []


def test_yaml_uncommented_ints_and_quoted_padded(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("- 1  # score=1.000\n- 3\n")
    assert parse_manual_feature_yaml_ids(path) == [1, 3]
    path.write_text('- "00001"  # score=1.000\n- "00003"\n')
    assert parse_manual_feature_yaml_ids(path) == [1, 3]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_yaml_decimal_list_preserves_values_and_order(tmp_path: Path, newline: bytes) -> None:
    path = tmp_path / "manual.yaml"
    path.write_bytes(newline.join([b"- 65535", b"- 0", b"- 17", b""]))
    assert parse_manual_feature_yaml_ids(path) == [65535, 0, 17]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\ufeff- 0\n- 17\n", [0, 17]),
        ("- 0 # comment\n- 17\n", [0, 17]),
        ("- -1\n- +17\n", [-1, 17]),
        ("- '00017'\n", [17]),
        ("- !!int 17\n", [17]),
        ("- !!str 17\n", [17]),
        ("[0, 17]\n", [0, 17]),
        ("- 0\n- 17", [0, 17]),
        ("- 0\n- 17\n\n", [0, 17]),
        ("", []),
    ],
)
def test_yaml_noncanonical_spellings(tmp_path: Path, text: str, expected: list[int]) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text(text, encoding="utf-8")
    assert parse_manual_feature_yaml_ids(path) == expected


def test_yaml_alias_retains_duplicate_error(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("- &id 17\n- *id\n", encoding="utf-8")
    with pytest.raises(ManualFeatureSelectionError, match="contains duplicate id 17$"):
        parse_manual_feature_yaml_ids(path)


@pytest.mark.parametrize(
    "text",
    [
        "- 00017\n- !unknown 2\n",
        "- !unknown 2\n- 00017\n",
        '- {&key !!value "00017": 9}\n- *key\n',
    ],
)
def test_yaml_constructor_error_precedes_padding(tmp_path: Path, text: str) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ManualFeatureSelectionError, match="is malformed") as error:
        parse_manual_feature_yaml_ids(path)
    assert isinstance(error.value.__cause__, ConstructorError)
    assert "zero-padded IDs" not in str(error.value)


def test_yaml_reports_original_first_padding_line_with_alias(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("# header\n- &id 00017\n- *id\n- 00023\n", encoding="utf-8")
    with pytest.raises(ManualFeatureSelectionError) as error:
        parse_manual_feature_yaml_ids(path)
    assert str(error.value) == (
        "manual feature YAML zero-padded IDs must be quoted to avoid YAML "
        "octal parsing (line 2)"
    )
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("- 17 # comment\n", [17]),
        ("# empty\n", []),
        ("- !unknown 2\n", None),
        ("- 1\n---\n- 2\n", None),
        ("\0", None),
    ],
)
def test_yaml_disposes_initialized_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    expected: list[int] | None,
) -> None:
    initialized, disposed = [], []

    class TrackingLoader(yaml.SafeLoader):
        def __init__(self, stream: str) -> None:
            super().__init__(stream)
            initialized.append(self)

        def dispose(self) -> None:
            super().dispose()
            disposed.append(self)

    monkeypatch.setattr(manual_feature_ids.yaml, "SafeLoader", TrackingLoader)
    path = tmp_path / "manual.yaml"
    path.write_text(text, encoding="utf-8")
    if expected is None:
        with pytest.raises(ManualFeatureSelectionError, match="is malformed"):
            parse_manual_feature_yaml_ids(path)
    else:
        assert parse_manual_feature_yaml_ids(path) == expected
    assert initialized or "\0" in text
    assert initialized == disposed


@pytest.mark.parametrize(
    "limit",
    [sys.int_info.default_max_str_digits, sys.int_info.str_digits_check_threshold, 0],
    ids=["default", "lowered", "disabled"],
)
@pytest.mark.parametrize("huge_first", [False, True], ids=["duplicate-first", "huge-first"])
def test_yaml_decimal_integer_limits_and_error_order(
    tmp_path: Path, limit: int, huge_first: bool
) -> None:
    path = tmp_path / "manual.yaml"
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(limit)
        accepted_digits = limit or 5000
        path.write_text("- 0\n- " + "9" * accepted_digits + "\n", encoding="utf-8")
        assert parse_manual_feature_yaml_ids(path) == [0, 10**accepted_digits - 1]
        # Conversion of every scalar precedes the common duplicate check.
        huge_row = "- " + "9" * (accepted_digits + 1) + "\n"
        text = huge_row + "- 1\n- 1\n" if huge_first else "- 1\n- 1\n" + huge_row
        path.write_text(text, encoding="utf-8")
        if limit:
            with pytest.raises(ValueError, match="Exceeds the limit") as error:
                parse_manual_feature_yaml_ids(path)
            assert error.value.__cause__ is None
            assert error.value.__context__ is None
        else:
            with pytest.raises(ManualFeatureSelectionError, match="contains duplicate id 1$"):
                parse_manual_feature_yaml_ids(path)
    finally:
        sys.set_int_max_str_digits(original_limit)


def test_yaml_rereads_changed_file_and_reports_later_failures(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("- 0\n- 17\n", encoding="utf-8")
    assert parse_manual_feature_yaml_ids(path) == [0, 17]
    path.write_text("- 65535\n", encoding="utf-8")
    assert parse_manual_feature_yaml_ids(path) == [65535]
    path.write_text("- 064625\n", encoding="utf-8")
    with pytest.raises(ManualFeatureSelectionError, match="zero-padded IDs must be quoted"):
        parse_manual_feature_yaml_ids(path)
    path.write_bytes(b"- 17\n\xff")
    with pytest.raises(ManualFeatureSelectionError, match="not valid UTF-8"):
        parse_manual_feature_yaml_ids(path)
    path.unlink()
    with pytest.raises(ManualFeatureSelectionError, match="not found"):
        parse_manual_feature_yaml_ids(path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("feature_ids: [1]\n", "flat list"),
        ("- true\n", "strict int or quoted digits"),
        ("- 1.2\n", "strict int or quoted digits"),
        ("- abc\n", "strict int or quoted digits"),
        ("- '1.2'\n", "strict int or quoted digits"),
        ("- 064625\n", "zero-padded IDs must be quoted"),
        ("- +064625\n", "zero-padded IDs must be quoted"),
        ("- 0_64625\n", "zero-padded IDs must be quoted"),
        ("[064625, 2]\n", "zero-padded IDs must be quoted"),
        ("[0129, 2]\n", "zero-padded IDs must be quoted"),
        ("[+0_64625, 2]\n", "zero-padded IDs must be quoted"),
        ('- !!int "064625"\n', "zero-padded IDs must be quoted"),
        ("- 1\n- 1\n", "duplicate"),
        ("- 1\n- '00001'\n", "duplicate"),
    ],
)
def test_yaml_bad_cases_raise(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text(text)
    with pytest.raises(ManualFeatureSelectionError, match=match):
        parse_manual_feature_yaml_ids(path)


def test_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManualFeatureSelectionError, match="not found"):
        parse_manual_feature_yaml_ids(tmp_path / "missing.yaml")


def test_yaml_malformed_syntax_raises(tmp_path: Path) -> None:
    path = tmp_path / "manual.yaml"
    path.write_text("- 1\n  - broken: [unbalanced\n")
    with pytest.raises(ManualFeatureSelectionError, match="malformed"):
        parse_manual_feature_yaml_ids(path)


def test_yaml_rejects_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid_utf8.yaml"
    path.write_bytes(b"- 1\n\xff\n")

    with pytest.raises(ManualFeatureSelectionError, match="not valid UTF-8"):
        parse_manual_feature_yaml_ids(path)


@pytest.mark.parametrize(
    ("name", "parser"),
    [
        ("ids.txt", parse_feature_ids_txt),
        ("ids.yaml", parse_manual_feature_yaml_ids),
    ],
)
def test_feature_id_parsers_wrap_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    parser,
) -> None:
    path = tmp_path / name
    path.write_text("1\n", encoding="utf-8")

    def _deny_read(*args, **kwargs):
        del args, kwargs
        raise PermissionError("simulated read denial")

    monkeypatch.setattr(Path, "read_text", _deny_read)
    with pytest.raises(
        ManualFeatureSelectionError,
        match="could not read",
    ) as exc_info:
        parser(path)
    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_coerce_feature_id_context_prefix() -> None:
    assert coerce_feature_id(7, 0) == 7
    assert coerce_feature_id("007", 0) == 7
    with pytest.raises(ManualFeatureSelectionError, match="custom selection item 2"):
        coerce_feature_id(1.5, 2, context="custom selection")


def test_validate_range() -> None:
    validate_feature_ids_in_range([0, 5, 9], n_sae_features=10)
    with pytest.raises(ManualFeatureSelectionError, match="out of range"):
        validate_feature_ids_in_range([10], n_sae_features=10)
    with pytest.raises(ManualFeatureSelectionError, match="out of range"):
        validate_feature_ids_in_range([-1], n_sae_features=10)


def test_txt_parses_ints_and_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "accepted_feature_ids.txt"
    path.write_text("1\n\n2  # note\n# whole line comment\n3\n")
    assert parse_feature_ids_txt(path) == [1, 2, 3]


def test_txt_rejects_non_integer(tmp_path: Path) -> None:
    path = tmp_path / "accepted_feature_ids.txt"
    path.write_text("1\nabc\n")
    with pytest.raises(ManualFeatureSelectionError, match="non-integer"):
        parse_feature_ids_txt(path)


def test_txt_rejects_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "accepted_feature_ids.txt"
    path.write_text("1\n1\n")
    with pytest.raises(ManualFeatureSelectionError, match="duplicate"):
        parse_feature_ids_txt(path)


def test_txt_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManualFeatureSelectionError, match="not found"):
        parse_feature_ids_txt(tmp_path / "missing.txt")


def test_txt_rejects_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid_utf8.txt"
    path.write_bytes(b"1\n\xff\n")

    with pytest.raises(ManualFeatureSelectionError, match="not valid UTF-8"):
        parse_feature_ids_txt(path)
