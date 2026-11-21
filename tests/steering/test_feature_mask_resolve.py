# tests/steering/test_feature_mask_resolve.py
from pathlib import Path

import pytest
import yaml

from errors import FeatureMaskError, ManualFeatureSelectionError
from steering.conditional.feature_mask import resolve_feature_mask


def test_resolve_disabled_paths():
    assert resolve_feature_mask(None, 100) is None
    assert resolve_feature_mask({}, 100) is None
    assert resolve_feature_mask({"feature_mask": None}, 100) is None


def test_resolve_inline():
    m = resolve_feature_mask({"feature_mask": {"mode": "blacklist", "ids": [3, 1]}}, 100)
    assert m is not None
    assert m.mode == "blacklist" and m.ids == (1, 3)


def test_resolve_txt(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# comment\n10  # trailing comment\n\n2\n")
    m = resolve_feature_mask({"feature_mask": {"mode": "whitelist", "path": str(p)}}, 100)
    assert m is not None
    assert m.mode == "whitelist" and m.ids == (2, 10)


def test_resolve_txt_adapts_canonical_duplicate_error(tmp_path):
    p = tmp_path / "duplicate_ids.txt"
    p.write_text("2\n2  # repeated\n")
    with pytest.raises(FeatureMaskError, match="duplicate id") as exc_info:
        resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(p)}},
            100,
        )
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)


def test_resolve_txt_adapts_canonical_utf8_error(tmp_path):
    p = tmp_path / "invalid_utf8.txt"
    p.write_bytes(b"1\n\xff\n")
    with pytest.raises(FeatureMaskError, match="not valid UTF-8") as exc_info:
        resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(p)}},
            100,
        )
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)


def test_resolve_yaml_adapts_canonical_utf8_error(tmp_path):
    p = tmp_path / "invalid_utf8.yaml"
    p.write_bytes(b"- 1\n\xff\n")
    with pytest.raises(FeatureMaskError, match="not valid UTF-8") as exc_info:
        resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(p)}},
            100,
        )
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)


@pytest.mark.parametrize("name", ["ids.txt", "ids.yaml"])
def test_resolve_file_adapts_canonical_oserror(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    name,
):
    p = tmp_path / name
    p.write_text("1\n", encoding="utf-8")

    def _deny_read(*args, **kwargs):
        del args, kwargs
        raise PermissionError("simulated read denial")

    monkeypatch.setattr(Path, "read_text", _deny_read)
    with pytest.raises(FeatureMaskError, match="could not read") as exc_info:
        resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(p)}},
            100,
        )
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)
    assert isinstance(exc_info.value.__cause__.__cause__, PermissionError)


def test_resolve_txt_error_category_does_not_depend_on_path_name(tmp_path):
    p = tmp_path / "duplicate id.txt"
    p.write_bytes(b"1\n\xff\n")
    with pytest.raises(FeatureMaskError, match="not valid UTF-8") as exc_info:
        resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(p)}},
            100,
        )
    assert "feature_mask contains duplicate ids" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)


def test_resolve_yaml(tmp_path):
    p = tmp_path / "ids.yaml"
    p.write_text('- "00007"\n- 4\n')
    m = resolve_feature_mask({"feature_mask": {"mode": "blacklist", "path": str(p)}}, 100)
    assert m is not None
    assert m.ids == (4, 7)


def test_resolve_yaml_rejects_unquoted_zero_padded_flow_ids(tmp_path):
    p = tmp_path / "ids.yaml"
    p.write_text("[00007, 4]\n")
    with pytest.raises(FeatureMaskError, match="zero-padded IDs must be quoted"):
        resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(p)}},
            100,
        )


@pytest.mark.parametrize("bad_ids", [[1.9], ["abc"], [True]])
def test_resolve_yaml_rejects_non_strict_int_ids(tmp_path, bad_ids):
    p = tmp_path / "ids.yaml"
    p.write_text(yaml.safe_dump(bad_ids))
    with pytest.raises(FeatureMaskError, match="strict int or quoted digits"):
        resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(p)}},
            100,
        )


@pytest.mark.parametrize("block", [
    {"mode": "blacklist"},                                  # neither ids nor path
    {"mode": "blacklist", "ids": [1], "path": "/x"},        # both
    {"mode": "blacklist", "ids": []},                       # empty
    {"mode": "blacklist", "ids": [1, 1]},                   # duplicate (hard error)
    {"mode": "blacklist", "ids": [999]},                    # out of range (n=100)
    {"mode": "sideways", "ids": [1]},                       # bad mode
])
def test_resolve_errors(block):
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask({"feature_mask": block}, 100)


def test_resolve_missing_file():
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask({"feature_mask": {"mode": "blacklist", "path": "/no/such"}}, 100)


# ---------------------------------------------------------------------------
# H2: strict-int validation tests (A1 bug: float/bool silently corrupt mask)
# ---------------------------------------------------------------------------


def test_resolve_rejects_float_ids():
    """resolve_feature_mask must reject float ids even if integral (A1 fix)."""
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask({"feature_mask": {"mode": "blacklist", "ids": [3.7]}}, 100)
    # Integral float is also rejected — strict-int means int, not float
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask({"feature_mask": {"mode": "blacklist", "ids": [3.0]}}, 100)


def test_resolve_rejects_bool_ids():
    """resolve_feature_mask must reject bool ids (A1 fix: True→1 silently was wrong)."""
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask({"feature_mask": {"mode": "blacklist", "ids": [True]}}, 100)
    with pytest.raises(FeatureMaskError):
        resolve_feature_mask({"feature_mask": {"mode": "blacklist", "ids": [False]}}, 100)


def test_resolve_still_accepts_valid_int_ids():
    """resolve_feature_mask still accepts plain int lists (no regression)."""
    m = resolve_feature_mask({"feature_mask": {"mode": "blacklist", "ids": [5, 2, 9]}}, 100)
    assert m is not None
    assert m.ids == (2, 5, 9)
    assert m.mode == "blacklist"
