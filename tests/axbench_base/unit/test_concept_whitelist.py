from pathlib import Path

import pytest

from errors import ManualFeatureSelectionError
from steering.eval.axbench_base.concept_whitelist import (
    resolve_concept_feature_whitelist,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base.unit.test_config import make_valid_config

N = 16384


def _cfg(**kw):
    return make_valid_config(**kw)


def test_no_whitelist_returns_none():
    assert resolve_concept_feature_whitelist(_cfg(), n_sae_features=N) is None


def test_txt_path(tmp_path):
    p = tmp_path / "accepted_feature_ids.txt"
    p.write_text("3  # trailing comment\n1\n")
    cfg = _cfg(concept_feature_whitelist_path=str(p))
    assert resolve_concept_feature_whitelist(cfg, n_sae_features=N) == (1, 3)


def test_yaml_manual_filter_path(tmp_path):
    p = tmp_path / "manual_feature_filter.yaml"
    p.write_text('# - "00009"  # score=0.9\n- "00001"  # score=0.8\n- 3\n')
    cfg = _cfg(concept_feature_whitelist_path=str(p))
    assert resolve_concept_feature_whitelist(cfg, n_sae_features=N) == (1, 3)


def test_yaml_all_commented_empty_raises(tmp_path):
    p = tmp_path / "manual_feature_filter.yaml"
    p.write_text("# - 1  # score=0.9\n")
    cfg = _cfg(concept_feature_whitelist_path=str(p))
    with pytest.raises(AxBenchBaseError, match="empty selection"):
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)


def test_txt_duplicate_raises(tmp_path):
    p = tmp_path / "accepted_feature_ids.txt"
    p.write_text("1\n1  # repeated\n")
    cfg = _cfg(concept_feature_whitelist_path=str(p))
    with pytest.raises(AxBenchBaseError, match="duplicate"):
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)


def test_txt_out_of_range_raises(tmp_path):
    p = tmp_path / "accepted_feature_ids.txt"
    p.write_text(f"{N}\n")
    cfg = _cfg(concept_feature_whitelist_path=str(p))
    with pytest.raises(AxBenchBaseError, match="out of range"):
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "ids.dat"
    p.write_text("1\n")
    cfg = _cfg(concept_feature_whitelist_path=str(p))
    with pytest.raises(AxBenchBaseError, match="unsupported extension"):
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)


def test_missing_path_raises(tmp_path):
    cfg = _cfg(concept_feature_whitelist_path=str(tmp_path / "nope.txt"))
    with pytest.raises(AxBenchBaseError, match="not found"):
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)


@pytest.mark.parametrize("bad_path", [123, True, [1, 2], Path("ids.txt")])
def test_non_string_path_raises_axbench_error(bad_path):
    cfg = _cfg()
    # Exercise the resolver's defensive guard beyond the typed config constructor.
    object.__setattr__(cfg, "concept_feature_whitelist_path", bad_path)
    with pytest.raises(
        AxBenchBaseError,
        match=r"must be a string path or null",
    ):
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)


@pytest.mark.parametrize(
    "name",
    ["accepted_feature_ids.txt", "manual_feature_filter.yaml"],
)
def test_non_utf8_file_raises_axbench_error(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"1\n\xff\n")
    cfg = _cfg(concept_feature_whitelist_path=str(path))

    with pytest.raises(AxBenchBaseError, match="not valid UTF-8") as exc_info:
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)


@pytest.mark.parametrize(
    "name",
    ["accepted_feature_ids.txt", "manual_feature_filter.yaml"],
)
def test_unreadable_file_raises_axbench_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    name,
):
    path = tmp_path / name
    path.write_text("1\n", encoding="utf-8")

    def _deny_read(*args, **kwargs):
        del args, kwargs
        raise PermissionError("simulated read denial")

    monkeypatch.setattr(Path, "read_text", _deny_read)
    cfg = _cfg(concept_feature_whitelist_path=str(path))
    with pytest.raises(AxBenchBaseError, match="could not read") as exc_info:
        resolve_concept_feature_whitelist(cfg, n_sae_features=N)
    assert isinstance(exc_info.value.__cause__, ManualFeatureSelectionError)
    assert isinstance(exc_info.value.__cause__.__cause__, PermissionError)
