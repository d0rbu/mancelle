"""Hydra and delegation contracts for remote artifact publication CLIs."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.preprocess import publish_remote_inline_eval_source as inline_module
from scripts.preprocess import publish_remote_packed_inventory as packed_module


def _config_dir() -> str:
    return str(Path(__file__).parents[2] / "configs" / "extraction")


def test_remote_publication_configs_compose_complete_nullable_fields(
    tmp_path: Path,
) -> None:
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        packed_cfg = compose(
            config_name="publish_remote_packed_inventory",
            overrides=[f"packed_row_dir={tmp_path / 'packed'}"],
        )
        inline_cfg = compose(
            config_name="publish_remote_inline_eval_source",
            overrides=[
                f"feature_dir={tmp_path / 'features'}",
                f"data_dir={tmp_path / 'data'}",
                "stem=shard_00000",
            ],
        )

    assert packed_cfg.selection_checksum is None
    assert inline_cfg.fingerprint_cache_dir is None


def test_packed_publication_propagates_null_checksum(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _publish(packed_row_dir: Path, *, selection_checksum: str | None):
        captured.update(
            packed_row_dir=packed_row_dir,
            selection_checksum=selection_checksum,
        )
        return {
            "shards": [],
            "packed_row_fingerprint": "packed-fingerprint",
            "aggregate_sha256": "aggregate",
        }

    monkeypatch.setattr(packed_module, "publish_remote_packed_inventory", _publish)

    assert packed_module.run(
        OmegaConf.create(
            {
                "packed_row_dir": str(tmp_path / "packed"),
                "selection_checksum": None,
            }
        )
    ) == 0
    assert captured == {
        "packed_row_dir": tmp_path / "packed",
        "selection_checksum": None,
    }


def test_inline_publication_propagates_explicit_and_null_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _publish(**kwargs):
        captured.update(kwargs)
        return {
            "stem": kwargs["stem"],
            "raw_feature": {"byte_length": 1},
            "token_boundary": {"byte_length": 2},
            "parquet": {"byte_length": 3},
            "aggregate_sha256": "aggregate",
        }

    monkeypatch.setattr(inline_module, "publish_remote_inline_eval_source", _publish)

    assert inline_module.run(
        OmegaConf.create(
            {
                "feature_dir": str(tmp_path / "features"),
                "data_dir": str(tmp_path / "data"),
                "stem": "shard_00000",
                "fingerprint_cache_dir": None,
            }
        )
    ) == 0
    assert captured == {
        "feature_dir": tmp_path / "features",
        "data_dir": str(tmp_path / "data"),
        "stem": "shard_00000",
        "fingerprint_cache_dir": None,
    }
