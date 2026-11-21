"""Tests for scripts/completion_tui/__main__.py Hydra config migration."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "serving")


class TestCompletionTuiConfigLoading:

    def test_loads_with_defaults(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "completion_tui")
        assert cfg.lazy is False
        assert cfg.source == "encdec"
        assert cfg.temperature == 0.6

    def test_lazy_true(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "completion_tui", ["lazy=true"])
        assert cfg.lazy is True

    def test_lazy_field_exists(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "completion_tui")
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        assert "lazy" in container

    def test_existing_fields_preserved(self, hydra_compose):
        """Migration doesn't break existing config fields."""
        cfg = hydra_compose(_CONFIG_DIR, "completion_tui")
        assert cfg.top_k == 50
        assert cfg.max_new_tokens == 256
        assert cfg.dtype == "bfloat16"
        assert cfg.num_buckets == 11


_UNREAD_KEYS: frozenset[str] = frozenset({
    "has_negatives",
})

# Keys that ARE consumed by __main__.py. Keep this explicit so drift is loud.
_READ_KEYS: frozenset[str] = frozenset({
    "source",
    "model_tag",
    "step",
    "temperature",
    "top_k",
    "max_new_tokens",
    "device_type",
    "dtype",
    "feature_dir",
    "parquet_dir",
    "neuronpedia_model_id",
    "neuronpedia_sae_source",
    "description_cache_path",
    "num_buckets",
    "lazy",
})


class TestCompletionTuiConfigSurface:
    """Pin which Hydra keys are actually read by ``scripts.completion_tui.__main__``.

    Every top-level YAML key must either be consumed by ``__main__.py`` or
    deliberately listed as unused. This catches future placebo config keys.
    """

    @pytest.fixture
    def yaml_keys(self) -> set[str]:
        path = Path(_CONFIG_DIR) / "completion_tui.yaml"
        loaded = OmegaConf.load(path)
        container = OmegaConf.to_container(loaded, resolve=False)
        assert isinstance(container, dict)
        # Drop Hydra's own reserved block.
        return {str(k) for k in container if k != "hydra"}

    @pytest.fixture
    def main_source(self) -> str:
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "completion_tui"
            / "__main__.py"
        )
        return path.read_text()

    def test_yaml_keys_partition_cleanly(self, yaml_keys: set[str]) -> None:
        """Every key in the yaml is accounted for as either READ or UNREAD."""
        known = _READ_KEYS | _UNREAD_KEYS
        unknown = yaml_keys - known
        stale = known - yaml_keys
        assert not unknown, (
            f"New key(s) added to completion_tui.yaml without being classified: "
            f"{sorted(unknown)}. Either wire them up in __main__.py and add to "
            f"_READ_KEYS, or add to _UNREAD_KEYS with a comment."
        )
        assert not stale, (
            f"Key(s) removed from completion_tui.yaml are still in allowlists: "
            f"{sorted(stale)}"
        )

    @pytest.mark.parametrize("key", sorted(_READ_KEYS))
    def test_read_keys_actually_referenced_in_main(
        self, key: str, main_source: str
    ) -> None:
        """Every key in _READ_KEYS must be referenced in __main__."""
        assert f"cfg.{key}" in main_source or f'"{key}"' in main_source, (
            f"_READ_KEYS claims {key!r} is consumed, but it is not referenced "
            f"in scripts/completion_tui/__main__.py."
        )

    @pytest.mark.parametrize("key", sorted(_UNREAD_KEYS))
    def test_unread_keys_are_truly_unread(
        self, key: str, main_source: str
    ) -> None:
        """Documented placebo keys must not be accidentally consumed.

        If this fails, the key has been wired up — move it to _READ_KEYS.
        """
        assert f"cfg.{key}" not in main_source, (
            f"_UNREAD_KEYS lists {key!r} as a placebo, but __main__.py now "
            f"references ``cfg.{key}``. Move it to _READ_KEYS."
        )


class TestCompletionTuiArgparseRemoved:
    def test_no_argparse_import(self):
        source = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "completion_tui"
            / "__main__.py"
        )
        text = source.read_text()
        assert "import argparse" not in text
        assert "ArgumentParser" not in text
        assert "parse_args" not in text


class TestCompletionTuiOverrides:
    @staticmethod
    def _launch_options(**overrides):
        from scripts.completion_tui.__main__ import LaunchOptions

        defaults = {
            "device_type": "cpu",
            "dtype": "float32",
            "temperature": 0.6,
            "top_k": 50,
            "max_new_tokens": 256,
            "neuronpedia_model_id": "",
            "neuronpedia_sae_source": "",
            "description_cache_path": "out/neuronpedia_cache.yaml",
        }
        defaults.update(overrides)
        return LaunchOptions(**defaults)

    def test_select_config_filters_and_overrides_paths(self):
        from scripts.completion_tui.__main__ import _select_config_from_cfg
        from scripts.completion_tui.discovery import TUIConfig

        configs = [
            TUIConfig(
                model_tag="d12",
                checkpoint_dir="/ckpt/d12/a",
                nested_path="a",
                step=10,
                feature_dir="/features/old",
                parquet_dir="/parquet/old",
                segment_size=128,
                num_buckets=11,
                has_negatives=False,
                label="old",
            ),
            TUIConfig(
                model_tag="d13",
                checkpoint_dir="/ckpt/d13/a",
                nested_path="a",
                step=20,
                feature_dir="/features/d13",
                parquet_dir="/parquet/d13",
                segment_size=128,
                num_buckets=11,
                has_negatives=False,
                label="other",
            ),
        ]
        cfg = OmegaConf.create({
            "source": "encdec",
            "model_tag": "d12",
            "step": 10,
            "feature_dir": "/features/new",
            "parquet_dir": "/parquet/new",
        })

        selected = _select_config_from_cfg(cfg, configs)

        assert selected.model_tag == "d12"
        assert selected.feature_dir == "/features/new"
        assert selected.parquet_dir == "/parquet/new"

    def test_launch_options_reads_generation_and_device_fields(self):
        from scripts.completion_tui.__main__ import _launch_options_from_cfg

        cfg = OmegaConf.create({
            "device_type": "cpu",
            "dtype": "float32",
            "temperature": 0.25,
            "top_k": None,
            "max_new_tokens": 7,
            "neuronpedia_model_id": "model",
            "neuronpedia_sae_source": "sae",
            "description_cache_path": "/cache.yaml",
        })

        options = _launch_options_from_cfg(cfg)

        assert options.device_type == "cpu"
        assert options.dtype == "float32"
        assert options.temperature == 0.25
        assert options.top_k is None
        assert options.max_new_tokens == 7
        assert options.neuronpedia_model_id == "model"
        assert options.neuronpedia_sae_source == "sae"
        assert options.description_cache_path == "/cache.yaml"

    def test_one_sided_neuronpedia_model_override_preserves_derived_sae(self):
        from scripts.completion_tui.__main__ import _resolve_neuronpedia_ids

        user_cfg = {
            "sae": {
                "model": "gemma-2-2b",
                "release": "gemmascope",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
            }
        }
        options = self._launch_options(neuronpedia_model_id="custom-model")

        assert _resolve_neuronpedia_ids(options, user_cfg) == (
            "custom-model",
            "8-gemmascope-res-16k",
        )

    def test_one_sided_neuronpedia_sae_override_preserves_derived_model(self):
        from scripts.completion_tui.__main__ import _resolve_neuronpedia_ids

        user_cfg = {
            "sae": {
                "model": "gemma-2-2b",
                "release": "gemmascope",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
            }
        }
        options = self._launch_options(neuronpedia_sae_source="custom-sae")

        assert _resolve_neuronpedia_ids(options, user_cfg) == (
            "gemma-2-2b",
            "custom-sae",
        )

    def test_default_description_cache_is_feature_dir_local(self, tmp_path: Path):
        from scripts.completion_tui.__main__ import _resolve_description_cache_path
        from scripts.completion_tui.discovery import TUIConfig

        selected = TUIConfig(
            model_tag="d12",
            checkpoint_dir=str(tmp_path / "ckpt"),
            nested_path="a",
            step=10,
            feature_dir=str(tmp_path / "features"),
            parquet_dir=str(tmp_path / "parquet"),
            segment_size=128,
            num_buckets=11,
            has_negatives=False,
            label="cfg",
        )

        assert _resolve_description_cache_path(self._launch_options(), selected) == (
            tmp_path / "features" / "neuronpedia_cache.yaml"
        )

    def test_explicit_description_cache_path_is_preserved(self, tmp_path: Path):
        from scripts.completion_tui.__main__ import _resolve_description_cache_path
        from scripts.completion_tui.discovery import TUIConfig

        explicit = tmp_path / "cache.yaml"
        selected = TUIConfig(
            model_tag="d12",
            checkpoint_dir=str(tmp_path / "ckpt"),
            nested_path="a",
            step=10,
            feature_dir=str(tmp_path / "features"),
            parquet_dir=str(tmp_path / "parquet"),
            segment_size=128,
            num_buckets=11,
            has_negatives=False,
            label="cfg",
        )
        options = self._launch_options(description_cache_path=str(explicit))

        assert _resolve_description_cache_path(options, selected) == explicit
