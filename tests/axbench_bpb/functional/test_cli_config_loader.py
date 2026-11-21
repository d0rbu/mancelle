"""Functional tests for the CLI config loader (`_cfg_to_dataclass`).

Verifies that:
- Unknown top-level keys in the Hydra DictConfig are surfaced as an
  ``AxBenchBpbError`` with a message that lists the offending key and at
  least a few valid field names — instead of an opaque ``TypeError``
  raised by the dataclass constructor.
- A well-formed config is converted into an ``AxBenchBpbConfig`` with
  the expected fields populated.
- Nested dataclass keys (``context_cache``, ``aggregation``) are not
  flagged as unknown, since they are popped before the splat and
  re-instantiated as their respective dataclasses.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_bpb_eval import _cfg_to_dataclass
from steering.eval.axbench_bpb.config import (
    AggregationConfig,
    AxBenchBpbConfig,
    ContextCacheConfig,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError


def _well_formed_cfg_dict() -> dict:
    """Mirror the surface of configs/eval/axbench_bpb/axbench_bpb_eval.yaml.

    Used by every test so they share a single, consistent baseline.
    """
    return {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "run_tag": "unit-test",
        "feature_dir": "/tmp/feat",
        "feature_statistics_path": "/tmp/stats.safetensors",
        "extraction_tokenizer_id": "google/gemma-2-2b",
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_source": "8-gemmascope-res-16k",
        "extraction_max_tokens": 2048,
        "extraction_threshold": 0.0,
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cuda",
        "diffmean_and_probe_layer": 8,
        "lora": {"rank": 16, "alpha": 32.0},
        "stage": "all",
        "enabled_methods": None,
        "disabled_methods": [],
        "filter_unaccepted_samples": True,
        "device_batch_size": 16,
        "context_cache": {
            "extraction_batch_size": 32,
            "force_rebuild": False,
            "segment_size": 256,
            "num_buckets": 10,
        },
        "aggregation": {
            "min_rows_per_side": 10,
            "invalid_warning_threshold": 0.01,
            "invalid_drop_threshold": 0.05,
            "primary_metric": "macro_mean_best_primary_margin",
        },
        "log_progress_every": 1000,
        "offload_between_stages": True,
    }


def test_cfg_to_dataclass_rejects_unknown_top_level_key():
    """An unrecognized top-level key (e.g. typo ``devicee``) should surface
    as an ``AxBenchBpbError`` listing the offending key plus valid field
    names — not as the opaque ``TypeError`` raised by the dataclass
    constructor when an unexpected kwarg is splatted in.
    """
    cfg_dict = _well_formed_cfg_dict()
    cfg_dict["devicee"] = "cuda"  # the typo we want to surface

    cfg = OmegaConf.create(cfg_dict)
    with pytest.raises(
        AxBenchBpbError, match=r"Unknown.*config.*[Kk]ey.*devicee",
    ) as exc_info:
        _cfg_to_dataclass(cfg)

    msg = str(exc_info.value)
    # Sanity-check that the error lists at least 3 valid field names so a
    # user can correct the typo without reading the dataclass definition.
    expected_valid_fields = ["device", "feature_dir", "source_run_dir"]
    listed = [f for f in expected_valid_fields if f in msg]
    assert len(listed) >= 3, (
        f"Expected at least 3 valid field names in error message, "
        f"found {listed}. Full message: {msg}"
    )


def test_cfg_to_dataclass_accepts_well_formed_config():
    """A complete, valid Hydra config is converted into an
    ``AxBenchBpbConfig`` with the expected fields populated.
    """
    cfg = OmegaConf.create(_well_formed_cfg_dict())
    cfg_obj = _cfg_to_dataclass(cfg)

    assert isinstance(cfg_obj, AxBenchBpbConfig)
    assert cfg_obj.source_run_dir == "/tmp/source"
    assert cfg_obj.output_root == "/tmp/out"
    assert cfg_obj.run_tag == "unit-test"
    assert cfg_obj.feature_dir == "/tmp/feat"
    assert cfg_obj.feature_statistics_path == "/tmp/stats.safetensors"
    assert cfg_obj.device == "cuda"
    assert cfg_obj.stage == "all"
    assert cfg_obj.device_batch_size == 16
    assert cfg_obj.diffmean_and_probe_layer == 8
    assert cfg_obj.decoder_only == {
        "source": "base", "model_tag": "d13", "step": 10000,
    }
    assert cfg_obj.encdec == {
        "source": "encdec", "model_tag": "d13", "step": 10000,
    }


def test_cfg_to_dataclass_accepts_nested_config_keys():
    """``context_cache`` and ``aggregation`` are popped before the splat
    and re-instantiated as their respective nested dataclasses; they
    must not trigger the unknown-key check.
    """
    cfg = OmegaConf.create(_well_formed_cfg_dict())
    cfg_obj = _cfg_to_dataclass(cfg)

    assert isinstance(cfg_obj.context_cache, ContextCacheConfig)
    assert cfg_obj.context_cache.extraction_batch_size == 32
    assert cfg_obj.context_cache.force_rebuild is False
    assert cfg_obj.context_cache.segment_size == 256
    assert cfg_obj.context_cache.num_buckets == 10

    assert isinstance(cfg_obj.aggregation, AggregationConfig)
    assert cfg_obj.aggregation.min_rows_per_side == 10
    assert cfg_obj.aggregation.invalid_warning_threshold == 0.01
    assert cfg_obj.aggregation.invalid_drop_threshold == 0.05
    assert cfg_obj.aggregation.primary_metric == "macro_mean_best_primary_margin"
