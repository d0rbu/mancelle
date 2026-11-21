"""Roundtrip probe statistics remain separate from direction scaling (#1228)."""

from __future__ import annotations

from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.roundtrip_scoring import (
    _roundtrip_probe_stats_path,
)
from steering.eval.roundtrip_steering import RoundtripEvalConfig


def _cfg(override: str, main: str) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path=main,
        roundtrip_steering=RoundtripEvalConfig(
            enabled=True, feature_statistics_path=override
        ),
    )


def test_probe_resolver_prefers_override() -> None:
    assert _roundtrip_probe_stats_path(_cfg("/override", "/main")) == "/override"


def test_probe_resolver_falls_back_to_main() -> None:
    assert _roundtrip_probe_stats_path(_cfg("", "/main")) == "/main"
