"""Config wiring for the cross-version overlay snapshot harness."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.benchmark import verify_overlay_cache_byte_identity as harness


def test_percentile_producer_receives_complete_compression_config(
    monkeypatch, tmp_path: Path
) -> None:
    captured: list[DictConfig] = []

    def _run_percentiles(cfg: DictConfig) -> int:
        captured.append(cfg)
        return 0

    monkeypatch.setattr(harness, "run_pct", _run_percentiles)
    monkeypatch.setattr(harness, "run_tok", lambda _cfg: 1)

    harness._run_producers(OmegaConf.create({"segment_size": 128}), tmp_path)

    assert len(captured) == 1
    assert captured[0].zstd_level == 12


def test_optional_packed_rows_producer_receives_complete_config(
    monkeypatch, tmp_path: Path
) -> None:
    captured: list[DictConfig] = []
    monkeypatch.setattr(harness, "run_pct", lambda _cfg: 0)
    monkeypatch.setattr(harness, "run_tok", lambda _cfg: 0)

    def _run_packed_rows(cfg: DictConfig) -> int:
        captured.append(cfg)
        return 0

    monkeypatch.setattr(harness, "run_packed", _run_packed_rows)

    harness._run_producers(
        OmegaConf.create(
            {
                "segment_size": 128,
                "sae_width": "65k",
                "context_len": 1_024,
                "max_shards": 4,
            }
        ),
        tmp_path,
    )

    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.sae_width == "65k"
    assert cfg.buffer_size == 1_000
    assert cfg.rows_per_shard == 16_384
    assert cfg.zstd_level == 12
    assert cfg.resume_from_shard is None
    assert cfg.compress_workers == 0
    assert cfg.profile is False
    assert cfg.profile_summary_output is None
