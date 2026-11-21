"""Filesystem pathologies: atomic rename, disk-full, read-only dir, symlinked dir, unicode paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeRating
from steering.eval.feature_filter.runner import FeatureFilterRunner
from tests.feature_filter.fixtures.payload_builder import build_payload


class _Tok:
    @staticmethod
    def convert_tokens_to_string(toks: list[str]) -> str:
        return "".join(toks)


def _cfg(output_dir: str, *, feature_ids: list[int]) -> FeatureFilterConfig:
    d = output_dir
    return FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d,
        feature_ids=feature_ids,
        n_sae_features=max(feature_ids) + 1,
        samples_per_bin=3,
        interp_min_score=1,
        consistency_min_score=1,
        max_concurrent_fetches=1,
        max_concurrent_judges=1,
        neuronpedia_cache_dir=f"{d}/np",
        features_output_path=f"{d}/scores.jsonl",
        accepted_features_path=f"{d}/accepted.txt",
        report_output_path=f"{d}/report.json",
        runner_manifest_path=f"{d}/manifest.json",
        lockfile_path=f"{d}/.lock",
        judge_cache_path=f"{d}/judge.sqlite",
        judge=JudgeConfig(),
    )


def _run(cfg: FeatureFilterConfig) -> None:
    payloads = {i: build_payload(feature_id=i) for i in cfg.feature_ids or []}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    FeatureFilterRunner(
        cfg,
        np_cache=cache,
        judge=lambda desc, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    ).run()


def test_symlinked_output_dir_works(tmp_path: Path) -> None:
    """output_dir can be a symlink; the runner writes through it normally."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    _run(_cfg(str(link), feature_ids=[0, 1]))
    # Artifacts appear in both views because they're the same directory.
    assert (link / "report.json").is_file()
    assert (real / "report.json").is_file()


def test_output_dir_with_unicode_path(tmp_path: Path) -> None:
    """Unicode/emoji characters in output_dir path must not break writes."""
    unicode_dir = tmp_path / "features_漢字_🌟"
    unicode_dir.mkdir()
    _run(_cfg(str(unicode_dir), feature_ids=[0]))
    report = json.loads((unicode_dir / "report.json").read_text())
    assert report["n_total"] == 1


def test_trailing_slash_in_output_dir(tmp_path: Path) -> None:
    """Trailing slash on output_dir is tolerated (normalized by Path)."""
    d = str(tmp_path) + "/"
    _run(_cfg(d, feature_ids=[0]))
    assert (tmp_path / "report.json").is_file()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX perms only")
def test_disk_full_surfaces_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulated ENOSPC during the final atomic report write must raise OSError."""
    from steering.eval.feature_filter import runner as runner_mod

    real_write_text = Path.write_text

    def flaky_write_text(self: Path, data: str, *args, **kwargs):
        # Allow lockfile + manifest + JSONL; fail on report.json.tmp write.
        if self.name.endswith("report.json.tmp"):
            raise OSError(28, "No space left on device")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(runner_mod.Path, "write_text", flaky_write_text)

    cfg = _cfg(str(tmp_path), feature_ids=[0])
    payloads = {0: build_payload(feature_id=0)}
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    runner = FeatureFilterRunner(
        cfg,
        np_cache=cache,
        judge=lambda d, s: InterpJudgeRating(3, 3),
        tokenizer=_Tok(),
    )
    with pytest.raises(OSError):
        runner.run()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX perms only")
def test_read_only_output_dir_fails_on_lock_acquire(tmp_path: Path) -> None:
    """If output_dir is read-only, the runner cannot create the lockfile and errors out."""
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        cfg = _cfg(str(ro), feature_ids=[0])
        payloads = {0: build_payload(feature_id=0)}
        cache = MagicMock()
        cache.get.side_effect = lambda fid: payloads[fid]
        runner = FeatureFilterRunner(
            cfg,
            np_cache=cache,
            judge=lambda d, s: InterpJudgeRating(3, 3),
            tokenizer=_Tok(),
        )
        with pytest.raises((PermissionError, OSError)):
            runner.run()
    finally:
        ro.chmod(0o700)
