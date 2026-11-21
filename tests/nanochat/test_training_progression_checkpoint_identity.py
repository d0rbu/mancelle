import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

import scripts.train.training_progression as tp

_EVALS = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}


def _cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "model_tag": "d12",
            "model_family": "decoder",
            "checkpoint_root": "base_checkpoints",
            "checkpoint_prefix": "",
            "checkpoint_source": "local",
            "checkpoints_dir": str(tmp_path / "data"),
            "results_file": str(tmp_path / "results.json"),
            "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
            "evals": "core",
            "num_seeds": 1,
            "base_seed": 1234,
            "device_type": "cpu",
            "dtype": "float32",
        }
    )


def _write_checkpoint(cfg, model_bytes: bytes = b"model-v1") -> Path:
    checkpoint_dir = Path(str(cfg.checkpoints_dir)) / "base_checkpoints" / "d12"
    checkpoint_dir.mkdir(parents=True)
    model_path = checkpoint_dir / "model_000500.pt"
    model_path.write_bytes(model_bytes)
    (checkpoint_dir / "meta_000500.json").write_text('{"step": 500}', encoding="utf-8")
    return model_path


def _mock_evaluator(cfg):
    def evaluate(spec, *_args):
        spec.checkpoint_identity = tp._checkpoint_identity_for_spec(spec, cfg)
        return _EVALS

    return patch.object(tp, "evaluate_single_checkpoint", side_effect=evaluate)


def test_replaced_checkpoint_marks_cached_result_pending(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model_path = _write_checkpoint(cfg)

    with _mock_evaluator(cfg) as evaluate:
        tp.collect_results(cfg)
        model_path.write_bytes(b"model-v2")
        tp.collect_results(cfg)

    assert evaluate.call_count == 2


def test_unchanged_checkpoint_reuses_cached_result(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model_path = _write_checkpoint(cfg)

    with _mock_evaluator(cfg) as evaluate:
        tp.collect_results(cfg)
        evaluate.reset_mock()
        tp.collect_results(cfg)

    evaluate.assert_not_called()
    result = json.loads(Path(str(cfg.results_file)).read_text(encoding="utf-8"))
    identity = result["checkpoints"][0]["checkpoint_identity"]
    assert identity["model"]["size"] == model_path.stat().st_size
    assert (set(identity), set(identity["model"])) == (
        {"model", "meta"},
        {"size", "sha256"},
    )


def test_legacy_result_is_accepted_with_one_time_warning(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg.checkpoint_source = "remote"
    legacy = {
        "model_tag": "d12",
        "checkpoints": [{"step": 500, "phase": "pretrain", "evals": _EVALS}],
    }
    Path(str(cfg.results_file)).write_text(
        json.dumps(legacy),
        encoding="utf-8",
    )
    monkeypatch.setattr(tp, "_legacy_checkpoint_identity_warned", False, raising=False)

    with (
        patch.object(tp, "fetch_remote_checkpoint") as fetch,
        patch.object(tp, "evaluate_single_checkpoint") as evaluate,
        patch.object(tp.logger, "warning") as warning,
    ):
        tp.collect_results(cfg)
        tp.collect_results(cfg)

    fetch.assert_not_called()
    evaluate.assert_not_called()
    warning.assert_called_once()
    assert "checkpoint_identity" in warning.call_args.args[0]
