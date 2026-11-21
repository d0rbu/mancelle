"""II-6: run_target_real_evals passes kind through report.log.

Behavior tests — assert mock report.log is called with kind in data when
checkpoint_dir is provided.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    PooledTargetRealConfig,
)
from steering.eval.target_real_runner import run_target_real_evals

# ---------------------------------------------------------------------------
# Helper: a fake per_config_eval_fn that returns a minimal result dict.
# ---------------------------------------------------------------------------


def _make_fake_eval_fn(name: str = "trc"):
    def _fn(trc, provider):
        return {
            "name": trc.name,
            "target_real_mode": "pooled",
            "results": {"ARC-Easy": 0.5},
        }
    return _fn


def _make_cfg():
    return OmegaConf.create({
        "conditioning": {
            "identity": {
                "model_id": "m", "sae_id": "s", "layer": 0, "pooling": "max_pool",
            },
        },
        "model_max_seq_len": 32,
    })


# ---------------------------------------------------------------------------
# Test 1: pooled trc → report.log called with kind=target_real_pooled
# ---------------------------------------------------------------------------


def test_run_target_real_evals_pooled_calls_report_log_with_kind(tmp_path):
    """When run_target_real_evals processes a pooled trc with checkpoint_dir,
    get_report().log must be called with data containing {'kind': 'target_real_pooled'}."""
    mock_report = MagicMock()
    trc = PooledTargetRealConfig(
        name="tr_pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    fake_provider = object()

    with patch(
        "steering.eval.target_real_runner.build_context_feature_provider",
        return_value=fake_provider,
    ), patch(
        "steering.eval.target_real_runner.get_report",
        return_value=mock_report,
    ) as mock_get_report:
        run_target_real_evals(
            target_real_configs=[trc],
            cfg=_make_cfg(),
            output_dir_base=str(tmp_path / "out"),
            ddp_rank=0,
            per_config_eval_fn=_make_fake_eval_fn(),
            checkpoint_dir=str(tmp_path / "ckpt"),
        )

    mock_get_report.assert_called_once_with(checkpoint_dir=str(tmp_path / "ckpt"))
    mock_report.log.assert_called_once()
    call_kwargs = mock_report.log.call_args
    data = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
    # data is a list of dicts; at least one must have kind=target_real_pooled.
    assert any(
        isinstance(d, dict) and d.get("kind") == "target_real_pooled"
        for d in data
    ), (
        f"report.log data must contain {{'kind': 'target_real_pooled'}}; got data={data}"
    )


# ---------------------------------------------------------------------------
# Test 2: lockstep trc → report.log called with kind=target_real_lockstep
# ---------------------------------------------------------------------------


def test_run_target_real_evals_lockstep_calls_report_log_with_kind(tmp_path):
    """When run_target_real_evals processes a lockstep trc with checkpoint_dir,
    get_report().log must be called with kind=target_real_lockstep."""
    mock_report = MagicMock()
    trc = LockstepTargetRealConfig(
        name="tr_lockstep",
        cache_dir=tmp_path / "cache",
    )
    fake_provider = object()

    with patch(
        "steering.eval.target_real_runner.build_context_feature_provider",
        return_value=fake_provider,
    ), patch(
        "steering.eval.target_real_runner.get_report",
        return_value=mock_report,
    ):
        run_target_real_evals(
            target_real_configs=[trc],
            cfg=_make_cfg(),
            output_dir_base=str(tmp_path / "out"),
            ddp_rank=0,
            per_config_eval_fn=lambda t, p: {
                "name": t.name,
                "target_real_mode": "lockstep",
                "results": {},
            },
            checkpoint_dir=str(tmp_path / "ckpt"),
        )

    mock_report.log.assert_called_once()
    call_kwargs = mock_report.log.call_args
    data = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
    assert any(
        isinstance(d, dict) and d.get("kind") == "target_real_lockstep"
        for d in data
    ), (
        f"report.log data must contain {{'kind': 'target_real_lockstep'}}; got data={data}"
    )


# ---------------------------------------------------------------------------
# Test 3: without checkpoint_dir, report.log is NOT called
# ---------------------------------------------------------------------------


def test_run_target_real_evals_without_checkpoint_dir_does_not_call_report_log(tmp_path):
    """When checkpoint_dir is not provided, get_report must not be called."""
    trc = PooledTargetRealConfig(
        name="tr_pooled",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    fake_provider = object()

    with patch(
        "steering.eval.target_real_runner.build_context_feature_provider",
        return_value=fake_provider,
    ), patch(
        "steering.eval.target_real_runner.get_report",
    ) as mock_get_report:
        run_target_real_evals(
            target_real_configs=[trc],
            cfg=_make_cfg(),
            output_dir_base=str(tmp_path / "out"),
            ddp_rank=0,
            per_config_eval_fn=_make_fake_eval_fn(),
            # checkpoint_dir NOT provided (defaults to None)
        )

    mock_get_report.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: report_section_prefix in section name
# ---------------------------------------------------------------------------


def test_run_target_real_evals_report_section_includes_trc_name(tmp_path):
    """The report section name must include the trc.name for per-config
    granularity, matching the static-yaml path convention."""
    mock_report = MagicMock()
    trc = PooledTargetRealConfig(
        name="safety_trc",
        cache_dir=tmp_path / "cache",
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    fake_provider = object()

    with patch(
        "steering.eval.target_real_runner.build_context_feature_provider",
        return_value=fake_provider,
    ), patch(
        "steering.eval.target_real_runner.get_report",
        return_value=mock_report,
    ):
        run_target_real_evals(
            target_real_configs=[trc],
            cfg=_make_cfg(),
            output_dir_base=str(tmp_path / "out"),
            ddp_rank=0,
            per_config_eval_fn=_make_fake_eval_fn("safety_trc"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            report_section_prefix="Chat evaluation test_source",
        )

    mock_report.log.assert_called_once()
    section = mock_report.log.call_args.kwargs.get("section") or mock_report.log.call_args.args[0]
    assert "safety_trc" in str(section), (
        f"report section must include trc.name 'safety_trc'; got section={section!r}"
    )
    assert "Chat evaluation test_source" in str(section), (
        f"section must include the prefix; got section={section!r}"
    )


# ---------------------------------------------------------------------------
# Test 5: multiple target_real configs each log their own report section.
# ---------------------------------------------------------------------------


def test_run_target_real_evals_multiple_trcs_each_gets_report_log(tmp_path):
    """Two target_real configs → report.log called twice, once per config."""
    mock_report = MagicMock()
    trc_pooled = PooledTargetRealConfig(
        name="pooled_one",
        cache_dir=tmp_path / "cache_p",
        pooling_op=PoolingOp.BUCKET_MAX,
    )
    trc_lock = LockstepTargetRealConfig(
        name="lockstep_two",
        cache_dir=tmp_path / "cache_l",
    )
    fake_provider = object()

    with patch(
        "steering.eval.target_real_runner.build_context_feature_provider",
        return_value=fake_provider,
    ), patch(
        "steering.eval.target_real_runner.get_report",
        return_value=mock_report,
    ):
        run_target_real_evals(
            target_real_configs=[trc_pooled, trc_lock],
            cfg=_make_cfg(),
            output_dir_base=str(tmp_path / "out"),
            ddp_rank=0,
            per_config_eval_fn=lambda t, p: {
                "name": t.name,
                "target_real_mode": "pooled",
                "results": {},
            },
            checkpoint_dir=str(tmp_path / "ckpt"),
        )

    assert mock_report.log.call_count == 2, (
        f"report.log must be called once per trc; got {mock_report.log.call_count} calls"
    )
    # Extract all kinds from all calls.
    kinds = set()
    for call in mock_report.log.call_args_list:
        data = call.kwargs.get("data") or call.args[1]
        for d in data:
            if isinstance(d, dict) and "kind" in d:
                kinds.add(d["kind"])
    assert "target_real_pooled" in kinds
    assert "target_real_lockstep" in kinds
