"""Behavioural turn-rejection coverage for the fixed-only eval consumers.

The AST census in ``tests/steering/test_fixed_consumer_segment_mode_guards.py``
proves each consumer *mentions* the shared guard and passes it the right
subject, but a static call cannot observe the guard's predicate at runtime nor
its position relative to the work it is supposed to precede. Every test here
drives the real entry point with a turn-delimited witness and turns the first
genuinely post-guard collaborator into ``pytest.fail``: a guard that is
inverted, dead, or moved below that collaborator fails the test.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.chat import steer_chat_cli
from scripts.eval import (
    base_eval,
    build_eval_context_cache,
    chat_eval,
    completion_steering_eval,
    shuffled_conditioning_eval,
    steering_eval,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval import dclm_eval
from steering.segmentation import SegmentMode
from tests.conftest import build_tiny_encdec
from tests.statistics_doubles import fixed_statistics_double

TURN_MODE = SegmentMode.turn_delimited_fixed_tokens.value
FIXED_MODE = SegmentMode.fixed_tokens.value


def _encdec_stub() -> EncoderDecoderGPT:
    """An ``isinstance``-true encoder-decoder that owns no weights."""

    return object.__new__(EncoderDecoderGPT)


def _meta(mode: str) -> dict[str, Any]:
    return {
        "sft_segment_mode": mode,
        "step": 7,
        "model_config": {"sequence_len": 128},
    }


def _stats(mode: str) -> SimpleNamespace:
    """A statistics holder carrying the real ``MetadataStats`` witness shape."""

    return fixed_statistics_double(
        segment_mode=SegmentMode(mode),
        segment_size=64,
        context_window=64,
        n_sae_features=4,
    )


def _compose(monkeypatch: pytest.MonkeyPatch, module: Any, cfg: Any) -> None:
    """Replace a ``compose``-driven entry point's config resolution."""

    monkeypatch.setattr(module.sys, "argv", [module.__name__])
    monkeypatch.setattr(module, "initialize_config_dir", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(module, "compose", lambda **_kwargs: cfg)


def _cpu_compute(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(module, "compute_init", lambda _dt: (False, 0, 0, 1, t.device("cpu")))


# ---------------------------------------------------------------------------
# Checkpoint-lineage consumers
# ---------------------------------------------------------------------------


def test_chat_eval_rejects_turn_without_target_real_before_feature_config_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: admitting unsupported turn work, or moving admission below feature-config parsing.
    cfg = OmegaConf.create(
        {
            "device_type": "cpu",
            "dtype": "float32",
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
            "task_name": None,
            "feature_configs": None,
        }
    )
    _cpu_compute(monkeypatch, chat_eval)
    meta = _meta(TURN_MODE)
    (tmp_path / "meta_000007.json").write_text(json.dumps(meta))
    model = build_tiny_encdec()
    t.save(model.state_dict(), tmp_path / "model_000007.pt")
    monkeypatch.setattr(chat_eval, "get_checkpoint_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(
        chat_eval,
        "load_model",
        lambda *_args, **_kwargs: (model, object(), meta),
    )
    monkeypatch.setattr(
        chat_eval,
        "parse_feature_configs",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached feature-config parsing"),
    )

    with pytest.raises(ValueError, match=r"chat_eval.*turn_delimited_fixed_tokens"):
        chat_eval.main.__wrapped__(cfg)


def test_dclm_eval_rejects_turn_checkpoint_before_conditioning_offset_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: inverting/removing dclm_eval's encoder-decoder guard, or moving it
    # below target-conditioning resolution.
    cfg = OmegaConf.create(
        {
            "device_type": "cpu",
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
            "feature_configs": None,
        }
    )
    _cpu_compute(monkeypatch, dclm_eval)
    monkeypatch.setattr(dclm_eval, "parse_feature_configs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        dclm_eval,
        "parse_target_real_configs_from_cfg",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        dclm_eval,
        "load_model",
        lambda *_args, **_kwargs: (_encdec_stub(), object(), _meta(TURN_MODE)),
    )
    monkeypatch.setattr(
        dclm_eval,
        "resolve_next_segment_conditioning_offset",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached conditioning-offset resolution"),
    )

    with pytest.raises(ValueError, match=r"dclm_eval.*turn_delimited_fixed_tokens"):
        dclm_eval.main.__wrapped__(cfg)


def test_base_eval_rejects_turn_checkpoint_before_sequence_len_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: inverting/removing base_eval's encoder-decoder guard, or moving it
    # below the model/sequence-length reconciliation.
    cfg = OmegaConf.create(
        {
            "device_type": "cpu",
            "eval": "core",
            "hf_path": None,
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
            "feature_configs": None,
        }
    )
    _cpu_compute(monkeypatch, base_eval)
    monkeypatch.setattr(base_eval, "parse_feature_configs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        base_eval,
        "load_model",
        lambda *_args, **_kwargs: (_encdec_stub(), object(), _meta(TURN_MODE)),
    )
    monkeypatch.setattr(
        base_eval,
        "_sync_cfg_model_max_seq_len_with_loaded_model",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached sequence-length sync"),
    )

    with pytest.raises(ValueError, match=r"base_eval.*turn_delimited_fixed_tokens"):
        base_eval.main.__wrapped__(cfg)


def test_steer_chat_cli_rejects_turn_checkpoint_before_tokenizer_specials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kills: inverting/removing steer_chat_cli's guard, or moving it below the
    # special-token resolution that starts engine construction.
    cfg = OmegaConf.create(
        {
            "device_type": "cpu",
            "dtype": "float32",
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
            "mode": "chat",
            "feature_conditions": None,
        }
    )
    _compose(monkeypatch, steer_chat_cli, cfg)
    _cpu_compute(monkeypatch, steer_chat_cli)
    tokenizer = SimpleNamespace(
        get_bos_token_id=lambda: pytest.fail("turn guard reached tokenizer specials"),
        encode_special=lambda _name: pytest.fail("turn guard reached tokenizer specials"),
    )
    monkeypatch.setattr(
        steer_chat_cli,
        "load_model",
        lambda *_args, **_kwargs: (_encdec_stub(), tokenizer, _meta(TURN_MODE)),
    )

    with pytest.raises(ValueError, match=r"steer_chat_cli.*turn_delimited_fixed_tokens"):
        steer_chat_cli.main()


def test_steering_eval_rejects_turn_checkpoint_before_steering_config_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: inverting/removing steering_eval's guard, or moving it below the
    # steering-config load that drives generation.
    steering_config = tmp_path / "steering.yaml"
    steering_config.write_text("features: []\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "steering_config": str(steering_config),
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
        }
    )
    _compose(monkeypatch, steering_eval, cfg)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-in-this-test")
    monkeypatch.setattr(steering_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        steering_eval,
        "_load_model_for_steering_eval",
        lambda *_args, **_kwargs: (_encdec_stub(), object(), _meta(TURN_MODE)),
    )
    monkeypatch.setattr(
        steering_eval,
        "load_config",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached steering-config load"),
    )

    with pytest.raises(ValueError, match=r"steering_eval.*turn_delimited_fixed_tokens"):
        steering_eval.main()


# ---------------------------------------------------------------------------
# Feature-artifact consumers
# ---------------------------------------------------------------------------


def _completion_cfg(feature_dir: Path) -> Any:
    return OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "output_dir": None,
            "stage": "all",
            "mode": "insertion",
            "run_judge": False,
            "run_feature_analysis": False,
            "force_recompute": False,
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
        }
    )


def test_completion_steering_eval_rejects_turn_checkpoint_before_tokenizer_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: inverting/removing completion_steering_eval's checkpoint guard, or
    # moving it below the eval-tokenizer contract.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _compose(monkeypatch, completion_steering_eval, _completion_cfg(feature_dir))
    monkeypatch.setattr(completion_steering_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        completion_steering_eval,
        "load_model",
        lambda **_kwargs: (_encdec_stub(), object(), _meta(TURN_MODE)),
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "require_bos_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached tokenizer contract"),
    )

    with pytest.raises(
        ValueError,
        match=r"completion_steering_eval.*turn_delimited_fixed_tokens",
    ):
        completion_steering_eval.main()

    assert not (feature_dir / "completion_steering_eval").exists()


def test_completion_steering_eval_rejects_turn_statistics_before_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: dropping the statistics witness from the feature-dir guard, or
    # restoring the output-directory creation above it.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _compose(monkeypatch, completion_steering_eval, _completion_cfg(feature_dir))
    monkeypatch.setattr(completion_steering_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        completion_steering_eval,
        "load_model",
        lambda **_kwargs: (_encdec_stub(), object(), _meta(FIXED_MODE)),
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "require_bos_tokenizer",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "resolve_next_segment_conditioning_offset",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "find_statistics_path",
        lambda _dir: feature_dir / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "load_statistics",
        lambda _path: _stats(TURN_MODE),
    )

    with pytest.raises(
        ValueError,
        match=r"completion_steering_eval.*turn_delimited_fixed_tokens",
    ):
        completion_steering_eval.main()

    assert not (feature_dir / "completion_steering_eval").exists()


class _GuardSinkReached(RuntimeError):
    """Raised by the first collaborator after a consumer's output-dir creation."""


def test_completion_steering_eval_creates_output_dir_only_after_both_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: deleting the relocated output-directory creation, which the two
    # rejection tests above would not notice.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _compose(monkeypatch, completion_steering_eval, _completion_cfg(feature_dir))
    model = _encdec_stub()
    model.encoder = cast(Any, SimpleNamespace(config=SimpleNamespace(num_buckets=2)))
    monkeypatch.setattr(completion_steering_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        completion_steering_eval,
        "load_model",
        lambda **_kwargs: (model, object(), _meta(FIXED_MODE)),
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "require_bos_tokenizer",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "resolve_next_segment_conditioning_offset",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "find_statistics_path",
        lambda _dir: feature_dir / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        completion_steering_eval,
        "load_statistics",
        lambda _path: _stats(FIXED_MODE),
    )

    def stop(*_args: Any, **_kwargs: Any) -> Any:
        raise _GuardSinkReached("feature pipeline reached")

    monkeypatch.setattr(
        completion_steering_eval,
        "build_feature_pipeline_from_cfg_block",
        stop,
    )

    with pytest.raises(_GuardSinkReached, match="feature pipeline reached"):
        completion_steering_eval.main()

    assert (feature_dir / "completion_steering_eval").is_dir()


def _shuffled_cfg(feature_dir: Path) -> Any:
    return OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "output_dir": None,
            "stage": "all",
            "force_recompute": False,
            "threshold": 2048.9,
            "source": "encdec",
            "model_tag": "d12",
            "step": 7,
        }
    )


def test_shuffled_conditioning_eval_rejects_turn_statistics_before_shard_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: dropping the statistics witness from the feature-dir guard, or
    # restoring the output-directory creation above it.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _compose(monkeypatch, shuffled_conditioning_eval, _shuffled_cfg(feature_dir))
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "find_statistics_path",
        lambda _dir: feature_dir / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "load_statistics",
        lambda _path: _stats(TURN_MODE),
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "glob_features",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached shard discovery"),
    )

    with pytest.raises(
        ValueError,
        match=r"shuffled_conditioning_eval.*turn_delimited_fixed_tokens",
    ):
        shuffled_conditioning_eval.main()

    assert not (feature_dir / "shuffled_conditioning_eval").exists()


def test_shuffled_conditioning_eval_rejects_turn_checkpoint_before_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: inverting/removing shuffled_conditioning_eval's checkpoint guard,
    # or restoring the output-directory creation above it.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _compose(monkeypatch, shuffled_conditioning_eval, _shuffled_cfg(feature_dir))
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "find_statistics_path",
        lambda _dir: feature_dir / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "load_statistics",
        lambda _path: _stats(FIXED_MODE),
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "glob_features",
        lambda *_args, **_kwargs: [feature_dir / "shard_00000.safetensors"],
    )
    monkeypatch.setattr(shuffled_conditioning_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "load_model",
        lambda **_kwargs: (_encdec_stub(), object(), _meta(TURN_MODE)),
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "require_bos_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached tokenizer contract"),
    )

    with pytest.raises(
        ValueError,
        match=r"shuffled_conditioning_eval.*turn_delimited_fixed_tokens",
    ):
        shuffled_conditioning_eval.main()

    assert not (feature_dir / "shuffled_conditioning_eval").exists()


def test_shuffled_conditioning_eval_creates_output_dir_only_after_the_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: deleting the relocated output-directory creation from the
    # generate stage.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _compose(monkeypatch, shuffled_conditioning_eval, _shuffled_cfg(feature_dir))
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "find_statistics_path",
        lambda _dir: feature_dir / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "load_statistics",
        lambda _path: _stats(FIXED_MODE),
    )
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "glob_features",
        lambda *_args, **_kwargs: [feature_dir / "shard_00000.safetensors"],
    )
    monkeypatch.setattr(shuffled_conditioning_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        shuffled_conditioning_eval,
        "load_model",
        lambda **_kwargs: (_encdec_stub(), object(), _meta(FIXED_MODE)),
    )

    def stop(*_args: Any, **_kwargs: Any) -> Any:
        raise _GuardSinkReached("tokenizer contract reached")

    monkeypatch.setattr(shuffled_conditioning_eval, "require_bos_tokenizer", stop)

    with pytest.raises(_GuardSinkReached, match="tokenizer contract reached"):
        shuffled_conditioning_eval.main()

    assert (feature_dir / "shuffled_conditioning_eval").is_dir()


def test_fixed_cache_builder_rejects_turn_statistics_before_pipeline_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # kills: admitting turn statistics through the legacy/default fixed converter.
    cfg = OmegaConf.create(
        {
            "segment_mode": FIXED_MODE,
            "feature_dir": str(tmp_path / "features"),
            "stats_path": str(tmp_path / "feature_statistics.safetensors"),
            "contexts_file": str(tmp_path / "contexts.yaml"),
            "n_sae_features": 4,
            "num_buckets": 2,
            "shard_size": 8,
            "seed": 42,
        }
    )
    monkeypatch.setattr(build_eval_context_cache, "load_context_records", lambda _path: [])
    monkeypatch.setattr(
        build_eval_context_cache,
        "_required_contexts_contract",
        lambda *_args, **_kwargs: (None, 0),
    )
    monkeypatch.setattr(
        build_eval_context_cache,
        "glob_features",
        lambda *_args, **_kwargs: [tmp_path / "features" / "shard_00000.safetensors"],
    )
    monkeypatch.setattr(
        build_eval_context_cache,
        "load_statistics",
        lambda _path: _stats(TURN_MODE),
    )
    monkeypatch.setattr(
        build_eval_context_cache,
        "build_feature_pipeline_from_cfg_block",
        lambda *_args, **_kwargs: pytest.fail("turn guard reached feature-pipeline build"),
    )

    with pytest.raises(
        ValueError,
        match=r"eval context cache conversion.*turn_delimited_fixed_tokens",
    ):
        build_eval_context_cache.build_cache_from_config(cfg)
