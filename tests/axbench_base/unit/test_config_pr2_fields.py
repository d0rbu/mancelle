"""Tests for Stage 0c batching and verification-prefill config fields."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path

from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    VerificationPrefillMode,
)


def test_qc_batch_size_field_default():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    assert "qc_batch_size" in fields
    assert fields["qc_batch_size"].default == 16


def test_qc_batch_size_is_int():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
    )
    assert isinstance(cfg.qc_batch_size, int)
    assert cfg.qc_batch_size == 16


def test_qc_pipeline_field_default():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    assert "qc_pipeline" in fields
    assert fields["qc_pipeline"].default is True


def test_qc_pipeline_is_bool():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
    )
    assert isinstance(cfg.qc_pipeline, bool)
    assert cfg.qc_pipeline is True


def test_verification_prefill_mode_field_default():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    assert "verification_prefill_mode" in fields
    assert "batched_generation_mode" not in fields
    assert fields["verification_prefill_mode"].default == "shared"


def test_verification_prefill_mode_is_str():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
    )
    assert isinstance(cfg.verification_prefill_mode, str)
    assert cfg.verification_prefill_mode == "shared"


def test_all_three_fields_overridable():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
        qc_batch_size=32,
        qc_pipeline=False,
        verification_prefill_mode=VerificationPrefillMode.REPEATED,
    )
    assert cfg.qc_batch_size == 32
    assert cfg.qc_pipeline is False
    assert cfg.verification_prefill_mode == "repeated"


def test_stage0c_multi_gpu_block_size_field_default():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    assert "stage0c_multi_gpu_block_size" in fields
    assert fields["stage0c_multi_gpu_block_size"].default == 16


def test_verification_cross_prefix_batch_size_field_default():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    assert "verification_cross_prefix_batch_size" in fields
    assert fields["verification_cross_prefix_batch_size"].default == 4


def test_verification_api_prefix_wave_size_field_default():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    assert "verification_api_prefix_wave_size" in fields
    assert fields["verification_api_prefix_wave_size"].default == 100


def test_stage0c_multi_gpu_block_size_is_int():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
    )
    assert isinstance(cfg.stage0c_multi_gpu_block_size, int)
    assert cfg.stage0c_multi_gpu_block_size == 16


def test_verification_cross_prefix_batch_size_is_int():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
    )
    assert isinstance(cfg.verification_cross_prefix_batch_size, int)
    assert cfg.verification_cross_prefix_batch_size == 4


def test_verification_api_prefix_wave_size_is_int():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
    )
    assert isinstance(cfg.verification_api_prefix_wave_size, int)
    assert cfg.verification_api_prefix_wave_size == 100


def test_stage0c_multi_gpu_block_size_overridable():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
        stage0c_multi_gpu_block_size=64,
    )
    assert cfg.stage0c_multi_gpu_block_size == 64


def test_verification_cross_prefix_batch_size_overridable():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
        verification_cross_prefix_batch_size=8,
    )
    assert cfg.verification_cross_prefix_batch_size == 8


def test_verification_api_prefix_wave_size_overridable():
    cfg = AxBenchBaseConfig(
        feature_dir="/x",
        feature_statistics_path="/y",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out",
        verification_api_prefix_wave_size=48,
    )
    assert cfg.verification_api_prefix_wave_size == 48


# ---------------------------------------------------------------------------
# VerificationPrefillMode StrEnum contract
# ---------------------------------------------------------------------------


def test_verification_prefill_mode_is_strenum_with_exact_members():
    assert set(VerificationPrefillMode) == {
        VerificationPrefillMode.SHARED,
        VerificationPrefillMode.REPEATED,
    }
    assert VerificationPrefillMode.SHARED == "shared"
    assert VerificationPrefillMode.REPEATED == "repeated"
    assert str(VerificationPrefillMode.SHARED) == "shared"
    assert VerificationPrefillMode("repeated") is VerificationPrefillMode.REPEATED


def test_verification_prefill_mode_json_serializes_to_string_value():
    assert json.dumps({"k": VerificationPrefillMode.SHARED}) == ('{"k": "shared"}')
    assert json.dumps({"k": VerificationPrefillMode.REPEATED}) == ('{"k": "repeated"}')


def test_verification_prefill_mode_default_is_strenum_member():
    fields = {f.name: f for f in dataclasses.fields(AxBenchBaseConfig)}
    default = fields["verification_prefill_mode"].default
    assert default is VerificationPrefillMode.SHARED
    assert default == "shared"


# The Transformers floor and the exact version the lock resolves to. Both are
# asserted below and must be updated together, with a note saying what was
# re-verified -- that coupling is the whole point of this test.
#
# The floor is >=5.10.0 because commit 02f828dc raised it there: FA3 raised a
# KeyError below that version. The lock resolves 5.14.1, the version the suite
# is actually exercised against.
_PROVEN_TRANSFORMERS_FLOOR = "5.10.0"
_PROVEN_TRANSFORMERS_LOCKED = "5.14.1"


def test_transformers_requirement_and_lock_are_pinned_to_proven_version() -> None:
    """The declared floor and the locked version must both stay proven.

    Bumping either without re-proving is exactly what this test exists to stop,
    so it fails on any drift rather than accepting a range.

    Historical note (issue #929): this sat red on ``main`` because ``02f828dc``
    raised the floor without updating it here. A pin asserting a version the
    project has deliberately moved off proves nothing about the version
    actually in use -- it fails unconditionally, which is indistinguishable
    from being ignored. If you are here because it failed, re-verify against
    the new version and update the two constants above; do not relax the
    assertions.
    """
    repo_root = Path(__file__).resolve().parents[3]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    # Whitespace-normalized comparison rather than literal substring
    # membership, so reformatting ``pyproject.toml`` (``transformers >= 5.10.0``)
    # cannot break this independently of the version, and a second
    # ``transformers`` entry is caught rather than silently ignored.
    declared = [
        dependency.replace(" ", "")
        for dependency in pyproject["project"]["dependencies"]
        if dependency.replace(" ", "").startswith("transformers")
    ]
    assert declared == [f"transformers>={_PROVEN_TRANSFORMERS_FLOOR}"], (
        f"pyproject.toml declares {declared!r}; expected exactly one entry "
        f"'transformers>={_PROVEN_TRANSFORMERS_FLOOR}'"
    )

    with (repo_root / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    packages = {package["name"]: package for package in lock["package"]}
    assert packages["transformers"]["version"] == _PROVEN_TRANSFORMERS_LOCKED
    requirements = packages["nano-gpt-pretrain-steer"]["metadata"]["requires-dist"]
    project_requirement = next(
        requirement for requirement in requirements if requirement["name"] == "transformers"
    )
    assert project_requirement["specifier"] == f">={_PROVEN_TRANSFORMERS_FLOOR}"
