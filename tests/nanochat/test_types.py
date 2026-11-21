"""Tests for typed enums (StrEnum) and dataclasses used across the codebase.

These tests are written TDD-style: they import types that do not yet exist, so
they will fail with ImportError until the corresponding modules define them.

Enums tested:
    - TaskType          (nanochat.core_eval)
    - DatasetSplit      (nanochat.dataset)
    - DeviceType        (nanochat.common)
    - FlashAttnImpl     (nanochat.flash_attention)
    - ModelPhase        (nanochat.checkpoint_manager)
    - ModelSource       (nanochat.checkpoint_manager)
    - EvalMode          (scripts.eval.base_eval)
    - OptimizerKind     (nanochat.gpt)

Dataclasses tested:
    - GitInfo           (nanochat.report)
    - GPUInfo           (nanochat.report)
    - SystemInfo        (nanochat.report)
    - CostEstimate      (nanochat.report)
    - TaskMeta          (nanochat.core_eval)
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum

import pytest
import torch


def _assert_frozen_assignment(instance: object, field: str, value: object) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, value)


# =============================================================================
# Enum tests — each test class imports lazily so tests can run incrementally
# =============================================================================


class TestTaskType:
    """Tests for TaskType StrEnum (nanochat.core_eval)."""

    def test_is_str_enum(self):
        from nanochat.core_eval import TaskType

        assert issubclass(TaskType, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.core_eval import TaskType

        assert set(TaskType.__members__.keys()) == {
            "MULTIPLE_CHOICE",
            "SCHEMA",
            "LANGUAGE_MODELING",
        }

    def test_string_values(self):
        from nanochat.core_eval import TaskType

        assert TaskType.MULTIPLE_CHOICE == "multiple_choice"
        assert TaskType.SCHEMA == "schema"
        assert TaskType.LANGUAGE_MODELING == "language_modeling"

    def test_str_equality(self):
        from nanochat.core_eval import TaskType

        assert TaskType.MULTIPLE_CHOICE == "multiple_choice"
        assert "schema" == TaskType.SCHEMA

    def test_usable_as_dict_key(self):
        from nanochat.core_eval import TaskType

        d: dict[str, int] = {TaskType.MULTIPLE_CHOICE: 1}
        assert d["multiple_choice"] == 1


class TestDatasetSplit:
    """Tests for DatasetSplit StrEnum (nanochat.dataset)."""

    def test_is_str_enum(self):
        from nanochat.dataset import DatasetSplit

        assert issubclass(DatasetSplit, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.dataset import DatasetSplit

        assert set(DatasetSplit.__members__.keys()) == {"TRAIN", "VAL"}

    def test_string_values(self):
        from nanochat.dataset import DatasetSplit

        assert DatasetSplit.TRAIN == "train"
        assert DatasetSplit.VAL == "val"

    def test_str_equality(self):
        from nanochat.dataset import DatasetSplit

        assert DatasetSplit.TRAIN == "train"
        assert "val" == DatasetSplit.VAL


class TestDeviceType:
    """Tests for DeviceType StrEnum (nanochat.common)."""

    def test_is_str_enum(self):
        from nanochat.common import DeviceType

        assert issubclass(DeviceType, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.common import DeviceType

        assert set(DeviceType.__members__.keys()) == {"CUDA", "CPU", "MPS"}

    def test_string_values(self):
        from nanochat.common import DeviceType

        assert DeviceType.CUDA == "cuda"
        assert DeviceType.CPU == "cpu"
        assert DeviceType.MPS == "mps"

    def test_str_equality(self):
        from nanochat.common import DeviceType

        assert DeviceType.CPU == "cpu"
        assert "cuda" == DeviceType.CUDA

    def test_works_with_torch_device(self):
        from nanochat.common import DeviceType

        dev = torch.device(DeviceType.CPU)
        assert dev.type == "cpu"


class TestFlashAttnImpl:
    """Tests for FlashAttnImpl StrEnum (nanochat.flash_attention)."""

    def test_is_str_enum(self):
        from nanochat.flash_attention import FlashAttnImpl

        assert issubclass(FlashAttnImpl, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.flash_attention import FlashAttnImpl

        assert set(FlashAttnImpl.__members__.keys()) == {
            "FA4",
            "FA3",
            "SDPA",
            "AUTO",
        }

    def test_string_values(self):
        from nanochat.flash_attention import FlashAttnImpl

        assert FlashAttnImpl.FA4 == "fa4"
        assert FlashAttnImpl.FA3 == "fa3"
        assert FlashAttnImpl.SDPA == "sdpa"
        assert FlashAttnImpl.AUTO == "auto"

    def test_str_equality(self):
        from nanochat.flash_attention import FlashAttnImpl

        assert FlashAttnImpl.SDPA == "sdpa"
        assert "auto" == FlashAttnImpl.AUTO


class TestModelPhase:
    """Tests for ModelPhase StrEnum (nanochat.checkpoint_manager)."""

    def test_is_str_enum(self):
        from nanochat.checkpoint_manager import ModelPhase

        assert issubclass(ModelPhase, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.checkpoint_manager import ModelPhase

        assert set(ModelPhase.__members__.keys()) == {"TRAIN", "EVAL"}

    def test_string_values(self):
        from nanochat.checkpoint_manager import ModelPhase

        assert ModelPhase.TRAIN == "train"
        assert ModelPhase.EVAL == "eval"

    def test_str_equality(self):
        from nanochat.checkpoint_manager import ModelPhase

        assert ModelPhase.TRAIN == "train"
        assert "eval" == ModelPhase.EVAL


class TestModelSource:
    """Tests for ModelSource StrEnum (nanochat.checkpoint_manager)."""

    def test_is_str_enum(self):
        from nanochat.checkpoint_manager import ModelSource

        assert issubclass(ModelSource, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.checkpoint_manager import ModelSource

        assert set(ModelSource.__members__.keys()) == {
            "BASE",
            "SFT",
            "STANDARD",
            "SAFETY",
            "CAPABILITIES",
            "RL",
            "ENCDEC",
        }

    def test_string_values(self):
        from nanochat.checkpoint_manager import ModelSource

        assert ModelSource.BASE == "base"
        assert ModelSource.SFT == "sft"
        assert ModelSource.STANDARD == "standard"
        assert ModelSource.SAFETY == "safety"
        assert ModelSource.CAPABILITIES == "capabilities"
        assert ModelSource.RL == "rl"
        assert ModelSource.ENCDEC == "encdec"

    def test_str_equality(self):
        from nanochat.checkpoint_manager import ModelSource

        assert ModelSource.BASE == "base"
        assert "encdec" == ModelSource.ENCDEC


class TestEvalMode:
    """Tests for EvalMode StrEnum (scripts.eval.base_eval)."""

    def test_is_str_enum(self):
        from scripts.eval.base_eval import EvalMode

        assert issubclass(EvalMode, StrEnum)

    def test_members_exhaustive(self):
        from scripts.eval.base_eval import EvalMode

        assert set(EvalMode.__members__.keys()) == {"CORE", "BPB", "SAMPLE"}

    def test_string_values(self):
        from scripts.eval.base_eval import EvalMode

        assert EvalMode.CORE == "core"
        assert EvalMode.BPB == "bpb"
        assert EvalMode.SAMPLE == "sample"

    def test_str_equality(self):
        from scripts.eval.base_eval import EvalMode

        assert EvalMode.CORE == "core"
        assert "sample" == EvalMode.SAMPLE


class TestOptimizerKind:
    """Tests for OptimizerKind StrEnum (nanochat.gpt)."""

    def test_is_str_enum(self):
        from nanochat.gpt import OptimizerKind

        assert issubclass(OptimizerKind, StrEnum)

    def test_members_exhaustive(self):
        from nanochat.gpt import OptimizerKind

        assert set(OptimizerKind.__members__.keys()) == {"ADAMW", "MUON"}

    def test_string_values(self):
        from nanochat.gpt import OptimizerKind

        assert OptimizerKind.ADAMW == "adamw"
        assert OptimizerKind.MUON == "muon"

    def test_str_equality(self):
        from nanochat.gpt import OptimizerKind

        assert OptimizerKind.ADAMW == "adamw"
        assert "muon" == OptimizerKind.MUON


# =============================================================================
# Dataclass tests
# =============================================================================


class TestGitInfo:
    """Tests for GitInfo frozen dataclass (nanochat.report)."""

    def test_is_dataclass(self):
        from nanochat.report import GitInfo

        assert dataclasses.is_dataclass(GitInfo)

    def test_construction(self):
        from nanochat.report import GitInfo

        info = GitInfo(commit="abc123", branch="main", dirty=False, message="init")
        assert info.commit == "abc123"
        assert info.branch == "main"
        assert info.dirty is False
        assert info.message == "init"

    def test_frozen_immutability(self):
        from nanochat.report import GitInfo

        info = GitInfo(commit="abc123", branch="main", dirty=False, message="init")
        _assert_frozen_assignment(info, "commit", "def456")

    def test_asdict_roundtrip(self):
        from nanochat.report import GitInfo

        info = GitInfo(commit="abc123", branch="main", dirty=True, message="wip")
        d = dataclasses.asdict(info)
        assert isinstance(d, dict)
        reconstructed = GitInfo(**d)
        assert reconstructed == info

    def test_fields(self):
        from nanochat.report import GitInfo

        names = {f.name for f in dataclasses.fields(GitInfo)}
        assert names == {"commit", "branch", "dirty", "message"}


class TestGPUInfo:
    """Tests for GPUInfo frozen dataclass (nanochat.report)."""

    def test_is_dataclass(self):
        from nanochat.report import GPUInfo

        assert dataclasses.is_dataclass(GPUInfo)

    def test_construction_minimal(self):
        from nanochat.report import GPUInfo

        info = GPUInfo(available=False)
        assert info.available is False
        assert info.count == 0
        assert info.names == []
        assert info.memory_gb == []
        assert info.cuda_version == ""

    def test_construction_full(self):
        from nanochat.report import GPUInfo

        info = GPUInfo(
            available=True,
            count=2,
            names=["A100", "A100"],
            memory_gb=[80.0, 80.0],
            cuda_version="12.4",
        )
        assert info.available is True
        assert info.count == 2
        assert info.names == ["A100", "A100"]
        assert info.memory_gb == [80.0, 80.0]
        assert info.cuda_version == "12.4"

    def test_frozen_immutability(self):
        from nanochat.report import GPUInfo

        info = GPUInfo(available=False)
        _assert_frozen_assignment(info, "available", True)

    def test_asdict_roundtrip(self):
        from nanochat.report import GPUInfo

        info = GPUInfo(
            available=True,
            count=1,
            names=["V100"],
            memory_gb=[16.0],
            cuda_version="11.8",
        )
        d = dataclasses.asdict(info)
        assert isinstance(d, dict)
        reconstructed = GPUInfo(**d)
        assert reconstructed == info

    def test_fields(self):
        from nanochat.report import GPUInfo

        names = {f.name for f in dataclasses.fields(GPUInfo)}
        assert names == {"available", "count", "names", "memory_gb", "cuda_version"}


class TestSystemInfo:
    """Tests for SystemInfo frozen dataclass (nanochat.report)."""

    def test_is_dataclass(self):
        from nanochat.report import SystemInfo

        assert dataclasses.is_dataclass(SystemInfo)

    def test_construction(self):
        from nanochat.report import SystemInfo

        info = SystemInfo(
            hostname="devbox",
            platform="Linux-6.1",
            python_version="3.11.0",
            torch_version="2.4.0",
            cpu_count=8,
            cpu_count_logical=16,
            memory_gb=64.0,
            user="testuser",
            nanochat_base_dir="/opt/nanochat",
            working_dir="/home/testuser/project",
        )
        assert info.hostname == "devbox"
        assert info.memory_gb == 64.0

    def test_frozen_immutability(self):
        from nanochat.report import SystemInfo

        info = SystemInfo(
            hostname="h",
            platform="p",
            python_version="3.11",
            torch_version="2.4",
            cpu_count=1,
            cpu_count_logical=2,
            memory_gb=8.0,
            user="u",
            nanochat_base_dir="/tmp",
            working_dir="/tmp",
        )
        _assert_frozen_assignment(info, "hostname", "other")

    def test_asdict_roundtrip(self):
        from nanochat.report import SystemInfo

        info = SystemInfo(
            hostname="h",
            platform="p",
            python_version="3.11",
            torch_version="2.4",
            cpu_count=1,
            cpu_count_logical=2,
            memory_gb=8.0,
            user="u",
            nanochat_base_dir="/tmp",
            working_dir="/tmp",
        )
        d = dataclasses.asdict(info)
        reconstructed = SystemInfo(**d)
        assert reconstructed == info

    def test_fields(self):
        from nanochat.report import SystemInfo

        names = {f.name for f in dataclasses.fields(SystemInfo)}
        assert names == {
            "hostname",
            "platform",
            "python_version",
            "torch_version",
            "cpu_count",
            "cpu_count_logical",
            "memory_gb",
            "user",
            "nanochat_base_dir",
            "working_dir",
        }


class TestCostEstimate:
    """Tests for CostEstimate frozen dataclass (nanochat.report)."""

    def test_is_dataclass(self):
        from nanochat.report import CostEstimate

        assert dataclasses.is_dataclass(CostEstimate)

    def test_construction_with_estimate(self):
        from nanochat.report import CostEstimate

        est = CostEstimate(hourly_rate=3.50, gpu_type="A100", estimated_total=28.00)
        assert est.hourly_rate == 3.50
        assert est.estimated_total == 28.00

    def test_construction_without_estimate(self):
        from nanochat.report import CostEstimate

        est = CostEstimate(hourly_rate=3.50, gpu_type="A100", estimated_total=None)
        assert est.estimated_total is None

    def test_frozen_immutability(self):
        from nanochat.report import CostEstimate

        est = CostEstimate(hourly_rate=1.0, gpu_type="V100", estimated_total=10.0)
        _assert_frozen_assignment(est, "hourly_rate", 2.0)

    def test_asdict_roundtrip(self):
        from nanochat.report import CostEstimate

        est = CostEstimate(hourly_rate=2.0, gpu_type="H100", estimated_total=None)
        d = dataclasses.asdict(est)
        reconstructed = CostEstimate(**d)
        assert reconstructed == est

    def test_fields(self):
        from nanochat.report import CostEstimate

        names = {f.name for f in dataclasses.fields(CostEstimate)}
        assert names == {"hourly_rate", "gpu_type", "estimated_total"}


class TestTaskMeta:
    """Tests for TaskMeta frozen dataclass (nanochat.core_eval)."""

    def test_is_dataclass(self):
        from nanochat.core_eval import TaskMeta

        assert dataclasses.is_dataclass(TaskMeta)

    def test_construction(self):
        from nanochat.core_eval import TaskMeta, TaskType

        meta = TaskMeta(
            task_type=TaskType.MULTIPLE_CHOICE,
            num_fewshot=5,
            continuation_delimiter="\n",
        )
        assert meta.task_type == TaskType.MULTIPLE_CHOICE
        assert meta.task_type == "multiple_choice"
        assert meta.num_fewshot == 5

    def test_frozen_immutability(self):
        from nanochat.core_eval import TaskMeta, TaskType

        meta = TaskMeta(
            task_type=TaskType.SCHEMA,
            num_fewshot=0,
            continuation_delimiter=" ",
        )
        _assert_frozen_assignment(meta, "task_type", TaskType.LANGUAGE_MODELING)

    def test_asdict_roundtrip(self):
        from nanochat.core_eval import TaskMeta, TaskType

        meta = TaskMeta(
            task_type=TaskType.LANGUAGE_MODELING,
            num_fewshot=3,
            continuation_delimiter=":",
        )
        d = dataclasses.asdict(meta)
        reconstructed = TaskMeta(**d)
        assert reconstructed == meta

    def test_fields(self):
        from nanochat.core_eval import TaskMeta

        names = {f.name for f in dataclasses.fields(TaskMeta)}
        assert names == {"task_type", "num_fewshot", "continuation_delimiter"}
