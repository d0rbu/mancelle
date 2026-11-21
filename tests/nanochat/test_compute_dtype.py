"""Tests for COMPUTE_DTYPE auto-detection and env var override (nanochat/common.py)."""

import os
from unittest.mock import patch

import torch
import pytest


class TestDetectComputeDtype:
    """Test _detect_compute_dtype logic from nanochat/common.py:235-248."""

    def test_env_override_bfloat16(self):
        """NANOCHAT_DTYPE=bfloat16 forces bfloat16 regardless of hardware."""
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "bfloat16"}):
            from nanochat.common import _detect_compute_dtype
            dtype, reason = _detect_compute_dtype()
            assert dtype is torch.bfloat16
            assert "NANOCHAT_DTYPE" in reason

    def test_env_override_float32(self):
        """NANOCHAT_DTYPE=float32 forces float32."""
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "float32"}):
            from nanochat.common import _detect_compute_dtype
            dtype, reason = _detect_compute_dtype()
            assert dtype is torch.float32

    def test_env_override_float16(self):
        """NANOCHAT_DTYPE=float16 forces float16."""
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "float16"}):
            from nanochat.common import _detect_compute_dtype
            dtype, reason = _detect_compute_dtype()
            assert dtype is torch.float16

    def test_no_cuda_returns_float32(self):
        """Without CUDA, auto-detection returns float32."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOCHAT_DTYPE", None)
            with patch("torch.cuda.is_available", return_value=False):
                from nanochat.common import _detect_compute_dtype
                dtype, reason = _detect_compute_dtype()
                assert dtype is torch.float32
                assert "no CUDA" in reason

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_auto_detection_returns_valid_dtype(self):
        """On CUDA, auto-detection returns bfloat16 (SM80+) or float32 (pre-Ampere)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOCHAT_DTYPE", None)
            from nanochat.common import _detect_compute_dtype
            dtype, reason = _detect_compute_dtype()
            assert dtype in (torch.bfloat16, torch.float32)
            assert "auto-detected" in reason

    def test_cuda_sm80_returns_bfloat16(self):
        """SM 8.0+ (Ampere) should return bfloat16."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOCHAT_DTYPE", None)
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                    from nanochat.common import _detect_compute_dtype
                    dtype, reason = _detect_compute_dtype()
                    assert dtype is torch.bfloat16

    def test_cuda_sm75_returns_float32(self):
        """SM 7.5 (T4) should return float32 (no bf16 tensor cores)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOCHAT_DTYPE", None)
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                    from nanochat.common import _detect_compute_dtype
                    dtype, reason = _detect_compute_dtype()
                    assert dtype is torch.float32
                    assert "pre-Ampere" in reason

    def test_dtype_map_completeness(self):
        """_DTYPE_MAP covers all three supported string names."""
        from nanochat.common import _DTYPE_MAP
        assert set(_DTYPE_MAP.keys()) == {"bfloat16", "float16", "float32"}
        assert _DTYPE_MAP["bfloat16"] is torch.bfloat16
        assert _DTYPE_MAP["float16"] is torch.float16
        assert _DTYPE_MAP["float32"] is torch.float32
