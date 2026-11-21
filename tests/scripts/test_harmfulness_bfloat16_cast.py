"""Tests for the bfloat16 dtype cast fix in harmfulness_eval.py.

The fix ensures that when running on CUDA, the model is explicitly cast to
bfloat16 to match the KV cache dtype used by InferenceEngine.  Without this
cast, a float32 model would produce float32 q/k/v tensors that mismatch
the bfloat16 KV cache, causing dtype errors in flash attention.

The InferenceEngine._make_kv_cache method hardcodes:
    dtype = bfloat16 if device.type == "cuda" else float32

So the model must be bfloat16 on CUDA and float32 on CPU for dtype
consistency between model activations and the KV cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch as t
import torch.nn.functional as F

from nanochat.gpt import GPT
from scripts.eval.harmfulness_eval import main
from steering.eval.harmfulness_generation import (
    build_generation_provenance,
    native_checkpoint_metadata_path,
)
from steering.eval.inference_engine import InferenceEngine, build_engine_for_features
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
)
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer, _make_encdec, _make_gpt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNCONDITIONED_PROVENANCE = {
    "conditioning_mode": "unconditioned",
    "eval_name": "harmfulness_eval",
}
_GENERATION_PROVENANCE = build_generation_provenance(
    model={
        "format": "nanochat_checkpoint",
        "backend": "nanochat",
        "architecture": "GPT",
        "model_type": "decoder_only",
        "step": 1,
        "checkpoint_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
    },
    condition_name="unconditional",
    device_type="cpu",
    inference_dtype="float32",
    feature_conditions=None,
    target_conditioning_mode="none",
    max_new_tokens=1,
    temperature=0.0,
    top_k=1,
    batch_size=8,
    seed=0,
    prompt_count=1,
)


def _encode_prompt(tokenizer: MockTokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, prepend=0)
    if not all(isinstance(token_id, int) for token_id in token_ids):
        raise TypeError("single prompt encoding must return a flat list of token ids")
    return cast(list[int], token_ids)


def _all_float_param_dtypes(model: t.nn.Module) -> set[t.dtype]:
    """Return the set of dtypes for all floating-point parameters."""
    return {p.dtype for p in model.parameters() if p.is_floating_point()}


class _AttrMapping(dict[str, object]):
    """Minimal mapping fixture with DictConfig-style attribute reads."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _SyntheticContextProvider:
    """Return one full-prefix feature segment for dtype-alignment generation tests."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def get(self, token_ids):
        token_ids = [int(tok) for tok in token_ids]
        if not token_ids:
            return None
        char_end = token_to_char_offsets(self._tokenizer, token_ids)[-1]
        if char_end <= 0:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


def _context_provider(tokenizer):
    return _SyntheticContextProvider(tokenizer)


# =====================================================================
# 1. KV cache dtype selection
# =====================================================================


class TestKVCacheDtype:
    """InferenceEngine._make_kv_cache must pick the right dtype per device."""

    def test_cpu_kv_cache_is_float32(self, gpt_model: GPT, mock_tokenizer: MockTokenizer) -> None:
        """KV cache on CPU uses float32."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        kv = engine._make_kv_cache(batch_size=1, seq_len=32)
        assert kv.k_cache.dtype == t.float32
        assert kv.v_cache.dtype == t.float32

    def test_cpu_kv_cache_matches_model_param_dtype(
        self,
        gpt_model: GPT,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """On CPU, model params and KV cache share float32 dtype."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        kv = engine._make_kv_cache(batch_size=1, seq_len=32)
        model_dtypes = _all_float_param_dtypes(gpt_model)
        assert t.float32 in model_dtypes, "model should have float32 params on CPU"
        assert kv.k_cache.dtype == t.float32

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_cuda_kv_cache_is_bfloat16(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """KV cache on CUDA uses bfloat16."""
        model = _make_gpt(device="cuda")
        engine = InferenceEngine(model, mock_tokenizer, t.device("cuda"))
        kv = engine._make_kv_cache(batch_size=1, seq_len=32)
        assert kv.k_cache.dtype == t.bfloat16
        assert kv.v_cache.dtype == t.bfloat16

    def test_kv_cache_dtype_determined_by_device_not_model(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """KV cache dtype is hardcoded per device type, independent of model."""
        model = _make_gpt(device="cpu")
        engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))
        kv = engine._make_kv_cache(batch_size=1, seq_len=32)
        assert kv.k_cache.dtype == t.float32, (
            "CPU KV cache must be float32 regardless of model dtype"
        )


# =====================================================================
# 2. bfloat16 cast correctness
# =====================================================================


class TestBfloat16Cast:
    """Verify that .to(dtype=bfloat16) converts model parameters correctly."""

    def test_gpt_all_float_params_become_bfloat16(self) -> None:
        """After .to(bfloat16), every floating-point GPT param is bfloat16."""
        model = _make_gpt(device="cpu")
        # Precondition: model starts as float32
        assert t.float32 in _all_float_param_dtypes(model)

        model = model.to(dtype=t.bfloat16)

        dtypes = _all_float_param_dtypes(model)
        assert dtypes == {t.bfloat16}, f"Expected only bfloat16 params, got {dtypes}"

    def test_encdec_all_float_params_become_bfloat16(self) -> None:
        """After .to(bfloat16), every floating-point EncoderDecoderGPT param is bfloat16."""
        model = _make_encdec(device="cpu")
        assert t.float32 in _all_float_param_dtypes(model)

        model = model.to(dtype=t.bfloat16)

        dtypes = _all_float_param_dtypes(model)
        assert dtypes == {t.bfloat16}, f"Expected only bfloat16 params, got {dtypes}"

    def test_cast_preserves_eval_mode(self) -> None:
        """Model remains in eval mode after dtype cast."""
        model = _make_gpt(device="cpu")
        model.eval()
        assert not model.training

        model = model.to(dtype=t.bfloat16)
        assert not model.training, "eval mode must survive dtype cast"

    def test_cast_preserves_device(self) -> None:
        """Model remains on the same device after dtype cast."""
        model = _make_gpt(device="cpu")
        model = model.to(dtype=t.bfloat16)

        for p in model.parameters():
            assert p.device.type == "cpu", "device must survive dtype cast"

    def test_integer_params_unchanged_by_cast(self) -> None:
        """Integer parameters (e.g., buffer indices) are unaffected by float cast."""
        model = _make_gpt(device="cpu")
        int_buffers_before = {
            name: buf.clone() for name, buf in model.named_buffers() if not buf.is_floating_point()
        }

        model = model.to(dtype=t.bfloat16)

        for name, buf in model.named_buffers():
            if not buf.is_floating_point() and name in int_buffers_before:
                assert t.equal(buf, int_buffers_before[name]), (
                    f"Integer buffer {name} was modified by dtype cast"
                )


# =====================================================================
# 3. Forward pass works after bfloat16 cast
# =====================================================================


class TestForwardAfterCast:
    """Verify forward pass still produces valid output after bfloat16 cast."""

    def test_gpt_forward_after_bfloat16_cast(self) -> None:
        """GPT forward pass produces finite logits after bfloat16 cast."""
        model = _make_gpt(device="cpu")
        model = model.to(dtype=t.bfloat16)

        idx = t.randint(0, 128, (1, 8))
        with t.inference_mode():
            logits = model(idx, targets=None)

        assert logits.shape == (1, 8, 128), f"Unexpected shape: {logits.shape}"
        assert t.isfinite(logits).all(), "Logits contain non-finite values"

    def test_encdec_forward_after_bfloat16_cast(self) -> None:
        """EncoderDecoderGPT forward pass works after bfloat16 cast."""
        model = _make_encdec(device="cpu")
        model = model.to(dtype=t.bfloat16)

        B, T, n_seg, F = 1, 8, 1, 4
        idx = t.randint(0, 128, (B, T))
        enc_fids = t.zeros(B, n_seg, F, dtype=t.int64)
        enc_mags = t.zeros(B, n_seg, F, dtype=t.int64)
        enc_mask = t.ones(B, n_seg, F, dtype=t.bool)
        seg_map = t.zeros(B, T, dtype=t.long)

        with t.inference_mode():
            logits = model(
                idx,
                targets=None,
                enc_feature_ids=enc_fids,
                enc_magnitudes=enc_mags,
                enc_mask=enc_mask,
                seg_map=seg_map,
            )

        assert logits.shape == (B, T, 128), f"Unexpected shape: {logits.shape}"
        assert t.isfinite(logits).all(), "Logits contain non-finite values"

    def test_logits_always_float32_due_to_softcap(self) -> None:
        """Model output logits are always float32 due to explicit .float() softcap.

        This is important context: the logits are float32 regardless of model
        dtype, but the intermediate q/k/v activations in attention follow the
        model's parameter dtype.  The KV cache dtype must match those
        intermediate activations, not the final logits.
        """
        # float32 model → float32 logits
        model_f32 = _make_gpt(device="cpu")
        idx = t.randint(0, 128, (1, 8))
        with t.inference_mode():
            logits_f32 = model_f32(idx, targets=None)
        assert logits_f32.dtype == t.float32

        # bfloat16 model → also float32 logits (due to logits.float() in forward)
        model_bf16 = _make_gpt(device="cpu")
        model_bf16 = model_bf16.to(dtype=t.bfloat16)
        with t.inference_mode():
            logits_bf16 = model_bf16(idx, targets=None)
        assert logits_bf16.dtype == t.float32, (
            "Even bfloat16 model produces float32 logits due to softcap"
        )


# =====================================================================
# 4. Dtype mismatch detection — proving the bug exists without the fix
# =====================================================================


class TestDtypeMismatch:
    """Demonstrate that mismatched model/KV-cache dtypes cause problems.

    The InferenceEngine creates bfloat16 KV caches on CUDA.  If the model
    is float32, activations written to the cache undergo silent precision
    loss, and attention may receive mixed-dtype inputs.
    """

    def test_float32_model_produces_float32_logits(self) -> None:
        """A float32 model produces float32 logits."""
        model = _make_gpt(device="cpu")
        assert t.float32 in _all_float_param_dtypes(model)

        idx = t.randint(0, 128, (1, 8))
        with t.inference_mode():
            logits = model(idx, targets=None)
        assert logits.dtype == t.float32

    def test_kv_cache_insertion_silently_truncates_float32_to_bfloat16(self) -> None:
        """Writing float32 values into a bfloat16 tensor silently truncates.

        This is the silent data corruption that happens when a float32 model
        writes activations into a bfloat16 KV cache — values lose precision.
        """
        # Simulate a value that has precision beyond bfloat16's mantissa
        float32_val = t.tensor([1.0009765625], dtype=t.float32)  # exact in float32
        bfloat16_cache = t.zeros(1, dtype=t.bfloat16)
        bfloat16_cache[0] = float32_val[0]  # silent truncation

        # bfloat16 can't represent this exactly
        roundtripped = bfloat16_cache[0].float()
        assert roundtripped != float32_val[0], (
            "Expected precision loss when storing float32 in bfloat16"
        )

    def test_sdpa_rejects_mixed_dtype_inputs(self) -> None:
        """PyTorch SDPA raises RuntimeError on mixed-dtype q/k/v.

        This is the exact error that occurs when a bfloat16 model produces
        bfloat16 q tensors but the KV cache supplies float32 k/v (or vice
        versa).  On CUDA with flash attention, this is also a hard error.
        """
        q = t.randn(1, 2, 4, 8, dtype=t.bfloat16)
        k = t.randn(1, 2, 4, 8, dtype=t.float32)
        v = t.randn(1, 2, 4, 8, dtype=t.float32)

        with pytest.raises(RuntimeError, match="same dtype"):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def test_bfloat16_model_on_cpu_generates_successfully(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """bfloat16 model on CPU + float32 KV cache works after SDPA dtype fix.

        The SDPA fallback path casts KV cache slices to match the query dtype,
        so mixed bf16 model / f32 cache no longer raises.
        """
        model = _make_gpt(device="cpu")
        model = model.to(dtype=t.bfloat16)
        engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))

        tokens = engine.generate("hello", max_new_tokens=4, temperature=0.0)
        assert len(tokens) > 0


# =====================================================================
# 5. Generation works with correct dtype alignment
# =====================================================================


class TestGenerationDtypeAlignment:
    """Verify that generation works when model and KV cache dtypes match."""

    def test_cpu_float32_generation_succeeds(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CPU: float32 model + float32 KV cache → generation works."""
        model = _make_gpt(device="cpu")
        engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))
        result = engine.generate(
            "hello",
            max_new_tokens=4,
            temperature=0.0,
        )
        assert isinstance(result, str)

    def test_cpu_float32_batch_generation_succeeds(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CPU: float32 model + float32 KV cache → batch generation works."""
        model = _make_gpt(device="cpu")
        engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))
        results = engine.generate_batch(
            ["hello", "world"],
            max_new_tokens=4,
            temperature=0.0,
        )
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_cpu_encdec_generation_succeeds(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CPU: float32 encoder-decoder + float32 KV cache → generation works."""
        model = _make_encdec(device="cpu")
        engine = build_engine_for_features(
            model,
            mock_tokenizer,
            t.device("cpu"),
            features=None,
            context_provider=_context_provider(mock_tokenizer),
        )
        result = engine.generate(
            "hello",
            max_new_tokens=4,
            temperature=0.0,
        )
        assert isinstance(result, str)

    def test_cpu_encdec_conditioned_generation_succeeds(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CPU: encoder-decoder context-conditioned generation works via wrapper."""
        model = _make_encdec(device="cpu")
        engine = build_engine_for_features(
            model,
            mock_tokenizer,
            t.device("cpu"),
            features={},
            context_provider=_context_provider(mock_tokenizer),
        )
        result = engine.generate(
            "hello",
            max_new_tokens=4,
            temperature=0.0,
        )
        assert isinstance(result, str)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_cuda_bfloat16_generation_succeeds(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CUDA: bfloat16 model + bfloat16 KV cache → generation works."""
        model = _make_gpt(device="cuda")
        model = model.to(dtype=t.bfloat16)
        engine = InferenceEngine(model, mock_tokenizer, t.device("cuda"))
        result = engine.generate(
            "hello",
            max_new_tokens=4,
            temperature=0.0,
        )
        assert isinstance(result, str)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_cuda_float32_model_without_cast_would_mismatch_kv_cache(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CUDA: float32 model with bfloat16 KV cache → dtype mismatch.

        This test demonstrates the bug that the bfloat16 cast fix addresses.
        Without the fix, the model is float32 but the KV cache is bfloat16.
        """
        model = _make_gpt(device="cuda")
        # Force model to float32 (simulating no bfloat16 cast)
        model = model.to(dtype=t.float32)

        engine = InferenceEngine(model, mock_tokenizer, t.device("cuda"))
        kv = engine._make_kv_cache(batch_size=1, seq_len=32)

        model_dtypes = _all_float_param_dtypes(model)
        assert t.float32 in model_dtypes, "model should be float32"
        assert kv.k_cache.dtype == t.bfloat16, "KV cache should be bfloat16"

        # The dtypes DON'T match — this is the bug.
        assert t.float32 in model_dtypes and kv.k_cache.dtype == t.bfloat16, (
            "Without the fix, model dtype (float32) != KV cache dtype (bfloat16)"
        )

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_cuda_bfloat16_model_matches_kv_cache(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CUDA: bfloat16 model + bfloat16 KV cache → dtypes match (the fix)."""
        model = _make_gpt(device="cuda")
        model = model.to(dtype=t.bfloat16)

        engine = InferenceEngine(model, mock_tokenizer, t.device("cuda"))
        kv = engine._make_kv_cache(batch_size=1, seq_len=32)

        model_dtypes = _all_float_param_dtypes(model)
        assert model_dtypes == {t.bfloat16}
        assert kv.k_cache.dtype == t.bfloat16
        assert kv.v_cache.dtype == t.bfloat16


# =====================================================================
# 6. Main function conditional bfloat16 cast logic
# =====================================================================


class TestMainBfloat16CastLogic:
    """Test the conditional bfloat16 cast in scripts/harmfulness_eval.main().

    The fix adds:
        if device.type == "cuda":
            model = model.to(dtype=t.bfloat16)

    These tests mock load_model and verify the cast is applied correctly
    depending on the device type.
    """

    def _make_mock_model(self) -> MagicMock:
        """Create a mock model that tracks .to() calls."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        return mock_model

    @patch(
        "scripts.eval.harmfulness_eval.current_harmfulness_payload",
        return_value=({"current": True}, "current artifact"),
    )
    @patch("scripts.eval.harmfulness_eval._get_responses")
    @patch("scripts.eval.harmfulness_eval.JudgeClient")
    @patch("scripts.eval.harmfulness_eval.InferenceEngine")
    @patch("scripts.eval.harmfulness_eval._load_model_for_harmfulness")
    @patch("scripts.eval.harmfulness_eval.autodetect_device_type")
    @patch("scripts.eval.harmfulness_eval.OmegaConf")
    @patch("scripts.eval.harmfulness_eval.compose")
    @patch("scripts.eval.harmfulness_eval.initialize_config_dir")
    def test_cuda_device_triggers_bfloat16_cast(
        self,
        mock_init_cfg,
        mock_compose,
        mock_omegaconf,
        mock_autodetect,
        mock_load_model,
        mock_engine_cls,
        mock_judge_cls,
        mock_get_responses,
        mock_current_payload,
        tmp_path: Path,
    ) -> None:
        """On CUDA, model.to(dtype=bfloat16) is called before creating engine."""
        mock_cfg = _AttrMapping(
            source="safety",
            model_tag="test",
            step=None,
            judge_only=False,
            responses_file=None,
            feature_configs=None,
            self_cond_segment_size=None,
            batch_size=8,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            judge_model="gpt-4",
            cache_path="",
            requests_per_minute=100,
            encdec_enabled=False,
            conditioning=None,
            next_segment_conditioning_offset=False,
            seed=0,
            num_harmful=0,
            num_benign=0,
        )
        mock_compose.return_value = mock_cfg
        mock_init_cfg.return_value.__enter__ = MagicMock()
        mock_init_cfg.return_value.__exit__ = MagicMock()
        mock_omegaconf.to_yaml.return_value = "source: safety"
        mock_autodetect.return_value = "cuda"

        mock_model = self._make_mock_model()
        mock_load_model.return_value = (
            mock_model,
            MagicMock(),
            {"checkpoint_dir": "/checkpoints/decoder-only"},
            None,
            _GENERATION_PROVENANCE["model"],
        )

        # Mark the isolated output as a current artifact so
        # the test remains about dtype routing and cannot write into source.
        output_path = tmp_path / "cuda-result.yaml"
        mock_get_responses.return_value = (
            "unconditional",
            str(output_path),
            [],
            _UNCONDITIONED_PROVENANCE,
            _GENERATION_PROVENANCE,
        )

        main()

        # Verify model.to(dtype=bfloat16) was called
        mock_model.to.assert_called_once_with(dtype=t.bfloat16)
        runtime = mock_get_responses.call_args.kwargs["generation_provenance"]
        assert runtime["device_type"] == "cuda"
        assert runtime["inference_dtype"] == "bfloat16"
        mock_current_payload.assert_called_once_with(
            str(output_path),
            model="gpt-4",
            pairs=[],
            expected_conditioning=_UNCONDITIONED_PROVENANCE,
            expected_generation=_GENERATION_PROVENANCE,
        )

    @patch(
        "scripts.eval.harmfulness_eval.current_harmfulness_payload",
        return_value=({"current": True}, "current artifact"),
    )
    @patch("scripts.eval.harmfulness_eval._get_responses")
    @patch("scripts.eval.harmfulness_eval.JudgeClient")
    @patch("scripts.eval.harmfulness_eval.InferenceEngine")
    @patch("scripts.eval.harmfulness_eval._load_model_for_harmfulness")
    @patch("scripts.eval.harmfulness_eval.autodetect_device_type")
    @patch("scripts.eval.harmfulness_eval.OmegaConf")
    @patch("scripts.eval.harmfulness_eval.compose")
    @patch("scripts.eval.harmfulness_eval.initialize_config_dir")
    def test_cpu_device_skips_bfloat16_cast(
        self,
        mock_init_cfg,
        mock_compose,
        mock_omegaconf,
        mock_autodetect,
        mock_load_model,
        mock_engine_cls,
        mock_judge_cls,
        mock_get_responses,
        mock_current_payload,
        tmp_path: Path,
    ) -> None:
        """On CPU, model.to(dtype=bfloat16) is NOT called."""
        mock_cfg = _AttrMapping(
            source="safety",
            model_tag="test",
            step=None,
            judge_only=False,
            responses_file=None,
            feature_configs=None,
            self_cond_segment_size=None,
            batch_size=8,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            judge_model="gpt-4",
            cache_path="",
            requests_per_minute=100,
            encdec_enabled=False,
            conditioning=None,
            next_segment_conditioning_offset=False,
            seed=0,
            num_harmful=0,
            num_benign=0,
        )
        mock_compose.return_value = mock_cfg
        mock_init_cfg.return_value.__enter__ = MagicMock()
        mock_init_cfg.return_value.__exit__ = MagicMock()
        mock_omegaconf.to_yaml.return_value = "source: safety"
        mock_autodetect.return_value = "cpu"

        mock_model = self._make_mock_model()
        mock_load_model.return_value = (
            mock_model,
            MagicMock(),
            {"checkpoint_dir": "/checkpoints/decoder-only"},
            None,
            _GENERATION_PROVENANCE["model"],
        )

        # Mark an isolated output as current.
        output_path = tmp_path / "cpu-result.yaml"
        mock_get_responses.return_value = (
            "unconditional",
            str(output_path),
            [],
            _UNCONDITIONED_PROVENANCE,
            _GENERATION_PROVENANCE,
        )

        main()

        # model.to should NOT have been called with bfloat16
        mock_model.to.assert_not_called()
        runtime = mock_get_responses.call_args.kwargs["generation_provenance"]
        assert runtime["device_type"] == "cpu"
        assert runtime["inference_dtype"] == "float32"
        mock_current_payload.assert_called_once_with(
            str(output_path),
            model="gpt-4",
            pairs=[],
            expected_conditioning=_UNCONDITIONED_PROVENANCE,
            expected_generation=_GENERATION_PROVENANCE,
        )

    @patch(
        "scripts.eval.harmfulness_eval.current_harmfulness_payload",
        return_value=({"current": True}, "current artifact"),
    )
    @patch("scripts.eval.harmfulness_eval._get_responses")
    @patch("scripts.eval.harmfulness_eval.JudgeClient")
    @patch("scripts.eval.harmfulness_eval.InferenceEngine")
    @patch("scripts.eval.harmfulness_eval._load_model_for_harmfulness")
    @patch("scripts.eval.harmfulness_eval.autodetect_device_type")
    @patch("scripts.eval.harmfulness_eval.OmegaConf")
    @patch("scripts.eval.harmfulness_eval.compose")
    @patch("scripts.eval.harmfulness_eval.initialize_config_dir")
    def test_judge_only_mode_skips_model_loading_and_cast(
        self,
        mock_init_cfg,
        mock_compose,
        mock_omegaconf,
        mock_autodetect,
        mock_load_model,
        mock_engine_cls,
        mock_judge_cls,
        mock_get_responses,
        mock_current_payload,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In judge_only mode, no model is loaded so no cast occurs."""
        native_checkpoint_metadata_path(tmp_path, 1).write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "scripts.eval.harmfulness_eval._resolve_native_checkpoint_for_harmfulness",
            lambda _cfg: (str(tmp_path), 1),
        )

        mock_cfg = _AttrMapping(
            source="safety",
            model_tag="test",
            step=None,
            judge_only=True,
            responses_file=None,
            feature_configs=None,
            self_cond_segment_size=None,
            batch_size=8,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            judge_model="gpt-4",
            cache_path="",
            requests_per_minute=100,
            encdec_enabled=False,
            seed=0,
        )
        mock_compose.return_value = mock_cfg
        mock_init_cfg.return_value.__enter__ = MagicMock()
        mock_init_cfg.return_value.__exit__ = MagicMock()
        mock_omegaconf.to_yaml.return_value = "source: safety"

        # Stub response generation so the unconditioned pass is inert
        # (judge_only mode loads pre-generated responses, not the model).
        output_path = tmp_path / "judge-only-result.yaml"
        mock_get_responses.return_value = (
            "unconditional",
            str(output_path),
            [],
            _UNCONDITIONED_PROVENANCE,
            _GENERATION_PROVENANCE,
        )

        main()

        # load_model should not have been called at all
        mock_load_model.assert_not_called()
        mock_current_payload.assert_called_once_with(
            str(output_path),
            model="gpt-4",
            pairs=[],
            expected_conditioning=_UNCONDITIONED_PROVENANCE,
            expected_generation=_GENERATION_PROVENANCE,
        )

    def test_cast_condition_logic_cuda(self) -> None:
        """Directly test: t.device('cuda') → device.type == 'cuda' is True."""
        device = t.device("cuda")
        assert device.type == "cuda"

    def test_cast_condition_logic_cpu(self) -> None:
        """Directly test: t.device('cpu') → device.type != 'cuda'."""
        device = t.device("cpu")
        assert device.type != "cuda"

    def test_cast_condition_logic_mps(self) -> None:
        """Directly test: t.device('mps') → device.type != 'cuda'."""
        device = t.device("mps")
        assert device.type != "cuda"

    def test_cast_condition_logic_cuda_with_index(self) -> None:
        """t.device('cuda:0') also has device.type == 'cuda'."""
        device = t.device("cuda:0")
        assert device.type == "cuda"


# =====================================================================
# 7. Rotary embedding dtype interaction
# =====================================================================


class TestRotaryEmbeddingDtype:
    """Rotary embeddings are always bfloat16 — verify they interact correctly."""

    def test_rotary_embeddings_are_bfloat16(self) -> None:
        """GPT rotary embeddings (cos, sin buffers) are always bfloat16."""
        model = _make_gpt(device="cpu")
        assert model.cos.dtype == t.bfloat16
        assert model.sin.dtype == t.bfloat16

    def test_rotary_bfloat16_survives_model_cast(self) -> None:
        """Casting model to bfloat16 keeps rotary embeddings in bfloat16."""
        model = _make_gpt(device="cpu")
        model = model.to(dtype=t.bfloat16)
        assert model.cos.dtype == t.bfloat16
        assert model.sin.dtype == t.bfloat16

    def test_float32_model_has_bfloat16_rotary(self) -> None:
        """Even a float32 model has bfloat16 rotary embeddings.

        This means the forward pass already does mixed-dtype ops (bfloat16
        rotary * float32 activations) on CPU via auto-promotion.  The real
        problem is the KV cache dtype mismatch, not the rotary ops.
        """
        model = _make_gpt(device="cpu")
        param_dtypes = _all_float_param_dtypes(model)
        assert t.float32 in param_dtypes, "params should be float32"
        assert model.cos.dtype == t.bfloat16, "rotary should be bfloat16"


# =====================================================================
# 8. EncoderDecoderGPT-specific tests
# =====================================================================


class TestEncoderDecoderBfloat16:
    """Encoder-decoder models have extra components that must all be cast."""

    def test_encoder_params_cast_to_bfloat16(self) -> None:
        """Encoder sub-module params become bfloat16 after whole-model cast."""
        model = _make_encdec(device="cpu")
        model = model.to(dtype=t.bfloat16)

        encoder_dtypes = {p.dtype for p in model.encoder.parameters() if p.is_floating_point()}
        assert encoder_dtypes == {t.bfloat16}, (
            f"Encoder params should all be bfloat16, got {encoder_dtypes}"
        )

    def test_transformer_params_cast_to_bfloat16(self) -> None:
        """Decoder transformer sub-module params become bfloat16 after cast."""
        model = _make_encdec(device="cpu")
        model = model.to(dtype=t.bfloat16)

        transformer_dtypes = {
            p.dtype for p in model.transformer.parameters() if p.is_floating_point()
        }
        assert transformer_dtypes == {t.bfloat16}, (
            f"Transformer params should all be bfloat16, got {transformer_dtypes}"
        )

    def test_lm_head_params_cast_to_bfloat16(self) -> None:
        """lm_head linear layer becomes bfloat16 after whole-model cast."""
        model = _make_encdec(device="cpu")
        model = model.to(dtype=t.bfloat16)

        lm_head_dtypes = {p.dtype for p in model.lm_head.parameters() if p.is_floating_point()}
        assert lm_head_dtypes == {t.bfloat16}, (
            f"lm_head params should be bfloat16, got {lm_head_dtypes}"
        )

    def test_cross_attention_params_cast_to_bfloat16(self) -> None:
        """Cross-attention layers become bfloat16 after whole-model cast."""
        model = _make_encdec(device="cpu")
        model = model.to(dtype=t.bfloat16)

        xattn_params = [
            p
            for name, p in model.named_parameters()
            if "cross_attn" in name and p.is_floating_point()
        ]
        if xattn_params:  # only if cross-attention layers exist
            xattn_dtypes = {p.dtype for p in xattn_params}
            assert xattn_dtypes == {t.bfloat16}, (
                f"Cross-attn params should be bfloat16, got {xattn_dtypes}"
            )

    def test_encdec_generation_with_conditions_on_cpu(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Encoder-decoder context-conditioned generation works via wrapper on CPU."""
        model = _make_encdec(device="cpu")
        engine = build_engine_for_features(
            model,
            mock_tokenizer,
            t.device("cpu"),
            features={},
            context_provider=_context_provider(mock_tokenizer),
        )

        result = engine.generate(
            "test prompt",
            max_new_tokens=4,
            temperature=0.0,
        )
        assert isinstance(result, str)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_encdec_cuda_bfloat16_generation(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """CUDA: bfloat16 encoder-decoder generation works via wrapper."""
        model = _make_encdec(device="cuda")
        model = model.to(dtype=t.bfloat16)
        engine = build_engine_for_features(
            model,
            mock_tokenizer,
            t.device("cuda"),
            features=None,
            context_provider=_context_provider(mock_tokenizer),
        )

        result = engine.generate(
            "test",
            max_new_tokens=4,
            temperature=0.0,
        )
        assert isinstance(result, str)


# =====================================================================
# 9. Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases for the bfloat16 cast behaviour."""

    def test_double_cast_is_idempotent(self) -> None:
        """Casting to bfloat16 twice doesn't change the result."""
        model = _make_gpt(device="cpu")
        model = model.to(dtype=t.bfloat16)
        params_after_first = {name: p.clone() for name, p in model.named_parameters()}

        model = model.to(dtype=t.bfloat16)

        for name, p in model.named_parameters():
            assert t.equal(p, params_after_first[name]), f"Double cast changed param {name}"

    def test_greedy_generation_is_deterministic(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Greedy generation on CPU (float32) produces same output twice."""
        model = _make_gpt(device="cpu")
        engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))
        r1 = engine.generate("test", max_new_tokens=4, temperature=0.0)
        r2 = engine.generate("test", max_new_tokens=4, temperature=0.0)
        assert r1 == r2, "Greedy generation should be deterministic"

    def test_cast_does_not_affect_model_config(self) -> None:
        """Model config attributes are unchanged after dtype cast."""
        model = _make_gpt(device="cpu")
        cfg_before = model.config

        model = model.to(dtype=t.bfloat16)

        assert model.config.n_layer == cfg_before.n_layer
        assert model.config.n_head == cfg_before.n_head
        assert model.config.n_embd == cfg_before.n_embd
        assert model.config.vocab_size == cfg_before.vocab_size
        assert model.config.sequence_len == cfg_before.sequence_len

    def test_stream_generation_works_on_cpu(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Streaming generation works with float32 model on CPU."""
        model = _make_gpt(device="cpu")
        engine = InferenceEngine(model, mock_tokenizer, t.device("cpu"))
        token_ids = _encode_prompt(mock_tokenizer, "hello")
        tokens = list(
            engine.generate_stream(
                token_ids,
                max_new_tokens=4,
                temperature=0.0,
            )
        )
        assert len(tokens) <= 4
        assert all(isinstance(tok, int) for tok in tokens)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_stream_generation_works_after_bfloat16_cast_on_cuda(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Streaming generation works with bfloat16 model on CUDA."""
        model = _make_gpt(device="cuda")
        model = model.to(dtype=t.bfloat16)
        engine = InferenceEngine(model, mock_tokenizer, t.device("cuda"))
        token_ids = _encode_prompt(mock_tokenizer, "hello")
        tokens = list(
            engine.generate_stream(
                token_ids,
                max_new_tokens=4,
                temperature=0.0,
            )
        )
        assert len(tokens) <= 4
        assert all(isinstance(tok, int) for tok in tokens)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_greedy_generation_is_deterministic_after_cast_on_cuda(
        self,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Greedy generation on CUDA (bfloat16) produces same output twice."""
        model = _make_gpt(device="cuda")
        model = model.to(dtype=t.bfloat16)
        engine = InferenceEngine(model, mock_tokenizer, t.device("cuda"))
        r1 = engine.generate("test", max_new_tokens=4, temperature=0.0)
        r2 = engine.generate("test", max_new_tokens=4, temperature=0.0)
        assert r1 == r2, "Greedy generation should be deterministic"
