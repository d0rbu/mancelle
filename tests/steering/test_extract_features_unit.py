"""Unit tests for steering/extract_features.py — slice 01 Phase 1 audit.

Pins fixes for plan items C-3 (hook name inference), C-7 (CSR invariant
exceptions), C-10 (independent empty tensors from extract_sparse_gpu),
C-12 (find_optimal_batch_sizes zero-candidate handling), C-6 (ActivationCapture
detach).
"""

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering import extract_features as ef
from steering.extract_features import (
    ActivationCapture,
    _assert_context_window_within_sae_max,
    _infer_activation_type_from_hook_name,
    _validate_csr_invariants,
    extract_sparse_gpu,
    find_optimal_batch_sizes,
)
from steering.refined_types import NonBlankStr, PositiveInt
from steering.sae_resolver import ActivationType

# ---------------------------------------------------------------------------
# C-3: _infer_activation_type_from_hook_name raises on unknown
# ---------------------------------------------------------------------------


class TestInferActivationTypeStrict:
    def test_residual_hook_post_recognized(self):
        assert (
            _infer_activation_type_from_hook_name("blocks.8.hook_resid_post")
            == ActivationType.residual
        )

    def test_residual_hook_pre_recognized(self):
        assert (
            _infer_activation_type_from_hook_name("blocks.8.hook_resid_pre")
            == ActivationType.residual
        )

    def test_residual_hook_mid_recognized(self):
        assert (
            _infer_activation_type_from_hook_name("blocks.8.hook_resid_mid")
            == ActivationType.residual
        )

    def test_mlp_hook_recognized(self):
        assert (
            _infer_activation_type_from_hook_name("blocks.8.hook_mlp_out")
            == ActivationType.mlp
        )

    def test_attention_hook_z_recognized(self):
        assert (
            _infer_activation_type_from_hook_name("blocks.8.attn.hook_z")
            == ActivationType.attention
        )

    def test_none_defaults_to_residual_for_legacy(self):
        """Per docstring: legacy GPT-2 path with hook_name=None → residual."""
        assert _infer_activation_type_from_hook_name(None) == ActivationType.residual

    def test_unknown_hook_raises(self):
        """An unrecognized hook name MUST raise rather than silently default to residual.

        Pins the invariant that the function never silently mis-classifies
        an unrecognized hook as residual. Uses ``hook_pattern`` (a real
        TransformerLens attention hook that is intentionally NOT in the
        recognized set) as the unknown example. Note: ``hook_attn_out`` is
        now a *recognized* hook (gemma-scope-2 post-projection attention,
        returns attention) and is covered by
        ``test_ensure_chat_template.py::TestHookAttnOutRouting``.
        """
        with pytest.raises(ValueError, match="hook"):
            _infer_activation_type_from_hook_name("blocks.8.attn.hook_pattern")

    def test_garbage_hook_raises(self):
        with pytest.raises(ValueError, match="hook"):
            _infer_activation_type_from_hook_name("not_a_hook_at_all")


# ---------------------------------------------------------------------------
# C-10: extract_sparse_gpu returns independent empty tensors
# ---------------------------------------------------------------------------


class TestExtractSparseEmptyTensorIndependence:
    def test_all_zero_input_returns_independent_empties(self):
        """All-zero pooled batch must return per-segment INDEPENDENT empty tensors."""
        n_segs = 4
        pooled = t.zeros(n_segs, 16, dtype=t.float32)
        all_ids, all_mags = extract_sparse_gpu(pooled)
        assert len(all_ids) == n_segs
        assert len(all_mags) == n_segs
        # Verify each tensor is independent: storage pointers differ.
        for i in range(n_segs):
            for j in range(i + 1, n_segs):
                assert all_ids[i].data_ptr() != all_ids[j].data_ptr() or (
                    all_ids[i].numel() == 0 and all_ids[j].numel() == 0
                ), "all_ids[i] and all_ids[j] must not alias"
                # For empty tensors, data_ptr is implementation-defined; the
                # critical check is that resize on one does not affect the other.
        # Mutate one via resize_; others must remain empty + valid.
        all_ids[0].resize_(3)
        all_ids[0][:] = 99
        for j in range(1, n_segs):
            assert all_ids[j].numel() == 0


# ---------------------------------------------------------------------------
# C-6: ActivationCapture stores detached tensor
# ---------------------------------------------------------------------------


class TestActivationCaptureDetached:
    def test_capture_detaches_from_grad_graph(self):
        """ActivationCapture must store a tensor detached from the autograd graph."""
        capture = ActivationCapture()
        x = t.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2  # has grad_fn
        capture.hook_fn(None, (x,), y)
        assert capture.activation is not None
        # The stored tensor must be detached (no grad_fn) so that downstream
        # storage reuse / pinning never mutates the autograd graph.
        assert capture.activation.requires_grad is False

    def test_capture_clones_so_storage_is_independent(self):
        """Two captures must yield independent storage."""
        capture = ActivationCapture()
        x = t.tensor([1.0, 2.0, 3.0])
        capture.hook_fn(None, None, x)
        assert capture.activation is not None
        first = capture.activation
        # Mutate the source: capture must not see it (independent storage).
        x[0] = 99.0
        assert first[0].item() == 1.0


# ---------------------------------------------------------------------------
# C-7: process_shard CSR invariants raise typed errors (-O safe)
# ---------------------------------------------------------------------------


class TestProcessShardCSRInvariantsAreRuntimeErrors:
    """Slice 1 doesn't synthesize a process_shard call here (heavy fixtures);
    we instead verify that the CSR invariant helper exists as a typed
    function (not assert) so it is robust under `python -O`."""

    def test_csr_invariant_helper_exists(self):
        # Empty / valid case: no error
        _validate_csr_invariants(
            offsets=t.tensor([0, 2, 5], dtype=t.int64),
            feature_ids=t.zeros(5, dtype=t.int32),
            feature_mags=t.ones(5, dtype=t.float32),
            n_segments=2,
            doc_ids=t.zeros(2, dtype=t.int32),
            segment_indices=t.zeros(2, dtype=t.int32),
            segment_positions=t.zeros(2, dtype=t.int32),
            segment_char_starts=t.zeros(2, dtype=t.int32),
            segment_char_ends=t.zeros(2, dtype=t.int32),
        )

    def test_csr_invariant_offsets_zero(self):
        with pytest.raises(RuntimeError, match="offsets"):
            _validate_csr_invariants(
                offsets=t.tensor([1, 5], dtype=t.int64),  # offsets[0] != 0
                feature_ids=t.zeros(5, dtype=t.int32),
                feature_mags=t.ones(5, dtype=t.float32),
                n_segments=1,
                doc_ids=t.zeros(1, dtype=t.int32),
                segment_indices=t.zeros(1, dtype=t.int32),
                segment_positions=t.zeros(1, dtype=t.int32),
                segment_char_starts=t.zeros(1, dtype=t.int32),
                segment_char_ends=t.zeros(1, dtype=t.int32),
            )

    def test_csr_invariant_offsets_last(self):
        with pytest.raises(RuntimeError, match="offsets"):
            _validate_csr_invariants(
                offsets=t.tensor([0, 4], dtype=t.int64),  # offsets[-1] != len(feature_ids)
                feature_ids=t.zeros(5, dtype=t.int32),
                feature_mags=t.ones(5, dtype=t.float32),
                n_segments=1,
                doc_ids=t.zeros(1, dtype=t.int32),
                segment_indices=t.zeros(1, dtype=t.int32),
                segment_positions=t.zeros(1, dtype=t.int32),
                segment_char_starts=t.zeros(1, dtype=t.int32),
                segment_char_ends=t.zeros(1, dtype=t.int32),
            )

    def test_csr_invariant_segment_array_lengths(self):
        with pytest.raises(RuntimeError, match="doc_ids|segment"):
            _validate_csr_invariants(
                offsets=t.tensor([0, 2], dtype=t.int64),
                feature_ids=t.zeros(2, dtype=t.int32),
                feature_mags=t.ones(2, dtype=t.float32),
                n_segments=1,
                doc_ids=t.zeros(2, dtype=t.int32),  # length 2 instead of 1
                segment_indices=t.zeros(1, dtype=t.int32),
                segment_positions=t.zeros(1, dtype=t.int32),
                segment_char_starts=t.zeros(1, dtype=t.int32),
                segment_char_ends=t.zeros(1, dtype=t.int32),
            )


# ---------------------------------------------------------------------------
# C-12: find_optimal_batch_sizes input validation
# ---------------------------------------------------------------------------


class TestFindOptimalBatchSizesInputValidation:
    """find_optimal_batch_sizes(max_gpu_batch_size=0) must raise a clean error.

    The function accepts model/SAE objects whose `.parameters()` is needed for
    device detection; we pass None and rely on the input-validation early
    return to fire BEFORE the device-detection step.
    """

    def test_max_gpu_zero_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                max_gpu_batch_size=0,
                max_cpu_batch_size=4,
            )

    def test_max_cpu_zero_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                max_gpu_batch_size=4,
                max_cpu_batch_size=0,
            )

    def test_negative_min_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                max_gpu_batch_size=4,
                max_cpu_batch_size=4,
                min_gpu_batch_size=-1,
            )

    def test_min_gpu_greater_than_max_raises_before_device_detection(self):
        with pytest.raises(ValueError, match="min_gpu_batch_size .*<= max_gpu_batch_size"):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                min_gpu_batch_size=64,
                max_gpu_batch_size=4,
                min_cpu_batch_size=1,
                max_cpu_batch_size=4,
            )

    def test_min_cpu_greater_than_max_raises_before_device_detection(self):
        with pytest.raises(ValueError, match="min_cpu_batch_size .*<= max_cpu_batch_size"):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                min_gpu_batch_size=1,
                max_gpu_batch_size=4,
                min_cpu_batch_size=64,
                max_cpu_batch_size=4,
            )

    def test_no_gpu_power_of_two_candidate_raises_before_device_detection(self):
        with pytest.raises(ValueError, match="No GPU batch-size candidates"):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                min_gpu_batch_size=3,
                max_gpu_batch_size=3,
                min_cpu_batch_size=1,
                max_cpu_batch_size=4,
            )

    def test_no_cpu_power_of_two_candidate_raises_before_device_detection(self):
        with pytest.raises(ValueError, match="No CPU batch-size candidates"):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                min_gpu_batch_size=1,
                max_gpu_batch_size=4,
                min_cpu_batch_size=3,
                max_cpu_batch_size=3,
            )

    def test_no_valid_gpu_cpu_pair_raises_before_device_detection(self):
        with pytest.raises(ValueError, match="gpu_batch_size <= cpu_batch_size"):
            find_optimal_batch_sizes(
                model=None,
                sae=None,
                layer=8,
                context_window=128,
                min_gpu_batch_size=8,
                max_gpu_batch_size=8,
                min_cpu_batch_size=1,
                max_cpu_batch_size=4,
            )


# ---------------------------------------------------------------------------
# Fix 6: fail loud when context_window exceeds the SAE truncation cap.
#
# Extraction head-truncates each doc to min(context_window, sae_max) but records
# the RAW context_window in shard metadata; an over-set context_window therefore
# yields shards whose recorded window exceeds reality, poisoning downstream
# consumers (fully-covered classification, feature-shard splice). The precondition
# must fail loud at startup — no clamp, no warn, no recorded min().
# ---------------------------------------------------------------------------


class TestContextWindowSaeMaxPrecondition:
    def test_raises_when_context_window_exceeds_sae_max(self, monkeypatch):
        monkeypatch.setattr(ef, "resolve_sae_max_context", lambda _model: 4096)
        with pytest.raises(ValueError) as exc:
            _assert_context_window_within_sae_max(
                NonBlankStr.parse("gemma-2-2b"),
                PositiveInt.parse(8192),
            )
        msg = str(exc.value)
        # Names both values and the forbidden silent-truncation behaviour.
        assert "8192" in msg and "4096" in msg
        assert "truncate" in msg.lower()

    def test_does_not_raise_when_context_window_equals_sae_max(self, monkeypatch):
        monkeypatch.setattr(ef, "resolve_sae_max_context", lambda _model: 4096)
        # cw == sae_max is the exact boundary: no truncation occurs, must not raise.
        _assert_context_window_within_sae_max(
            NonBlankStr.parse("gemma-2-2b"),
            PositiveInt.parse(4096),
        )

    def test_does_not_raise_when_context_window_below_sae_max(self, monkeypatch):
        monkeypatch.setattr(ef, "resolve_sae_max_context", lambda _model: 8192)
        _assert_context_window_within_sae_max(
            NonBlankStr.parse("gemma-2-2b"),
            PositiveInt.parse(2048),
        )
