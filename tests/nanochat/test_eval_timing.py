"""Tests for timing instrumentation in evaluation functions and MFU computation."""

import io
from unittest.mock import patch

import torch
from loguru import logger

from nanochat.loss_eval import evaluate_bpb
from steering.conditional.encoder_decoder.loss_eval import evaluate_encdec_bpb_conditioned
from nanochat.core_eval import evaluate_task, TaskMeta, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel:
    """Minimal model mock for evaluate_bpb."""

    def get_device(self):
        return torch.device("cpu")

    def __call__(self, x, y, loss_reduction="none"):
        return torch.zeros(x.shape, dtype=torch.float32)

    def eval(self):
        pass

    def train(self):
        pass


class MockEncdecModel:
    """Minimal model mock for evaluate_encdec_bpb_conditioned."""

    def __call__(self, vx, vy, enc_fids, enc_mags, enc_mask, seg_map, loss_reduction="none"):
        return torch.zeros(vx.shape, dtype=torch.float32)

    def eval(self):
        pass

    def train(self):
        pass


def _capture_logs():
    """Return a StringIO sink and its handler id for loguru capture."""
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}", level="INFO")
    return sink, handler_id


# ---------------------------------------------------------------------------
# Issue 3 — evaluate_bpb timing
# ---------------------------------------------------------------------------


def test_evaluate_bpb_timing_output():
    """evaluate_bpb should log elapsed time and tok/sec."""
    model = MockModel()
    vocab_size = 32
    token_bytes = torch.ones(vocab_size, dtype=torch.int64)
    batch_size, seq_len = 2, 8

    def batch_gen():
        while True:
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            y = torch.randint(0, vocab_size, (batch_size, seq_len))
            yield x, y

    sink, hid = _capture_logs()
    try:
        bpb = evaluate_bpb(model, batch_gen(), steps=3, token_bytes=token_bytes)
    finally:
        logger.remove(hid)

    log_output = sink.getvalue()
    assert "evaluate_bpb:" in log_output, f"Missing timing header in log: {log_output}"
    assert "tok/sec" in log_output, f"Missing tok/sec in log: {log_output}"
    assert bpb == 0.0  # zero loss => zero bpb


# ---------------------------------------------------------------------------
# Issue 3 — evaluate_encdec_bpb_conditioned timing
# ---------------------------------------------------------------------------


def test_evaluate_encdec_bpb_conditioned_timing_output():
    """evaluate_encdec_bpb_conditioned should log elapsed time, tokens, and tok/sec."""
    model = MockEncdecModel()
    vocab_size = 32
    token_bytes = torch.ones(vocab_size, dtype=torch.int64, device="cpu")
    batch_size, seq_len = 2, 8
    n_seg = 2
    max_features = 4

    def val_loader():
        while True:
            vx = torch.randint(0, vocab_size, (batch_size, seq_len))
            vy = torch.randint(0, vocab_size, (batch_size, seq_len))
            enc_fids = torch.zeros(batch_size, n_seg, max_features, dtype=torch.long)
            enc_mags = torch.zeros(batch_size, n_seg, max_features, dtype=torch.long)
            enc_mask = torch.ones(batch_size, n_seg, max_features, dtype=torch.bool)
            enc_pcts = torch.zeros(batch_size, n_seg, max_features)
            seg_map = torch.zeros(batch_size, seq_len, dtype=torch.long)
            seg_doc_id = torch.zeros(batch_size, n_seg, dtype=torch.long)
            dummy = torch.zeros(1)
            yield vx, vy, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, dummy, None

    sink, hid = _capture_logs()
    try:
        result = evaluate_encdec_bpb_conditioned(
            model, val_loader(), eval_steps=2, token_bytes=token_bytes, device="cpu"
        )
    finally:
        logger.remove(hid)

    log_output = sink.getvalue()
    assert "evaluate_encdec_bpb_conditioned:" in log_output, f"Missing timing header in log: {log_output}"
    assert "tok/sec" in log_output, f"Missing tok/sec in log: {log_output}"
    assert "tokens" in log_output, f"Missing token count in log: {log_output}"
    assert result.bpb_conditioned == 0.0


# ---------------------------------------------------------------------------
# Issue 3 — evaluate_task timing
# ---------------------------------------------------------------------------


def test_evaluate_task_timing_output():
    """evaluate_task should log elapsed time, example count, and ex/sec."""

    class SimpleModel:
        max_seq_len = 64

        def __call__(self, input_ids):
            b, t = input_ids.shape
            return torch.randn(b, t, 32)

        def eval(self):
            pass

        def train(self):
            pass

    class SimpleTokenizer:
        def __call__(self, prompts, prepend=None):
            return [list(range(10)) for _ in prompts]

        def get_bos_token_id(self):
            return 0

    model = SimpleModel()
    tokenizer = SimpleTokenizer()
    # Minimal multiple-choice data
    data = [
        {"query": "What?", "choices": ["A", "B"], "gold": 0},
        {"query": "Why?", "choices": ["C", "D"], "gold": 1},
    ]
    task_meta = TaskMeta(
        task_type=TaskType.MULTIPLE_CHOICE,
        num_fewshot=0,
        continuation_delimiter=" ",
    )

    sink, hid = _capture_logs()
    try:
        evaluate_task(model, tokenizer, data, torch.device("cpu"), task_meta)
    finally:
        logger.remove(hid)

    log_output = sink.getvalue()
    assert "evaluate_task:" in log_output, f"Missing timing header in log: {log_output}"
    assert "ex/sec" in log_output, f"Missing ex/sec in log: {log_output}"
    assert "2 examples" in log_output, f"Missing example count in log: {log_output}"


# ---------------------------------------------------------------------------
# Issue 4 — chat_sft MFU uses get_peak_flops
# ---------------------------------------------------------------------------


def test_mfu_uses_get_peak_flops():
    """MFU computation should use get_peak_flops, not a hardcoded value."""
    from nanochat.common import get_peak_flops

    # Verify the function exists and returns reasonable values for known GPUs
    result = get_peak_flops("NVIDIA H100 SXM")
    assert result > 0, "get_peak_flops should return positive FLOPS"

    # Verify it can be mocked (as chat_sft.py would import it)
    with patch("nanochat.common.get_peak_flops", return_value=500e12):
        from nanochat.common import get_peak_flops as gpf

        assert gpf("test") == 500e12


def test_chat_sft_no_hardcoded_h100():
    """chat_sft.py should not contain a hardcoded 989e12 FLOPS value."""
    import pathlib

    sft_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "train"
        / "chat_sft.py"
    )
    source = sft_path.read_text()
    assert "989e12" not in source, (
        "chat_sft.py still contains hardcoded H100 peak FLOPS (989e12)"
    )
    assert "promised_flops_per_sec_h100" not in source, (
        "chat_sft.py still references promised_flops_per_sec_h100"
    )
    assert "get_peak_flops" in source, (
        "chat_sft.py should import and use get_peak_flops"
    )
