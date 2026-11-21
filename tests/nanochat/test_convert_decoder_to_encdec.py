"""Direct tests for scripts/convert_decoder_to_encdec.py.

Wave 3C P0 coverage: the converter had zero tests and a stale docstring
claiming cross-attention c_proj is zero-initialized for "identical-to-decoder"
output at step 0. The FP8 amax=0 fix (cross_attention.init_weights) replaced
the zero init with Uniform(+-sqrt(3/dec_n_embd)), breaking that promise.

These tests:
- lock in bit-exact preservation of every decoder weight through the conversion
- lock in the new Uniform init for cross-attention c_proj (prevents regression
  to zero init, which causes FP8 amax=0 degeneracy)
- lock in idempotency
- verify that a renamed / missing source key is surfaced (not silently absorbed)
- verify decoder-only forward equivalence (the real "identical output" property)

The script is procedural (runs at module import via hydra + sys.argv), so we
exercise it in a subprocess with NANOCHAT_BASE_DIR pointing at a tmp_path and
a tiny pre-populated decoder checkpoint. `get_tokenizer` is stubbed in the
subprocess to avoid needing a trained tokenizer on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nanochat.checkpoint_manager import save_checkpoint
from tests.conftest import (
    TINY_N_EMBD,
    TINY_VOCAB,
    _make_gpt,
    _make_gpt_config,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_TAG = "d2"
STEP = 0
SEG = 16
THRESHOLD = 2048.9
THRESHOLD_DIR = f"t{THRESHOLD}"


def _write_decoder_checkpoint(base_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Build a tiny decoder GPT, save a checkpoint under base_checkpoints/d2/.

    Returns (model, state_dict) for in-test comparison.
    """
    torch.manual_seed(0)
    model = _make_gpt(device="cpu")
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt_dir = base_dir / "base_checkpoints" / MODEL_TAG
    save_checkpoint(
        str(ckpt_dir),
        STEP,
        sd,
        None,
        {
            "step": STEP,
            "val_bpb": 1.23,
            "model_config": asdict(_make_gpt_config()),
            "user_config": {},
        },
    )
    return model, sd


RUNNER_TEMPLATE = textwrap.dedent(
    """
    import sys
    import types

    # Stub get_tokenizer before the converter imports it — the converter
    # only calls tokenizer at the very end for a log message. Avoiding
    # the real tokenizer keeps the test hermetic (no tokenizer.json on disk).
    import nanochat.tokenizer as _tok_mod

    def _stub_get_tokenizer():
        class _StubTok:
            def get_vocab_size(self):
                return {vocab}
        return _StubTok()

    _tok_mod.get_tokenizer = _stub_get_tokenizer

    sys.argv = ["convert_decoder_to_encdec"] + {overrides!r}
    import runpy
    runpy.run_module("scripts.preprocess.convert_decoder_to_encdec", run_name="__main__")
    """
)


def _run_converter(base_dir: Path, extra_overrides: list[str] | None = None) -> None:
    """Invoke the converter script in a subprocess with NANOCHAT_BASE_DIR set."""
    overrides = [
        "source=base",
        f"model_tag={MODEL_TAG}",
        f"step={STEP}",
        f"segment_size={SEG}",
        # Tiny encoder/xattn config: real dim (not ratio), depth=1, xattn at every layer.
        "encoder.n_embd=32",
        "encoder.depth=1",
        "encoder.n_head=2",
        "encoder.n_kv_head=2",
        "encoder.head_dim=16",
        "cross_attention.frequency=1",
        "n_sae_features=32",
        "num_buckets=4",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    runner = RUNNER_TEMPLATE.format(vocab=TINY_VOCAB, overrides=overrides)

    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(base_dir)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Converter subprocess failed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def _encdec_checkpoint_path(base_dir: Path) -> Path:
    return (
        base_dir
        / "conditional_encdec_checkpoints"
        / MODEL_TAG
        / f"s{SEG}"
        / THRESHOLD_DIR
        / f"model_{STEP:06d}.pt"
    )


def _load_encdec_state(base_dir: Path) -> dict[str, torch.Tensor]:
    path = _encdec_checkpoint_path(base_dir)
    assert path.exists(), f"Converter did not write {path}"
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_decoder_weights_preserved_exactly(tmp_path: Path) -> None:
    """Every decoder weight must survive conversion bit-exactly (torch.equal)."""
    _dec_model, dec_sd = _write_decoder_checkpoint(tmp_path)
    _run_converter(tmp_path)
    encdec_sd = _load_encdec_state(tmp_path)

    missing_in_encdec: list[str] = []
    mismatches: list[str] = []
    for key, dec_tensor in dec_sd.items():
        if key not in encdec_sd:
            missing_in_encdec.append(key)
            continue
        enc_tensor = encdec_sd[key]
        if enc_tensor.shape != dec_tensor.shape or not torch.equal(
            enc_tensor, dec_tensor
        ):
            mismatches.append(key)

    assert not missing_in_encdec, (
        f"Decoder keys missing from encdec state_dict: {missing_in_encdec}"
    )
    assert not mismatches, (
        f"Decoder weights not bit-exactly preserved in encdec for keys: {mismatches}"
    )


def test_cross_attention_c_proj_is_uniform_not_zero(tmp_path: Path) -> None:
    """Lock in the FP8 fix: c_proj is Uniform(+-sqrt(3/dec_n_embd)), NOT zero.

    Zero init was replaced because FP8 tensorwise scaling degenerates on
    all-zero tensors (amax=0 -> inf gradients on first backward).
    """
    _dec_model, _ = _write_decoder_checkpoint(tmp_path)
    _run_converter(tmp_path)
    encdec_sd = _load_encdec_state(tmp_path)

    dec_n_embd = TINY_N_EMBD
    expected_bound = (3.0 / dec_n_embd) ** 0.5
    expected_std = (1.0 / dec_n_embd) ** 0.5  # Uniform(-b, b) with b=sqrt(3/d) has std sqrt(1/d)

    c_proj_keys = [k for k in encdec_sd if "cross_attn_layers" in k and "c_proj.weight" in k]
    assert c_proj_keys, "Expected at least one cross-attention c_proj in encdec state_dict"

    for key in c_proj_keys:
        w = encdec_sd[key].float()
        assert torch.any(w != 0), f"{key} is all-zero — FP8 regression (c_proj zero init)"
        # Bounds check (Uniform samples strictly within +- bound)
        assert w.abs().max().item() <= expected_bound + 1e-5, (
            f"{key} exceeds expected Uniform bound {expected_bound}: max={w.abs().max().item()}"
        )
        # Distribution sanity: mean ~ 0, std ~ sqrt(1/d). Tolerance scales with 1/sqrt(N).
        n = w.numel()
        assert n >= 64, f"{key} too small ({n}) to meaningfully check distribution"
        mean = w.mean().item()
        std = w.std().item()
        # 4 sigma envelope on mean (std of sample mean = expected_std / sqrt(n)).
        mean_tol = 4.0 * expected_std / (n**0.5)
        assert abs(mean) < mean_tol, f"{key} mean={mean} outside +-{mean_tol}"
        # std should be within 20% of the theoretical value for n>=64 samples.
        assert abs(std - expected_std) / expected_std < 0.2, (
            f"{key} std={std} far from expected {expected_std}"
        )


def test_converter_detects_missing_source_key(tmp_path: Path) -> None:
    """If a source checkpoint has a renamed key that no longer matches the
    target model, `load_state_dict(strict=False)` silently leaves random init
    in the target. The converter must surface this via an 'unexpected keys'
    warning (it already logs `missing`/`unexpected` counts) or by raising.
    """
    _dec_model, _ = _write_decoder_checkpoint(tmp_path)
    # Monkey-rename one decoder key to simulate a schema drift that would
    # silently skip a real weight. We rewrite the saved model_*.pt.
    ckpt_path = (
        tmp_path / "base_checkpoints" / MODEL_TAG / f"model_{STEP:06d}.pt"
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Rename wte.weight -> old_wte.weight so the target keeps random wte
    # and gains one unexpected key.
    sd["transformer.old_wte.weight"] = sd.pop("transformer.wte.weight")
    torch.save(sd, ckpt_path)

    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    overrides = [
        "source=base",
        f"model_tag={MODEL_TAG}",
        f"step={STEP}",
        f"segment_size={SEG}",
        "encoder.n_embd=32",
        "encoder.depth=1",
        "encoder.n_head=2",
        "encoder.n_kv_head=2",
        "encoder.head_dim=16",
        "cross_attention.frequency=1",
        "n_sae_features=32",
        "num_buckets=4",
    ]
    runner = RUNNER_TEMPLATE.format(vocab=TINY_VOCAB, overrides=overrides)
    result = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    combined = result.stdout + result.stderr
    # Either the converter must raise OR it must prominently warn about the
    # unexpected key. We accept a warning path because that is the current
    # logger.warning("Unexpected keys ...") behavior — but the test would
    # fail loudly if that warning got silently deleted.
    if result.returncode == 0:
        assert "old_wte" in combined, (
            "Converter silently accepted a renamed source key without warning. "
            f"Output:\n{combined}"
        )
        assert (
            "Unexpected keys" in combined
            or "unexpected" in combined.lower()
        ), f"No 'unexpected keys' diagnostic emitted. Output:\n{combined}"


def test_converter_is_idempotent(tmp_path: Path) -> None:
    """Running the converter twice must produce bit-identical encdec state."""
    _write_decoder_checkpoint(tmp_path)

    _run_converter(tmp_path)
    first = _load_encdec_state(tmp_path)
    first_meta_path = _encdec_checkpoint_path(tmp_path).with_name(
        f"meta_{STEP:06d}.json"
    )
    first_meta = json.loads(first_meta_path.read_text())

    _run_converter(tmp_path)
    second = _load_encdec_state(tmp_path)
    second_meta = json.loads(first_meta_path.read_text())

    assert set(first) == set(second), "Key set changed between conversions"
    for key in first:
        assert torch.equal(first[key], second[key]), (
            f"Non-idempotent: {key} differs between two conversions"
        )
    assert first_meta == second_meta, "Metadata differs between two conversions"


def test_decoder_forward_unchanged_post_conversion(tmp_path: Path) -> None:
    """The converted encdec's decoder stack, loaded with the preserved weights,
    must produce identical decoder-side hidden states as the original GPT.

    We cannot feed the EncoderDecoderGPT without an encoder input (it asserts
    enc_mask is required) — so instead we verify the decoder-owned parameters
    are bit-identical AND rebuild a GPT with them to confirm logit equality on
    a fixed input. This isolates the real "decoder preserved" claim from the
    stale "identical model output at step 0" docstring promise, which no
    longer holds now that cross-attention c_proj is non-zero.
    """
    dec_model, dec_sd = _write_decoder_checkpoint(tmp_path)
    _run_converter(tmp_path)
    encdec_sd = _load_encdec_state(tmp_path)

    # Extract the decoder-owned subset of the encdec state dict and rebuild a GPT.
    from nanochat.gpt import GPT

    gpt_config = _make_gpt_config()
    with torch.device("meta"):
        reloaded = GPT(gpt_config, pad_vocab_size_to=TINY_VOCAB)
    reloaded = reloaded.to_empty(device="cpu")
    reloaded.init_weights()
    decoder_subset = {k: v for k, v in encdec_sd.items() if k in dec_sd}
    missing, unexpected = reloaded.load_state_dict(decoder_subset, strict=False)
    assert not missing, f"Missing decoder keys when rebuilding GPT: {missing}"
    reloaded.eval()
    dec_model.eval()

    torch.manual_seed(42)
    idx = torch.randint(0, TINY_VOCAB, (1, 32))
    with torch.no_grad():
        orig_logits = dec_model(idx, targets=None)
        new_logits = reloaded(idx, targets=None)
    assert torch.equal(orig_logits, new_logits), (
        "Decoder logits diverged after round-tripping through the converter. "
        "This means the converter does not preserve decoder weights bit-exactly."
    )
