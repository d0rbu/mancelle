"""Opt-in two-prompt generation using a staged checkpoint, fitted target and cache.

CAMPAIGN2_SAFETY_GPU_CONFIG points to campaign2 config overrides. Invoke under
`timeout 840 .venv/bin/python -m pytest ... -m slow`; no fitting or judging.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.eval import campaign2 as runner
from scripts.utils.campaign2 import install_json_once
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.campaign2 import cell_output_dir, validate_safety_record
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool

ROOT = Path(__file__).resolve().parents[2]
GPU_MEMORY_LIMIT_BYTES = 10_000_000_000
pytestmark = pytest.mark.slow


def wait_for_idle_gpu() -> None:
    deadline = time.monotonic() + 60
    while True:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True)
        other = {int(line) for line in result.stdout.splitlines() if line.strip()} - {os.getpid()}
        if not other:
            return
        if time.monotonic() >= deadline:
            pytest.skip(f"shared GPU still busy after 60 seconds: {sorted(other)}")
        time.sleep(60)


def test_local_masked_checkpoint_safety_generation(tmp_path, monkeypatch):
    config_path = os.environ.get("CAMPAIGN2_SAFETY_GPU_CONFIG")
    if not config_path:
        pytest.skip("set CAMPAIGN2_SAFETY_GPU_CONFIG for the staged masked checkpoint smoke")
    with initialize_config_dir(config_dir=str(ROOT / "configs/eval"), version_base=None):
        cfg = compose(config_name="campaign2", overrides=["variant=encdec_safety"])
    cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path), {
        "mode": "generate", "variant": "encdec_safety", "device": "cuda",
        "output_dir": str(tmp_path / "cell"), "max_new_tokens": 40,
        "safety": {"strength": 3}, "allow_paid_judge": False})
    # This admission must remain GPU-free even when cfg.device requests CUDA.
    identity = runner.preflight(cfg)
    assert identity["safety_binding"]["target"]["fit_contract"]["feature_vocab"] is not None
    output = cell_output_dir(cfg)
    install_json_once(output / "identity.json", identity)
    pool = resolve_harmfulness_prompt_pool(seed=cfg.seed, max_harmful=cfg.num_harmful, num_benign=cfg.num_benign)
    rows = [dict(row) for row in pool.rows]
    pending = [(next(index for index, row in enumerate(rows) if row["pool"] == kind), kind)
               for kind in ("harmful", "benign")]
    wait_for_idle_gpu()
    assert t.cuda.is_available()
    total = t.cuda.get_device_properties(0).total_memory
    t.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEMORY_LIMIT_BYTES / total))
    t.cuda.reset_peak_memory_stats()
    original_to, original_forward = EncoderDecoderGPT.to, EncoderDecoderGPT.forward
    forwards = []

    def guarded_to(self, *args, **kwargs):
        if t.device(kwargs.get("device", "cpu")).type == "cuda":
            wait_for_idle_gpu()  # Recheck immediately before the actual GPU transfer.
        return original_to(self, *args, **kwargs)

    def checked_forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        logits = result[0] if isinstance(result, tuple) else result
        assert bool(t.isfinite(logits).all())
        forwards.append(True)
        return result

    monkeypatch.setattr(EncoderDecoderGPT, "to", guarded_to)
    monkeypatch.setattr(EncoderDecoderGPT, "forward", checked_forward)
    monkeypatch.setattr(runner, "judge", lambda *args: pytest.fail("GPU smoke called paid judge"))
    started = time.monotonic()
    runner.generate_encdec(cfg, [(index, rows[index]) for index, _ in pending], output)
    elapsed = time.monotonic() - started
    assert elapsed < 15 * 60 and forwards
    assert t.cuda.max_memory_reserved() < 12_000_000_000
    for index, _ in pending:
        record = json.loads((output / "records" / f"{index:06d}.json").read_text())
        validate_safety_record(record, rows[index], index, identity)
    install_json_once(tmp_path / "qualification.json", {"seconds": elapsed, "forwards": len(forwards),
        "peak_reserved_bytes": t.cuda.max_memory_reserved(), "prompt_indexes": [index for index, _ in pending],
        "safety_binding": identity["safety_binding"]})
