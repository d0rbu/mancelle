"""Regression tests for the ``val_bpb`` NameError bug fix.

Background
----------
``scripts/train/chat_sft.py`` previously assigned ``val_bpb`` only inside the
eval block (``if last_step or step % eval_every == 0:``). When a
``save_every`` boundary fired on a step where eval did NOT — most
obviously when ``eval_every`` is set to 0 (disabled), or when
``save_every`` and ``eval_every`` share no common step before the
first save — the save block tried to read ``val_bpb`` and crashed with
``NameError``.

The fix initializes ``val_bpb = float("nan")`` before the training loop,
so an intermediate save can always write a valid meta JSON (the saved
``val_bpb`` is NaN if no eval has run yet, instead of crashing).

These tests pin two things:

1. A source-inspection guard that the pre-loop initialization is still
   present (regressions tend to come as "remove the suspicious-looking
   line" refactors).
2. A runtime guard that ``json.dump`` on a meta dict containing
   ``val_bpb = NaN`` round-trips safely — the concrete thing
   ``save_checkpoint`` does with the ``meta_*.json`` file.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

_CHAT_SFT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "train" / "chat_sft.py"
)


class TestValBpbPreLoopInitialization:
    """Source-inspection guards for the `val_bpb = float('nan')`
    initialization that precedes the training loop."""

    @pytest.fixture(scope="class")
    def chat_sft_source(self) -> str:
        return _CHAT_SFT_PATH.read_text()

    def test_val_bpb_initialized_before_training_loop(
        self, chat_sft_source: str
    ) -> None:
        """The initialization must appear BEFORE the `while True:` line that
        starts the training loop. Prevents a regression that would put the
        init inside the eval block (where it already was — that's the bug).
        """
        # Find the while True line that starts the training loop. There's
        # only one in the script.
        while_true_match = re.search(r"^\s+while True:\s*$", chat_sft_source, re.MULTILINE)
        assert while_true_match is not None, (
            "scripts/train/chat_sft.py missing the `while True:` training loop line"
        )
        prelude = chat_sft_source[: while_true_match.start()]
        assert re.search(r"^\s+val_bpb\s*=\s*float\(\"nan\"\)", prelude, re.MULTILINE), (
            "val_bpb must be initialized before the training loop "
            "(e.g. `val_bpb = float(\"nan\")`). If `save_every` fires on a "
            "step where eval does not, the save block references val_bpb "
            "and crashes with NameError. See test_chat_sft_save_val_bpb.py."
        )

    def test_val_bpb_init_comes_after_min_val_bpb(
        self, chat_sft_source: str
    ) -> None:
        """Small housekeeping guard: `val_bpb` init is grouped with
        `min_val_bpb` init (they're the two bpb-related pre-loop state
        variables). This prevents someone from stashing the init far from
        its semantic neighbors where it would be easier to accidentally
        delete."""
        min_val_match = re.search(
            r"^\s+min_val_bpb\s*=", chat_sft_source, re.MULTILINE
        )
        val_bpb_match = re.search(
            r"^\s+val_bpb\s*=\s*float\(\"nan\"\)", chat_sft_source, re.MULTILINE
        )
        assert min_val_match is not None and val_bpb_match is not None
        # val_bpb is initialized within ~10 lines of min_val_bpb
        between = chat_sft_source[min_val_match.start() : val_bpb_match.start()]
        assert between.count("\n") <= 10, (
            "val_bpb initialization drifted far from min_val_bpb — keep "
            "them together so a future refactor doesn't silently delete "
            "the val_bpb init and restore the NameError."
        )


class TestValBpbNaNMetaSerializes:
    """Runtime guard: the concrete save path (``json.dump`` on a meta
    dict containing ``val_bpb = NaN``) must not raise. atomic_json_dump
    uses ``json.dump`` under the hood, which by default emits ``NaN``
    literally (non-standard JSON but Python's parser accepts it).

    This test protects against a future change to the json serializer
    config that would start rejecting NaN (e.g. ``json.dump(allow_nan=False)``).
    """

    def test_meta_with_nan_val_bpb_dumps_and_loads(self, tmp_path: Path) -> None:
        """Mirror the exact shape of chat_sft.py's save-time meta dict."""
        meta = {
            "step": 500,
            "val_bpb": float("nan"),
            "model_config": {
                "sequence_len": 1024,
                "vocab_size": 32768,
                "n_layer": 12,
                "n_head": 6,
                "n_kv_head": 6,
                "n_embd": 768,
                "window_pattern": "SSSL",
            },
            "user_config": {"mixture": "safety", "save_every": 500},
        }
        path = tmp_path / "meta_000500.json"

        # Dump (as chat_sft.py does via save_checkpoint → atomic_json_dump).
        with open(path, "w") as f:
            json.dump(meta, f)

        # Round-trip back.
        with open(path) as f:
            loaded = json.load(f)

        assert loaded["step"] == 500
        assert math.isnan(loaded["val_bpb"])
        assert loaded["model_config"]["n_layer"] == 12

    def test_finite_val_bpb_still_serializes(self, tmp_path: Path) -> None:
        """Regression guard that finite floats continue to work."""
        meta = {"val_bpb": 1.2345, "step": 100}
        path = tmp_path / "meta_000100.json"
        with open(path, "w") as f:
            json.dump(meta, f)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["val_bpb"] == pytest.approx(1.2345)


class TestSaveFiresWithoutEvalBoundary:
    """Integration-style guard: the configuration that triggered the
    original NameError (`eval_every=0` or misaligned `save_every`/
    `eval_every`) must not crash the save block.

    This test asserts the source-level structure that makes the
    scenario safe — the explicit `val_bpb = float("nan")` initialization
    — because spinning up the full training loop is too heavy for a
    unit test. See also the integration test under
    `tests/integration/` (if/when added) that runs the real loop.
    """

    def test_eval_every_zero_path_does_not_reference_undefined_val_bpb(
        self,
    ) -> None:
        """Under ``eval_every=0``, the `if last_step or (eval_every > 0 and ...)`
        guard prevents the eval block from running. ``val_bpb`` must be
        pre-initialized so the subsequent save block reads a defined value.
        """
        source = _CHAT_SFT_PATH.read_text()
        # Pin the guard shape: eval_every > 0 is the condition for the eval
        # block. If someone changes this to `>= 0`, eval runs even with
        # eval_every=0 and the NaN default stops mattering — still not a
        # bug, but this test catches the guard drift.
        assert "int(cfg.eval_every) > 0" in source, (
            "scripts/train/chat_sft.py eval block must gate on `cfg.eval_every > 0` "
            "so save_every can be used standalone (eval_every=0) without "
            "running eval unnecessarily."
        )
