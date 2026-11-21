"""Tests for the optimizer-state skip on intermediate ``save_every`` saves.

``scripts/train/chat_sft.py`` saves a checkpoint on every ``save_every``
boundary when that config is set. Since the docstring explicitly notes
that intermediate checkpoints are NOT resume-able (dataloader state is
not persisted), persisting the optimizer state at every intermediate
boundary is just disk bloat — AdamW + Muon state for a depth-26 model
is typically 2-4× the model state size. Multiplied by (num_iterations /
save_every) intermediate saves, the overhead matters.

Slice 11 audit P1 #5 refined this further: the FINAL save must run on
every rank (passing ``rank=ddp_rank``) so each rank's ZeRO-2 sharded
optimizer slice is persisted; previously the entire save block was
gated on ``master_process``, so ranks 1..N-1 silently dropped their
Muon momentum / 2nd-moment buffers and any downstream resume restarted
those slices from scratch.

This test file guards the behavior via source inspection. A full
end-to-end training-loop test would require spinning up DDP, a model,
optimizers, etc. — too heavy. The source-string guards catch the most
likely regression: someone re-introducing the master-only gating
around the final save.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CHAT_SFT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "train" / "chat_sft.py"
)


class TestOptimizerStateOnlyOnLastStep:
    @pytest.fixture(scope="class")
    def chat_sft_source(self) -> str:
        return _CHAT_SFT_PATH.read_text()

    def test_intermediate_save_block_passes_none_for_optimizer_state(
        self, chat_sft_source: str
    ) -> None:
        """The intermediate (master-only) save_checkpoint call must
        explicitly pass ``None`` for ``optimizer_data``. Otherwise every
        intermediate boundary serializes 2-4× the model size of
        optimizer buffers — wasteful since intermediate saves are
        eval-only (dataloader state is not persisted, so they aren't
        resume-able anyway).

        Slice 11 audit P1 #5 split the original single ``save_checkpoint``
        call into two: a final save (all ranks, with optim state) and
        an intermediate save (master-only, no optim state). This guard
        pins the intermediate branch's optimizer arg to ``None``.
        """
        # Match the intermediate save_checkpoint block: identified by
        # the ``orig_model.state_dict()`` followed by exactly ``None``
        # in the optim slot.
        intermediate_save = re.search(
            r"save_checkpoint\(\s*checkpoint_dir,\s*completed_steps,\s*orig_model\.state_dict\(\),\s*None,",
            chat_sft_source,
        )
        assert intermediate_save is not None, (
            "scripts/train/chat_sft.py must contain a save_checkpoint(...) "
            "call passing ``None`` for optimizer_data on the intermediate "
            "(master-only) branch. Slice 11 audit P1 #5: intermediate "
            "saves are eval-only and must skip optimizer state."
        )

    def test_final_save_block_passes_optimizer_state_with_rank(
        self, chat_sft_source: str
    ) -> None:
        """The final-step save MUST pass ``optimizer.state_dict()`` AND
        ``rank=ddp_rank`` so every rank persists its ZeRO-2 sharded
        slice. Slice 11 audit P1 #5: the previous master-gated form
        silently dropped ranks 1..N-1's Muon momentum / 2nd-moment.
        """
        final_save = re.search(
            r"save_checkpoint\(\s*checkpoint_dir,\s*completed_steps,\s*orig_model\.state_dict\(\),"
            r"\s*optimizer\.state_dict\(\),"
            r"[\s\S]*?rank=ddp_rank",
            chat_sft_source,
        )
        assert final_save is not None, (
            "scripts/train/chat_sft.py must contain a save_checkpoint(...) "
            "call that passes ``optimizer.state_dict()`` AND "
            "``rank=ddp_rank`` for the final save. Slice 11 audit P1 #5: "
            "every rank writes its sharded optim slice for downstream "
            "resume / inspection."
        )

    def test_docstring_flags_non_resumability(self, chat_sft_source: str) -> None:
        """Double-check that the docstring explicitly warns intermediates
        are not resume-able, so the skip is intentional and documented."""
        # The comment block above the save describes the intent.
        assert "NOT resume-able" in chat_sft_source, (
            "scripts/train/chat_sft.py must document the intermediate saves' "
            "non-resumability so the optimizer-state skip is justified "
            "to future readers."
        )
