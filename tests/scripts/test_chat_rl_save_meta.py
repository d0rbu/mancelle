"""Source-inspection tests for ``scripts/train/chat_rl.py`` save-meta.

Slice 11 audit P1 #7 surfaced two bugs in the chat_rl save_checkpoint
meta dict:

* missing ``user_config`` -> downstream ancestor walks (e.g.
  ``_recover_encdec_ancestor_config``) skip RL ancestors silently
  because they look up ``user_config.encoder``/etc.
* uses ``model.config.__dict__`` -> "slightly naughty" per the inline
  comment; survives only because GPTConfig has no nested fields.
  Replace with ``dataclasses.asdict(model.config)`` for resilience.

Both are pinned here as source-inspection guards (the full save path
requires a real DDP + model + optimizer setup; out of scope for unit
tests — DDP integration smoke is left to ``test_chat_rl_save_smoke.py``
when/if added).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAT_RL_PATH = REPO_ROOT / "scripts" / "train" / "chat_rl.py"


def test_save_meta_contains_user_config() -> None:
    """The meta dict in chat_rl's save_checkpoint must include
    ``"user_config": user_config`` so ancestor walks can recover it."""
    src = CHAT_RL_PATH.read_text()
    # Look for the save_checkpoint(...) block by anchoring on its known
    # shape (passes ``model.state_dict()`` then ``None`` for optim
    # state) and reading 30 following lines.
    save_match = re.search(
        r"save_checkpoint\(\s*checkpoint_dir,\s*step,\s*model\.state_dict\(\),\s*None,",
        src,
    )
    assert save_match is not None, (
        "save_checkpoint(...) call in chat_rl.py not found; this test "
        "must be updated if the save site has been moved."
    )
    # Slice the next 1500 chars (covers the multi-line dict + close paren).
    save_block = src[save_match.start() : save_match.start() + 1500]
    assert '"user_config"' in save_block, (
        "scripts/train/chat_rl.py save_checkpoint meta dict is missing "
        '``"user_config": user_config``. Slice 11 audit P1 #7: any '
        "downstream tooling that walks RL ancestors expecting the "
        "user_config.encoder/cross_attention/feature_prediction subkeys "
        "will silently skip RL nodes."
    )


def test_save_meta_uses_dataclass_asdict_not_dunder_dict() -> None:
    """``model.config.__dict__`` accesses raw instance attributes which
    break for nested dataclasses. Use ``dataclasses.asdict`` instead.

    Slice 11 audit P1 #7 / P3 #20.
    """
    src = CHAT_RL_PATH.read_text()
    # Look for the literal dunder-dict access (now banned).
    assert re.search(r"model\.config\.__dict__", src) is None, (
        "scripts/train/chat_rl.py uses model.config.__dict__ — replace "
        "with dataclasses.asdict(model.config) for nested-dataclass "
        "robustness. Slice 11 audit P3 #20."
    )
    # And that the asdict-based form is present.
    assert "asdict(model.config)" in src, (
        "scripts/train/chat_rl.py must serialize the model config via "
        "``asdict(model.config)`` (slice 11 audit P1 #7 / P3 #20)."
    )


def test_save_meta_does_not_persist_optimizer_state() -> None:
    """RL is not resume-able by design; the save passes None for
    optimizer_data. Pin this so a future "let's also save optim state"
    refactor surfaces as a test diff.
    """
    src = CHAT_RL_PATH.read_text()
    # The exact arg pattern: state_dict() ... None
    assert re.search(
        r"save_checkpoint\([\s\S]*?model\.state_dict\(\),\s*None,",
        src,
    ) is not None, (
        "chat_rl.py save_checkpoint must pass None for optimizer_data "
        "(RL is not resume-able by design)."
    )
