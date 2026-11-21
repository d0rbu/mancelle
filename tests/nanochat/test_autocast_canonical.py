"""Wave 3D task 6: canonical autocast/mixed-precision contract tests.

Enforces the "Mixed Precision Contract" documented in docs/architecture.md:

1. Canonical training scripts wrap forward in ``torch.amp.autocast(...)`` via an
   ``autocast_ctx`` alias built once at startup.
2. Inference (``nanochat/engine.py``) is intentionally off-pattern: no ambient
   autocast; instead it uses explicit ``.to(dtype=...)`` on KV cache + Linear
   cast-on-use.
3. ``GPT.forward`` logits output is always fp32 regardless of autocast state
   (softcap is computed in fp32).
4. ``docs/architecture.md`` contains the "Mixed Precision" contract section.

Scripts deliberately off-pattern must carry an inline ``AUTOCAST_OFFPATTERN``
comment and be listed in ``OFFPATTERN_TRAINING_SCRIPTS`` below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Tests in this file need the SDPA backend so the tiny-GPT forward
    works on CPU without hitting the flex_attention Triton compile path
    (which assumes CUDA). Save and restore so the override doesn't leak
    into other test files."""
    saved = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = saved


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Training scripts that MUST wrap the forward pass in ``with autocast_ctx:`` (or
# equivalent ``with torch.amp.autocast(...)``).
#
# Paths updated post script reorg (PR #394 moved scripts/ -> scripts/{train,eval,...}/).
# If the reorg moves them again, this is the canonical place to update; the
# existence assertion below will surface a stale path immediately.
CANONICAL_TRAINING_SCRIPTS = [
    REPO_ROOT / "scripts" / "train" / "base_train.py",
    REPO_ROOT / "scripts" / "train" / "chat_sft.py",
    REPO_ROOT / "scripts" / "train" / "chat_rl.py",
    REPO_ROOT / "scripts" / "train" / "conditional_encdec_sft.py",
]

# Scripts deliberately off-pattern (must carry AUTOCAST_OFFPATTERN inline comment).
OFFPATTERN_TRAINING_SCRIPTS = [
    REPO_ROOT / "scripts" / "train" / "conditional_encdec_train.py",
    REPO_ROOT / "scripts" / "train" / "conditional_encdec_selfcond.py",
]


def _iter_with_items(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                yield item.context_expr


def _has_autocast_with(tree: ast.AST) -> bool:
    """True if the tree contains a ``with autocast_ctx`` or
    ``with torch.amp.autocast(...)`` (or ``torch.autocast(...)``) context."""
    for ctx_expr in _iter_with_items(tree):
        # ``with autocast_ctx:`` — bare Name
        if isinstance(ctx_expr, ast.Name) and "autocast" in ctx_expr.id.lower():
            return True
        # Tuple/BoolOp with multiple managers: ``with disable_fp8(m), autocast_ctx:``
        # ast.With.items already flattens this so the Name form above catches it.
        if isinstance(ctx_expr, ast.Call):
            func = ctx_expr.func
            # ``torch.amp.autocast(...)`` → Attribute chain ending in ``autocast``
            # ``torch.autocast(...)`` → same
            name_chain: list[str] = []
            cur: ast.AST | None = func
            while isinstance(cur, ast.Attribute):
                name_chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                name_chain.append(cur.id)
            if "autocast" in name_chain:
                return True
    return False


# ---------------------------------------------------------------------------
# AST-based contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_path", CANONICAL_TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_training_scripts_use_autocast_context(script_path: Path) -> None:
    """Training scripts must wrap forward in a ``with autocast_ctx`` block."""
    assert script_path.exists(), f"Training script missing: {script_path}"
    source = script_path.read_text()
    tree = ast.parse(source)
    assert _has_autocast_with(tree), (
        f"{script_path.name} does not contain a ``with autocast_ctx`` / "
        f"``with torch.amp.autocast(...)`` block. Training scripts MUST wrap "
        f"forward in autocast per docs/architecture.md 'Mixed Precision Contract'. "
        f"If this script is deliberately off-pattern, add AUTOCAST_OFFPATTERN "
        f"inline comment and move it to OFFPATTERN_TRAINING_SCRIPTS in the test."
    )
    # Sanity: the script also references .backward() (i.e. it is a training script,
    # not misclassified).
    assert "backward()" in source, (
        f"{script_path.name} does not contain .backward() — not a training script?"
    )


@pytest.mark.parametrize("script_path", OFFPATTERN_TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_offpattern_training_scripts_are_annotated(script_path: Path) -> None:
    """Off-pattern scripts must carry an AUTOCAST_OFFPATTERN comment."""
    assert script_path.exists(), f"Off-pattern script missing: {script_path}"
    source = script_path.read_text()
    assert "AUTOCAST_OFFPATTERN" in source, (
        f"{script_path.name} is listed as off-pattern but has no "
        f"AUTOCAST_OFFPATTERN inline comment. Either add one explaining why "
        f"it skips the canonical autocast wrap, or migrate it back to canonical."
    )


def test_inference_engine_does_not_rely_on_ambient_autocast() -> None:
    """``nanochat/engine.py`` must either (a) wrap forward in its own autocast
    or (b) rely on explicit ``.to(dtype=...)`` + KV-cache dtype. Currently we
    expect (b) and require an AUTOCAST_OFFPATTERN comment documenting it."""
    engine_path = REPO_ROOT / "nanochat" / "engine.py"
    source = engine_path.read_text()
    has_offpattern_comment = "AUTOCAST_OFFPATTERN" in source
    has_explicit_dtype = "dtype = torch.bfloat16" in source or "dtype=torch.bfloat16" in source
    assert has_offpattern_comment, (
        "nanochat/engine.py must carry an AUTOCAST_OFFPATTERN comment "
        "explaining why the main generate() path does not open an autocast "
        "context (it pre-allocates the KV cache with explicit dtype instead). "
        "See docs/architecture.md 'Mixed Precision Contract'."
    )
    assert has_explicit_dtype, (
        "nanochat/engine.py must set an explicit dtype for the KV cache "
        "allocation (currently expected: bf16 on CUDA, fp32 otherwise)."
    )


def test_docs_mentions_autocast_contract() -> None:
    """docs/reference/architecture.md must contain the Mixed Precision contract section."""
    arch = REPO_ROOT / "docs" / "reference" / "architecture.md"
    assert arch.exists(), "docs/reference/architecture.md is missing"
    text = arch.read_text()
    assert "autocast" in text.lower(), "architecture.md does not mention autocast"
    assert "Mixed Precision" in text, (
        "architecture.md must contain a 'Mixed Precision' heading documenting "
        "the canonical autocast contract."
    )


# ---------------------------------------------------------------------------
# Dtype-propagation contract tests — build a real tiny GPT and exercise forward
# ---------------------------------------------------------------------------


def _build_tiny_gpt() -> GPT:
    """Build a tiny CPU GPT (fp32 master weights) for dtype-contract tests."""
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    return model


def test_gpt_forward_dtype_contract_no_autocast() -> None:
    """Outside any autocast context, GPT.forward logits must be fp32 (softcap
    forces fp32 regardless of internal COMPUTE_DTYPE)."""
    model = _build_tiny_gpt()
    idx = torch.randint(0, 64, (1, 8), dtype=torch.long)
    with torch.no_grad():
        logits = model(idx)
    assert logits.dtype == torch.float32, (
        f"Expected fp32 logits outside autocast, got {logits.dtype}. "
        f"Contract: GPT.forward logits are always fp32 (softcap in fp32)."
    )


def test_gpt_forward_dtype_contract_autocast() -> None:
    """Inside a CPU autocast bf16 context, GPT.forward logits must STILL be fp32
    because the softcap step explicitly calls .float() before returning.

    This is the key contract: autocast affects intermediate compute only. The
    returned logits are always fp32 so loss / cross_entropy operates in fp32."""
    model = _build_tiny_gpt()
    idx = torch.randint(0, 64, (1, 8), dtype=torch.long)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(idx)
    assert logits.dtype == torch.float32, (
        f"Expected fp32 logits even inside autocast (softcap forces fp32), "
        f"got {logits.dtype}. If this changed, update the Mixed Precision "
        f"Contract in docs/architecture.md."
    )


def test_gpt_forward_loss_dtype_is_fp32() -> None:
    """When targets are provided, the returned loss must also be fp32 since
    cross_entropy is computed on fp32 logits."""
    model = _build_tiny_gpt()
    idx = torch.randint(0, 64, (1, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (1, 8), dtype=torch.long)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(idx, targets)
    assert loss.dtype == torch.float32, f"Expected fp32 loss, got {loss.dtype}."
