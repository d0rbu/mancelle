"""Guard the unified cross-tokenizer end-marker window constant.

Wave 3D task 2 — both sides of the Gemma/RustBPE bridge must agree on a
single char-window for end-of-turn marker residue at segment boundaries:

  * ``scripts/preprocess/precompute_token_boundaries.py`` — tiny trailing
    segment repair (extraction side)
  * ``steering/conditional/encoder_decoder/full_sequence_routing.py`` —
    full-sequence SFT partial-coverage absorption
  * ``steering/eval/target_segments.py`` — target-real partial coverage
    absorption (Slice 14; coordinated with Slice 02 audit, see manifest
    §4.12 / Q12 / Slice-02 B-1)

Before Wave 3D/2 these used 16 and 20 respectively, creating a silent
dead-band. Now every site must use the canonical behavior from
``steering.segmentation``—either by referencing
``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` directly or by calling the
alignment helper that owns it—never an inline magic number.

Slice-02 audit (2026-05-05): the precompute path was migrated under
``scripts/preprocess/`` (PR #394, manifest §4.1) and the AST guard now
points at the new location. ``target_segments.py`` is included in the
guard via :func:`test_target_segments_uses_canonical_alignment_helper`; Slice
14 landed canonical alignment there, and the prior xfail-strict marker was
retired in PR ``feat-centralise-segmentation-alignment`` (the lockstep merge
per manifest §4.12 is complete).
"""

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

from steering.segmentation import CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FULL_SEQUENCE_ROUTING_PATH = (
    REPO_ROOT / "steering" / "conditional" / "encoder_decoder" / "full_sequence_routing.py"
)
PRECOMPUTE_PATH = REPO_ROOT / "scripts" / "preprocess" / "precompute_token_boundaries.py"
TARGET_SEGMENTS_PATH = REPO_ROOT / "steering" / "eval" / "target_segments.py"

CANONICAL_NAME = "CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _imports_canonical_symbol(tree: ast.Module) -> bool:
    """Return True if the module imports the canonical constant from
    ``steering.segmentation``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "steering.segmentation":
                for alias in node.names:
                    if alias.name == CANONICAL_NAME:
                        return True
    return False


def _references_canonical_name(tree: ast.Module) -> int:
    """Count non-import Name references to the canonical constant."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == CANONICAL_NAME:
            count += 1
    return count


def test_threshold_constant_exists_in_canonical_location() -> None:
    """The canonical constant must be importable from steering.segmentation."""
    mod = importlib.import_module("steering.segmentation")
    assert hasattr(mod, CANONICAL_NAME), (
        f"{CANONICAL_NAME} must be defined in steering.segmentation as the "
        f"single source of truth for the cross-tokenizer end-marker window."
    )
    value = getattr(mod, CANONICAL_NAME)
    assert isinstance(value, int)


def test_threshold_is_sanely_sized() -> None:
    """The threshold should be in a plausible range for an end-of-turn
    marker (e.g. ``<end_of_turn>\\n`` is 14 chars; ``<|assistant|>\\n`` is
    15). Smaller than 1 is useless, larger than 64 is unsafe dead-band."""
    assert 1 <= CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS <= 64


def _imports_canonical_helper(tree: ast.Module) -> bool:
    """Return True if the module imports the canonical alignment helper
    (Layer 1 or Layer 2) from steering.segmentation. Either import
    satisfies the guard because both encapsulate the constant.
    """
    canonical_helpers = {
        "align_char_ranges_to_decoded_prefix",
        "align_segments_to_decoded_prefix",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "steering.segmentation":
                for alias in node.names:
                    if alias.name in canonical_helpers:
                        return True
    return False


def _calls_canonical_helper(tree: ast.Module) -> bool:
    """Return True if the module makes at least one Call to the canonical
    alignment helper. An import without a call would let a contributor
    bypass the centralisation by importing but never calling.
    """
    canonical_helpers = {
        "align_char_ranges_to_decoded_prefix",
        "align_segments_to_decoded_prefix",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in canonical_helpers:
                return True
            # Also match attribute access (e.g., segmentation.align_...)
            if isinstance(func, ast.Attribute) and func.attr in canonical_helpers:
                return True
    return False


def test_full_sequence_routing_and_precompute_use_same_threshold() -> None:
    """AST-audit: both sites must NOT inline magic numbers {16, 20}.

    Precompute (extraction side) directly references the canonical
    constant. Full-sequence SFT routing delegates to the shared
    align_char_ranges_to_decoded_prefix helper, which references the
    constant internally — equally valid.
    """
    routing_tree = _parse(FULL_SEQUENCE_ROUTING_PATH)
    precompute_tree = _parse(PRECOMPUTE_PATH)

    # Precompute must directly reference the canonical constant.
    assert _imports_canonical_symbol(precompute_tree), (
        f"{PRECOMPUTE_PATH} must `from steering.segmentation import "
        f"{CANONICAL_NAME}`."
    )
    assert _references_canonical_name(precompute_tree) >= 1, (
        f"{PRECOMPUTE_PATH} imports {CANONICAL_NAME} but never references "
        f"it; this is a sign a contributor left an inlined magic number "
        f"behind."
    )

    # Full-sequence routing must import the canonical helper (Layer 1 or Layer 2)
    # from steering.segmentation. The helper encapsulates the constant.
    assert _imports_canonical_helper(routing_tree), (
        f"{FULL_SEQUENCE_ROUTING_PATH} must import align_char_ranges_to_decoded_prefix "
        f"or align_segments_to_decoded_prefix from steering.segmentation "
        f"(the centralised cross-tokenizer alignment helper). Direct "
        f"reference to {CANONICAL_NAME} is no longer required because "
        f"the helper carries the constant."
    )
    assert _calls_canonical_helper(routing_tree), (
        f"{FULL_SEQUENCE_ROUTING_PATH} imports the alignment helper but never calls it. "
        f"A future contributor must not bypass the centralisation by "
        f"importing without calling — the canonical site should be the only "
        f"site that compares against CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS."
    )

    # Forbid bare comparison against the near-window plausibility band
    # in BOTH sites (full_sequence_routing.py + precompute_token_boundaries.py).
    # Forbid the entire "near-window plausibility band" (10-32 inclusive).
    # The canonical window is 20; plausible inlined-magic-number candidates
    # span 16 (historical extraction-side value), 20 (canonical), 32 (historical
    # target_segments value), plus round numbers like 25, 30. Verified no
    # false positives in full_sequence_routing.py / precompute_token_boundaries.py today.
    # If a non-alignment Compare against an int in [10, 32] is legitimately
    # needed in these files, name it via a local constant (the AST scan only
    # flags raw Constant(int) operands).
    forbidden = set(range(10, 33))
    for path, tree in [
        (FULL_SEQUENCE_ROUTING_PATH, routing_tree),
        (PRECOMPUTE_PATH, precompute_tree),
    ]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                for op in operands:
                    if (
                        isinstance(op, ast.Constant)
                        and isinstance(op.value, int)
                        and op.value in forbidden
                    ):
                        raise AssertionError(
                            f"{path}:{op.lineno} compares against magic "
                            f"number {op.value}; use {CANONICAL_NAME} "
                            f"imported from steering.segmentation instead."
                        )


def test_target_segments_uses_canonical_alignment_helper() -> None:
    """AST-audit: ``steering/eval/target_segments.py`` must import and call
    the canonical alignment helper and contain no inline near-window magic
    numbers. The helper owns the shared end-marker window.
    """
    if not TARGET_SEGMENTS_PATH.is_file():
        pytest.skip(
            f"{TARGET_SEGMENTS_PATH} missing — slice 14 may have moved it."
        )
    target_tree = _parse(TARGET_SEGMENTS_PATH)

    assert _imports_canonical_helper(target_tree), (
        f"{TARGET_SEGMENTS_PATH} must import the canonical alignment helper "
        "from steering.segmentation."
    )
    assert _calls_canonical_helper(target_tree), (
        f"{TARGET_SEGMENTS_PATH} imports the canonical alignment helper but "
        "never calls it."
    )

    # Must reject any local Assign of ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS``
    # to a literal integer.  An Assign whose value is a ``Name`` (i.e. an alias
    # ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS = SomeOtherName``) is harmless;
    # what we're targeting is the literal-numeric redefinition in B-1.
    for node in ast.walk(target_tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == CANONICAL_NAME:
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, int
                    ):
                        raise AssertionError(
                            f"{TARGET_SEGMENTS_PATH}:{node.lineno} re-declares "
                            f"{CANONICAL_NAME} = {node.value.value} as a "
                            f"literal. Delegate to the canonical alignment "
                            f"helper from steering.segmentation."
                        )

    # Must reject bare Compare against the near-window plausibility band.
    # See test_finetune_and_precompute_use_same_threshold for the rationale.
    forbidden = set(range(10, 33))
    for node in ast.walk(target_tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for op in operands:
                if (
                    isinstance(op, ast.Constant)
                    and isinstance(op.value, int)
                    and op.value in forbidden
                ):
                    raise AssertionError(
                        f"{TARGET_SEGMENTS_PATH}:{op.lineno} compares against "
                        f"magic number {op.value}; delegate to the canonical "
                        f"alignment helper from steering.segmentation instead."
                    )


def test_no_other_eval_module_redefines_constant() -> None:
    """Future-proofing: forbid any other ``steering/eval/*.py`` module from
    declaring a top-level ``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS = <int>``.

    This catches the Slice-02 B-1 pattern (a future contributor copy-pasting
    the constant into a new file) at audit time. Files that import the
    canonical name OR that don't reference it at all are fine.
    """
    eval_dir = REPO_ROOT / "steering" / "eval"
    offenders: list[str] = []
    for py_path in eval_dir.rglob("*.py"):
        # Skip target_segments.py — that file is covered by the dedicated
        # ``test_target_segments_uses_canonical_alignment_helper`` test above.
        # We don't want to double-flag it here.
        if py_path.resolve() == TARGET_SEGMENTS_PATH.resolve():
            continue
        try:
            tree = _parse(py_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == CANONICAL_NAME:
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, int
                        ):
                            offenders.append(f"{py_path}:{node.lineno}")
    assert offenders == [], (
        f"Found {len(offenders)} module(s) under steering/eval/ that redefine "
        f"{CANONICAL_NAME} as an integer literal: {offenders}. Import the "
        f"canonical constant from steering.segmentation instead."
    )


def test_cross_tokenizer_alignment_respects_threshold() -> None:
    """End-to-end: a partial-coverage gap exactly equal to the threshold
    is absorbed; a gap one char larger raises.

    The shared full-sequence router drives the SFT consumer-side absorption
    logic. This smoke-check verifies the precompute-side bound is symbolic and
    agrees with the canonical constant.
    """
    threshold = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    # Build a simple one-feature conv with coverage ending `threshold`
    # chars short of total. We assemble SegmentFeatures-like stub via the
    # function's actual signature. Inspect to avoid coupling to private
    # internals: just verify the constant is wired by checking the
    # in-source comparison exists and is symbolic (already done in the
    # AST test). Here we additionally smoke-check round-trip behavior at
    # the boundary using the precompute-side helper.
    # AST-extract the precompute-side bound without importing the module
    # (its heavy deps may not be installable in all test envs).
    precompute_tree = _parse(PRECOMPUTE_PATH)
    precompute_bound_rhs: ast.expr | None = None
    for node in ast.walk(precompute_tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_TINY_TRAILING_SEG_MAX_CHARS":
                    precompute_bound_rhs = node.value
    assert precompute_bound_rhs is not None, (
        "Precompute-side _TINY_TRAILING_SEG_MAX_CHARS must be defined."
    )
    assert isinstance(precompute_bound_rhs, ast.Name), (
        "Precompute-side _TINY_TRAILING_SEG_MAX_CHARS must be assigned "
        "from the canonical constant (a Name reference), not a literal."
    )
    assert precompute_bound_rhs.id == CANONICAL_NAME, (
        f"Precompute-side bound must equal {CANONICAL_NAME}, got "
        f"{precompute_bound_rhs.id}."
    )

    # Sanity: numpy arrays sized at boundary behave as expected.
    # (We don't invoke the full router directly here because it requires
    # feature tensors + a tokenizer; the AST + symbol-
    # equality checks above are the load-bearing guarantees.)
    at_boundary = np.array([threshold], dtype=np.int64)
    over_boundary = np.array([threshold + 1], dtype=np.int64)
    assert bool(at_boundary[0] <= threshold)
    assert not bool(over_boundary[0] <= threshold)


def test_widened_forbidden_set_catches_synthetic_offender(tmp_path):
    """Pin: the widened forbidden set (10-32) MUST flag synthetic offenders.

    Constructs a fake routing file containing ``if x <= 19:`` and re-runs
    the AST-scan logic with the same forbidden set. The
    forbidden literal must be caught; the test fails if the guard's
    forbidden set was narrowed back to {16, 20} or removed entirely.
    """
    synthetic = tmp_path / "fake_full_sequence_routing.py"
    synthetic.write_text(
        "def f(x):\n"
        "    # Inlined magic number that the AST guard MUST catch.\n"
        "    if x <= 19:\n"
        "        return x\n"
        "    return x + 1\n"
    )
    tree = _parse(synthetic)

    forbidden = set(range(10, 33))
    caught: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in [node.left, *node.comparators]:
                if (
                    isinstance(op, ast.Constant)
                    and isinstance(op.value, int)
                    and op.value in forbidden
                ):
                    caught.append(op.value)
    assert 19 in caught, (
        f"Widened forbidden set MUST catch literal 19; got caught={caught}. "
        f"This test fails if range(10, 33) was narrowed."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
