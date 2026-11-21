"""Regression guards for issue #1303's test-only SFT patch lifetime bug."""

from __future__ import annotations

import ast
from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
import torch as t

import scripts.train.conditional_encdec_sft as conditional_encdec_sft
from steering.conditional.encoder_decoder import sft_features
from tests.steering.test_encdec_sft_conditioned_integration import (
    make_lazy_patched_sft_generator,
    patch_load_sft_features,
)


def _is_sft_feature_patch(expression: ast.expr) -> bool:
    """Return whether an expression calls ``patch_load_sft_features``."""

    if not isinstance(expression, ast.Call):
        return False
    function = expression.func
    return (
        isinstance(function, ast.Name)
        and function.id == "patch_load_sft_features"
    ) or (
        isinstance(function, ast.Attribute)
        and function.attr == "patch_load_sft_features"
    )


def _yield_nodes(node: ast.AST) -> list[ast.Yield | ast.YieldFrom]:
    """Return all yield nodes in a context-manager body, including ``yield from``."""

    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Yield | ast.YieldFrom)
    ]


def _assert_no_yield_under_sft_patch(path: Path) -> None:
    """Reject every direct SFT patch context that can suspend while patched."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for context in ast.walk(tree):
        if not isinstance(context, ast.With):
            continue
        for item in context.items:
            if not _is_sft_feature_patch(item.context_expr):
                continue
            yields = _yield_nodes(ast.Module(body=context.body, type_ignores=[]))
            if yields:
                first = yields[0]
                violations.append(
                    f"{path}:{context.lineno}: patch_load_sft_features block "
                    f"contains {type(first).__name__} at line {first.lineno}"
                )
    if violations:
        raise AssertionError("\n".join(violations))


def test_all_sft_feature_patches_release_before_any_yield() -> None:
    """The repository-wide source invariant must hold for every test file."""

    tests_root = Path(__file__).resolve().parents[1]
    for path in sorted(tests_root.rglob("*.py")):
        _assert_no_yield_under_sft_patch(path)


@pytest.mark.parametrize(
    "source",
    [
        """
from tests.steering.test_encdec_sft_conditioned_integration import patch_load_sft_features

def broken():
    with patch_load_sft_features({}):
        yield from make_generator()
""",
        """
from contextlib import nullcontext
from tests.steering.test_encdec_sft_conditioned_integration import patch_load_sft_features

def broken():
    with (patch_load_sft_features({}), nullcontext()):
        yield 1
""",
    ],
)
def test_static_guard_fails_for_mutated_suspended_patch(tmp_path: Path, source: str) -> None:
    """The guard is non-vacuous: both direct and tuple mutations fail."""

    mutated = tmp_path / "mutated_sft_test.py"
    mutated.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError, match=r"patch_load_sft_features block contains"):
        _assert_no_yield_under_sft_patch(mutated)


def _deterministic_batches(
    *,
    seed: int,
    consumed: list[int],
    loader_seen: list[object],
) -> Iterator[tuple[t.Tensor, t.Tensor | None]]:
    """Produce realistic tensor batches while recording generator lifecycle."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    loader_seen.append(conditional_encdec_sft.load_sft_features)
    for batch_index in range(4):
        consumed.append(batch_index)
        yield (
            t.randint(0, 128, (2, 16), generator=generator),
            t.rand((2, 16), generator=generator),
        )


def test_lazy_helper_preserves_exact_batches_and_state_parity() -> None:
    """The isolation fix must not change any produced value or state timing."""

    expected_loader = sft_features.load_sft_features
    candidate_consumed: list[int] = []
    reference_consumed: list[int] = []
    candidate_seen: list[object] = []
    reference_seen: list[object] = []

    candidate = make_lazy_patched_sft_generator(
        {},
        lambda: _deterministic_batches(
            seed=1303,
            consumed=candidate_consumed,
            loader_seen=candidate_seen,
        ),
    )

    def reference() -> Iterator[tuple[t.Tensor, t.Tensor | None]]:
        context = patch_load_sft_features({})
        context.__enter__()
        try:
            yield from _deterministic_batches(
                seed=1303,
                consumed=reference_consumed,
                loader_seen=reference_seen,
            )
        finally:
            context.__exit__(None, None, None)

    baseline = reference()
    assert candidate_consumed == []
    assert reference_consumed == []
    assert conditional_encdec_sft.load_sft_features is expected_loader

    candidate_batches = list(candidate)
    reference_batches = list(baseline)

    assert candidate_seen and candidate_seen[0] is not expected_loader
    assert reference_seen and reference_seen[0] is not expected_loader
    assert candidate_consumed == reference_consumed == [0, 1, 2, 3]
    assert len(candidate_batches) == len(reference_batches) == 4
    for candidate_batch, reference_batch in zip(
        candidate_batches,
        reference_batches,
        strict=True,
    ):
        candidate_inputs, candidate_targets = candidate_batch
        reference_inputs, reference_targets = reference_batch
        assert t.equal(candidate_inputs, reference_inputs)
        if candidate_targets is None or reference_targets is None:
            assert candidate_targets is reference_targets
        else:
            assert t.equal(candidate_targets, reference_targets)
    assert conditional_encdec_sft.load_sft_features is expected_loader


def test_lazy_helper_releases_patch_after_partial_close_without_gc() -> None:
    """Stopping after one batch must restore the loader immediately."""

    expected_loader = sft_features.load_sft_features
    generator = make_lazy_patched_sft_generator(
        {},
        lambda: _deterministic_batches(
            seed=1303,
            consumed=[],
            loader_seen=[],
        ),
    )

    assert isinstance(generator, Generator)
    next(generator)
    assert conditional_encdec_sft.load_sft_features is expected_loader
    generator.close()
    assert conditional_encdec_sft.load_sft_features is expected_loader


def test_lazy_helper_restores_patch_when_first_iteration_raises() -> None:
    """Factory and first-batch failures must not leave a mocked loader behind."""

    expected_loader = sft_features.load_sft_features

    def broken_factory() -> Iterator[tuple[object, object]]:
        raise RuntimeError("synthetic first-batch failure")
        yield (None, None)

    generator = make_lazy_patched_sft_generator({}, broken_factory)
    with pytest.raises(RuntimeError, match="synthetic first-batch failure"):
        next(generator)
    assert conditional_encdec_sft.load_sft_features is expected_loader


def test_lazy_helper_restores_patch_for_empty_generator() -> None:
    """An empty inner iterator exits cleanly and still restores the loader."""

    expected_loader = sft_features.load_sft_features

    def empty_factory() -> Iterator[tuple[object, object]]:
        return iter(())

    assert list(make_lazy_patched_sft_generator({}, empty_factory)) == []
    assert conditional_encdec_sft.load_sft_features is expected_loader
