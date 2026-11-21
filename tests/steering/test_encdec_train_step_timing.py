"""Training-step synchronization/timing contract for issue #876."""

from __future__ import annotations

import ast
import inspect
import textwrap

from scripts.train.conditional_encdec_train import train_one_config


def _assignment_line(tree: ast.AST, name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    assert len(lines) == 1, f"expected one assignment to {name}, found {lines}"
    return lines[0]


def test_device_synchronize_is_inside_step_timer() -> None:
    """Do not re-serialize the async CUDA-staging stream before ``t0``."""
    source = textwrap.dedent(inspect.getsource(train_one_config))
    tree = ast.parse(source)
    synchronize_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "synchronize"
    ]

    t0_line = _assignment_line(tree, "t0")
    t1_line = _assignment_line(tree, "t1")
    optimizer_step_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "optimizer"
        and node.func.attr == "step"
    ]

    assert len(synchronize_lines) == 1, (
        "train_one_config must retain exactly the post-optimizer device synchronize; "
        f"found calls at {synchronize_lines}"
    )
    assert len(optimizer_step_lines) == 1, (
        f"expected one optimizer.step call, found calls at {optimizer_step_lines}"
    )
    assert t0_line < optimizer_step_lines[0] < synchronize_lines[0] < t1_line, (
        "the device synchronize must remain inside dt so staging is not blocked "
        "before the step and all paid step work is reported"
    )
