"""Tests for ``scripts/train/chat_rl.py``'s rollout seed.

The map flagged this as P1 #6: the original ``hash((step, example_idx,
sampling_step)) & 0x7FFFFFFF`` was non-deterministic across processes
(PYTHONHASHSEED randomization), breaking RL run reproducibility.

The fix replaces ``hash()`` with an explicit linear combination of the
three index variables. We pin the determinism contract here because
the seed flows directly into the engine sampler and downstream
gradients — silent regression to ``hash()`` would silently make every
RL run irreproducible again.

We exercise the seed function in two ways:

1. Run it inside a subprocess with ``PYTHONHASHSEED=random`` set, and
   assert the output is identical to running it in this process. A
   regression to ``hash(...)`` would make the subprocess produce
   different values from the parent.
2. Run it inside a subprocess with a specific ``PYTHONHASHSEED=0`` and
   compare against another subprocess with ``PYTHONHASHSEED=12345`` —
   they must agree.

Both subprocess flavors import the helper and skip the rest of
``chat_rl.py``'s top-level (Hydra compose, model load, etc.). The
helper module is small enough that we use ``importlib`` to grab it
without paying the rest of the module's startup cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _import_seed_helper_inline():
    """Import ``_compute_rollout_seed`` from chat_rl WITHOUT importing the
    full module (which would trigger Hydra compose and model load).

    Strategy: read the source, extract just the helper definition,
    exec into a fresh namespace.
    """
    chat_rl_src = (REPO_ROOT / "scripts" / "train" / "chat_rl.py").read_text()
    # Find the helper. It's a top-level def with a known signature.
    import ast

    tree = ast.parse(chat_rl_src)
    helper_src: str | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_rollout_seed":
            helper_src = ast.get_source_segment(chat_rl_src, node)
            break
    if helper_src is None:
        raise AssertionError(
            "_compute_rollout_seed not found in scripts/train/chat_rl.py — "
            "regression check: slice 11 audit P1 #6"
        )
    ns: dict = {}
    exec(helper_src, ns)
    return ns["_compute_rollout_seed"]


_compute_rollout_seed = _import_seed_helper_inline()


def test_returns_non_negative_int31() -> None:
    """Seed must fit in non-negative int32 (Engine.generate_batch contract)."""
    for step, ex, ss in [(0, 0, 0), (10, 5, 3), (1_000_000, 1_000, 7)]:
        seed = _compute_rollout_seed(step, ex, ss)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31


def test_different_inputs_yield_different_seeds() -> None:
    """At least within the index ranges chat_rl uses, the helper must
    produce distinct seeds for distinct ``(step, example_idx, sampling_step)``
    triples. This is the whole point of mixing three primes."""
    seeds = set()
    for step in range(50):
        for ex in range(20):
            for ss in range(8):
                seeds.add(_compute_rollout_seed(step, ex, ss))
    # In a 50*20*8 = 8000 grid, collisions in a 2^31 space are
    # astronomically unlikely. We allow up to 4 (extremely loose).
    assert len(seeds) >= 8000 - 4, f"Too many collisions: got {len(seeds)} unique seeds"


def test_same_inputs_yield_same_seed_in_process() -> None:
    """Trivial in-process determinism guard."""
    a = _compute_rollout_seed(7, 11, 3)
    b = _compute_rollout_seed(7, 11, 3)
    assert a == b


SUBPROC_PROBE = textwrap.dedent(
    """
    import ast, sys, os
    src_path = sys.argv[1]
    src = open(src_path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_compute_rollout_seed"
        ):
            helper_src = ast.get_source_segment(src, node)
            break
    else:
        sys.exit("helper not found")
    ns = {}
    exec(helper_src, ns)
    fn = ns["_compute_rollout_seed"]
    print(fn(7, 11, 3))
    """
).strip()


def _run_with_pythonhashseed(seed: str) -> int:
    chat_rl_path = REPO_ROOT / "scripts" / "train" / "chat_rl.py"
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.check_output(
        [sys.executable, "-c", SUBPROC_PROBE, str(chat_rl_path)],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode().strip()
    return int(out)


def test_deterministic_across_pythonhashseed_settings() -> None:
    """The KEY contract: the seed must NOT depend on PYTHONHASHSEED.

    Run the helper in two subprocesses with different PYTHONHASHSEED
    values. A regression to ``hash(...)`` would make their outputs
    diverge.
    """
    a = _run_with_pythonhashseed("0")
    b = _run_with_pythonhashseed("12345")
    c = _run_with_pythonhashseed("random")
    assert a == b == c, (
        f"PYTHONHASHSEED-dependent seed detected: a={a}, b={b}, c={c}. "
        "Slice 11 audit P1 #6 regressed: ``hash()`` is non-deterministic "
        "across processes by default."
    )


@pytest.mark.parametrize(
    "step,ex,ss,expected",
    [
        (0, 0, 0, 0),
        (1, 0, 0, 1_000_003),
        (0, 1, 0, 1_009),
        (0, 0, 1, 1),
        (1, 1, 1, 1_000_003 + 1_009 + 1),
    ],
)
def test_known_seed_values(step: int, ex: int, ss: int, expected: int) -> None:
    """Pin specific (step, ex, ss) -> seed mappings so future
    refactoring of the linear combination would surface as a test
    diff rather than silently changing every RL run's rollouts."""
    assert _compute_rollout_seed(step, ex, ss) == expected
