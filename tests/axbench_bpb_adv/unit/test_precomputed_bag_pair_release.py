"""#1245: the extraction pair must be released before the derive families run.

`build_candidates` loaded Gemma-3-4B + the 65k SAE for the clustered warm-start
precompute and then held them through the feature-GCG derive, alongside the
EncoderDecoderGPT and the d12 decoder. On a 23.51 GiB card that OOMs ~9s into the
derive, inside `cross_attention.forward`.

It is not batch-bound. Cutting `mean_g_steer_batch` 16->4, `max_forward_batch`
8->4 and `device_batch_size` 16->8 moved peak 23.19 -> 23.24 -> 23.42 GiB: three
attempts, no reduction. The binding term is fixed model residency.

The release is safe precisely because every derive family branches on
`if clustered_warm_starts is None` and reads `base_model`/`sae` only in the None
arm -- `build_ours_gcg_bags_{fair,roofline}` and
`build_feature_diffmean_bags_{fair,roofline}`. `test_every_derive_family_guards_pair_use`
pins that invariant against the real sources, so a future family that reads the
pair unconditionally fails here rather than at runtime on a released model.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv import model_optimize
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import _PrecomputedBagSetup

_SRC = Path(__file__).resolve().parents[3] / "steering" / "eval" / "axbench_bpb_adv"

_DERIVE_FAMILIES = (
    ("gcg_bags.py", "build_ours_gcg_bags_fair"),
    ("gcg_bags.py", "build_ours_gcg_bags_roofline"),
    ("feature_diffmean_bags.py", "build_feature_diffmean_bags_fair"),
    ("feature_diffmean_bags.py", "build_feature_diffmean_bags_roofline"),
)


def _function_body(filename: str, func: str) -> str:
    src = (_SRC / filename).read_text()
    start = src.find(f"def {func}(")
    assert start >= 0, f"{func} not found in {filename}"
    end = src.find("\ndef ", start + 10)
    body = src[start : end if end > 0 else len(src)]
    return re.sub(r'""".*?"""', "", body, count=1, flags=re.S)


@pytest.mark.parametrize(("filename", "func"), _DERIVE_FAMILIES)
def test_every_derive_family_guards_pair_use(filename: str, func: str) -> None:
    """The invariant that makes releasing the pair safe.

    Each family must gate its `base_model`/`sae` reads behind
    `if clustered_warm_starts is None`. A family that reads them unconditionally
    would crash on a released (None) model once #1245's release is in place, so
    this must fail at test time instead.
    """
    body = _function_body(filename, func)
    assert "if clustered_warm_starts is None" in body, (
        f"{func} must branch on clustered_warm_starts; without that guard the "
        "#1245 release would hand it a None base_model at runtime"
    )
    guard = body.index("if clustered_warm_starts is None")
    signature_end = body.index(")", body.index("("))
    # Every base_model/sae mention after the signature must sit at or after the
    # guard -- i.e. no unconditional read between the signature and the branch.
    for match in re.finditer(r"\b(base_model|sae)\b", body):
        if match.start() <= signature_end:
            continue  # parameter declarations
        assert match.start() > guard, (
            f"{func} reads {match.group(0)!r} before the clustered_warm_starts "
            "guard; it would be read even when the pair has been released"
        )


def _setup(**overrides: object) -> _PrecomputedBagSetup:
    base = _PrecomputedBagSetup(
        prefixes=[],
        model=object(),
        base_model=object(),
        sae=object(),
        percentile_table=object(),
        context_segments_by_prefix_hash={},
        device="cpu",
    )
    return replace(base, **overrides) if overrides else base


def test_setup_is_frozen_so_release_cannot_mutate_in_place() -> None:
    """`replace()` is required, not attribute assignment.

    If the dataclass ever becomes mutable, an in-place `setup.base_model = None`
    would silently diverge from the copy any caller already holds. Pinning
    frozenness keeps the release a single well-defined rebind.
    """
    setup = _setup()
    with pytest.raises(Exception, match="frozen|immutable|cannot assign"):
        setup.base_model = None  # ty: ignore[invalid-assignment]


def test_replace_drops_only_the_pair() -> None:
    """The release must not disturb anything else the derives depend on."""
    setup = _setup()
    released = replace(setup, base_model=None, sae=None)
    assert released.base_model is None
    assert released.sae is None
    for field_name in (
        "prefixes",
        "model",
        "percentile_table",
        "context_segments_by_prefix_hash",
        "device",
    ):
        assert getattr(released, field_name) is getattr(setup, field_name), (
            f"{field_name} must survive the release untouched"
        )


def test_released_pair_without_warm_starts_raises_naming_both_facts() -> None:
    """Fail loud, not deep.

    Reaching a derive with a released pair AND no warm starts is a programming
    error. Without this guard the derive fails somewhere inside on a None model,
    far from the cause.
    """
    with pytest.raises(AxBenchBpbAdvError, match="requires the base extraction pair"):
        model_optimize._build_model_backed_precomputed_bags(
            cfg=_DummyCfg(),  # ty: ignore[invalid-argument-type]
            paths=object(),  # ty: ignore[invalid-argument-type]
            runtime_state=_DummyState(),  # ty: ignore[invalid-argument-type]
            tokenizer=object(),
            modes={SteeringMode.POSITIVE},
            derive=lambda **_: pytest.fail("derive must not be reached"),
            label="fair feature-GCG",
            shared=_setup(base_model=None, sae=None, prefixes=[_DummyPrefix()]),
            clustered_warm_starts=None,
        )


def test_release_guard_message_states_which_half_is_missing() -> None:
    """A half-released pair must be distinguishable in the error text."""
    with pytest.raises(AxBenchBpbAdvError, match=r"base_model=None, sae=set"):
        model_optimize._build_model_backed_precomputed_bags(
            cfg=_DummyCfg(),  # ty: ignore[invalid-argument-type]
            paths=object(),  # ty: ignore[invalid-argument-type]
            runtime_state=_DummyState(),  # ty: ignore[invalid-argument-type]
            tokenizer=object(),
            modes={SteeringMode.POSITIVE},
            derive=lambda **_: pytest.fail("derive must not be reached"),
            label="fair feature-GCG",
            shared=_setup(base_model=None, prefixes=[_DummyPrefix()]),
            clustered_warm_starts=None,
        )


def test_warm_starts_present_bypasses_the_guard_entirely() -> None:
    """With warm starts, a released pair is the EXPECTED state, not an error."""
    seen: dict[str, object] = {}

    def _derive(**kwargs: object) -> None:
        seen.update(kwargs)

    model_optimize._build_model_backed_precomputed_bags(
        cfg=_DummyCfg(),  # ty: ignore[invalid-argument-type]
        paths=object(),  # ty: ignore[invalid-argument-type]
        runtime_state=_DummyState(),  # ty: ignore[invalid-argument-type]
        tokenizer=object(),
        modes={SteeringMode.POSITIVE},
        derive=_derive,
        label="fair feature-GCG",
        shared=_setup(base_model=None, sae=None, prefixes=[_DummyPrefix()]),
        clustered_warm_starts=object(),  # ty: ignore[invalid-argument-type]
    )
    assert seen, "derive must run when warm starts are supplied"
    assert seen["base_model"] is None and seen["sae"] is None, (
        "the released pair is passed through as None; the derive's warm-start "
        "branch must not read it"
    )
    assert "clustered_warm_starts" in seen


class _DummyPrefix:
    mode = "positive"


class _DummyCfg:
    source_run_dir = "/nonexistent"


class _DummyState:
    cluster_contract = None


# --- the release SITE ------------------------------------------------------
#
# The guard above makes a mis-gated release loud rather than silent, but loud
# still means "every unclustered run dies". These pin the site itself.
#
# They are AST-based, not textual: the first version of this fix placed the
# release at the same indent as `if cluster_contract is not None:` -- a sibling,
# not a child -- so it fired on the unclustered path too, where no warm starts
# exist. A `grep` for the two statements passed. Only the nesting was wrong.

_MODEL_OPTIMIZE = _SRC / "model_optimize.py"


def _parsed() -> tuple[ast.Module, dict[ast.AST, ast.AST]]:
    tree = ast.parse(_MODEL_OPTIMIZE.read_text())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    return tree, parents


def _find_call(tree: ast.Module, attr: str) -> ast.Call:
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]
    assert len(found) == 1, f"expected exactly one {attr}() call, found {len(found)}"
    return found[0]


def _enclosing_if_tests(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    """Tests of every `if` whose BODY (not `else`) contains ``node``."""
    tests: list[str] = []
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.If):
            in_body = any(current is stmt or current in set(ast.walk(stmt)) for stmt in parent.body)
            if in_body:
                tests.append(ast.unparse(parent.test))
        current = parent
    return tests


def test_release_is_gated_on_the_warm_start_cache_existing() -> None:
    """The bug this test exists for: an UNGATED release.

    Releasing the pair is only sound once the warm-start cache exists. Ungated,
    the unclustered path -- which never builds a cache and whose derives read
    base_model/sae directly -- would hit the guard above and abort every run.
    """
    tree, parents = _parsed()
    tests = _enclosing_if_tests(_find_call(tree, "release_base_gemma_and_sae"), parents)
    assert any(
        "clustered_warm_starts is not None" in test or "cluster_contract is not None" in test
        for test in tests
    ), (
        "release_base_gemma_and_sae() must sit inside a branch conditioned on the "
        "clustered warm-start cache existing; enclosing if-tests were "
        f"{tests!r}. Ungated, this fires on the unclustered path where the "
        "derives still need the pair."
    )


def test_setup_pair_null_and_state_release_share_one_gate() -> None:
    """Both halves, or neither.

    Nulling the frozen setup's fields leaves RuntimeState's cache holding the
    models; calling release_base_gemma_and_sae() leaves the setup holding them.
    Either half alone frees nothing, so they must not drift onto separate gates.
    """
    tree, parents = _parsed()
    release_gates = _enclosing_if_tests(_find_call(tree, "release_base_gemma_and_sae"), parents)
    replaces = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replace"
        and any(kw.arg == "base_model" for kw in node.keywords)
    ]
    assert len(replaces) == 1, "expected exactly one pair-nulling replace()"
    null_pair = replaces[0]
    assert {kw.arg for kw in null_pair.keywords} == {"base_model", "sae"}, (
        "the release must drop exactly the pair -- both halves, nothing else"
    )
    assert all(
        isinstance(kw.value, ast.Constant) and kw.value.value is None for kw in null_pair.keywords
    )
    assert _enclosing_if_tests(null_pair, parents) == release_gates, (
        "the setup-level null and the state-level release must share one gate"
    )


def test_release_happens_after_the_cache_is_built_not_before() -> None:
    """Ordering: the cache is derived FROM the pair, so it must exist first."""
    tree, _ = _parsed()
    build = _find_call(tree, "release_base_gemma_and_sae")
    cache_builds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_clustered_warm_start_cache"
    ]
    assert len(cache_builds) == 1
    assert cache_builds[0].lineno < build.lineno, (
        "the pair is released before the warm-start cache that consumes it is "
        "built; the cache derive would run on a released model"
    )


# --- does the release actually FREE anything? ------------------------------
#
# Everything above pins code shape. None of it proves the models die. They
# die only if the release drops the LAST strong reference -- and an
# nn.Module graph routinely holds itself in a reference cycle (a forward hook
# whose closure captures its module; the nnterp wrapper hung off
# `model.nnterp_model`). Cycles survive refcounting, so their CUDA blocks are
# still allocated when `empty_cache()` runs, and `empty_cache()` only returns
# blocks that are already free: the release would free nothing and #1245 would
# reproduce with the fix in place.


class _CyclicModel:
    """Stands in for a module graph that reaches itself.

    Shape, not size: one self-reference through a captured closure, which is
    what a registered forward hook creates.
    """

    def __init__(self) -> None:
        self.hook = lambda: self  # closure captures self -> cycle
        self.nnterp_model = self  # the side-channel wrapper, same shape


def _state_holding(model: object, sae: object) -> RuntimeState:
    state = RuntimeState.__new__(RuntimeState)
    state._base_gemma = model
    state._base_sae = sae
    state._base_gemma_tail_offloaded_after = 22
    return state


def test_release_frees_models_held_in_a_reference_cycle() -> None:
    """The regression #1245's fix silently depends on.

    Without a collection in the release path this fails: both weakrefs stay
    alive, meaning ~10 GiB of Gemma + SAE would still be resident when the
    feature-GCG derive starts -- the exact OOM the fix exists to prevent.
    """
    import gc
    import weakref

    model, sae = _CyclicModel(), _CyclicModel()
    model_ref, sae_ref = weakref.ref(model), weakref.ref(sae)
    state = _state_holding(model, sae)

    # Drop every reference this test holds; the state is now the sole owner.
    del model, sae
    gc.collect()
    assert model_ref() is not None, "precondition: the state still owns the pair"

    state.release_base_gemma_and_sae()

    assert model_ref() is None, (
        "base model survived release_base_gemma_and_sae(); it is held in a "
        "reference cycle, so nulling the attribute cannot reclaim it and "
        "empty_cache() has no free blocks to return"
    )
    assert sae_ref() is None, "SAE survived release_base_gemma_and_sae()"


def test_release_leaves_no_reference_on_the_setup_side_either() -> None:
    """Both owners, one collection.

    The state-side release cannot free a model the frozen setup still points
    at. This is the same both-halves invariant as the structural test, proved
    by liveness rather than by code shape.
    """
    import gc
    import weakref

    model, sae = _CyclicModel(), _CyclicModel()
    model_ref = weakref.ref(model)
    state = _state_holding(model, sae)
    setup = _setup(base_model=model, sae=sae)
    del model, sae

    state.release_base_gemma_and_sae()
    gc.collect()
    assert model_ref() is not None, (
        "precondition: the setup is the second owner and still holds it"
    )

    setup = replace(setup, base_model=None, sae=None)
    gc.collect()
    assert model_ref() is None, (
        "with both owners released the pair must be reclaimable; something "
        "else is still holding the extraction models"
    )
    assert setup.base_model is None
