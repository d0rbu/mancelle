"""Self-tests for the shared equivalence harness (tests/steering/feature_gcg/_equivalence.py).

These verify the harness itself is correct:

* ``differential`` PASSES on identical inputs and RAISES (with the first
  differing flat index in the message) on differing inputs.
* ``feature_grads_equal`` exact-mode RAISES on a 1-ULP bf16 difference.
* ``feature_grads_equal`` allclose-mode passes / raises as expected.

All tests run on CPU with hand-built tensors — no model required, no ``slow``
mark, and no GPU dependency.
"""

from __future__ import annotations

import pytest
import torch

from steering.conditional.feature_gcg.gradient import FeatureGrad
from tests.steering.feature_gcg._equivalence import (
    _DEFAULT_APPROX_ATOL,
    differential,
    feature_grads_equal,
)
from tests.steering.feature_gcg.conftest import (
    SLOW_GATED_FIXTURES,
    pytest_collection_modifyitems,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fg(val: float, dtype: torch.dtype = torch.float32, shape: tuple[int, ...] = (2, 4)) -> FeatureGrad:
    """FeatureGrad with all four fields filled with ``val``."""
    t = torch.full(shape, val, dtype=dtype)
    return FeatureGrad(
        id_grad=t.clone(),
        bucket_grad=t.clone(),
        id_weight=t.clone(),
        bucket_weight=t.clone(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# differential — identical inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_differential_passes_on_identical_tensor() -> None:
    """differential does not raise when both functions return the same tensor."""
    t = torch.tensor([1.0, 2.0, 3.0])
    differential(lambda: t, lambda: t.clone())


def test_differential_passes_on_identical_feature_grad() -> None:
    """differential does not raise when both return an identical FeatureGrad."""
    differential(lambda: _fg(1.0), lambda: _fg(1.0))


def test_differential_passes_on_identical_list_of_grads() -> None:
    """differential does not raise when both return identical lists of FeatureGrad."""
    ref = [_fg(0.5), _fg(1.5)]
    cand = [_fg(0.5), _fg(1.5)]
    differential(lambda: ref, lambda: cand)


def test_differential_raises_on_differing_list_element_field() -> None:
    """differential on two list[FeatureGrad] differing in element 1's id_grad RAISES.

    T3-T6 compare lists of per-member gradients, so the oracle must bite inside a
    list and name the index-prefixed location ``[1].id_grad`` (element 1, field
    id_grad) in the message.
    """
    base = torch.zeros(2, 4)
    elem1_diff = base.clone()
    elem1_diff[0, 1] = 7.0  # element 1's id_grad differs from the reference
    ref = [_fg(0.0), _fg(0.0)]
    cand = [
        _fg(0.0),
        FeatureGrad(
            id_grad=elem1_diff,
            bucket_grad=base.clone(),
            id_weight=base.clone(),
            bucket_weight=base.clone(),
        ),
    ]
    with pytest.raises(AssertionError, match=r"\[1\]\.id_grad"):
        differential(lambda: ref, lambda: cand)


# ─────────────────────────────────────────────────────────────────────────────
# differential — differing inputs: raises with index in message
# ─────────────────────────────────────────────────────────────────────────────


def test_differential_raises_on_differing_tensor() -> None:
    """differential raises AssertionError when tensors differ."""
    ref = torch.zeros(5)
    cand = torch.zeros(5)
    cand[3] = 1.0
    with pytest.raises(AssertionError):
        differential(lambda: ref, lambda: cand)


def test_differential_message_contains_first_flat_index() -> None:
    """The AssertionError message names the first differing flat index (here 7)."""
    ref = torch.zeros(10)
    cand = ref.clone()
    cand[7] = 99.0  # only position 7 differs → flat index 7
    with pytest.raises(AssertionError, match=r"\b7\b"):
        differential(lambda: ref, lambda: cand)


def test_differential_first_index_not_last_index() -> None:
    """Only the FIRST differing index is reported (positions 2 and 8 differ; 2 wins)."""
    ref = torch.zeros(10)
    cand = ref.clone()
    cand[2] = 5.0
    cand[8] = 9.0
    err = None
    try:
        differential(lambda: ref, lambda: cand)
    except AssertionError as e:
        err = str(e)
    assert err is not None, "expected AssertionError was not raised"
    assert "index=2" in err, f"expected 'index=2' in message, got: {err}"


def test_differential_raises_on_differing_feature_grad_id_grad() -> None:
    """differential raises when FeatureGrad.id_grad differs and names the field."""
    a = _fg(0.0)
    t = torch.zeros(2, 4)
    t_diff = t.clone()
    t_diff[0, 2] = 1.0
    b = FeatureGrad(
        id_grad=t_diff,
        bucket_grad=t.clone(),
        id_weight=t.clone(),
        bucket_weight=t.clone(),
    )
    with pytest.raises(AssertionError, match="id_grad"):
        differential(lambda: a, lambda: b)


def test_differential_custom_tensors_eq_passes_near_equal() -> None:
    """A loose tensors_eq lets differential pass on near-equal tensors."""
    ref = torch.tensor([1.0])
    cand = torch.tensor([1.001])
    # Exact equality would fail.
    with pytest.raises(AssertionError):
        differential(lambda: ref, lambda: cand)
    # Allclose with generous atol passes.
    differential(
        lambda: ref,
        lambda: cand,
        tensors_eq=lambda a, b: torch.allclose(a, b, atol=0.01),
    )


def test_differential_same_position_nan_raises_with_coherent_index() -> None:
    """Two tensors identical except a same-position NaN RAISE with a coherent index.

    ``torch.equal`` treats NaN as unequal (NaN != NaN), so two tensors that hold
    NaN at the SAME position (and are otherwise identical) are reported as
    differing.  The mismatch message must name a real, non-negative flat index
    (here 1) — not the incoherent ``index=-1`` the old same-position-NaN-as-equal
    logic produced.
    """
    ref = torch.tensor([1.0, float("nan"), 3.0])
    cand = torch.tensor([1.0, float("nan"), 3.0])
    # Sanity: the default exact predicate genuinely fires on this pair.
    assert not torch.equal(ref, cand), "bf NaN sanity: torch.equal should be False"
    err = None
    try:
        differential(lambda: ref, lambda: cand)
    except AssertionError as e:
        err = str(e)
    assert err is not None, "expected AssertionError was not raised"
    assert "index=1" in err, f"expected coherent 'index=1' in message, got: {err}"
    assert "index=-1" not in err, f"incoherent negative index leaked: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# feature_grads_equal — exact mode
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_grads_equal_exact_passes_identical() -> None:
    """exact mode does not raise on two identical FeatureGrad objects."""
    feature_grads_equal(_fg(1.0), _fg(1.0), exact=True)


def test_feature_grads_equal_exact_raises_on_1ulp_bf16() -> None:
    """exact mode RAISES on a 1-ULP bf16 difference in id_grad.

    bf16 epsilon ≈ 2^-7 ≈ 0.0078125; ``torch.equal(1.0_bf16, 1.0078125_bf16)``
    is False, so this proves the oracle catches even the smallest representable
    difference in bf16.
    """
    ulp = torch.finfo(torch.bfloat16).eps  # ≈ 0.0078125
    base = torch.full((2, 2), 1.0, dtype=torch.bfloat16)
    perturbed = torch.tensor(
        [[1.0 + ulp, 1.0], [1.0, 1.0]], dtype=torch.bfloat16
    )
    # Verify the 1-ULP difference is representable (sanity check on the test).
    assert not torch.equal(base[0, 0:1], perturbed[0, 0:1]), (
        "bf16 1-ULP sanity check failed: the two values are equal in bf16"
    )

    a = FeatureGrad(
        id_grad=perturbed,
        bucket_grad=base.clone(),
        id_weight=base.clone(),
        bucket_weight=base.clone(),
    )
    b = FeatureGrad(
        id_grad=base.clone(),
        bucket_grad=base.clone(),
        id_weight=base.clone(),
        bucket_weight=base.clone(),
    )
    with pytest.raises(AssertionError, match="id_grad"):
        feature_grads_equal(a, b, exact=True)


def test_feature_grads_equal_exact_raises_includes_flat_index() -> None:
    """The AssertionError from exact mode includes the flat index of the first diff.

    A 3×4 tensor with first diff at (0, 2) has flat index 2.
    """
    t = torch.zeros(3, 4)
    t_diff = t.clone()
    t_diff[0, 2] = 99.0  # flat index 2 in a 3×4 = 12-element tensor
    a = FeatureGrad(
        id_grad=t_diff,
        bucket_grad=t.clone(),
        id_weight=t.clone(),
        bucket_weight=t.clone(),
    )
    b = FeatureGrad(
        id_grad=t.clone(),
        bucket_grad=t.clone(),
        id_weight=t.clone(),
        bucket_weight=t.clone(),
    )
    with pytest.raises(AssertionError, match=r"\b2\b"):
        feature_grads_equal(a, b, exact=True)


def test_feature_grads_equal_exact_bucket_grad_field() -> None:
    """exact mode also catches differences in bucket_grad and names the field."""
    base = torch.zeros(2, 4)
    diff = base.clone()
    diff[1, 3] = 7.0
    a = FeatureGrad(id_grad=base.clone(), bucket_grad=diff, id_weight=base.clone(), bucket_weight=base.clone())
    b = FeatureGrad(id_grad=base.clone(), bucket_grad=base.clone(), id_weight=base.clone(), bucket_weight=base.clone())
    with pytest.raises(AssertionError, match="bucket_grad"):
        feature_grads_equal(a, b, exact=True)


# ─────────────────────────────────────────────────────────────────────────────
# feature_grads_equal — allclose mode
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_grads_equal_allclose_passes_within_atol() -> None:
    """allclose mode passes when all diffs are within atol."""
    a = _fg(1.0)
    b = FeatureGrad(
        id_grad=torch.full((2, 4), 1.0 + 5e-6),
        bucket_grad=torch.full((2, 4), 1.0),
        id_weight=torch.full((2, 4), 1.0),
        bucket_weight=torch.full((2, 4), 1.0),
    )
    feature_grads_equal(a, b, exact=False, atol=1e-5)


def test_feature_grads_equal_allclose_raises_when_exceeding_atol() -> None:
    """allclose mode raises when any diff exceeds atol."""
    a = _fg(1.0)
    b = _fg(2.0)
    with pytest.raises(AssertionError):
        feature_grads_equal(a, b, exact=False, atol=1e-5)


def test_feature_grads_equal_allclose_raises_names_field() -> None:
    """allclose mode raises naming the specific differing field."""
    base = torch.full((2, 4), 1.0)
    a = FeatureGrad(
        id_grad=base.clone(),
        bucket_grad=base.clone(),
        id_weight=base.clone(),
        bucket_weight=torch.full((2, 4), 100.0),  # only bucket_weight differs
    )
    b = _fg(1.0)
    with pytest.raises(AssertionError, match="bucket_weight"):
        feature_grads_equal(a, b, exact=False, atol=1e-5)


def test_feature_grads_equal_default_atol_is_pinned() -> None:
    """The module default atol (used when ``exact=False`` with NO ``atol=`` kwarg).

    Exercises the default path (no ``atol=`` supplied) and pins the constant:
    a diff just below the default passes, a diff just above it raises, and the
    exported constant equals 1e-5.  Any future change to ``_DEFAULT_APPROX_ATOL``
    that crosses these bounds is caught.
    """
    # The constant itself is pinned (catches ANY change to the default value).
    assert _DEFAULT_APPROX_ATOL == 1e-5

    base = torch.zeros(2, 4)
    # Diff 9e-6 < default 1e-5 → passes when called with NO atol= kwarg.
    near = FeatureGrad(
        id_grad=torch.full((2, 4), 9e-6),
        bucket_grad=base.clone(),
        id_weight=base.clone(),
        bucket_weight=base.clone(),
    )
    feature_grads_equal(_fg(0.0), near, exact=False)  # uses module default atol

    # Diff 1.1e-5 > default 1e-5 → raises when called with NO atol= kwarg.
    far = FeatureGrad(
        id_grad=torch.full((2, 4), 1.1e-5),
        bucket_grad=base.clone(),
        id_weight=base.clone(),
        bucket_weight=base.clone(),
    )
    with pytest.raises(AssertionError):
        feature_grads_equal(_fg(0.0), far, exact=False)  # uses module default atol


def test_feature_grads_equal_exact_is_default() -> None:
    """feature_grads_equal defaults to exact mode (no explicit exact= needed)."""
    a = _fg(1.0)
    b = _fg(1.0 + 1e-7)  # very small diff — passes allclose but not exact
    with pytest.raises(AssertionError):
        feature_grads_equal(a, b)  # default exact=True


# ─────────────────────────────────────────────────────────────────────────────
# A1: the loss path (float / list[float]) is ACTUALLY compared, not vacuous
# ─────────────────────────────────────────────────────────────────────────────


def test_differential_scalar_float_mismatch_raises() -> None:
    """A1: ``differential`` on a float-returning pair RAISES on a value mismatch.

    Before the scalar branch in ``_extract_tensors``, ``differential(lambda: 1.0,
    lambda: 999.0)`` extracted ZERO tensors and PASSED vacuously — silently
    blessing the entire loss path (``losses`` / ``reference_masked_loss`` /
    ``reference_g_steer`` all return ``float``).
    """
    with pytest.raises(AssertionError, match="differs|index"):
        differential(lambda: 1.0, lambda: 999.0)


def test_differential_scalar_float_match_passes() -> None:
    """A1: equal floats compare equal (the loss path is compared, and passes when equal)."""
    differential(lambda: 4.6052, lambda: 4.6052)


def test_differential_list_float_mismatch_raises() -> None:
    """A1: a ``list[float]`` (the per-member ``losses`` shape) RAISES on a mismatch."""
    with pytest.raises(AssertionError):
        differential(lambda: [1.0, 2.0], lambda: [9.9, 8.8])


def test_differential_list_float_match_passes() -> None:
    """A1: equal ``list[float]`` results compare equal."""
    differential(lambda: [1.0, 2.0, 3.0], lambda: [1.0, 2.0, 3.0])


def test_differential_nonnull_zero_tensor_payload_raises() -> None:
    """A1: a NON-None payload yielding zero tensors (e.g. a dict) RAISES (not vacuous)."""
    with pytest.raises(AssertionError, match="zero tensors"):
        differential(lambda: {"loss": 1.0}, lambda: {"loss": 1.0})


def test_differential_none_results_are_exempt() -> None:
    """A1: ``None`` is an explicit 'no result' and is exempt from the vacuity guard."""
    differential(lambda: None, lambda: None)


# ─────────────────────────────────────────────────────────────────────────────
# A4: differential's own error branches (count + shape) are self-tested
# ─────────────────────────────────────────────────────────────────────────────


def test_differential_count_mismatch_fewer_raises() -> None:
    """A4: a candidate returning FEWER tensors than the reference RAISES."""
    with pytest.raises(AssertionError, match="returned"):
        differential(
            lambda: [torch.zeros(3), torch.zeros(3)],
            lambda: [torch.zeros(3)],
        )


def test_differential_count_mismatch_more_raises() -> None:
    """A4: a candidate returning MORE tensors than the reference RAISES."""
    with pytest.raises(AssertionError, match="returned"):
        differential(
            lambda: [torch.zeros(3)],
            lambda: [torch.zeros(3), torch.zeros(3)],
        )


def test_differential_shape_mismatch_under_torch_equal_raises() -> None:
    """A4: a non-broadcastable shape divergence RAISES under the default ``torch.equal``."""
    with pytest.raises(AssertionError, match="shape mismatch"):
        differential(lambda: torch.zeros(2, 4), lambda: torch.zeros(2, 5))


def test_differential_broadcastable_shape_under_allclose_raises() -> None:
    """A4 (+A2): a BROADCASTABLE shape divergence is rejected even under ``allclose``.

    ``torch.allclose(ones(10), ones(10, 1))`` BROADCASTS to (10, 10) all-ones and
    returns True — so without the unconditional shape guard (checked BEFORE the
    predicate) this divergence would pass vacuously under an allclose ``tensors_eq``.
    """
    ref = torch.ones(10)
    cand = torch.ones(10, 1)
    # Sanity: the allclose predicate alone WOULD broadcast-pass (proves the teeth
    # come from the shape guard, not the predicate).
    assert torch.allclose(ref, cand, atol=1e-3)
    with pytest.raises(AssertionError, match="shape mismatch"):
        differential(
            lambda: ref,
            lambda: cand,
            tensors_eq=lambda x, y: torch.allclose(x, y, atol=1e-3),
        )


# ─────────────────────────────────────────────────────────────────────────────
# A2: feature_grads_equal rejects a broadcastable shape / wrong dtype (both modes)
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_grads_equal_approx_rejects_broadcastable_shape() -> None:
    """A2: a (1, 4) vs (2, 4) id_grad is rejected in APPROX mode (allclose broadcasts).

    The sole gate for the batched FeatureGrad is ``feature_grads_equal(exact=False)``.
    ``torch.allclose(full((2,4),1).float(), full((1,4),1).float())`` broadcasts and
    returns True — a collapsed/duplicated slot would slip through without the shape
    guard.
    """
    a = _fg(1.0, shape=(2, 4))
    b = FeatureGrad(
        id_grad=torch.full((1, 4), 1.0),  # broadcastable to (2, 4), same value
        bucket_grad=torch.full((2, 4), 1.0),
        id_weight=torch.full((2, 4), 1.0),
        bucket_weight=torch.full((2, 4), 1.0),
    )
    # Sanity: the bare allclose WOULD broadcast-pass.
    assert torch.allclose(a.id_grad.float(), b.id_grad.float(), atol=1e-5)
    with pytest.raises(AssertionError, match="shape mismatch"):
        feature_grads_equal(a, b, exact=False, atol=1e-5)


def test_feature_grads_equal_exact_rejects_broadcastable_shape() -> None:
    """A2: a broadcastable shape divergence is rejected in EXACT mode too."""
    a = _fg(0.0, shape=(3, 2))
    b = FeatureGrad(
        id_grad=torch.zeros(1, 2),  # broadcastable to (3, 2)
        bucket_grad=torch.zeros(3, 2),
        id_weight=torch.zeros(3, 2),
        bucket_weight=torch.zeros(3, 2),
    )
    with pytest.raises(AssertionError, match="shape mismatch"):
        feature_grads_equal(a, b, exact=True)


def test_feature_grads_equal_rejects_dtype_mismatch_both_modes() -> None:
    """A2: a same-shape, same-value float32-vs-bf16 id_grad is rejected (both modes).

    ``.float()`` in approx mode would erase the storage dtype and let a bf16 slot
    masquerade as float32; the explicit dtype guard rejects it.
    """
    a = _fg(1.0, dtype=torch.float32, shape=(2, 4))
    b = _fg(1.0, dtype=torch.bfloat16, shape=(2, 4))
    with pytest.raises(AssertionError, match="dtype mismatch"):
        feature_grads_equal(a, b, exact=False, atol=1e-2)
    with pytest.raises(AssertionError, match="dtype mismatch"):
        feature_grads_equal(a, b, exact=True)


# ─────────────────────────────────────────────────────────────────────────────
# conftest slow-gating hook (item 1: fixture is the single slow-gating choke point)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeItem:
    """Minimal stand-in for a collected pytest item (fixturenames + add_marker)."""

    def __init__(self, fixturenames: tuple[str, ...]) -> None:
        self.fixturenames = fixturenames
        self.markers: list[object] = []

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


def test_conftest_hook_auto_marks_real_model_users_slow() -> None:
    """The collection hook marks any test whose fixture closure includes real_model.

    pytest 9 makes ``@pytest.mark.slow`` on the fixture itself a hard collection
    error, so the conftest hook is the slow-gating choke point.  A test that uses
    ``real_model`` is marked ``slow``; one that does not is left untouched.
    """
    assert "real_model" in SLOW_GATED_FIXTURES
    item_real = _FakeItem(("real_model", "tmp_path"))
    item_plain = _FakeItem(("tmp_path",))

    pytest_collection_modifyitems(config=None, items=[item_real, item_plain])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    assert any(
        getattr(m, "name", None) == "slow" for m in item_real.markers
    ), "real_model user was not auto-marked slow"
    assert item_plain.markers == [], "a non-real_model test must not be marked slow"
