"""Storage, eager update/state, and scripted collective contracts; no GPU claims."""

import copy
import gc
import io
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch as t

from steering.conditional.encoder_decoder import packed_optimizer as packed


def make_model(count=5, shape=(3, 4), world=3):
    model = t.nn.Module()
    generator = t.Generator().manual_seed(53)
    model.weights = t.nn.ParameterList(
        [t.nn.Parameter(t.randn(shape, generator=generator)) for _ in range(count)]
    )
    model.small = t.nn.Parameter(t.ones(4))
    model.large = t.nn.Parameter(t.ones(world * 16, 64))
    return model


def groups(model):
    return [
        {
            "kind": "muon",
            "params": list(model.weights),
            "lr": 0.01,
            "initial_lr": 0.01,
            "momentum": 0.85,
            "beta2": 0.95,
            "ns_steps": 5,
            "weight_decay": 0.02,
        },
        {
            "kind": "adamw",
            "params": [model.small, model.large],
            "lr": 0.002,
            "initial_lr": 0.002,
            "betas": (0.8, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    ]


def assert_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, t.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert t.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_equal(a, b)
    else:
        assert actual == expected


class ScriptedCollectives:
    """Script remote receive values; check local routing/order against upstream.

    Reductions model identical rank gradients. Gather supplies fixed nonzero
    remote rows and the actual local send; this is intentionally not a multi-rank
    arithmetic proof. Receives write only when their Future is waited.
    """

    def __init__(self, monkeypatch, rank, world, count):
        self.rank, self.world, self.count = rank, world, count
        self.events, self.sends = [], []
        self.fail_gather = False
        monkeypatch.setattr(packed.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(packed.dist, "get_rank", lambda: self.rank)
        monkeypatch.setattr(packed.dist, "get_world_size", lambda: self.world)
        monkeypatch.setattr(packed.dist, "group", SimpleNamespace(WORLD=object()))
        monkeypatch.setattr(packed.dist, "all_reduce", self.all_reduce)
        monkeypatch.setattr(packed.dist, "reduce_scatter_tensor", self.reduce_scatter)
        monkeypatch.setattr(packed.dist, "all_gather_into_tensor", self.all_gather)

    def work(self, label, action=lambda: None):
        def wait():
            self.events.append(("wait", label))
            action()

        return SimpleNamespace(get_future=lambda: SimpleNamespace(wait=wait))

    def all_reduce(self, value, *, op, async_op):
        assert op is packed.dist.ReduceOp.AVG and async_op
        self.events.append(("reduce", tuple(value.shape)))
        return self.work("reduce")

    def reduce_scatter(self, output, source, *, op, async_op):
        assert op is packed.dist.ReduceOp.AVG and async_op
        self.events.append(("scatter", tuple(source.shape), tuple(output.shape)))
        if source.ndim == 3:
            assert not t.count_nonzero(source[self.count :])
        chunk = source.chunk(self.world)[self.rank].clone()
        return self.work("scatter", lambda: output.copy_(chunk))

    def all_gather(self, output, source, *, async_op):
        assert async_op
        self.events.append(("gather", tuple(source.shape), tuple(output.shape)))
        if output.ndim == 3:
            # Candidate must not rely on backend-specific in-place all-gather.
            assert output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
            owned = min(source.shape[0], max(0, self.count - self.rank * source.shape[0]))
            assert not t.count_nonzero(source[owned:])
        sent = source.detach().clone()
        self.sends.append(sent)

        def receive():
            if self.fail_gather:
                raise RuntimeError("injected gather failure")
            output.fill_(0.125)
            if output.ndim == 3:
                output[self.count :].zero_()
            start = self.rank * source.shape[0]
            output[start : start + source.shape[0]].copy_(sent)

        return self.work("gather", receive)


@pytest.fixture
def eager(monkeypatch):
    for name in ("muon_step_fused", "adamw_step_fused"):
        monkeypatch.setattr(
            packed.upstream, name, getattr(packed.upstream, name)._torchdynamo_orig_callable
        )


def set_gradients(model, step):
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = t.linspace(-0.13, 0.21, parameter.numel()).reshape_as(parameter).to(
            parameter.dtype
        ) * (step + index + 1)


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3)])
@pytest.mark.parametrize("rank,world,count", [(0, 1, 3), (0, 3, 5), (2, 3, 5), (2, 3, 2)])
def test_distributed_evolving_math_collective_order_padding_and_checkpoint(
    monkeypatch, eager, shape, rank, world, count
):
    collective = ScriptedCollectives(monkeypatch, rank, world, count)
    kernel_calls = []
    original_kernel = packed.upstream.muon_step_fused

    def record_kernel(*args):
        kernel_calls.append(
            (
                [(tuple(value.shape), value.stride(), value.dtype) for value in args[:4]],
                [value.item() for value in args[4:8]],
                args[8:],
            )
        )
        return original_kernel(*args)

    monkeypatch.setattr(packed.upstream, "muon_step_fused", record_kernel)
    baseline_model = make_model(count, shape, world)
    actual_model = copy.deepcopy(baseline_model)
    baseline = packed.upstream.DistMuonAdamW(groups(baseline_model))
    ids = [id(value) for value in actual_model.parameters()]
    actual = packed.DistPackedMuonAdamW(groups(actual_model), owner=actual_model)
    assert [id(value) for value in actual_model.parameters()] == ids
    assert len(actual._packing.buffers) == 1
    assert not t.count_nonzero(next(iter(actual._packing.buffers.values()))[count:])
    for step in range(3):
        for model in (baseline_model, actual_model):
            set_gradients(model, step)
        for optimizer in (baseline, actual):
            optimizer.param_groups[0]["lr"] = 0.01 / (step + 1)
            optimizer.param_groups[0]["momentum"] = 0.85 + step * 0.01
        collective.events.clear()
        collective.sends.clear()
        kernel_calls.clear()
        baseline.step()
        events = list(collective.events)
        sends = list(collective.sends)
        expected_kernel_calls = list(kernel_calls)
        chunk = (count + world - 1) // world
        assert len(expected_kernel_calls) == int(rank * chunk < count)
        collective.events.clear()
        collective.sends.clear()
        kernel_calls.clear()
        versions = [value._version for value in actual_model.weights]
        actual.step()
        assert [value._version for value in actual_model.weights] == [
            value + 1 for value in versions
        ]
        assert kernel_calls == expected_kernel_calls
        assert collective.events == events
        assert_equal(collective.sends, sends)
        assert_equal(actual_model.state_dict(), baseline_model.state_dict())
        assert_equal(actual.state_dict(), baseline.state_dict())
        assert not actual._pending_muon
        assert not t.count_nonzero(next(iter(actual._packing.buffers.values()))[count:])
        for a, b in zip(actual_model.parameters(), baseline_model.parameters(), strict=True):
            assert t.equal(a.grad, b.grad)
        if step == 1:
            # Original checkpoint layout loads into freshly rebound packed
            # storage; no packed tensor is serialized in optimizer state.
            actual_model = copy.deepcopy(baseline_model)
            actual = packed.DistPackedMuonAdamW(groups(actual_model), owner=actual_model)
            actual.load_state_dict(copy.deepcopy(baseline.state_dict()))
            baseline.load_state_dict(copy.deepcopy(actual.state_dict()))


@pytest.mark.parametrize(
    "change", ["rank", "world", "group", "data", "order", "owner_assign", "missing_gradient"]
)
def test_distributed_binding_changes_fail_before_any_collective(monkeypatch, change):
    collective = ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    optimizer = packed.DistPackedMuonAdamW(groups(model), owner=model)
    set_gradients(model, 0)
    if change == "rank":
        collective.rank = 1
    elif change == "world":
        collective.world = 2
    elif change == "group":
        monkeypatch.setattr(packed.dist.group, "WORLD", object())
    elif change == "data":
        model.weights[0].data = model.weights[0].data.clone()
    elif change == "order":
        optimizer.param_groups[0]["params"].reverse()
    elif change == "owner_assign":
        model.load_state_dict(copy.deepcopy(model.state_dict()), assign=True)
    else:
        model.weights[0].grad = None
    with pytest.raises((RuntimeError, ValueError), match="changed|gradient"):
        optimizer.step()
    assert not collective.events


@pytest.mark.parametrize("failure", ["kernel", "gather"])
def test_distributed_partial_failure_invalidates_versions_and_poison_without_recovery(
    monkeypatch, eager, failure
):
    collective = ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    optimizer = packed.DistPackedMuonAdamW(groups(model), owner=model)
    set_gradients(model, 0)
    versions = [parameter._version for parameter in model.weights]
    if failure == "gather":
        collective.fail_gather = True
    else:
        original = packed.upstream.muon_step_fused

        def fail(*args):
            original(*args)
            raise RuntimeError("injected kernel failure after mutation")

        monkeypatch.setattr(packed.upstream, "muon_step_fused", fail)
    with pytest.raises(RuntimeError, match="injected"):
        optimizer.step()
    assert all(
        parameter._version > before
        for parameter, before in zip(model.weights, versions, strict=True)
    )
    if failure == "gather":
        assert optimizer._pending_muon  # retain asynchronous send/receive lifetime
    events = list(collective.events)
    for operation in (optimizer.step, optimizer.state_dict, lambda: optimizer.load_state_dict({})):
        with pytest.raises(RuntimeError, match="poisoned"):
            operation()
    assert collective.events == events


@pytest.mark.parametrize("failure", ["second_reduce", "second_kernel", "adamw", "second_gather"])
def test_distributed_late_failure_retains_earlier_gathers_and_invalidates_all_groups(
    monkeypatch, eager, failure
):
    collective = ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    model.second = t.nn.ParameterList(
        [t.nn.Parameter(value.detach().t().contiguous()) for value in model.weights]
    )
    group_list = groups(model)
    group_list.insert(1, {**group_list[0], "params": list(model.second)})
    optimizer = packed.DistPackedMuonAdamW(group_list, owner=model)
    set_gradients(model, 0)
    muon_parameters = list(model.weights) + list(model.second)
    versions = [value._version for value in muon_parameters]
    saved = model.weights[0].square().sum()
    before = model.weights[0].detach().clone()

    # The first gather is already in flight when a later compute fails. In the
    # second-gather case its wait succeeds before a subsequent gather fails.
    original_work = collective.work
    work_counts = {}

    def work(label, action=lambda: None):
        work_counts[label] = work_counts.get(label, 0) + 1
        if (failure, label, work_counts[label]) in (
            ("second_reduce", "scatter", 2),
            ("second_gather", "gather", 2),
        ):

            def fail():
                raise RuntimeError("injected later collective failure")

            return original_work(label, fail)
        return original_work(label, action)

    monkeypatch.setattr(collective, "work", work)
    if failure in ("second_kernel", "adamw"):
        kernel_name = "muon_step_fused" if failure == "second_kernel" else "adamw_step_fused"
        original_kernel = getattr(packed.upstream, kernel_name)
        kernel_calls = 0

        def fail_after_mutation(*args):
            nonlocal kernel_calls
            kernel_calls += 1
            original_kernel(*args)
            if kernel_calls == (2 if failure == "second_kernel" else 1):
                raise RuntimeError("injected later kernel failure")

        monkeypatch.setattr(packed.upstream, kernel_name, fail_after_mutation)

    with pytest.raises(RuntimeError, match="injected later"):
        optimizer.step()
    assert not t.equal(model.weights[0], before)
    assert all(
        value._version > version for value, version in zip(muon_parameters, versions, strict=True)
    )
    assert len(optimizer._pending_muon) == (
        1 if failure in ("second_reduce", "second_kernel") else 2
    )
    muon_sends = collective.sends[: len(optimizer._pending_muon)]
    for pending, sent in zip(optimizer._pending_muon, muon_sends, strict=True):
        assert t.equal(pending["updated_params"], sent)
        received = optimizer._packing.buffers[id(pending["params"][0])]
        assert (
            received.untyped_storage().data_ptr()
            != pending["updated_params"].untyped_storage().data_ptr()
        )
    gather_waits = [event for event in collective.events if event == ("wait", "gather")]
    assert len(gather_waits) == (2 if failure == "second_gather" else 0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()
    events = list(collective.events)
    for operation in (optimizer.step, optimizer.state_dict, lambda: optimizer.load_state_dict({})):
        with pytest.raises(RuntimeError, match="poisoned"):
            operation()
    assert collective.events == events  # no draining or unsafe attempted recovery


def test_distributed_saved_forward_is_invalidated_by_successful_owned_update(monkeypatch, eager):
    ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    optimizer = packed.DistPackedMuonAdamW(groups(model), owner=model)
    saved = model.weights[0].square().sum()
    versions = [parameter._version for parameter in model.weights]
    set_gradients(model, 0)
    optimizer.step()
    assert all(
        parameter._version > before
        for parameter, before in zip(model.weights, versions, strict=True)
    )
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()


def single_model(dtype=t.float32):
    model = t.nn.Module()
    generator = t.Generator().manual_seed(28)
    model.weights = t.nn.ParameterList(
        [
            t.nn.Parameter(t.randn(shape, generator=generator).to(dtype))
            for shape in ((3, 4), (3, 4), (4, 3), (3, 3), (4,))
        ]
    )
    return model


def single_groups(model):
    values = list(model.weights)
    template = {
        "kind": "muon",
        "lr": 0.01,
        "initial_lr": 0.01,
        "momentum": 0.85,
        "beta2": 0.95,
        "ns_steps": 5,
        "weight_decay": 0.02,
    }
    return [
        {**template, "params": values[:2]},
        {**template, "params": values[2:3], "beta2": None},
        {**template, "params": values[3:4]},
        {
            "kind": "adamw",
            "params": values[4:],
            "lr": 0.002,
            "initial_lr": 0.002,
            "betas": (0.8, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    ]


def test_single_parameter_identity_disjoint_views_and_adam_storage_unchanged():
    model = single_model()
    values = list(model.parameters())
    identities = [id(value) for value in values]
    original = [value.detach().clone() for value in values]
    versions = [value._version for value in values]
    adam_pointer = values[-1].data_ptr()
    optimizer = packed.PackedMuonAdamW(single_groups(model), owner=model)
    assert len(optimizer._packing.buffers) == 3
    assert [id(value) for value in values] == identities
    assert_equal([value.detach() for value in values], original)
    assert [value._version for value in values[:-1]] == [value + 1 for value in versions[:-1]]
    assert values[-1]._version == versions[-1]
    assert values[0].untyped_storage().data_ptr() == values[1].untyped_storage().data_ptr()
    assert values[1].storage_offset() - values[0].storage_offset() == values[0].numel()
    assert values[-1].data_ptr() == adam_pointer
    with t.no_grad():
        values[0][0, 0] += 1
    assert t.equal(values[1], original[1])
    assert not optimizer.state


@pytest.mark.parametrize(
    "constructor,upstream_constructor",
    [
        (packed.PackedMuonAdamW, packed.upstream.MuonAdamW),
        (packed.DistPackedMuonAdamW, packed.upstream.DistMuonAdamW),
    ],
)
def test_disjoint_source_views_preserve_shared_version_counter_and_exact_updates(
    monkeypatch, eager, constructor, upstream_constructor
):
    ScriptedCollectives(monkeypatch, 0, 1, 2)

    def shared_model():
        model = make_model(count=2, world=1)
        values = t.arange(24, dtype=t.float32).reshape(2, 3, 4) / 24
        model.weights = t.nn.ParameterList([t.nn.Parameter(value) for value in values])
        return model

    baseline_model, model = shared_model(), shared_model()
    baseline = upstream_constructor(groups(baseline_model))
    optimizer = constructor(groups(model), owner=model)
    assert [value._version for value in model.weights] == [2, 2]
    for step in range(3):
        for owner in (baseline_model, model):
            set_gradients(owner, step)
        saved = model.weights[0].square().sum()
        baseline_versions = [value._version for value in baseline_model.weights]
        versions = [value._version for value in model.weights]
        baseline.step()
        optimizer.step()
        assert_equal(model.state_dict(), baseline_model.state_dict())
        assert_equal(optimizer.state_dict(), baseline.state_dict())
        # Both parameter objects still share a counter, so updating both must
        # invalidate it twice, just as upstream's two copy-back operations do.
        assert [value._version for value in model.weights] == [value + 2 for value in versions]
        assert [value._version for value in baseline_model.weights] == [
            value + 2 for value in baseline_versions
        ]
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            saved.backward()


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_single_evolving_updates_original_dtype_state_and_cross_resume(eager, dtype):
    baseline_model = single_model(dtype)
    actual_model = copy.deepcopy(baseline_model)
    baseline = packed.upstream.MuonAdamW(single_groups(baseline_model))
    actual = packed.PackedMuonAdamW(single_groups(actual_model), owner=actual_model)
    for step in range(4):
        for model in (baseline_model, actual_model):
            set_gradients(model, step)
            if step == 2:
                for value in model.parameters():
                    value.grad.zero_()
            if step == 1:
                model.weights[-1].grad = None  # upstream single AdamW skips missing gradients
        for optimizer in (baseline, actual):
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] / (step + 1)
                if group["kind"] == "muon":
                    group["momentum"] = 0.85 + 0.01 * step
        gradients = [
            None if value.grad is None else value.grad.clone()
            for value in actual_model.parameters()
        ]
        baseline.step()
        actual.step()
        assert_equal(actual_model.state_dict(), baseline_model.state_dict())
        assert_equal(actual.state_dict(), baseline.state_dict())
        assert_equal([value.grad for value in actual_model.parameters()], gradients)
        assert set(actual.state) == {actual_model.weights[index] for index in (0, 2, 3, 4)}
        if step == 1:
            stream = io.BytesIO()
            t.save({"model": actual_model.state_dict(), "optimizer": actual.state_dict()}, stream)
            stream.seek(0)
            checkpoint = t.load(stream, weights_only=True)
            saved_values = list(checkpoint["model"].values())
            assert (
                saved_values[0].untyped_storage().data_ptr()
                == saved_values[1].untyped_storage().data_ptr()
            )
            # Packed -> upstream and upstream -> packed, with subsequent evolving updates.
            baseline_model.load_state_dict(checkpoint["model"])
            baseline = packed.upstream.MuonAdamW(single_groups(baseline_model))
            baseline.load_state_dict(checkpoint["optimizer"])
            actual_model = single_model(dtype)
            actual_model.load_state_dict(checkpoint["model"], assign=True)
            actual = packed.PackedMuonAdamW(single_groups(actual_model), owner=actual_model)
            actual.load_state_dict(copy.deepcopy(baseline.state_dict()))


@pytest.mark.parametrize(
    "change",
    [
        "data",
        "dtype",
        "shape",
        "parameter",
        "order",
        "kind",
        "assign",
        "owner_gone",
        "requires_grad",
    ],
)
def test_single_binding_changes_reject_step_and_serialization_before_mutation(change):
    model = single_model()
    optimizer = packed.PackedMuonAdamW(single_groups(model), owner=model)
    if change == "data":
        model.weights[0].data = model.weights[0].data.clone()
    elif change == "dtype":
        model.weights[0].data = model.weights[0].data.double()
    elif change == "shape":
        model.weights[0].data = model.weights[0].data.reshape(4, 3)
    elif change == "parameter":
        optimizer.param_groups[0]["params"][0] = t.nn.Parameter(model.weights[0].detach().clone())
    elif change == "order":
        optimizer.param_groups[0]["params"].reverse()
    elif change == "kind":
        optimizer.param_groups[0]["kind"] = "adamw"
    elif change == "assign":
        model.load_state_dict(copy.deepcopy(model.state_dict()), assign=True)
    elif change == "owner_gone":
        del model
        gc.collect()
    elif change == "requires_grad":
        model.weights[0].requires_grad_(False)
    for operation in (optimizer.step, optimizer.state_dict, lambda: optimizer.load_state_dict({})):
        with pytest.raises(RuntimeError, match="changed|replaced"):
            operation()
    assert not optimizer.state


@pytest.mark.parametrize("constructor", [packed.PackedMuonAdamW, packed.DistPackedMuonAdamW])
@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate",
        "overlap",
        "owner_alias",
        "noncontiguous",
        "dtype",
        "existing_grad",
        "shape",
        "foreign",
        "compiled_owner",
        "owner_type",
        "unknown_kind",
        "empty_group",
        "bare_tensor",
        "group_duplicate",
        "empty_parameter",
        "meta_parameter",
    ],
)
def test_invalid_later_group_is_rejected_before_any_storage_rebinding(
    monkeypatch, constructor, invalid
):
    ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = single_model()
    # First group stays valid, so late validation must precede its rebinding.
    if invalid == "duplicate":
        model.weights[3] = model.weights[2]
    elif invalid == "overlap":
        model.weights[3] = t.nn.Parameter(model.weights[2].detach())
    elif invalid == "owner_alias":
        model.alias_excluded_from_optimizer = t.nn.Parameter(model.weights[2].detach())
    elif invalid == "noncontiguous":
        model.weights[2] = t.nn.Parameter(t.zeros(3, 4).t())
    elif invalid == "dtype":
        model.weights[2] = t.nn.Parameter(t.zeros(4, 3, dtype=t.float64))
    elif invalid == "existing_grad":
        model.weights[2].grad = t.ones_like(model.weights[2])
    elif invalid == "empty_parameter":
        model.weights[2] = t.nn.Parameter(t.empty(4, 0))
    elif invalid == "meta_parameter":
        model.weights[2] = t.nn.Parameter(t.empty(4, 3, device="meta"))
    group_list = single_groups(model)
    group_list[1]["beta2"] = 0.95
    owner = model
    if invalid == "shape":
        group_list[1]["params"].append(group_list.pop(2)["params"][0])
    elif invalid == "foreign":
        group_list[1]["params"] = [t.nn.Parameter(t.ones(4, 3))]
    elif invalid == "compiled_owner":
        owner = t.nn.Module()
        owner._orig_mod = model
    elif invalid == "owner_type":
        owner = object()
    elif invalid == "unknown_kind":
        group_list[1]["kind"] = "sgd"
    elif invalid == "empty_group":
        group_list[1]["params"] = []
    elif invalid == "bare_tensor":
        group_list[1]["params"] = [model.weights[2].detach()]
    elif invalid == "group_duplicate":
        group_list[1]["params"].append(model.weights[2])
    values = list(model.parameters())
    pointers = [value.data_ptr() for value in values]
    versions = [value._version for value in values]
    warning = (
        pytest.warns(UserWarning, match="duplicate parameters")
        if invalid == "group_duplicate"
        else nullcontext()
    )
    with warning, pytest.raises(ValueError):
        constructor(group_list, owner=owner)
    assert [value.data_ptr() for value in values] == pointers
    assert [value._version for value in values] == versions


@pytest.mark.parametrize("constructor", [packed.PackedMuonAdamW, packed.DistPackedMuonAdamW])
def test_owner_is_required_and_assign_false_load_preserves_bindings(monkeypatch, constructor):
    ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    with pytest.raises(TypeError, match="owner"):
        constructor(groups(model))
    optimizer = constructor(groups(model), owner=model)
    model.load_state_dict(copy.deepcopy(model.state_dict()), assign=False)
    optimizer._validate_bindings()


@pytest.mark.parametrize("constructor", [packed.PackedMuonAdamW, packed.DistPackedMuonAdamW])
def test_owner_tied_alias_added_after_construction_is_rejected(monkeypatch, constructor):
    collective = ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    optimizer = constructor(groups(model), owner=model)
    model.new_alias = model.weights[0]
    with pytest.raises(RuntimeError, match="owning model"):
        optimizer.step()
    assert not optimizer.state and not collective.events


@pytest.mark.parametrize("failure", [False, True])
def test_single_saved_tensor_guard_hook_count_and_kernel_failure(monkeypatch, eager, failure):
    model = single_model()
    stale_before_construction = model.weights[0].square().sum()
    optimizer = packed.PackedMuonAdamW(single_groups(model), owner=model)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        stale_before_construction.backward()
    saved = model.weights[0].square().sum()
    versions = [value._version for value in model.weights[:2]]
    set_gradients(model, 0)
    calls = []
    optimizer.register_step_pre_hook(lambda *args: calls.append(True))
    if failure:
        original = packed.upstream.muon_step_fused

        def fail(*args):
            original(*args)
            raise RuntimeError("injected failure after mutation")

        monkeypatch.setattr(packed.upstream, "muon_step_fused", fail)
        with pytest.raises(RuntimeError, match="injected"):
            optimizer.step()
    else:
        optimizer.step()
    assert calls == [True]
    assert [value._version for value in model.weights[:2]] == [value + 1 for value in versions]
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved.backward()


def test_distributed_failed_future_acquisition_retains_buffers(monkeypatch, eager):
    collective = ScriptedCollectives(monkeypatch, 0, 3, 5)
    model = make_model()
    optimizer = packed.DistPackedMuonAdamW(groups(model), owner=model)
    set_gradients(model, 0)
    versions = [value._version for value in model.weights]
    original = packed.dist.all_gather_into_tensor

    def gather(*args, **kwargs):
        original(*args, **kwargs)

        def failure():
            raise RuntimeError("injected get_future failure after launch")

        return SimpleNamespace(get_future=failure)

    monkeypatch.setattr(packed.dist, "all_gather_into_tensor", gather)
    with pytest.raises(RuntimeError, match="get_future failure"):
        optimizer.step()
    assert optimizer._failed and len(optimizer._pending_muon) == 1
    pending = optimizer._pending_muon[0]
    assert pending["updated_params"].shape == (2, 3, 4)
    assert [value._version for value in model.weights] == [value + 1 for value in versions]
    events = list(collective.events)
    with pytest.raises(RuntimeError, match="poisoned"):
        optimizer.step()
    assert collective.events == events


def test_distributed_requires_initialized_group_before_rebinding(monkeypatch):
    model = make_model()
    pointers = [value.data_ptr() for value in model.parameters()]
    monkeypatch.setattr(packed.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized process group"):
        packed.DistPackedMuonAdamW(groups(model), owner=model)
    assert [value.data_ptr() for value in model.parameters()] == pointers


@pytest.mark.parametrize(
    "constructor,upstream_constructor",
    [
        (packed.PackedMuonAdamW, packed.upstream.MuonAdamW),
        (packed.DistPackedMuonAdamW, packed.upstream.DistMuonAdamW),
    ],
)
@pytest.mark.parametrize("muon", [False, True])
def test_step_hooks_run_once_after_upstream_optimizer_construction(
    monkeypatch, eager, constructor, upstream_constructor, muon
):
    ScriptedCollectives(monkeypatch, 0, 3, 5)
    baseline_model = make_model()
    # Optimizer.__init__ installs a hook wrapper on the upstream class's step.
    baseline = upstream_constructor(groups(baseline_model)[0 if muon else 1 :])
    set_gradients(baseline_model, 0)
    baseline.step()
    model = make_model()
    # AdamW-only groups occur when training a head without residual matrices.
    optimizer = constructor(groups(model)[0 if muon else 1 :], owner=model)
    set_gradients(model, 0)
    hooks, grad_modes = [], []
    optimizer.register_step_pre_hook(lambda *args: hooks.append("pre"))
    optimizer.register_step_post_hook(lambda *args: hooks.append("post"))
    for name in ("muon_step_fused", "adamw_step_fused"):
        kernel = getattr(packed.upstream, name)

        def record_grad_mode(*args, kernel=kernel):
            grad_modes.append(t.is_grad_enabled())
            return kernel(*args)

        monkeypatch.setattr(packed.upstream, name, record_grad_mode)
    with t.enable_grad():
        optimizer.step()
    assert hooks == ["pre", "post"]
    assert grad_modes and not any(grad_modes)
    assert_equal(model.state_dict(), baseline_model.state_dict())
    assert_equal(optimizer.state_dict(), baseline.state_dict())
