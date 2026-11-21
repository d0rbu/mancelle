"""Direct unit tests for ``tasks.mixtures.build_mixture``.

Audit scope (Wave 3C task 1): the public mixture-composition entry point had
zero direct tests. Indirect coverage via capability/safety integration tests
is aggregate-level only and cannot detect an off-by-one in
``range(params["<name>_epochs"])`` where a single source could be silently
halved (or doubled).

These tests monkeypatch every task constructor in ``tasks.mixtures`` with a
cheap stub that records its init arguments. We then assert on the exact
composition of the returned ``(train_tasks, val_tasks)`` lists: class
identities, instance counts per class, and the constructor kwargs each
instance was built with.

Each test pins a contract that would fail if the off-by-one were
re-introduced — in particular, ``_epochs=N`` must produce exactly ``N``
instances (not ``N-1`` nor ``N+1``), and ``_epochs=1`` must mean one pass.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tasks import mixtures as mixtures_module
from tasks.common import Task, TaskMixture
from tasks.mixtures import _DEFAULTS as _DEFAULTS_RAW
from tasks.mixtures import STEERED_MIXTURES, SFTMixture, steered_mixture_overrides
from tasks.mixtures import build_mixture as _build_mixture


def _d(key: str) -> int:
    """Return a default as ``int`` (helper for mypy/ty — values in _DEFAULTS for
    *_epochs keys are always int, never None)."""
    v = _DEFAULTS_RAW[key]
    assert isinstance(v, int), f"expected int default for {key}, got {v!r}"
    return v


_DEFAULTS = _DEFAULTS_RAW


def build_mixture(mixture: Any, **overrides: Any) -> tuple[list[Any], list[Any]]:
    """Typing-neutral wrapper so tests can freely access stub attributes."""
    train, val = _build_mixture(mixture, **overrides)
    return list(train), list(val)


def _write_steered_jsonl(path: Path, rows: int) -> str:
    """Write ``rows`` valid single-turn steered conversations and return the path."""
    path.write_text(
        "\n".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt {index}"},
                    {"role": "assistant", "content": f"steered answer {index}"},
                ]
            )
            for index in range(rows)
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _overrides_for(mixture: SFTMixture, steered_jsonl: str) -> dict[str, Any]:
    """Feed the JSONL to the members that consume it, and only to those."""
    return steered_mixture_overrides(
        mixture, steered_jsonl if mixture in STEERED_MIXTURES else None
    )


@pytest.fixture
def steered_jsonl(tmp_path: Path) -> str:
    """Three steered single-turn rows, enough to exercise the apportionment."""
    return _write_steered_jsonl(tmp_path / "steered.jsonl", 3)


# ---------------------------------------------------------------------------
# Stub task classes
# ---------------------------------------------------------------------------


_TASK_NAMES = [
    "ARC",
    "SmolTalk",
    "CustomJSON",
    "MMLU",
    "GSM8K",
    "SimpleSpelling",
    "SpellingBee",
    "OrcaMath",
    "MetaMathQA",
    "OpenCoder",
    "WildJailbreak",
    "WildGuardMix",
    "CoCoNot",
    "NemotronSafety",
]


def _make_stub(name: str) -> type:
    """Create a stub Task subclass that records its constructor kwargs.

    The stub deliberately does *not* call ``Task.__init__`` with any dataset
    loading — it just stores the kwargs on the instance so tests can assert
    on exact construction parameters.
    """

    class _Stub(Task):
        _stub_name = name

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            # Intentionally skip Task.__init__ to avoid dataset loading.
            self.stub_name = name
            self.stub_args = args
            self.stub_kwargs = dict(kwargs)

        def num_examples(self) -> int:
            return 0

    _Stub.__name__ = f"Stub{name}"
    _Stub.__qualname__ = _Stub.__name__
    return _Stub


@pytest.fixture
def stub_tasks(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Replace every task class in ``tasks.mixtures`` with a recording stub."""
    stubs: dict[str, type] = {}
    monkeypatch.setattr(mixtures_module, "exclude_heldout_questions", lambda train, val: None)
    for task_name in _TASK_NAMES:
        stub = _make_stub(task_name)
        stubs[task_name] = stub
        monkeypatch.setattr(mixtures_module, task_name, stub)
    # Also make get_base_dir deterministic so identity_path is stable.
    monkeypatch.setattr(
        mixtures_module,
        "get_base_dir",
        lambda: "/tmp/nanochat_test_base",
    )
    return stubs


_SIZED_TASK_COUNTS = {
    "ARC": 11,
    "SmolTalk": 17,
    "CustomJSON": 3,
    "MMLU": 6000,
    "GSM8K": 7000,
    "OrcaMath": 13,
    "MetaMathQA": 19,
    "OpenCoder": 23,
    "WildJailbreak": 2,
    "WildGuardMix": 3,
    "CoCoNot": 5,
    "NemotronSafety": 29,
}


def _make_sized_stub(name: str) -> type[Task]:
    """Create a recording stub with realistic, deterministic task lengths."""

    class _SizedStub(Task):
        def __init__(
            self,
            *args: Any,
            start: int = 0,
            stop: int | None = None,
            step: int = 1,
            **kwargs: Any,
        ) -> None:
            super().__init__(start=start, stop=stop, step=step)
            self.stub_name = name
            self.stub_args = args
            self.stub_kwargs = {
                **kwargs,
                "start": start,
                "stop": stop,
                "step": step,
            }
            self.size = kwargs.get("size")

        def num_examples(self) -> int:
            if name in {"SimpleSpelling", "SpellingBee"}:
                assert isinstance(self.size, int)
                return self.size
            return _SIZED_TASK_COUNTS[name]

    _SizedStub.__name__ = f"SizedStub{name}"
    _SizedStub.__qualname__ = _SizedStub.__name__
    return _SizedStub


@pytest.fixture
def sized_stub_tasks(monkeypatch: pytest.MonkeyPatch) -> dict[str, type[Task]]:
    """Replace task classes with sized stubs suitable for ``TaskMixture``."""
    stubs: dict[str, type[Task]] = {}
    monkeypatch.setattr(mixtures_module, "exclude_heldout_questions", lambda train, val: None)
    for task_name in _TASK_NAMES:
        stub = _make_sized_stub(task_name)
        stubs[task_name] = stub
        monkeypatch.setattr(mixtures_module, task_name, stub)
    monkeypatch.setattr(
        mixtures_module,
        "get_base_dir",
        lambda: "/tmp/nanochat_test_base",
    )
    return stubs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SlicedStub(Task):
    """A task constructed with a non-trivial logical slice."""

    def __init__(self, count: int, **task_kwargs: Any) -> None:
        super().__init__(**task_kwargs)
        self.count = count

    def num_examples(self) -> int:
        return self.count

    def get_example(self, index: int) -> dict[str, int]:
        return {"index": index}


def _class_counts(tasks: list[Any]) -> Counter[str]:
    return Counter(getattr(t, "stub_name", type(t).__name__) for t in tasks)


def _class_sequence(tasks: list[Any]) -> list[str]:
    return [getattr(t, "stub_name", type(t).__name__) for t in tasks]


def _stub_kwargs(task: Any) -> dict[str, Any]:
    return task.stub_kwargs  # type: ignore[no-any-return]


def _stub_name(task: Any) -> str:
    return task.stub_name  # type: ignore[no-any-return]


def test_chatcore_uses_only_benchmark_train_and_full_test_splits(stub_tasks) -> None:
    train, val = build_mixture(SFTMixture.CHATCORE)
    assert _class_sequence(train) == ["ARC", "ARC", "MMLU", "GSM8K"]
    assert _class_sequence(val) == ["ARC", "ARC", "MMLU", "GSM8K"]
    assert [_stub_kwargs(task) for task in train] == [
        {"subset": "ARC-Easy", "split": "train"},
        {"subset": "ARC-Challenge", "split": "train"},
        {"subset": "auxiliary_train", "split": "train"},
        {"subset": "main", "split": "train"},
    ]
    assert [_stub_kwargs(task) for task in val] == [
        {"subset": "ARC-Easy", "split": "test"},
        {"subset": "ARC-Challenge", "split": "test"},
        {"subset": "all", "split": "test"},
        {"subset": "main", "split": "test"},
    ]
    with pytest.raises(ValueError, match="does not consume"):
        build_mixture(SFTMixture.CHATCORE, mmlu_epochs=2)


class TestDefaultEquivalencePin:
    """Pin default task lengths and shuffled index maps for the mix contract."""

    _EXPECTED = {
        SFTMixture.CHATCORE: {
            "train": (
                [11, 11, 6000, 7000],
                "0d050766aa24fdf207766815a6512cb7aaf71a991370d46b1b7fbe204e98f81d",
                13022,
            ),
            "val": (
                [11, 11, 6000, 7000],
                "0d050766aa24fdf207766815a6512cb7aaf71a991370d46b1b7fbe204e98f81d",
                13022,
            ),
        },
        SFTMixture.STANDARD: {
            "train": (
                [
                    17,
                    3,
                    3,
                    6000,
                    7000,
                    200000,
                    80000,
                ],
                "0a1d172ae721f86047a62cff0735289d41d12dfbd6ddb2185db3ce0c8ad83188",
                293023,
            ),
            "val": (
                [17, 5200, 420],
                "1b88a5c895cfa86eb958be1f328cdf9a486322551f0c7135f0f8af072100390b",
                5637,
            ),
        },
        SFTMixture.CAPABILITIES: {
            "train": (
                [120000, 3, 3, 6000, 6000, 23, 13, 19, 200000, 80000],
                "51ca84cdb2bdc27afb47c5543ad35a7226cc7c47d4a3a79b86b9d111fffc4e76",
                412061,
            ),
            "val": (
                [17, 5200, 13, 420],
                "3eebc22b0be43bad58f96ea2af9cfd3d1fbf6159bd2e4f1d46a32d911fff36e6",
                5650,
            ),
        },
        SFTMixture.SAFETY: {
            "train": (
                [
                    2,
                    2,
                    3,
                    3,
                    3,
                    5,
                    5,
                    120000,
                    29,
                    3,
                    3,
                ],
                "52f0408c3679420bb28a9542fcc473ede6d13d58be74e9ecb25ccf985f69f890",
                120058,
            ),
            "val": (
                [5000],
                "4e8630367f14c1909ca896f7fc7322b5ff461828ffa834f35b9f3bf55c109a07",
                5000,
            ),
        },
        # The steered members append the JSONL as the LAST train task and reuse
        # STANDARD's val split byte for byte (note the identical val digest).
        SFTMixture.STANDARD_STEERED: {
            "train": (
                [17, 3, 3, 6000, 7000, 200000, 80000, 3],
                "5f50fcd35046204d81f848b91d0aa817ebd041b0ca65f8a9111bf56648ea7bbb",
                293026,
            ),
            "val": (
                [17, 5200, 420],
                "1b88a5c895cfa86eb958be1f328cdf9a486322551f0c7135f0f8af072100390b",
                5637,
            ),
        },
        # target=3 apportioned over [17, 3, 3, 6000, 7000, 200000, 80000]: every
        # floor is 0 except SimpleSpelling (2), and the single remainder seat
        # goes to SpellingBee (largest fraction, 0.819).
        SFTMixture.STANDARD_STEERED_BALANCED: {
            "train": (
                [0, 0, 0, 0, 0, 2, 1, 3],
                "867360499e9763d6bef0199f0294ffb46ec959afa5449f3ec1364073a7697155",
                6,
            ),
            "val": (
                [17, 5200, 420],
                "1b88a5c895cfa86eb958be1f328cdf9a486322551f0c7135f0f8af072100390b",
                5637,
            ),
        },
    }

    @pytest.mark.parametrize("mixture", list(SFTMixture))
    def test_default_lengths_and_index_map(
        self,
        sized_stub_tasks: dict[str, type[Task]],
        mixture: SFTMixture,
        steered_jsonl: str,
    ) -> None:
        train_tasks, val_tasks = build_mixture(mixture, **_overrides_for(mixture, steered_jsonl))
        for split, tasks in (("train", train_tasks), ("val", val_tasks)):
            task_mixture = TaskMixture(tasks)
            lengths, digest, count = self._EXPECTED[mixture][split]
            assert [len(task) for task in tasks] == lengths
            assert hashlib.sha256(repr(task_mixture.index_map).encode()).hexdigest() == digest
            assert task_mixture.num_conversations == count

    def test_none_stop_is_equivalent_to_omitted_stop(
        self, sized_stub_tasks: dict[str, type[Task]]
    ) -> None:
        stub_smoltalk: Any = sized_stub_tasks["SmolTalk"]
        assert len(stub_smoltalk(split="train")) == len(stub_smoltalk(split="train", stop=None))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_standard_two_calls_identical_sequence(self, stub_tasks) -> None:
        """Two calls with identical params yield identical class sequences and kwargs."""
        train1, val1 = build_mixture(SFTMixture.STANDARD)
        train2, val2 = build_mixture(SFTMixture.STANDARD)
        assert _class_sequence(train1) == _class_sequence(train2)
        assert _class_sequence(val1) == _class_sequence(val2)
        for a, b in zip(train1, train2, strict=True):
            assert a.stub_kwargs == b.stub_kwargs
            assert a.stub_args == b.stub_args

    def test_capabilities_two_calls_identical_sequence(self, stub_tasks) -> None:
        train1, val1 = build_mixture(SFTMixture.CAPABILITIES)
        train2, val2 = build_mixture(SFTMixture.CAPABILITIES)
        assert _class_sequence(train1) == _class_sequence(train2)
        assert _class_sequence(val1) == _class_sequence(val2)

    def test_safety_two_calls_identical_sequence(self, stub_tasks) -> None:
        train1, val1 = build_mixture(SFTMixture.SAFETY)
        train2, val2 = build_mixture(SFTMixture.SAFETY)
        assert _class_sequence(train1) == _class_sequence(train2)
        assert _class_sequence(val1) == _class_sequence(val2)


# ---------------------------------------------------------------------------
# Exact composition — pins the off-by-one contract
# ---------------------------------------------------------------------------


class TestStandardComposition:
    def test_default_composition_exact(self, stub_tasks) -> None:
        """Pin the full class sequence for the default STANDARD mixture."""
        train, val = build_mixture(SFTMixture.STANDARD)
        expected_train = (
            ["SmolTalk", "CustomJSON", "CustomJSON"]
            + ["MMLU"] * _d("mmlu_epochs")  # 1
            + ["GSM8K"] * _d("gsm8k_epochs")  # 1
            + ["SimpleSpelling", "SpellingBee"]
        )
        assert _class_sequence(train) == expected_train
        assert _class_sequence(val) == ["SmolTalk", "MMLU", "GSM8K"]

    def test_mmlu_epochs_exact_count(self, stub_tasks) -> None:
        """mmlu_epochs=N must produce exactly N MMLU instances (off-by-one guard)."""
        for n in (1, 2, 5, 7):
            train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=n)
            counts = _class_counts(train)
            assert counts["MMLU"] == n, (
                f"mmlu_epochs={n} must yield exactly {n} MMLU train instances, got {counts['MMLU']}"
            )

    def test_mmlu_epochs_one_means_one_pass(self, stub_tasks) -> None:
        """Boundary: epochs=1 must mean exactly one MMLU instance, not zero, not two."""
        train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=1)
        assert _class_counts(train)["MMLU"] == 1

    def test_gsm8k_epochs_one_means_one_pass(self, stub_tasks) -> None:
        train, _ = build_mixture(SFTMixture.STANDARD, gsm8k_epochs=1)
        assert _class_counts(train)["GSM8K"] == 1

    def test_mmlu_epochs_zero_yields_zero_instances(self, stub_tasks) -> None:
        """Empty component: zero-weight (epochs=0) must yield zero instances."""
        train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=0)
        assert _class_counts(train)["MMLU"] == 0

    def test_gsm8k_epochs_zero_yields_zero_instances(self, stub_tasks) -> None:
        train, _ = build_mixture(SFTMixture.STANDARD, gsm8k_epochs=0)
        assert _class_counts(train)["GSM8K"] == 0

    def test_epochs_override_ratio(self, stub_tasks) -> None:
        """A 3:1 weight ratio between MMLU and GSM8K must be reflected exactly."""
        train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=6, gsm8k_epochs=2)
        counts = _class_counts(train)
        assert counts["MMLU"] == 6
        assert counts["GSM8K"] == 2
        assert counts["MMLU"] / counts["GSM8K"] == 3.0

    def test_double_epochs_doubles_instances(self, stub_tasks) -> None:
        """Off-by-one guard: 2x the epoch override must yield 2x the instances."""
        default_train, _ = build_mixture(SFTMixture.STANDARD)
        doubled_train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=2 * _d("mmlu_epochs"))
        default_mmlu = _class_counts(default_train)["MMLU"]
        doubled_mmlu = _class_counts(doubled_train)["MMLU"]
        assert doubled_mmlu == 2 * default_mmlu
        # And default must match the documented default (contract lock-in).
        assert default_mmlu == _DEFAULTS["mmlu_epochs"]

    def test_spelling_size_kwarg_is_forwarded(self, stub_tasks) -> None:
        """Spelling sizes flow from params; defaults are 200000 / 80000."""
        train, _ = build_mixture(SFTMixture.STANDARD)
        simple = [t for t in train if t.stub_name == "SimpleSpelling"]
        bee = [t for t in train if t.stub_name == "SpellingBee"]
        assert len(simple) == 1
        assert len(bee) == 1
        assert simple[0].stub_kwargs["size"] == 200000
        assert bee[0].stub_kwargs["size"] == 80000


class TestStandardParamsFix:
    def test_spelling_sizes_forwarded_standard(self, stub_tasks) -> None:
        train, _ = build_mixture(
            SFTMixture.STANDARD,
            simple_spelling_size=999,
            spelling_bee_size=111,
        )
        simple = [task for task in train if task.stub_name == "SimpleSpelling"][0]
        bee = [task for task in train if task.stub_name == "SpellingBee"][0]
        assert simple.stub_kwargs["size"] == 999
        assert bee.stub_kwargs["size"] == 111

    def test_standard_capabilities_parity_on_shared_keys(self, stub_tasks) -> None:
        overrides = {
            "simple_spelling_size": 999,
            "spelling_bee_size": 111,
        }
        standard, _ = build_mixture(SFTMixture.STANDARD, **overrides)
        capabilities, _ = build_mixture(SFTMixture.CAPABILITIES, **overrides)
        for task_name in ("SimpleSpelling", "SpellingBee"):
            standard_task = [task for task in standard if task.stub_name == task_name][0]
            capabilities_task = [task for task in capabilities if task.stub_name == task_name][0]
            assert standard_task.stub_kwargs == capabilities_task.stub_kwargs


class TestCapabilitiesComposition:
    def test_default_composition_exact(self, stub_tasks) -> None:
        train, val = build_mixture(SFTMixture.CAPABILITIES)
        expected_train = (
            ["SmolTalk", "CustomJSON", "CustomJSON"]
            + ["MMLU"] * _d("capabilities_mmlu_epochs")  # 2
            + ["OpenCoder", "OrcaMath", "MetaMathQA", "SimpleSpelling", "SpellingBee"]
        )
        assert _class_sequence(train) == expected_train
        assert _class_sequence(val) == [
            "SmolTalk",
            "MMLU",
            "OrcaMath",
            "GSM8K",
        ]

    def test_opencoder_limit_forwarded(self, stub_tasks) -> None:
        train, _ = build_mixture(SFTMixture.CAPABILITIES, opencoder_limit=1234)
        opencoder = [t for t in train if t.stub_name == "OpenCoder"][0]
        assert opencoder.stub_kwargs["max_rows"] == 1234

    def test_capabilities_has_identity(self, stub_tasks) -> None:
        """Capabilities carries two CustomJSON / identity passes."""
        train, val = build_mixture(SFTMixture.CAPABILITIES)
        assert _class_counts(train)["CustomJSON"] == 2
        assert _class_counts(val)["CustomJSON"] == 0

    def test_smoltalk_limit_forwarded_as_stop(self, stub_tasks) -> None:
        train, _ = build_mixture(SFTMixture.CAPABILITIES, smoltalk_limit=1234)
        smoltalk = [t for t in train if t.stub_name == "SmolTalk"]
        assert len(smoltalk) == 1
        assert smoltalk[0].stub_kwargs["stop"] == 1234

    def test_orca_metamath_limits_forwarded(self, stub_tasks) -> None:
        train, _ = build_mixture(
            SFTMixture.CAPABILITIES,
            orca_math_limit=42,
            metamath_limit=7,
        )
        orca = [t for t in train if t.stub_name == "OrcaMath"][0]
        meta = [t for t in train if t.stub_name == "MetaMathQA"][0]
        assert orca.stub_kwargs["stop"] == 42
        assert meta.stub_kwargs["stop"] == 7
        assert meta.stub_kwargs["gsm_only"] is True

    def test_simple_spelling_size_flows_through(self, stub_tasks) -> None:
        train, _ = build_mixture(
            SFTMixture.CAPABILITIES,
            simple_spelling_size=999,
            spelling_bee_size=111,
        )
        simple = [t for t in train if t.stub_name == "SimpleSpelling"][0]
        bee = [t for t in train if t.stub_name == "SpellingBee"][0]
        assert simple.stub_kwargs["size"] == 999
        assert bee.stub_kwargs["size"] == 111


class TestSafetyComposition:
    def test_default_composition_exact(self, stub_tasks) -> None:
        train, val = build_mixture(SFTMixture.SAFETY)
        expected_train = (
            ["WildJailbreak"] * _d("wildjailbreak_epochs")  # 2
            + ["WildGuardMix"] * _d("wildguardmix_epochs")  # 3
            + ["CoCoNot"] * _d("coconot_epochs")  # 2
            + ["SmolTalk"]
            + ["NemotronSafety"]
            + ["CustomJSON", "CustomJSON"]
        )
        assert _class_sequence(train) == expected_train
        assert _class_sequence(val) == ["SmolTalk"]

    @pytest.mark.parametrize(
        "param",
        [
            "wildjailbreak_epochs",
            "wildguardmix_epochs",
            "coconot_epochs",
        ],
    )
    def test_each_epoch_override_exact_count(self, stub_tasks, param: str) -> None:
        """Each *_epochs override in SAFETY must produce exactly N instances for N in {0,1,2,5}."""
        # Map param -> class name
        mapping = {
            "wildjailbreak_epochs": "WildJailbreak",
            "wildguardmix_epochs": "WildGuardMix",
            "coconot_epochs": "CoCoNot",
        }
        cls_name = mapping[param]
        for n in (0, 1, 2, 5):
            train, _ = build_mixture(SFTMixture.SAFETY, **{param: n})
            got = _class_counts(train)[cls_name]
            assert got == n, f"{param}={n} -> {cls_name} count should be {n}, got {got}"

    def test_safety_epoch_one_is_one_pass(self, stub_tasks) -> None:
        """Boundary: every *_epochs=1 in SAFETY must mean exactly one instance."""
        train, _ = build_mixture(
            SFTMixture.SAFETY,
            wildjailbreak_epochs=1,
            wildguardmix_epochs=1,
            coconot_epochs=1,
        )
        counts = _class_counts(train)
        for name in (
            "WildJailbreak",
            "WildGuardMix",
            "CoCoNot",
        ):
            assert counts[name] == 1, f"{name} should appear exactly once, got {counts[name]}"

    def test_smoltalk_limit_forwarded(self, stub_tasks) -> None:
        train, _ = build_mixture(
            SFTMixture.SAFETY,
            smoltalk_safety_limit=555,
        )
        # SmolTalk appears once in SAFETY train tasks.
        smoltalks = [t for t in train if t.stub_name == "SmolTalk"]
        assert len(smoltalks) == 1
        assert smoltalks[0].stub_kwargs["stop"] == 555

    def test_safety_has_identity(self, stub_tasks) -> None:
        """SAFETY carries over two CustomJSON identity passes from STANDARD."""
        train, _ = build_mixture(SFTMixture.SAFETY)
        assert _class_counts(train)["CustomJSON"] == 2


# ---------------------------------------------------------------------------
# Single-component / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_component_mixture_when_all_others_zero(self, stub_tasks) -> None:
        """Collapse STANDARD to only the SmolTalk train task by zeroing all epochs.

        CustomJSON identity lines and the two spelling tasks are hard-coded, so we
        can only zero the *_epochs knobs. Assert the resulting class Counter.
        """
        train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=0, gsm8k_epochs=0)
        counts = _class_counts(train)
        assert counts["MMLU"] == 0
        assert counts["GSM8K"] == 0
        # Non-overridable components remain.
        assert counts["SmolTalk"] == 1
        assert counts["CustomJSON"] == 2
        assert counts["SimpleSpelling"] == 1
        assert counts["SpellingBee"] == 1

    def test_unknown_mixture_raises(self, stub_tasks) -> None:
        with pytest.raises((KeyError, ValueError), match="does-not-exist"):
            build_mixture("does-not-exist")  # type: ignore[arg-type]

    def test_returns_tuple_of_two_lists(self, stub_tasks) -> None:
        result = build_mixture(SFTMixture.STANDARD)
        assert isinstance(result, tuple)
        assert len(result) == 2
        train, val = result
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert len(train) > 0
        assert len(val) > 0

    def test_unknown_override_raises(self, stub_tasks) -> None:
        """The stricter validator explicitly rejects unknown override keys."""
        with pytest.raises(ValueError, match="Unknown mixture override"):
            build_mixture(SFTMixture.STANDARD, totally_unused_key=999)

    def test_overrides_do_not_mutate_defaults(self, stub_tasks) -> None:
        before = dict(_DEFAULTS)
        build_mixture(SFTMixture.STANDARD, mmlu_epochs=99)
        assert _DEFAULTS == before

    def test_negative_epochs_yields_zero_instances(self, stub_tasks) -> None:
        """range(n) for n<=0 is empty; negative epochs must not crash and must yield 0."""
        train, _ = build_mixture(SFTMixture.STANDARD, mmlu_epochs=-3)
        assert _class_counts(train)["MMLU"] == 0


class TestMixtureKeyGuards:
    @pytest.mark.parametrize(
        ("mixture", "override"),
        [
            (SFTMixture.SAFETY, {"mmlu_epochs": 5}),
            (SFTMixture.STANDARD, {"wildjailbreak_epochs": 1}),
        ],
    )
    def test_out_of_set_override_raises(
        self,
        stub_tasks,
        mixture: SFTMixture,
        override: dict[str, int],
    ) -> None:
        with pytest.raises(ValueError, match="does not consume override"):
            build_mixture(mixture, **override)

    def test_declared_key_ownership_is_complete(self) -> None:
        declared = mixtures_module._MIXTURE_KEYS
        assert set(declared) == set(SFTMixture)
        assert all(keys <= set(_DEFAULTS) for keys in declared.values())
        assert set().union(*declared.values()) == set(_DEFAULTS)

    def test_only_the_steered_members_declare_steered_jsonl(self) -> None:
        declared = mixtures_module._MIXTURE_KEYS
        owners = {mixture for mixture, keys in declared.items() if "steered_jsonl" in keys}
        assert owners == STEERED_MIXTURES
        for mixture in STEERED_MIXTURES:
            assert declared[mixture] == declared[SFTMixture.STANDARD] | {"steered_jsonl"}

    @pytest.mark.parametrize("mixture", sorted(STEERED_MIXTURES))
    def test_steered_member_without_jsonl_raises(self, stub_tasks, mixture: SFTMixture) -> None:
        with pytest.raises(
            ValueError, match=rf"mixture '{mixture.value}' requires steered_jsonl=<path>"
        ):
            build_mixture(mixture)

    @pytest.mark.parametrize(
        "mixture",
        [SFTMixture.STANDARD, SFTMixture.CAPABILITIES, SFTMixture.SAFETY],
    )
    def test_non_steered_member_rejects_steered_jsonl(
        self, stub_tasks, mixture: SFTMixture, steered_jsonl: str
    ) -> None:
        with pytest.raises(ValueError, match="does not consume override"):
            build_mixture(mixture, steered_jsonl=steered_jsonl)
        with pytest.raises(ValueError, match="does not consume steered_jsonl"):
            steered_mixture_overrides(mixture, steered_jsonl)

    @pytest.mark.parametrize("mixture", sorted(STEERED_MIXTURES))
    def test_steered_overrides_helper_round_trips(
        self, mixture: SFTMixture, steered_jsonl: str
    ) -> None:
        assert steered_mixture_overrides(mixture, steered_jsonl) == {
            "steered_jsonl": steered_jsonl
        }
        with pytest.raises(ValueError, match="requires steered_jsonl=<path>"):
            steered_mixture_overrides(mixture, None)

    def test_balanced_caps_every_standard_task_and_keeps_zero_length_ones(
        self, sized_stub_tasks: dict[str, type[Task]], steered_jsonl: str
    ) -> None:
        standard_train, _ = build_mixture(SFTMixture.STANDARD)
        balanced_train, _ = build_mixture(
            SFTMixture.STANDARD_STEERED_BALANCED, steered_jsonl=steered_jsonl
        )
        # Same catalogue plus the steered task; nothing is dropped, not even the
        # constituents whose apportioned share rounds to zero.
        assert len(balanced_train) == len(standard_train) + 1
        assert _class_sequence(balanced_train)[:-1] == _class_sequence(standard_train)
        assert [len(task) for task in balanced_train[:-1]].count(0) == 5
        steered_rows = len(balanced_train[-1])
        assert sum(len(task) for task in balanced_train[:-1]) == steered_rows == 3
        # The per-task apportionment itself is pinned by the hand-checked
        # ``_EXPECTED[STANDARD_STEERED_BALANCED]`` literal in
        # ``TestDefaultEquivalencePin``; comparing it against
        # ``balanced_standard_caps`` here would check the builder's own helper
        # against itself.

    def test_balanced_cap_honours_an_existing_task_slice(
        self, steered_jsonl: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cap narrows the task's OWN slice, not its absolute ``stop``.

        Every shipped standard constituent has ``start=0, step=1``, which makes
        ``task.stop = task.start + cap * task.step`` indistinguishable from
        ``task.stop = cap``. A sliced constituent separates them: with
        ``start=3, step=2`` and cap 3 the correct stop is 9, while ``stop=cap``
        would leave the task empty.
        """
        sliced = _SlicedStub(40, start=3, step=2)
        assert len(sliced) == 19
        monkeypatch.setattr(
            mixtures_module,
            "_build_standard",
            lambda params, identity_path: ([sliced], []),
        )

        train_tasks, val_tasks = mixtures_module._build_standard_steered_balanced(
            {"steered_jsonl": steered_jsonl},
            "/unused/identity_conversations.jsonl",
        )

        assert val_tasks == []
        assert len(train_tasks[-1]) == 3
        assert sliced.stop == 9
        assert len(sliced) == 3
        # The retained rows are the sliced task's OWN first three logical rows.
        assert [sliced[index]["index"] for index in range(len(sliced))] == [3, 5, 7]

    def test_balanced_rejects_more_steered_rows_than_standard_rows(
        self, sized_stub_tasks: dict[str, type[Task]], tmp_path: Path
    ) -> None:
        # SmolTalk 17 + identity 3 + 3 + spelling 1 + 1 = 25 standard rows.
        with pytest.raises(
            ValueError, match=r"target=30 exceeds the total available rows \(25\)"
        ):
            build_mixture(
                SFTMixture.STANDARD_STEERED_BALANCED,
                steered_jsonl=_write_steered_jsonl(tmp_path / "big.jsonl", 30),
                mmlu_epochs=0,
                gsm8k_epochs=0,
                simple_spelling_size=1,
                spelling_bee_size=1,
            )

    def test_declared_but_never_read_raises(
        self, stub_tasks, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            mixtures_module._MIXTURE_KEYS,
            SFTMixture.STANDARD,
            mixtures_module._MIXTURE_KEYS[SFTMixture.STANDARD] | {"bogus_key"},
        )
        with pytest.raises(
            ValueError,
            match=r"declared-but-never-read=\['bogus_key'\], read-but-undeclared=\[\]",
        ):
            build_mixture(SFTMixture.STANDARD)

    def test_read_but_undeclared_raises(self, stub_tasks, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            mixtures_module._MIXTURE_KEYS,
            SFTMixture.STANDARD,
            mixtures_module._MIXTURE_KEYS[SFTMixture.STANDARD] - {"gsm8k_epochs"},
        )
        with pytest.raises(
            ValueError,
            match=r"declared-but-never-read=\[\], read-but-undeclared=\['gsm8k_epochs'\]",
        ):
            build_mixture(SFTMixture.STANDARD)


class TestSelectMixtureOverrides:
    def test_missing_key_raises(self) -> None:
        values = dict(_DEFAULTS)
        values.pop("gsm8k_epochs")
        with pytest.raises(
            ValueError,
            match=r"missing=\['gsm8k_epochs'\] unexpected=\[\]",
        ):
            mixtures_module.select_mixture_overrides(SFTMixture.STANDARD, values)

    def test_extra_key_raises(self) -> None:
        values = {**_DEFAULTS, "extra_key": 1}
        with pytest.raises(
            ValueError,
            match=r"missing=\[\] unexpected=\['extra_key'\]",
        ):
            mixtures_module.select_mixture_overrides(SFTMixture.STANDARD, values)

    def test_returns_exact_in_set_subset(self) -> None:
        values = dict(_DEFAULTS)
        values["mmlu_epochs"] = 9
        selected = mixtures_module.select_mixture_overrides(SFTMixture.STANDARD, values)
        declared = mixtures_module._MIXTURE_KEYS[SFTMixture.STANDARD]
        assert list(selected) == sorted(declared)
        assert selected == {key: values[key] for key in sorted(declared)}

    def test_out_of_set_default_is_tolerated_and_omitted(self) -> None:
        values = dict(_DEFAULTS)
        selected = mixtures_module.select_mixture_overrides(SFTMixture.STANDARD, values)
        assert "wildjailbreak_epochs" not in selected

    def test_out_of_set_nondefault_raises_with_ownership_details(self) -> None:
        values = dict(_DEFAULTS)
        values["wildjailbreak_epochs"] = 99
        with pytest.raises(ValueError, match="silently ignored") as exc_info:
            mixtures_module.select_mixture_overrides(SFTMixture.STANDARD, values)
        message = str(exc_info.value)
        assert "wildjailbreak_epochs" in message
        assert "standard" in message
        assert "safety" in message


# ---------------------------------------------------------------------------
# Contract lock-in on defaults (so a silent change is caught)
# ---------------------------------------------------------------------------


class TestDefaultsContract:
    def test_defaults_values(self) -> None:
        """Lock the default epoch counts so a silent halving is caught."""
        assert _DEFAULTS["mmlu_epochs"] == 1
        assert _DEFAULTS["gsm8k_epochs"] == 1
        assert _DEFAULTS["wildjailbreak_epochs"] == 2
        assert _DEFAULTS["wildguardmix_epochs"] == 3
        assert _DEFAULTS["coconot_epochs"] == 2
        assert _DEFAULTS["smoltalk_limit"] == 120000
        assert _DEFAULTS["smoltalk_safety_limit"] == 120000
        assert _DEFAULTS["simple_spelling_size"] == 200000
        assert _DEFAULTS["spelling_bee_size"] == 80000
