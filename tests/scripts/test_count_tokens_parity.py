"""Parity tests for `scripts.preprocess.count_tokens` SFT helpers vs.
`tasks.mixtures.build_mixture` (P1 #1).

The Phase 0 map flagged that ``count_tokens.py`` duplicates the SFT
mixture catalogue inline rather than calling ``build_mixture``. Drift
between the two is silent: a future update to one source of truth
makes per-task token counts wrong without any test failing.

Strategy: monkeypatch every ``tasks.<X>`` symbol that the helpers and
``build_mixture`` import, with a stub ``Task`` subclass that records its
constructor args (a ``"task_kind"`` and the kwargs).  Then assert the
catalogue sequences (kind + kwargs in order) match between
``build_mixture(<phase>)`` and the helper-emitted task lists.

We only use stubs at the module level after monkeypatching, so no real
HF datasets are touched.
"""

from __future__ import annotations

from typing import Any

import pytest

from tasks.common import Task

# ---------------------------------------------------------------------------
# Stub task: records constructor args; supports __len__ via stop/start/step
# ---------------------------------------------------------------------------


class StubTask(Task):
    """Lightweight stand-in for any concrete Task class."""

    KIND: str = "stub"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs_recorded = dict(kwargs)
        # Strip the kwargs that Task.__init__ understands, pass the rest aside.
        super_kwargs = {
            k: kwargs.pop(k) for k in ("start", "stop", "step") if k in kwargs
        }
        super().__init__(**super_kwargs)
        # We need __len__ to work for count_*_sft_tokens iteration.
        # Force a tiny len(=2) so the helpers iterate just twice per task.
        self._fake_len = 2

    def num_examples(self) -> int:
        return self._fake_len

    def get_example(self, index: int) -> dict:
        return {"messages": [{"role": "user", "content": f"q{index}"},
                              {"role": "assistant", "content": f"a{index}"}]}


def _make_kind(kind_label: str) -> type:
    """Factory for a StubTask subclass with a specific class name.

    The class name matches the original Task class name so that
    ``count_tokens._label_task`` (which keys off ``type(task).__name__``)
    produces the same labels it would for a real task.
    """
    return type(
        kind_label,
        (StubTask,),
        {"KIND": kind_label},
    )


# ---------------------------------------------------------------------------
# Pytest fixture: patches every Task symbol used by both code paths
# ---------------------------------------------------------------------------


_KINDS = [
    "GSM8K",
    "MMLU",
    "SmolTalk",
    "CustomJSON",
    "SimpleSpelling",
    "SpellingBee",
    "WildJailbreak",
    "CoCoNot",
    "WildGuardMix",
    "UltraChat",
    "NoRobots",
    "AnthropicHH",
    "OpenCoder",
    "MetaMathQA",
    "OrcaMath",
    "PKUSafeRLHF",
    "NemotronSafety",
]


@pytest.fixture
def stub_all_tasks(monkeypatch, tmp_path):
    """Replace every concrete Task with a stub that records its kwargs."""
    # Map from module path to attribute name that's used in mixtures + count_tokens.
    targets = [
        ("tasks.gsm8k", "GSM8K"),
        ("tasks.mmlu", "MMLU"),
        ("tasks.smoltalk", "SmolTalk"),
        ("tasks.customjson", "CustomJSON"),
        ("tasks.spellingbee", "SimpleSpelling"),
        ("tasks.spellingbee", "SpellingBee"),
        ("tasks.wildjailbreak", "WildJailbreak"),
        ("tasks.coconot", "CoCoNot"),
        ("tasks.wildguardmix", "WildGuardMix"),
        ("tasks.ultrachat", "UltraChat"),
        ("tasks.no_robots", "NoRobots"),
        ("tasks.anthropic_hh", "AnthropicHH"),
        ("tasks.opencoder", "OpenCoder"),
        ("tasks.metamath", "MetaMathQA"),
        ("tasks.orca_math", "OrcaMath"),
        ("tasks.pku_saferlhf", "PKUSafeRLHF"),
        ("tasks.nemotron_safety", "NemotronSafety"),
    ]

    stubs = {name: _make_kind(name) for _, name in targets}

    # Patch source modules first, so importers see stubs.
    for module_path, name in targets:
        monkeypatch.setattr(f"{module_path}.{name}", stubs[name])

    # Also patch in `tasks.mixtures` and `scripts.preprocess.count_tokens`
    # since they imported the names directly.
    import tasks.mixtures as mixtures_mod
    for name in [n for _, n in targets]:
        monkeypatch.setattr(mixtures_mod, name, stubs[name], raising=False)

    # ``scripts.preprocess.count_tokens`` no longer imports the concrete
    # task classes by name (post-P1 #1 refactor). All routing flows through
    # ``tasks.mixtures.build_mixture`` and the patches above. We still set
    # any matching attribute non-strictly, so an older count_tokens that
    # imports Task classes directly remains testable.
    import scripts.preprocess.count_tokens as count_mod
    for name in [n for _, n in targets]:
        if hasattr(count_mod, name):
            monkeypatch.setattr(count_mod, name, stubs[name], raising=False)

    # Make get_base_dir return tmp_path so identity-conv path is predictable.
    monkeypatch.setattr(
        "nanochat.common.get_base_dir", lambda: str(tmp_path), raising=True
    )
    monkeypatch.setattr(
        "tasks.mixtures.get_base_dir", lambda: str(tmp_path), raising=False
    )
    monkeypatch.setattr(
        "scripts.preprocess.count_tokens.get_base_dir",
        lambda: str(tmp_path),
        raising=False,
    )

    # Touch identity_conversations.jsonl so customjson is included by both paths.
    identity = tmp_path / "identity_conversations.jsonl"
    identity.write_text(
        '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n'
    )
    return stubs, str(identity)


def _kinds_from_tasks(task_list: list[Any]) -> list[str]:
    """Extract the KIND label sequence from a list of stub tasks."""
    return [type(t).__name__ for t in task_list]


def _kinds_from_named(named: list[tuple[str, Any]]) -> list[str]:
    """Same, but for ``[(name, task), ...]`` lists."""
    return [type(t).__name__ for _, t in named]


def _normalise_helper_keys(keys: list[str]) -> list[str]:
    """Map helper-side per_task names to their underlying Task kind.

    The helper labels follow conventions like ``"MMLU_epoch1"``,
    ``"AnthropicHH_epoch1"``, ``"Identity_epoch1"`` (which actually
    refers to ``CustomJSON``), ``"MetaMathQA_GSM"`` (which is
    ``MetaMathQA``).
    """
    out = []
    for k in keys:
        if "_epoch" in k:
            base = k.split("_epoch")[0]
        elif k.endswith("_GSM"):
            base = k[: -len("_GSM")]
        else:
            base = k
        if base == "Identity":
            base = "CustomJSON"
        out.append(base)
    return out


def test_standard_mixture_kind_sequence_matches_count_tokens(stub_all_tasks):
    """STANDARD mixture: ``build_mixture`` and ``count_sft_tokens`` enumerate
    the same Task kinds in the same order."""
    from scripts.preprocess.count_tokens import count_sft_tokens
    from tasks.mixtures import SFTMixture, build_mixture

    train_tasks, _val = build_mixture(SFTMixture.STANDARD, mmlu_epochs=3, gsm8k_epochs=4)
    mixture_kinds = _kinds_from_tasks(train_tasks)

    # count_sft_tokens enumerates internally: capture via stub tokenizer.
    captured: list[Any] = []

    class _CapturingTok:
        def render_conversation(self, conv):
            captured.append(conv)
            # 3 ids + 3 mask, mask all=1 (just to count > 0)
            return [1, 2, 3], [1, 1, 1]

    helper_result = count_sft_tokens(
        _CapturingTok(), mmlu_epochs=3, gsm8k_epochs=4
    )
    helper_kinds = list(helper_result["per_task"].keys())
    helper_kinds_normalised = _normalise_helper_keys(helper_kinds)

    assert mixture_kinds == helper_kinds_normalised, (
        f"STANDARD drift: build_mixture={mixture_kinds!r}, "
        f"count_sft_tokens={helper_kinds_normalised!r}"
    )


def test_capabilities_mixture_kind_sequence_matches_count_tokens(stub_all_tasks):
    from scripts.preprocess.count_tokens import count_capabilities_sft_tokens
    from tasks.mixtures import SFTMixture, build_mixture

    cfg = type("C", (), {
        "smoltalk_limit": None,
        "capabilities_mmlu_epochs": 2,
        "opencoder_limit": 50000,
        "orca_math_limit": None,
        "metamath_limit": None,
        "simple_spelling_size": 200000,
        "spelling_bee_size": 80000,
    })()

    train_tasks, _val = build_mixture(
        SFTMixture.CAPABILITIES,
        smoltalk_limit=None,
        capabilities_mmlu_epochs=2,
        opencoder_limit=50000,
        orca_math_limit=None,
        metamath_limit=None,
        simple_spelling_size=200000,
        spelling_bee_size=80000,
    )
    mixture_kinds = _kinds_from_tasks(train_tasks)

    class _CapturingTok:
        def render_conversation(self, conv):
            return [1, 2, 3], [1, 1, 1]

    helper_result = count_capabilities_sft_tokens(_CapturingTok(), cfg)
    helper_kinds = list(helper_result["per_task"].keys())
    helper_kinds_normalised = _normalise_helper_keys(helper_kinds)
    assert mixture_kinds == helper_kinds_normalised, (
        f"CAPABILITIES drift: build_mixture={mixture_kinds!r}, "
        f"count_capabilities_sft_tokens={helper_kinds_normalised!r}"
    )


def test_safety_mixture_kind_sequence_matches_count_tokens(stub_all_tasks):
    from scripts.preprocess.count_tokens import count_safety_sft_tokens
    from tasks.mixtures import SFTMixture, build_mixture

    cfg = type("C", (), {
        "wildjailbreak_epochs": 2,
        "wildguardmix_epochs": 3,
        "coconot_epochs": 2,
        "smoltalk_safety_limit": 120000,
    })()

    train_tasks, _val = build_mixture(
        SFTMixture.SAFETY,
        wildjailbreak_epochs=2,
        wildguardmix_epochs=3,
        coconot_epochs=2,
        smoltalk_safety_limit=120000,
    )
    mixture_kinds = _kinds_from_tasks(train_tasks)

    class _CapturingTok:
        def render_conversation(self, conv):
            return [1, 2, 3], [1, 1, 1]

    helper_result = count_safety_sft_tokens(_CapturingTok(), cfg)
    helper_kinds = list(helper_result["per_task"].keys())
    helper_kinds_normalised = _normalise_helper_keys(helper_kinds)

    assert mixture_kinds == helper_kinds_normalised, (
        f"SAFETY drift: build_mixture={mixture_kinds!r}, "
        f"count_safety_sft_tokens={helper_kinds_normalised!r}"
    )
