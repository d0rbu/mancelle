"""Config and source-wiring tests for selective chat SFT mixture params."""

from __future__ import annotations

from pathlib import Path

import pytest

from tasks.mixtures import (
    _DEFAULTS,
    MIXTURE_PARAM_KEYS,
    STEERED_MIXTURES,
    SFTMixture,
    select_mixture_overrides,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs" / "training")


def _compose_variant(hydra_compose, config_name: str):
    """Compose a mixture variant standalone.

    These variants carry ``defaults: [chat_sft, _self_]``, so composing the
    name alone yields the full runnable config. They previously did not, and
    this helper had to merge the base in by hand — which meant
    ``chat_sft config_name=chat_sft_safety`` died on the first ``cfg.*`` access
    even though the test suite was green.
    """
    return hydra_compose(_CONFIG_DIR, config_name)


def test_mixture_variants_compose_standalone(hydra_compose) -> None:
    """Each variant must be runnable by name, not just as a merge overlay."""
    for config_name, expected_mixture in (
        ("chat_sft_safety", "safety"),
        ("chat_sft_capabilities", "capabilities"),
    ):
        cfg = hydra_compose(_CONFIG_DIR, config_name)
        assert cfg.mixture == expected_mixture
        # The keys chat_sft.py reads at startup, before any mixture work.
        for key in ("source", "run", "device_type", "dtype", "max_seq_len", "sft_loss_mask"):
            assert key in cfg, f"{config_name} is missing {key!r} required by chat_sft.py"


def test_default_config_mixture_params_match_code_defaults(hydra_compose) -> None:
    cfg = hydra_compose(_CONFIG_DIR, "chat_sft")
    assert {key: cfg[key] for key in sorted(MIXTURE_PARAM_KEYS)} == dict(sorted(_DEFAULTS.items()))


def test_capabilities_variant_selects_only_consumed_params(hydra_compose) -> None:
    cfg = _compose_variant(hydra_compose, "chat_sft_capabilities")
    selected = select_mixture_overrides(
        SFTMixture.CAPABILITIES,
        {key: cfg[key] for key in MIXTURE_PARAM_KEYS},
    )
    assert selected["smoltalk_limit"] == 120000
    assert selected["capabilities_mmlu_epochs"] == 2
    assert selected["opencoder_limit"] == 50000


def test_safety_variant_selects_only_consumed_params(hydra_compose) -> None:
    cfg = _compose_variant(hydra_compose, "chat_sft_safety")
    selected = select_mixture_overrides(
        SFTMixture.SAFETY,
        {key: cfg[key] for key in MIXTURE_PARAM_KEYS},
    )
    assert set(selected) == {
        "coconot_epochs",
        "smoltalk_safety_limit",
        "wildguardmix_epochs",
        "wildjailbreak_epochs",
    }


@pytest.mark.parametrize("mixture", sorted(STEERED_MIXTURES))
def test_steered_mixtures_select_the_steered_jsonl(
    hydra_compose, mixture: SFTMixture
) -> None:
    """chat_sft forwards steered_jsonl automatically: it is a declared key."""
    cfg = hydra_compose(_CONFIG_DIR, "chat_sft")
    values = {key: cfg[key] for key in MIXTURE_PARAM_KEYS}
    values["steered_jsonl"] = "/data/steered.jsonl"

    selected = select_mixture_overrides(mixture, values)

    assert selected["steered_jsonl"] == "/data/steered.jsonl"
    assert set(selected) == set(
        select_mixture_overrides(SFTMixture.STANDARD, {**values, "steered_jsonl": None})
    ) | {"steered_jsonl"}


@pytest.mark.parametrize(
    "mixture", [SFTMixture.STANDARD, SFTMixture.CAPABILITIES, SFTMixture.SAFETY]
)
def test_non_steered_mixtures_reject_a_set_steered_jsonl(
    hydra_compose, mixture: SFTMixture
) -> None:
    cfg = hydra_compose(_CONFIG_DIR, "chat_sft")
    values = {key: cfg[key] for key in MIXTURE_PARAM_KEYS}
    assert values["steered_jsonl"] is None
    values["steered_jsonl"] = "/data/steered.jsonl"

    with pytest.raises(ValueError, match="silently ignored") as exc_info:
        select_mixture_overrides(mixture, values)
    assert "steered_jsonl" in str(exc_info.value)


def test_chat_sft_source_uses_selective_forwarding() -> None:
    source = (_REPO_ROOT / "scripts" / "train" / "chat_sft.py").read_text()
    assert "select_mixture_overrides(" in source
    assert "build_task_mixtures(mixture, **mixture_params)" in source
    assert "chat_sft requires a val split" in source
    assert "smoltalk_limit=cfg.smoltalk_limit" not in source
