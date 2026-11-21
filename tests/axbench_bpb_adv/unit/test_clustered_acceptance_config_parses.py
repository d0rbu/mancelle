"""The clustered GPU acceptance config must parse and stay inside the allowlist.

Cheap guards for two failures that a GPU run was otherwise the first thing to catch.

A hand-edit to ``axbench_bpb_adv_clustered_gpu_acceptance.yaml`` once left a
duplicated ``ours_gcg:`` block and an orphaned run of ``faster_gcg`` fields at the
wrong indentation. The whole ``tests/axbench_bpb_adv`` suite -- 2526 tests -- passed
against that broken file, because nothing loaded it; the break only surfaced ~30
seconds into a very_slow GPU acceptance run as a ``yaml.scanner.ScannerError``.

These tests are the second in the trade the project already makes elsewhere: a
sub-second static check standing in front of an expensive one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from steering.eval.cluster_config import CLUSTERED_AXBENCH_BPB_ADV_METHODS

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "eval"
    / "axbench_bpb_adv"
    / "axbench_bpb_adv_clustered_gpu_acceptance.yaml"
)


@pytest.fixture(scope="module")
def acceptance_config() -> dict:
    assert _CONFIG.is_file(), f"acceptance config missing at {_CONFIG}"
    loaded = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "acceptance config must be a mapping"
    return loaded


def test_acceptance_config_is_valid_yaml(acceptance_config: dict) -> None:
    """Plain parseability. The failure this exists for is a scanner error."""
    assert acceptance_config["run_tag"] == "h3-clustered-gpu"


def test_acceptance_config_has_exactly_one_ours_gcg_block() -> None:
    """A duplicated top-level key is silent in YAML -- last wins.

    ``yaml.safe_load`` would happily accept two ``ours_gcg:`` blocks and keep only
    the second, so parseability alone does not catch the duplication that actually
    happened here. This asserts on the raw text instead.
    """
    text = _CONFIG.read_text(encoding="utf-8")
    assert text.count("\nours_gcg:") == 1, "duplicated or missing ours_gcg block"


def test_acceptance_base_methods_are_within_the_clustered_allowlist(
    acceptance_config: dict,
) -> None:
    """The config layer must not request what clustered evaluation cannot accept.

    ``validate_clustered_method_allowlist`` enforces this at runtime; asserting it
    here means a bad edit fails in a second rather than after model load.
    """
    requested = acceptance_config["base_methods"]
    assert requested, "clustered runs require an explicit non-empty allowlist"
    assert len(requested) == len(set(requested)), "duplicate method in base_methods"
    unknown = sorted(set(requested) - set(CLUSTERED_AXBENCH_BPB_ADV_METHODS))
    assert not unknown, f"methods outside the clustered allowlist: {unknown}"


def test_acceptance_diffmean_layer_matches_the_fixture_stamp(
    acceptance_config: dict,
) -> None:
    """The fixture stamps ``layer`` into every direction artifact by hand.

    ``source_validation`` requires that key, and ``_build_direction_intervention``
    compares it against ``cfg.diffmean_and_probe_layer`` at consumption time. The
    fixture therefore hardcodes ``_DIFFMEAN_LAYER = 6``, which is only correct while
    the composed config agrees. Pin it here so a config change fails in a second
    instead of ~30s into a very_slow GPU run.
    """
    # Absent from the YAML today, so the dataclass default applies.
    configured = acceptance_config.get("diffmean_and_probe_layer", 6)
    assert configured == 6, (
        "diffmean_and_probe_layer changed; update _DIFFMEAN_LAYER in "
        "tests/axbench_bpb_adv/integration/test_clustered_gpu_acceptance.py to match"
    )
