"""Identity regressions for the AxBench-BPB-Adv run tag and profiler config."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

import scripts.benchmark.profile_axbench_bpb_adv as profile_wrapper
import scripts.eval.axbench_bpb_adv_eval as cli
from scripts.eval.axbench_bpb_adv_eval import _generate_run_tag
from steering.eval import cluster_config
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.cluster_config import axbench_bpb_adv_config_to_dict

REPO_ROOT = Path(__file__).parents[3]


def test_default_config_run_tag_digest_is_frozen() -> None:
    """Freeze the current digest and require an explicit re-pin audit.

    This literal is deliberately brittle: it fails whenever a newly added or
    changed default survives the shared serializer. Managed optional fields
    that default to null, including G-stream-B's planned
    ``feature_extraction_model_id``, are intentionally byte-neutral and do not
    trip this alarm. Re-pin only after confirming
    ``test_run_tag_is_invariant_under_null_optional_fields`` still passes and
    auditing both whole-dataclass hash sites in the eval script.
    """

    tag = _generate_run_tag(AxBenchBpbAdvConfig())

    # Re-pinned 2026-08-03 for #1245's `OursGcgConfig.max_grad_batch`, per the
    # audit this docstring prescribes: the null-optional invariance test still
    # passes, and both whole-dataclass hash sites (the rank-0 and broadcast
    # branches at _generate_run_tag's call sites) go through this one generator,
    # so they cannot diverge.
    #
    # The field is NOT byte-neutral, and that is correct rather than incidental:
    # it is stamped into the GCG bag identity for the same reason
    # `max_forward_batch` is, so two runs differing only in it can derive
    # different bags and must not share an auto-generated tag.
    assert tag.rsplit("-", 1)[1] == "ef1fed3a"


def test_serializer_strips_real_config_null_cluster_fields() -> None:
    """G2c's real dataclass retains the legacy serialized config shape."""

    cfg = AxBenchBpbAdvConfig()
    expected = dataclasses.asdict(cfg)
    assert expected.pop("feature_mask") is None
    assert expected.pop("feature_clusters") is None
    assert expected.pop("feature_extraction_model_id") is None
    assert expected.pop("force_cluster_shard_revalidation") is False
    # Clustered selection provenance, added with the BPB clustered preflight fix.
    # Managed-optional like the three above: stripped when null so an existing
    # config keeps its run tag instead of orphaning its artifacts.
    assert expected.pop("selection_splice_batch_size") is None
    assert expected.pop("selection_manifest_path") is None
    # PRE-EXISTING GAP (not from the alpha work): #1249 added this strip to the
    # serializer but never to this guard, so the guard has been red on main.
    # Same rule as the rest -- an empty exclusion list keeps existing tags.
    assert expected.pop("exclude_concept_modes") == []
    # Steering-coefficient pinning: null means the full default grid, which is
    # what every existing run swept, so null must not enter the tag.
    assert expected.pop("steering_coefficient_grid") is None
    # Class-margin alpha: the default grid makes the derive walk exactly the
    # pre-alpha reg_lambda-only sequence, so it must not enter the tag either.
    # Nested one level down -- this ONE key would otherwise rehash every tag.
    assert expected["ours_gcg"].pop("objective_alpha_grid") == (1.0,)
    faster_gcg = next(
        method for method in expected["adversarial_methods"]
        if method["method"] == "faster_gcg"
    )
    assert faster_gcg.pop("candidate_eval_device_batch_size") is None

    assert axbench_bpb_adv_config_to_dict(cfg) == expected


def test_serializer_records_nondefault_candidate_eval_batch_size() -> None:
    cfg = AxBenchBpbAdvConfig(
        adversarial_methods=[
            dataclasses.replace(
                AxBenchBpbAdvConfig().adversarial_methods[0],
                candidate_eval_device_batch_size=32,
            )
        ]
    )

    faster_gcg = axbench_bpb_adv_config_to_dict(cfg)["adversarial_methods"][0]
    assert faster_gcg["candidate_eval_device_batch_size"] == 32


def test_force_revalidation_true_is_recorded_in_serialized_config() -> None:
    cfg = AxBenchBpbAdvConfig(force_cluster_shard_revalidation=True)

    assert axbench_bpb_adv_config_to_dict(cfg)[
        "force_cluster_shard_revalidation"
    ] is True


def test_run_tag_is_invariant_under_null_optional_fields() -> None:
    @dataclasses.dataclass(frozen=True)
    class _LegacyConfig:
        run_tag: str = "legacy"
        seed: int = 17

    @dataclasses.dataclass(frozen=True)
    class _ExtendedConfig:
        run_tag: str = "legacy"
        seed: int = 17
        feature_mask: dict[str, str] | None = None
        feature_clusters: dict[str, str] | None = None
        feature_extraction_model_id: str | None = None

    assert axbench_bpb_adv_config_to_dict(_ExtendedConfig()) == dataclasses.asdict(_LegacyConfig())

    feature_mask = {"mode": "whitelist", "path": "/mask.txt"}
    feature_clusters = {"path": "/clusters.yaml"}
    populated = axbench_bpb_adv_config_to_dict(
        _ExtendedConfig(
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
            feature_extraction_model_id="google/gemma-2-2b",
        )
    )
    assert populated["feature_mask"] == feature_mask
    assert populated["feature_clusters"] == feature_clusters
    assert populated["feature_extraction_model_id"] == "google/gemma-2-2b"


def test_shared_serializer_binding_is_genuine() -> None:
    """SEMANTIC guard: the name must be bound to the real shared serializer.

    This is the load-bearing anti-drift check. Both independent reviewers of
    this PR defeated a purely syntactic scan with::

        from dataclasses import asdict as axbench_bpb_adv_config_to_dict

    which reinstates the raw whole-dataclass hash at every call site while
    leaving the call spelling untouched -- all four tests passed. Comparing the
    bound object by identity closes that hole, and with it module aliases,
    ``getattr``, ``functools.partial`` and same-name rebinding, none of which a
    spelling-based pattern match can see.
    """

    for module in (cli, profile_wrapper):
        assert (
            module.axbench_bpb_adv_config_to_dict
            is cluster_config.axbench_bpb_adv_config_to_dict
        ), module.__name__


@pytest.mark.parametrize(
    ("relative_path", "expected_shared_calls"),
    [
        ("scripts/eval/axbench_bpb_adv_eval.py", 2),
        ("scripts/benchmark/profile_axbench_bpb_adv.py", 1),
    ],
)
def test_no_raw_whole_config_serialization_remains(
    relative_path: str,
    expected_shared_calls: int,
) -> None:
    """Structural companion: no raw ``asdict`` on a WHOLE-CONFIG variable.

    Deliberately argument-aware rather than a blanket ban: the profiler wrapper
    legitimately calls ``dataclasses.asdict(method)`` on a *different* dataclass
    (each ``AdversarialMethodConfig``), which must stay raw. Only serialization
    of the whole ``AxBenchBpbAdvConfig`` has to go through the shared
    null-stripping serializer.

    Scope note: this catches the reverted-spelling regression. It does NOT catch
    aliasing or rebinding -- ``test_shared_serializer_binding_is_genuine`` above
    is what covers those.
    """

    config_vars = {"cfg", "cfg_obj"}
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    raw_whole_config: list[int] = []
    shared_calls: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_raw_asdict = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dataclasses"
            and node.func.attr == "asdict"
        ) or (isinstance(node.func, ast.Name) and node.func.id == "asdict")
        targets_whole_config = (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in config_vars
        )
        if is_raw_asdict and targets_whole_config:
            raw_whole_config.append(node.lineno)
        if isinstance(node.func, ast.Name) and node.func.id == "axbench_bpb_adv_config_to_dict":
            shared_calls.append(node.lineno)

    assert raw_whole_config == [], (relative_path, raw_whole_config)
    assert len(shared_calls) == expected_shared_calls, (relative_path, shared_calls)
