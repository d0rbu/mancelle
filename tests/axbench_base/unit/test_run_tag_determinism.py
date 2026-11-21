"""Slice 16 audit — pin ``_generate_run_tag`` determinism (map §7.2 hole).

Map §7.2 coverage holes table flagged "no test that two successive
runs with the same cfg in the same UTC second produce the same tag.
Pin the determinism contract."

This test exercises ``_generate_run_tag`` with a frozen ``datetime``
(via ``monkeypatch``) so the timestamp portion is constant; the
hash portion is then verified to be:

- Identical for identical configs (within the same UTC second).
- Different for configs that differ in an identity-bearing field
  (i.e. one that survives ``axbench_base_config_to_dict``).
- A predictable ``YYYYMMDD-HHMMSS-<8 hex>`` shape.

Not "any field": a ``conditioning`` mapping with an explicit
``feature_clusters: None`` and one omitting the key are different config
values that deliberately share a suffix, because the serializer drops the
disabled key. See
``test_run_tag_omits_null_feature_clusters_without_silent_break``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

import scripts.eval.axbench_base_eval as cli
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.cluster_config import axbench_base_config_to_dict

FIXED: dict[str, Any] = {
    "feature_dir": "out/features/s128/l8/a_relu/p_max/f16384/m_gemma-2-2b/base",
    "feature_statistics_path": "out/features/statistics.json",
    "decoder_only": {"source": "local", "model_tag": "d12", "step": 54600},
    "encdec": {"source": "local", "model_tag": "d12_encdec", "step": 54600},
    "device": "cpu",
    "output_root": "out/axbench_base",
}

# Representative subset, not a literal copy of the production YAML block (which
# also has ``identity``); the literal suffix goldens are pinned to this fixture.
CONDITIONING_WITHOUT_CLUSTER_KEY: dict[str, Any] = {
    "context_cache_dir": None,
    "n_sae_features": 16384,
    "num_buckets": 2,
    "feature_mask": None,
}


def _fixed_cfg(conditioning: dict[str, Any] | None) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(**FIXED, conditioning=conditioning)


def _run_tag_suffix(cfg: AxBenchBaseConfig) -> str:
    return cli._generate_run_tag(cfg).split("-")[-1]


def _legacy_config_payload(cfg: AxBenchBaseConfig) -> dict[str, Any]:
    """Return the payload from before selection overrides and the force flag."""

    payload = dataclasses.asdict(cfg)
    # ``encdec_fit_bag_store`` (#1249) postdates these tags; at its None default
    # it is stripped from identity exactly like the selection overrides below.
    # Assert-before-pop so this helper cannot mask a changed default.
    for key in (
        "selection_splice_batch_size",
        "selection_manifest_path",
        "encdec_fit_bag_store",
    ):
        assert payload[key] is None
        payload.pop(key)
    assert payload.pop("force_cluster_shard_revalidation") is False
    # ``transport_retry_budget_seconds`` postdates these tags too. It is pure
    # transport policy (how long a call waits out an upstream 429) and never
    # changes a prompt or a completion, so at its default it is stripped from
    # identity rather than retagging every pre-existing run. Assert the value
    # before dropping it, so this helper cannot mask a changed default.
    for section, legacy_default in (
        ("judge", 0.0),
        ("synth", 300.0),
        ("concept_examples_qc_judge", 0.0),
    ):
        client = payload[section]
        assert isinstance(client, dict)
        assert client["transport_retry_budget_seconds"] == legacy_default
        client.pop("transport_retry_budget_seconds")
    return payload


def _make_cfg(**overrides: object) -> AxBenchBaseConfig:
    """Build a valid AxBenchBaseConfig with sensible mandatory fields.

    We construct the dataclass directly (rather than spreading a dict)
    so that ``ty``'s typed-fields constraint is satisfied. Overrides
    are applied via ``replace`` after construction.
    """
    cfg = AxBenchBaseConfig(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        ),
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cuda",
        output_root="out/eval/axbench_base",
        run_tag=None,
    )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


def test_run_tag_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run-tag matches ``YYYYMMDD-HHMMSS-<8hex>`` format."""
    fixed = datetime(2026, 5, 5, 12, 34, 56, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cli, "_DateTime", _FrozenDateTime)

    cfg = _make_cfg()
    tag = cli._generate_run_tag(cfg)
    # Format: YYYYMMDD-HHMMSS-<8hex> = 24 chars (8+1+6+1+8)
    assert len(tag) == 24
    assert tag.startswith("20260505-123456-")
    suffix = tag.split("-")[-1]
    assert len(suffix) == 8
    # Suffix is hex.
    int(suffix, 16)


def test_same_config_same_second_yields_same_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls with the same config in the same UTC second yield the same tag."""
    fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cli, "_DateTime", _FrozenDateTime)

    cfg = _make_cfg()
    a = cli._generate_run_tag(cfg)
    b = cli._generate_run_tag(cfg)
    assert a == b


def test_different_config_yields_different_hash_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configs differing in an identity-bearing field get distinct suffixes.

    This varies ``device``, which the serializer always retains, so the
    property holds. It is deliberately NOT a claim about "any field" -- a
    null ``conditioning.feature_clusters`` is dropped by
    ``axbench_base_config_to_dict`` and so does not change the suffix.
    """
    fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cli, "_DateTime", _FrozenDateTime)

    cfg_a = _make_cfg(device="cuda")
    cfg_b = _make_cfg(device="cpu")
    tag_a = cli._generate_run_tag(cfg_a)
    tag_b = cli._generate_run_tag(cfg_b)
    # Same timestamp prefix, different hash suffix.
    prefix_a, suffix_a = tag_a.rsplit("-", 1)
    prefix_b, suffix_b = tag_b.rsplit("-", 1)
    assert prefix_a == prefix_b
    assert suffix_a != suffix_b


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("accept_threshold", 0.2),
        ("positive_prefix_headroom_ceiling", 0.8),
        ("min_prefix_chars", 99),
        ("max_prefix_chars", 1024),
        ("positive_max_attempts_per_seed", 7),
        ("positive_zero_hit_bail_after", None),
        ("max_concept_drop_fraction", 0.4),
        ("max_dropped_concept_fraction", 0.4),
    ],
)
def test_positive_qc_contract_fields_change_auto_run_tag_hash(
    field_name,
    value,
):
    cfg = _make_cfg()
    changed = dataclasses.replace(cfg, **{field_name: value})

    assert _run_tag_suffix(changed) != _run_tag_suffix(cfg)
    assert axbench_base_config_to_dict(changed)[field_name] == value


def test_dict_field_insertion_order_does_not_affect_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map P2-1: ``json.dumps(..., sort_keys=True)`` makes dict insertion
    order irrelevant. This test pins that contract by passing the
    nested ``decoder_only`` dict in two different insertion orders.
    """
    fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(cli, "_DateTime", _FrozenDateTime)

    cfg_a = _make_cfg(
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
    )
    cfg_b = _make_cfg(
        # Same fields, different insertion order.
        decoder_only={"step": 10000, "source": "base", "model_tag": "d13"},
    )
    tag_a = cli._generate_run_tag(cfg_a)
    tag_b = cli._generate_run_tag(cfg_b)
    assert tag_a == tag_b, (
        f"Map P2-1: insertion-order independence violated. "
        f"a={tag_a!r}, b={tag_b!r}"
    )


@pytest.mark.parametrize(
    "conditioning",
    [
        None,
        CONDITIONING_WITHOUT_CLUSTER_KEY,
    ],
)
def test_run_tag_helper_is_universally_legacy_equivalent_without_cluster_key(
    conditioning: dict[str, Any] | None,
) -> None:
    """Pin ``axbench_base_config_to_dict`` behavior, not call-site wiring.

    This catches a future change to that helper. It does not by itself detect
    a revert of the call-site migration: without a ``feature_clusters`` key or
    enabled selection override, the helper matches the dataclass schema from
    before those fields existed. Of the prior tests, the revert-sensitive
    call-site guards are
    ``test_run_tag_omits_null_feature_clusters_without_silent_break`` and
    ``test_profiler_payload_uses_cluster_aware_config_serialization``.
    """

    cfg = _fixed_cfg(conditioning)
    legacy_suffix = hashlib.sha256(
        json.dumps(_legacy_config_payload(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    assert _run_tag_suffix(cfg) == legacy_suffix


@pytest.mark.parametrize(
    ("conditioning", "expected_suffix"),
    [
        (None, "9751e4bc"),
        (CONDITIONING_WITHOUT_CLUSTER_KEY, "6fad21c8"),
    ],
)
def test_run_tag_legacy_literal_goldens(
    conditioning: dict[str, Any] | None,
    expected_suffix: str,
) -> None:
    """Pin ``axbench_base_config_to_dict`` with external legacy goldens.

    This catches a future change to that helper. It does not by itself detect
    a revert of the call-site migration: without a ``feature_clusters`` key or
    enabled selection override, the helper matches the legacy dataclass schema,
    so the golden holds whichever function production calls. Of the prior
    tests, the revert-sensitive call-site guards are
    ``test_run_tag_omits_null_feature_clusters_without_silent_break`` and
    ``test_profiler_payload_uses_cluster_aware_config_serialization``.
    """

    assert _run_tag_suffix(_fixed_cfg(conditioning)) == expected_suffix


def test_run_tag_omits_null_feature_clusters_without_silent_break() -> None:
    """A future null YAML default preserves the existing unclustered identity."""

    conditioning = {
        **CONDITIONING_WITHOUT_CLUSTER_KEY,
        "feature_clusters": None,
    }
    cfg = _fixed_cfg(conditioning)
    suffix = _run_tag_suffix(cfg)
    old_formula_suffix = hashlib.sha256(
        json.dumps(_legacy_config_payload(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]

    assert suffix == "6fad21c8"
    # This is the distinct value produced by retaining the disabled
    # feature-clusters key under the current config schema.
    assert old_formula_suffix == "ad1fc9a3"
    assert suffix != old_formula_suffix


def test_run_tag_retains_real_feature_clusters_and_matches_legacy_formula() -> None:
    """Pin real-cluster retention in ``axbench_base_config_to_dict``.

    This catches a future change to that helper. It does not by itself detect
    a revert of the call-site migration: this real-value case is unchanged
    relative to the legacy schema because only disabled fields are stripped.
    Thus the equality holds whichever function production calls. Of the prior
    tests, the revert-sensitive call-site guards are
    ``test_run_tag_omits_null_feature_clusters_without_silent_break`` and
    ``test_profiler_payload_uses_cluster_aware_config_serialization``.
    """

    conditioning = {
        **CONDITIONING_WITHOUT_CLUSTER_KEY,
        "feature_clusters": {"path": "clusters.yaml"},
    }
    cfg = _fixed_cfg(conditioning)
    legacy_suffix = hashlib.sha256(
        json.dumps(_legacy_config_payload(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    helper_suffix = hashlib.sha256(
        json.dumps(axbench_base_config_to_dict(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    suffix = _run_tag_suffix(cfg)

    assert suffix == "a32fed53"
    assert legacy_suffix == helper_suffix == suffix


def test_run_tag_retains_real_selection_manifest_overrides() -> None:
    """Configured selection identity inputs remain in the serialized payload."""

    cfg = dataclasses.replace(
        _fixed_cfg(CONDITIONING_WITHOUT_CLUSTER_KEY),
        selection_splice_batch_size=64,
        selection_manifest_path="packed/sel/example/manifest.json",
    )

    payload = axbench_base_config_to_dict(cfg)

    assert payload["selection_splice_batch_size"] == 64
    assert payload["selection_manifest_path"] == "packed/sel/example/manifest.json"


def test_profiler_payload_uses_cluster_aware_config_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real ``_run`` metadata payload preserves absent/null identity."""

    captured_metadata: list[dict[str, Any]] = []
    dispatched_configs: list[AxBenchBaseConfig] = []

    class _CapturingProfiler:
        def __init__(
            self,
            _output_dir: Path,
            *,
            metadata: dict[str, Any],
            **_kwargs: Any,
        ) -> None:
            captured_metadata.append(metadata)

        def __enter__(self) -> _CapturingProfiler:
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    class _FakeRuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "AxBenchProfiler", _CapturingProfiler)
    monkeypatch.setattr(cli, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(cli, "validate_env_for_stages", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda *, device_type: (False, 0, 0, 1, device_type),
    )
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(
        cli,
        "_dispatch_stage",
        lambda cfg, _state: dispatched_configs.append(cfg),
    )

    for conditioning in (
        CONDITIONING_WITHOUT_CLUSTER_KEY,
        {**CONDITIONING_WITHOUT_CLUSTER_KEY, "feature_clusters": None},
    ):
        raw_cfg = {
            **FIXED,
            "conditioning": conditioning,
            "run_tag": "profile-payload-test",
            "stage": "0a",
            "profile": {
                "enabled": True,
                "gpu_utilization": False,
                "synchronize_cuda": False,
            },
        }
        cli._run(OmegaConf.create(raw_cfg))

    assert len(captured_metadata) == len(dispatched_configs) == 2

    no_key_payload = captured_metadata[0]["config"]
    no_key_cfg = dispatched_configs[0]
    assert no_key_payload == axbench_base_config_to_dict(no_key_cfg)
    assert no_key_payload == _legacy_config_payload(no_key_cfg)

    null_key_payload = captured_metadata[1]["config"]
    null_key_cfg = dispatched_configs[1]
    assert null_key_payload == axbench_base_config_to_dict(null_key_cfg)
    assert "feature_clusters" not in null_key_payload["conditioning"]


def test_auto_run_tag_and_profiler_payload_share_cluster_aware_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise both migrated call sites in one profiling-enabled ``_run``."""

    fixed = datetime(2026, 7, 26, 12, 34, 56, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    captured_metadata: list[dict[str, Any]] = []
    dispatched_configs: list[AxBenchBaseConfig] = []

    class _CapturingProfiler:
        def __init__(
            self,
            _output_dir: Path,
            *,
            metadata: dict[str, Any],
            **_kwargs: Any,
        ) -> None:
            captured_metadata.append(metadata)

        def __enter__(self) -> _CapturingProfiler:
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            return None

    class _FakeRuntimeState:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "_DateTime", _FrozenDateTime)
    monkeypatch.setattr(cli, "AxBenchProfiler", _CapturingProfiler)
    monkeypatch.setattr(cli, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(cli, "validate_config", lambda _cfg: None)
    monkeypatch.setattr(cli, "validate_env_for_stages", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "compute_init",
        lambda *, device_type: (False, 0, 0, 1, device_type),
    )
    monkeypatch.setattr(cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(
        cli,
        "_dispatch_stage",
        lambda cfg, _state: dispatched_configs.append(cfg),
    )

    old_formula_suffixes: list[str] = []
    for conditioning in (
        CONDITIONING_WITHOUT_CLUSTER_KEY,
        {**CONDITIONING_WITHOUT_CLUSTER_KEY, "feature_clusters": None},
    ):
        raw_cfg = OmegaConf.create(
            {
                **FIXED,
                "conditioning": conditioning,
                "run_tag": None,
                "stage": "0a",
                "profile": {
                    "enabled": True,
                    "gpu_utilization": False,
                    "synchronize_cuda": False,
                },
            }
        )
        cfg_obj_with_run_tag_none = cli.cfg_to_dataclass(raw_cfg)
        assert cfg_obj_with_run_tag_none.run_tag is None
        old_formula_suffixes.append(
            hashlib.sha256(
                json.dumps(
                    dataclasses.asdict(cfg_obj_with_run_tag_none),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:8]
        )
        cli._run(raw_cfg)

    assert len(captured_metadata) == len(dispatched_configs) == 2

    run_tags = [metadata["run_tag"] for metadata in captured_metadata]
    assert all(
        re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_tag)
        for run_tag in run_tags
    )
    assert run_tags[0].rsplit("-", 1)[1] == run_tags[1].rsplit("-", 1)[1]
    assert old_formula_suffixes[0] != old_formula_suffixes[1]

    null_key_payload = captured_metadata[1]["config"]
    assert "feature_clusters" not in null_key_payload["conditioning"]


# ---------------------------------------------------------------------------
# transport_retry_budget_seconds must not perturb run identity (PR #1098)
# ---------------------------------------------------------------------------


def test_default_transport_retry_budget_is_absent_from_every_client_section() -> None:
    """No LLM-client section may carry the budget at its default.

    A missed section silently retags every pre-existing run and orphans its
    artifacts under a new output root. Sweeping the payload (rather than
    naming ``judge``/``synth``) is what makes this catch a client section
    added later -- exactly how ``concept_examples_qc_judge`` was missed.
    """
    payload = axbench_base_config_to_dict(_fixed_cfg(None))
    offenders = [
        section
        for section, client in payload.items()
        if isinstance(client, dict) and "transport_retry_budget_seconds" in client
    ]
    assert offenders == []


def test_client_sections_exist_so_the_strip_assertion_is_not_vacuous() -> None:
    """Guard the test above: it passes trivially if no client sections exist."""
    payload = dataclasses.asdict(_fixed_cfg(None))
    carriers = {
        section
        for section, client in payload.items()
        if isinstance(client, dict) and "transport_retry_budget_seconds" in client
    }
    assert carriers == {"judge", "synth", "concept_examples_qc_judge"}


@pytest.mark.parametrize(
    ("section", "tuned"),
    [("synth", 30.0), ("judge", 45.0), ("concept_examples_qc_judge", 45.0)],
)
def test_non_default_transport_retry_budget_retags(section: str, tuned: float) -> None:
    """A deliberately tuned budget is identity-bearing and must change the tag.

    Under ``qc_fallback_on_exhaust=warn`` a shorter budget drops rows a longer
    one would have kept, so the tuned value can move outputs.
    """
    baseline = _fixed_cfg(None)
    tuned_cfg = dataclasses.replace(
        baseline,
        **{
            section: dataclasses.replace(
                getattr(baseline, section), transport_retry_budget_seconds=tuned
            )
        },
    )
    baseline_payload = axbench_base_config_to_dict(baseline)
    tuned_payload = axbench_base_config_to_dict(tuned_cfg)
    assert tuned_payload != baseline_payload
    assert tuned_payload[section]["transport_retry_budget_seconds"] == tuned
    assert _run_tag_suffix(tuned_cfg) != _run_tag_suffix(baseline)
