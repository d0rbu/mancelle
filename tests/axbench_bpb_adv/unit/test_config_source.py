from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb.config import ContextCacheConfig
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GepaConfig,
    Stage,
    validate_config,
)
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.source_validation import validate_stage_0_3_source_run
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from tests.axbench_bpb_adv.source_run_builders import (
    synth_row,
    write_stage_0_3_source_run,
)
from tests.runtime_attribute_helpers import (
    call_runtime_attribute,
    set_runtime_attribute,
)

_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "eval"
    / "axbench_bpb_adv"
)


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 256


def _write_source_run(root: Path) -> None:
    # concept_rich/concept_avoiding completions of two prefixes, with a distinct
    # output_text per class, plus a single-file marker baselines/ directory.
    def _de_fg_pair(prefix: str) -> list[dict]:
        return [
            synth_row(prefix, content_class="concept_rich", output_text="de"),
            synth_row(prefix, content_class="concept_avoiding", output_text="fg"),
        ]

    write_stage_0_3_source_run(
        root,
        positive_rows=_de_fg_pair("abc"),
        negative_rows=_de_fg_pair("xyz"),
    )


def test_yaml_composes_with_defaults() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )

    assert "baseline_x_mode" not in cfg
    assert "neutral_references" not in cfg
    assert "include_unsteered_baseline_series" not in cfg
    assert "context_signal_modes" not in cfg
    assert cfg.adversarial_methods[0].method == "faster_gcg"
    assert cfg.adversarial_methods[0].optimized_token_counts == (
        0,
        1,
        2,
        4,
        8,
        16,
        32,
    )
    assert "gepa" not in [method.method for method in cfg.adversarial_methods]
    assert "additive_direction" in [method.method for method in cfg.adversarial_methods]
    assert list(cfg.target_models) == ["decoder_only", "encdec"]
    assert cfg.diffmean_and_probe_layer == 6
    assert "decoder_only_target" in cfg
    assert "encdec_target" in cfg
    assert "lora_adapter" in cfg

    # base_methods is now an explicit whitelist (not null). The encoder-decoder
    # no-steering control is excluded by default -- it stays in the registry and
    # can be re-added here to compute it. The whitelisted methods must be valid.
    base_methods = list(cfg.base_methods)
    assert base_methods, "base_methods whitelist must be non-empty"
    assert "control_encdec" not in base_methods
    assert "neg_control_encdec" not in base_methods
    known = {method.name for method in ALL_METHODS}
    assert set(base_methods) <= known, sorted(set(base_methods) - known)


def test_cfg_to_dataclass_rejects_unknown_keys() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="Unknown"):
        cfg_to_dataclass(
            OmegaConf.create(
                {
                    "source_run_dir": "/tmp/source",
                    "output_root": "/tmp/out",
                    "feature_dir": "/tmp/features",
                    "feature_statistics_path": "/tmp/stats",
                    "not_a_key": True,
                }
            )
        )


def test_cfg_to_dataclass_rejects_context_signal_modes_key() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="Unknown axbench-bpb-adv config keys"):
        cfg_to_dataclass(
            OmegaConf.create(
                {
                    "source_run_dir": "/tmp/source",
                    "output_root": "/tmp/out",
                    "feature_dir": "/tmp/features",
                    "feature_statistics_path": "/tmp/stats",
                    "context_signal_modes": ["fixed_context"],
                }
            )
        )


def test_config_has_no_context_signal_modes_field() -> None:
    names = {f.name for f in dataclasses.fields(AxBenchBpbAdvConfig)}
    assert "context_signal_modes" not in names


def test_cfg_to_dataclass_requires_tagged_adversarial_method_objects() -> None:
    base = {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/stats",
    }
    with pytest.raises(AxBenchBpbAdvError, match="tagged objects"):
        cfg_to_dataclass(OmegaConf.create({**base, "adversarial_methods": ["gepa"]}))

    cfg = cfg_to_dataclass(
        OmegaConf.create(
            {
                **base,
                "adversarial_methods": [
                    {
                        "method": "faster_gcg",
                        "optimized_token_counts": [0, 1],
                    }
                ],
            }
        )
    )

    assert cfg.adversarial_methods == [
        FasterGcgConfig(
            optimized_token_counts=(0, 1),
        )
    ]


def test_cfg_to_dataclass_programmatic_defaults_match_yaml_method_defaults() -> None:
    cfg = cfg_to_dataclass(
        OmegaConf.create(
            {
                "source_run_dir": "/tmp/source",
                "output_root": "/tmp/out",
                "feature_dir": "/tmp/features",
                "feature_statistics_path": "/tmp/stats",
            }
        )
    )

    assert cfg.adversarial_methods == [FasterGcgConfig(), AdditiveDirectionConfig()]


def test_cfg_to_dataclass_rejects_old_target_config_aliases() -> None:
    base = {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/stats",
    }
    for old_key in ("decoder_only", "encdec", "lora"):
        with pytest.raises(AxBenchBpbAdvError, match="Unknown"):
            cfg_to_dataclass(OmegaConf.create({**base, old_key: {"source": "old"}}))

    cfg = cfg_to_dataclass(
        OmegaConf.create(
            {
                **base,
                "decoder_only_target": {"source": "local"},
                "encdec_target": {"source": "artifact"},
                "lora_adapter": {"rank": 4},
            }
        )
    )

    assert cfg.decoder_only_target == {"source": "local"}
    assert cfg.encdec_target == {"source": "artifact"}
    assert cfg.lora_adapter == {"rank": 4}

    with pytest.raises(AxBenchBpbAdvError, match="Unknown"):
        cfg_to_dataclass(
            OmegaConf.create(
                {
                    **base,
                    "decoder_only": {"source": "old"},
                    "decoder_only_target": {"source": "new"},
                }
            )
        )


def test_method_configs_are_deeply_frozen() -> None:
    cfg = FasterGcgConfig()

    with pytest.raises(FrozenInstanceError):
        set_runtime_attribute(cfg, "max_steps", 1)
    with pytest.raises(AttributeError):
        call_runtime_attribute(cfg.optimized_token_counts, "append", 99)


def test_cfg_to_dataclass_wraps_nested_config_errors() -> None:
    base = {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/stats",
    }
    with pytest.raises(AxBenchBpbAdvError, match="invalid context_cache"):
        cfg_to_dataclass(
            OmegaConf.create({**base, "context_cache": {"not_a_context_key": 1}})
        )
    # neutral_references is no longer a valid config block; it must be rejected
    # as an unknown key rather than parsed into a nested config.
    with pytest.raises(AxBenchBpbAdvError, match="Unknown"):
        cfg_to_dataclass(
            OmegaConf.create(
                {**base, "neutral_references": {"continuation_token_budget": 256}}
            )
        )


def test_validate_config_gepa_requires_explicit_model_config() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[GepaConfig()],
    )

    with pytest.raises(AxBenchBpbAdvError, match="reflection_model"):
        validate_config(cfg)


def test_validate_config_accepts_additive_modes() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )

    # additive on an encdec-only run is now valid (additive composes a
    # counter-direction on top of the encdec `ours` baselines).
    cfg.adversarial_methods = [AdditiveDirectionConfig(families=("ours",))]
    cfg.target_models = ["encdec"]
    validate_config(cfg)

    # The boundary magnitude 0.0 (the pure-baseline control documented on
    # intervention_build.additive_counter_coefficient) plus a positive magnitude
    # must remain valid: the non-negativity guard rejects only values strictly < 0.
    cfg.adversarial_methods = [AdditiveDirectionConfig(magnitudes=(0.0, 1.0))]
    cfg.target_models = ["decoder_only"]
    validate_config(cfg)


def test_prepare_neutral_references_is_not_a_valid_stage() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="Unknown axbench-bpb-adv stage"):
        Stage.coerce("prepare_neutral_references")
    assert "prepare_neutral_references" not in {stage.value for stage in Stage}


def test_resolved_all_stage_chain_excludes_neutral_references() -> None:
    stages = bpb_adv_cli._resolve_stages(Stage.ALL)
    stage_values = [stage.value for stage in stages]
    assert "prepare_neutral_references" not in stage_values
    assert stage_values == [
        "validate_source",
        "prepare_context_cache",
        "prepare_attack_rows",
        "build_candidates",
        "score",
        "score_general_val",
        "aggregate",
        "plot",
    ]


def test_source_validation_requires_stage_0_3_only(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _write_source_run(root)
    (root / "generations").mkdir()
    (root / "judge_scores").mkdir()

    validation = validate_stage_0_3_source_run(root)

    assert validation.n_concepts == 1
    assert validation.n_positive_synth_rows == 2
    assert validation.n_negative_synth_rows == 2
    assert validation.n_positive_prefixes == 1
    assert validation.n_negative_prefixes == 1
    assert "generations" not in validation.contract.artifact_byte_sizes
    assert "judge_scores" not in validation.contract.artifact_byte_sizes
    assert "generations" not in validation.contract.artifact_entry_counts
    assert "judge_scores" not in validation.contract.artifact_entry_counts


def test_source_validation_requires_both_classes_per_prefix(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _write_source_run(root)
    # Drop the concept_avoiding side of the positive prefix -> one-sided prefix.
    schema = pa.schema(
        [
            ("concept_id", pa.int32()),
            ("content_class", pa.string()),
            ("prefix_text", pa.string()),
            ("output_text", pa.string()),
            ("qc_accepted", pa.bool_()),
            ("prefix_source_shard", pa.int32()),
            ("prefix_source_doc", pa.int32()),
        ]
    )
    one_sided = [
        {
            "concept_id": 1,
            "content_class": "concept_rich",
            "prefix_text": "abc",
            "output_text": "de",
            "qc_accepted": True,
            "prefix_source_shard": 0,
            "prefix_source_doc": 0,
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(
            one_sided,
            schema=schema_with_stage1_continuation_contract(
                schema_with_positive_qc_contract(
                    schema,
                    accept_threshold=0.3,
                ),
                role="stage1b_positive",
            ),
        ),
        root / "synth_training_positive.parquet",
    )

    with pytest.raises(AxBenchBpbAdvError, match="concept_avoiding"):
        validate_stage_0_3_source_run(root)


def test_source_validation_requires_synth_columns_and_hashes_baselines(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _write_source_run(root)
    first = validate_stage_0_3_source_run(root)
    (root / "baselines" / "prompting" / "positive_descriptions.json").write_text(
        '{"changed": true}',
        encoding="utf-8",
    )
    second = validate_stage_0_3_source_run(root)

    assert first.contract.artifact_hashes["baselines/"] != second.contract.artifact_hashes[
        "baselines/"
    ]

    broken = tmp_path / "broken"
    _write_source_run(broken)
    pq.write_table(
        pa.Table.from_pylist(
            [{"concept_id": 1, "prefix_text": "x", "output_text": "y"}],
            schema=schema_with_stage1_continuation_contract(
                pa.schema(
                    [
                        ("concept_id", pa.int64()),
                        ("prefix_text", pa.string()),
                        ("output_text", pa.string()),
                    ]
                ),
                role="stage1c_negative",
            ),
        ),
        broken / "synth_training_negative.parquet",
    )
    with pytest.raises(AxBenchBpbAdvError, match="prefix_source_doc"):
        validate_stage_0_3_source_run(broken)


def test_prepare_attack_rows_filters_concepts_not_in_source() -> None:
    rows = [
        {"concept_id": 1, "prefix_text": "kept", "output_text": "x"},
        {"concept_id": -1, "prefix_text": "global negative", "output_text": "x"},
    ]

    filtered = bpb_adv_cli._filter_synth_rows_to_source_concepts(
        rows,
        concept_ids={1},
        mode=SteeringMode.POSITIVE,
    )

    assert [row["concept_id"] for row in filtered] == [1]


def test_prepare_attack_rows_detects_real_target_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRuntimeState:
        offloaded = False

        def __init__(self, cfg, ddp_rank: int, ddp_world_size: int) -> None:
            self.cfg = cfg
            self.ddp_rank = ddp_rank
            self.ddp_world_size = ddp_world_size

        def decoder_only(self):
            return SimpleNamespace(model="decoder-model", tokenizer=_TinyTokenizer())

        def encdec(self):
            return SimpleNamespace(model="encdec-model", tokenizer=_TinyTokenizer())

        def offload_models(self) -> None:
            FakeRuntimeState.offloaded = True

    def fake_detect(model, _cfg):
        return TargetModelMetadata(
            num_buckets=7 if model == "encdec-model" else None,
            n_sae_features=11,
            negative_selection={"1": 2} if model == "encdec-model" else None,
            feature_selection=None,
            feature_mask=None,
            per_feature_stats=None,
            device="cpu",
            sae_layer=3,
            segment_size=128,
        )

    monkeypatch.setattr(bpb_adv_cli, "RuntimeState", FakeRuntimeState)
    monkeypatch.setattr(bpb_adv_cli, "detect_target_model_metadata", fake_detect)
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        target_models=["decoder_only", "encdec"],
        device="cpu",
    )

    metas, _tokenizer = bpb_adv_cli._target_runtime_info_from_runtime(cfg)

    assert metas["encdec"].num_buckets == 7
    assert metas["encdec"].negative_selection == {"1": 2}
    assert FakeRuntimeState.offloaded is True


def test_run_tag_is_broadcast_from_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[list[str | None]] = []

    def fake_broadcast_object_list(payload: list[str | None], src: int) -> None:
        assert src == 0
        payloads.append(list(payload))
        payload[0] = "rank0-tag"

    monkeypatch.setattr(bpb_adv_cli.dist, "is_available", lambda: True)
    monkeypatch.setattr(bpb_adv_cli.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        bpb_adv_cli.dist,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        run_tag=None,
    )

    bpb_adv_cli._synchronize_run_tag(cfg, ddp_rank=1, ddp_world_size=2)

    assert payloads == [[None]]
    assert cfg.run_tag == "rank0-tag"


def test_context_cache_stage_participates_on_nonzero_ddp_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_context_stage(
        _cfg: AxBenchBpbAdvConfig,
        *,
        ddp_rank: int,
        ddp_world_size: int,
    ) -> None:
        calls.append((ddp_rank, ddp_world_size))

    monkeypatch.setattr(bpb_adv_cli, "_stage_prepare_context_cache", fake_context_stage)
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )

    bpb_adv_cli._run_stage(
        Stage.PREPARE_CONTEXT_CACHE,
        cfg,
        ddp_rank=1,
        ddp_world_size=4,
    )

    assert calls == [(1, 4)]


def _valid_config(**overrides: object) -> AxBenchBpbAdvConfig:
    base = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _mutate_required_string(field_name: str) -> AxBenchBpbAdvConfig:
    return _valid_config(**{field_name: ""})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        # --- empty required-string guards ---
        pytest.param(
            lambda: _mutate_required_string("source_run_dir"),
            "source_run_dir is required",
            id="empty_source_run_dir",
        ),
        pytest.param(
            lambda: _mutate_required_string("output_root"),
            "output_root is required",
            id="empty_output_root",
        ),
        pytest.param(
            lambda: _mutate_required_string("feature_dir"),
            "feature_dir is required",
            id="empty_feature_dir",
        ),
        pytest.param(
            lambda: _mutate_required_string("feature_statistics_path"),
            "feature_statistics_path is required",
            id="empty_feature_statistics_path",
        ),
        # --- additive wrong dataclass type guard ---
        pytest.param(
            lambda: _valid_config(
                adversarial_methods=[
                    SimpleNamespace(method="additive_direction")
                ],
                target_models=["decoder_only"],
            ),
            "additive_direction must use AdditiveDirectionConfig",
            id="additive_wrong_dataclass_type",
        ),
    ],
)
def test_validate_config_rejects_each_guard(mutate, match: str) -> None:
    cfg = mutate()
    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_config(cfg)


def test_source_validation_reports_positive_and_negative_prefix_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _write_source_run(root)
    # Add a SECOND distinct prefix (with both class sides) to each synth split
    # so the positive/negative prefix counts are non-trivial and distinguishable
    # from the row counts.
    schema = pa.schema(
        [
            ("concept_id", pa.int32()),
            ("content_class", pa.string()),
            ("prefix_text", pa.string()),
            ("output_text", pa.string()),
            ("qc_accepted", pa.bool_()),
            ("prefix_source_shard", pa.int32()),
            ("prefix_source_doc", pa.int32()),
        ]
    )

    def _both_class_rows(prefix: str) -> list[dict]:
        return [
            {
                "concept_id": 1,
                "content_class": cls,
                "prefix_text": prefix,
                "output_text": "de",
                "qc_accepted": True,
                "prefix_source_shard": 0,
                "prefix_source_doc": 0,
            }
            for cls in ("concept_rich", "concept_avoiding")
        ]

    pq.write_table(
        pa.Table.from_pylist(
            _both_class_rows("abc") + _both_class_rows("abc2"),
            schema=schema_with_stage1_continuation_contract(
                schema_with_positive_qc_contract(
                    schema,
                    accept_threshold=0.3,
                ),
                role="stage1b_positive",
            ),
        ),
        root / "synth_training_positive.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _both_class_rows("xyz"),
            schema=schema_with_stage1_continuation_contract(
                schema,
                role="stage1c_negative",
            ),
        ),
        root / "synth_training_negative.parquet",
    )

    validation = validate_stage_0_3_source_run(root)

    assert validation.n_positive_synth_rows == 4
    assert validation.n_negative_synth_rows == 2
    assert validation.n_positive_prefixes == 2
    assert validation.n_negative_prefixes == 1


def test_non_context_stages_are_rank_zero_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_on_nonzero_rank(
        _cfg: AxBenchBpbAdvConfig,
        _cluster_contract: ClusterFeatureContract | None,
    ) -> None:
        raise AssertionError("rank 1 should not run rank-0-only stages")

    monkeypatch.setitem(
        bpb_adv_cli._STAGE_HANDLERS,
        Stage.PREPARE_ATTACK_ROWS,
        fail_on_nonzero_rank,
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )

    bpb_adv_cli._run_stage(
        Stage.PREPARE_ATTACK_ROWS,
        cfg,
        ddp_rank=1,
        ddp_world_size=4,
    )


# ---------------------------------------------------------------------------
# config_loader.py coverage gaps (Feature B). These exercise the parser
# (``cfg_to_dataclass`` / its ``_parse_*`` helpers) directly from a wrapped
# ``OmegaConf`` dict, mirroring the existing pattern above.
# ---------------------------------------------------------------------------


def _loader_base() -> dict[str, str]:
    return {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/features",
        "feature_statistics_path": "/tmp/stats",
    }


def test_cfg_to_dataclass_parses_gepa_tagged_dict() -> None:
    # GEPA dict-dispatch: a tagged ``{"method": "gepa", ...}`` block must route
    # through the GepaConfig arm of the config_cls dispatch table. A swapped
    # dispatch arm (e.g. mapping GEPA -> FasterGcgConfig) would either raise on the
    # gepa-only kwargs or build the wrong dataclass; this pins the GepaConfig
    # result with the passed fields preserved.
    cfg = cfg_to_dataclass(
        OmegaConf.create(
            {
                **_loader_base(),
                "adversarial_methods": [
                    {
                        "method": "gepa",
                        "reflection_model": "r",
                        "k_editable_tokens": 8,
                    }
                ],
            }
        )
    )

    assert cfg.adversarial_methods == [
        GepaConfig(reflection_model="r", k_editable_tokens=8)
    ]
    # And it really is the GEPA dataclass, not a coincidentally-equal sibling.
    assert isinstance(cfg.adversarial_methods[0], GepaConfig)


def test_cfg_to_dataclass_rejects_unknown_adversarial_method() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="Unknown adversarial method"):
        cfg_to_dataclass(
            OmegaConf.create(
                {
                    **_loader_base(),
                    "adversarial_methods": [{"method": "not_a_real_method"}],
                }
            )
        )


def test_cfg_to_dataclass_wraps_general_val_bpb_type_error() -> None:
    # Asymmetric twin of the covered context_cache wrap: an unknown nested key
    # makes ``GeneralValBpbConfig(**mapping)`` raise TypeError, which the loader
    # must re-wrap as the project error type naming the block.
    with pytest.raises(AxBenchBpbAdvError, match="invalid general_val_bpb"):
        cfg_to_dataclass(
            OmegaConf.create(
                {**_loader_base(), "general_val_bpb": {"not_a_gv_key": 1}}
            )
        )


def test_cfg_to_dataclass_wraps_per_method_type_error() -> None:
    # A bogus kwarg makes ``config_cls(**payload)`` raise TypeError inside
    # ``_parse_adversarial_method_config``; it must be re-wrapped naming the method.
    with pytest.raises(
        AxBenchBpbAdvError,
        match="invalid faster_gcg adversarial method config",
    ):
        cfg_to_dataclass(
            OmegaConf.create(
                {
                    **_loader_base(),
                    "adversarial_methods": [
                        {"method": "faster_gcg", "bogus_kwarg": 1}
                    ],
                }
            )
        )


def test_cfg_to_dataclass_rejects_non_list_adversarial_methods() -> None:
    # A mapping (not a list) hits the ``_parse_adversarial_methods`` list guard.
    with pytest.raises(AxBenchBpbAdvError, match="adversarial_methods must be a list"):
        cfg_to_dataclass(
            OmegaConf.create(
                {**_loader_base(), "adversarial_methods": {"method": "faster_gcg"}}
            )
        )


def test_cfg_to_dataclass_rejects_adversarial_method_without_method_field() -> None:
    # A tagged-dict entry that omits ``method`` hits the missing-'method' guard.
    with pytest.raises(AxBenchBpbAdvError, match="missing 'method'"):
        cfg_to_dataclass(
            OmegaConf.create(
                {
                    **_loader_base(),
                    "adversarial_methods": [{"optimized_token_counts": [0, 1]}],
                }
            )
        )


def test_cfg_to_dataclass_rejects_non_mapping_nested_block() -> None:
    # A list where a nested config OBJECT is expected hits ``_nested_config_mapping``'s
    # "must be a config object" guard (via the context_cache block).
    with pytest.raises(AxBenchBpbAdvError, match="context_cache must be a config object"):
        cfg_to_dataclass(
            OmegaConf.create({**_loader_base(), "context_cache": [1, 2]})
        )


def test_cfg_to_dataclass_coalesces_explicit_null_context_cache() -> None:
    # An explicit ``context_cache: null`` coalesces to ``{}`` (the None branch of
    # ``_nested_config_mapping``), yielding the default ContextCacheConfig.
    cfg = cfg_to_dataclass(
        OmegaConf.create({**_loader_base(), "context_cache": None})
    )

    assert cfg.context_cache == ContextCacheConfig()
