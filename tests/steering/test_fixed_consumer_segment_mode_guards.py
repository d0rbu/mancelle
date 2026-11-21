"""Regression coverage for fixed-only inference/eval segment-mode guards."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

from scripts.train import training_progression
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
)
from steering.eval import (
    context_cache_build,
    multiseg_generator,
    segment_mode_guard,
)
from steering.eval.axbench_base import seed_pools
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb_adv import clustered_context_cache
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.eval.segment_mode_guard import (
    require_fixed_checkpoint_segment_mode,
    require_fixed_feature_dir_segment_mode,
)
from steering.feature_io import TensorKey
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
)
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"sft_segment_mode": SegmentMode.fixed_tokens.value},
        {
            "selfcond": {"data_source": "sft"},
            "selfcond_data": {"sft_segment_mode": SegmentMode.fixed_tokens.value},
        },
    ],
)
def test_fixed_checkpoint_guard_preserves_fixed_and_legacy_metadata(meta) -> None:
    assert (
        require_fixed_checkpoint_segment_mode(
            meta,
            surface="unit consumer",
            checkpoint_dir="/checkpoints/fixed",
        )
        is None
    )


@pytest.mark.parametrize(
    "meta",
    [
        {"sft_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value},
        {
            "selfcond": {"data_source": "sft"},
            "selfcond_data": {
                "sft_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value,
            },
        },
    ],
)
def test_fixed_checkpoint_guard_rejects_every_turn_metadata_carrier(meta) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"unit consumer.*checkpoint /checkpoints/turn.*"
            r"turn_delimited_fixed_tokens.*only 'fixed_tokens'"
        ),
    ):
        require_fixed_checkpoint_segment_mode(
            meta,
            surface="unit consumer",
            checkpoint_dir="/checkpoints/turn",
        )


def _canonical_feature_dir(tmp_path: Path, mode: SegmentMode) -> Path:
    return build_feature_dir(
        root=tmp_path,
        segment_size=64,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.STANDARD_SFT,
        segment_mode=mode,
    )


def test_feature_guard_uses_statistics_for_noncanonical_legacy_paths(tmp_path: Path) -> None:
    assert (
        require_fixed_feature_dir_segment_mode(
            tmp_path / "historical-explicit-feature-dir",
            surface="unit consumer",
            statistics_segment_mode=SegmentMode.fixed_tokens,
        )
        is None
    )


def test_feature_guard_rejects_turn_statistics_even_off_canonical_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unit consumer.*turn_delimited_fixed_tokens"):
        require_fixed_feature_dir_segment_mode(
            tmp_path / "copied-artifact",
            surface="unit consumer",
            statistics_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )


def test_feature_guard_rejects_malformed_path_with_turn_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unit consumer.*turn_delimited_fixed_tokens"):
        require_fixed_feature_dir_segment_mode(
            tmp_path / "noncanonical" / "turn_delimited",
            surface="unit consumer",
        )


def test_feature_guard_does_not_swallow_non_parse_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        segment_mode_guard,
        "parse_feature_dir",
        lambda _path: (_ for _ in ()).throw(RuntimeError("parser bug")),
    )
    with pytest.raises(RuntimeError, match="parser bug"):
        require_fixed_feature_dir_segment_mode(
            tmp_path / "features",
            surface="unit consumer",
        )


def test_feature_guard_rejects_arbitrary_symlink_to_turn_directory(
    tmp_path: Path,
) -> None:
    turn_dir = _canonical_feature_dir(
        tmp_path,
        SegmentMode.turn_delimited_fixed_tokens,
    )
    turn_dir.mkdir(parents=True)
    alias = tmp_path / "copied-features"
    alias.symlink_to(turn_dir, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match=r"unit consumer.*turn_delimited_fixed_tokens",
    ):
        require_fixed_feature_dir_segment_mode(alias, surface="unit consumer")


def test_feature_guard_rejects_literal_fixed_path_resolving_to_turn(
    tmp_path: Path,
) -> None:
    turn_dir = _canonical_feature_dir(
        tmp_path / "target",
        SegmentMode.turn_delimited_fixed_tokens,
    )
    turn_dir.mkdir(parents=True)
    literal_fixed = _canonical_feature_dir(
        tmp_path / "alias",
        SegmentMode.fixed_tokens,
    )
    literal_fixed.parent.mkdir(parents=True)
    literal_fixed.symlink_to(turn_dir, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match=r"feature artifact segment-mode mismatch.*feature path='fixed_tokens'.*"
        r"resolved feature path='turn_delimited_fixed_tokens'",
    ):
        require_fixed_feature_dir_segment_mode(
            literal_fixed,
            surface="unit consumer",
        )


def test_feature_guard_rejects_path_statistics_disagreement(tmp_path: Path) -> None:
    turn_dir = _canonical_feature_dir(
        tmp_path,
        SegmentMode.turn_delimited_fixed_tokens,
    )
    with pytest.raises(
        ValueError,
        match=r"feature artifact segment-mode mismatch for .*turn_delimited: feature path.*turn.*statistics.*fixed",
    ):
        require_fixed_feature_dir_segment_mode(
            turn_dir,
            surface="unit consumer",
            statistics_segment_mode=SegmentMode.fixed_tokens,
        )


def test_feature_guard_rejects_unrecognised_statistics_witness_type(
    tmp_path: Path,
) -> None:
    # kills: dropping @beartype, which would let a non-SegmentMode witness
    # (a raw metadata mapping, or an unknown mode string) be resolved as the
    # legacy fixed_tokens default instead of failing loud.
    with pytest.raises(BeartypeCallHintParamViolation, match=r"statistics_segment_mode.*sliding"):
        require_fixed_feature_dir_segment_mode(
            tmp_path / "features",
            surface="unit consumer",
            statistics_segment_mode=cast(Any, "sliding"),
        )


def test_feature_guard_turn_suffix_backstop_tracks_the_canonical_leaf(
    tmp_path: Path,
) -> None:
    # kills: re-spelling the turn leaf in the guard independently of
    # build_feature_dir, which would silently disable the backstop.
    leaf = _canonical_feature_dir(tmp_path, SegmentMode.turn_delimited_fixed_tokens).name
    with pytest.raises(ValueError, match=r"unit consumer.*turn_delimited_fixed_tokens"):
        require_fixed_feature_dir_segment_mode(
            tmp_path / "relocated-artifact" / leaf,
            surface="unit consumer",
        )


def test_seed_pool_entry_rejects_turn_path_before_tokenizer_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = _canonical_feature_dir(
        tmp_path,
        SegmentMode.turn_delimited_fixed_tokens,
    )
    monkeypatch.setattr(
        seed_pools,
        "get_tokenizer",
        lambda: pytest.fail("turn guard reached tokenizer loading"),
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"segment-mode preflight failed collectively.*turn_delimited_fixed_tokens",
    ):
        seed_pools.prepare_seed_pools(
            cast(
                AxBenchBaseConfig,
                SimpleNamespace(run_tag="guard-test", feature_dir=str(feature_dir)),
            ),
            base_model=object(),
            base_sae=object(),
        )


def test_seed_pool_ddp_guard_uses_collective_failure_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = _canonical_feature_dir(
        tmp_path,
        SegmentMode.turn_delimited_fixed_tokens,
    )
    observed: list[tuple[str, int, int]] = []

    def record_and_run(label, operation, *, rank: int, world_size: int):
        observed.append((label, rank, world_size))
        return real_collective_call(label, operation, rank=rank, world_size=world_size)

    real_collective_call = seed_pools._collective_call
    monkeypatch.setattr(seed_pools.dist, "is_available", lambda: True)
    monkeypatch.setattr(seed_pools.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(seed_pools, "_collective_call", record_and_run)

    # Only the rank-exchange primitive is stubbed: the real _collective_call
    # still decides how a rank-local failure is surfaced.
    def echo_gather(gathered, record, **_kwargs):
        for index in range(len(gathered)):
            gathered[index] = record

    monkeypatch.setattr(seed_pools.dist, "all_gather_object", echo_gather)
    monkeypatch.setattr(
        seed_pools,
        "get_tokenizer",
        lambda: pytest.fail("turn guard reached tokenizer loading"),
    )

    with pytest.raises(
        AxBenchBaseError,
        match=r"segment-mode preflight failed collectively.*turn_delimited_fixed_tokens",
    ) as excinfo:
        seed_pools.prepare_seed_pools(
            cast(
                AxBenchBaseConfig,
                SimpleNamespace(run_tag="guard-test", feature_dir=str(feature_dir)),
            ),
            base_model=object(),
            base_sae=object(),
            rank=1,
            world_size=2,
        )

    # kills: dropping the AxBenchBaseError wrapping under DDP, which the
    # previous pass-through stub could not observe.
    assert not isinstance(excinfo.value, ValueError)
    assert observed == [("segment-mode preflight", 1, 2)]


def test_clustered_bpb_cache_preflight_rejects_turn_contract() -> None:
    contract = SimpleNamespace(
        feature_directory=SimpleNamespace(
            segment_mode=SegmentMode.turn_delimited_fixed_tokens.value,
        )
    )
    with pytest.raises(ValueError, match=r"clustered context-cache.*turn_delimited"):
        clustered_context_cache._validate_runtime_configuration(
            cfg=SimpleNamespace(),
            contract=cast(ClusterFeatureContract, contract),
        )


def test_progression_loader_rejects_turn_checkpoint_before_returning_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    monkeypatch.setattr(
        training_progression,
        "_should_load_encoder_decoder",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        training_progression,
        "build_encdec_model",
        lambda *_args, **_kwargs: (
            model,
            object(),
            {"sft_segment_mode": "turn_delimited_fixed_tokens"},
        ),
    )
    monkeypatch.setattr(
        training_progression,
        "assert_conditioned_eval_model",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(
        ValueError,
        match=r"training_progression.*turn_delimited_fixed_tokens",
    ):
        training_progression.load_progression_model(
            "/checkpoints/turn",
            1,
            t.device("cpu"),
            OmegaConf.create({}),
        )


def _turn_self_cond_provider() -> SelfConditionedFeatureProvider:
    provider = cast(Any, object.__new__(SelfConditionedFeatureProvider))
    provider.cfg = SimpleNamespace(
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    return provider


def test_multiseg_public_selfcond_rejects_turn_before_zero_budget_shortcut() -> None:
    stream = multiseg_generator.generate_completion_stream_self_conditioned(
        model=object(),
        prefix_tokens=[1],
        provider=_turn_self_cond_provider(),
        encoder_inputs=object.__new__(multiseg_generator.MultisegInputs),
        max_new_tokens=0,
        device=t.device("cpu"),
    )

    with pytest.raises(
        ValueError,
        match=r"multisegment self-conditioned.*turn_delimited_fixed_tokens",
    ):
        next(stream)


# Harmfulness supports both fixed and turn checkpoints through authenticated
# context contracts. Its mode-specific admission is covered by
# tests/steering/test_harmfulness_safety_runtime.py.


def test_raw_context_converter_validates_every_shard_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def load_features(path: Path) -> dict:
        loaded.append(path.name)
        if path.name == "a.safetensors":
            return {
                "segment_mode": SegmentMode.fixed_tokens.value,
                TensorKey.DOC_TOKEN_COUNTS: t.empty(0, dtype=t.int64),
            }
        return {"segment_mode": SegmentMode.turn_delimited_fixed_tokens.value}

    monkeypatch.setattr(context_cache_build, "load_features", load_features)
    monkeypatch.setattr(
        context_cache_build,
        "_validate_context_conversion_contract",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        context_cache_build,
        "_entry_from_shard_doc",
        lambda **_kwargs: pytest.fail("turn shard reached context selection"),
    )

    with pytest.raises(
        ValueError,
        match=r"eval context-cache conversion.*b\.safetensors.*turn_delimited_fixed_tokens",
    ):
        context_cache_build.build_cache_entries_from_extracted_feature_shards(
            context_records=[],
            feature_shard_paths=[
                tmp_path / "b.safetensors",
                tmp_path / "a.safetensors",
            ],
            pipeline=object.__new__(context_cache_build.FeaturePipelineConfig),
            n_sae_features=PositiveInt.parse(1),
            num_buckets=PositiveInt.parse(1),
            tokenizer=object(),
            source_bos_char_len=NonNegativeInt.parse(0),
        )

    assert loaded == ["a.safetensors", "b.safetensors"]


_GUARD_NAMES = (
    "require_fixed_checkpoint_segment_mode",
    "require_fixed_feature_dir_segment_mode",
    "require_fixed_segment_mode",
)


def _files_calling_any_guard() -> set[str]:
    """Every production file that calls a shared guard, discovered from source."""

    discovered: set[str] = set()
    guard_module = _ROOT / "steering" / "eval" / "segment_mode_guard.py"
    for package in ("scripts", "steering"):
        for path in sorted((_ROOT / package).rglob("*.py")):
            if path == guard_module:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name in _GUARD_NAMES:
                    discovered.add(path.relative_to(_ROOT).as_posix())
                    break
    return discovered


# Every production file that calls a shared guard. Behavioural turn-rejection
# tests pin what each consumer hands its guard; this ratchet pins the *set* of
# guarded files, which no behavioural test can: a NEW fixed-only consumer that
# never calls a guard at all is invisible to every other test in the suite.
# Adding a file here without a behavioural turn-rejection test (in this module,
# in tests/steering/test_fixed_consumer_turn_rejection.py, or beside the
# consumer) leaves that consumer unpinned.
# Standalone chat_eval uses mode-aware admission for schema-5 turn target-real
# caches, not a shared fixed-only guard. Its rejection paths are covered in
# test_fixed_consumer_turn_rejection.py and test_chat_eval_turn_cache_integration.py.
# Harmfulness likewise resolves both modes through its self-conditioning adapter.
_GUARDED_FILES = frozenset(
    {
        "scripts/utils/verify_bf16_threshold.py",
        "scripts/chat/steer_chat_cli.py",
        "scripts/chat_tui/setup.py",
        "scripts/completion_tui/__main__.py",
        "scripts/eval/axbench_bpb_eval.py",
        "scripts/eval/base_eval.py",
        "scripts/eval/build_eval_context_cache.py",
        "scripts/eval/completion_steering_eval.py",
        "scripts/eval/dump_chat_responses.py",
        "scripts/eval/shuffled_conditioning_eval.py",
        "scripts/eval/steering_eval.py",
        "scripts/train/training_progression.py",
        "steering/eval/axbench_base/runtime_state.py",
        "steering/eval/axbench_base/seed_pools.py",
        "steering/eval/axbench_bpb/runtime_state.py",
        "steering/eval/axbench_bpb_adv/clustered_context_cache.py",
        "steering/eval/axbench_bpb_adv/general_val.py",
        "steering/eval/bpb_pipeline.py",
        "steering/eval/context_cache_build.py",
        "steering/eval/dclm_eval.py",
        "steering/eval/multiseg_generator.py",
    }
)


def test_guard_census_enumerates_every_guarded_file() -> None:
    # kills: adding a fixed-only consumer that calls a guard (or dropping the
    # last guard call from one) without enrolling it in this suite, which is
    # what let bpb_pipeline.py and general_val.py go unguarded.
    assert _files_calling_any_guard() == _GUARDED_FILES
