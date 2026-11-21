from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from torch import nn

from nanochat.tokenizer import LogicalTurnSpan
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_online_sae import (
    ONLINE_SAE_LINEAGE_KEY,
    OnlineSAEInitialConditioningProvider,
    OnlineSAELineage,
    build_online_sae_lineage_metadata,
)
from steering.conditional.encoder_decoder.sft_selection import SegmentSelectionContext
from steering.conditional.missing_percentile import MissingPercentileResolver
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.probe_model import ProbeModel
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.sft.structured_rollout import ConversationEnvironment, TargetSegment


class _FakeTokenizer:
    bos_token_id = 1

    def __init__(self, encoded: list[int]) -> None:
        self.encoded = encoded
        self.seen: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        self.seen.append(text)
        return list(self.encoded)

    def decode(self, token_ids: list[int]) -> str:
        del token_ids
        return self.seen[-1]


class _FakeSAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(t.zeros(()), requires_grad=False)
        self.cfg = SimpleNamespace(apply_b_dec_to_input=False)

    def encode(self, activations: t.Tensor) -> t.Tensor:
        return activations


def _lineage() -> OnlineSAELineage:
    return OnlineSAELineage(
        model="fixture",
        release="fixture",
        activation_type="residual",
        layer=2,
        width="4",
        l0=None,
        source_segment_size=2,
        feature_dir="unused",
        statistics_path="unused",
        statistics_sha256="0" * 64,
        n_sae_features=4,
        num_buckets=2,
        feature_selection={},
        negative_selection=None,
        missing_percentile_policy="error",
        selection_seed=42,
        source_segment_mode=SegmentMode.fixed_tokens.value,
    )


def _feature_dir(tmp_path: Path, mode: SegmentMode) -> Path:
    directory = build_feature_dir(
        root=tmp_path,
        segment_size=256,
        layer=22,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_65K,
        model="gemma-3-4b-pt",
        dataset_phase=DatasetPhase.STANDARD_SFT,
        context_independent_features=True,
        ignore_bos_features=True,
        segment_mode=mode,
    )
    directory.mkdir(parents=True)
    (directory / "feature_statistics.safetensors").write_bytes(b"pinned-statistics")
    return directory


def _lineage_meta(
    feature_dir: Path,
    mode: SegmentMode,
    **overrides: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "feature_dir": str(feature_dir),
        "statistics_path": feature_dir / "feature_statistics.safetensors",
        "segment_mode": mode,
        "sae_config": {
            "model": "gemma-3-4b-pt",
            "release": "gemmascope-2",
            "activation_type": "residual",
            "layer": 22,
            "width": "65k",
        },
        "feature_selection": {"mode": "top_k_percentile"},
        "negative_selection": None,
        "missing_percentile_policy": "error",
        "n_sae_features": 65_536,
        "num_buckets": 2,
        "selection_seed": 42,
    }
    kwargs.update(overrides)
    return build_online_sae_lineage_metadata(**kwargs)


def _environment() -> ConversationEnvironment:
    return ConversationEnvironment(
        original_index=9,
        inputs=(0, 10, 11, 12),
        targets=(10, 11, 12, 13),
        reward_mask=(False, False, True, True),
        logical_turns=(
            LogicalTurnSpan("user", 0, 2),
            LogicalTurnSpan("assistant", 2, 4),
        ),
        segments=(
            TargetSegment(0, 0, 2, None),
            TargetSegment(1, 2, 4, None),
        ),
        seg_map=(0, 0, 0, 1),
    )


def _selection_context(_original_index: int) -> SegmentSelectionContext:
    return SegmentSelectionContext(
        threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 2),
        precomputed_thresholds=None,
        sampling_bundle=None,
        negative_bundle=None,
        boundaries=t.zeros((4, 1), dtype=t.float64),
        fp_percentile_table=None,
        cluster_threshold_percentile_table=None,
        feature_mask=None,
        consolidation=None,
        missing_percentile=MissingPercentileResolver.error(),
    )


def test_online_provider_is_lazy_excludes_all_bos_rows_and_reuses_probe() -> None:
    tokenizer = _FakeTokenizer([1, 2, 1, 3])
    model = nn.Linear(1, 1, bias=False)
    probe = ProbeModel(
        model=model,
        sae=_FakeSAE(),
        tokenizer=tokenizer,
        hook_name="hook",
        layer=2,
        context_length=16,
        activation_type=ActivationType.residual,
    )
    loads = 0
    decoded: list[tuple[int, ...]] = []

    def load_probe() -> ProbeModel:
        nonlocal loads
        loads += 1
        return probe

    def rust_decode(token_ids) -> str:
        decoded.append(tuple(token_ids))
        return "first segment"

    def activations(_probe: ProbeModel, tokens: t.Tensor) -> t.Tensor:
        assert tokens.tolist() == [[1, 2, 1, 3]]
        return t.tensor(
            [
                [
                    [99.0, 99.0, 99.0, 99.0],
                    [0.5, 4.0, 0.0, 1.0],
                    [88.0, 88.0, 88.0, 88.0],
                    [3.0, 0.0, 2.0, 0.0],
                ]
            ]
        )

    provider = OnlineSAEInitialConditioningProvider(
        _lineage(),
        decode_rustbpe=rust_decode,
        selection_context_factory=_selection_context,
        device="cpu",
        probe_factory=load_probe,
        activation_provider=activations,
    )
    assert provider.sae_loaded is False
    first = provider.provide(_environment())
    second = provider.provide(_environment())
    assert provider.sae_loaded is True
    assert loads == 1
    assert decoded == [(10, 11), (10, 11)]
    assert tokenizer.seen == ["first segment", "first segment"]
    assert first == second
    assert first.feature_ids == (1, 0)
    assert first.magnitudes == (1, 1)
    assert first.mask == (True, True)
    assert provider.raw_empty_count == 0
    assert provider.selected_empty_count == 0


def test_online_provider_records_empty_without_becoming_none() -> None:
    tokenizer = _FakeTokenizer([])
    probe = ProbeModel(
        model=nn.Linear(1, 1, bias=False),
        sae=_FakeSAE(),
        tokenizer=tokenizer,
        hook_name="hook",
        layer=2,
        context_length=16,
        activation_type=ActivationType.residual,
    )
    provider = OnlineSAEInitialConditioningProvider(
        _lineage(),
        decode_rustbpe=lambda _ids: "",
        selection_context_factory=_selection_context,
        device="cpu",
        probe_factory=lambda: probe,
        activation_provider=lambda _probe, _tokens: t.zeros((1, 1, 4)),
    )
    action = provider.provide(_environment())
    assert action.feature_ids == ()
    assert provider.mode.value == "online_sae"
    assert provider.sae_loaded is True
    assert provider.raw_empty_count == 1
    assert provider.selected_empty_count == 1


def test_lineage_builder_and_parser_pin_feature_path_and_statistics(tmp_path) -> None:
    feature_dir = (
        tmp_path
        / "s256"
        / "l22"
        / "a_res"
        / "p_max_pool"
        / "f65k"
        / "m_gemma-3-4b-pt"
        / "standard_sft"
        / "context_independent"
        / "no_bos"
    )
    feature_dir.mkdir(parents=True)
    stats = feature_dir / "feature_statistics.safetensors"
    stats.write_bytes(b"pinned-statistics")
    meta = build_online_sae_lineage_metadata(
        feature_dir=str(feature_dir),
        statistics_path=stats,
        segment_mode=SegmentMode.fixed_tokens,
        sae_config={
            "model": "gemma-3-4b-pt",
            "release": "gemmascope-2",
            "activation_type": "residual",
            "layer": 22,
            "width": "65k",
        },
        feature_selection={"mode": "top_k_percentile"},
        negative_selection={"mean": 10.0},
        missing_percentile_policy="error",
        n_sae_features=65_536,
        num_buckets=2,
        selection_seed=42,
    )
    lineage_block = meta[ONLINE_SAE_LINEAGE_KEY]
    assert "source_segment_mode" not in lineage_block
    assert lineage_block == {
        "schema_version": 1,
        "model": "gemma-3-4b-pt",
        "release": "gemmascope-2",
        "activation_type": "residual",
        "layer": 22,
        "width": "65k",
        "l0": None,
        "source_segment_size": 256,
        "pooling": "max_pool",
        "context_independent_features": True,
        "ignore_bos_features": True,
        "feature_dir": str(feature_dir),
        "statistics_path": str(stats),
        "statistics_sha256": (
            "9839df3cd946fef127899c59e6a9b92d08c9caab546996b237a2efe793969c0d"
        ),
        "n_sae_features": 65_536,
        "num_buckets": 2,
        "feature_selection": {"mode": "top_k_percentile"},
        "negative_selection": {"mean": 10.0},
        "missing_percentile_policy": "error",
        "selection_seed": 42,
    }
    parsed = OnlineSAELineage.from_checkpoint_meta(meta)
    assert parsed.model == "gemma-3-4b-pt"
    assert parsed.layer == 22
    assert parsed.n_sae_features == 65_536
    stats.write_bytes(b"drift")
    with pytest.raises(SFTGRPOError, match="digest changed"):
        OnlineSAELineage.from_checkpoint_meta(meta).validate_paths()


def test_lineage_builder_resolves_staged_statistics_symlink(tmp_path) -> None:
    canonical = (
        tmp_path
        / "canonical"
        / "s256"
        / "l22"
        / "a_res"
        / "p_max_pool"
        / "f65k"
        / "m_gemma-3-4b-pt"
        / "base"
        / "context_independent"
        / "no_bos"
    )
    canonical.mkdir(parents=True)
    canonical_stats = canonical / "feature_statistics.safetensors"
    canonical_stats.write_bytes(b"pinned-statistics")
    staged = tmp_path / "features_pretrain_stats" / "train"
    staged.mkdir(parents=True)
    staged_stats = staged / canonical_stats.name
    staged_stats.symlink_to(canonical_stats)

    meta = build_online_sae_lineage_metadata(
        feature_dir=str(staged),
        statistics_path=staged_stats,
        segment_mode=SegmentMode.fixed_tokens,
        sae_config={
            "model": "gemma-3-4b-pt",
            "release": "gemmascope-2",
            "activation_type": "residual",
            "layer": 22,
            "width": "65k",
        },
        feature_selection={"mode": "top_k_percentile"},
        negative_selection={"mean": 10.0},
        missing_percentile_policy="error",
        n_sae_features=65_536,
        num_buckets=2,
        selection_seed=42,
    )

    lineage = OnlineSAELineage.from_checkpoint_meta(meta)
    assert lineage.feature_dir == str(canonical)
    assert lineage.statistics_path == str(canonical_stats)

    with pytest.raises(SFTGRPOError, match="SAE config conflicts"):
        build_online_sae_lineage_metadata(
            feature_dir=str(staged),
            statistics_path=staged_stats,
            segment_mode=SegmentMode.fixed_tokens,
            sae_config={
                "model": "gemma-2-2b",
                "release": "gemmascope",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
            },
            feature_selection={"mode": "top_k_percentile"},
            negative_selection={"mean": 10.0},
            missing_percentile_policy="error",
            n_sae_features=65_536,
            num_buckets=2,
            selection_seed=42,
        )


def test_online_lineage_missing_block_fails_before_any_probe_activity() -> None:
    with pytest.raises(SFTGRPOError, match=ONLINE_SAE_LINEAGE_KEY):
        OnlineSAELineage.from_checkpoint_meta({})


# kills: dropping strict=True from EITHER path resolve in validate_paths. Each
# case leaves the other path present, so only its own check can raise, and the
# match pins the exact path that check is responsible for.
def test_lineage_path_validation_checks_feature_dir_and_statistics_separately(
    tmp_path: Path,
) -> None:
    feature_dir = _feature_dir(tmp_path / "present", SegmentMode.fixed_tokens)
    meta = _lineage_meta(feature_dir, SegmentMode.fixed_tokens)
    raw = meta[ONLINE_SAE_LINEAGE_KEY]

    missing_statistics = feature_dir / "absent_statistics.safetensors"
    stats_meta = copy.deepcopy(meta)
    stats_meta[ONLINE_SAE_LINEAGE_KEY]["statistics_path"] = str(missing_statistics)
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_statistics))):
        OnlineSAELineage.from_checkpoint_meta(stats_meta).validate_paths()

    absent_root = tmp_path / "absent"
    missing_feature_dir = Path(
        str(feature_dir).replace(str(tmp_path / "present"), str(absent_root))
    )
    missing_feature_dir.parent.mkdir(parents=True)
    assert not missing_feature_dir.exists()
    dir_meta = copy.deepcopy(meta)
    dir_meta[ONLINE_SAE_LINEAGE_KEY]["feature_dir"] = str(missing_feature_dir)
    assert Path(raw["statistics_path"]).is_file()
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_feature_dir))):
        OnlineSAELineage.from_checkpoint_meta(dir_meta).validate_paths()


def test_turn_lineage_parser_is_policy_neutral_but_grpo_rejects(tmp_path) -> None:
    feature_dir = _feature_dir(
        tmp_path,
        SegmentMode.turn_delimited_fixed_tokens,
    )
    meta = _lineage_meta(
        feature_dir,
        SegmentMode.turn_delimited_fixed_tokens,
        negative_selection={"mean": 10.0},
    )

    assert meta[ONLINE_SAE_LINEAGE_KEY]["source_segment_mode"] == (
        SegmentMode.turn_delimited_fixed_tokens.value
    )
    lineage = OnlineSAELineage.from_checkpoint_meta(meta)
    assert lineage.source_segment_mode == SegmentMode.turn_delimited_fixed_tokens.value
    with pytest.raises(
        SFTGRPOError,
        match=r"supports fixed_tokens.*turn_delimited_fixed_tokens",
    ):
        lineage.assert_grpo_supported()


@pytest.mark.parametrize(
    ("path_mode", "artifact_mode"),
    [
        (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
        (SegmentMode.turn_delimited_fixed_tokens, SegmentMode.fixed_tokens),
    ],
)
def test_lineage_builder_rejects_path_and_discovered_mode_disagreement(
    tmp_path,
    path_mode: SegmentMode,
    artifact_mode: SegmentMode,
) -> None:
    feature_dir = _feature_dir(tmp_path, path_mode)

    with pytest.raises(
        SFTGRPOError,
        match=(
            rf"leaf disagrees with the shard metadata.*{path_mode.value}.*"
            rf"{artifact_mode.value}.*canonical output_dir"
        ),
    ):
        _lineage_meta(feature_dir, artifact_mode)


def test_legacy_lineage_field_set_is_accepted(tmp_path) -> None:
    feature_dir = _feature_dir(tmp_path, SegmentMode.fixed_tokens)
    meta = _lineage_meta(feature_dir, SegmentMode.fixed_tokens)

    # Together with the turn-path rejection below, this pins the legacy
    # field-set default to fixed_tokens without storing a redundant field.
    assert OnlineSAELineage.from_checkpoint_meta(meta).feature_dir == str(feature_dir)


def test_unknown_or_missing_lineage_fields_are_rejected(tmp_path: Path) -> None:
    feature_dir = _feature_dir(tmp_path, SegmentMode.fixed_tokens)
    meta = _lineage_meta(feature_dir, SegmentMode.fixed_tokens)

    unknown = copy.deepcopy(meta)
    unknown[ONLINE_SAE_LINEAGE_KEY]["source_segment_modes"] = (
        SegmentMode.turn_delimited_fixed_tokens.value
    )
    with pytest.raises(
        SFTGRPOError,
        match=r"lineage fields are incomplete or unknown.*source_segment_modes",
    ):
        OnlineSAELineage.from_checkpoint_meta(unknown)

    missing = copy.deepcopy(meta)
    del missing[ONLINE_SAE_LINEAGE_KEY]["selection_seed"]
    with pytest.raises(
        SFTGRPOError,
        match=r"lineage fields are incomplete or unknown",
    ):
        OnlineSAELineage.from_checkpoint_meta(missing)


def test_unknown_source_segment_mode_names_valid_values(tmp_path: Path) -> None:
    feature_dir = _feature_dir(tmp_path, SegmentMode.fixed_tokens)
    meta = _lineage_meta(feature_dir, SegmentMode.fixed_tokens)
    meta[ONLINE_SAE_LINEAGE_KEY]["source_segment_mode"] = "turn_delimited"

    with pytest.raises(
        SFTGRPOError,
        match=(
            r"online-SAE.*unknown .*source_segment_mode='turn_delimited'.*"
            r"fixed_tokens.*turn_delimited_fixed_tokens"
        ),
    ):
        OnlineSAELineage.from_checkpoint_meta(meta)


def test_validate_paths_rejects_legacy_lineage_for_turn_path(tmp_path) -> None:
    feature_dir = _feature_dir(
        tmp_path,
        SegmentMode.turn_delimited_fixed_tokens,
    )
    meta = _lineage_meta(feature_dir, SegmentMode.turn_delimited_fixed_tokens)
    del meta[ONLINE_SAE_LINEAGE_KEY]["source_segment_mode"]

    with pytest.raises(
        SFTGRPOError,
        match=r"feature path conflicts.*segment_mode.*turn_delimited_fixed_tokens",
    ):
        OnlineSAELineage.from_checkpoint_meta(meta).validate_paths()
