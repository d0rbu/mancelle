"""Conditioned-SFT config-surface contracts for issue #1281."""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train import conditional_encdec_sft as sft

REPO_ROOT = Path(__file__).parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "training"
SFT_CONFIG_NAMES = (
    "conditional_encdec_sft",
    "conditional_encdec_sft_4090",
    "conditional_encdec_sft_2xh200",
    "conditional_encdec_sft_4xh200",
    "conditional_encdec_sft_2xb200",
    "conditional_encdec_sft_profile_4090",
)

# These keys are consumed by helpers that deliberately accept the composed cfg
# as a mapping. Keep the call-site pins beside the annotations so they cannot
# become a hand-waved exemption from the source-derived read set.
INDIRECT_READ_KEYS = {
    "feature_mask": "resolve_feature_mask consumes cfg and resolves this key",
}

# Existing SFT provenance metadata, not one of #1281's inherited pretraining
# controls. The runtime authenticates SAE identity through feature/checkpoint
# artifacts; retaining the structured block is a separately reviewable surface.
ALLOW_LIST = {
    "sae": "retained provenance block; runtime identity is artifact-authenticated",
}


def _parse_config_accesses(source: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(source)
    reads: set[str] = set()
    writes: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cfg"
            and node.attr != "get"
        ):
            reads.add(node.attr)

        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cfg"
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            reads.add(node.args[0].value)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "cfg"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            reads.add(node.args[1].value)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "OmegaConf"
            and node.func.attr == "update"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "cfg"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            writes.add(node.args[1].value)

    return reads, writes


def _source_config_accesses() -> tuple[set[str], set[str]]:
    source = Path(sft.__file__).read_text(encoding="utf-8")
    reads, writes = _parse_config_accesses(source)
    assert "resolve_feature_mask(cfg" in source
    reads.update(INDIRECT_READ_KEYS)
    return reads, writes


def _composed_top_level_keys(config_name: str) -> set[str]:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    data = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(data, dict)
    data.pop("hydra", None)
    # Structured knobs remain one top-level key: a cfg.feature_selection or
    # cfg.sae read consumes the parent block wholesale rather than requiring a
    # synthetic cfg access for each nested field.
    assert all(isinstance(key, str) for key in data)
    return {key for key in data if isinstance(key, str)}


def _guard_match(key: str) -> str:
    return (
        rf"does not honour pretrain-only key '{re.escape(key)}'.*"
        rf"scripts/train/conditional_encdec_train\.py.*checkpoint metadata"
    )


def _assert_composed_surface_is_explained(config_name: str) -> None:
    reads, writes = _source_config_accesses()
    unexplained = _composed_top_level_keys(config_name) - reads - writes
    assert unexplained == set(ALLOW_LIST), {
        "config_name": config_name,
        "unexplained": sorted(unexplained),
        "allowed": ALLOW_LIST,
        "indirect_reads": INDIRECT_READ_KEYS,
    }


@pytest.mark.parametrize("key", sft._UNSUPPORTED_SFT_PRETRAIN_KEYS)
def test_appended_pretrain_only_key_hits_post_compose_guard(
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sft.sys,
        "argv",
        ["conditional_encdec_sft.py", f"+{key}=ignored"],
    )

    with pytest.raises(ValueError, match=_guard_match(key)):
        sft.main()


def test_inherited_pretrain_only_key_hits_post_compose_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "training"
    shutil.copytree(CONFIG_DIR, config_dir)
    (config_dir / "conditional_encdec_sft_inherited_fp8.yaml").write_text(
        "defaults:\n  - conditional_encdec_sft\n  - _self_\nfp8: true\n",
        encoding="utf-8",
    )
    initialize = sft.initialize_config_dir
    monkeypatch.setattr(
        sft,
        "initialize_config_dir",
        lambda **_: initialize(config_dir=str(config_dir), version_base=None),
    )
    monkeypatch.setattr(
        sft.sys,
        "argv",
        [
            "conditional_encdec_sft.py",
            "config_name=conditional_encdec_sft_inherited_fp8",
        ],
    )

    with pytest.raises(ValueError, match=_guard_match("fp8")):
        sft.main()


@pytest.mark.parametrize("config_name", SFT_CONFIG_NAMES)
def test_composed_sft_surface_has_only_read_written_or_allowed_keys(
    config_name: str,
) -> None:
    _assert_composed_surface_is_explained(config_name)


def test_composed_surface_detector_catches_reintroduced_dead_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dead_key = "synthetic_reintroduced_dead_key"
    composed_keys = _composed_top_level_keys("conditional_encdec_sft")
    original_composed_keys = _composed_top_level_keys

    def compose_with_reintroduced_dead_key(config_name: str) -> set[str]:
        assert config_name == "conditional_encdec_sft"
        return original_composed_keys(config_name) | {dead_key}

    monkeypatch.setattr(
        sys.modules[__name__],
        "_composed_top_level_keys",
        compose_with_reintroduced_dead_key,
    )
    assert dead_key not in composed_keys

    with pytest.raises(AssertionError) as failure:
        _assert_composed_surface_is_explained("conditional_encdec_sft")
    assert dead_key in str(failure.value)


def test_cross_chain_sft_surface_reads_and_removals_are_detectable() -> None:
    reads, _ = _source_config_accesses()
    assert {
        "alignment_probe_samples",
        "feature_clusters",
        "sft_loss_mask",
        "val_feature_dir",
        "val_sft_parquet_dir",
    } <= reads

    for config_name in SFT_CONFIG_NAMES:
        composed = _composed_top_level_keys(config_name)
        assert "feature_mask_freq_min" not in composed
        assert "feature_mask_freq_max" not in composed


def test_pretrain_only_and_removed_frequency_guards_do_not_overlap() -> None:
    assert len(sft._UNSUPPORTED_SFT_PRETRAIN_KEYS) == 19
    assert set(sft._UNSUPPORTED_SFT_PRETRAIN_KEYS).isdisjoint(
        sft._REMOVED_SFT_FREQUENCY_KEYS
    )


def test_config_access_parser_covers_all_supported_source_forms() -> None:
    source = """
value_x = cfg.x
value_y = cfg.get("y")
value_z = getattr(cfg, "z")
OmegaConf.update(cfg, "w", value_x)
"""

    reads, writes = _parse_config_accesses(source)

    assert reads == {"x", "y", "z"}
    assert writes == {"w"}


class _TinyOptimizerModel:
    def __init__(self) -> None:
        self.parameters = [t.nn.Parameter(t.tensor(float(i))) for i in range(7)]
        self.setup_kwargs: dict[str, object] = {}

    def setup_optimizer(self, **kwargs):
        self.setup_kwargs = kwargs
        lr_keys = (
            "unembedding_lr",
            "embedding_lr",
            "matrix_lr",
            "fp_head_matrix_lr",
            "encoder_embedding_lr",
            "encoder_matrix_lr",
            "xattn_matrix_lr",
        )
        groups = [
            {"params": [parameter], "lr": kwargs[key]}
            for parameter, key in zip(self.parameters, lr_keys, strict=True)
        ]
        return t.optim.SGD(groups, lr=0.0)


def _optimizer_capture(
    init_lr_frac: float,
    *,
    packing: bool | None = None,
) -> tuple[_TinyOptimizerModel, list[float]]:
    model = _TinyOptimizerModel()
    cfg = OmegaConf.create(
        {
            "unembedding_lr": 0.004,
            "embedding_lr": 0.3,
            "matrix_lr": 0.02,
            "fp_head_lr": None,
            "fp_head_matrix_lr": 0.02,
            "weight_decay": 0.0,
            "encoder_embedding_lr": 0.2,
            "encoder_matrix_lr": 0.02,
            "xattn_matrix_lr": 0.02,
            "lr_multiplier": 0.25,
            "init_lr_frac": init_lr_frac,
        }
    )
    if packing is not None:
        cfg.muon_parameter_packing = packing
    optimizer = sft._setup_sft_optimizer(model, cfg)
    initial_lrs = [group["initial_lr"] for group in optimizer.param_groups]
    assert [group["lr"] for group in optimizer.param_groups] == initial_lrs
    return model, initial_lrs


def test_init_lr_frac_halves_every_optimizer_group_initial_lr() -> None:
    full_model, full_lrs = _optimizer_capture(1.0)
    half_model, half_lrs = _optimizer_capture(0.5)

    assert len(full_lrs) == len(half_lrs) == 7
    assert half_lrs == pytest.approx([lr * 0.5 for lr in full_lrs])
    assert full_model.setup_kwargs["embedding_lr"] == pytest.approx(0.3 * 0.25)
    assert full_model.setup_kwargs["fp_head_matrix_lr"] == pytest.approx(0.02 * 0.25)
    assert half_model.setup_kwargs == full_model.setup_kwargs


def test_packing_opt_in_preserves_sft_learning_rate_scaling() -> None:
    default_model, default_lrs = _optimizer_capture(0.5)
    packed_model, packed_lrs = _optimizer_capture(0.5, packing=True)
    assert default_model.setup_kwargs["muon_parameter_packing"] is False
    assert packed_model.setup_kwargs["muon_parameter_packing"] is True
    assert packed_lrs == default_lrs
    assert packed_model.setup_kwargs == {
        **default_model.setup_kwargs,
        "muon_parameter_packing": True,
    }


def test_dry_run_skips_checkpoint_file_but_false_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta, *, rank):
        del model_data, optimizer_data, meta, rank
        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"checkpoint_{step}.pt").write_bytes(b"checkpoint")

    monkeypatch.setattr(sft, "save_checkpoint", write_checkpoint)
    model = t.nn.Linear(2, 2)
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    dry_dir = tmp_path / "dry"
    sft._save_sft_checkpoint(
        dry_run=True,
        checkpoint_dir=str(dry_dir),
        step=7,
        model=model,
        optimizer=optimizer,
        meta={"step": 7},
        rank=0,
    )
    assert not list(dry_dir.glob("checkpoint_*.pt"))

    live_dir = tmp_path / "live"
    sft._save_sft_checkpoint(
        dry_run=False,
        checkpoint_dir=str(live_dir),
        step=7,
        model=model,
        optimizer=optimizer,
        meta={"step": 7},
        rank=0,
    )
    assert [path.name for path in live_dir.glob("checkpoint_*.pt")] == ["checkpoint_7.pt"]
