"""Hardware-variant SFT config tests.

Before PR 7, ``configs/training/`` had four hardware-variant configs
for pretraining (``conditional_encdec_4090.yaml``,
``conditional_encdec_4xh200.yaml``, ``conditional_encdec_2xh200.yaml``,
``conditional_encdec_2xb200.yaml``) but **zero** for SFT — the SFT
script only had the single default ``conditional_encdec_sft.yaml``
tuned for 8xH100.  Operators on smaller or different hardware had to
pass batch sizes and sequence lengths on the CLI every time.

PR 7 adds four SFT variants that mirror the pretraining layout:

* ``conditional_encdec_sft_4090.yaml`` — 24GB 4090
* ``conditional_encdec_sft_2xh200.yaml`` — 2xH200 (144GB each)
* ``conditional_encdec_sft_4xh200.yaml`` — 4xH200
* ``conditional_encdec_sft_2xb200.yaml`` — 2xB200 (180GB each)

Each variant:

1. **Inherits from the default ``conditional_encdec_sft`` config**
   via ``defaults: - conditional_encdec_sft`` so every new knob
   added to the default propagates automatically.
2. **Overrides ``device_batch_size``** to fit the target hardware's
   HBM budget.
3. **May override ``max_seq_len``** when a hardware-specific run needs
   a shorter window; the shared project default is 1024.
4. **Preserves the inherited ``total_batch_size``** (524288 tokens)
   so gradient-accumulation automatically scales to match.  The
   divisibility invariant
   ``total_batch_size % (device_batch_size * max_seq_len * num_gpus) == 0``
   must hold or the SFT script's startup assert will fire.

These tests pin the above invariants: every variant loads through
Hydra, inherits the SFT default, actually overrides a field, and
produces a valid grad-accum divisor for its target GPU count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class HardwareVariant:
    """Metadata for a single hardware-variant SFT config.

    ``num_gpus`` is the assumed ``ddp_world_size`` the YAML targets
    — used to verify the ``total_batch_size`` divisibility invariant
    without actually spinning up DDP.
    """

    config_name: str
    num_gpus: int
    expected_device_batch_size: int
    description: str


HARDWARE_VARIANTS: list[HardwareVariant] = [
    HardwareVariant(
        config_name="conditional_encdec_sft_4090",
        num_gpus=1,
        expected_device_batch_size=4,
        description="RTX 4090 (24GB HBM); single-GPU SFT",
    ),
    HardwareVariant(
        config_name="conditional_encdec_sft_2xh200",
        num_gpus=2,
        expected_device_batch_size=8,
        description="2xH200 (144GB each); 2-GPU DDP",
    ),
    HardwareVariant(
        config_name="conditional_encdec_sft_4xh200",
        num_gpus=4,
        expected_device_batch_size=16,
        description="4xH200; 4-GPU DDP",
    ),
    HardwareVariant(
        config_name="conditional_encdec_sft_2xb200",
        num_gpus=2,
        expected_device_batch_size=32,
        description="2xB200 (180GB each); 2-GPU DDP",
    ),
]


# ---------------------------------------------------------------------------
# Composability: each variant loads through Hydra
# ---------------------------------------------------------------------------


class TestHardwareVariantsCompose:
    """Every variant must load through ``hydra.compose`` without
    raising.  A composition failure here usually means a typo in the
    ``defaults`` list or an override that references a field the
    parent doesn't have."""

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_variant_composes_via_hydra(
        self, hydra_compose, config_dir: str, variant: HardwareVariant
    ) -> None:
        """Hydra must resolve the variant's defaults list, apply
        overrides, and return a config object."""
        cfg = hydra_compose(config_dir, variant.config_name)
        assert cfg is not None

    def test_default_source_bos_char_len_matches_gemma(
        self, hydra_compose, config_dir: str
    ) -> None:
        """SFT feature coordinates include Gemma's decoded ``<bos>`` span."""
        cfg = hydra_compose(config_dir, "conditional_encdec_sft")
        assert cfg.feature_source_bos_char_len == 5
        assert cfg.alignment_probe_samples == 8

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_variant_inherits_sft_defaults(
        self, hydra_compose, config_dir: str, variant: HardwareVariant
    ) -> None:
        """Every variant must carry forward keys that only exist in
        ``conditional_encdec_sft``: ``dataset``, ``sft_parquet_dir``,
        ``grad_clip_max_norm``, ``min_loss_tokens``.  If one of these is missing the
        variant is not inheriting properly and operators who switch
        to it lose the SFT-specific runtime defaults."""
        cfg = hydra_compose(config_dir, variant.config_name)
        for key in (
            "dataset",
            "sft_parquet_dir",
            "grad_clip_max_norm",
            "min_loss_tokens",
            "alignment_probe_samples",
            "model_tag",  # from the SFT default (required, set to ???)
        ):
            assert key in cfg, (
                f"{variant.config_name} must inherit ``{key}`` from "
                f"conditional_encdec_sft.yaml"
            )
        assert cfg.alignment_probe_samples == 8

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_variant_overrides_device_batch_size(
        self, hydra_compose, config_dir: str, variant: HardwareVariant
    ) -> None:
        """The whole point of a hardware variant is to override the
        ``device_batch_size`` for the target GPU's HBM budget.  If
        this override is missing, the variant is dead weight."""
        cfg = hydra_compose(config_dir, variant.config_name)
        assert cfg.device_batch_size == variant.expected_device_batch_size, (
            f"{variant.config_name} should set device_batch_size to "
            f"{variant.expected_device_batch_size}, got {cfg.device_batch_size}"
        )


# ---------------------------------------------------------------------------
# Batch-size math: total_batch_size must be cleanly divisible
# ---------------------------------------------------------------------------


class TestHardwareVariantsBatchMath:
    """``conditional_encdec_sft.py``'s main() asserts at startup that
    ``total_batch_size`` is divisible by
    ``device_batch_size * max_seq_len * ddp_world_size``.  A variant
    that violates this would crash on run — an expensive failure
    mode for operators who just copy-pasted the config.  These
    tests catch the violation at config-load time."""

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_total_batch_size_divisible(
        self, hydra_compose, config_dir: str, variant: HardwareVariant
    ) -> None:
        """The inherited ``total_batch_size`` (524288 tokens) must
        evenly divide into ``device_batch_size * max_seq_len *
        num_gpus`` chunks.  Otherwise the SFT startup assert fires
        and the run crashes before the first step."""
        cfg = hydra_compose(config_dir, variant.config_name)
        world_tokens_per_fwdbwd = (
            cfg.device_batch_size * (cfg.max_seq_len or 1024) * variant.num_gpus  # 1024 fallback for null (checkpoint-derived) configs
        )
        assert cfg.total_batch_size % world_tokens_per_fwdbwd == 0, (
            f"{variant.config_name}: total_batch_size={cfg.total_batch_size} "
            f"is not divisible by device_batch_size={cfg.device_batch_size} "
            f"* max_seq_len={cfg.max_seq_len} * num_gpus={variant.num_gpus} "
            f"= {world_tokens_per_fwdbwd}"
        )

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_positive_grad_accum_steps(
        self, hydra_compose, config_dir: str, variant: HardwareVariant
    ) -> None:
        """Grad-accum steps must be >= 1 — i.e. the total batch is at
        least one world-step.  A zero or negative value would mean
        the variant oversubscribes the HBM budget for its target
        hardware (``total_batch_size`` smaller than a single world
        step), which is a config error."""
        cfg = hydra_compose(config_dir, variant.config_name)
        world_tokens_per_fwdbwd = (
            cfg.device_batch_size * (cfg.max_seq_len or 1024) * variant.num_gpus  # 1024 fallback for null (checkpoint-derived) configs
        )
        grad_accum = cfg.total_batch_size // world_tokens_per_fwdbwd
        assert grad_accum >= 1, (
            f"{variant.config_name}: grad_accum={grad_accum} must be >= 1 "
            f"(total_batch_size={cfg.total_batch_size}, "
            f"world_tokens_per_fwdbwd={world_tokens_per_fwdbwd})"
        )


# ---------------------------------------------------------------------------
# Variant-specific expectations
# ---------------------------------------------------------------------------


class TestHardwareVariantSpecifics:
    """Per-variant behaviors that don't fit the parametrized pattern:
    4090 explicitly pins ``max_seq_len=1024``; H200/B200 variants
    inherit the checkpoint-derived value, which is 1024 for the
    current from-scratch project default."""

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    def test_4090_shrinks_max_seq_len(
        self, hydra_compose, config_dir: str
    ) -> None:
        cfg = hydra_compose(config_dir, "conditional_encdec_sft_4090")
        assert cfg.max_seq_len == 1024, (
            "4090 variant must shrink max_seq_len to 1024 to fit 24GB HBM; "
            f"got {cfg.max_seq_len}"
        )

    def test_h200_variants_inherit_null_max_seq_len(
        self, hydra_compose, config_dir: str
    ) -> None:
        """H200 variants inherit null max_seq_len (use checkpoint value)."""
        for name in (
            "conditional_encdec_sft_2xh200",
            "conditional_encdec_sft_4xh200",
        ):
            cfg = hydra_compose(config_dir, name)
            assert cfg.max_seq_len is None, (
                f"{name} should inherit null max_seq_len; "
                f"got {cfg.max_seq_len}"
            )

    def test_b200_inherits_null_max_seq_len(
        self, hydra_compose, config_dir: str
    ) -> None:
        """B200 inherits null max_seq_len (use checkpoint value)."""
        cfg = hydra_compose(config_dir, "conditional_encdec_sft_2xb200")
        assert cfg.max_seq_len is None

    def test_all_variants_inherit_null_grad_clip_default(
        self, hydra_compose, config_dir: str
    ) -> None:
        """Hardware choices must not re-enable the disabled default."""
        for variant in HARDWARE_VARIANTS:
            cfg = hydra_compose(config_dir, variant.config_name)
            assert cfg.grad_clip_max_norm is None, (
                f"{variant.config_name} must inherit the null grad clip "
                f"default from the SFT config; got {cfg.grad_clip_max_norm}"
            )

        # This profiling variant lives outside HARDWARE_VARIANTS but must still
        # inherit the null default from the SFT config.
        cfg = hydra_compose(config_dir, "conditional_encdec_sft_profile_4090")
        assert cfg.grad_clip_max_norm is None

    def test_all_variants_remove_legacy_val_mode_default(
        self, hydra_compose, config_dir: str
    ) -> None:
        """Hardware choice must not reintroduce removed validation modes."""
        for variant in HARDWARE_VARIANTS:
            cfg = hydra_compose(config_dir, variant.config_name)
            assert "label_features" not in cfg
            assert "val_label_features" not in cfg


# ---------------------------------------------------------------------------
# File presence and shape
# ---------------------------------------------------------------------------


class TestHardwareVariantFiles:
    """Basic sanity on the files themselves — they exist, they are
    YAML, they have a ``defaults`` list, they have a usage comment
    at the top.  These checks don't need Hydra at all and catch the
    fastest-to-diagnose failures."""

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_variant_file_exists(
        self, config_dir: str, variant: HardwareVariant
    ) -> None:
        path = os.path.join(config_dir, f"{variant.config_name}.yaml")
        assert os.path.isfile(path), (
            f"{variant.config_name}.yaml must exist at {path}"
        )

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_variant_inherits_from_sft_default(
        self, config_dir: str, variant: HardwareVariant
    ) -> None:
        """The ``defaults`` list must reference
        ``conditional_encdec_sft``, not e.g. ``base_train`` or
        ``conditional_encdec_train``.  Inheriting from the wrong
        parent would silently lose SFT-specific keys."""
        path = os.path.join(config_dir, f"{variant.config_name}.yaml")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "- conditional_encdec_sft" in content, (
            f"{variant.config_name}.yaml must list ``- conditional_encdec_sft`` "
            f"in its defaults so it inherits the SFT-specific knobs"
        )

    @pytest.mark.parametrize(
        "variant", HARDWARE_VARIANTS, ids=lambda v: v.config_name
    )
    def test_variant_has_usage_comment(
        self, config_dir: str, variant: HardwareVariant
    ) -> None:
        """Every variant should have a top-of-file comment showing
        an operator how to invoke it — which ``torchrun`` flags,
        which ``nproc_per_node``, etc.  These are copied into
        run scripts all the time and serve as self-documentation."""
        path = os.path.join(config_dir, f"{variant.config_name}.yaml")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # The first non-blank line should be a # comment
        first_lines = [
            line for line in content.splitlines() if line.strip()
        ][:5]
        assert any(line.startswith("#") for line in first_lines), (
            f"{variant.config_name}.yaml should have a usage comment "
            f"near the top"
        )
        # And the comment block should mention the hardware
        comment_block = "\n".join(
            line for line in content.splitlines() if line.startswith("#")
        )
        hw_keywords = {
            "conditional_encdec_sft_4090": ["4090"],
            "conditional_encdec_sft_2xh200": ["2xH200", "H200"],
            "conditional_encdec_sft_4xh200": ["4xH200", "H200"],
            "conditional_encdec_sft_2xb200": ["2xB200", "B200"],
        }
        expected = hw_keywords[variant.config_name]
        assert any(kw in comment_block for kw in expected), (
            f"{variant.config_name}.yaml comment block must mention one "
            f"of {expected}; got:\n{comment_block}"
        )


class TestSFTProfilingConfig:
    """The dedicated 4090 profiling config should compose and keep the
    run short enough for iterative d8 bottleneck investigations."""

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    def test_profile_4090_config_composes(self, hydra_compose, config_dir: str) -> None:
        cfg = hydra_compose(config_dir, "conditional_encdec_sft_profile_4090")
        assert cfg.model_tag == "d8"
        assert cfg.device_batch_size == 4
        assert cfg.max_seq_len == 1024
        assert cfg.profile_timing is True
        assert cfg.profile_trace == 2
        assert cfg.profile_trace_steps == 2
        assert cfg.profile_output_dir.endswith("encdec_sft_d8_4090")

    def test_profile_4090_batch_math(self, hydra_compose, config_dir: str) -> None:
        cfg = hydra_compose(config_dir, "conditional_encdec_sft_profile_4090")
        world_tokens_per_fwdbwd = cfg.device_batch_size * cfg.max_seq_len
        assert cfg.total_batch_size % world_tokens_per_fwdbwd == 0
        assert cfg.total_batch_size // world_tokens_per_fwdbwd == 8
