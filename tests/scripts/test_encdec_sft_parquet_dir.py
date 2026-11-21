"""Configurable SFT parquet directory tests.

``main()`` resolves train and validation parquet directories independently.
The train default is ``{base_dir}/sft_data/{cfg.dataset}``; the val default is
its ``{cfg.dataset}_val`` sibling, and each has an explicit override.

These tests cover:

* the resolver's mapping (null → default layout, explicit → passthrough)
* the config round-trip through Hydra compose
* source inspection of ``main()`` to catch regressions where the
  resolve call or its use at either call site gets accidentally
  removed
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.train.conditional_encdec_sft import (
    _build_sft_val_loader,
    _resolve_sft_parquet_dir,
    _resolve_sft_val_parquet_dir,
)
from steering.sft.loss_mask import LossMaskMode

# ---------------------------------------------------------------------------
# Unit: _resolve_sft_parquet_dir
# ---------------------------------------------------------------------------


class TestResolveSFTParquetDir:
    """``_resolve_sft_parquet_dir`` turns the config override into the
    absolute path the SFT data generator should read from.

    Two inputs: the operator's override value (``cfg.sft_parquet_dir``)
    and the run context (``base_dir``, ``dataset``).  The contract is
    simple: a ``None`` override returns the default
    ``{base_dir}/sft_data/{dataset}`` path, any other string is
    returned as-is so operators can point at arbitrary filesystem
    locations."""

    def test_none_returns_default_layout(self) -> None:
        """A null override uses the standard per-dataset parquet layout."""
        result = _resolve_sft_parquet_dir(override=None, base_dir="/base", dataset="safety")
        assert result == os.path.join("/base", "sft_data", "safety")

    def test_explicit_override_is_passthrough(self) -> None:
        """Operators supply absolute paths; the resolver does not
        rewrite or normalise them.  Passing ``/mnt/scratch/parquets``
        returns ``/mnt/scratch/parquets`` unchanged."""
        override = "/mnt/scratch/safety_parquets"
        result = _resolve_sft_parquet_dir(override=override, base_dir="/base", dataset="safety")
        assert result == override

    def test_override_ignores_dataset_name(self) -> None:
        """When an override is set, the ``dataset`` parameter is
        irrelevant — the override wins wholesale.  This catches a
        regression where the resolver tries to "help" by appending
        the dataset name to the override."""
        result = _resolve_sft_parquet_dir(
            override="/a/b/c", base_dir="/base", dataset="capabilities"
        )
        assert result == "/a/b/c"
        assert "capabilities" not in result
        assert "sft_data" not in result

    def test_override_ignores_base_dir(self) -> None:
        """Symmetric: ``base_dir`` is also ignored when an override is
        set."""
        result = _resolve_sft_parquet_dir(override="/elsewhere", base_dir="/base", dataset="safety")
        assert result == "/elsewhere"
        assert "/base" not in result

    @pytest.mark.parametrize(
        "dataset",
        ["standard", "capabilities", "safety"],
    )
    def test_default_path_for_each_dataset(self, dataset: str) -> None:
        """The default layout interpolates the dataset name verbatim
        — verify this for every configured dataset so a
        rename of any dataset would surface here."""
        result = _resolve_sft_parquet_dir(override=None, base_dir="/base", dataset=dataset)
        assert result == f"/base/sft_data/{dataset}"

    def test_path_is_stable_across_calls(self) -> None:
        """Two calls with the same arguments must produce the exact
        same path string.  Guards against accidental use of anything
        non-deterministic (tmp dir, timestamp, etc.)."""
        kwargs = {"override": None, "base_dir": "/base", "dataset": "safety"}
        assert _resolve_sft_parquet_dir(**kwargs) == _resolve_sft_parquet_dir(**kwargs)

    def test_tmp_path_override(self, tmp_path: Path) -> None:
        """A realistic override uses a pytest ``tmp_path`` — verify
        the resolver handles it without path-normalization surprises."""
        override = str(tmp_path / "my_parquets")
        result = _resolve_sft_parquet_dir(override=override, base_dir="/base", dataset="safety")
        assert result == override

    def test_empty_string_override_is_passthrough(self) -> None:
        """An empty-string override is treated as a value, not as
        ``None``: it is returned as-is.  This is a documented
        quirk — an empty path will fail downstream when
        ``make_sft_data_generator`` tries to load features from it,
        but the failure surfaces at feature-load time (with a clear
        filesystem error) rather than being silently rewritten to
        the default path.

        If this behavior is ever changed to treat ``""`` as
        equivalent to ``None``, update this test to reflect the
        new contract — a silent rewrite should be deliberate, not
        accidental.
        """
        result = _resolve_sft_parquet_dir(override="", base_dir="/base", dataset="safety")
        assert result == ""
        # Critically, it must NOT silently fall back to the default.
        # An operator passing ``""`` should see their value
        # honored (even though it will fail later), not mysteriously
        # get the default.
        assert result != os.path.join("/base", "sft_data", "safety")

    def test_trailing_slash_override_is_passthrough(self) -> None:
        """Operators may supply paths with or without trailing
        slashes; the resolver does not normalise them.  Filesystem
        operations downstream handle both correctly."""
        result = _resolve_sft_parquet_dir(override="/a/b/", base_dir="/base", dataset="safety")
        assert result == "/a/b/"

    def test_relative_path_override_is_passthrough(self) -> None:
        """Relative paths are passed through unchanged.  Operators
        who want absolute paths should supply absolute paths —
        the resolver does not abspath-ify."""
        result = _resolve_sft_parquet_dir(
            override="sft_data/custom", base_dir="/base", dataset="safety"
        )
        assert result == "sft_data/custom"


# ---------------------------------------------------------------------------
# Default path contract
# ---------------------------------------------------------------------------


class TestSFTParquetDirDefaultPath:
    """A null override produces the standard generated-parquet location."""

    def test_null_override_matches_default_layout(self) -> None:
        base_dir = "/tmp/nanochat_cache"
        dataset = "safety"
        default_path = os.path.join(base_dir, "sft_data", dataset)
        resolved = _resolve_sft_parquet_dir(override=None, base_dir=base_dir, dataset=dataset)
        assert resolved == default_path


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestSFTParquetDirConfig:
    """``configs/training/conditional_encdec_sft.yaml`` must expose
    ``sft_parquet_dir`` with a ``null`` default so standard generated
    parquet location is used unless overridden."""

    @pytest.fixture()
    def cfg(self, hydra_compose):
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        return hydra_compose(config_dir, "conditional_encdec_sft")

    def test_default_is_none(self, cfg) -> None:
        """Default ``sft_parquet_dir`` must be ``None`` (YAML ``null``)
        so the default path wins unless explicitly overridden."""
        assert cfg.sft_parquet_dir is None

    def test_override_to_explicit_path(self, hydra_compose) -> None:
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        cfg = hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            ["sft_parquet_dir=/mnt/parquets/safety"],
        )
        assert cfg.sft_parquet_dir == "/mnt/parquets/safety"

    def test_val_default_and_override(self, cfg, hydra_compose) -> None:
        assert cfg.val_sft_parquet_dir is None
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        overridden = hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            ["val_sft_parquet_dir=/mnt/parquets/safety_val"],
        )
        assert overridden.val_sft_parquet_dir == "/mnt/parquets/safety_val"
        assert _resolve_sft_val_parquet_dir(None, "/base", "safety") == "/base/sft_data/safety_val"


# ---------------------------------------------------------------------------
# Source inspection: main() wires the override through
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# End-to-end: override flows through to make_sft_data_generator
# ---------------------------------------------------------------------------


class TestOverrideFlowsThroughValLoader:
    """Integration-style tests that verify the resolved override
    actually reaches ``make_sft_data_generator`` via
    ``_build_sft_val_loader``.  Unit tests on the resolver prove the
    value is computed correctly; these tests prove the **plumbing**
    from ``cfg.sft_parquet_dir`` through the val loader factory is
    intact.

    Uses a ``make_sft_data_generator`` mock so we can inspect the
    kwargs it receives without actually reading the filesystem.
    """

    def test_val_loader_passes_resolved_override_through(self) -> None:
        """When the caller hands ``_build_sft_val_loader`` a
        ``parquet_dir`` override, it must show up verbatim in the
        ``make_sft_data_generator`` kwargs for the conditioned path."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        override = "/mnt/shared/safety_parquets"
        with mock_patch(
            "scripts.train.conditional_encdec_sft.make_sft_data_generator",
            new=MagicMock(return_value=iter([])),
        ) as mock_gen:
            _build_sft_val_loader(
                val_task=MagicMock(),
                tokenizer=MagicMock(),
                B=2,
                T=64,
                device="cpu",
                ddp_rank=0,
                ddp_world_size=1,
                feature_dir="/features",
                expected_feature_dir="/features",
                parquet_dir=override,
                threshold_cfg=None,
                boundaries=None,
                precomputed_thresholds=None,
                sampling_bundle=None,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
                min_loss_tokens=8,
            )
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["parquet_dir"] == override, (
            f"override must be passed through unchanged; got {kwargs['parquet_dir']!r}"
        )

    def test_resolver_plus_val_loader_end_to_end(self) -> None:
        """Compose the resolver + val loader the way ``main()`` does:
        resolve once, pass the result to ``_build_sft_val_loader``.
        Verify the override from the *config* flows all the way into
        the ``make_sft_data_generator`` call."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        cfg_override = "/mnt/ops/custom"
        resolved = _resolve_sft_parquet_dir(
            override=cfg_override, base_dir="/base", dataset="safety"
        )
        assert resolved == cfg_override  # sanity

        with mock_patch(
            "scripts.train.conditional_encdec_sft.make_sft_data_generator",
            new=MagicMock(return_value=iter([])),
        ) as mock_gen:
            _build_sft_val_loader(
                val_task=MagicMock(),
                tokenizer=MagicMock(),
                B=2,
                T=64,
                device="cpu",
                ddp_rank=0,
                ddp_world_size=1,
                feature_dir="/features",
                expected_feature_dir="/features",
                parquet_dir=resolved,
                threshold_cfg=None,
                boundaries=None,
                precomputed_thresholds=None,
                sampling_bundle=None,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
                min_loss_tokens=8,
            )
        assert mock_gen.call_args.kwargs["parquet_dir"] == cfg_override

    def test_resolver_plus_val_loader_null_override_uses_default(self) -> None:
        """When the config override is ``None``, the resolver produces
        the default path and that default path must flow through the
        val loader into ``make_sft_data_generator``."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        resolved = _resolve_sft_parquet_dir(override=None, base_dir="/base", dataset="safety")
        default_path = os.path.join("/base", "sft_data", "safety")
        assert resolved == default_path  # sanity

        with mock_patch(
            "scripts.train.conditional_encdec_sft.make_sft_data_generator",
            new=MagicMock(return_value=iter([])),
        ) as mock_gen:
            _build_sft_val_loader(
                val_task=MagicMock(),
                tokenizer=MagicMock(),
                B=2,
                T=64,
                device="cpu",
                ddp_rank=0,
                ddp_world_size=1,
                feature_dir="/features",
                expected_feature_dir="/features",
                parquet_dir=resolved,
                threshold_cfg=None,
                boundaries=None,
                precomputed_thresholds=None,
                sampling_bundle=None,
                loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
                min_loss_tokens=8,
            )
        assert mock_gen.call_args.kwargs["parquet_dir"] == default_path


class TestSFTParquetDirMainWiring:
    """Lightweight source inspection of ``main()`` to ensure both the
    train and val call sites read from the resolver's return value
    rather than building their own path."""

    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        import inspect

        from scripts.train.conditional_encdec_sft import main

        return inspect.getsource(main)

    def test_main_calls_resolve_helper(self, main_source: str) -> None:
        """main() must call the resolver — if someone inlines the
        logic, we lose the single point of truth and the test
        coverage that goes with it."""
        assert "_resolve_sft_parquet_dir" in main_source, (
            "main() must call _resolve_sft_parquet_dir rather than inlining the path construction"
        )

    def test_main_references_sft_parquet_dir_config(self, main_source: str) -> None:
        assert "sft_parquet_dir" in main_source, (
            "main() must read cfg.sft_parquet_dir so operator overrides take effect"
        )
        assert "val_sft_parquet_dir" in main_source
        assert "resolved_sft_val_parquet_dir" in main_source
        assert "parquet_dir=resolved_sft_val_parquet_dir" in main_source, (
            "build_val_loader must pass the resolved VAL parquet dir (#1278)"
        )
        assert "val_feature_dir" in main_source
        assert "feature_dir=resolved_val_feature_dir" in main_source, (
            "build_val_loader must pass the resolved VAL feature dir, not cfg.feature_dir (#1278)"
        )
        assert "expected_feature_dir=resolved_val_feature_dir" in main_source

    def test_main_no_longer_inlines_hardcoded_path(self, main_source: str) -> None:
        """Catch a regression where the hardcoded
        ``os.path.join(base_dir, "sft_data", cfg.dataset)`` expression
        sneaks back in after a refactor, bypassing the resolver."""
        # The literal ``"sft_data"`` must not appear in main() anymore —
        # it lives exclusively inside ``_resolve_sft_parquet_dir``.
        assert '"sft_data"' not in main_source, (
            'main() should not reference the "sft_data" string '
            "literal directly; use _resolve_sft_parquet_dir instead"
        )
