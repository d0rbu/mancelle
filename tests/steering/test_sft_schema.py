from __future__ import annotations

from steering.sft_schema import SFT_LEGACY_COLUMNS, legacy_sft_columns


def test_legacy_sft_columns_includes_removed_loss_mask_path() -> None:
    assert "assistant_loss_mask" in SFT_LEGACY_COLUMNS


def test_legacy_sft_columns_returns_only_removed_sft_columns() -> None:
    assert legacy_sft_columns(
        ["text", "original_index", "assistant_loss_mask", "messages"],
    ) == {"assistant_loss_mask", "messages"}
