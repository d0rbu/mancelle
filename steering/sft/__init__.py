"""Shared supervised fine-tuning data utilities."""

from steering.sft.iteration import (
    SFTGeneratorProgress,
    SFTLoopPosition,
    find_bestfit_index,
    resolve_sft_loop_position,
    sft_data_generator_bos_bestfit_core,
    should_save_checkpoint,
)
from steering.sft.loss_mask import (
    LossMaskMode,
    apply_completion_only_row_masks,
    assert_min_one_valid_target,
    assert_resume_sft_loss_mask_matches,
    parse_loss_mask_mode,
    render_conversation_with_mask,
    supervised_target_count,
    truncate_conv_and_mask,
)

__all__ = [
    "LossMaskMode",
    "SFTGeneratorProgress",
    "SFTLoopPosition",
    "apply_completion_only_row_masks",
    "assert_min_one_valid_target",
    "assert_resume_sft_loss_mask_matches",
    "find_bestfit_index",
    "parse_loss_mask_mode",
    "render_conversation_with_mask",
    "resolve_sft_loop_position",
    "sft_data_generator_bos_bestfit_core",
    "should_save_checkpoint",
    "supervised_target_count",
    "truncate_conv_and_mask",
]
