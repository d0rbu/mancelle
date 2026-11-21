"""_warn_if_path_flag_mismatch handles the /no_bos leaf too."""

import logging

from steering.extract_features import _warn_if_path_flag_mismatch
from steering.segmentation import SegmentMode


def test_no_bos_flag_true_leaf_present_no_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/no_bos",
            context_independent_flag=False,
            ignore_bos_flag=True,
        )
    assert "ignore_bos" not in caplog.text.lower()


def test_no_bos_flag_true_leaf_absent_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
            context_independent_flag=False,
            ignore_bos_flag=True,
        )
    assert "ignore_bos_features" in caplog.text


def test_both_leaves_present_both_flags_no_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/x/base/context_independent/no_bos",
            context_independent_flag=True,
            ignore_bos_flag=True,
        )
    assert caplog.text == "" or "mismatch" not in caplog.text


def test_no_bos_leaf_present_flag_false_warns(caplog):
    """Reverse mismatch: path ends .../no_bos but both flags are False → warns."""
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/no_bos",
            context_independent_flag=False,
            ignore_bos_flag=False,
        )
    assert "ignore_bos_features" in caplog.text


def test_no_bos_flag_false_leaf_absent_no_warn(caplog):
    """Clean legacy case: path has no no_bos leaf and flag is False → no warn."""
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
            context_independent_flag=False,
            ignore_bos_flag=False,
        )
    assert "ignore_bos" not in caplog.text.lower()


def test_turn_mode_leaf_absent_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/x/base/no_bos",
            context_independent_flag=False,
            ignore_bos_flag=True,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )
    assert "segment_mode" in caplog.text
    assert "turn_delimited_fixed_tokens" in caplog.text


def test_turn_leaf_with_fixed_mode_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/x/base/no_bos/turn_delimited",
            context_independent_flag=False,
            ignore_bos_flag=True,
            segment_mode=SegmentMode.fixed_tokens,
        )
    assert "turn_delimited leaf present=True" in caplog.text


def test_turn_leaf_matching_mode_preserves_other_leaf_detection(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_if_path_flag_mismatch(
            "out/x/base/no_bos/turn_delimited",
            context_independent_flag=False,
            ignore_bos_flag=True,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )
    assert caplog.text == "" or "mismatch" not in caplog.text
