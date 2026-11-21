"""Issue #506: recomputed_context mode is fully removed; fixed_context is the
sole prompt-token mode (additive direction records not_applicable)."""

from __future__ import annotations

import pytest

from steering.eval.axbench_bpb_adv import context as context_mod
from steering.eval.axbench_bpb_adv.aggregation import _context_linestyle
from steering.eval.axbench_bpb_adv.config import ContextSignalMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def test_context_signal_mode_has_only_fixed_and_not_applicable() -> None:
    assert {mode.value for mode in ContextSignalMode} == {"fixed_context", "not_applicable"}
    assert not hasattr(ContextSignalMode, "RECOMPUTED_CONTEXT")


def test_coerce_recomputed_context_raises() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="Unknown context signal mode"):
        ContextSignalMode.coerce("recomputed_context")


@pytest.mark.parametrize(
    "name",
    [
        "recompute_tail_context_segments",
        "extract_attacked_prompt_token_features",
        "ATTACKED_PROMPT_FEATURE_SEGMENT_SIZE",
        "_aggregate_recomputed_features_for_projected_segment",
        "_char_ranges_overlap",
    ],
)
def test_recompute_symbols_removed_from_context_module(name: str) -> None:
    assert not hasattr(context_mod, name)


def test_context_linestyle_has_no_recomputed_entry() -> None:
    assert _context_linestyle("fixed_context") == "-"
    # not_applicable draws solid (omitted from the context legend); the removed
    # recomputed_context has no dedicated entry, so it falls to the ":" default.
    assert _context_linestyle("not_applicable") == "-"
    assert _context_linestyle("recomputed_context") == ":"
