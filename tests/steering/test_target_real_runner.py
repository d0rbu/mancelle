from __future__ import annotations

import pytest

from steering.eval.context_cache_io import ContextCacheEntry
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.eval.target_real_runner import (
    context_segments_for_target_real_prefix,
)


class _Provider:
    def __init__(self, rows):
        self._rows = {
            tuple(int(tok) for tok in token_ids): segments
            for token_ids, segments in rows.items()
        }

    def get(self, token_ids):
        return self._rows.get(tuple(int(tok) for tok in token_ids))


def test_context_segments_for_target_real_prefix_requires_exact_prompt_row():
    exact_segments = (
        ContextFeatureSegment((9,), (1,), 0, 3),
    )
    full_entry = ContextCacheEntry.from_sequences(
        [1, 2, 3, 4],
        [ContextFeatureSegment((1,), (1,), 0, 10)],
    )

    result = context_segments_for_target_real_prefix(
        provider=_Provider({(1, 2, 3): exact_segments}),
        context_token_ids=[1, 2, 3],
        trc_name="lockstep",
    )

    assert result == exact_segments

    with pytest.raises(KeyError, match="no prompt/context cache row"):
        context_segments_for_target_real_prefix(
            provider=_Provider({(1, 2, 3, 4): full_entry.segments}),
            context_token_ids=[1, 2, 3],
            trc_name="lockstep",
        )


def test_context_segments_for_target_real_prefix_rejects_empty_exact_row():
    with pytest.raises(KeyError, match="has no segments"):
        context_segments_for_target_real_prefix(
            provider=_Provider({(1, 2, 3): ()}),
            context_token_ids=[1, 2, 3],
            trc_name="lockstep",
        )
