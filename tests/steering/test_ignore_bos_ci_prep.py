"""Context-independent prep populates per-segment keep_mask when ignore_bos."""

from steering.extract_features import _prepare_context_independent_segment_inputs

BOS = 2


def test_no_ignore_bos_mask_is_none():
    tok_info = [([2, 10, 11, 12], 4)]
    precomp = [([(0, 4)], 4, [0], [4])]
    out = _prepare_context_independent_segment_inputs(
        tok_info, precomp, bos_token_id=BOS, ignore_bos_features=False
    )
    assert len(out) == 1
    assert out[0].keep_mask is None


def test_ignore_bos_natural_seg0_mask():
    tok_info = [([2, 10, 11, 12], 4)]
    precomp = [([(0, 4)], 4, [0], [4])]
    out = _prepare_context_independent_segment_inputs(
        tok_info, precomp, bos_token_id=BOS, ignore_bos_features=True
    )
    assert out[0].bos_injected is False
    assert out[0].keep_mask == (False, True, True, True)


def test_ignore_bos_injected_segment_mask_all_true():
    tok_info = [([2, 9, 10, 11], 4)]
    precomp = [([(1, 3)], 4, [0], [2])]  # segment (1,3) → tokens [9,10]
    out = _prepare_context_independent_segment_inputs(
        tok_info, precomp, bos_token_id=BOS, ignore_bos_features=True
    )
    assert out[0].bos_injected is True
    assert out[0].keep_mask == (True, True)


def test_ci_prep_multi_doc_keep_masks():
    """Two docs, each with ≥1 segment; keep_mask matches tok != BOS for each seg."""
    # Doc 0: tokens [BOS, 10, 11, BOS, 12, 13], two segments (0,3) and (3,6)
    # Doc 1: tokens [BOS, 20, 21], one segment (0,3)
    tok_info = [
        ([BOS, 10, 11, BOS, 12, 13], 6),
        ([BOS, 20, 21], 3),
    ]
    precomp = [
        ([(0, 3), (3, 6)], 6, [0, 3], [3, 6]),
        ([(0, 3)], 3, [0], [3]),
    ]
    out = _prepare_context_independent_segment_inputs(
        tok_info, precomp, bos_token_id=BOS, ignore_bos_features=True
    )
    # Three total segments
    assert len(out) == 3
    # doc 0, seg 0: tokens [BOS, 10, 11] → keep_mask (False, True, True)
    assert out[0].doc_idx == 0
    assert out[0].seg_idx == 0
    seg0_tokens = tok_info[0][0][0:3]  # [BOS, 10, 11]
    assert out[0].keep_mask == tuple(tok != BOS for tok in seg0_tokens)
    assert out[0].keep_mask == (False, True, True)
    # doc 0, seg 1: tokens [BOS, 12, 13] → keep_mask (False, True, True)
    assert out[1].doc_idx == 0
    assert out[1].seg_idx == 1
    seg1_tokens = tok_info[0][0][3:6]  # [BOS, 12, 13]
    assert out[1].keep_mask == tuple(tok != BOS for tok in seg1_tokens)
    assert out[1].keep_mask == (False, True, True)
    # doc 1, seg 0: tokens [BOS, 20, 21] → keep_mask (False, True, True)
    assert out[2].doc_idx == 1
    assert out[2].seg_idx == 0
    seg2_tokens = tok_info[1][0][0:3]  # [BOS, 20, 21]
    assert out[2].keep_mask == tuple(tok != BOS for tok in seg2_tokens)
    assert out[2].keep_mask == (False, True, True)


def test_ci_prep_multi_bos_in_segment():
    """Segment with ≥2 BOS tokens: keep_mask has False at every BOS position."""
    # Doc 0: tokens [BOS, 10, BOS, 11, BOS], one segment covering all 5 tokens
    tok_info = [([BOS, 10, BOS, 11, BOS], 5)]
    precomp = [([(0, 5)], 5, [0], [5])]
    out = _prepare_context_independent_segment_inputs(
        tok_info, precomp, bos_token_id=BOS, ignore_bos_features=True
    )
    assert len(out) == 1
    seg_tokens = tok_info[0][0][0:5]  # [BOS, 10, BOS, 11, BOS]
    expected_mask = tuple(tok != BOS for tok in seg_tokens)
    # expected: (False, True, False, True, False)
    assert expected_mask == (False, True, False, True, False)
    assert out[0].keep_mask == expected_mask
