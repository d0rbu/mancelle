"""Tests that shard CSR format is valid when some docs have zero segments."""

import torch as t


def test_csr_valid_with_zero_segment_docs():
    """Simulate the CSR accumulation logic with mixed zero/nonzero segment docs."""
    # Simulate 3 docs: doc0 has 2 segments, doc1 has 0 segments, doc2 has 1 segment
    all_feature_ids: list[t.Tensor] = []
    all_feature_mags: list[t.Tensor] = []
    doc_token_counts: list[int] = []
    doc_segment_info: list[tuple[int, int, list[int], list[int], list[int]]] = []

    doc_counter = 0

    # Doc 0: 2 segments with features
    for _ in range(2):
        all_feature_ids.append(t.tensor([1, 2, 3], dtype=t.int32))
        all_feature_mags.append(t.tensor([0.5, 0.3, 0.1], dtype=t.float32))
    doc_segment_info.append((0, 2, [0, 128], [0, 100], [100, 200]))
    doc_token_counts.append(256)
    doc_counter += 1

    # Doc 1: 0 segments (fully truncated SFT response)
    doc_segment_info.append((1, 0, [], [], []))
    doc_token_counts.append(300)
    doc_counter += 1

    # Doc 2: 1 segment
    all_feature_ids.append(t.tensor([5, 6], dtype=t.int32))
    all_feature_mags.append(t.tensor([0.8, 0.2], dtype=t.float32))
    doc_segment_info.append((2, 1, [0], [0], [150]))
    doc_token_counts.append(150)
    doc_counter += 1

    # Replicate the CSR accumulation logic from extract_features.py
    feat_counts = t.tensor(
        [len(fids) for fids in all_feature_ids],
        dtype=t.int64,
    )
    offsets = t.cat([t.zeros(1, dtype=t.int64), feat_counts.cumsum(0)])
    n_total_segments = len(all_feature_ids)

    doc_ids_list: list[int] = []
    seg_indices_list: list[int] = []
    seg_positions_list: list[int] = []
    seg_char_starts_list: list[int] = []
    seg_char_ends_list: list[int] = []
    for doc_id, n_segs, positions, cstarts, cends in doc_segment_info:
        doc_ids_list.extend([doc_id] * n_segs)
        seg_indices_list.extend(range(n_segs))
        seg_positions_list.extend(positions)
        seg_char_starts_list.extend(cstarts)
        seg_char_ends_list.extend(cends)

    # Validate CSR invariants
    assert offsets[0] == 0
    assert offsets[-1] == sum(len(fids) for fids in all_feature_ids)
    assert len(offsets) == n_total_segments + 1  # 3 segments + 1
    assert n_total_segments == 3  # 2 from doc0 + 0 from doc1 + 1 from doc2
    assert doc_counter == 3  # 3 docs tracked
    assert len(doc_token_counts) == 3
    assert len(doc_ids_list) == 3  # only 3 segment entries (doc1 contributes 0)
    assert doc_ids_list == [0, 0, 2]  # doc1 absent from segment metadata
    assert len(seg_char_starts_list) == 3
    assert len(seg_char_ends_list) == 3


def test_csr_valid_all_zero_segment_docs():
    """Edge case: ALL docs have zero segments (entire shard of truncated SFT)."""
    all_feature_ids: list[t.Tensor] = []
    all_feature_mags: list[t.Tensor] = []

    feat_counts = t.tensor(
        [len(fids) for fids in all_feature_ids],
        dtype=t.int64,
    )
    offsets = t.cat([t.zeros(1, dtype=t.int64), feat_counts.cumsum(0)])
    n_total_segments = len(all_feature_ids)

    # With no segments, offsets = [0], n_total_segments = 0
    assert offsets.tolist() == [0]
    assert n_total_segments == 0
    # The cat guard: `t.cat(all_feature_ids) if all_feature_ids else t.tensor([])`
    # handles the empty case
    feature_ids = (
        t.cat(all_feature_ids) if all_feature_ids else t.tensor([], dtype=t.int32)
    )
    feature_mags = (
        t.cat(all_feature_mags) if all_feature_mags else t.tensor([], dtype=t.float32)
    )
    assert len(feature_ids) == 0
    assert len(feature_mags) == 0
    # offsets[-1] == len(feature_ids)
    assert offsets[-1] == len(feature_ids)
    # len(offsets) == n_total_segments + 1
    assert len(offsets) == n_total_segments + 1
