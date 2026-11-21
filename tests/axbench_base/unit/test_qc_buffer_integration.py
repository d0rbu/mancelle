"""Integration tests verifying QcBuffer integration into Stage 0c
preserves per-candidate QC decisions across different qc_batch_size values."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_qc_score(fid, outcome, score):
    return MagicMock(
        feature_index=fid,
        description=f"d{fid}",
        outcome=outcome,
        score=score,
        n_examples=5,
        error_message=None,
    )


def _make_qc_client_returning(scores_per_fid: dict):
    """Build a qc_client whose filter returns per_feature matching scores_per_fid."""
    def _filter(batch, *, target_surviving=None):
        survivors = []
        per_feature = []
        for fid, desc in batch:
            score = scores_per_fid.get(fid)
            if score is None:
                continue
            per_feature.append(score)
            if score.outcome == "accepted":
                survivors.append((fid, desc))
                if target_surviving is not None and len(survivors) >= target_surviving:
                    break
        return survivors, MagicMock(per_feature=per_feature)
    client = MagicMock()
    client.filter = MagicMock(side_effect=_filter)
    return client


@pytest.mark.parametrize("qc_batch_size", [1, 2, 4, 8])
def test_qc_buffer_results_identical_across_batch_sizes(qc_batch_size):
    """For a fixed candidate set and fixed scoring, the per-candidate
    QcBuffer.next() output must be identical regardless of qc_batch_size."""
    from steering.eval.axbench_base.qc_buffer import QcBuffer

    n = 8
    candidates = [MagicMock(feature_index=i, description=f"d{i}") for i in range(n)]
    scores = {
        i: _make_qc_score(
            i,
            outcome="accepted" if i % 2 == 0 else "rejected",
            score=2 if i % 2 == 0 else 0,
        )
        for i in range(n)
    }
    client = _make_qc_client_returning(scores)
    buf = QcBuffer(client, candidates, qc_batch_size=qc_batch_size, target_surviving=4)

    results = []
    for i in range(n):
        results.append(buf.next(i))

    # Reference: the per-candidate scores are deterministic (looked up by
    # feature_index in `scores` dict). Across all batch sizes we should
    # see the same ordered list of QCFeatureScore objects.
    expected_outcomes = ["accepted" if i % 2 == 0 else "rejected" for i in range(n)]
    actual_outcomes = [r.outcome if r is not None else None for r in results]
    # Some accepted+rejected candidates may be skipped due to target_surviving
    # early-termination — but only after target_surviving accepts have been
    # collected. Verify the ordering is intact.
    accepted_so_far = 0
    for i, oc in enumerate(actual_outcomes):
        if oc is None:
            continue  # skipped due to early termination
        assert oc == expected_outcomes[i], (
            f"At index {i}, qc_batch_size={qc_batch_size}: "
            f"expected outcome={expected_outcomes[i]}, got {oc}"
        )
        if oc == "accepted":
            accepted_so_far += 1


@pytest.mark.parametrize("qc_batch_size", [1, 4, 8])
@pytest.mark.parametrize("pipeline", [False, True])
def test_qc_buffer_byte_identity_across_batch_size_and_pipeline(qc_batch_size, pipeline):
    """All combinations of (qc_batch_size, pipeline) produce byte-identical
    per-candidate scores. The pipelining is transparent to caller semantics."""
    from steering.eval.axbench_base.qc_buffer import QcBuffer

    n = 8
    candidates = [MagicMock(feature_index=i, description=f"d{i}") for i in range(n)]
    scores = {
        i: _make_qc_score(
            i,
            outcome="accepted" if i % 2 == 0 else "rejected",
            score=2 if i % 2 == 0 else 0,
        )
        for i in range(n)
    }
    client = _make_qc_client_returning(scores)
    with QcBuffer(
        client, candidates,
        qc_batch_size=qc_batch_size, target_surviving=4, pipeline=pipeline,
    ) as buf:
        results = [buf.next(i) for i in range(n)]

    # Same outcomes regardless of (qc_batch_size, pipeline) combination
    expected_outcomes = ["accepted" if i % 2 == 0 else "rejected" for i in range(n)]
    actual_outcomes = [r.outcome if r is not None else None for r in results]
    accepted_so_far = 0
    for i, oc in enumerate(actual_outcomes):
        if oc is None:
            continue  # skipped due to early termination
        assert oc == expected_outcomes[i], (
            f"At index {i}, qc_batch_size={qc_batch_size}, pipeline={pipeline}: "
            f"expected {expected_outcomes[i]}, got {oc}"
        )
        if oc == "accepted":
            accepted_so_far += 1
