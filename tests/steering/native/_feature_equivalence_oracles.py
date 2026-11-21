"""Test-only feature-equivalence acceptance and measurement oracles."""

from __future__ import annotations

import torch as t

from steering.feature_equivalence import _seg_jaccard_and_count_rdiff


def _seg_common_mag_max_abs_and_rel(
    aid: t.Tensor, amg: t.Tensor, bid: t.Tensor, bmg: t.Tensor
) -> tuple[float, float]:
    """Return worst absolute and relative drift over common feature IDs."""
    common = sorted(set(aid.tolist()) & set(bid.tolist()))
    if not common:
        return 0.0, 0.0
    ct = t.tensor(common, dtype=t.long)
    a_mask = t.isin(aid, ct)
    b_mask = t.isin(bid, ct)
    a_sel = amg[a_mask][aid[a_mask].argsort()].float()
    b_sel = bmg[b_mask][bid[b_mask].argsort()].float()
    abs_diff = (a_sel - b_sel).abs()
    max_abs = abs_diff.max().item()
    nz = b_sel.abs() > 0
    max_rel = (abs_diff[nz] / b_sel[nz].abs()).max().item() if nz.any() else 0.0
    return max_abs, max_rel


def feature_equivalence_passed(
    a_ids: list[t.Tensor],
    a_mags: list[t.Tensor],
    b_ids: list[t.Tensor],
    b_mags: list[t.Tensor],
    *,
    mag_rtol: float,
    mag_atol: float,
    jaccard_threshold: float,
    count_rtol: float,
) -> bool:
    """Return whether every segment passes the test parity envelope."""
    if not (len(a_ids) == len(b_ids) == len(a_mags) == len(b_mags)):
        return False
    for aid, amg, bid, bmg in zip(a_ids, a_mags, b_ids, b_mags, strict=True):
        jaccard, count_rdiff = _seg_jaccard_and_count_rdiff(aid, bid)
        if count_rdiff > count_rtol or jaccard < jaccard_threshold:
            return False
        common = sorted(set(aid.tolist()) & set(bid.tolist()))
        if common:
            ct = t.tensor(common, dtype=t.long)
            a_mask = t.isin(aid, ct)
            b_mask = t.isin(bid, ct)
            a_sel = amg[a_mask][aid[a_mask].argsort()]
            b_sel = bmg[b_mask][bid[b_mask].argsort()]
            if not t.allclose(a_sel.float(), b_sel.float(), rtol=mag_rtol, atol=mag_atol):
                return False
    return True


def feature_magnitude_metrics(
    a_ids: list[t.Tensor],
    a_mags: list[t.Tensor],
    b_ids: list[t.Tensor],
    b_mags: list[t.Tensor],
) -> tuple[float, float]:
    """Return the worst common-feature absolute and relative deviations."""
    assert len(a_ids) == len(b_ids) == len(a_mags) == len(b_mags), (
        f"feature_magnitude_metrics: segment count mismatch "
        f"({len(a_ids)},{len(a_mags)}) vs ({len(b_ids)},{len(b_mags)})"
    )
    max_abs = 0.0
    max_rel = 0.0
    for aid, amg, bid, bmg in zip(a_ids, a_mags, b_ids, b_mags, strict=True):
        seg_abs, seg_rel = _seg_common_mag_max_abs_and_rel(aid, amg, bid, bmg)
        max_abs = max(max_abs, seg_abs)
        max_rel = max(max_rel, seg_rel)
    return max_abs, max_rel
