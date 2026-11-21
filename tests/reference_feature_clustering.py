"""Frozen shared primitive from the implementation removed at ``49fb3cf``.

This is intentionally test-only. It preserves the old
``FeatureConsolidation.build`` / ``consolidate_segment`` / negative-mask logic
without importing deleted discovery, Ising, coactivation, overlay, or runtime
machinery. Production tests compare the new static-file implementation against
this independent reference with exact tensor and serialized-shard equality.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as t

from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode


@dataclass(frozen=True)
class HistoricalFeatureConsolidation:
    """The valid shared consolidation subset as implemented at ``49fb3cf``."""

    rep_of: t.Tensor
    consolidated_away: t.Tensor
    augmented_allowed_universe: t.Tensor
    n_sae_features: int

    @classmethod
    def build(
        cls,
        clusters: tuple[tuple[int, ...], ...],
        mask: FeatureMask | None,
        n_sae_features: int,
    ) -> HistoricalFeatureConsolidation:
        n = n_sae_features
        rep_of = t.arange(n, dtype=t.int64)
        consolidated_away = t.zeros(n, dtype=t.bool)
        allowed = (
            mask.allowed_universe_mask(n)
            if mask is not None
            else t.ones(n, dtype=t.bool)
        )
        for cluster in clusters:
            survivors = [int(feature_id) for feature_id in cluster if bool(allowed[feature_id])]
            if not survivors:
                continue
            representative = survivors[0]
            survivor_tensor = t.tensor(survivors, dtype=t.int64)
            rep_of[survivor_tensor] = representative
            if len(survivors) > 1:
                consolidated_away[survivor_tensor[1:]] = True
        return cls(
            rep_of=rep_of,
            consolidated_away=consolidated_away,
            augmented_allowed_universe=allowed & ~consolidated_away,
            n_sae_features=n,
        )

    def consolidate_segment(
        self,
        feature_ids: t.Tensor,
        percentiles: t.Tensor,
        mask: FeatureMask | None,
    ) -> tuple[t.Tensor, t.Tensor]:
        """Verbatim old ordering and ``amax`` reduction for one segment."""
        if mask is not None and feature_ids.numel() > 0:
            keep = mask.keep_mask(feature_ids)
            feature_ids = feature_ids[keep]
            percentiles = percentiles[keep]
        if feature_ids.numel() == 0:
            return feature_ids, percentiles
        representatives = self.rep_of.to(feature_ids.device)[feature_ids.long()]
        unique_representatives, inverse = t.unique(
            representatives,
            return_inverse=True,
        )
        maxima = t.zeros(
            unique_representatives.shape[0],
            dtype=percentiles.dtype,
            device=percentiles.device,
        ).scatter_reduce(
            0,
            inverse,
            percentiles,
            reduce="amax",
            include_self=False,
        )
        return unique_representatives, maxima

    def effective_negative_mask(
        self,
        mask: FeatureMask | None,
    ) -> FeatureMask | None:
        away = t.nonzero(self.consolidated_away, as_tuple=True)[0].tolist()
        if not away:
            return mask
        away_set = {int(feature_id) for feature_id in away}
        if mask is None:
            return FeatureMask(FeatureMaskMode.blacklist, tuple(sorted(away_set)))
        if mask.mode is FeatureMaskMode.blacklist:
            return FeatureMask(
                FeatureMaskMode.blacklist,
                tuple(sorted(set(mask.ids) | away_set)),
            )
        return FeatureMask(
            FeatureMaskMode.whitelist,
            tuple(sorted(set(mask.ids) - away_set)),
        )

    def assert_consolidated(self, feature_ids: t.Tensor) -> None:
        if feature_ids.numel() == 0:
            return
        ids = feature_ids.long()
        if bool(self.consolidated_away[ids].any()):
            bad = int(ids[self.consolidated_away[ids]][0].item())
            raise AssertionError(f"historical consolidated-away feature {bad} reached selection")
        if not bool((self.rep_of[ids] == ids).all()):
            bad = int(ids[self.rep_of[ids] != ids][0].item())
            raise AssertionError(f"historical non-representative feature {bad} reached selection")


@dataclass(frozen=True)
class HistoricalPreselectionAdapter:
    """Expose the frozen primitive through the current producer's narrow seam."""

    state: HistoricalFeatureConsolidation
    mask: FeatureMask | None

    def consolidate_segment(
        self,
        feature_ids: t.Tensor,
        percentiles: t.Tensor,
        *aligned_scores: t.Tensor,
    ) -> tuple[t.Tensor, ...]:
        if aligned_scores:
            raise AssertionError("historical parity adapter covers the non-gradient shared subset")
        # The current producer has already applied the mask. Passing None here
        # isolates the consolidation primitive while retaining old exact math.
        return self.state.consolidate_segment(feature_ids, percentiles, None)

    def effective_negative_mask(self) -> FeatureMask | None:
        return self.state.effective_negative_mask(self.mask)

    def assert_representatives_only(
        self,
        feature_ids: t.Tensor,
        *,
        surface: str = "historical clustered selection",
    ) -> None:
        del surface
        self.state.assert_consolidated(feature_ids)
