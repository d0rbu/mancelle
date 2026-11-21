"""Immutable display index derived from static feature-cluster consolidation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch as t

from steering.conditional.feature_clusters import FeatureClusterConsolidation


@dataclass(frozen=True)
class ClusterView:
    """Composite representatives and their mask-allowed source members."""

    rep_to_members: Mapping[int, tuple[int, ...]]
    member_to_rep: Mapping[int, int]

    @classmethod
    def build(cls, consolidation: FeatureClusterConsolidation) -> ClusterView:
        rep_to_members: dict[int, tuple[int, ...]] = {}
        member_to_rep: dict[int, int] = {}

        for cluster in consolidation.resolved.definition.clusters:
            allowed_members = tuple(
                member for member in cluster if bool(consolidation.allowed_universe[member])
            )
            if len(allowed_members) < 2:
                continue
            representative = allowed_members[0]
            for member in allowed_members:
                observed_representative = int(consolidation.rep_of[member].item())
                if observed_representative != representative:
                    raise ValueError(
                        "ClusterView invariant violated: allowed member "
                        f"{member} maps to representative "
                        f"{observed_representative}, expected {representative}"
                    )
                member_to_rep[member] = representative
            if bool(consolidation.consolidated_away[representative]):
                raise ValueError(
                    "ClusterView invariant violated: representative "
                    f"{representative} is marked consolidated away"
                )
            rep_to_members[representative] = allowed_members

        away_ids = {
            int(feature_id)
            for feature_id in t.nonzero(consolidation.consolidated_away).squeeze(1).tolist()
        }
        for feature_id in away_ids:
            if feature_id not in member_to_rep:
                raise ValueError(
                    "ClusterView invariant violated: consolidated-away feature "
                    f"{feature_id} is absent from member_to_rep"
                )
        away_representatives = {
            int(consolidation.rep_of[feature_id].item()) for feature_id in away_ids
        }
        if set(rep_to_members) != away_representatives:
            raise ValueError(
                "ClusterView invariant violated: composite representatives "
                f"{sorted(rep_to_members)} do not match representatives of "
                f"consolidated-away members {sorted(away_representatives)}"
            )

        return cls(
            rep_to_members=MappingProxyType(rep_to_members),
            member_to_rep=MappingProxyType(member_to_rep),
        )

    def is_composite(self, feature_id: int) -> bool:
        return feature_id in self.rep_to_members

    def n_members(self, rep_id: int) -> int:
        return len(self.rep_to_members[rep_id])
