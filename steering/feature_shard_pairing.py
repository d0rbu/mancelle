"""Bounded splice pairing in original SAE IDs.

Compare first segments of replacement suffixes and displaced continuations. Cut
selection and tokenizer safety remain in ``feature_shard_splice``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch as t
from numpy.typing import NDArray

from steering.conditional.percentile_table import (
    _magnitude_to_percentile,
    compute_percentile_checksum,
)

if TYPE_CHECKING:
    from steering.feature_shard_splice import SourceSegment

_PERCENTILE_LOOKUP_CHUNK_SIZE = 1_000_000
_INTEGER_DTYPES = frozenset(
    {t.int8, t.int16, t.int32, t.int64, t.uint8, t.uint16, t.uint32, t.uint64}
)


@dataclass(frozen=True)
class _SparsePercentileRows:
    ids: NDArray[np.int64]
    values: NDArray[np.float64]
    offsets: NDArray[np.int64]


def _percentile_rows(
    segments: list[SourceSegment],
    percentile_table: t.Tensor,
    *,
    table_is_finite: bool = False
) -> _SparsePercentileRows:
    """Convert one bounded group, retaining only its sparse feature values."""
    lengths: list[int] = []
    for segment in segments:
        ids, mags = segment.feature_ids, segment.feature_mags
        if ids.device.type != "cpu" or mags.device.type != "cpu":
            raise ValueError("splice pairing requires CPU feature tensors")
        if ids.ndim != 1 or mags.shape != ids.shape or ids.numel() == 0:
            raise ValueError("splice pairing requires matched, nonempty rank-1 feature tensors")
        if ids.dtype not in _INTEGER_DTYPES or not mags.is_floating_point():
            raise ValueError("splice pairing requires integer feature IDs and floating magnitudes")
        lengths.append(ids.numel())

    offsets = np.empty(len(segments) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    feature_ids = t.cat([segment.feature_ids.to(t.int64) for segment in segments])
    magnitudes = t.cat([segment.feature_mags.detach() for segment in segments])
    ids = feature_ids.numpy()
    if np.any(ids < 0) or np.any(ids >= percentile_table.shape[0]):
        raise ValueError("splice pairing feature IDs must be within the percentile table")
    ascending = ids[1:] > ids[:-1]
    ascending[offsets[1:-1] - 1] = True  # ordering applies within each row only
    if not ascending.all():
        raise ValueError("splice pairing feature IDs must be sorted and unique in each segment")
    if not t.isfinite(magnitudes).all() or not (magnitudes > 0).all():
        raise ValueError("splice pairing feature magnitudes must be positive and finite")

    # Cat can preserve a Tensor subclass even when the owned table is native.
    table_is_finite = (
        table_is_finite and type(feature_ids) is t.Tensor and type(magnitudes) is t.Tensor
    )
    values = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), _PERCENTILE_LOOKUP_CHUNK_SIZE):
        end = min(start + _PERCENTILE_LOOKUP_CHUNK_SIZE, len(ids))
        values[start:end] = _magnitude_to_percentile(
            feature_ids[start:end],
            magnitudes[start:end],
            percentile_table,
            table_is_finite=table_is_finite,
        ).numpy()
    return _SparsePercentileRows(ids, values, offsets)


def _shared_minimum_matrix(
    references: _SparsePercentileRows,
    candidates: _SparsePercentileRows,
    workspace: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return sum_f min(reference_f, candidate_f), with absent features zero.

    Only the group-sized matrix is dense. ``workspace`` is one feature-width
    vector, reused across rows and groups; all other work follows stored IDs.
    """
    overlap = np.empty((len(references.offsets) - 1, len(candidates.offsets) - 1), dtype=np.float64)
    for row, (start, end) in enumerate(
        zip(references.offsets[:-1], references.offsets[1:], strict=True)
    ):
        ids = references.ids[start:end]
        workspace[ids] = references.values[start:end]
        overlap[row] = np.add.reduceat(
            np.minimum(workspace[candidates.ids], candidates.values), candidates.offsets[:-1]
        )
        workspace[ids] = 0.0
    return overlap


@dataclass(frozen=True)
class OppositePercentilePairing:
    """Improve L1 percentile distance with one pass of bounded suffix swaps.

    Reference cut ``(doc, boundary)`` uses its replaced continuation at
    ``(doc, boundary + 1)``. Each group's candidate set stays fixed; improving
    swaps preserve the derangement. Later tokenizer repairs may reduce the gain.

    A bijection fixes total L1 masses, so maximizing L1 minimizes shared minima.
    Comparing those directly avoids cancellation for disjoint vectors. Ranks
    use canonical units (0.5–99.5). With K cuts, bound B, S features per segment,
    and vocabulary F, scoring uses O(K*B*S) work and O(B*S + B*B + F) storage,
    plus the F-by-99 table.

    The table is copied to pin its checksum, then read-only by contract.
    """

    percentile_table: t.Tensor = field(repr=False, compare=False)
    candidate_pool_size: int = 64
    percentile_checksum: str = field(init=False)
    _table_is_finite: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.candidate_pool_size) is not int or self.candidate_pool_size < 2:
            raise ValueError("pairing_candidate_pool_size must be an integer >= 2")
        table = self.percentile_table
        if (
            table.device.type != "cpu"
            or table.dtype != t.float64
            or table.ndim != 2
            or table.shape[0] == 0
            or table.shape[1] != 99
        ):
            raise ValueError(
                "splice pairing percentile table must be CPU float64 with shape [F, 99], F > 0"
            )
        if not t.isfinite(table).all() or not (table >= 0).all():
            raise ValueError("splice pairing percentile table must be finite and nonnegative")
        if not (table[:, 1:] >= table[:, :-1]).all():
            raise ValueError("splice pairing percentile table rows must be monotone")
        owned_table = table.detach().clone(memory_format=t.contiguous_format)
        object.__setattr__(self, "percentile_table", owned_table)
        object.__setattr__(self, "percentile_checksum", compute_percentile_checksum(owned_table))
        # The existing read-only contract keeps this constructor evidence valid.
        object.__setattr__(
            self, "_table_is_finite", type(table) is t.Tensor and type(owned_table) is t.Tensor
        )

    def provenance_params(self) -> dict[str, object]:
        """Semantic identity shared by the producer and resume verification."""
        return {
            "pairing_mode": "opposite_percentile",
            "pairing_candidate_pool_size": self.candidate_pool_size,
            "pairing_percentile_checksum": self.percentile_checksum,
            "pairing_algorithm": "bounded_swap_v1",
        }

    def improve_pairing(
        self,
        source_segments: list[list[SourceSegment]],
        cuts: list[tuple[int, int]],
        sigma: list[int],
    ) -> list[int]:
        """Return an improved derangement without modifying inputs or drawing RNG.

        Gains indistinguishable at float64 accumulation precision are ties:
        candidates use their current group-slot order, and zero gain keeps the
        current assignment. A final singleton group is unchanged. This bounded
        local search does not promise a globally maximal assignment.
        """
        n_cuts = len(cuts)
        if (
            len(sigma) != n_cuts
            or set(sigma) != set(range(n_cuts))
            or any(target == source for source, target in enumerate(sigma))
        ):
            raise ValueError("splice pairing requires a derangement over all cut slots")
        if len(set(cuts)) != n_cuts:
            raise ValueError("splice pairing requires distinct cuts")
        for doc, boundary in cuts:
            if not (
                0 <= doc < len(source_segments) and 0 <= boundary < len(source_segments[doc]) - 1
            ):
                raise ValueError("splice pairing cut must name a source inter-segment boundary")

        improved = list(sigma)
        workspace = np.zeros(self.percentile_table.shape[0], dtype=np.float64)
        # A final singleton cannot swap; slices clamp every other partial group.
        for start in range(0, n_cuts - 1, self.candidate_pool_size):
            stop = start + self.candidate_pool_size
            target_slots = np.asarray(sigma[start:stop], dtype=np.int64)
            reference_segments = [source_segments[d][b + 1] for d, b in cuts[start:stop]]
            candidate_segments = [source_segments[cuts[j][0]][cuts[j][1] + 1] for j in target_slots]
            references = _percentile_rows(
                reference_segments, self.percentile_table, table_is_finite=self._table_is_finite
            )
            candidates = _percentile_rows(
                candidate_segments, self.percentile_table, table_is_finite=self._table_is_finite
            )
            overlap = _shared_minimum_matrix(references, candidates, workspace)
            # Each overlap sums at most max_nnz nonnegative float64 values.
            # Bound accumulation plus the two pair sums/subtraction, scaled by
            # their actual magnitudes. This conservative allowance prevents an
            # exact zero-gain swap becoming positive through reduction rounding;
            # it is not a percentile quantization or a fixed score threshold.
            max_nnz = int(np.diff(candidates.offsets).max())
            roundoff_factor = 8 * np.finfo(np.float64).eps * (max_nnz + 1)

            rows = np.arange(len(target_slots))
            assignment = rows.copy()
            for row in range(len(target_slots)):
                target_column = assignment[row]
                current = overlap[row, target_column] + overlap[rows, assignment]
                proposed = overlap[row, assignment] + overlap[rows, target_column]
                gain = current - proposed
                roundoff = roundoff_factor * (current + proposed)
                # The row itself has exactly zero gain, so it cannot qualify.
                eligible = (
                    (target_slots[assignment] != start + row)
                    & (target_slots[target_column] != start + rows)
                    & (gain > roundoff)
                )
                if not eligible.any():
                    continue
                gain[~eligible] = -np.inf
                best = int(np.argmax(gain))
                tied_best = eligible & (gain[best] - gain <= roundoff[best] + roundoff)
                partner = int(np.argmax(tied_best))
                assignment[row], assignment[partner] = assignment[partner], assignment[row]
            improved[start:stop] = target_slots[assignment].tolist()
        return improved
