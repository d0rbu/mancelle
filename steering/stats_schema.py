"""Typed histogram schema for segment-level SAE statistics.

Defines the histogram data contract used by ``compute_statistics()``.

Run the companion tests with:
    uv run pytest tests/test_stats_schema.py -v
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistogramData:
    """A pre-binned histogram ready for plotting.

    Parameters
    ----------
    bin_edges : list[float]
        Monotonically non-decreasing edge values.  Length is
        ``len(counts) + 1`` (one more than the number of bins).
    counts : list[int]
        Non-negative count for each bin.
    """

    bin_edges: list[float]
    counts: list[int]

    def validate(self) -> None:
        """Raise ``ValueError`` if the histogram is malformed."""
        if len(self.bin_edges) != len(self.counts) + 1:
            raise ValueError(
                f"len(bin_edges) must be len(counts) + 1, "
                f"got {len(self.bin_edges)} and {len(self.counts)}"
            )
        for i in range(1, len(self.bin_edges)):
            if self.bin_edges[i] < self.bin_edges[i - 1]:
                raise ValueError(
                    f"bin_edges must be monotonically non-decreasing, "
                    f"but bin_edges[{i}]={self.bin_edges[i]} < "
                    f"bin_edges[{i - 1}]={self.bin_edges[i - 1]}"
                )
        for idx, c in enumerate(self.counts):
            if c < 0:
                raise ValueError(f"counts must be non-negative, but counts[{idx}]={c}")
