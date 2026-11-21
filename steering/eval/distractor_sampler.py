"""Distractor feature sampling for robustness evaluation.

Samples random "distractor" SAE features (weighted by activation frequency)
to activate alongside a target feature during steering evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file


@dataclass(frozen=True)
class DistractorSet:
    """A set of distractor features with their magnitudes."""

    feature_ids: tuple[int, ...]
    magnitudes: tuple[int, ...]

    def to_feature_conditions(self) -> dict[int, int]:
        """Convert to a feature_conditions dict for the inference engine."""
        return dict(zip(self.feature_ids, self.magnitudes, strict=True))


class DistractorSampler:
    """Samples distractor features weighted by activation frequency.

    Loads feature activation counts from a safetensors statistics file and
    builds a sampling pool excluding blacklisted features and zero-count
    features. Each ``sample()`` call is stateless (creates a fresh RNG from
    the given seed), so resume works automatically.

    Parameters
    ----------
    stats_path:
        Path to the safetensors statistics file containing ``feature_ids``
        and ``counts`` tensors.
    magnitude_levels:
        Possible magnitude values to assign to each distractor.
    excluded_feature_ids:
        Feature IDs to exclude from the distractor pool (e.g. all eval
        target features).
    """

    def __init__(
        self,
        stats_path: Path,
        magnitude_levels: list[int],
        excluded_feature_ids: set[int],
    ) -> None:
        self._magnitude_levels = magnitude_levels

        tensors = load_file(str(stats_path))
        feature_ids = tensors["feature_ids"].tolist()
        counts = tensors["counts"].tolist()

        self._pool_ids: list[int] = []
        self._pool_weights: list[float] = []

        for fid, count in zip(feature_ids, counts, strict=True):
            if fid in excluded_feature_ids:
                continue
            if count <= 0:
                continue
            self._pool_ids.append(int(fid))
            self._pool_weights.append(float(count))

        if not self._pool_ids:
            raise ValueError(
                "Distractor pool is empty after excluding blacklisted and "
                "zero-count features"
            )

    @property
    def pool_size(self) -> int:
        """Number of features available for sampling."""
        return len(self._pool_ids)

    def sample(self, n: int, sample_seed: int) -> DistractorSet:
        """Sample n distractor features with weighted sampling (no replacement).

        Parameters
        ----------
        n:
            Number of distractors to sample.
        sample_seed:
            Seed for this specific sample — ensures determinism and
            reproducibility.

        Returns
        -------
        A ``DistractorSet`` with exactly ``n`` features.

        Raises
        ------
        ValueError
            If ``n`` exceeds the pool size.
        """
        if n > self.pool_size:
            raise ValueError(
                f"Requested {n} distractors but pool only has {self.pool_size} features"
            )
        if n == 0:
            return DistractorSet(feature_ids=(), magnitudes=())

        rng = random.Random(sample_seed)

        # Weighted sampling without replacement via iterative removal
        ids_remaining = list(self._pool_ids)
        weights_remaining = list(self._pool_weights)
        selected_ids: list[int] = []

        for _ in range(n):
            # random.choices returns a list; we pick one at a time
            chosen = rng.choices(ids_remaining, weights=weights_remaining, k=1)[0]
            idx = ids_remaining.index(chosen)
            selected_ids.append(chosen)
            ids_remaining.pop(idx)
            weights_remaining.pop(idx)

        magnitudes = tuple(rng.choice(self._magnitude_levels) for _ in range(n))

        return DistractorSet(
            feature_ids=tuple(selected_ids),
            magnitudes=magnitudes,
        )
