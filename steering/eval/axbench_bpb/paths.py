"""Output-side path resolvers for axbench-bpb.

Every resolver is anchored at ``OutputPaths.output_root`` — never writes
under ``source_run_dir`` (the read-only axbench-base output that provides
inputs for this harness).

The ``run_tag`` is sanitized via :func:`steering.eval.axbench_base.paths._sanitize`,
which deterministically maps path separators / traversal markers (``/``,
``\\``, ``..``) to a hashed ``traversal_<sha256[:16]>`` form so that a
malicious or malformed tag cannot escape ``output_root``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from steering.eval.axbench_base.paths import _sanitize
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.paths import format_reg_lambda
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
    coerce_precomputed_bag_family,
)


@dataclass(frozen=True)
class OutputPaths:
    """Resolve every path this run writes to.

    Read-only reads may also happen (e.g., the cache manifest).

    Attributes
    ----------
    output_root:
        Root directory into which all run outputs are written. A
        sub-directory named after the sanitized ``run_tag`` is created
        beneath this root.
    run_tag:
        Identifier for this run; sanitized before use as a directory
        name.
    """

    output_root: Path
    run_tag: str

    def run_output_root(self) -> Path:
        return self.output_root / _sanitize(self.run_tag)

    # --- Context cache ---
    def context_cache_dir(self) -> Path:
        return self.run_output_root() / "context_cache"

    def context_cache_file(self, prefix_hash: str) -> Path:
        return self.context_cache_dir() / prefix_hash[:2] / f"{prefix_hash}.safetensors"

    def context_cache_manifest(self) -> Path:
        return self.context_cache_dir() / "manifest.json"

    # --- BPB scores ---
    def bpb_scores_dir(self) -> Path:
        return self.run_output_root() / "bpb_scores"

    def bpb_scores_rank_dir(self, rank: int) -> Path:
        return self.bpb_scores_dir() / f"rank_{rank:03d}"

    # --- BPB cache (SQLite) ---
    def bpb_cache_dir(self) -> Path:
        return self.run_output_root() / "bpb_cache"

    def bpb_cache_path(self, rank: int) -> Path:
        return self.bpb_cache_dir() / f"rank_{rank:03d}.sqlite"

    # --- Aggregation ---
    def aggregation_dir(self) -> Path:
        return self.run_output_root() / "aggregation"

    def leaderboard_csv(self) -> Path:
        return self.aggregation_dir() / "leaderboard.csv"

    def report_yaml(self) -> Path:
        return self.aggregation_dir() / "report.yaml"

    # --- Precomputed feature-conditioning bags ---
    def ours_gcg_bags_dir(self) -> Path:
        return self.run_output_root() / "ours_gcg_bags"

    def ours_gcg_bag(
        self,
        concept_id: int,
        mode: SteeringMode | str,
        reg_lambda: float,
        family: PrecomputedBagFamily | str,
        prefix_id: str | None = None,
    ) -> Path:
        mode_value = SteeringMode(mode).value
        family_slug = coerce_precomputed_bag_family(family).value
        lam_slug = _sanitize(format_reg_lambda(reg_lambda))
        parent = (
            self.ours_gcg_bags_dir()
            / _sanitize(family_slug)
            / _sanitize(mode_value)
            / f"c{int(concept_id)}"
        )
        if prefix_id is not None:
            parent = parent / f"p{_sanitize(str(prefix_id))}"
        return parent / f"lambda_{lam_slug}.safetensors"
