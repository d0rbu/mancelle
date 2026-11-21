"""Explicit output-path contract for AxBench-Base artifacts.

All output paths are derived from one immutable :class:`AxBenchBasePaths`
instance. A null configured root preserves the historical location beneath
``get_base_dir() / "eval" / "axbench_base"``; a configured root is honored
literally. Run-tag sanitization prevents directory escape.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from nanochat.common import get_base_dir
from steering.eval.axbench_base.types import AxBenchBaseError, SteeringMode

if TYPE_CHECKING:
    from steering.eval.axbench_base.config import AxBenchBaseConfig

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_\-.]")
_NEURONPEDIA_CACHE_SCHEMA = "v1"


def _sanitize(run_tag: str) -> str:
    """Strip path separators and unsafe characters from a run tag.

    - If the input contains path separators or traversal markers (``/``,
      ``\\``, or ``..``), it is considered suspicious: the returned name
      is a deterministic hash-based fallback that cannot collide with the
      original dangerous substrings (e.g. ``passwd``, ``etc``).
    - Otherwise, any character outside ``[A-Za-z0-9_\\-.]`` is replaced
      with ``_``.
    - Dot-only names (``"."``, ``".."`` variants, ``"..."`` etc.) would
      collapse to the parent directory on filesystem operations, so they
      are bounced to the ``_`` fallback.
    - Falls back to ``_`` if the result is empty.
    """
    if "/" in run_tag or "\\" in run_tag or ".." in run_tag:
        digest = hashlib.sha256(run_tag.encode("utf-8")).hexdigest()[:16]
        return f"traversal_{digest}"
    safe = _SAFE_NAME.sub("_", run_tag)
    if not safe.lstrip("."):
        return "_"
    return safe or "_"


def default_output_root() -> Path:
    """Return the exact historical AxBench-Base artifact parent."""
    return Path(get_base_dir()) / "eval" / "axbench_base"


def effective_output_root(configured: str | None) -> Path:
    """Resolve the configured root without changing relative-path semantics."""
    if configured is None:
        return default_output_root()
    if not isinstance(configured, str) or not configured.strip():
        raise AxBenchBaseError(
            f"output_root must be null or a non-empty string, got {configured!r}"
        )
    return Path(configured)


def _neuronpedia_cache_identity(model_id: str, sae_source: str) -> str:
    """Return a stable full digest for the exact Neuronpedia identity."""
    payload = json.dumps(
        {"model_id": model_id, "sae_source": sae_source},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AxBenchBasePaths:
    """Resolve every output path for one AxBench-Base invocation."""

    output_root: Path
    run_tag: str

    @classmethod
    def from_config(cls, cfg: AxBenchBaseConfig) -> Self:
        """Construct paths from a config after its run tag has been resolved."""
        if cfg.run_tag is None:
            raise AxBenchBaseError("cfg.run_tag must be set before resolving AxBench-Base paths")
        return cls(
            output_root=effective_output_root(cfg.output_root),
            run_tag=cfg.run_tag,
        )

    def run_output_root(self) -> Path:
        return self.output_root / _sanitize(self.run_tag)

    def concepts_json(self) -> Path:
        return self.run_output_root() / "concepts.json"

    def candidate_concepts_json(self) -> Path:
        return self.run_output_root() / "candidate_concepts.json"

    def feature_contract_json(self) -> Path:
        """Cluster-only root commit marker for a source run."""
        return self.run_output_root() / "feature_contract.json"

    def candidate_concepts_cluster_provenance_json(self) -> Path:
        """Cluster-only Stage 0a provenance commit marker."""
        return self.run_output_root() / "candidate_concepts.cluster_provenance.json"

    def rejected_concepts_json(self) -> Path:
        return self.run_output_root() / "rejected_concepts.json"

    def concept_examples_qc_report(self) -> Path:
        return self.run_output_root() / "concept_examples_qc_report.json"

    def seeds_train(self) -> Path:
        return self.run_output_root() / "seeds_train.safetensors"

    def seeds_eval(self) -> Path:
        return self.run_output_root() / "seeds_eval.safetensors"

    def seed_pool_manifest(self) -> Path:
        """Sole commit marker for the schema-v2 train/eval seed pair."""
        return self.run_output_root() / "seed_pool_manifest.json"

    def seed_extraction_root(self) -> Path:
        """Resumable Stage 0b candidate and feature-shard workspace."""
        return self.run_output_root() / "_seed_extraction_v2"

    def profile_dir(self) -> Path:
        """Default profiler artifact directory for this run."""
        return self.run_output_root() / "profile"

    def stage4_contexts_jsonl(self) -> Path:
        return self.run_output_root() / "stage4_contexts.jsonl"

    def stage4_contexts_parquet(self) -> Path:
        return self.run_output_root() / "stage4_contexts_extraction" / "shard_00000.parquet"

    def stage4_contexts_manifest(self) -> Path:
        return self.run_output_root() / "stage4_contexts_manifest.json"

    def negative_mode_prefixes(self) -> Path:
        return self.run_output_root() / "negative_mode_prefixes.parquet"

    def stage0c_partial_rank_json(self, rank: int) -> Path:
        """Per-rank Stage 0c JSON checkpoint."""
        return self.run_output_root() / f"stage0c_partial_rank_{rank}.json"

    def stage0c_partial_rank_parquet(self, rank: int) -> Path:
        """Per-rank Stage 0c verified-prefix checkpoint."""
        return self.run_output_root() / f"stage0c_partial_rank_{rank}.parquet"

    def synth_training_positive(self) -> Path:
        return self.run_output_root() / "synth_training_positive.parquet"

    def stage1b_partial_parquet(self) -> Path:
        """Stage 1b durable complete-pair rows (JSON is authoritative)."""
        return self.run_output_root() / "stage1b_partial.parquet"

    def stage1b_partial_json(self) -> Path:
        """Stage 1b checkpoint commit marker."""
        return self.run_output_root() / "stage1b_partial.json"

    def stage1b_concept_yield_report(self) -> Path:
        return self.run_output_root() / "stage1b_concept_yield.json"

    def stage1b_concepts(self) -> Path:
        return self.run_output_root() / "concepts_stage1b.json"

    def synth_training_negative(self) -> Path:
        return self.run_output_root() / "synth_training_negative.parquet"

    def stage1c_concept_yield_report(self) -> Path:
        return self.run_output_root() / "stage1c_concept_yield.json"

    def stage1c_concepts(self) -> Path:
        return self.run_output_root() / "concepts_stage1c.json"

    def downstream_concepts(self, *, run_negative_mode: bool) -> Path:
        """Authoritative post-synthesis concept population for later stages."""

        if run_negative_mode:
            return self.stage1c_concepts()
        return self.stage1b_concepts()

    def synth_training(self, mode: SteeringMode) -> Path:
        """Synthetic training corpus for ``mode``."""
        if mode is SteeringMode.POSITIVE:
            return self.synth_training_positive()
        return self.synth_training_negative()

    def concept_features(self) -> Path:
        return self.run_output_root() / "concept_features.safetensors"

    def concept_features_neg(self) -> Path:
        return self.run_output_root() / "concept_features_neg.safetensors"

    def generations_dir(self) -> Path:
        return self.run_output_root() / "generations"

    def generations_feature_contract_json(self) -> Path:
        """Cluster-only Stage 4 generation/resume commit marker."""
        return self.generations_dir() / "feature_contract.json"

    def judge_cache(self) -> Path:
        return self.run_output_root() / "judge_cache.sqlite"

    def synth_response_cache(self) -> Path:
        return self.run_output_root() / "synth_response_cache.sqlite"

    def judge_scores_dir(self) -> Path:
        """Directory-dataset root for incremental Stage 5 score batches."""
        return self.run_output_root() / "judge_scores"

    def report_yaml(self) -> Path:
        return self.run_output_root() / "report.yaml"

    def leaderboard_csv(self) -> Path:
        return self.run_output_root() / "leaderboard.csv"

    def aggregation_report(self) -> Path:
        return self.run_output_root() / "aggregation_report.yaml"

    def diffmean_direction(
        self,
        target_model: str,
        mode: SteeringMode,
        concept_id: int,
    ) -> Path:
        return (
            self.run_output_root()
            / "baselines"
            / "diffmean"
            / target_model
            / mode
            / f"concept_{concept_id:04d}.safetensors"
        )

    def linear_probe_direction(
        self,
        target_model: str,
        mode: SteeringMode,
        concept_id: int,
    ) -> Path:
        return (
            self.run_output_root()
            / "baselines"
            / "linear_probe"
            / target_model
            / mode
            / f"concept_{concept_id:04d}.safetensors"
        )

    def lora_adapter(self, mode: SteeringMode, concept_id: int) -> Path:
        return (
            self.run_output_root()
            / "baselines"
            / "lora"
            / "decoder_only"
            / mode
            / f"concept_{concept_id:04d}.pt"
        )

    def prompting_descriptions(self, mode: SteeringMode) -> Path:
        return self.run_output_root() / "baselines" / "prompting" / f"{mode}_descriptions.json"

    def neuronpedia_cache(self, *, model_id: str, sae_source: str) -> Path:
        identity = _neuronpedia_cache_identity(model_id, sae_source)
        return (
            self.output_root / "_neuronpedia_cache" / _NEURONPEDIA_CACHE_SCHEMA / f"{identity}.yaml"
        )


def assert_no_legacy_output_shadow(cfg: AxBenchBaseConfig) -> None:
    """Reject ambiguous resume when a former custom-root run lives elsewhere.

    Before #1001, custom ``output_root`` values were ignored for run artifacts.
    Reusing that config after the fix would otherwise start a second tree while
    the expensive historical run remains beneath the default root.
    """
    if cfg.output_root is None:
        return
    output_paths = AxBenchBasePaths.from_config(cfg)
    configured_run = output_paths.run_output_root()
    legacy_run = default_output_root() / _sanitize(output_paths.run_tag)
    if configured_run.resolve(strict=False) == legacy_run.resolve(strict=False):
        return
    if configured_run.exists() or not legacy_run.exists():
        return
    raise AxBenchBaseError(
        "Refusing to start a second AxBench-Base artifact tree: custom "
        f"output_root resolves this run to {configured_run}, but a historical "
        f"run with the same run_tag already exists at {legacy_run}. Before "
        "#1001, custom output_root values were ignored. To resume in place, "
        "set output_root=null and keep the explicit run_tag; otherwise copy "
        "and validate the complete run directory or choose a fresh run_tag."
    )


PROMPT_DIR = Path(__file__).parent / "prompts"
