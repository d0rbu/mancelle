"""Read-only path resolvers anchored at cfg.source_run_dir.

This module intentionally offers NO write helpers. Every function
returns a path you are expected to open in read mode. The axbench-bpb
contract is that nothing in the source run directory is ever mutated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.trainable_concepts import (
    require_complete_concept_population_from_corpus,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_base.types import SteeringMode as BaseSteeringMode
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    InterventionKind,
    TargetModel,
)


@dataclass(frozen=True)
class SourcePaths:
    source_run_dir: Path

    def concepts_json(self) -> Path:
        """Final post-Stage-1c concept authority (method name kept for API stability)."""

        return self.source_run_dir / "concepts_stage1c.json"

    def synth_training_positive(self) -> Path:
        return self.source_run_dir / "synth_training_positive.parquet"

    def synth_training_negative(self) -> Path:
        return self.source_run_dir / "synth_training_negative.parquet"

    def seeds_train(self) -> Path:
        return self.source_run_dir / "seeds_train.safetensors"

    def seeds_eval(self) -> Path:
        return self.source_run_dir / "seeds_eval.safetensors"

    def seed_pool_manifest(self) -> Path:
        return self.source_run_dir / "seed_pool_manifest.json"

    def feature_contract(self) -> Path:
        return self.source_run_dir / "feature_contract.json"

    def candidate_concepts(self) -> Path:
        return self.source_run_dir / "candidate_concepts.json"

    def candidate_cluster_provenance(self) -> Path:
        return self.source_run_dir / "candidate_concepts.cluster_provenance.json"

    def negative_mode_prefixes(self) -> Path:
        return self.source_run_dir / "negative_mode_prefixes.parquet"

    def concept_features(self) -> Path:
        return self.source_run_dir / "concept_features.safetensors"

    def concept_features_neg(self) -> Path:
        return self.source_run_dir / "concept_features_neg.safetensors"

    def baselines_root(self) -> Path:
        return self.source_run_dir / "baselines"

    def diffmean_direction(self, target: str, mode: str, concept_id: int) -> Path:
        return (
            self.baselines_root()
            / InterventionKind.DIFFMEAN.value
            / target
            / mode
            / f"concept_{concept_id:04d}.safetensors"
        )

    def linear_probe_direction(self, target: str, mode: str, concept_id: int) -> Path:
        return (
            self.baselines_root()
            / InterventionKind.LINEAR_PROBE.value
            / target
            / mode
            / f"concept_{concept_id:04d}.safetensors"
        )

    def lora_adapter(self, mode: str, concept_id: int) -> Path:
        # LoRA only trains on decoder_only per axbench-base convention
        return (
            self.baselines_root()
            / InterventionKind.LORA.value
            / TargetModel.DECODER_ONLY.value
            / mode
            / f"concept_{concept_id:04d}.pt"
        )

    def prompting_descriptions(self, mode: str) -> Path:
        return (
            self.baselines_root() / InterventionKind.PROMPTING.value / f"{mode}_descriptions.json"
        )


def _load_final_source_concept_ids(path: Path) -> list[int]:
    """Load the final population identity needed by the source preflight."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbError(f"failed to load final source concepts from {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise AxBenchBpbError(f"final source concepts at {path} must be a non-empty JSON array")
    concept_ids: list[int] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AxBenchBpbError(f"final source concepts[{index}] at {path} must be an object")
        item_dict = cast(dict[str, Any], item)
        concept_id = item_dict.get("concept_id")
        if isinstance(concept_id, bool) or not isinstance(concept_id, int) or concept_id < 0:
            raise AxBenchBpbError(
                f"final source concepts[{index}] at {path} has invalid concept_id={concept_id!r}"
            )
        concept_ids.append(concept_id)
    if concept_ids != sorted(set(concept_ids)):
        raise AxBenchBpbError(
            f"final source concepts at {path} must preserve unique, increasing original concept IDs"
        )
    return concept_ids


def validate_stage_0_3_artifacts(
    sp: SourcePaths,
    *,
    filter_unaccepted_samples: bool = True,
    validate_population: bool = True,
) -> None:
    """Raise AxBenchBpbError naming every Stage 0-3 artifact missing from
    the source run. Called at CLI startup before heavy resource loads."""
    required = [
        ("concepts_stage1c.json", sp.concepts_json()),
        ("synth_training_positive.parquet", sp.synth_training_positive()),
        ("synth_training_negative.parquet", sp.synth_training_negative()),
        ("seeds_train.safetensors", sp.seeds_train()),
        ("seeds_eval.safetensors", sp.seeds_eval()),
        ("seed_pool_manifest.json", sp.seed_pool_manifest()),
        ("concept_features.safetensors", sp.concept_features()),
        ("concept_features_neg.safetensors", sp.concept_features_neg()),
        ("baselines/", sp.baselines_root()),
    ]
    missing = [name for name, path in required if not path.exists()]
    if missing:
        raise AxBenchBpbError(
            f"Source run at {sp.source_run_dir} is missing required Stage 0-3 "
            f"artifacts: {', '.join(missing)}"
        )
    try:
        concept_ids = _load_final_source_concept_ids(sp.concepts_json())
        validate_positive_qc_contract(
            sp.synth_training_positive(),
            consumer="AxBench-BPB source preflight",
        )
        validate_positive_selection_contract(
            sp.synth_training_positive(),
            consumer="AxBench-BPB source preflight",
        )
        validate_stage1_continuation_contract(
            sp.synth_training_positive(),
            consumer="AxBench-BPB source preflight",
            expected_role="stage1b_positive",
        )
        validate_stage1_continuation_contract(
            sp.synth_training_negative(),
            consumer="AxBench-BPB source preflight",
            expected_role="stage1c_negative",
        )
        if validate_population:
            for mode, corpus_path in (
                (BaseSteeringMode.POSITIVE, sp.synth_training_positive()),
                (BaseSteeringMode.NEGATIVE, sp.synth_training_negative()),
            ):
                require_complete_concept_population_from_corpus(
                    corpus_path,
                    concept_ids,
                    mode=mode,
                    consumer="AxBench-BPB source preflight",
                    filter_unaccepted=filter_unaccepted_samples,
                )
    except AxBenchBaseError as exc:
        raise AxBenchBpbError(str(exc)) from exc
