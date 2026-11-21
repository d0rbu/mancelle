"""Strict Stage 4 artifact contracts for integration tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_base import paths
from steering.eval.axbench_base import seed_contract as seed_contract_module
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.seed_contract import load_seed_pool_fingerprint
from steering.eval.axbench_base.stage4_contexts import (
    enumerate_stage4_contexts,
    load_stage4_contexts_manifest,
)
from steering.eval.axbench_base.stage4_prompts import prompt_token_key
from steering.eval.axbench_base.types import (
    Concept,
    SeedRow,
    SteeringMode,
    TargetModelMetadata,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.axbench_base.fixtures.seed_contract import write_committed_seed_pair


def commit_seed_contract(
    cfg: AxBenchBaseConfig,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    eval_rows: list[SeedRow],
) -> str:
    """Commit and validate the exact schema-v2 seed artifacts Stage 4 loads."""

    if not eval_rows:
        raise ValueError("Stage 4 test contract requires at least one eval seed")
    monkeypatch.setattr(seed_contract_module, "get_tokenizer", lambda: tokenizer)
    write_committed_seed_pair(
        cfg,
        tokenizer=tokenizer,
        train_rows=[eval_rows[0]],
        eval_rows=eval_rows,
    )
    assert cfg.run_tag is not None
    return load_seed_pool_fingerprint(paths.AxBenchBasePaths.from_config(cfg))


class Stage4TestContextProvider:
    """Exact-key provider carrying the cache metadata Stage 4 validates."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        context_keys: Sequence[str],
        contexts_fingerprint: str,
        model_max_seq_len: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._context_keys = frozenset(context_keys)
        self.manifest = {
            "metadata": {
                "model_max_seq_len": model_max_seq_len,
                "contexts_fingerprint": contexts_fingerprint,
            }
        }

    def get(
        self,
        token_ids: Sequence[int],
    ) -> tuple[ContextFeatureSegment, ...] | None:
        tokens = [int(token) for token in token_ids]
        if prompt_token_key(tokens) not in self._context_keys:
            return None
        decoded_length = len(self._tokenizer.decode(tokens))
        if decoded_length <= 0:
            raise ValueError("Stage 4 test context decoded to empty text")
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(0,),
                char_start=0,
                char_end=decoded_length,
            ),
        )


def commit_stage4_context_contract(
    cfg: AxBenchBaseConfig,
    *,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    encdec_meta: TargetModelMetadata,
    encdec_sequence_len: int,
) -> tuple[str, Stage4TestContextProvider]:
    """Commit seeds plus a production-enumerated Stage 3e context manifest."""

    seed_fingerprint = commit_seed_contract(
        cfg,
        monkeypatch=monkeypatch,
        tokenizer=tokenizer,
        eval_rows=[*eval_seeds_positive, *eval_seeds_negative],
    )
    assert cfg.run_tag is not None
    descriptions = {str(concept.concept_id): concept.description for concept in concepts}
    for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
        description_path = paths.AxBenchBasePaths.from_config(cfg).prompting_descriptions(mode)
        description_path.parent.mkdir(parents=True, exist_ok=True)
        description_path.write_text(
            json.dumps(descriptions, sort_keys=True),
            encoding="utf-8",
        )
    manifest_path = enumerate_stage4_contexts(
        cfg,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
        tokenizer=tokenizer,
        encdec_meta=encdec_meta,
        encdec_sequence_len=encdec_sequence_len,
    )
    manifest = load_stage4_contexts_manifest(Path(manifest_path))
    provider = Stage4TestContextProvider(
        tokenizer=tokenizer,
        context_keys=manifest["context_keys"],
        contexts_fingerprint=manifest["contexts_fingerprint"],
        model_max_seq_len=encdec_sequence_len,
    )
    return seed_fingerprint, provider


__all__ = [
    "Stage4TestContextProvider",
    "commit_seed_contract",
    "commit_stage4_context_contract",
]
