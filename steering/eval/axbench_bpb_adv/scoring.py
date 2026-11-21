"""Dual-completion class-margin BPB scoring for AxBench-BPB-Adv.

Each candidate is one attacked prefix + one steering configuration, keyed by
``core_prefix_id``. A candidate owns BOTH AxBench class-margin completions of
the SAME prefix: ``concept_rich`` (the concept-favoring continuation) and
``concept_avoiding`` (the contrast). Both completions are teacher-forced under
ONE steered intervention; the signed class margin ``primary_margin_y`` is the
metric:

    positive mode: primary_margin_y = BPB(avoid) - BPB(rich)
    negative mode: primary_margin_y = BPB(rich) - BPB(avoid)   (sign flip)

Higher ``primary_margin_y`` = stronger steering in both modes. There is NO
neutral reference; the neutral-vs-steered apparatus has been removed.

The additive adversary adds residual counter-pressure ON TOP of the steered
intervention (additive composes a counter-direction on top); both completions are scored
under that single intervention. Magnitude 0 yields a pure-steered control.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from loguru import logger

from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.direction_scale import (
    DirectionScalePreflight,
    DirectionScaleRequest,
    preflight_direction_scales,
)
from steering.eval.axbench_base.method_registry import (
    BaselineKind,
    get_method_by_name,
)
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.context_cache import load_context_segments
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle, score_sample
from steering.eval.axbench_bpb.tokenization import truncate_prefix_left
from steering.eval.axbench_bpb.types import (
    BpbScoreRow,
    Intervention,
    LoraParams,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.axbench_bpb_adv.artifact_identity import (
    ClusteredCandidateSnapshot,
    ClusteredSourceIdentity,
    assert_table_cluster_identity,
    require_clustered_candidate_snapshot,
)
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    CLUSTERED_CONTEXT_CACHE_KIND,
    load_clustered_context_segments_by_prefix_hash,
)
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    require_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    AxBenchBpbAdvConfig,
    ContextSignalMode,
    get_additive_direction_config,
)
from steering.eval.axbench_bpb_adv.context import (
    project_fixed_context_segments,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    read_evaluation_population,
    require_table_in_evaluation_population,
)
from steering.eval.axbench_bpb_adv.intervention_build import (
    apply_additive_counter,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    read_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)
from steering.eval.axbench_bpb_adv.progress import StageProgress
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    SCORE_SCHEMA,
)
from steering.eval.target_conditioning import ContextFeatureSegment

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.source_validation import (
        BpbAdvSourceValidation,
    )
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


class DualCompletionScoreFn(Protocol):
    def __call__(
        self,
        candidate_row: Mapping[str, Any],
        rich_input: ScoringInputs,
        avoid_input: ScoringInputs,
    ) -> tuple[BpbScoreRow, BpbScoreRow]: ...


def _ensure_candidates_schema(candidates: pa.Table) -> None:
    if not candidates.schema.equals(CANDIDATES_SCHEMA):
        raise AxBenchBpbAdvError("expected candidates with CANDIDATES_SCHEMA")


def _token_list(value: Any, *, field_name: str) -> list[int]:
    if value is None:
        raise AxBenchBpbAdvError(f"{field_name} is required")
    return [int(token_id) for token_id in value]


def _validate_candidate_digests(row: Mapping[str, Any]) -> None:
    source_tokens = _token_list(
        row["source_prompt_tokens"],
        field_name="source_prompt_tokens",
    )
    attacked_tokens = _token_list(
        row["attacked_prompt_tokens"],
        field_name="attacked_prompt_tokens",
    )
    if len(source_tokens) != len(attacked_tokens):
        raise AxBenchBpbAdvError(
            f"candidate {row['candidate_id']!r} is not length-preserving"
        )
    if source_tokens and source_tokens[0] != attacked_tokens[0]:
        raise AxBenchBpbAdvError(
            f"candidate {row['candidate_id']!r} changed the immutable BOS "
            "token at position 0"
        )
    if token_sequence_digest(source_tokens) != str(row["source_prompt_token_digest"]):
        raise AxBenchBpbAdvError(
            f"source prompt digest mismatch for candidate_id={row['candidate_id']!r}"
        )
    if token_sequence_digest(attacked_tokens) != str(row["attacked_prompt_token_digest"]):
        raise AxBenchBpbAdvError(
            f"attacked prompt digest mismatch for candidate_id={row['candidate_id']!r}"
        )
    start = int(row["editable_token_start"])
    count = int(row["editable_token_count"])
    stop = start + count
    if start < 0 or count < 0 or stop > len(source_tokens):
        raise AxBenchBpbAdvError(
            f"candidate {row['candidate_id']!r} editable span is outside prompt"
        )
    replacement_tokens = _token_list(
        row["replacement_tokens"],
        field_name="replacement_tokens",
    )
    if attacked_tokens[start:stop] != replacement_tokens:
        raise AxBenchBpbAdvError(
            f"candidate {row['candidate_id']!r} replacement_tokens do not match "
            "the attacked prompt editable span"
        )
    expected_attacked = (
        source_tokens[:start] + replacement_tokens + source_tokens[stop:]
    )
    if attacked_tokens != expected_attacked:
        raise AxBenchBpbAdvError(
            f"candidate {row['candidate_id']!r} changed tokens outside the "
            "editable span"
        )


def _source_context_segments(
    *,
    candidate_row: Mapping[str, Any],
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]],
) -> tuple[ContextFeatureSegment, ...]:
    prefix_hash = str(candidate_row["prefix_hash"])
    source_segments = context_segments_by_prefix_hash.get(prefix_hash)
    if source_segments is None:
        raise AxBenchBpbAdvError(
            "context cache miss for source prefix hash "
            f"{prefix_hash!r} on candidate_id={candidate_row['candidate_id']!r}"
        )
    if not source_segments:
        raise AxBenchBpbAdvError(
            f"context cache returned no segments for source prefix hash {prefix_hash!r}"
        )
    return source_segments


def _resolve_context_segments(
    *,
    candidate_row: Mapping[str, Any],
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]],
    tokenizer: Any,
) -> tuple[ContextFeatureSegment, ...]:
    mode = ContextSignalMode.coerce(str(candidate_row["context_signal_mode"]))
    source_segments = _source_context_segments(
        candidate_row=candidate_row,
        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
    )
    source_tokens = _token_list(
        candidate_row["source_prompt_tokens"],
        field_name="source_prompt_tokens",
    )
    attacked_tokens = _token_list(
        candidate_row["attacked_prompt_tokens"],
        field_name="attacked_prompt_tokens",
    )
    if mode == ContextSignalMode.FIXED_CONTEXT:
        return project_fixed_context_segments(
            tokenizer=tokenizer,
            source_tokens=source_tokens,
            attacked_tokens=attacked_tokens,
            source_segments=source_segments,
        )
    if mode == ContextSignalMode.NOT_APPLICABLE:
        return source_segments
    raise AxBenchBpbAdvError(f"unsupported context_signal_mode={mode.value!r}")


def _completion_scoring_input(
    *,
    candidate_row: Mapping[str, Any],
    content_class: str,
    sample_id: str,
    output_tokens: list[int],
    prompt_tokens: list[int],
    prefix_context_segments: tuple[ContextFeatureSegment, ...],
    token_bytes: t.Tensor,
    max_seq_len: int,
) -> ScoringInputs:
    if not output_tokens:
        raise AxBenchBpbAdvError(
            f"candidate {candidate_row['candidate_id']!r} has no {content_class} "
            "output tokens"
        )
    output_tensor = t.tensor(output_tokens, dtype=t.long)
    try:
        output_byte_weights = token_bytes[output_tensor]
    except IndexError as exc:
        raise AxBenchBpbAdvError(
            f"candidate {candidate_row['candidate_id']!r} has {content_class} output "
            "token outside tokenizer vocab"
        ) from exc
    if int(output_byte_weights.sum().item()) <= 0:
        raise AxBenchBpbAdvError(
            f"candidate {candidate_row['candidate_id']!r} {content_class} output "
            "tokens decode to zero bytes"
        )
    _, truncated = truncate_prefix_left(
        prefix=prompt_tokens,
        n_output=len(output_tokens),
        max_seq_len=max_seq_len,
    )
    return ScoringInputs(
        sample_id=sample_id,
        concept_id=int(candidate_row["concept_id"]),
        mode=SteeringMode(str(candidate_row["mode"])),
        content_class=content_class,
        prefix_text=str(candidate_row["attacked_prompt_text"]),
        prefix_hash=make_prefix_hash(str(candidate_row["attacked_prompt_text"])),
        prefix_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=prefix_context_segments,
        n_prefix=len(prompt_tokens),
        n_output=len(output_tokens),
        output_byte_weights=output_byte_weights,
        truncated=truncated,
    )


def candidate_row_to_scoring_inputs(
    *,
    candidate_row: Mapping[str, Any],
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]],
    tokenizer: Any,
    token_bytes: t.Tensor,
    max_seq_len: int,
) -> tuple[ScoringInputs, ScoringInputs]:
    """Build the ``(rich_input, avoid_input)`` pair for one candidate.

    Both inputs share the attacked prompt tokens, prefix text/hash, and context
    segments (encdec conditions on the prompt only, never the completion); they
    differ only in output tokens/bytes, ``content_class``, and ``sample_id``.
    """

    _validate_candidate_digests(candidate_row)
    prompt_tokens = _token_list(
        candidate_row["attacked_prompt_tokens"],
        field_name="attacked_prompt_tokens",
    )
    target_model = str(candidate_row["target_model"])
    if target_model == "encdec":
        prefix_context_segments = _resolve_context_segments(
            candidate_row=candidate_row,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
        )
    elif target_model == "decoder_only":
        prefix_context_segments = ()
    else:
        raise AxBenchBpbAdvError(f"unsupported target_model={target_model!r}")

    rich_input = _completion_scoring_input(
        candidate_row=candidate_row,
        content_class="concept_rich",
        sample_id=str(candidate_row["concept_rich_sample_id"]),
        output_tokens=_token_list(
            candidate_row["concept_rich_output_tokens"],
            field_name="concept_rich_output_tokens",
        ),
        prompt_tokens=prompt_tokens,
        prefix_context_segments=prefix_context_segments,
        token_bytes=token_bytes,
        max_seq_len=max_seq_len,
    )
    avoid_input = _completion_scoring_input(
        candidate_row=candidate_row,
        content_class="concept_avoiding",
        sample_id=str(candidate_row["concept_avoiding_sample_id"]),
        output_tokens=_token_list(
            candidate_row["concept_avoiding_output_tokens"],
            field_name="concept_avoiding_output_tokens",
        ),
        prompt_tokens=prompt_tokens,
        prefix_context_segments=prefix_context_segments,
        token_bytes=token_bytes,
        max_seq_len=max_seq_len,
    )
    return rich_input, avoid_input


def _validate_side_row(
    *,
    candidate_row: Mapping[str, Any],
    scoring_input: ScoringInputs,
    side_name: str,
    side_row: BpbScoreRow,
) -> None:
    expected = {
        "method_name": str(candidate_row["method_name"]),
        "concept_id": int(candidate_row["concept_id"]),
        "mode": str(candidate_row["mode"]),
        "content_class": scoring_input.content_class,
        "target": str(candidate_row["target_model"]),
        "coefficient_str": str(candidate_row["coefficient_str"]),
        "sample_id": scoring_input.sample_id,
    }
    actual = {
        "method_name": side_row.method_name,
        "concept_id": side_row.concept_id,
        "mode": side_row.mode,
        "content_class": side_row.content_class,
        "target": side_row.target,
        "coefficient_str": side_row.coefficient_str,
        "sample_id": side_row.sample_id,
    }
    for field_name, expected_value in expected.items():
        if actual[field_name] != expected_value:
            raise AxBenchBpbAdvError(
                f"{side_name} score row for candidate_id="
                f"{candidate_row['candidate_id']!r} has {field_name}="
                f"{actual[field_name]!r}, expected {expected_value!r}"
            )
    if side_row.n_output_tokens != scoring_input.n_output:
        raise AxBenchBpbAdvError(
            f"{side_name} score row token count mismatch for candidate_id="
            f"{candidate_row['candidate_id']!r}"
        )
    if int(side_row.n_output_bytes) <= 0:
        raise AxBenchBpbAdvError(
            f"{side_name} score row for candidate_id="
            f"{candidate_row['candidate_id']!r} has non-positive output bytes"
        )


def _is_control_candidate(candidate_row: Mapping[str, Any]) -> bool:
    """A control candidate applies no adversary (k=0 / magnitude 0).

    Prompt-token controls keep the attacked prompt identical to the source
    prompt; additive controls have zero counter-direction magnitude. Both are
    captured by ``adversarial_strength == 0`` combined with an unchanged prompt.
    """

    if float(candidate_row["adversarial_strength"]) != 0.0:
        return False
    source_tokens = _token_list(
        candidate_row["source_prompt_tokens"],
        field_name="source_prompt_tokens",
    )
    attacked_tokens = _token_list(
        candidate_row["attacked_prompt_tokens"],
        field_name="attacked_prompt_tokens",
    )
    return source_tokens == attacked_tokens


_SCORING_IDENTITY_EXCLUDED = frozenset(
    {"candidate_id", "attack_row_id"}
)


def _hashable(value: Any) -> Any:
    """Convert a candidate cell into a hashable form for the scoring-identity key.

    Token columns arrive as Python lists (from ``to_pylist``); everything else is
    already a scalar. Lists become tuples (recursively, defensively) so the
    per-candidate identity is hashable.
    """
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return value


def _scoring_identity_key(candidate_row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity of a candidate's *scoring computation*.

    Two candidates with the same key produce byte-identical dual-completion
    scores. The key is EVERY candidate column EXCEPT ``candidate_id`` and
    ``attack_row_id``. Those two are excluded because
    they are per-row identifiers that do NOT feed the
    forward pass: ``attack_row_id`` is never read in scoring at all, and
    ``candidate_id`` -- though it IS read (the duplicate guard, and copied to the
    output) -- is per-row UNIQUE, so including it would make every key distinct
    and silently disable the cache. (Do NOT "fix" this by adding ``candidate_id``
    to the key on the belief that it is unread: it is read, just not as a scoring
    input.)

    All OTHER columns are kept, including the raw token lists even though each
    already has a 1:1 SHA-256 ``*_digest`` column in the key. This conservatism
    is deliberate: the key can then only ever UNDER-dedup (still byte-identical),
    never over-share, with NO reasoning about which columns are redundant. The
    extra list hashing is negligible beside the dual GPU forward pass it guards.
    Distinct attack rows whose candidates share every scoring input (e.g. k=0
    control twins) differ in EXACTLY the two excluded id fields, so they collapse
    to one forward-pass evaluation (#511).
    """
    return tuple(
        _hashable(candidate_row[name])
        for name in CANDIDATES_SCHEMA.names
        if name not in _SCORING_IDENTITY_EXCLUDED
    )


def build_bpb_adv_score_table(
    *,
    candidates: pa.Table,
    context_segments_by_prefix_hash: Mapping[str, tuple[ContextFeatureSegment, ...]],
    tokenizer: Any,
    token_bytes: t.Tensor,
    max_seq_len_by_target: Mapping[str, int],
    score_pair: DualCompletionScoreFn,
    stats: dict[str, int] | None = None,
    log_progress_every: int = 1000,
) -> pa.Table:
    _ensure_candidates_schema(candidates)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Memoize the dual-completion forward passes: distinct attack rows whose
    # candidates share every scoring input (e.g. k=0 control twins) are identical
    # except candidate_id/attack_row_id (both unused by scoring), so they score
    # identically. We run score_pair ONCE per scoring identity and reuse it --
    # byte-identical output, fewer forward passes for the no-adversary controls
    # (issue #511).
    score_cache: dict[tuple[Any, ...], tuple[BpbScoreRow, BpbScoreRow]] = {}
    # This loop is the score stage's entire cost: 1.9 h for 160k rows on a real
    # clustered sweep, during which the stage logged NOTHING between its setup
    # lines and its final table. The only way to tell progress from a hang was to
    # watch for output files that are written once, at the end.
    #
    # Memo hits are reported as `reused`, not as work. They skip the forward
    # passes entirely, so counting them in the rate would understate the true
    # per-row cost and project a finish that arrives late -- the k=0 control
    # twins alone are a large share of the rows (#511).
    progress = StageProgress(
        "bpb-adv score", candidates.num_rows, "row", log_every=log_progress_every
    )
    for candidate_row in candidates.to_pylist():
        candidate_id = str(candidate_row["candidate_id"])
        if candidate_id in seen:
            raise AxBenchBpbAdvError(f"duplicate candidate_id={candidate_id!r}")
        seen.add(candidate_id)
        target_model = str(candidate_row["target_model"])
        max_seq_len = max_seq_len_by_target.get(target_model)
        if max_seq_len is None:
            raise AxBenchBpbAdvError(
                f"missing max_seq_len for target_model={target_model!r}"
            )
        rich_input, avoid_input = candidate_row_to_scoring_inputs(
            candidate_row=candidate_row,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
            token_bytes=token_bytes,
            max_seq_len=max_seq_len,
        )
        identity = _scoring_identity_key(candidate_row)
        cached = score_cache.get(identity)
        if cached is None:
            rich_score, avoid_score = score_pair(candidate_row, rich_input, avoid_input)
            score_cache[identity] = (rich_score, avoid_score)
            progress.record(reused=False)
        else:
            rich_score, avoid_score = cached
            progress.record(reused=True)
        _validate_side_row(
            candidate_row=candidate_row,
            scoring_input=rich_input,
            side_name="concept_rich",
            side_row=rich_score,
        )
        _validate_side_row(
            candidate_row=candidate_row,
            scoring_input=avoid_input,
            side_name="concept_avoiding",
            side_row=avoid_score,
        )
        loss_rich = float(rich_score.bpb_intervened)
        loss_avoid = float(avoid_score.bpb_intervened)
        rows.append(
            {
                "candidate_id": candidate_id,
                "core_prefix_id": str(candidate_row["core_prefix_id"]),
                "concept_id": int(candidate_row["concept_id"]),
                "mode": str(candidate_row["mode"]),
                "method_name": str(candidate_row["method_name"]),
                "target_model": target_model,
                "coefficient_str": str(candidate_row["coefficient_str"]),
                "adversarial_method": str(candidate_row["adversarial_method"]),
                "adversarial_strength": float(candidate_row["adversarial_strength"]),
                "strength_unit": str(candidate_row["strength_unit"]),
                "strength_index": int(candidate_row["strength_index"]),
                "context_signal_mode": str(candidate_row["context_signal_mode"]),
                "source_prompt_tokens": _token_list(
                    candidate_row["source_prompt_tokens"],
                    field_name="source_prompt_tokens",
                ),
                "source_prompt_token_digest": str(candidate_row["source_prompt_token_digest"]),
                "attacked_prompt_tokens": _token_list(
                    candidate_row["attacked_prompt_tokens"],
                    field_name="attacked_prompt_tokens",
                ),
                "attacked_prompt_token_digest": str(candidate_row["attacked_prompt_token_digest"]),
                "editable_token_start": int(candidate_row["editable_token_start"]),
                "editable_token_count": int(candidate_row["editable_token_count"]),
                "replacement_tokens": _token_list(
                    candidate_row["replacement_tokens"],
                    field_name="replacement_tokens",
                ),
                "loss_concept_rich": loss_rich,
                "loss_concept_avoiding": loss_avoid,
                "is_control": _is_control_candidate(candidate_row),
                "n_rich_output_tokens": int(rich_score.n_output_tokens),
                "n_avoid_output_tokens": int(avoid_score.n_output_tokens),
                "n_rich_output_bytes": int(rich_score.n_output_bytes),
                "n_avoid_output_bytes": int(avoid_score.n_output_bytes),
                "rich_truncated": bool(rich_score.truncated),
                "avoid_truncated": bool(avoid_score.truncated),
            }
        )
    # Derived (no running counters to keep in sync with the loop): exactly one
    # cache entry is added per unique scoring identity; every other candidate
    # reused one.
    scored_unique = len(score_cache)
    cache_hits = candidates.num_rows - scored_unique
    if stats is not None:
        stats["scored_unique"] = scored_unique
        stats["cache_hits"] = cache_hits
    if cache_hits:
        logger.info(
            "bpb-adv scoring: {} candidates, {} unique dual-completion "
            "evaluations, {} reused (deduplicated)",
            candidates.num_rows,
            scored_unique,
            cache_hits,
        )
    logger.info(progress.summary())
    return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)


def load_candidate_table(
    path: Path,
    *,
    paths: AxBenchBpbAdvPaths | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    source_identity: ClusteredSourceIdentity | None = None,
) -> pa.Table:
    table = pq.read_table(path)
    _ensure_candidates_schema(table)
    if paths is not None:
        assert_table_cluster_identity(
            table,
            paths=paths,
            contract=cluster_contract,
            artifact_family="candidates",
            source_identity=source_identity,
        )
    elif cluster_contract is not None:
        raise AxBenchBpbAdvError(
            "clustered candidate loading requires AxBenchBpbAdvPaths for source identity"
        )
    return table


def load_context_segments_by_prefix_hash(
    paths: AxBenchBpbAdvPaths,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    if cluster_contract is not None:
        return load_clustered_context_segments_by_prefix_hash(
            paths=paths,
            contract=cluster_contract,
        )
    try:
        manifest = json.loads(paths.context_cache_manifest().read_text())
    except Exception as exc:
        raise AxBenchBpbAdvError(
            "BPB-Adv scoring requires a BPB context cache under "
            f"{paths.context_cache_dir()}"
        ) from exc
    if not isinstance(manifest, dict):
        raise AxBenchBpbAdvError("BPB-Adv context-cache manifest must be a JSON object")
    if manifest.get("kind") == CLUSTERED_CONTEXT_CACHE_KIND:
        raise AxBenchBpbAdvError(
            "unclustered BPB-Adv reader rejects a clustered context-cache manifest"
        )
    out: dict[str, tuple[ContextFeatureSegment, ...]] = {}
    for prefix_hash in manifest:
        out[str(prefix_hash)] = load_context_segments(
            paths.context_cache_file(str(prefix_hash))
        )
    return out


def _build_token_bytes(tokenizer: Any) -> t.Tensor:
    vocab_size = int(tokenizer.get_vocab_size())
    weights = t.zeros(vocab_size, dtype=t.int64)
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
        except Exception as exc:
            # Fail loudly: silently leaving a 0 byte-weight under-counts this
            # token's bytes and inflates BPB downstream with no error. The id is
            # always in-vocab here, so a decode failure is a real problem.
            raise AxBenchBpbAdvError(
                f"failed to decode token id {token_id} while building the "
                "byte-weight table"
            ) from exc
        weights[token_id] = len(str(text).encode("utf-8"))
    return weights


def _bpb_config_from_adv_config(cfg: AxBenchBpbAdvConfig) -> AxBenchBpbConfig:
    return AxBenchBpbConfig(
        source_run_dir=cfg.source_run_dir,
        output_root=cfg.output_root,
        run_tag=cfg.run_tag,
        feature_dir=cfg.feature_dir,
        feature_statistics_path=cfg.feature_statistics_path,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
        neuronpedia_model_id=cfg.neuronpedia_model_id,
        feature_extraction_model_id=cfg.feature_extraction_model_id,
        neuronpedia_source=cfg.neuronpedia_source,
        extraction_max_tokens=cfg.extraction_max_tokens,
        extraction_threshold=cfg.extraction_threshold,
        decoder_only=dict(cfg.decoder_only_target),
        encdec=dict(cfg.encdec_target),
        device=cfg.device,
        next_segment_conditioning_offset=cfg.next_segment_conditioning_offset,
        diffmean_and_probe_layer=cfg.diffmean_and_probe_layer,
        stage="bpb_score",
        enabled_methods=cfg.base_methods,
        disabled_methods=[],
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
        device_batch_size=cfg.device_batch_size,
        context_cache=cfg.context_cache,
        lora=dict(cfg.lora_adapter),
        log_progress_every=cfg.log_progress_every,
        offload_between_stages=cfg.offload_between_stages,
    )


def _read_concepts(source_paths: SourcePaths) -> dict[int, Concept]:
    return _concepts_from_entries(
        json.loads(source_paths.concepts_json().read_text())
    )


def _concepts_from_entries(concepts_data: Any) -> dict[int, Concept]:
    """Build the scorer's concept index from already-parsed source entries."""

    if not isinstance(concepts_data, list):
        raise AxBenchBpbAdvError("concepts.json must contain a list of concepts")
    concepts: list[Concept] = []
    for entry in concepts_data:
        # A non-Mapping entry would otherwise raise an opaque ``TypeError`` from
        # ``Concept(**entry)``; surface a clear project error instead.
        if not isinstance(entry, Mapping):
            raise AxBenchBpbAdvError(
                "concepts.json entries must be objects (mappings); got "
                f"{type(entry).__name__}"
            )
        concepts.append(Concept(**entry))
    return {concept.concept_id: concept for concept in concepts}


@dataclass(kw_only=True)
class ModelBackedDualCompletionScorer:
    """Score BOTH class-margin completions under ONE steered intervention.

    There is no neutral reference: ``__call__`` builds a single steered
    intervention for the candidate and teacher-forces ``concept_rich`` and
    ``concept_avoiding`` under it, returning ``(rich_score, avoid_score)``.
    """

    target_bundles: Mapping[str, TargetModelBundle]
    target_model_metas: Mapping[str, Any]
    concepts_by_id: Mapping[int, Concept]
    source_paths: SourcePaths
    tokenizer: Any
    device: t.device
    diffmean_layer: int
    inactive_sampler: Callable[..., dict[int, int]] | None
    apply_max_act_scale: bool
    lora: LoraParams | None = None
    # additive counter-steering: the concept-direction family pushed (opposing the
    # concept) on top of the full baseline steering. Only consulted on additive rows.
    counter_direction_family: str = "diffmean"
    # Precomputed bag-backed methods: run-level artifact paths needed by
    # _build_ours_intervention to locate and load conditioning bags. Ignored
    # (and safe to omit) for all other methods.
    adv_paths: Any | None = None
    cluster_contract: ClusterFeatureContract | None = None
    source_validation: BpbAdvSourceValidation | None = None

    def __call__(
        self,
        candidate_row: Mapping[str, Any],
        rich_input: ScoringInputs,
        avoid_input: ScoringInputs,
    ) -> tuple[BpbScoreRow, BpbScoreRow]:
        target_model = str(candidate_row["target_model"])
        target_bundle = self.target_bundles.get(target_model)
        if target_bundle is None:
            raise AxBenchBpbAdvError(
                f"no target bundle loaded for target_model={target_model!r}"
            )
        method = get_method_by_name(str(candidate_row["method_name"]))
        concept = self.concepts_by_id.get(int(candidate_row["concept_id"]))
        if concept is None:
            raise AxBenchBpbAdvError(
                f"source concepts do not contain concept_id={candidate_row['concept_id']}"
            )
        coefficient_str = str(candidate_row["coefficient_str"])
        coefficient = _coefficient_from_str(coefficient_str)
        intervention = self._steered_intervention(
            method=method,
            concept=concept,
            coefficient=coefficient,
            candidate_row=candidate_row,
        )
        rich_score = score_sample(
            inputs=rich_input,
            target_bundle=target_bundle,
            intervention=intervention,
            device=self.device,
            coefficient_str=coefficient_str,
            method_name=method.name,
            lora=self.lora,
        )
        avoid_score = score_sample(
            inputs=avoid_input,
            target_bundle=target_bundle,
            intervention=intervention,
            device=self.device,
            coefficient_str=coefficient_str,
            method_name=method.name,
            lora=self.lora,
        )
        return rich_score, avoid_score

    def _base_steered_intervention(
        self,
        *,
        method: Any,
        concept: Concept,
        coefficient: int | float | None,
        candidate_row: Mapping[str, Any],
    ) -> Intervention:
        rng_seed_bytes = hashlib.sha256(
            (
                f"{candidate_row['method_name']}|{candidate_row['concept_id']}|"
                f"{candidate_row['core_prefix_id']}|{candidate_row['coefficient_str']}"
            ).encode()
        ).digest()
        target_model = str(candidate_row["target_model"])
        target_bundle = self.target_bundles.get(target_model)
        if target_bundle is None:
            raise AxBenchBpbAdvError(f"no target bundle loaded for target_model={target_model!r}")
        direction_payload = None
        if (
            self.source_validation is not None
            and method.baseline in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE)
        ):
            direction_payload = self.source_validation.direction_artifact_bytes(
                family=method.baseline.value,
                target=target_model,
                mode=method.mode.value,
                concept_id=concept.concept_id,
            )
        intervention_kwargs: dict[str, Any] = {
            "method": method,
            "concept": concept,
            "coefficient": coefficient,
            "source_paths": self.source_paths,
            "diffmean_layer": self.diffmean_layer,
            "target_model_meta": self.target_model_metas[target_model],
            "inactive_sampler": self.inactive_sampler,
            "tokenizer_nanochat": self.tokenizer,
            "rng": random.Random(rng_seed_bytes),
            "adv_paths": self.adv_paths,
            "core_prefix_id": str(candidate_row["core_prefix_id"]),
            "apply_max_act_scale": self.apply_max_act_scale,
            "model": target_bundle.model,
        }
        if self.cluster_contract is not None:
            intervention_kwargs["cluster_contract"] = self.cluster_contract
        if direction_payload is not None:
            intervention_kwargs["direction_payload"] = direction_payload
        return build_intervention(**intervention_kwargs)

    def _steered_intervention(
        self,
        *,
        method: Any,
        concept: Concept,
        coefficient: int | float | None,
        candidate_row: Mapping[str, Any],
    ) -> Intervention:
        # Base steering is still built standalone (model_optimize depends on
        # ``_base_steered_intervention``); the additive counter is attached via the
        # shared ``apply_additive_counter`` so the scored intervention is the SAME
        # one generation builds via ``build_steered_intervention``.
        return apply_additive_counter(
            self._base_steered_intervention(
                method=method,
                concept=concept,
                coefficient=coefficient,
                candidate_row=candidate_row,
            ),
            adversarial_method=str(candidate_row["adversarial_method"]),
            adversarial_strength=float(candidate_row["adversarial_strength"]),
            method=method,
            concept=concept,
            source_paths=self.source_paths,
            diffmean_layer=self.diffmean_layer,
            target=str(candidate_row["target_model"]),
            counter_direction_family=self.counter_direction_family,
            apply_max_act_scale=self.apply_max_act_scale,
            source_validation=self.source_validation,
        )

    def _control_intervention(
        self,
        *,
        method: Any,
        concept: Concept,
        candidate_row: Mapping[str, Any],
    ) -> Intervention:
        target_model = str(candidate_row["target_model"])
        if target_model == "encdec":
            base_intervention = Intervention.ours(feature_conditions={})
        elif target_model == "decoder_only":
            base_intervention = Intervention.none()
        else:
            raise AxBenchBpbAdvError(f"unsupported target_model={target_model!r}")

        return apply_additive_counter(
            base_intervention,
            adversarial_method=str(candidate_row["adversarial_method"]),
            adversarial_strength=float(candidate_row["adversarial_strength"]),
            method=method,
            concept=concept,
            source_paths=self.source_paths,
            diffmean_layer=self.diffmean_layer,
            target=target_model,
            counter_direction_family=self.counter_direction_family,
            apply_max_act_scale=self.apply_max_act_scale,
            source_validation=self.source_validation,
        )


def _require_precomputed_bag_scored_prefix_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    paths: AxBenchBpbAdvPaths,
    stats: dict[str, int] | None = None,
    label: str = "bpb-adv precomputed-bag coverage guard",
    feature_contract_fingerprint: str | None = None,
) -> None:
    """Require bag-backed rows to be covered by their scored-prefix artifacts.

    This is the SINGLE source of the ``(concept_id, mode, family, core_prefix_id)``
    keying truth: both the candidate-Table score-stage guard
    (:func:`require_precomputed_bag_candidates_cover_scored_prefixes`) and the
    attack-row optimize-stage guard
    (:func:`require_precomputed_bag_prompt_rows_cover_scored_prefixes`) build
    their mask here, so the two stages can never drift on which prefixes a
    bag-backed method is allowed to attack/score.

    Bag-backed methods are crossed with EVERY prefix, but a conditioning bag
    exists only for the prefixes the derive recorded for that family. The scored
    set per (concept, mode, family) is the artifact the derive persisted. A
    bag-backed row whose ``core_prefix_id`` is absent from that set is stale
    derive state and raises loudly. Non-bag-backed rows never read artifacts.
    """
    method_names = {str(row["method_name"]) for row in rows}
    if not any(
        precomputed_bag_method_family(name) is not None for name in method_names
    ):
        return

    scored_cache: dict[tuple[int, str, PrecomputedBagFamily], frozenset[str]] = {}
    kept_by_method: dict[str, int] = {}
    dropped_by_method: dict[str, int] = {}
    missing_examples: list[str] = []
    for row in rows:
        method_name = str(row["method_name"])
        family = precomputed_bag_method_family(method_name)
        if family is None:
            continue
        concept_id = int(row["concept_id"])
        mode = str(row["mode"])
        core_prefix_id = str(row["core_prefix_id"])
        cache_key = (concept_id, mode, family)
        scored = scored_cache.get(cache_key)
        if scored is None:
            scored = frozenset(
                read_ours_gcg_scored_prefixes(
                    paths.ours_gcg_scored_prefixes(cache_key[0], cache_key[1], family),
                    **(
                        {}
                        if feature_contract_fingerprint is None
                        else {
                            "expected_feature_contract_fingerprint": (
                                feature_contract_fingerprint
                            )
                        }
                    ),
                )
            )
            scored_cache[cache_key] = scored
        keep = core_prefix_id in scored
        bucket = kept_by_method if keep else dropped_by_method
        bucket[method_name] = bucket.get(method_name, 0) + 1
        if not keep and len(missing_examples) < 8:
            missing_examples.append(
                f"method={method_name} concept_id={concept_id} mode={mode} "
                f"family={family.value} core_prefix_id={core_prefix_id}"
            )

    n_kept = sum(kept_by_method.values())
    n_dropped = sum(dropped_by_method.values())
    kept_sorted = dict(sorted(kept_by_method.items()))
    dropped_sorted = dict(sorted(dropped_by_method.items()))
    logger.info(
        f"{label}: kept {n_kept} dropped {n_dropped} bag-backed row(s) "
        f"(kept per method={kept_sorted}, dropped per method={dropped_sorted})"
    )
    if stats is not None:
        stats["precomputed_bag_coverage_kept"] = n_kept
        stats["precomputed_bag_coverage_dropped"] = n_dropped
    if n_dropped:
        examples = "; ".join(missing_examples)
        raise AxBenchBpbAdvError(
            f"{label}: {n_dropped} bag-backed row(s) passed the universal "
            "evaluation population but are absent from their per-family "
            "scored-prefix artifact; refusing to score or optimize an "
            f"incomparable subset (dropped per method={dropped_sorted}; "
            f"examples={examples}). Regenerate the run from prepare_attack_rows "
            "through build_candidates with the current code, or remove stale "
            "precomputed-bag artifacts for this run_tag."
        )


def require_precomputed_bag_candidates_cover_scored_prefixes(
    candidates: pa.Table,
    *,
    paths: AxBenchBpbAdvPaths,
    stats: dict[str, int] | None = None,
    feature_contract_fingerprint: str | None = None,
) -> pa.Table:
    """Require bag-backed candidate rows to be covered by family scored-prefix artifacts.

    Production score stages call this after enforcing the universal evaluation
    population. At that point a missing per-family scored prefix is stale or
    incomplete derive state, not an allowed post-hoc subset selection.
    """
    _require_precomputed_bag_scored_prefix_coverage(
        candidates.to_pylist(),
        paths=paths,
        stats=stats,
        label="bpb-adv precomputed-bag coverage guard",
        **(
            {}
            if feature_contract_fingerprint is None
            else {"feature_contract_fingerprint": feature_contract_fingerprint}
        ),
    )
    return candidates


def require_precomputed_bag_prompt_rows_cover_scored_prefixes(
    prompt_rows: list[dict[str, Any]],
    *,
    paths: AxBenchBpbAdvPaths,
    feature_contract_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Require universal-held-out bag-backed prompt rows to have family coverage."""
    _require_precomputed_bag_scored_prefix_coverage(
        prompt_rows,
        paths=paths,
        stats=None,
        label="bpb-adv precomputed-bag prompt coverage guard",
        **(
            {}
            if feature_contract_fingerprint is None
            else {"feature_contract_fingerprint": feature_contract_fingerprint}
        ),
    )
    return prompt_rows


def _preflight_direction_scales_for_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    concepts_by_id: Mapping[int, Concept],
    source_paths: SourcePaths,
    expected_layer: int,
    counter_direction_family: str,
    source_validation: BpbAdvSourceValidation | None = None,
    stage: str = "score",
) -> DirectionScalePreflight:
    """Preflight exactly the base/counter direction artifacts rows consume."""
    requests: list[DirectionScaleRequest] = []
    for row in candidate_rows:
        method = get_method_by_name(str(row["method_name"]))
        is_scalar = method.baseline in (
            BaselineKind.DIFFMEAN,
            BaselineKind.LINEAR_PROBE,
        )
        is_scaled_counter = (
            str(row.get("adversarial_method", ""))
            == AdversarialMethod.ADDITIVE_DIRECTION.value
            and float(row.get("adversarial_strength") or 0.0) != 0.0
        )
        concept_id = int(row["concept_id"])
        if concept_id not in concepts_by_id:
            raise AxBenchBpbAdvError(
                f"bpb-adv {stage} direction preflight has unknown concept_id={concept_id}"
            )
        target = str(row["target_model"])
        mode = method.mode.value
        families = []
        if is_scalar:
            families.append(method.baseline.value)
        if is_scaled_counter:
            families.append(counter_direction_family)
        for family in families:
            path_fn = (
                source_paths.diffmean_direction
                if family == "diffmean"
                else source_paths.linear_probe_direction
            )
            path: Path | None = None
            payload: bytes | None = None
            if source_validation is not None:
                payload = source_validation.direction_artifact_bytes(
                    family=family,
                    target=target,
                    mode=mode,
                    concept_id=concept_id,
                )
            else:
                path = path_fn(target, mode, concept_id)
            requests.append(
                DirectionScaleRequest(
                    family=family,
                    target_model=target,
                    mode=mode,
                    concept_id=concept_id,
                    path=path,
                    payload=payload,
                )
            )
    return preflight_direction_scales(
        requests,
        expected_layer=expected_layer,
        stage_label=f"bpb-adv {stage}",
        error_cls=AxBenchBpbAdvError,
    )


def run_model_backed_bpb_adv_score_stage(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    candidates: pa.Table,
    stats: dict[str, int] | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    stage_source: ClusteredStageSource | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> pa.Table:
    resolved_stage_source = require_clustered_stage_source(
        stage_source=stage_source,
        contract=cluster_contract,
        surface="class-margin score stage",
    )
    source_validation = (
        None
        if resolved_stage_source is None
        else resolved_stage_source.validation
    )
    require_clustered_candidate_snapshot(
        candidate_snapshot,
        paths=paths,
        contract=cluster_contract,
        source_identity=(
            None
            if resolved_stage_source is None
            else resolved_stage_source.source_identity
        ),
        surface="class-margin score stage",
        table=candidates,
    )
    if cfg.run_tag is None:
        raise AxBenchBpbAdvError("cfg.run_tag must be set before scoring")
    if candidates.num_rows == 0:
        raise AxBenchBpbAdvError("no candidates to score")
    record_counter("axbench_bpb_adv.score.input_candidate_rows", candidates.num_rows)
    with profile_region("axbench_bpb_adv.score.population_guard", rows=candidates.num_rows):
        candidates = require_table_in_evaluation_population(
            candidates,
            population=read_evaluation_population(
                paths.evaluation_population(),
                **(
                    {}
                    if cluster_contract is None
                    else {
                        "expected_feature_contract_fingerprint": (
                            cluster_contract.feature_contract_fingerprint
                        )
                    }
                ),
            ),
            label="score-stage candidates",
            stats=stats,
        )
    with profile_region(
        "axbench_bpb_adv.score.precomputed_bag_coverage_guard",
        rows=candidates.num_rows,
    ):
        candidates = require_precomputed_bag_candidates_cover_scored_prefixes(
            candidates,
            paths=paths,
            stats=stats,
            **(
                {}
                if cluster_contract is None
                else {
                    "feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
    candidate_rows = candidates.to_pylist()
    bpb_cfg = _bpb_config_from_adv_config(cfg)
    runtime_kwargs: dict[str, Any] = {
        "cfg": bpb_cfg,
        "ddp_rank": 0,
        "ddp_world_size": 1,
    }
    if cluster_contract is not None:
        runtime_kwargs["cluster_contract"] = cluster_contract
    runtime_state = RuntimeState(**runtime_kwargs)
    try:
        targets = {str(row["target_model"]) for row in candidate_rows}
        with profile_region("axbench_bpb_adv.score.target_runtime_setup", n_targets=len(targets)):
            target_bundles: dict[str, TargetModelBundle] = {}
            if "decoder_only" in targets:
                target_bundles["decoder_only"] = runtime_state.decoder_only()
            if "encdec" in targets:
                target_bundles["encdec"] = runtime_state.encdec()
        unknown_targets = targets - set(target_bundles)
        if unknown_targets:
            raise AxBenchBpbAdvError(
                f"unsupported target_model values: {sorted(unknown_targets)}"
            )
        tokenizer = next(iter(target_bundles.values())).tokenizer
        with profile_region("axbench_bpb_adv.score.build_token_bytes"):
            token_bytes = _build_token_bytes(tokenizer)
        with profile_region("axbench_bpb_adv.score.context_segments_load"):
            context_segments_by_hash = (
                load_context_segments_by_prefix_hash(
                    paths,
                    **(
                        {}
                        if cluster_contract is None
                        else {"cluster_contract": cluster_contract}
                    ),
                )
                if "encdec" in targets
                else {}
            )
        record_counter(
            "axbench_bpb_adv.score.context_prefixes_loaded",
            len(context_segments_by_hash),
        )
        detect_kwargs = (
            {}
            if cluster_contract is None
            else {"cluster_contract": cluster_contract}
        )
        target_metas = {
            target: detect_target_model_metadata(
                bundle.model,
                bpb_cfg,
                **detect_kwargs,
            )
            for target, bundle in target_bundles.items()
        }
        source_paths = SourcePaths(Path(cfg.source_run_dir))
        with profile_region("axbench_bpb_adv.score.concepts_and_scale_preflight"):
            concepts_by_id = (
                _read_concepts(source_paths)
                if source_validation is None
                else _concepts_from_entries(
                    source_validation.concept_entries()
                )
            )
            additive_cfg = get_additive_direction_config(cfg)
            _preflight_direction_scales_for_rows(
                candidate_rows=candidate_rows,
                concepts_by_id=concepts_by_id,
                source_paths=source_paths,
                expected_layer=cfg.diffmean_and_probe_layer,
                counter_direction_family=(
                    additive_cfg.counter_direction_family
                    if additive_cfg is not None
                    else "diffmean"
                ),
                source_validation=source_validation,
            )
        lora = (
            LoraParams(
                cfg=LoraConfig(**cfg.lora_adapter),
                layer=cfg.diffmean_and_probe_layer,
            )
            if cfg.lora_adapter
            else None
        )
        additive_cfg = get_additive_direction_config(cfg)
        scorer_kwargs: dict[str, Any] = (
            {}
            if cluster_contract is None
            else {
                "cluster_contract": cluster_contract,
                "source_validation": source_validation,
            }
        )
        scorer = ModelBackedDualCompletionScorer(
            target_bundles=target_bundles,
            target_model_metas=target_metas,
            concepts_by_id=concepts_by_id,
            source_paths=source_paths,
            tokenizer=tokenizer,
            device=t.device(runtime_state.target_device()),
            diffmean_layer=cfg.diffmean_and_probe_layer,
            inactive_sampler=(
                runtime_state.inactive_sampler() if "encdec" in targets else None
            ),
            apply_max_act_scale=True,
            lora=lora,
            counter_direction_family=(
                additive_cfg.counter_direction_family
                if additive_cfg is not None
                else "diffmean"
            ),
            adv_paths=paths,
            **scorer_kwargs,
        )
        with profile_region("axbench_bpb_adv.score.score_table", rows=candidates.num_rows):
            return build_bpb_adv_score_table(
                candidates=candidates,
                context_segments_by_prefix_hash=context_segments_by_hash,
                tokenizer=tokenizer,
                token_bytes=token_bytes,
                max_seq_len_by_target={
                    target: bundle.max_seq_len for target, bundle in target_bundles.items()
                },
                score_pair=scorer,
                stats=stats,
                log_progress_every=cfg.log_progress_every,
            )
    finally:
        if cfg.offload_between_stages:
            runtime_state.offload_models()
