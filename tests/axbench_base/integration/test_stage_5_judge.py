"""Stage 5 ``judge_all`` end-to-end integration tests.

Exercises :func:`steering.eval.axbench_base.judge.judge_all` with a
duck-typed fake judge client. The fake supports:

* A canned per-call response queue (concept + fluency alternating).
* ``None`` responses to exercise the invalid-sentinel path.
* Ordered batch scoring and call-count tracking so we can assert 2 API calls per row.

These tests use the production generations schema to write Stage 4
rows into a tmp directory-dataset, then call ``judge_all`` against
an injected client. No real OpenRouter traffic.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, cast

import pytest

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    JudgeConfig,
)
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import (
    _JUDGE_SCORES_SCHEMA,
    AxBenchJudgeClient,
    judge_all,
)
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.types import Concept
from tests.axbench_base import paths as test_paths

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fake judge client
# ---------------------------------------------------------------------------


class _FakeJudgeClient:
    """Structural fake used in place of :class:`AxBenchJudgeClient`.

    The real class requires ``OPENROUTER_API_KEY`` at construction
    time and opens a SQLite cache, neither of which we need for these
    integration tests. ``judge_all`` calls ``score_many`` plus the
    context-manager protocol so a duck-typed fake works cleanly.
    """

    def __init__(self, responses: list[tuple[int | None, int]]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.closed = False

    def score(self, prompt: str) -> tuple[int | None, int]:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError(
                f"_FakeJudgeClient: unexpected extra score() call "
                f"(prompt={prompt[:80]!r})",
            )
        return self._responses.pop(0)

    def score_many(self, prompts: list[str]) -> list[tuple[int | None, int]]:
        return [self.score(prompt) for prompt in prompts]

    def __enter__(self) -> _FakeJudgeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


def _as_client(fake: _FakeJudgeClient) -> AxBenchJudgeClient:
    return cast(AxBenchJudgeClient, fake)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    tmp_path: Path,
    *,
    run_tag: str = "stage5_int",
    invalid_warning_threshold: float = 0.5,
    invalid_drop_threshold: float = 0.9,
) -> AxBenchBaseConfig:
    judge_cfg = JudgeConfig(
        invalid_warning_threshold=invalid_warning_threshold,
        invalid_drop_threshold=invalid_drop_threshold,
    )
    return AxBenchBaseConfig(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        ),
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=str(tmp_path / "eval_out"),
        run_tag=run_tag,
        judge=judge_cfg,
    )


def _concept(concept_id: int = 0, description: str = "the color blue") -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id * 11,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=description,
        corpus_count=100,
        description_source="stub",
    )


def _make_generation_row(
    *,
    generation_key: str,
    concept_id: int = 0,
    method_name: str = "diffmean_decoder_only",
    mode: str = "positive",
    eval_seed: int = 0,
    coefficient: str = "1.0",
    generation_text: str = "the elephants eat every evening",
    target_model: str = "decoder_only",
    prefix_text: str = "once upon a time",
    token_count: int = 16,
    stop_reason: str = "bos_token",
) -> dict[str, Any]:
    return {
        "generation_key": generation_key,
        "concept_id": concept_id,
        "method_name": method_name,
        "mode": mode,
        "eval_seed": eval_seed,
        "coefficient": coefficient,
        "generation_text": generation_text,
        "target_model": target_model,
        "prefix_text": prefix_text,
        "prompt_token_key": "tokens_sha256:" + "0" * 64,
        "prompt_token_count": 4,
        "seed_pool_fingerprint": "f" * 64,
        "token_count": token_count,
        "stop_reason": stop_reason,
    }


def _write_generations(run_tag: str, rows: list[dict[str, Any]]) -> Path:
    gen_dir = test_paths.default_paths(run_tag).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=100,
    ) as writer:
        for row in rows:
            writer.append(row)
    return gen_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_5_judge_mocked_all_rows_scored(
    tmp_run_root: Path,
) -> None:
    """5 rows with a mocked (1, 1) judge → every generation row has a
    corresponding judge_scores row with concept/fluency=1."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage5_mocked")
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(
            generation_key=f"k{i}",
            concept_id=i % 2,
            eval_seed=100 + i,
            coefficient=f"{i}.0",
            generation_text=f"sample {i}",
        )
        for i in range(5)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0, "concept zero"), _concept(1, "concept one")]
    fake = _FakeJudgeClient([(1, 1)] * 10)

    out_dir = judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert out_dir == paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    assert len(fake.prompts) == 10  # 2 calls per row
    table = load_existing(out_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 5
    for judge_row in table.to_pylist():
        assert judge_row["concept_score"] == 1
        assert judge_row["fluency_score"] == 1
        assert judge_row["concept_invalid"] is False
        assert judge_row["fluency_invalid"] is False


def test_stage_5_judge_invalid_flag_tracked(
    tmp_run_root: Path,
) -> None:
    """A ``(None, 2)`` concept response must become
    ``concept_score=-1`` + ``concept_invalid=True`` in the output row,
    with ``concept_attempts=2`` surfaced."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage5_invalid")
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(
            generation_key="only_row",
            concept_id=0,
        ),
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    # concept → None (2 attempts); fluency → 2 (1 attempt)
    fake = _FakeJudgeClient([(None, 2), (2, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    )
    assert table.num_rows == 1
    row_dict = table.to_pylist()[0]
    assert row_dict["concept_invalid"] is True
    assert row_dict["concept_score"] == -1
    assert row_dict["concept_attempts"] == 2
    assert row_dict["fluency_invalid"] is False
    assert row_dict["fluency_score"] == 2
    assert row_dict["fluency_attempts"] == 1


def test_stage_5_judge_unicode_prompts(
    tmp_run_root: Path,
) -> None:
    """Unicode in concept descriptions and generation text flows
    through the judge prompts unchanged."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage5_unicode")
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(
            generation_key="unicode_row",
            concept_id=0,
            generation_text="café résumé naïve 日本語 🔥",
        ),
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0, description="français é café")]
    fake = _FakeJudgeClient([(2, 1), (2, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert len(fake.prompts) == 2
    # Concept-relevance prompt carries both the description and the text.
    assert "café résumé naïve 日本語 🔥" in fake.prompts[0]
    assert "français é café" in fake.prompts[0]
    # Fluency prompt carries only the sentence, not the concept.
    assert "café résumé naïve 日本語 🔥" in fake.prompts[1]
    assert "français é café" not in fake.prompts[1]


def test_stage_5_judge_respects_both_templates(
    tmp_run_root: Path,
) -> None:
    """Exactly 2 ``score`` calls per row: one concept-relevance
    template, one fluency template. We can tell them apart by the
    distinct marker strings every template carries."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage5_templates")
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(
            generation_key=f"r{i}",
            concept_id=0,
            eval_seed=i,
            generation_text=f"generation {i}",
        )
        for i in range(3)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(2, 1)] * 6)

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    # 3 rows × 2 judge axes = 6 prompts.
    assert len(fake.prompts) == 6

    # Per-row prompts arrive in pairs: (concept, fluency, concept, fluency...).
    for i, concept_prompt in enumerate(fake.prompts[0::2]):
        # The concept-relevance template carries the ``[Concept Start]``
        # marker from the prompt file; fluency does not.
        assert "[Concept Start]" in concept_prompt
        assert f"generation {i}" in concept_prompt

    for i, fluency_prompt in enumerate(fake.prompts[1::2]):
        # The fluency template carries a ``[Sentence Fragment Start]``
        # marker but does NOT include a concept section.
        assert "[Sentence Fragment Start]" in fluency_prompt
        assert "[Concept Start]" not in fluency_prompt
        assert f"generation {i}" in fluency_prompt


def test_stage_5_judge_resume_skips_already_judged_rows(
    tmp_run_root: Path,
) -> None:
    """3 of 5 rows already in judge_scores → only the 2 missing rows
    get new judge calls (2 × 2 = 4 prompts total)."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage5_resume")
    assert cfg.run_tag is not None

    rows = [
        _make_generation_row(
            generation_key=f"row_{i}",
            concept_id=0,
            eval_seed=100 + i,
            generation_text=f"text {i}",
        )
        for i in range(5)
    ]
    _write_generations(cfg.run_tag, rows)

    # Pre-populate judge_scores_dir with 3 already-judged rows.
    judge_dir = paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    judge_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=judge_dir,
        schema=_JUDGE_SCORES_SCHEMA,
        batch_size=100,
    ) as writer:
        for row in rows[:3]:
            writer.append(
                {
                    "generation_key": row["generation_key"],
                    "concept_id": row["concept_id"],
                    "method_name": row["method_name"],
                    "mode": row["mode"],
                    "eval_seed": row["eval_seed"],
                    "coefficient": row["coefficient"],
                    "concept_score": 2,
                    "concept_attempts": 1,
                    "concept_invalid": False,
                    "fluency_score": 2,
                    "fluency_attempts": 1,
                    "fluency_invalid": False,
                    "judge_model": cfg.judge.model,
                },
            )

    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(1, 1), (1, 1), (2, 2), (1, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    # Only 2 new rows → 4 prompts.
    assert len(fake.prompts) == 4

    table = load_existing(judge_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 5
    keys = set(table.column("generation_key").to_pylist())
    assert len(keys) == 5
    assert keys == {row["generation_key"] for row in rows}
