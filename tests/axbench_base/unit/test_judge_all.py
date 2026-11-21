"""Unit tests for Stage 5 ``judge_all`` orchestration (Task 34).

Covers:
  - Calls the injected judge client twice per generation row
    (once for concept relevance, once for fluency)
  - Resume semantics: existing judge_scores rows are skipped by
    ``generation_key``
  - Invalid tracking: ``None`` judge scores become ``-1`` sentinel rows
    with ``*_invalid = True``
  - Schema pinning: output parquet batches carry exactly the 13-column
    ``_JUDGE_SCORES_SCHEMA``
  - Unicode passthrough via ``_safe_substitute``
  - Empty generations dir yields an empty judge_scores dir
  - Warning / error thresholds are logged (never raised) when the
    per-row invalid rate exceeds ``cfg.judge.*_threshold``
  - ``cfg.run_tag is None`` and missing generations dir raise
    :class:`AxBenchBaseError`
  - A generations row pointing at a concept_id that doesn't exist in
    the caller-supplied ``concepts`` list raises with a clear message
  - The caller-owned client is used directly and closed on exit

All judge API calls are routed through a fake
:class:`AxBenchJudgeClient` substitute with a hardcoded ``score_many`` method.
No network access, no SQLite, no monkeypatching needed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest
import requests
from loguru import logger

from steering.eval.axbench_base import judge as judge_module
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig, JudgeConfig
from steering.eval.axbench_base.judge import (
    _JUDGE_SCORES_SCHEMA,
    AxBenchJudgeClient,
    _expected_generations_schema,
    _make_cache_key,
    judge_all,
)
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.templates import (
    _PLACEHOLDER_RE,
    assert_no_unfilled_placeholders,
    load_template,
)
from steering.eval.axbench_base.types import AxBenchBaseError, Concept
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config


def _as_client(fake: _FakeJudgeClient) -> AxBenchJudgeClient:
    """Adapter for injecting a fake duck-typed judge client.

    ``judge_all`` only touches ``client.score_many`` plus the context-manager
    protocol, so a structural fake is a legitimate substitute. Casting
    keeps type checkers happy without leaking ``# type: ignore`` noise at
    every call site.
    """
    return cast(AxBenchJudgeClient, fake)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeJudgeClient:
    """Minimal stand-in for :class:`AxBenchJudgeClient`.

    We don't inherit — the real class requires ``OPENROUTER_API_KEY`` and
    opens a SQLite file at construction time. ``judge_all`` accesses the
    client only via ``.score_many(prompts)`` plus the context-manager
    protocol.
    """

    def __init__(
        self,
        responses: list[tuple[int | None, int]],
        prompt_tag_fn: Any | None = None,
    ) -> None:
        self._responses = list(responses)
        self._prompt_tag_fn = prompt_tag_fn
        self.prompts: list[str] = []
        self.closed = False

    def score(self, prompt: str) -> tuple[int | None, int]:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError(
                "_FakeJudgeClient: unexpected extra score() call",
            )
        return self._responses.pop(0)

    def score_many(self, prompts: list[str]) -> list[tuple[int | None, int]]:
        return [self.score(prompt) for prompt in prompts]

    def __enter__(self) -> _FakeJudgeClient:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _FakeBatchJudgeClient(_FakeJudgeClient):
    """Fake judge client that exercises judge_all's score_many path."""

    def __init__(
        self,
        responses: list[tuple[int | None, int]],
        prompt_tag_fn: Any | None = None,
        on_score_many: Any | None = None,
    ) -> None:
        super().__init__(responses, prompt_tag_fn=prompt_tag_fn)
        self._on_score_many = on_score_many
        self.score_many_batch_sizes: list[int] = []

    def score_many(self, prompts: list[str]) -> list[tuple[int | None, int]]:
        self.score_many_batch_sizes.append(len(prompts))
        if self._on_score_many is not None:
            self._on_score_many(prompts)
        return [self.score(prompt) for prompt in prompts]


@contextmanager
def _capture_loguru(level: str = "WARNING") -> Iterator[StringIO]:
    """Capture loguru messages at or above ``level`` into a StringIO."""
    sink = StringIO()
    handler_id = logger.add(
        sink,
        format="{level} | {message}",
        level=level,
    )
    try:
        yield sink
    finally:
        logger.remove(handler_id)


def _cfg(
    tmp_path: Path,
    *,
    run_tag: str | None = "t5",
    invalid_warning_threshold: float = 0.01,
    invalid_drop_threshold: float = 0.05,
) -> AxBenchBaseConfig:
    judge_cfg = JudgeConfig(
        invalid_warning_threshold=invalid_warning_threshold,
        invalid_drop_threshold=invalid_drop_threshold,
    )
    return make_valid_config(
        run_tag=run_tag,
        output_root=None,
        judge=judge_cfg,
    )


def _concept(concept_id: int, description: str = "the letter e") -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id * 13,
        layer=8,
        sae_source="stub",
        description=description,
        corpus_count=100,
        description_source="stub",
    )


def _make_generation_row(
    *,
    concept_id: int = 0,
    method_name: str = "diffmean_decoder_only",
    mode: str = "positive",
    eval_seed: int = 42,
    coefficient: str = "1.5",
    generation_text: str = "the elephants eat every evening",
    target_model: str = "decoder_only",
    prefix_text: str = "once upon a time",
    token_count: int = 16,
    stop_reason: str = "bos_token",
) -> dict[str, Any]:
    return {
        "generation_key": (
            f"c{concept_id:04d}_{method_name}_{mode}"
            f"_s{eval_seed}_k{coefficient}"
        ),
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
    schema = _expected_generations_schema()
    with ParquetAppendWriter(
        out_dir=gen_dir, schema=schema, batch_size=10,
    ) as writer:
        for row in rows:
            writer.append(row)
    return gen_dir


def _write_judge_scores(run_tag: str, rows: list[dict[str, Any]]) -> Path:
    judge_dir = test_paths.default_paths(run_tag).judge_scores_dir()
    judge_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=judge_dir,
        schema=_JUDGE_SCORES_SCHEMA,
        batch_size=10,
    ) as writer:
        for row in rows:
            writer.append(row)
    return judge_dir


def _set_base_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))


# ===========================================================================
# Test 1: Calls judge twice per generation
# ===========================================================================


def test_calls_judge_twice_per_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """5 rows → client.score is called exactly 10 times (2 × rows)."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None  # help ty narrow Optional
    rows = [
        _make_generation_row(
            concept_id=i % 2,
            eval_seed=100 + i,
            coefficient=f"{i}.0",
            generation_text=f"sample {i}",
        )
        for i in range(5)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0, "concept zero"), _concept(1, "concept one")]
    fake = _FakeJudgeClient([(2, 1)] * 10)

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert len(fake.prompts) == 10
    # Prompts alternate: concept-relevance, fluency, concept-relevance, …
    assert "Rating: [[score]]" in fake.prompts[0]
    assert "[Concept Start]" in fake.prompts[0]
    assert "[Sentence Fragment Start]" in fake.prompts[1]
    assert "[Concept Start]" not in fake.prompts[1]


def test_score_many_path_writes_rows_in_generation_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Batched scoring keeps parquet rows in Stage 4 input order."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(eval_seed=200 + i, generation_text=f"text {i}")
        for i in range(3)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeBatchJudgeClient(
        [
            (2, 1), (1, 1),
            (0, 1), (1, 1),
            (1, 1), (2, 1),
        ],
    )

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert fake.score_many_batch_sizes == [6]
    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    )
    output_rows = table.to_pylist()
    assert [row["generation_key"] for row in output_rows] == [row["generation_key"] for row in rows]
    assert [row["concept_score"] for row in output_rows] == [2, 0, 1]
    assert [row["fluency_score"] for row in output_rows] == [1, 1, 2]


def test_score_many_wrong_score_count_raises_without_partial_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(eval_seed=210 + i, generation_text=f"text {i}")
        for i in range(2)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]

    class _WrongCountClient(_FakeJudgeClient):
        def score_many(self, prompts: list[str]) -> list[tuple[int | None, int]]:
            assert len(prompts) == 4
            return [(2, 1), (1, 1), (2, 1)]

    fake = _WrongCountClient([])

    with pytest.raises(AxBenchBaseError, match="wrong number of scores"):
        judge_all(cfg, concepts=concepts, client=_as_client(fake))

    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    )
    assert table.num_rows == 0


def test_score_many_path_crosses_chunks_and_preserves_prompt_pairing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Multiple score_many chunks keep row order and concept/fluency pairing."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path).replace(
        judge=JudgeConfig(
            max_concurrent=1,
            invalid_warning_threshold=1.0,
            invalid_drop_threshold=1.0,
        ),
    )
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(eval_seed=250 + i, generation_text=f"chunk text {i}")
        for i in range(7)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    checked_prompt_pairs: list[tuple[str, str]] = []

    def _check_prompt_pairing(prompts: list[str]) -> None:
        assert len(prompts) % 2 == 0
        for prompt_index in range(0, len(prompts), 2):
            concept_prompt = prompts[prompt_index]
            fluency_prompt = prompts[prompt_index + 1]
            assert "[Concept Start]" in concept_prompt
            assert "[Concept Start]" not in fluency_prompt
            assert "[Sentence Fragment Start]" in fluency_prompt
            checked_prompt_pairs.append((concept_prompt, fluency_prompt))

    fake = _FakeBatchJudgeClient(
        [(2, 1), (1, 1)] * len(rows),
        on_score_many=_check_prompt_pairing,
    )

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert fake.score_many_batch_sizes == [8, 6]
    assert len(checked_prompt_pairs) == len(rows)
    for i, (concept_prompt, fluency_prompt) in enumerate(checked_prompt_pairs):
        assert f"chunk text {i}" in concept_prompt
        assert f"chunk text {i}" in fluency_prompt

    output_rows = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    ).to_pylist()
    assert [row["generation_key"] for row in output_rows] == [row["generation_key"] for row in rows]


def test_judge_all_streams_generation_batches_without_load_existing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stage 5 must not materialize all Stage 4 generations via load_existing."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(eval_seed=270 + i, generation_text=f"stream {i}")
        for i in range(3)
    ]
    gen_dir = _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeBatchJudgeClient([(2, 1), (1, 1)] * len(rows))
    original_load_existing = judge_module.load_existing

    def _load_existing_guard(out_dir: Path, schema: pa.Schema | None = None) -> pa.Table:
        if out_dir == gen_dir:
            raise AssertionError("judge_all should stream generations in chunks")
        return original_load_existing(out_dir, schema=schema)

    monkeypatch.setattr(judge_module, "load_existing", _load_existing_guard)

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert len(fake.prompts) == 2 * len(rows)


def test_score_many_and_sequential_judge_all_parquet_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two score_many implementations emit the same ordered rows."""
    _set_base_dir(monkeypatch, tmp_path)
    rows = [
        _make_generation_row(eval_seed=300 + i, generation_text=f"sample {i}")
        for i in range(5)
    ]
    responses = [
        (i % 3, 1) if i % 4 else (None, 2)
        for i in range(2 * len(rows))
    ]
    concepts = [_concept(0)]

    seq_cfg = _cfg(tmp_path, run_tag="seq")
    assert seq_cfg.run_tag is not None
    _write_generations(seq_cfg.run_tag, rows)
    seq_fake = _FakeJudgeClient(list(responses))
    judge_all(seq_cfg, concepts=concepts, client=_as_client(seq_fake))
    seq_rows = load_existing(
        test_paths.default_paths(seq_cfg.run_tag).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    ).to_pylist()

    batch_cfg = _cfg(tmp_path, run_tag="batch")
    assert batch_cfg.run_tag is not None
    _write_generations(batch_cfg.run_tag, rows)
    batch_fake = _FakeBatchJudgeClient(list(responses))
    judge_all(batch_cfg, concepts=concepts, client=_as_client(batch_fake))
    batch_rows = load_existing(
        test_paths.default_paths(batch_cfg.run_tag).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    ).to_pylist()

    assert batch_rows == seq_rows


def test_judge_all_cache_warm_real_client_makes_zero_api_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real AxBenchJudgeClient uses score_many and honors warm cache hits."""
    _set_base_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-xyz")
    cfg = _cfg(
        tmp_path,
    ).replace(judge=JudgeConfig(requests_per_minute=0, max_concurrent=4))
    assert cfg.run_tag is not None
    row = _make_generation_row(generation_text="cached generation")
    _write_generations(cfg.run_tag, [row])
    concepts = [_concept(0)]

    concept_template = load_template("judge_concept_relevance.txt")
    fluency_template = load_template("judge_fluency.txt")
    concept_prompt = judge_module._safe_substitute(
        concept_template,
        CONCEPT=concepts[0].description,
        SENTENCE=row["generation_text"],
    )
    fluency_prompt = judge_module._safe_substitute(
        fluency_template,
        SENTENCE=row["generation_text"],
    )

    real = AxBenchJudgeClient(cfg.judge, cache_path=tmp_path / "warm.sqlite")
    try:
        real._cache_put(
            key=_make_cache_key(cfg.judge.model, concept_prompt),
            score=2,
            raw_response="Rating: [[2]]",
            attempts=1,
        )
        real._cache_put(
            key=_make_cache_key(cfg.judge.model, fluency_prompt),
            score=1,
            raw_response="Rating: [[1]]",
            attempts=1,
        )

        def _post(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("cache-warm judge_all should not call API")

        monkeypatch.setattr(
            requests.Session, "post",
            lambda _self, *args, **kwargs: _post(*args, **kwargs),
        )
        judge_all(cfg, concepts=concepts, client=real)
    finally:
        # judge_all closes injected clients; this keeps the test tolerant
        # if close() has already run.
        try:
            real.close()
        except Exception:
            pass

    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    )
    scored = table.to_pylist()[0]
    assert scored["concept_score"] == 2
    assert scored["fluency_score"] == 1


def test_judge_all_real_client_respects_max_concurrent_peak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stage 5's batched real client caps concurrent HTTP calls."""
    _set_base_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-xyz")
    cfg = _cfg(tmp_path).replace(
        judge=JudgeConfig(requests_per_minute=0, max_concurrent=2),
    )
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(eval_seed=400 + i, generation_text=f"peak {i}")
        for i in range(4)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]

    in_flight = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5.0)

    def _post(
        url: str,
        *,
        json: Any,
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        del url, json, headers, timeout
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        barrier.wait()
        with lock:
            in_flight -= 1
        return type(
            "Resp",
            (),
            {
                "status_code": 200,
                "text": "",
                "json": lambda self: {
                    "choices": [
                        {"message": {"content": "Explanation. Rating: [[1]]"}},
                    ],
                },
            },
        )()

    monkeypatch.setattr(
        requests.Session, "post",
        lambda _self, *args, **kwargs: _post(*args, **kwargs),
    )

    client = AxBenchJudgeClient(
        cfg.judge,
        cache_path=paths.AxBenchBasePaths.from_config(cfg).judge_cache(),
    )
    judge_all(cfg, concepts=concepts, client=client)

    assert peak == 2


def test_stage5_heartbeat_counts_written_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Heartbeat progress is row-based and keeps rate/ETA fields."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path).replace(
        judge=JudgeConfig(requests_per_minute=0, max_concurrent=1),
    )
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(eval_seed=500 + i, generation_text=f"hb {i}")
        for i in range(2)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    clock = {"now": 1000.0}

    def _monotonic() -> float:
        return clock["now"]

    def _advance(_prompts: list[str]) -> None:
        clock["now"] += 31.0

    monkeypatch.setattr(judge_module.time, "monotonic", _monotonic)
    fake = _FakeBatchJudgeClient([(2, 1)] * 4, on_score_many=_advance)

    with _capture_loguru(level="INFO") as sink:
        judge_all(cfg, concepts=concepts, client=_as_client(fake))

    captured = sink.getvalue()
    assert "[Stage 5] judge progress: 1/2" in captured
    assert "rate=" in captured
    assert "eta≈" in captured


# ===========================================================================
# Test 2: Unfilled placeholders raise assertion
# ===========================================================================


def test_unfilled_placeholders_raise_assertion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A template with a bogus placeholder still in it after substitution
    must trip the assertion guard."""
    _set_base_dir(monkeypatch, tmp_path)

    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [_make_generation_row()]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(2, 1), (2, 1)])

    # Replace the cached concept-relevance template with a corrupt one
    # containing an extra {{MYSTERY}} placeholder that substitution won't
    # fill. load_template uses functools.lru_cache, so we poke the cache.
    corrupt = load_template("judge_concept_relevance.txt")
    assert "{{MYSTERY}}" not in corrupt
    corrupt_text = corrupt + "\n\nUnfilled: {{MYSTERY}}"
    monkeypatch.setattr(
        "steering.eval.axbench_base.judge.load_template",
        lambda name: (
            corrupt_text
            if name == "judge_concept_relevance.txt"
            else load_template(name)
        ),
    )

    with pytest.raises(AxBenchBaseError, match="unfilled"):
        judge_all(cfg, concepts=concepts, client=_as_client(fake))


# ---------------------------------------------------------------------------
# Template placeholder detection (direct tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "{{CONCEPT}}",
        "{{SENTENCE}}",
        "{{PREFIX_TEXT}}",
        "{{ABC_DEF_GHI}}",
        "prefix {{KEY}} suffix",
        "{{A}}{{B}}",
        # Digits-after-first-char: forward-looking for ``{{SLOT_1}}`` /
        # ``{{ROUND_2_DESC}}``-style placeholders a future template
        # might introduce. The stricter ``[A-Z_]+`` pattern used before
        # the widening would silently miss these.
        "{{SLOT_1}}",
        "{{ROUND_2_DESC}}",
        "{{A1}}",
    ],
)
def test_template_placeholder_regex_matches_upper_snake(prompt: str) -> None:
    """The regex fires on the canonical ``{{UPPER_SNAKE_DIGITS}}``
    placeholder shape. Current templates under ``prompts/`` only use
    letter+underscore keys, but the regex allows digits after the
    first character so a new template adding ``{{SLOT_1}}`` wouldn't
    silently bypass the guard.
    """
    assert _PLACEHOLDER_RE.search(prompt) is not None


@pytest.mark.parametrize(
    "prompt",
    [
        "no braces at all",
        "{{lowercase}}",          # lowercase is not a placeholder today
        "{{Mixed_Case}}",         # mixed case
        "{{ VIN Vin}}",           # space + mixed (the overnight-run garbled case)
        "{{}}",                   # empty body
        "{{AB",                   # unclosed
        "{{ {}}",                 # garbage
        "plain {{ prose }} here", # braces with arbitrary prose inside
        # Digit-first is NOT a valid placeholder: ``{{0}}`` could appear
        # naturally in tokeniser output / JSON-like prose; placeholder
        # keys must start with a letter or underscore to avoid clashing.
        "{{0}}",
        "{{1_KEY}}",
    ],
)
def test_template_placeholder_regex_does_not_match_non_templates(
    prompt: str,
) -> None:
    """The regex must NOT fire on garbled generation text. Pins the
    false-positive fix for commit 10ddba6 at the regex level so a
    change to the regex without a corresponding test would be caught.
    """
    assert _PLACEHOLDER_RE.search(prompt) is None


def test_assert_no_unfilled_placeholders_accepts_clean_prompt() -> None:
    """A prompt with no placeholders passes silently."""
    assert_no_unfilled_placeholders("plain prompt", label="x")


def test_assert_no_unfilled_placeholders_error_includes_label() -> None:
    """The error message prefixes the supplied label so the caller
    can identify which of the two per-row templates drifted."""
    with pytest.raises(AxBenchBaseError, match="my-label"):
        assert_no_unfilled_placeholders("{{UNFILLED}}", label="my-label")


def test_assert_no_unfilled_placeholders_error_shows_window() -> None:
    """The error message includes the 60-char window starting at the
    first match so operators can see which placeholder drifted without
    dumping the entire prompt."""
    # 30 chars of filler, then the placeholder, then more filler.
    prompt = "abc" * 10 + "{{DRIFT_KEY}}" + "xyz" * 30
    with pytest.raises(AxBenchBaseError) as excinfo:
        assert_no_unfilled_placeholders(prompt, label="lbl")
    msg = str(excinfo.value)
    assert "{{DRIFT_KEY}}" in msg
    # The window is 60 chars; make sure the full tail isn't dumped.
    assert "xyz" * 30 not in msg


def test_assert_no_unfilled_placeholders_survives_python_O() -> None:
    """The guard must be an explicit ``raise``, not a bare ``assert``, so
    ``python -O`` doesn't strip it from bytecode. Under ``-O``, a bare
    ``assert`` becomes a no-op and an unfilled ``{{CONCEPT}}`` template
    variable would reach OpenRouter — real money for a garbage judgment.

    We can't literally run the test under ``-O`` from within a normal
    pytest invocation, so we pin the mechanism: the function's bytecode
    contains a ``RAISE_VARARGS`` opcode (the ``raise`` statement), not
    an ``ASSERT`` opcode pattern. If a future refactor reverts to
    ``assert match is None, ...``, this test catches it.
    """
    import dis

    code = assert_no_unfilled_placeholders.__code__
    opnames = [instr.opname for instr in dis.get_instructions(code)]
    assert "RAISE_VARARGS" in opnames, (
        "Unfilled-placeholder guard must use `raise`, not bare `assert` "
        "(compiled out under python -O). Found opcodes: "
        f"{set(opnames)}"
    )


def test_generation_text_with_literal_braces_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A garbled generation containing literal ``{{`` must not false-trip
    the unfilled-placeholder guard. Real SAE-steered runs occasionally
    emit ``{{ Vin... [Sentence Fragment End`` and the like; those are
    legitimate generations, not template drift.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(
            generation_text=(
                "{{ Vin Vin opio holid powd\n[Sentence Fragment End"
            ),
        ),
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(2, 1), (2, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert len(fake.prompts) == 2
    assert "{{ Vin" in fake.prompts[0]
    assert "{{ Vin" in fake.prompts[1]


# ===========================================================================
# Test 3: Resume skips already-judged rows
# ===========================================================================


def test_resume_skips_already_judged_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """3 of 5 rows already judged → only 2 × 2 = 4 new API calls."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    rows = [
        _make_generation_row(
            concept_id=0,
            eval_seed=100 + i,
            coefficient=f"{i}.0",
            generation_text=f"text {i}",
        )
        for i in range(5)
    ]
    _write_generations(cfg.run_tag, rows)

    # Pre-populate judge_scores with 3 rows (the first 3 generation_keys)
    judge_dir = paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    judge_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=judge_dir,
        schema=_JUDGE_SCORES_SCHEMA,
        batch_size=10,
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

    # Only the 2 new rows got 2 calls each = 4 prompts
    assert len(fake.prompts) == 4
    # Verify that the new rows were appended (8 total, not 5)
    table = load_existing(judge_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 5  # 3 pre-existing + 2 new
    # generation_keys form a set
    keys = set(table.column("generation_key").to_pylist())
    assert len(keys) == 5


def test_resume_requires_existing_scores_to_match_current_judge_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path).replace(judge=JudgeConfig(model="new-judge"))
    assert cfg.run_tag is not None
    row = _make_generation_row(generation_text="stale judge model")
    _write_generations(cfg.run_tag, [row])
    _write_judge_scores(
        cfg.run_tag,
        [
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
                "judge_model": "old-judge",
            }
        ],
    )

    with pytest.raises(AxBenchBaseError, match="old-judge"):
        judge_all(cfg, concepts=[_concept(0)], client=_as_client(_FakeJudgeClient([])))


def test_resume_requires_existing_prompt_fingerprint_to_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    row = _make_generation_row(generation_text="same generation key")
    _write_generations(cfg.run_tag, [row])
    judge_dir = _write_judge_scores(
        cfg.run_tag,
        [
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
            }
        ],
    )
    judge_module._write_judge_resume_manifest(
        judge_dir,
        judge_model=cfg.judge.model,
        prompt_fingerprint="stale-fingerprint",
    )

    with pytest.raises(AxBenchBaseError, match="prompt_fingerprint"):
        judge_all(cfg, concepts=[_concept(0)], client=_as_client(_FakeJudgeClient([])))


def test_fully_resumed_run_uses_no_client_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    row = _make_generation_row(generation_text="already judged")
    _write_generations(cfg.run_tag, [row])
    _write_judge_scores(
        cfg.run_tag,
        [
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
            }
        ],
    )

    fake = _FakeJudgeClient([])
    out_dir = judge_all(
        cfg,
        concepts=[_concept(0)],
        client=_as_client(fake),
    )

    assert out_dir == paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    assert fake.prompts == []
    assert fake.closed
    table = load_existing(out_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 1


# ===========================================================================
# Test 4: Invalid flag tracked per row
# ===========================================================================


def test_invalid_flag_tracked_per_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concept score = None → concept_invalid=True, concept_score=-1."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, invalid_warning_threshold=0.99)
    assert cfg.run_tag is not None
    rows = [_make_generation_row()]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    # First (concept) → None with 2 attempts. Second (fluency) → valid.
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


# ===========================================================================
# Test 5: Schema matches spec
# ===========================================================================


def test_judge_scores_schema_matches_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The on-disk parquet carries exactly the 13 expected columns
    with the expected types."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, invalid_warning_threshold=0.99)
    assert cfg.run_tag is not None
    rows = [_make_generation_row()]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(2, 1), (1, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    schema = _JUDGE_SCORES_SCHEMA
    assert schema.names == [
        "generation_key",
        "concept_id",
        "method_name",
        "mode",
        "eval_seed",
        "coefficient",
        "concept_score",
        "concept_attempts",
        "concept_invalid",
        "fluency_score",
        "fluency_attempts",
        "fluency_invalid",
        "judge_model",
    ]
    assert schema.field("generation_key").type == pa.string()
    assert schema.field("concept_id").type == pa.int32()
    assert schema.field("concept_score").type == pa.int32()
    assert schema.field("concept_invalid").type == pa.bool_()
    assert schema.field("coefficient").type == pa.string()

    table = load_existing(paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(), schema=schema)
    assert table.schema.equals(schema)


# ===========================================================================
# Test 6: Unicode passthrough
# ===========================================================================


def test_unicode_generation_text_passes_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unicode in generation text survives prompt building + substitution."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [
        _make_generation_row(
            generation_text="café résumé naïve 日本語",
        ),
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0, description="français é")]
    fake = _FakeJudgeClient([(2, 1), (2, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert len(fake.prompts) == 2
    assert "café résumé naïve 日本語" in fake.prompts[0]
    assert "français é" in fake.prompts[0]
    assert "café résumé naïve 日本語" in fake.prompts[1]


# ===========================================================================
# Test 7: Empty generations dir
# ===========================================================================


def test_empty_generations_dir_produces_empty_judge_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Zero generations → zero judge scores, empty dir, no client calls."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    # Create an empty generations dir (no batch files).
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)

    concepts = [_concept(0)]
    fake = _FakeJudgeClient([])  # zero responses expected

    out_dir = judge_all(cfg, concepts=concepts, client=_as_client(fake))

    assert out_dir == paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    assert out_dir.exists()
    assert len(fake.prompts) == 0
    table = load_existing(out_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 0


def test_empty_generations_dir_closes_supplied_client_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    paths.AxBenchBasePaths.from_config(cfg).generations_dir().mkdir(parents=True, exist_ok=True)

    fake = _FakeJudgeClient([])
    out_dir = judge_all(
        cfg,
        concepts=[_concept(0)],
        client=_as_client(fake),
    )

    assert out_dir == paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    assert out_dir.exists()
    assert fake.closed
    table = load_existing(out_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 0


# ===========================================================================
# Test 8: Warning threshold
# ===========================================================================


def test_invalid_rate_warning_logged_at_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Half-invalid rows with 0.01 threshold → WARNING logged."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(
        tmp_path,
        invalid_warning_threshold=0.01,
        invalid_drop_threshold=1.0,
    )
    assert cfg.run_tag is not None

    rows = [
        _make_generation_row(
            concept_id=0,
            eval_seed=100 + i,
            coefficient=f"{i}.0",
        )
        for i in range(4)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    # Rows 0 + 2 invalid on concept (None); rows 1 + 3 valid. Fluency all valid.
    # Total invalid = 2 / 8 calls = 0.25, above 0.01.
    fake = _FakeJudgeClient(
        [
            (None, 2), (2, 1),   # row 0
            (2, 1), (2, 1),       # row 1
            (None, 2), (2, 1),   # row 2
            (2, 1), (2, 1),       # row 3
        ],
    )

    with _capture_loguru(level="WARNING") as sink:
        judge_all(cfg, concepts=concepts, client=_as_client(fake))
    captured = sink.getvalue()
    assert "WARNING" in captured
    assert "invalid" in captured.lower()


def test_resumed_invalid_rows_are_included_in_threshold_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _cfg(
        tmp_path,
        invalid_warning_threshold=0.01,
        invalid_drop_threshold=1.0,
    )
    assert cfg.run_tag is not None
    row = _make_generation_row(generation_text="already invalid")
    _write_generations(cfg.run_tag, [row])
    _write_judge_scores(
        cfg.run_tag,
        [
            {
                "generation_key": row["generation_key"],
                "concept_id": row["concept_id"],
                "method_name": row["method_name"],
                "mode": row["mode"],
                "eval_seed": row["eval_seed"],
                "coefficient": row["coefficient"],
                "concept_score": -1,
                "concept_attempts": 2,
                "concept_invalid": True,
                "fluency_score": 2,
                "fluency_attempts": 1,
                "fluency_invalid": False,
                "judge_model": cfg.judge.model,
            }
        ],
    )

    with _capture_loguru(level="WARNING") as sink:
        judge_all(
            cfg,
            concepts=[_concept(0)],
            client=_as_client(_FakeJudgeClient([])),
        )

    captured = sink.getvalue()
    assert "WARNING" in captured
    assert "invalid" in captured.lower()


# ===========================================================================
# Test 9: Error threshold
# ===========================================================================


def test_invalid_rate_error_logged_at_drop_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """High-invalid rate crosses drop threshold → ERROR logged, no raise."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(
        tmp_path,
        invalid_warning_threshold=0.01,
        invalid_drop_threshold=0.05,
    )
    assert cfg.run_tag is not None

    rows = [
        _make_generation_row(
            concept_id=0,
            eval_seed=100 + i,
            coefficient=f"{i}.0",
        )
        for i in range(5)
    ]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    # 4/5 rows invalid on both concept + fluency → 8 invalid out of 10
    fake = _FakeJudgeClient(
        [
            (None, 2), (None, 2),   # row 0 — both invalid
            (None, 2), (None, 2),   # row 1 — both invalid
            (None, 2), (None, 2),   # row 2 — both invalid
            (None, 2), (None, 2),   # row 3 — both invalid
            (2, 1), (2, 1),          # row 4 — both valid
        ],
    )

    with _capture_loguru(level="ERROR") as sink:
        # Must not raise.
        out = judge_all(cfg, concepts=concepts, client=_as_client(fake))
    captured = sink.getvalue()
    assert "ERROR" in captured
    assert "invalid" in captured.lower()
    assert out.exists()


# ===========================================================================
# Test 10: run_tag not set
# ===========================================================================


def test_run_tag_not_set_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        judge_all(
            cfg,
            concepts=[_concept(0)],
            client=_as_client(_FakeJudgeClient([])),
        )


# ===========================================================================
# Test 11: generations dir missing
# ===========================================================================


def test_generations_dir_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    # Deliberately do NOT create the generations dir.
    fake = _FakeJudgeClient([])
    with pytest.raises(AxBenchBaseError, match="generations"):
        judge_all(cfg, concepts=[_concept(0)], client=_as_client(fake))


# ===========================================================================
# Test 12: concept_id not in lookup
# ===========================================================================


def test_concept_not_in_lookup_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A generations row with concept_id=999 but concepts=[0] → clear error."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [_make_generation_row(concept_id=999)]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(2, 1), (2, 1)])

    with pytest.raises(AxBenchBaseError, match="999"):
        judge_all(cfg, concepts=concepts, client=_as_client(fake))


# ===========================================================================
# Test 13: Supplied client lifecycle
# ===========================================================================


def test_supplied_client_used_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The caller-owned client scores every prompt and closes on exit."""
    _set_base_dir(monkeypatch, tmp_path)

    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [_make_generation_row()]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(2, 1), (2, 1)])

    judge_all(cfg, concepts=concepts, client=_as_client(fake))
    assert len(fake.prompts) == 2
    assert fake.closed  # context-manager exit ran


# ===========================================================================
# Test 14: Batch-index resume cursor survives partial crashes
# ===========================================================================


def test_new_rows_appended_to_next_batch_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing batches on disk → new rows land in the next batch_NNNNNN.
    Enforces that the ParquetAppendWriter scan correctly picks up where
    prior runs left off."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    rows = [
        _make_generation_row(
            eval_seed=100 + i, coefficient=f"{i}.0",
        )
        for i in range(3)
    ]
    _write_generations(cfg.run_tag, rows)

    judge_dir = paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir()
    judge_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=judge_dir,
        schema=_JUDGE_SCORES_SCHEMA,
        batch_size=1,
    ) as writer:
        writer.append(
            {
                "generation_key": rows[0]["generation_key"],
                "concept_id": rows[0]["concept_id"],
                "method_name": rows[0]["method_name"],
                "mode": rows[0]["mode"],
                "eval_seed": rows[0]["eval_seed"],
                "coefficient": rows[0]["coefficient"],
                "concept_score": 2,
                "concept_attempts": 1,
                "concept_invalid": False,
                "fluency_score": 2,
                "fluency_attempts": 1,
                "fluency_invalid": False,
                "judge_model": cfg.judge.model,
            },
        )

    pre_files = sorted(judge_dir.glob("batch_*.parquet"))
    assert [f.name for f in pre_files] == ["batch_000000.parquet"]

    concepts = [_concept(0)]
    fake = _FakeJudgeClient([(1, 1), (1, 1), (2, 1), (2, 1)])
    judge_all(cfg, concepts=concepts, client=_as_client(fake))

    post_files = sorted(judge_dir.glob("batch_*.parquet"))
    assert post_files[0].name == "batch_000000.parquet"
    assert len(post_files) >= 2  # new batch file(s) appended
    table = load_existing(judge_dir, schema=_JUDGE_SCORES_SCHEMA)
    assert table.num_rows == 3


# ===========================================================================
# Test 16: expected generations schema helper is stable
# ===========================================================================


def test_expected_generations_schema_has_minimum_columns() -> None:
    """The tentative generations schema exposes the columns judge_all
    dereferences. Stage 4 must emit at least these."""
    schema = _expected_generations_schema()
    names = schema.names
    # judge_all reads these:
    assert "generation_key" in names
    assert "concept_id" in names
    assert "method_name" in names
    assert "mode" in names
    assert "eval_seed" in names
    assert "coefficient" in names
    assert "generation_text" in names
    assert schema.field("generation_key").type == pa.string()
    assert schema.field("concept_id").type == pa.int32()
    assert schema.field("generation_text").type == pa.string()


# ===========================================================================
# Test 17: No rows → no threshold log lines
# ===========================================================================


def test_empty_run_does_not_log_threshold_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Division-by-zero safety: a run with 0 rows must not log a WARNING
    or ERROR about invalid rates."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    concepts = [_concept(0)]
    fake = _FakeJudgeClient([])

    with _capture_loguru(level="WARNING") as sink:
        judge_all(cfg, concepts=concepts, client=_as_client(fake))
    captured = sink.getvalue()
    assert "invalid" not in captured.lower()


# ===========================================================================
# Test 18: Real AxBenchJudgeClient type can still be used by callers
# ===========================================================================


def test_injection_accepts_real_client_type_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``client: AxBenchJudgeClient | None`` accepts an actual instance.
    We avoid the OpenRouter network by monkey-patching ``score_many``."""
    _set_base_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-xyz")
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None
    rows = [_make_generation_row()]
    _write_generations(cfg.run_tag, rows)
    concepts = [_concept(0)]

    real = AxBenchJudgeClient(
        cfg.judge, cache_path=tmp_path / "j.sqlite",
    )
    try:
        monkeypatch.setattr(
            real, "score_many", lambda _prompts: [(2, 1), (1, 1)],
        )
        judge_all(cfg, concepts=concepts, client=real)
    finally:
        try:
            real.close()
        except Exception:
            pass

    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    )
    assert table.num_rows == 1
    row_dict = table.to_pylist()[0]
    assert row_dict["concept_score"] == 2
    assert row_dict["fluency_score"] == 1
