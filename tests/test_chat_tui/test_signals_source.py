"""CPU-only contracts for the lazy AxBench steering-signal source."""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch as t

from steering.eval.axbench_upstream.data import ConceptTrainingTexts


class ManualModelThread:
    def __init__(self) -> None:
        self.jobs: list[Callable[[], None]] = []

    def submit(self, job: Callable[[], None]) -> None:
        self.jobs.append(job)

    def run_next(self) -> None:
        self.jobs.pop(0)()


class ExtractionStub:
    def __init__(self) -> None:
        self.calls = 0
        self.base_model = object()
        self.sae = object()

    def extraction_stack(self) -> tuple[object, object]:
        self.calls += 1
        return self.base_model, self.sae


def _write_metadata(prepared_dir: Path) -> None:
    generate_dir = prepared_dir / "generate"
    generate_dir.mkdir(parents=True)
    rows = [
        {
            "concept_id": concept_id,
            "concept": name,
            "concept_genres_map": {name: [genre]},
        }
        for concept_id, name, genre in (
            (0, "alpha", "letters"),
            (1, "bravo", "letters"),
            (2, "calm", "traits"),
        )
    ]
    (generate_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _texts(*, changed: bool = False) -> dict[int, ConceptTrainingTexts]:
    suffix = " changed" if changed else ""
    return {
        concept_id: ConceptTrainingTexts(
            concept_id=concept_id,
            concept=name,
            genre=genre,
            positive=(f"{name} positive one{suffix}", f"{name} positive two"),
            negative=(f"{name} negative",),
        )
        for concept_id, name, genre in (
            (0, "alpha", "letters"),
            (1, "bravo", "letters"),
            (2, "calm", "traits"),
        )
    }


@pytest.fixture
def prepared_dir(tmp_path: Path) -> Path:
    path = tmp_path / "prepared"
    _write_metadata(path)
    return path


@pytest.fixture
def signal_module(monkeypatch: pytest.MonkeyPatch):
    from scripts.chat_tui import signals

    monkeypatch.setattr(signals, "contract_digest_for_setup", lambda _setup: "contract12345678")
    return signals


def _marshal_recorder(calls: list[str]) -> Callable[[Callable[[], None]], None]:
    def marshal(callback: Callable[[], None]) -> None:
        calls.append("marshal")
        callback()

    return marshal


def test_metadata_is_eager_and_first_fit_loads_texts_once_then_caches(
    prepared_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_module,
) -> None:
    text_calls: list[tuple[Path, int]] = []
    fit_calls: list[tuple[Any, ...]] = []
    model_thread = ManualModelThread()
    extraction = ExtractionStub()
    marshals: list[str] = []

    def select(actual_dir: Path, *, max_concepts: int):
        text_calls.append((actual_dir, max_concepts))
        return _texts()

    def fit(*args: Any, concept_id: int) -> t.Tensor:
        fit_calls.append((*args, concept_id))
        return t.tensor([-0.5, 0.0, 0.5], dtype=t.float32)

    monkeypatch.setattr(signal_module, "select_training_texts", select)
    monkeypatch.setattr(signal_module, "fit_dense_diffmean", fit)
    source = signal_module.AxbenchSignalsSource(
        setup=object(),
        prepared_dir=prepared_dir,
        cache_dir=tmp_path / "cache",
        model_thread=model_thread,
        extraction_service=extraction,
        marshal=_marshal_recorder(marshals),
    )

    assert source.list_concepts() == ((0, "alpha"), (1, "bravo"), (2, "calm"))
    assert source.state_of(0) == "unfitted"
    ready = []
    errors: list[str] = []
    source.ensure_fitted(0, ready.append, errors.append)
    assert source.state_of(0) == "fitting"
    assert text_calls == []

    model_thread.run_next()

    assert source.state_of(0) == "cached"
    assert ready == [source.signal_of(0)]
    assert errors == []
    assert text_calls == [(prepared_dir, 3)]
    assert extraction.calls == 1
    assert len(fit_calls) == 1
    assert fit_calls[0][0] is not None
    assert fit_calls[0][1:3] == (extraction.base_model, extraction.sae)
    assert fit_calls[0][3:5] == (
        ("alpha positive one", "alpha positive two"),
        ("alpha negative",),
    )
    assert fit_calls[0][5] == 0

    source.ensure_fitted(0, ready.append, errors.append)

    assert ready == [source.signal_of(0), source.signal_of(0)]
    assert model_thread.jobs == []
    assert text_calls == [(prepared_dir, 3)]
    assert len(fit_calls) == 1

    source.ensure_fitted(1, ready.append, errors.append)
    assert source.state_of(1) == "fitting"
    model_thread.run_next()

    assert source.state_of(1) == "cached"
    assert text_calls == [(prepared_dir, 3)]
    assert len(fit_calls) == 2
    assert marshals == ["marshal", "marshal", "marshal"]


def test_fit_error_is_marshaled_and_resets_state(
    prepared_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_module,
) -> None:
    model_thread = ManualModelThread()
    monkeypatch.setattr(
        signal_module,
        "select_training_texts",
        lambda _path, *, max_concepts: _texts(),
    )

    def fail(*_args: Any, **_kwargs: Any) -> t.Tensor:
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(signal_module, "fit_dense_diffmean", fail)
    marshals: list[str] = []
    errors: list[str] = []
    source = signal_module.AxbenchSignalsSource(
        setup=object(),
        prepared_dir=prepared_dir,
        cache_dir=tmp_path / "cache",
        model_thread=model_thread,
        extraction_service=ExtractionStub(),
        marshal=_marshal_recorder(marshals),
    )

    source.ensure_fitted(1, lambda _signal: pytest.fail("unexpected ready callback"), errors.append)
    assert source.state_of(1) == "fitting"
    model_thread.run_next()

    assert source.state_of(1) == "unfitted"
    assert errors == ["fit exploded"]
    assert marshals == ["marshal"]

    monkeypatch.setattr(
        signal_module,
        "fit_dense_diffmean",
        lambda *_args, **_kwargs: t.tensor([-0.5, 0.5], dtype=t.float32),
    )
    ready = []
    source.ensure_fitted(1, ready.append, errors.append)
    assert source.state_of(1) == "fitting"
    model_thread.run_next()
    assert source.state_of(1) == "cached"
    assert ready == [source.signal_of(1)]


def test_text_digest_is_order_sensitive_and_changed_texts_refit(
    prepared_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_module,
) -> None:
    fit_calls: list[int] = []

    def fit(*_args: Any, concept_id: int) -> t.Tensor:
        fit_calls.append(concept_id)
        return t.tensor([-0.25, 0.25], dtype=t.float32)

    monkeypatch.setattr(signal_module, "fit_dense_diffmean", fit)
    cache_dir = tmp_path / "cache"
    identities: list[str] = []

    for changed in (False, True):
        selected = _texts(changed=changed)
        monkeypatch.setattr(
            signal_module,
            "select_training_texts",
            lambda _path, *, max_concepts, selected=selected: selected,
        )
        thread = ManualModelThread()
        source = signal_module.AxbenchSignalsSource(
            setup=object(),
            prepared_dir=prepared_dir,
            cache_dir=cache_dir,
            model_thread=thread,
            extraction_service=ExtractionStub(),
            marshal=lambda callback: callback(),
        )
        source.ensure_fitted(0, lambda signal: identities.append(signal.identity), pytest.fail)
        thread.run_next()

    original = _texts()[0]
    expected_digest = sha256(
        "\x1f".join((*original.positive, *original.negative)).encode("utf-8")
    ).hexdigest()[:16]
    assert identities[0].endswith(f"/0_{expected_digest}")
    assert identities[0] != identities[1]
    assert fit_calls == [0, 0]
    assert len(list(cache_dir.rglob("*.safetensors"))) == 2
