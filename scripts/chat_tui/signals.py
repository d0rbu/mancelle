"""Lazy AxBench concept fitting for the chat TUI steering menu."""

from __future__ import annotations

import threading
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from scripts.chat_tui.extraction import GemmaExtractionService
from scripts.chat_tui.steering import (
    SignalSource,
    SignalStore,
    SteeringSignal,
    contract_digest_for_setup,
)
from scripts.chat_tui.widgets.steer_menu import SignalState
from steering.eval.axbench_upstream.data import (
    ConceptTrainingTexts,
    load_metadata,
    select_training_texts,
)
from steering.eval.axbench_upstream.methods import MethodSetup, fit_dense_diffmean


class _ModelThread(Protocol):
    def submit(self, job: Callable[[], None]) -> None: ...


_ReadyCallback = Callable[[SteeringSignal], None]
_ErrorCallback = Callable[[str], None]


class AxbenchSignalsSource:
    """Eager concept metadata plus model-thread-lazy texts and DiffMean fits."""

    def __init__(
        self,
        *,
        setup: MethodSetup,
        prepared_dir: Path,
        cache_dir: Path,
        model_thread: _ModelThread,
        extraction_service: GemmaExtractionService,
        marshal: Callable[[Callable[[], None]], None],
    ) -> None:
        self._setup = setup
        self._prepared_dir = Path(prepared_dir)
        metadata = load_metadata(self._prepared_dir / "generate" / "metadata.jsonl")
        concepts: list[tuple[int, str]] = []
        for concept_id, row in enumerate(metadata):
            name = row.get("concept")
            if not isinstance(name, str) or not name:
                raise ValueError(f"metadata position {concept_id} has no nonempty concept")
            concepts.append((concept_id, name))
        self._concepts = tuple(concepts)
        self._names = dict(concepts)
        self._store = SignalStore(cache_dir, contract_digest_for_setup(setup))
        self._model_thread = model_thread
        self._extraction_service = extraction_service
        self._marshal = marshal
        self._texts: dict[int, ConceptTrainingTexts] | None = None
        self._states: dict[int, SignalState] = {
            concept_id: "unfitted" for concept_id, _name in concepts
        }
        self._signals: dict[int, SteeringSignal] = {}
        self._pending: dict[
            int,
            list[tuple[_ReadyCallback, _ErrorCallback]],
        ] = {}
        self._lock = threading.Lock()

    def list_concepts(self) -> tuple[tuple[int, str], ...]:
        return self._concepts

    def state_of(self, concept_id: int) -> SignalState:
        with self._lock:
            return self._states[concept_id]

    def signal_of(self, concept_id: int) -> SteeringSignal:
        with self._lock:
            if self._states[concept_id] != "cached":
                raise RuntimeError(f"signal for concept {concept_id} is not cached")
            return self._signals[concept_id]

    def ensure_fitted(
        self,
        concept_id: int,
        on_ready: _ReadyCallback,
        on_error: _ErrorCallback,
    ) -> None:
        with self._lock:
            state = self._states[concept_id]
            if state == "cached":
                signal = self._signals[concept_id]
            else:
                signal = None
                self._pending.setdefault(concept_id, []).append((on_ready, on_error))
                if state == "fitting":
                    return
                self._states[concept_id] = "fitting"

        if signal is not None:
            self._marshal(lambda: on_ready(signal))
            return
        self._model_thread.submit(lambda: self._fit(concept_id))

    def _fit(self, concept_id: int) -> None:
        try:
            texts_by_concept = self._texts
            if texts_by_concept is None:
                texts_by_concept = select_training_texts(
                    self._prepared_dir,
                    max_concepts=len(self._concepts),
                )
                self._texts = texts_by_concept
            try:
                texts = texts_by_concept[concept_id]
            except KeyError as exc:
                raise ValueError(f"AxBench training texts have no concept_id {concept_id}") from exc
            base_model, sae = self._extraction_service.extraction_stack()
            text_digest = self._text_digest(texts)
            signal = self._store.get_or_fit(
                concept_id,
                text_digest,
                lambda: fit_dense_diffmean(
                    self._setup,
                    base_model,
                    sae,
                    texts.positive,
                    texts.negative,
                    concept_id=concept_id,
                ),
                name=self._names[concept_id],
                source=SignalSource.AXBENCH_CONCEPT,
            )
        except Exception as exc:
            self._finish_error(concept_id, str(exc))
            return
        self._finish_ready(concept_id, signal)

    @staticmethod
    def _text_digest(texts: ConceptTrainingTexts) -> str:
        payload = "\x1f".join((*texts.positive, *texts.negative)).encode("utf-8")
        return sha256(payload).hexdigest()[:16]

    def _finish_ready(self, concept_id: int, signal: SteeringSignal) -> None:
        with self._lock:
            self._signals[concept_id] = signal
            self._states[concept_id] = "cached"
            callbacks = self._pending.pop(concept_id)
        for on_ready, _on_error in callbacks:
            self._marshal(lambda on_ready=on_ready: on_ready(signal))

    def _finish_error(self, concept_id: int, detail: str) -> None:
        with self._lock:
            self._states[concept_id] = "unfitted"
            callbacks = self._pending.pop(concept_id)
        for _on_ready, on_error in callbacks:
            self._marshal(lambda on_error=on_error: on_error(detail))


__all__ = ["AxbenchSignalsSource"]
