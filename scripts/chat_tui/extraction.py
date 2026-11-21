import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from scripts.chat_tui.setup import resolve_conditioning_seed, resolve_stats_sha
from steering.conditional.sampling import SamplingBundle
from steering.eval.axbench_upstream.methods import (
    MethodSetup,
    PreparedPromptContext,
    load_extraction_stack,
    prepare_prompt_context,
)


class ExtractionService(Protocol):
    """Prepare exact rendered prompt tokens for self-conditioned generation."""

    def prepare(self, token_ids: tuple[int, ...]) -> PreparedPromptContext: ...


class GemmaExtractionService:
    """Lazy, content-addressed prompt extraction backed by Gemma and its SAE."""

    def __init__(self, setup: MethodSetup, tokenizer: Any, device: str):
        self.setup = setup
        self._tokenizer = tokenizer
        self.device = device
        self._conditioning_seed = resolve_conditioning_seed(setup)
        self._stack: tuple[Any, Any] | None = None
        self._cache: dict[tuple[int, ...], PreparedPromptContext] = {}

    def _ensure_stack(self) -> tuple[Any, Any]:
        if self._stack is None:
            self._stack = load_extraction_stack(self.setup, self.device)
        return self._stack

    def extraction_stack(self) -> tuple[Any, Any]:
        """Return the shared (base_model, sae) pair, loading it on first use.

        Public accessor for consumers that fit concept signals on the same
        stack (the signals source); loading stays lazy and single-instance.
        """
        return self._ensure_stack()

    def prepare(self, token_ids: tuple[int, ...]) -> PreparedPromptContext:
        cached = self._cache.get(token_ids)
        if cached is not None:
            return cached

        base_model, sae = self._ensure_stack()
        selection_config = self.setup.prompt_pipeline.selection_config
        if not isinstance(selection_config, SamplingBundle):
            raise ValueError("checkpoint prompt conditioning must expose a sampling generator")
        assistant_end_id = self._tokenizer.encode_special("<assistant_end>")
        selection_config.rng.manual_seed(self._conditioning_seed)
        prepared = prepare_prompt_context(
            self.setup,
            base_model,
            sae,
            self._tokenizer,
            token_ids=token_ids,
            assistant_end_id=assistant_end_id,
        )
        if prepared.token_ids != tuple(token_ids):
            raise ValueError("prepared extraction token_ids do not match requested token_ids")
        self._cache[token_ids] = prepared
        return prepared


class TurnGemmaExtractionService(GemmaExtractionService):
    """Instance-local turn entries over one checkpoint/statistics snapshot.

    The shared producer performs the Gemma-token preflight, turn extraction,
    SFT selection and conversion. File replacement invalidates even an exact
    token cache hit; no new file contents enter the already-loaded snapshot.
    Prompt selection never advances the session or fitter's RNG streams.
    """

    def __init__(
        self, setup: MethodSetup, tokenizer: Any, device: str, statistics_path: Path
    ) -> None:
        super().__init__(setup, tokenizer, device)
        self._statistics_path = Path(statistics_path)
        self._statistics_sha = resolve_stats_sha(setup)
        self._metadata_identity = json.dumps(setup.meta, sort_keys=True)

    def prepare(self, token_ids: tuple[int, ...]) -> PreparedPromptContext:
        if json.dumps(self.setup.meta, sort_keys=True) != self._metadata_identity:
            raise ValueError("turn Chat setup metadata changed; reload the checkpoint")
        loaded = json.loads(self.setup.target_meta_path.read_text(encoding="utf-8"))
        if json.dumps(loaded, sort_keys=True) != self._metadata_identity:
            raise ValueError("turn Chat checkpoint metadata changed; reload the checkpoint")
        payload = self._statistics_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self._statistics_sha:
            raise ValueError("turn Chat statistics raw SHA-256 changed; reload the checkpoint")
        # Positive/negative bundles normally share a generator. Preserve both
        # identities instead of relying on that implementation detail.
        positive = self.setup.prompt_pipeline.selection_config
        if not isinstance(positive, SamplingBundle):
            raise ValueError("turn Chat prompt conditioning must use sampling selection")
        bundles = (positive, self.setup.prompt_pipeline.negative_bundle)
        generators = {id(bundle.rng): bundle.rng for bundle in bundles if bundle is not None}
        states = [(rng, rng.get_state().clone()) for rng in generators.values()]
        try:
            for rng in generators.values():
                rng.manual_seed(self._conditioning_seed)
            return super().prepare(token_ids)
        finally:
            for rng, state in states:
                rng.set_state(state)
