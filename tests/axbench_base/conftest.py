"""Shared test fixtures for the axbench_base test tree.

Consolidates patterns that were previously duplicated across unit tests
so that integration tests (Tasks 46 and 47) have a clean slate to work
from. Anything tests import from here is considered a stable fixture
API — extend it rather than hand-rolling more private helpers.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base import runtime_state as _rs_mod
from steering.eval.axbench_base.config import AxBenchBaseConfig
from tests.axbench_base._tiny_run_dir import build_tiny_run_dir as _build_tiny_run_dir
from tests.axbench_base.fixtures.build_tiny_shard import build_tiny_shard
from tests.axbench_base.fixtures.mock_base_sae import MockBaseSAE
from tests.axbench_base.fixtures.mock_openrouter import MockOpenRouterClient
from tests.axbench_base.fixtures.offline_openrouter import (
    OfflineOpenRouterTransport,
    OpenRouterRequest,
)

# Path to the canned concepts JSON committed alongside this module. Used
# by the ``tiny_concepts_path`` fixture, which copies the file into each
# test's ``tmp_path`` so test code can write back without mutating the
# shared source.
_TINY_CONCEPTS_SOURCE: Path = Path(__file__).parent / "fixtures" / "tiny_concepts.json"


@pytest.fixture
def mock_openrouter_client() -> Callable[..., MockOpenRouterClient]:
    """Factory producing a :class:`MockOpenRouterClient`.

    Usage::

        client = mock_openrouter_client({"prompt-prefix": "response"})
        assert client.complete("prompt-prefix foo") == "response"

    The returned factory accepts an optional ``response_map`` dict
    keyed by prompt (or ``None`` for a default fallback) and any extra
    keyword arguments are forwarded to :class:`MockOpenRouterClient`.
    """

    def factory(
        response_map: dict[str | None, str] | None = None,
        **kwargs: Any,
    ) -> MockOpenRouterClient:
        return MockOpenRouterClient(
            response_map=response_map or {},
            **kwargs,
        )

    return factory


@pytest.fixture
def mock_base_sae() -> Callable[..., MockBaseSAE]:
    """Factory producing a :class:`MockBaseSAE`.

    Usage::

        sae = mock_base_sae(feature_dim=1024, seed=7)
        activations = sae.encode(residual)

    Default values match the production gemmascope-res-16k config
    (``feature_dim=16384``, ``seed=42``).
    """

    def factory(
        feature_dim: int = 16384,
        seed: int = 42,
    ) -> MockBaseSAE:
        return MockBaseSAE(feature_dim=feature_dim, seed=seed)

    return factory


@pytest.fixture
def tiny_concepts_path(tmp_path: Path) -> Path:
    """Return a path to a tiny concepts JSON in ``tmp_path``.

    Three concepts:

    - ``concept_id=0``: ``"the color blue"``
    - ``concept_id=1``: ``"scientific measurement"``
    - ``concept_id=2``: ``"unicode test: café résumé"``

    The source file is committed under ``tests/axbench_base/fixtures/``
    and copied into ``tmp_path`` so test code can mutate it without
    affecting other tests.
    """
    target = tmp_path / "tiny_concepts.json"
    shutil.copy(_TINY_CONCEPTS_SOURCE, target)
    return target


@pytest.fixture
def tiny_validation_shard_path(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Build a tiny (parquet, feature shard) pair for seed_pool tests.

    Delegates to :func:`tests.axbench_base.fixtures.build_tiny_shard.build_tiny_shard`
    (the shared helper written for Task 8). Returns the tuple unchanged
    so callers can pick either half.
    """
    return build_tiny_shard(tmp_path / "shards", shard_idx=0)


@pytest.fixture
def tmp_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Sets ``NANOCHAT_BASE_DIR`` to ``tmp_path`` for this test.

    Any stage output that touches :func:`nanochat.common.get_base_dir`
    lands under ``tmp_path`` when this fixture is in scope.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def make_valid_axbench_cfg(
    tmp_path: Path,
) -> Callable[..., AxBenchBaseConfig]:
    """Factory for a valid :class:`AxBenchBaseConfig` with test defaults.

    The factory uses the explicit-null local Stage 0c mode so unrelated tests
    cannot make external API calls merely because the shipped production
    default is API-backed. Everything else inherits from the dataclass
    defaults. Pass keyword overrides to replace any field.

    Usage::

        cfg = make_valid_axbench_cfg(max_concepts=3, run_tag="t7")

    ``output_root`` defaults to null, exercising the legacy-compatible path
    contract beneath the autouse ``NANOCHAT_BASE_DIR=tmp_path`` sandbox.
    Tests of custom relocation override it explicitly.
    """

    def factory(**overrides: Any) -> AxBenchBaseConfig:
        base = AxBenchBaseConfig(
            feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
            feature_statistics_path="stats.safetensors",
            decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
            encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
            device="cuda",
            output_root=None,
            run_tag="test",
            verification_api_model=None,
        )
        if overrides:
            return base.replace(**overrides)
        return base

    return factory


# ---------------------------------------------------------------------------
# Phase 8 integration fixtures
#
# The Tier 1 integration tests in
# ``tests/axbench_base/integration/test_full_pipeline_dev_tier.py``
# exercise the full ``_run`` pipeline against a real
# :class:`RuntimeState` with heavy loaders replaced by stubs and the
# OpenRouter wire replaced by a canned cached-``Session.post`` transport.
# Fixtures
# below are the shared construction machinery.
# ---------------------------------------------------------------------------


def _build_minimal_feature_statistics(path: Path, *, n_sae_features: int) -> Path:
    """Write a minimal safetensors file whose metadata carries
    ``n_sae_features``.

    Matches the exact shape the axbench_base pipeline expects —
    :func:`steering.eval.axbench_base._stats_reader.read_n_sae_features_from_stats`
    only reads the safetensors header metadata and never touches the
    tensors, so a placeholder tensor is sufficient.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )
    return path


@pytest.fixture
def tiny_feature_statistics_path(tmp_path: Path) -> Path:
    """Return the path to a minimal feature-statistics safetensors file.

    Used by Phase 8 integration tests that need
    :attr:`AxBenchBaseConfig.feature_statistics_path` to resolve to a
    real file whose header exposes ``n_sae_features`` (16384 to match
    the default gemmascope-res-16k SAE). Written under ``tmp_path`` so
    tests remain fully isolated.
    """
    stats_path = tmp_path / "tiny_stats.safetensors"
    return _build_minimal_feature_statistics(stats_path, n_sae_features=16384)


class _RuntimeTestTokenizer:
    """Deterministic reversible tokenizer for runtime integration fixtures."""

    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        tokens = [ord(char) + 2 for char in text]
        return ([prepend] if prepend is not None else []) + tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 2) for token_id in token_ids if token_id != 0)


class _MockStubModel:
    """Callable no-op stand-in for the four heavy-model loader returns.

    Implements the minimum surface exercised by
    :class:`steering.eval.axbench_base.runtime_state.RuntimeState` when
    it manages a heavy resource via its context-manager protocol:

    - ``to(device)`` returns ``self`` so chained ``.to("cpu")`` works.
    - ``_name`` is a debug tag so test failures can print a meaningful
      identifier.
    - ``_loaded_meta_data`` is an empty dict so the encdec loader's
      post-load attribute assignment still succeeds even when no test
      bothers to populate it.

    The stub intentionally does NOT implement ``.trace`` or the
    nnterp StandardizedTransformer API. Phase 8 integration tests run
    with every stage function (``prepare_negative_prefixes``,
    ``synthesize_positive_mode``, ``synthesize_negative_mode``,
    ``concept_feature_mapping_*``, ``train_diffmean_all``, ...)
    monkey-patched at the cli module level so the stage bodies never
    run and thus never call into the base Gemma / SAE models. This is
    the pragmatic tradeoff: fully stubbing nnterp's forward-pass API
    would require a ~500-line shim, while the tests we actually care
    about here are the ``_run`` control-flow + RuntimeState lifecycle
    + dispatcher wiring contracts, not the stage bodies themselves
    (which are covered by the dedicated per-stage unit tests).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._loaded_meta_data: dict[str, Any] = {}
        self.config = SimpleNamespace(sequence_len=2048)
        self.dec_config = SimpleNamespace(sequence_len=2048)

    def to(self, device: Any) -> _MockStubModel:
        del device
        return self

    def __repr__(self) -> str:
        return f"_MockStubModel(name={self._name!r})"


@pytest.fixture
def stub_heavy_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, _MockStubModel]:
    """Replace the four heavy-model loaders with deterministic stubs.

    Mirrors the private ``_stub_heavy_loaders`` fixture in
    ``tests.axbench_base.unit.test_cli_entry_point`` but lives at the
    conftest level so integration tests can consume it without
    importing private test-module symbols.

    Each stub loader returns a shared :class:`_MockStubModel` instance
    for its name, so re-entering a context manager reuses the same
    cached stub and the refcount + CPU-offload contract works exactly
    like production. CUDA helpers are also neutralized so the tests
    stay hardware-agnostic.

    Additional resources are also stubbed at the module boundary so
    Phase 8 integration tests can invoke every runtime-state accessor
    without touching the network or the real tokenizer bundle:

    - the imported nanochat tokenizer factory returns a test tokenizer
      so ``state.nanochat_tokenizer()`` never tries to read the
      tokenizer.pkl under ``$NANOCHAT_BASE_DIR``.
    - :meth:`AutoTokenizer.from_pretrained` returns a :class:`MagicMock` so
      ``state.extraction_tokenizer()`` never hits HuggingFace Hub.
    - :func:`detect_target_model_metadata` returns a fixed
      :class:`TargetModelMetadata` with ``num_buckets=4`` and no
      negative selection so ``state.decoder_only_meta()`` /
      ``state.encdec_meta()`` never inspect the stubbed model.

    Returns a mapping ``{loader_name: stub_model}`` so callers can
    grab the shared instance if they need it.

    --- Audit slice 16, X2 §6.1 mocking-trap notice ---

    This fixture is a deliberate "stub everything heavy" tool. Production-side
    hooks are replaced. Tests using this fixture
    verify the dispatch layer's *plumbing* but NOT that any real
    model code runs. Specifically, surfaces NOT exercised by tests
    using ``stub_heavy_loaders``:

    1. ``_load_feature_extraction_pair`` — actual SAE+model loading
       (``steering.extract_features.{build_nnterp_model,load_sae_and_model}``).
    2. ``_load_decoder_only_model`` / ``_load_encdec_model`` — real
       ``nanochat.checkpoint_manager.load_model``.
    3. ``detect_target_model_metadata`` — the live model-introspection
       path that reads ``user_config`` and resolves negative-selection
       semantics.
    4. The imported nanochat and HuggingFace tokenizer factories — real
       tokenizer initialization (RustBPE + HuggingFace Hub).
    5. ``torch.cuda.is_available`` / ``torch.cuda.empty_cache`` —
       CUDA-side refcount accounting.

    Slice owners adding new tests should:

    - For dispatch-layer regressions: KEEP using this fixture.
    - For correctness-of-stage regressions: AVOID this fixture and
      use a real-stage path. See
      ``tests/audit_2026_05/slice_16_axbench_base/test_real_stage_integration.py``
      for a CPU-only example that exercises real ``aggregation`` /
      ``judge_parser`` / ``paths`` code without the stubs.
    """
    from steering.eval.axbench_base.types import TargetModelMetadata

    loaded: dict[str, _MockStubModel] = {}

    def _make_loader(name: str) -> Callable[[Any], _MockStubModel]:
        def _loader(cfg: Any) -> _MockStubModel:
            del cfg
            if name not in loaded:
                loaded[name] = _MockStubModel(name)
            return loaded[name]

        return _loader

    def _make_pair_loader() -> Callable[[Any], tuple[Any, Any]]:
        def _loader(cfg: Any) -> tuple[Any, Any]:
            del cfg
            if "base_model" not in loaded:
                loaded["base_model"] = _MockStubModel("base_model")
            if "base_sae" not in loaded:
                loaded["base_sae"] = _MockStubModel("base_sae")
            return loaded["base_model"], loaded["base_sae"]

        return _loader

    monkeypatch.setattr(
        _rs_mod, "_load_feature_extraction_pair", _make_pair_loader(),
    )
    monkeypatch.setattr(
        _rs_mod, "_load_decoder_only_model", _make_loader("decoder_only"),
    )
    monkeypatch.setattr(
        _rs_mod, "_load_encdec_model", _make_loader("encdec"),
    )
    monkeypatch.setattr(_rs_mod.t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_rs_mod.t.cuda, "empty_cache", MagicMock())
    # Tokenizer stubs — never hit disk or HF Hub.
    monkeypatch.setattr(
        _rs_mod, "_nanochat_get_tokenizer", _RuntimeTestTokenizer
    )
    monkeypatch.setattr(
        _rs_mod.AutoTokenizer, "from_pretrained",
        lambda model_id: MagicMock(name="extraction_tokenizer"),
    )
    # Meta detector stub — returns a fixed dataclass instance so
    # ``state.decoder_only_meta()`` / ``state.encdec_meta()`` never
    # poke at the stubbed target models.
    stub_meta = TargetModelMetadata(
        num_buckets=4,
        n_sae_features=16384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )

    def _fake_detect(
        target_model: Any, cfg: Any,
    ) -> TargetModelMetadata:
        del target_model, cfg
        return stub_meta

    monkeypatch.setattr(_rs_mod, "detect_target_model_metadata", _fake_detect)

    return loaded


@pytest.fixture
def mock_base_gemma_for_generation():
    """Factory fixture: pass a `seed_to_outputs` map; get a MockBaseGemma."""
    from tests.axbench_base._mock_models import MockBaseGemma

    def _factory(seed_to_outputs):
        return MockBaseGemma(seed_to_outputs)

    return _factory


@pytest.fixture
def mock_openrouter_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Any]]:
    """Patch the production ``Session.post`` OpenRouter wire.

    Production clients use the cached thread-local
    :class:`requests.Session`, so the fixture patches that exact boundary.
    The module-level function and OpenRouter lower-level transport are
    tripwires: an accidental new OpenRouter client path fails immediately
    instead of reaching the provider.

    The fixture also sets ``OPENROUTER_API_KEY`` so
    :func:`runtime_state.validate_env_for_stages` and the
    ``AxBenchJudgeClient`` / ``OpenRouterLLMClient`` constructors pass
    their ``api_key is not None`` checks. Tests that want the env var
    UNSET must use ``monkeypatch.delenv`` AFTER this fixture runs.

    The historical ``{"post": <mock>}`` return shape is preserved.  The
    returned mapping also exposes the transport and request records for
    stronger assertions.
    """

    def _response_factory(request: OpenRouterRequest) -> str:
        prompt_text = request.messages[-1]["content"] if request.messages else ""
        if request.request_class == "stage5_judge" or "rating" in prompt_text.lower():
            return "Rating: [[1]]"
        return "Generated continuation text."

    transport = OfflineOpenRouterTransport(_response_factory)
    transport.install(monkeypatch)
    transport.bind_production_clients(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-mock")
    try:
        yield {
            "post": transport.post_mock,
            "transport": transport,
            "requests": transport.requests,
        }
    finally:
        transport.close()


@pytest.fixture
def tiny_axbench_run_dir(tmp_path):
    """A tiny but complete AxBench run dir skeleton for e2e tests."""
    return _build_tiny_run_dir(tmp_path)
