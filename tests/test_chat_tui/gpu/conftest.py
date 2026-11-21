"""Session-scoped real-artifact fixtures for the Chat TUI GPU gates."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch as t

from scripts.chat_tui.extraction import GemmaExtractionService
from scripts.chat_tui.setup import load_chat_setup, load_chat_target_model
from steering.eval.axbench_upstream.methods import MethodSetup, fit_dense_diffmean

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")

LEAF_META = (
    Path.home() / ".cache/nanochat/conditional_encdec_checkpoints/"
    "l22_gemma3pt_clustered_splice_cf_c1024_100m_370_2node_pr1407_v1/"
    "s128/dist_geometric/mean_12.0/std_12.0/wl_9172c540310ba842/"
    "fc_5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039/"
    "standard_sft_conditioned_checkpoints/meta_000749.json"
)
PRETRAIN_META = (
    Path.home() / ".cache/nanochat/conditional_encdec_checkpoints/"
    "l22_main_h100_8rank_db16_v1/s128/dist_geometric/mean_12.0/std_12.0/"
    "wl_9172c540310ba842/"
    "fc_5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039/"
    "meta_054600.json"
)
STATS = Path(
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos/"
    "feature_statistics.safetensors"
)
PREPARED = Path("out/axbench/prepared")
VAL_FEATURE_FAMILY = Path(
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/standard_sft_val/context_independent/no_bos"
)

PROMPT_TEXT = (
    "Explain why careful measurements are important when comparing two machine-learning "
    "systems. Give a concrete example."
)
POSITIVE_TEXTS = (
    "A careful scientist records measurements and checks every assumption.",
    "The laboratory report compares repeated observations with controls.",
    "Reliable experiments use evidence, calibration, and reproducible methods.",
    "The engineer verifies the result against an independent reference.",
)
NEGATIVE_TEXTS = (
    "A dragon sleeps beneath a castle beside the sea.",
    "The painter mixed violet pigment for the evening sky.",
    "A wooden boat drifted quietly across the moonlit lake.",
    "Fresh bread cooled on the kitchen counter after sunrise.",
)
SECOND_NEGATIVE_TEXTS = (
    "A musician tuned the old piano before the concert.",
    "Snow covered the narrow trail through the silent forest.",
    "The gardener planted tulips along the stone wall.",
    "A brass clock chimed softly in the empty hallway.",
)


def _require_file(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"required artifact path is absent: {path}")


def _resolve_repo_artifact(path: Path) -> Path:
    """Resolve a mandated repo-relative constant from a linked worktree."""

    if path.is_absolute() or path.exists():
        return path
    for parent in Path(__file__).resolve().parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def _require_checkpoint(meta_path: Path) -> None:
    _require_file(meta_path)
    model_path = meta_path.with_name(
        meta_path.name.replace("meta_", "model_").replace(".json", ".pt")
    )
    _require_file(model_path)


@pytest.fixture(scope="session", autouse=True)
def configured_nanochat_base():
    """Make the canonical tokenizer/data cache visible from linked worktrees."""

    prior = os.environ.get("NANOCHAT_BASE_DIR")
    if prior is None:
        candidate = Path.home() / ".cache/nanochat"
        _require_file(candidate / "tokenizer/tokenizer.pkl")
        os.environ["NANOCHAT_BASE_DIR"] = str(candidate)
    yield
    if prior is None:
        os.environ.pop("NANOCHAT_BASE_DIR", None)


@pytest.fixture(scope="session")
def stats_path() -> Path:
    resolved = _resolve_repo_artifact(STATS)
    _require_file(resolved)
    return resolved


@pytest.fixture(scope="session")
def val_feature_family_path() -> Path:
    return _resolve_repo_artifact(VAL_FEATURE_FAMILY)


class _SharedStackExtractionService(GemmaExtractionService):
    """Use one resident Gemma+SAE stack with setup-specific prompt pipelines.

    ``load_extraction_stack`` has no process cache. Both checkpoints resolve the
    identical Gemma-3-4B L22 SAE, so loading it twice would duplicate the large
    resident weights. Only the stack is shared; each service retains its own
    setup, deterministic prompt RNG, and content-addressed cache.
    """

    def __init__(
        self,
        setup: MethodSetup,
        tokenizer: Any,
        device: str,
        shared_stack: tuple[Any, Any],
    ) -> None:
        super().__init__(setup, tokenizer, device)
        self._shared_stack = shared_stack

    def _ensure_stack(self) -> tuple[Any, Any]:
        return self._shared_stack


@pytest.fixture(scope="session")
def leaf_setup(stats_path: Path) -> MethodSetup:
    _require_file(LEAF_META)
    return load_chat_setup(LEAF_META, stats_path)


@pytest.fixture(scope="session")
def leaf_model_tokenizer(leaf_setup: MethodSetup):
    _require_checkpoint(LEAF_META)
    model, tokenizer = load_chat_target_model(leaf_setup, "cuda")
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def pretrain_setup(stats_path: Path) -> MethodSetup:
    _require_file(PRETRAIN_META)
    return load_chat_setup(PRETRAIN_META, stats_path)


@pytest.fixture(scope="session")
def pretrain_model_tokenizer(pretrain_setup: MethodSetup):
    _require_checkpoint(PRETRAIN_META)
    model, tokenizer = load_chat_target_model(pretrain_setup, "cuda")
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def extraction_stack(
    leaf_setup: MethodSetup,
    leaf_model_tokenizer,
) -> GemmaExtractionService:
    """Leaf service owning the one resident full-size extraction stack."""

    _model, tokenizer = leaf_model_tokenizer
    return GemmaExtractionService(leaf_setup, tokenizer, "cuda")


@pytest.fixture(scope="session")
def shared_extraction_weights(extraction_stack: GemmaExtractionService) -> tuple[Any, Any]:
    return extraction_stack.extraction_stack()


@pytest.fixture(scope="session")
def pretrain_extraction_stack(
    pretrain_setup: MethodSetup,
    pretrain_model_tokenizer,
    leaf_setup: MethodSetup,
    shared_extraction_weights: tuple[Any, Any],
) -> GemmaExtractionService:
    if (
        pretrain_setup.sae_model,
        pretrain_setup.sae_layer,
        pretrain_setup.resolved_sae.sae_id,
    ) != (
        leaf_setup.sae_model,
        leaf_setup.sae_layer,
        leaf_setup.resolved_sae.sae_id,
    ):
        pytest.fail("leaf and pretrain checkpoints do not resolve the same extraction stack")
    _model, tokenizer = pretrain_model_tokenizer
    return _SharedStackExtractionService(
        pretrain_setup,
        tokenizer,
        "cuda",
        shared_extraction_weights,
    )


@pytest.fixture(scope="session")
def extraction_service_factory(
    leaf_setup: MethodSetup,
    shared_extraction_weights: tuple[Any, Any],
) -> Callable[[MethodSetup, Any], GemmaExtractionService]:
    def make(setup: MethodSetup, tokenizer: Any) -> GemmaExtractionService:
        if (
            setup.sae_model,
            setup.sae_layer,
            setup.resolved_sae.sae_id,
        ) != (
            leaf_setup.sae_model,
            leaf_setup.sae_layer,
            leaf_setup.resolved_sae.sae_id,
        ):
            raise ValueError("fresh setup does not match the shared extraction stack")
        return _SharedStackExtractionService(
            setup,
            tokenizer,
            "cuda",
            shared_extraction_weights,
        )

    return make


@pytest.fixture(scope="session")
def first_concept_prompt() -> str:
    """Fixed-literal prompt used on both parity sides.

    The AxBench runner consumes ``row.original_prompt``; using the same literal
    on both sides exercises that exact prompt pipeline without making parity
    depend on a particular prepared-frame row or factor ordering.
    """

    return PROMPT_TEXT


def _fit(setup: MethodSetup, service: GemmaExtractionService, concept_id: int) -> t.Tensor:
    base_model, sae = service.extraction_stack()
    return fit_dense_diffmean(
        setup,
        base_model,
        sae,
        POSITIVE_TEXTS,
        NEGATIVE_TEXTS,
        concept_id=concept_id,
    )


@pytest.fixture(scope="session")
def leaf_diffmean(
    leaf_setup: MethodSetup,
    extraction_stack: GemmaExtractionService,
) -> t.Tensor:
    return _fit(leaf_setup, extraction_stack, 0)


@pytest.fixture(scope="session")
def pretrain_diffmean(
    pretrain_setup: MethodSetup,
    pretrain_extraction_stack: GemmaExtractionService,
) -> t.Tensor:
    return _fit(pretrain_setup, pretrain_extraction_stack, 0)


@pytest.fixture(scope="session")
def leaf_second_diffmean(
    leaf_setup: MethodSetup,
    extraction_stack: GemmaExtractionService,
) -> t.Tensor:
    base_model, sae = extraction_stack.extraction_stack()
    return fit_dense_diffmean(
        leaf_setup,
        base_model,
        sae,
        POSITIVE_TEXTS,
        SECOND_NEGATIVE_TEXTS,
        concept_id=1,
    )
