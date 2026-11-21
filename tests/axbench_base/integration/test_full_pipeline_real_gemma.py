"""Very-slow end-to-end integration test with real Gemma-2b + real SAE + real nanochat d12.

This file exists to PROVE the axbench_base benchmark actually works
against production models. The test is gated behind
``@pytest.mark.very_slow`` so default CI skips it. Run manually via::

    uv run pytest tests/axbench_base/integration/test_full_pipeline_real_gemma.py -v -m very_slow

Prerequisites (the test fails loud if any is missing)
-----------------------------------------------------

1. ``~/.cache/nanochat/`` is a populated nanochat cache containing a real
   d12 decoder-only checkpoint under
   ``base_checkpoints/d12_base/model_XXXXXX.pt`` and a d12 encoder-decoder
   checkpoint under
   ``conditional_encdec_checkpoints/d12_baseline/.../model_XXXXXX.pt``.
   The d12 checkpoints are produced by the full nanochat pretrain +
   encdec-SFT pipeline; their exact sub-directory layout varies depending
   on the training recipe (segment size, feature selection, etc.), so
   the prerequisite check only looks for the outermost layer.

2. OpenRouter is offline-mocked at the production cached
   ``requests.Session.post`` boundary. The test deletes any inherited
   ``OPENROUTER_API_KEY`` and injects ``sk-or-test-offline`` only so the
   runtime environment-validation path is exercised. No provider credential
   is required. Neuronpedia remains a separate real prerequisite for Stage
   0a and Stage 0c activation evidence.

3. The canonical feature directory exists on disk:
   ``out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base``
   with a ``feature_statistics.safetensors`` header next to it. This is
   produced by the feature-extraction pipeline; see
   ``scripts/extract_features.py`` for how to build it.

4. ~8GB of disk for the HuggingFace Hub cache on first run to materialize
   ``google/gemma-2-2b`` and ``gemma-scope-2b-pt-res-canonical``. Cached
   locally by HF Hub for subsequent runs.

5. Enough RAM + optional GPU to host all three models simultaneously
   (Gemma-2b bf16 ~4GB, d12 decoder-only ~4GB, d12 encdec ~4GB). The
   RuntimeState CPU-offload pattern keeps peak GPU memory bounded.

The test runs a one-concept pipeline through the real offline Stage 3e/4
boundary and asserts:

- ``stage=all`` reaches Stage 3e and the missing-cache guard stops Stage 4
- The Stage 3e parquet is extracted with the real Gemma/SAE pair
- The resulting context cache is converted and validated against the target
  checkpoint's actual feature-selection configuration
- Stages 4--6 resume from that cache
- Every artifact exists at the expected path
- ``leaderboard.csv`` contains the exact ordinary Stage 4 method set
- ``report.yaml`` is non-empty
- No exceptions during the run
- RuntimeState closes cleanly at the end

Expected runtime: roughly 20-45 minutes with a modern GPU.  The single real
concept still traverses every ordinary method family; broader concept-count
coverage belongs to the synthetic integration suite rather than this model and
artifact compatibility gate.
When it fails, check the stdout for the specific fail-loud message and
follow the instructions in that message to fix your environment.

The prerequisite helpers at the bottom of this module
(``_require_nanochat_cache``,
``_require_feature_dir``) are unit-tested in
``tests/axbench_base/unit/test_real_gemma_prerequisites.py``.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.eval import axbench_base_eval as _cli
from scripts.eval import build_eval_context_cache as _context_cache_cli
from steering.eval.axbench_base import _stage0c_internals as _stage0c_internals
from steering.eval.axbench_base import openrouter, paths
from steering.eval.axbench_base import seed_pools as _seed_pools
from steering.eval.axbench_base import stage0c_api_generation as _stage0c_api
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.seed_contract import sha256_file
from steering.eval.axbench_base.synth import batching as _synth_batching
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cluster_config import parse_cluster_eval_paths
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.eval.context_cache_io import load_context_cache_manifest
from steering.extract_features import process_shard
from steering.feature_paths import parse_feature_dir
from steering.segmentation import SegmentMode
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.offline_openrouter import (
    OfflineOpenRouterTransport,
)
from tests.axbench_base.fixtures.real_gemma_offline import RealGemmaOfflineOracle

_NANOCHAT_CACHE: Path = Path.home() / ".cache" / "nanochat"

#: The canonical decoder-only d12 checkpoint directory relative to the
#: nanochat cache root. The inner step file is auto-discovered by
#: ``nanochat.checkpoint_manager.find_last_step``.
_D12_BASE_REL: Path = Path("base_checkpoints") / "d12_base"

#: The canonical encoder-decoder d12 root directory inside the nanochat
#: cache. The nanochat checkpoint loader traverses sub-directories
#: (``s<segment_size>/...``) to find the actual checkpoint files, so we
#: only assert on the top-level directory here.
_D12_ENCDEC_REL: Path = Path("conditional_encdec_checkpoints") / "d12_baseline"

#: Canonical feature directory built by ``scripts/extract_features.py``.
#: The path is deliberately the same as the CLI example in
#: ``configs/eval/axbench_base/axbench_base.yaml``.
_FEATURE_DIR: Path = Path(
    os.environ.get(
        "AXBENCH_REAL_FEATURE_DIR",
        "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
    )
).expanduser().resolve()

_FEATURE_STATISTICS_FILE: Path = _FEATURE_DIR / "feature_statistics.safetensors"

# These methods consume fair/roofline bags derived by AxBench-BPB-Adv. Base
# Stage 4 intentionally has no live bag optimizer/loader, so dispatching them
# here must remain fail-loud instead of fabricating a conditioning bag.
_BPB_ONLY_PRECOMPUTED_BAG_METHODS: tuple[str, ...] = (
    "ours_gcg_encdec",
    "ours_gcg_roofline_encdec",
    "feature_diffmean_encdec",
    "feature_diffmean_roofline_encdec",
    "neg_ours_gcg_encdec",
    "neg_ours_gcg_roofline_encdec",
    "neg_feature_diffmean_encdec",
    "neg_feature_diffmean_roofline_encdec",
)

_EXPECTED_REAL_GEMMA_STAGE4_METHODS = frozenset(
    {
        "diffmean_decoder_only",
        "diffmean_encdec",
        "linear_probe_decoder_only",
        "linear_probe_encdec",
        "lora_decoder_only",
        "prompting_decoder_only",
        "prompting_encdec",
        "control_decoder_only",
        "control_encdec",
        "ours_extract_and_add_encdec",
        "ours_replace_random_encdec",
        "ours_replace_all_encdec",
        "neg_diffmean_decoder_only",
        "neg_diffmean_encdec",
        "neg_linear_probe_decoder_only",
        "neg_linear_probe_encdec",
        "neg_lora_decoder_only",
        "neg_prompting_decoder_only",
        "neg_prompting_encdec",
        "neg_control_decoder_only",
        "neg_control_encdec",
        "neg_ours_remove_encdec",
        "neg_ours_add_to_negatives_encdec",
        "neg_ours_replace_random_negatives_encdec",
        "neg_ours_replace_all_negatives_encdec",
    }
)


def _real_stage4_conditioning_config(
    context_cache_root: Path,
    *,
    stats_source_sha256: str,
) -> dict[str, Any]:
    """Return the real d12 baseline's canonical Stage 4 cache identity inputs."""

    return {
        "context_cache_dir": str(context_cache_root),
        "identity": {
            "model_id": "gemma-2-2b",
            "sae_id": "gemmascope-residual-l8-16k",
            "layer": 8,
            "pooling": "max_pool",
            # Deliberately derived from the canonical inputs below. The cache
            # builder later commits this hash, and Stage 4 reads it back.
            "feature_selection_hash": None,
        },
        "n_sae_features": 16384,
        "num_buckets": 2,
        "feature_selection": {
            "distribution": "geometric",
            "mean": 10.0,
            "std": 10.0,
            "min_k": 1,
            "max_k": 40,
            "mode": "top_k_percentile",
            "delta_weights": None,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 10.0,
            "min_k": 0,
            "max_k": 40,
        },
        "feature_delta": None,
        "stats_source_sha256": stats_source_sha256,
        "seed": 42,
        "feature_mask": None,
        "feature_clusters": None,
    }


def _require_nanochat_cache() -> None:
    """Fail loud if the nanochat cache isn't populated with d12 models.

    Raises ``pytest.fail`` with an actionable message naming what's
    missing and how to produce it. We use ``pytest.fail`` (not
    ``pytest.skip``) because this test is ONLY run manually via
    ``-m very_slow``; skipping would swallow the failure and make the
    operator think the test passed.
    """
    if not _NANOCHAT_CACHE.exists():
        pytest.fail(
            f"Nanochat cache directory {_NANOCHAT_CACHE} not found. "
            f"This test requires real nanochat d12 checkpoints. Run the "
            f"nanochat pretrain + encdec-SFT pipeline to populate "
            f"$NANOCHAT_BASE_DIR (or symlink ~/.cache/nanochat to your "
            f"existing nanochat_cache/ directory)."
        )

    decoder_dir = _NANOCHAT_CACHE / _D12_BASE_REL
    if not decoder_dir.exists():
        pytest.fail(
            f"d12 decoder-only base checkpoint directory not found at "
            f"{decoder_dir}. Run the nanochat pretrain pipeline to "
            f"produce base_checkpoints/d12_base/model_XXXXXX.pt, or "
            f"point NANOCHAT_BASE_DIR at a cache that already contains "
            f"it."
        )
    base_checkpoint_files = list(decoder_dir.glob("model_*.pt"))
    if not base_checkpoint_files:
        pytest.fail(
            f"No model_*.pt files in {decoder_dir}. The directory "
            f"exists but appears to be empty or only contains metadata. "
            f"Re-run the nanochat pretrain pipeline."
        )

    encdec_dir = _NANOCHAT_CACHE / _D12_ENCDEC_REL
    if not encdec_dir.exists():
        pytest.fail(
            f"d12 encdec checkpoint directory not found at {encdec_dir}. "
            f"Run the nanochat encdec SFT pipeline to produce "
            f"conditional_encdec_checkpoints/d12_baseline/.../model_XXXXXX.pt, "
            f"or point NANOCHAT_BASE_DIR at a cache that already "
            f"contains it."
        )
    encdec_checkpoint_files = list(encdec_dir.rglob("model_*.pt"))
    if not encdec_checkpoint_files:
        pytest.fail(
            f"No model_*.pt files anywhere under {encdec_dir}. The "
            f"directory exists but appears to be empty. Re-run the "
            f"nanochat encdec SFT pipeline."
        )


def _require_feature_dir() -> None:
    """Fail loud if the feature dir or its statistics file is missing.

    The axbench_base pipeline reads ``cfg.feature_dir`` for seed-pool
    sampling and ``cfg.feature_statistics_path`` for the SAE feature
    count. Both must exist on disk before the test starts — we do not
    materialize placeholders for them because the pipeline touches real
    shards for seed construction.
    """
    if not _FEATURE_DIR.is_dir():
        pytest.fail(
            f"Feature directory not found at {_FEATURE_DIR.resolve()}. "
            f"This test needs a real Gemma-2b feature dir. Build it "
            f"with:\n"
            f"    uv run python -m scripts.extract_features "
            f"seg_size=128 layer=8 activation=res pool=max_pool "
            f"model_tag=gemma-2-2b"
        )
    if not _FEATURE_STATISTICS_FILE.exists():
        pytest.fail(
            f"Feature statistics file not found at "
            f"{_FEATURE_STATISTICS_FILE.resolve()}. Re-run "
            f"scripts/extract_features.py with the statistics step "
            f"enabled, or verify that "
            f"{_FEATURE_STATISTICS_FILE.name} lives next to the feature "
            f"shards in {_FEATURE_DIR}."
        )


def _cuda_available() -> bool:
    """Thin wrapper around ``torch.cuda.is_available`` so test code can
    express the "use GPU if present, else CPU" device choice in one
    call. Wrapped as a function (rather than a module-level constant)
    so monkey-patching ``t.cuda.is_available`` in a helper test still
    affects the real read.
    """
    return bool(t.cuda.is_available())


def _install_openrouter_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[OfflineOpenRouterTransport, RealGemmaOfflineOracle]:
    """Install the fail-closed wire and its local Gemma/SAE response oracle."""

    transport = OfflineOpenRouterTransport()
    oracle = RealGemmaOfflineOracle(transport)
    transport.response_factory = oracle.response
    transport.install(monkeypatch)
    transport.bind_production_clients(monkeypatch)
    return transport, oracle


_EXPECTED_OPENROUTER_REQUEST_CLASSES = frozenset(
    {
        "stage0c_definition",
        "stage0c_continuation",
        "stage1_synth_rich",
        "stage1_synth_avoid",
        "stage5_judge",
    }
)

_EXPECTED_OPENROUTER_MESSAGE_ROLES = {
    "stage0c_definition": ("user",),
    "stage0c_continuation": ("system", "user"),
    "stage1_synth_rich": ("user",),
    "stage1_synth_avoid": ("user",),
    "stage5_judge": ("user",),
}


def _assert_exact_openrouter_request_classes(
    transport: OfflineOpenRouterTransport,
) -> None:
    """Require every expected request class and reject every unexpected one."""

    seen_classes = {request.request_class for request in transport.requests}
    assert seen_classes == _EXPECTED_OPENROUTER_REQUEST_CLASSES, (
        "offline OpenRouter request classes changed: "
        f"expected={_EXPECTED_OPENROUTER_REQUEST_CLASSES!r}, "
        f"seen={seen_classes!r}"
    )


def _assert_openrouter_message_roles(request: Any) -> None:
    """Pin the production message-role contract for each request class."""

    expected_roles = _EXPECTED_OPENROUTER_MESSAGE_ROLES.get(
        request.request_class
    )
    assert expected_roles is not None, (
        f"no message-role contract for request class {request.request_class!r}"
    )
    assert request.message_roles == expected_roles, (
        "offline OpenRouter message roles changed: "
        f"request_class={request.request_class!r}, "
        f"expected={expected_roles!r}, actual={request.message_roles!r}"
    )


def _assert_exact_stage4_methods(actual_methods: set[str]) -> None:
    """Require every ordinary Stage 4 method and reject unexpected methods."""

    assert actual_methods == _EXPECTED_REAL_GEMMA_STAGE4_METHODS, (
        "real-Gemma Stage 4 method set changed: "
        f"missing={sorted(_EXPECTED_REAL_GEMMA_STAGE4_METHODS - actual_methods)!r}, "
        f"unexpected={sorted(actual_methods - _EXPECTED_REAL_GEMMA_STAGE4_METHODS)!r}"
    )


def _bind_live_oracle_contexts(
    monkeypatch: pytest.MonkeyPatch,
    oracle: RealGemmaOfflineOracle,
) -> None:
    """Attach the oracle to live stage and Stage 0c request identities."""

    original_definition_for_feature = (
        _stage0c_api.Stage0cApiGenerator._definition_for_feature
    )

    def definition_for_feature_with_oracle(
        generator: Any,
        feature_id: int,
    ) -> Any:
        # Production hashes the definition tag before the request reaches the
        # Session.post boundary. Carry the actual method argument instead.
        with oracle.stage0c_member_context(feature_id):
            return original_definition_for_feature(generator, feature_id)

    monkeypatch.setattr(
        _stage0c_api.Stage0cApiGenerator,
        "_definition_for_feature",
        definition_for_feature_with_oracle,
    )

    original_run_one = _synth_batching._run_one

    def run_one_with_oracle(
        llm_client: Any,
        request: Any,
        *,
        profile_region_name: str,
        submitted_at: float,
    ) -> Any:
        # Clustered Stage 0c puts the selected member in SynthRequest.attrs
        # before hashing the cache key. _run_one is the last per-request seam
        # before the real client and preserves bounded-parallel execution.
        member_id = request.attrs.get("description_feature_id")
        if member_id is None:
            return original_run_one(
                llm_client,
                request,
                profile_region_name=profile_region_name,
                submitted_at=submitted_at,
            )
        if type(member_id) is not int or member_id < 0:
            pytest.fail(
                "Stage 0c description_feature_id must be a strict "
                f"non-negative int, got {member_id!r}"
            )
        with oracle.stage0c_member_context(member_id):
            return original_run_one(
                llm_client,
                request,
                profile_region_name=profile_region_name,
                submitted_at=submitted_at,
            )

    monkeypatch.setattr(_synth_batching, "_run_one", run_one_with_oracle)

    original_prepare = _cli.prepare_negative_prefixes

    def prepare_with_oracle(
        cfg: Any,
        concepts_path: Any,
        base_model: Any,
        base_sae: Any,
        target_model_meta: Any,
        response_cache: Any = None,
        *,
        rank: int = 0,
        world_size: int = 1,
        cluster_contract: Any = None,
    ) -> Any:
        with oracle.stage_context(
            cfg,
            base_model=base_model,
            base_sae=base_sae,
            target_model_meta=target_model_meta,
            concepts=[],
            cluster_contract=cluster_contract,
        ):
            return original_prepare(
                cfg,
                concepts_path,
                base_model,
                base_sae,
                target_model_meta,
                response_cache=response_cache,
                rank=rank,
                world_size=world_size,
                cluster_contract=cluster_contract,
            )

    monkeypatch.setattr(_cli, "prepare_negative_prefixes", prepare_with_oracle)

    original_positive = _cli.synthesize_positive_mode

    def positive_with_oracle(
        cfg: Any,
        concepts: Any,
        seeds_train: Any,
        base_model: Any,
        base_sae: Any,
        target_model_meta: Any,
        *,
        seed_rows_sha256: str,
        cluster_contract: Any = None,
    ) -> Any:
        with oracle.stage_context(
            cfg,
            base_model=base_model,
            base_sae=base_sae,
            target_model_meta=target_model_meta,
            concepts=concepts,
            cluster_contract=cluster_contract,
        ):
            return original_positive(
                cfg,
                concepts,
                seeds_train,
                base_model,
                base_sae,
                target_model_meta,
                seed_rows_sha256=seed_rows_sha256,
                cluster_contract=cluster_contract,
            )

    monkeypatch.setattr(_cli, "synthesize_positive_mode", positive_with_oracle)

    original_negative = _cli.synthesize_negative_mode

    def negative_with_oracle(
        cfg: Any,
        concepts: Any,
        base_model: Any,
        base_sae: Any,
        target_model_meta: Any,
        *,
        cluster_contract: Any = None,
    ) -> Any:
        with oracle.stage_context(
            cfg,
            base_model=base_model,
            base_sae=base_sae,
            target_model_meta=target_model_meta,
            concepts=concepts,
            cluster_contract=cluster_contract,
        ):
            return original_negative(
                cfg,
                concepts,
                base_model,
                base_sae,
                target_model_meta,
                cluster_contract=cluster_contract,
            )

    monkeypatch.setattr(_cli, "synthesize_negative_mode", negative_with_oracle)

    original_verify = _stage0c_internals._verify_concept

    def verify_with_oracle(*args: Any, **kwargs: Any) -> Any:
        concept = args[1]
        candidate_index = kwargs.get("candidate_index")
        candidates = (
            candidate_index.candidates_for_feature(concept.feature_index)
            if candidate_index is not None
            else []
        )
        with oracle.feature_context(concept.feature_index):
            with oracle.candidate_context(candidates):
                return original_verify(*args, **kwargs)

    monkeypatch.setattr(_stage0c_internals, "_verify_concept", verify_with_oracle)


def _build_real_stage4_context_cache(
    cfg: DictConfig,
    *,
    run_tag: str,
    out_root: Path,
    oracle: RealGemmaOfflineOracle,
) -> Path:
    """Extract, convert, and validate the exact Stage 3e context cache.

    This deliberately uses the production extractor and cache converter rather
    than a synthetic provider. The target checkpoint is inspected through the
    same ``RuntimeState`` loader as Stage 4, and its sampling configuration is
    threaded into cache construction. For clustered runs, the exact resolved
    mask, cluster source, checkpoint pin, and feature-contract fingerprint are
    forwarded into both the cache producer and the resumed Stage 4 config.
    The current unclustered converter's historical RNG contract fixes the
    private selection seed at 42, so fail loudly if this real checkpoint ever
    stops matching that contract.
    """

    cluster_contract: ClusterFeatureContract | None = oracle.cluster_contract
    cfg_obj = _cli.cfg_to_dataclass(cfg)
    output_paths = paths.AxBenchBasePaths.from_config(cfg_obj)
    state_kwargs: dict[str, Any] = {"cfg": cfg_obj}
    if cluster_contract is not None:
        state_kwargs["cluster_contract"] = cluster_contract
    state = _cli.RuntimeState(**state_kwargs)
    extracted_feature_dir = out_root / "stage4_context_features"
    try:
        with state.encdec_model() as encdec_model:
            encdec_meta = state.encdec_meta()
            model_max_seq_len = _cli._encdec_model_sequence_len(encdec_model)
            loaded_meta = getattr(encdec_model, "_loaded_meta_data", None)
            if not isinstance(loaded_meta, dict):
                pytest.fail(
                    "Real encoder-decoder checkpoint did not expose its loaded "
                    "metadata; cannot prove Stage 4 cache-selection parity."
                )
            user_config = loaded_meta.get("user_config")
            if not isinstance(user_config, dict):
                pytest.fail(
                    "Real encoder-decoder checkpoint metadata has no user_config; "
                    "cannot prove Stage 4 cache-selection parity."
                )
            checkpoint_seed = user_config.get("seed")
            if checkpoint_seed != 42:
                pytest.fail(
                    "The ordinary eval-context converter preserves the historical "
                    "selection seed 42, but the real checkpoint records "
                    f"seed={checkpoint_seed!r}. Generalize the legacy converter's "
                    "seed contract before accepting this checkpoint."
                )

        if encdec_meta.n_sae_features is None:
            pytest.fail("Real encoder-decoder metadata has no n_sae_features")
        if encdec_meta.num_buckets is None:
            pytest.fail("Real encoder-decoder metadata has no num_buckets")
        if encdec_meta.feature_selection is None:
            pytest.fail("Real encoder-decoder metadata has no feature_selection")
        cache_feature_selection = dict(encdec_meta.feature_selection)

        parsed = parse_feature_dir(_FEATURE_DIR)
        extracted_feature_dir.mkdir(parents=True, exist_ok=False)
        with state.base_model() as base_model, state.base_sae() as base_sae:
            extraction_result = process_shard(
                str(output_paths.stage4_contexts_parquet()),
                base_model,
                base_sae,
                layer=parsed.layer,
                output_dir=str(extracted_feature_dir),
                context_window=model_max_seq_len,
                pooling=parsed.pooling,
                shard_position=0,
                gpu_batch_size=cfg_obj.seed_extraction_gpu_batch_size,
                cpu_batch_size=cfg_obj.seed_extraction_cpu_batch_size,
                nnterp_model=getattr(base_model, "nnterp_model", None),
                native_model=getattr(base_model, "native_model", None),
                segment_mode=SegmentMode.fixed_tokens,
                segment_size=parsed.segment_size,
                hook_name=_seed_pools._sae_hook_name(base_sae),
                activation_type=parsed.activation_type,
                dataset_phase=parsed.dataset_phase.value,
                context_independent_features=parsed.context_independent_features,
                ignore_bos_features=parsed.ignore_bos_features,
                run_l0_tier=_seed_pools._run_l0_tier(cfg_obj),
            )
        extracted_path = Path(extraction_result.output_path)
        if not extracted_path.is_file():
            pytest.fail(
                f"Real Stage 3e feature extraction did not commit {extracted_path}"
            )
    finally:
        state.close()

    statistics_sha = sha256_file(_FEATURE_STATISTICS_FILE)
    cache_root = out_root / "stage4_context_cache"
    identity = OmegaConf.to_container(cfg.conditioning.identity, resolve=True)
    if not isinstance(identity, dict):
        pytest.fail("conditioning.identity did not resolve to a mapping")
    identity["feature_selection_hash"] = None
    feature_mask = encdec_meta.feature_mask
    feature_clusters = None
    checkpoint_config: dict[str, Any] = {}
    metadata = {"source": "axbench-base-real-gemma-integration"}
    if cluster_contract is not None:
        # A clustered prompt cache has no current/reference pair. Preserve the
        # checkpoint's sampling policy but remove its inline-only delta weight;
        # clustered cache identity injects the canonical null key internally.
        cache_feature_selection.pop("delta_weights", None)
        cluster_provenance = cluster_contract.provenance_dict()
        feature_mask_path = cluster_provenance.get("feature_mask_path")
        if not isinstance(feature_mask_path, str) or not feature_mask_path:
            pytest.fail(
                "Clustered feature contract did not retain its path-backed feature_mask provenance"
            )
        feature_mask = {
            "mode": cluster_contract.feature_mask.mode.value,
            "path": feature_mask_path,
        }
        feature_clusters = {"path": str(cluster_contract.feature_clusters.source_path)}
        checkpoint_config = {
            "checkpoint_dir": str(cluster_contract.checkpoint.checkpoint_dir),
            "checkpoint_step": cluster_contract.checkpoint.step,
            "checkpoint_tag": cluster_contract.checkpoint.resolved_tag,
        }
        metadata["feature_contract_fingerprint"] = cluster_contract.feature_contract_fingerprint
    cache_cfg = OmegaConf.create(
        {
            "contexts_file": str(output_paths.stage4_contexts_jsonl()),
            "context_parquet_dir": str(output_paths.stage4_contexts_parquet().parent),
            "required_contexts_manifest": str(output_paths.stage4_contexts_manifest()),
            "feature_dir": str(extracted_feature_dir),
            "output_dir": str(cache_root),
            "n_sae_features": encdec_meta.n_sae_features,
            "num_buckets": encdec_meta.num_buckets,
            "source_bos_char_len": None,
            "extraction_tokenizer_id": cfg_obj.extraction_tokenizer_id,
            "stats_path": str(_FEATURE_STATISTICS_FILE),
            "feature_selection": cache_feature_selection,
            "negative_selection": encdec_meta.negative_selection,
            "feature_mask": feature_mask,
            "feature_clusters": feature_clusters,
            "feature_delta": None,
            "stats_source_sha256": statistics_sha,
            "seed": checkpoint_seed,
            **checkpoint_config,
            "identity": identity,
            "metadata": metadata,
            "shard_size": 1024,
            "overwrite": False,
            "zstd_level": 12,
            "model_max_seq_len": model_max_seq_len,
        }
    )
    manifest_path = _context_cache_cli.build_cache_from_config(cache_cfg)
    _context_cache_cli.validate_cache_from_config(cache_cfg)
    cache_manifest = load_context_cache_manifest(manifest_path.parent)
    feature_selection_hash = cache_manifest["identity"][
        "feature_selection_hash"
    ]

    # The resumed Stage 4 independently recomputes this identity from the
    # checkpoint selection config and statistics digest, then checks it against
    # the explicit hash and the committed cache manifest.
    cfg.conditioning.identity.feature_selection_hash = feature_selection_hash
    cfg.conditioning.feature_selection = cache_feature_selection
    cfg.conditioning.negative_selection = encdec_meta.negative_selection
    cfg.conditioning.feature_mask = feature_mask
    cfg.conditioning.feature_clusters = feature_clusters
    cfg.conditioning.feature_delta = None
    cfg.conditioning.stats_source_sha256 = statistics_sha
    cfg.conditioning.seed = checkpoint_seed
    if cluster_contract is not None:
        resumed_paths = parse_cluster_eval_paths(
            feature_mask=cfg.conditioning.feature_mask,
            feature_clusters=cfg.conditioning.feature_clusters,
        )
        if resumed_paths is None:
            pytest.fail("Clustered Stage 4 resume parsed as an unclustered config")
        resumed_mask, resumed_clusters = resumed_paths
        assert resumed_mask.canonical_dict() == feature_mask
        assert resumed_clusters.canonical_dict() == feature_clusters
    return manifest_path


@pytest.mark.very_slow
def test_full_pipeline_dev_tier_with_real_gemma_2b(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end dev-tier run with real Gemma-2b, real SAE, real nanochat d12.

    Mocks OpenRouter at the production cached ``requests.Session.post``
    boundary, with lower transport tripwires so no OpenRouter calls go out.
    Every other resource is real: base Gemma (downloaded from HF Hub on
    first run), GemmaScope SAE (same), nanochat d12 targets (from
    ``~/.cache/nanochat/``), SQLite caches, InferenceEngines.

    Output artifacts and the identity-addressed Neuronpedia cache land under
    pytest's temporary directory, independently of the real model/checkpoint
    cache, so the operator's cache is never polluted.
    """
    _require_nanochat_cache()
    _require_feature_dir()

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-offline")
    transport, oracle = _install_openrouter_oracle(monkeypatch)
    _bind_live_oracle_contexts(monkeypatch, oracle)

    # Point the pipeline at the real nanochat cache so load_model
    # resolves source="base" / source="encdec" to the real checkpoints.
    # We explicitly set the env var even when it's already set so the
    # test is robust to operators with a different default.
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(_NANOCHAT_CACHE))

    run_tag = "phase-9-real-gemma"
    effective_output_root = tmp_path / "axbench-output"
    out_root = effective_output_root / run_tag
    context_cache_root = out_root / "stage4_context_cache"
    statistics_sha = sha256_file(_FEATURE_STATISTICS_FILE)

    # Clean any leftover output from a prior aborted run.
    if out_root.exists():
        shutil.rmtree(out_root)

    cfg_dict: dict[str, Any] = {
        "feature_dir": str(_FEATURE_DIR),
        "feature_statistics_path": str(_FEATURE_STATISTICS_FILE),
        "decoder_only": {
            "source": "base",
            "model_tag": "d12_base",
            "step": None,
        },
        "encdec": {
            "source": "encdec",
            "model_tag": "d12_baseline",
            "step": None,
        },
        "device": "cuda" if _cuda_available() else "cpu",
        "output_root": str(effective_output_root),
        "run_tag": run_tag,
        "stage": "all",
        "run_negative_mode": True,
        # One real concept exercises every ordinary method family.  Using the
        # full dev tier caused common-feature dirty-seed ranking to turn this
        # compatibility gate into a multi-hour corpus benchmark.
        "max_concepts": 1,
        # This is an end-to-end model/integration gate, not an exhaustive
        # Neuronpedia-catalogue benchmark.  Keep a broad real candidate pool
        # while avoiding a fresh 16k-feature API scan on every isolated run.
        "concept_description_fetch_limit": 1024,
        "enable_concept_examples_qc": False,
        "verification_api_model": "inclusionai/ling-2.6-flash",
        # Negative-mode LoRA trains from one avoiding row per verified prefix,
        # independently of n_positives_per_concept.  Materialize the same
        # minimum number of verified prefixes so both steering modes exercise
        # the real LoRA path instead of failing at its corpus-size guard.
        "n_verified_prefixes_per_concept_target": LoraConfig().min_training_rows,
        "n_verification_seeds": 1,
        "min_consistency_passes": 1,
        "verification_api_prefix_wave_size": 1,
        # Stage 3 includes LoRA in the ordinary all-method profile.  Its
        # fail-loud training contract requires at least eight rows per class;
        # generating only one made this test stop before Stage 4 by design.
        "n_positives_per_concept": LoraConfig().min_training_rows,
        # Pin issue #1048's corrected fraction-to-raw-point contract in the
        # real-model gate rather than relying on the dataclass default alone.
        "accept_threshold": 0.3,
        # This fixture expects every deterministic Stage 1b oracle response
        # to pass on its first semantic attempt.
        "positive_max_attempts_per_seed": 1,
        "max_retries": 0,
        "synth": {"requests_per_minute": 0, "max_concurrent": 1},
        # All judge traffic is intercepted by the deterministic in-process
        # oracle above.  Preserve the real concurrent Stage 5 path without
        # making this offline acceptance gate sleep at the production API
        # admission rate.
        "judge": {"requests_per_minute": 0, "max_concurrent": 10},
        "qc_fallback_on_exhaust": "error",
        "validation_shards": [369],
        # Feature-DiffMean/GCG fair and roofline bags belong to BPB-Adv. This
        # real Base gate covers every directly executable Stage 4 method and
        # leaves the precomputed-bag placeholders fail-loud by construction.
        "disabled_methods": list(_BPB_ONLY_PRECOMPUTED_BAG_METHODS),
        "conditioning": _real_stage4_conditioning_config(
            context_cache_root,
            stats_source_sha256=statistics_sha,
        ),
    }
    cfg: DictConfig = OmegaConf.create(cfg_dict)
    output_paths = paths.AxBenchBasePaths(effective_output_root, run_tag)
    legacy_output_paths = test_paths.default_paths(run_tag)
    assert legacy_output_paths.run_output_root() != out_root
    assert not legacy_output_paths.run_output_root().exists()
    pipeline_completed = False

    try:
        # ``stage=all`` correctly refuses to invent a live context cache. Let
        # the production Stage 4 provider reach the real missing-cache failure
        # after Stage 3e commits its exact source artifacts.
        with pytest.raises(
            AxBenchBaseError,
            match="Failed to build the AxBench-Base Stage 4 encoder-decoder",
        ) as missing_cache_error:
            _cli._run(cfg)
        assert missing_cache_error.value.__cause__ is not None
        assert context_cache_root.as_posix() in str(missing_cache_error.value.__cause__)
        assert output_paths.stage4_contexts_jsonl().is_file()
        assert output_paths.stage4_contexts_parquet().is_file()
        assert output_paths.stage4_contexts_manifest().is_file()
        assert not output_paths.generations_dir().exists()

        cache_manifest_path = _build_real_stage4_context_cache(
            cfg,
            run_tag=run_tag,
            out_root=out_root,
            oracle=oracle,
        )
        assert cache_manifest_path.is_file()

        for stage in ("4", "5", "6"):
            cfg.stage = stage
            _cli._run(cfg)
        pipeline_completed = True

        # Every canonical artifact exists.
        assert output_paths.concepts_json().exists()
        assert output_paths.seeds_train().exists()
        assert output_paths.seeds_eval().exists()
        assert output_paths.negative_mode_prefixes().exists()
        assert output_paths.synth_training_positive().exists()
        assert output_paths.synth_training_negative().exists()
        assert output_paths.concept_features().exists()
        assert output_paths.concept_features_neg().exists()
        assert output_paths.generations_dir().is_dir()
        assert output_paths.judge_scores_dir().is_dir()
        assert output_paths.report_yaml().exists()
        assert output_paths.aggregation_report().exists()
        assert not legacy_output_paths.run_output_root().exists()

        # Prove the real Gemma + real SAE rows were accepted under the
        # corrected +30 raw-percentile-point contract and that the producer
        # committed the exact currency/algorithm provenance.
        positive_path = output_paths.synth_training_positive()
        positive_table = pq.read_table(positive_path)
        positive_contract = validate_positive_qc_contract(
            positive_path,
            consumer="real Gemma end-to-end acceptance test",
            schema=positive_table.schema,
            expected_accept_threshold=0.3,
        )
        assert positive_contract.accept_threshold_fraction == 0.3
        assert positive_contract.accept_threshold_points == 30.0
        rich_rows = [
            row
            for row in positive_table.to_pylist()
            if row["content_class"] == "concept_rich"
        ]
        assert len(rich_rows) == LoraConfig().min_training_rows
        assert all(row["qc_accepted"] is True for row in rich_rows)
        assert all(float(row["qc_delta"]) >= 30.0 for row in rich_rows)

        # The leaderboard must retain every ordinary Stage 4 method. A generic
        # header-plus-row check would allow all but one method to disappear.
        leaderboard_path = output_paths.leaderboard_csv()
        assert leaderboard_path.exists()
        leaderboard_lines = leaderboard_path.read_text(
            encoding="utf-8",
        ).splitlines()
        leaderboard_reader = csv.DictReader(leaderboard_lines)
        assert leaderboard_reader.fieldnames, (
            f"leaderboard.csv has no header: {leaderboard_lines!r}"
        )
        assert leaderboard_reader.fieldnames[0] == "method_name", (
            "leaderboard first column is not method_name: "
            f"{leaderboard_reader.fieldnames!r}"
        )
        leaderboard_rows = list(leaderboard_reader)
        _assert_exact_stage4_methods(
            {str(row["method_name"]) for row in leaderboard_rows}
        )

        # Every data row's concept_score lives in AxBench's [0, 2] range.
        for row in leaderboard_rows:
            if "concept_score" in row:
                raw = row["concept_score"]
                try:
                    score = float(raw)
                except ValueError:
                    # Non-numeric cells (e.g. NaN or empty) are acceptable
                    # so long as the aggregator emitted them deliberately.
                    continue
                assert 0.0 <= score <= 2.0, f"concept_score {score} out of [0, 2] in row {row!r}"

        # Report.yaml has a metadata section with our run_tag.
        report_text = output_paths.report_yaml().read_text(encoding="utf-8")
        assert run_tag in report_text, (
            f"report.yaml does not reference run_tag {run_tag!r}; content: {report_text[:500]!r}"
        )
    finally:
        try:
            transport.assert_all_handled()
            assert transport.top_level_calls == []
            assert transport.transport_tripwire_calls == []
            if pipeline_completed:
                _assert_exact_openrouter_request_classes(transport)
            for request in transport.requests:
                assert request.url == ("https://openrouter.ai/api/v1/chat/completions")
                assert request.payload["model"] == request.model
                _assert_openrouter_message_roles(request)
                assert set(request.payload) <= {"model", "messages", "max_tokens"}
                assert request.cache_key
                assert request.response_hash
                assert request.headers == {
                    "Authorization": "Bearer sk-or-test-offline",
                    "Content-Type": "application/json",
                }
                assert request.timeout == (10, 60)
                if request.request_class in {
                    "stage0c_definition",
                    "stage0c_continuation",
                }:
                    assert request.payload["max_tokens"] == 1024
                else:
                    assert "max_tokens" not in request.payload
        finally:
            transport.close()
            try:
                assert not hasattr(openrouter._thread_local, "session")
            finally:
                if out_root.exists():
                    shutil.rmtree(out_root)
