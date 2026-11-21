"""Real Gemma-3-4B-PT + GemmaScope-2 acceptance gate for Issue #1051.

Run only when the audited feature/run roots and a free 24 GB CUDA device are
available::

    AXBENCH_ISSUE_1051_RUN_DIR=/abs/path/to/l22_f65k_clustered_std50 \
    AXBENCH_ISSUE_1051_FEATURE_DIR=/abs/path/to/\
      s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/\
      context_independent/no_bos \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      uv run pytest \
        tests/axbench_base/integration/test_issue_1051_real_gpu.py \
        -v -m 'very_slow and requires_feature_data'

The synth transport is not invoked. The test drives the production Stage 1b
attempt handler through the real extraction tokenizer, full base model,
65,536-feature SAE, real statistics table, and real 2,499-character failing
seed prefix.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors import safe_open

from steering.eval.axbench_base.config import AxBenchBaseConfig, validate_config
from steering.eval.axbench_base.extraction import (
    _infer_n_sae_features,
    build_percentile_table_from_stats,
)
from steering.eval.axbench_base.prefix_text import window_prefix_text
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.synth import positive_mode
from steering.eval.axbench_base.synth.response_parser import (
    get_extraction_tokenizer,
)
from steering.eval.axbench_base.types import Concept, SeedRow
from tests.axbench_base.positive_mode_reference import (
    handle_positive_attempt_result,
)
from tests.axbench_base.stage1_protocol_helpers import frame_stage1_response

_SEED_INDEX = 20_968
_FEATURE_INDEX = 488
_MAX_PREFIX_CHARS = 2_048
_PREFIX_SHA256 = (
    "a5d27eae1b2d30dbbe8e15c03243eb5243551fa2b8dcc7d1b60763362934045b"
)
_TOKENIZER_ID = "google/gemma-3-4b-pt"
_CONTINUATION = (
    "A separate engineering team documented a new calibration procedure, "
    "then independently verified every measurement before publication."
)


def _configured_path(
    env_name: str,
    default: Path,
) -> tuple[Path, bool]:
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser().resolve(), True
    return default.resolve(), False


def _real_inputs() -> tuple[Path, Path, bool]:
    repo_root = Path(__file__).resolve().parents[3]
    run_dir, run_explicit = _configured_path(
        "AXBENCH_ISSUE_1051_RUN_DIR",
        (
            repo_root
            / "nanochat_cache/eval/axbench_base/l22_f65k_clustered_std50"
        ),
    )
    feature_dir, feature_explicit = _configured_path(
        "AXBENCH_ISSUE_1051_FEATURE_DIR",
        (
            repo_root
            / "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
            "m_gemma-3-4b-pt/base/context_independent/no_bos"
        ),
    )
    return run_dir, feature_dir, run_explicit or feature_explicit


def _real_prefix(path: Path) -> str:
    with safe_open(path, framework="pt", device="cpu") as handle:
        offsets = handle.get_tensor("prefix_text_offsets")
        start = int(offsets[_SEED_INDEX].item())
        end = int(offsets[_SEED_INDEX + 1].item())
        payload = (
            handle.get_tensor("prefix_text_bytes")[start:end]
            .numpy()
            .tobytes()
        )
    return payload.decode("utf-8", errors="strict")


def _placeholder_seed(prefix_text: str) -> SeedRow:
    """Minimal immutable carrier; the handler reads only its resolved index."""
    return SeedRow(
        shard_idx=369,
        doc_idx=0,
        source_nanochat_token_count=2,
        prefix_source_token_start=0,
        prefix_source_token_end=1,
        prefix_text=prefix_text,
        prefix_content_nanochat_tokens=[2],
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="",
        continuation_content_nanochat_tokens=[3],
    )


def _concept() -> Concept:
    return Concept(
        concept_id=0,
        feature_index=_FEATURE_INDEX,
        layer=22,
        sae_source="22-gemmascope-2-res-65k",
        description="first name last name",
        corpus_count=20_325_208,
        description_source="neuronpedia",
    )


def _cfg(feature_dir: Path) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir=str(feature_dir),
        feature_statistics_path=str(
            feature_dir / "feature_statistics.safetensors"
        ),
        decoder_only={
            "source": "base",
            "model_tag": "unused-real-gpu-gate",
            "step": 1,
        },
        encdec={
            "source": "encdec",
            "model_tag": "unused-real-gpu-gate",
            "step": 1,
        },
        device="cuda",
        output_root=None,
        run_tag="issue-1051-real-gpu-readonly",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_model_id="gemma-3-4b",
        neuronpedia_source="22-gemmascope-2-res-65k",
        extraction_tokenizer_id=_TOKENIZER_ID,
        min_prefix_chars=100,
        max_prefix_chars=_MAX_PREFIX_CHARS,
        output_length=64,
    )


def _run_real_attempt(
    *,
    cfg: AxBenchBaseConfig,
    prefix_text: str,
    full_prefix: str,
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    tokenizer: Any,
    seed_percentile: float,
) -> dict[str, Any]:
    seed = _placeholder_seed(full_prefix)
    item = positive_mode._PositivePending(
        order=0,
        selected=positive_mode._SelectedPositiveSeed(
            seed_idx=_SEED_INDEX,
            seed=seed,
            prefix_text=prefix_text,
            seed_percentile=seed_percentile,
            stored_clean=True,
        ),
    )
    concept = _concept()
    request = positive_mode._stage1b_continuation_request(
        positive_mode.load_template("synth_continue_with_concept.txt"),
        concept=concept,
        prefix_text=prefix_text,
        seed_idx=_SEED_INDEX,
        attempt=0,
        content_class="concept_rich",
        cluster_target=None,
    )
    result = positive_mode.SynthResult(
        index=0,
        response_text=frame_stage1_response(request.prompt, _CONTINUATION),
    )
    handle_positive_attempt_result(
        result,
        item,
        request,
        cfg,
        concept,
        base_model,
        base_sae,
        percentile_table,
        tokenizer,
        "real-gpu:fixture",
        sae_layer=22,
        attempt=0,
    )
    row = item.accepted_row
    assert row is not None
    return row


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_issue_1051_bounded_prefix_real_stage1b_qc_is_repeat_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, feature_dir, explicitly_configured = _real_inputs()
    seed_path = run_dir / "seeds_train.safetensors"
    stats_path = feature_dir / "feature_statistics.safetensors"
    required = [seed_path, stats_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        if explicitly_configured:
            pytest.fail(f"Audited Issue #1051 GPU inputs are missing: {missing}")
        pytest.skip(
            "set AXBENCH_ISSUE_1051_RUN_DIR and "
            "AXBENCH_ISSUE_1051_FEATURE_DIR for the real GPU gate"
        )

    nanochat_base = run_dir.parents[2]
    nanochat_tokenizer = nanochat_base / "tokenizer/tokenizer.pkl"
    if not nanochat_tokenizer.is_file():
        if explicitly_configured:
            pytest.fail(
                f"Audited nanochat tokenizer is missing: {nanochat_tokenizer}"
            )
        pytest.skip("set the audited run directory with its nanochat tokenizer")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(nanochat_base))

    full_prefix = _real_prefix(seed_path)
    assert len(full_prefix) == 2_499
    assert hashlib.sha256(
        full_prefix.encode("utf-8")
    ).hexdigest() == _PREFIX_SHA256
    bounded_prefix = window_prefix_text(
        full_prefix,
        len(full_prefix),
        _MAX_PREFIX_CHARS,
    )
    assert bounded_prefix == full_prefix[-_MAX_PREFIX_CHARS:]
    assert len(bounded_prefix) == _MAX_PREFIX_CHARS

    cfg = _cfg(feature_dir)
    validate_config(cfg)
    tokenizer = get_extraction_tokenizer(_TOKENIZER_ID)
    bounded_token_ids = tokenizer.encode(
        bounded_prefix,
        add_special_tokens=True,
    )
    assert len(bounded_token_ids) == 428
    assert len(bounded_token_ids) < cfg.extraction_max_tokens

    state = RuntimeState(cfg=cfg)
    try:
        with state.base_model() as base_model:
            with state.base_sae() as base_sae:
                assert _infer_n_sae_features(base_sae) == 65_536
                percentile_table = build_percentile_table_from_stats(
                    cfg.feature_statistics_path,
                    n_sae_features=65_536,
                )
                seed_percentile = positive_mode.fc_percentile_on_prefix(
                    bounded_prefix,
                    _FEATURE_INDEX,
                    base_model,
                    base_sae,
                    percentile_table,
                    sae_layer=22,
                    tokenizer_id=cfg.extraction_tokenizer_id,
                )
                first = _run_real_attempt(
                    cfg=cfg,
                    prefix_text=bounded_prefix,
                    full_prefix=full_prefix,
                    base_model=base_model,
                    base_sae=base_sae,
                    percentile_table=percentile_table,
                    tokenizer=tokenizer,
                    seed_percentile=seed_percentile,
                )
                second = _run_real_attempt(
                    cfg=cfg,
                    prefix_text=bounded_prefix,
                    full_prefix=full_prefix,
                    base_model=base_model,
                    base_sae=base_sae,
                    percentile_table=percentile_table,
                    tokenizer=tokenizer,
                    seed_percentile=seed_percentile,
                )
    finally:
        state.close()

    assert first == second
    assert first["prefix_text"] == bounded_prefix
    assert first["prefix_text"] != full_prefix
    assert first["output_text"] == _CONTINUATION
    assert first["qc_feature_id"] == _FEATURE_INDEX
    assert first["qc_attempts"] == 1
    assert first["qc_accepted"] in {True, False}
    assert t.isfinite(t.tensor(first["qc_seed_percentile"]))
    assert t.isfinite(t.tensor(first["qc_output_percentile"]))
    assert t.isfinite(t.tensor(first["qc_delta"]))
    assert first["qc_delta"] == (
        first["qc_output_percentile"] - first["qc_seed_percentile"]
    )
    assert first["n_prefix_nanochat_tokens"] == (
        positive_mode._exact_output_start(
            bounded_prefix,
            first["output_text"],
        )
    )
