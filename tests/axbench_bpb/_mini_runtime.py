"""Minimal RuntimeState builder for axbench_bpb DDP tests.

Stubs out heavy loaders (base Gemma, SAE) so distributed tests can run
on CPU without GPU/network access.

Slice 17 audit (2026-05-05): replaced direct-private-attr writes with a
``_StubRuntimeState`` subclass that overrides the public lazy-loader
methods. This keeps the test surface aligned with the public API even
when the cache slot names on ``RuntimeState`` are renamed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from steering.eval.axbench_bpb.config import (
    AggregationConfig,
    AxBenchBpbConfig,
    ContextCacheConfig,
)
from steering.eval.axbench_bpb.runtime_state import RuntimeState


class _StubRuntimeState(RuntimeState):
    """RuntimeState subclass that returns mocks for the heavy accessors.

    Overrides the public lazy-loader API (``base_gemma_and_sae``,
    ``extraction_tokenizer``, ``decoder_only``) instead of writing to
    private cache slots. The stubbed accessors are deterministic: every
    call returns the same mock instances cached on first call.

    Test-only contract: ``encdec`` and ``inactive_sampler`` are NOT
    overridden — distributed tests that need those must construct a
    richer subclass or monkey-patch the methods individually.

    The ``decoder_only`` override exists because slice-17's P1.B runtime
    guards (``assert_tokenizer_checksum_matches`` /
    ``assert_bos_lengths_match``) run at the top of Stage 0 and resolve
    ``decoder_only()``; without this stub rank 0 would crash with
    "Unknown source: 'x'" before reaching the post-Stage-0 barrier,
    causing cooperating ranks to see "Connection closed by peer" on
    their barrier wait.
    """

    def __init__(
        self,
        *,
        cfg: AxBenchBpbConfig,
        ddp_rank: int,
        ddp_world_size: int,
        d_sae: int = 16,
    ) -> None:
        super().__init__(
            cfg=cfg, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size,
        )
        # Build canned return values once and cache them. The lazy-loader
        # overrides return these objects on every call (matching
        # RuntimeState's public contract: repeated calls return the same
        # instance).
        self._stub_base_model = MagicMock()
        self._stub_base_sae = MagicMock(cfg=MagicMock(d_sae=d_sae))
        self._stub_extraction_tokenizer = MagicMock()
        # The decoder-only bundle needs at least a ``tokenizer`` attribute
        # because slice-17's runtime guards read ``decoder_bundle.tokenizer``.
        self._stub_decoder_only_bundle = MagicMock(tokenizer=MagicMock())

    def base_gemma_and_sae(
        self,
        *,
        offload_transformer_tail_after: int | None = None,
    ) -> tuple[Any, Any]:
        del offload_transformer_tail_after  # stub: always returns the canned mocks
        return self._stub_base_model, self._stub_base_sae

    def extraction_tokenizer(self) -> Any:
        return self._stub_extraction_tokenizer

    def decoder_only(self) -> Any:
        return self._stub_decoder_only_bundle


def build_mini_runtime_state(*, run_root: Path, rank: int, world_size: int) -> RuntimeState:
    """Construct a stripped-down RuntimeState for distributed tests.

    The returned state has heavy loaders pre-stubbed so callers don't trip
    on ``base_gemma_and_sae()`` or ``extraction_tokenizer()`` accessors.
    """
    source_run_dir = run_root / "source"
    source_run_dir.mkdir(parents=True, exist_ok=True)
    (source_run_dir / "concepts_stage1c.json").write_text(
        json.dumps(
            [
                {
                    "concept_id": 0,
                    "feature_index": 7,
                    "layer": 8,
                    "sae_source": "gemmascope",
                    "description": "mini concept",
                    "corpus_count": 10,
                    "description_source": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = AxBenchBpbConfig(
        run_tag="test",
        source_run_dir=str(source_run_dir),
        output_root=str(run_root / "output"),
        feature_dir=str(run_root / "features"),
        feature_statistics_path=str(run_root / "stats.safetensors"),
        decoder_only={"source": "x", "model_tag": "y", "step": 1},
        encdec={"source": "x", "model_tag": "y", "step": 1},
        device="cpu",
        neuronpedia_source="8-gemmascope-res-16k",
        neuronpedia_model_id="google/gemma-2-2b",
        stage="prepare_context_cache",
        context_cache=ContextCacheConfig(num_buckets=16, segment_size=128),
        aggregation=AggregationConfig(),
    )
    return _StubRuntimeState(
        cfg=cfg, ddp_rank=rank, ddp_world_size=world_size,
    )


__all__ = ["build_mini_runtime_state"]
