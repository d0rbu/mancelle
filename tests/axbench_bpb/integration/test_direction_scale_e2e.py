"""Constructed end-to-end proof for direction-owned max-activation scaling."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors import safe_open
from safetensors.torch import load_file

import nanochat.flash_attention as _fa_mod
import nanochat.gpt as _gpt_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines import _capture
from steering.eval.axbench_base.baselines import diffmean as diffmean_mod
from steering.eval.axbench_base.baselines.diffmean import train_diffmean_all
from steering.eval.axbench_base.config import AxBenchBaseConfig, LayerHooksConfig
from steering.eval.axbench_base.direction_scale import MAX_ACTIVATION_METADATA_KEY
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import Concept, SteeringMode
from steering.eval.axbench_bpb.intervention import (
    apply_intervention,
    build_intervention,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths

pytestmark = pytest.mark.slow

_VOCAB_SIZE = 128
_N_EMBD = 32
_LAYER = 0
_CONCEPT_ID = 0


class _StubTokenizer:
    """Reversible character tokenizer with nanochat's owned-BOS convention."""

    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        tokens = [(ord(char) % (_VOCAB_SIZE - 2)) + 2 for char in text]
        return ([prepend] if prepend is not None else []) + tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 2) for token_id in token_ids)


def _make_tiny_gpt() -> GPT:
    """Build the deterministic CPU GPT used for both fit and intervention."""

    t.manual_seed(0x1228)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=_VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=_N_EMBD,
        window_pattern="L",
    )
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=_VOCAB_SIZE)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _row(
    tokenizer: _StubTokenizer,
    *,
    content_class: str,
    prefix_text: str,
    output_text: str,
) -> dict[str, Any]:
    return {
        "concept_id": _CONCEPT_ID,
        "category": "positive" if content_class == "concept_rich" else "negative",
        "content_class": content_class,
        "prefix_text": prefix_text,
        "output_text": output_text,
        "output_concept": "constructed direction-scale concept",
        "n_prefix_nanochat_tokens": len(
            tokenizer.encode(prefix_text, prepend=tokenizer.get_bos_token_id())
        ),
        "gen_source": "test:constructed",
        "gen_seed": 0,
        "qc_feature_id": 7,
        "qc_seed_percentile": 0.0,
        "qc_output_percentile": 1.0,
        "qc_delta": 1.0,
        "qc_attempts": 1,
        "qc_accepted": True,
    }


def _write_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            schema_with_extraction_tokenizer_metadata(
                TRAINING_SCHEMA,
                "google/gemma-2-2b",
            ),
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_direction_scale_fit_artifact_consumer_and_hook_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the fitted statistic reaches the installed residual hook unchanged."""

    # A float32 model makes the hook-cast equality in spec §8 observable without
    # bf16 addition/subtraction rounding obscuring the semantic invariant.
    monkeypatch.setattr(_gpt_mod, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(_fa_mod, "_override_impl", FlashAttnImpl.SDPA)
    tokenizer = _StubTokenizer()
    monkeypatch.setattr(diffmean_mod, "get_tokenizer", lambda: tokenizer)

    cfg = AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path="unused-by-stage3.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        run_tag="direction_scale_e2e",
        device="cpu",
        output_root=str(tmp_path / "axbench_base"),
        verification_api_model=None,
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=_LAYER),
    )
    model = _make_tiny_gpt()
    concept = Concept(
        concept_id=_CONCEPT_ID,
        feature_index=7,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="constructed direction-scale concept",
        corpus_count=4,
        description_source="test",
    )
    rows = [
        _row(
            tokenizer,
            content_class="concept_rich",
            prefix_text="RICH_A",
            output_text="alpha alpha",
        ),
        _row(
            tokenizer,
            content_class="concept_rich",
            prefix_text="RICH_B",
            output_text="alpha beta",
        ),
        _row(
            tokenizer,
            content_class="concept_avoiding",
            prefix_text="AVOID_A",
            output_text="omega omega",
        ),
        _row(
            tokenizer,
            content_class="concept_avoiding",
            prefix_text="AVOID_B",
            output_text="omega delta",
        ),
    ]
    corpus_path = tmp_path / "direction_scale_e2e.parquet"
    _write_corpus(corpus_path, rows)

    [artifact_path] = train_diffmean_all(
        cfg=cfg,
        target_model=model,
        target_model_name="decoder_only",
        mode=SteeringMode.POSITIVE,
        concepts=[concept],
        training_corpus_path=corpus_path,
    )
    direction = load_file(str(artifact_path))["direction"]
    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        stored_value = float((handle.metadata() or {})[MAX_ACTIVATION_METADATA_KEY])

    fit_acts = [
        _capture._capture_per_row_continuation(
            model=model,
            layer=_LAYER,
            tokenizer=tokenizer,
            prefix_text=row["prefix_text"],
            output_text=row["output_text"],
            n_prefix_nanochat_tokens=row["n_prefix_nanochat_tokens"],
        )[1]
        for row in rows
    ]
    independently_recomputed = max(
        float((acts.detach().float().cpu() @ direction.detach().float().cpu()).max().item())
        for acts in fit_acts
    )
    # Spec §8 non-vacuity: recompute from captured residuals, not metadata.
    assert stored_value == independently_recomputed

    run_root = paths.AxBenchBasePaths.from_config(cfg).run_output_root()
    method = get_method_by_name("diffmean_decoder_only")
    intervention = build_intervention(
        method=method,
        concept=concept,
        coefficient=1.0,
        source_paths=SourcePaths(source_run_dir=run_root),
        diffmean_layer=_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=tokenizer,
        rng=random.Random(0),
        apply_max_act_scale=True,
    )

    prompt = t.tensor(
        [tokenizer.encode("HOOK PROOF", prepend=tokenizer.get_bos_token_id())],
        dtype=t.long,
    )
    transformer = model.transformer
    blocks = cast(t.nn.ModuleList, transformer["h"])
    block = blocks[_LAYER]
    observed: dict[str, t.Tensor] = {}

    def _capture_output(
        _module: t.nn.Module,
        _inputs: tuple[t.Tensor, ...],
        output: t.Tensor,
    ) -> None:
        observed["residual"] = output.detach().clone()

    baseline_handle = block.register_forward_hook(_capture_output)
    unhooked_logits = model(prompt)
    unhooked_residual = observed.pop("residual")
    baseline_handle.remove()

    with apply_intervention(model, intervention):
        steered_handle = block.register_forward_hook(_capture_output)
        hooked_logits = model(prompt)
        hooked_residual = observed.pop("residual")
        steered_handle.remove()

    perturbation = hooked_residual - unhooked_residual
    expected = stored_value * direction
    # Spec §8 applied-hook proof: every token receives value × unit d, within
    # the one-ULP noise introduced by recovering the addend via f32 subtraction.
    t.testing.assert_close(
        perturbation,
        expected[None, None, :].expand_as(perturbation),
        rtol=2e-6,
        atol=2e-7,
    )
    t.testing.assert_close(
        perturbation[0, 0].float().norm(),
        t.tensor(stored_value, dtype=t.float32),
        rtol=2e-6,
        atol=2e-6,
    )

    max_fit_residual_norm = max(
        float(acts.detach().float().norm(dim=-1).max().item()) for acts in fit_acts
    )
    # Spec §8 / Cauchy-Schwarz: max(h·d) cannot exceed max ||h|| for unit d.
    assert stored_value <= max_fit_residual_norm
    # Spec §8 frontier smoke proof: the live hooked model executes and changes.
    assert not t.equal(hooked_logits, unhooked_logits)
