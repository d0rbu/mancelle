"""Stage 3 (training) end-to-end integration tests.

Exercises the four Stage 3 baselines on a tiny CPU GPT and synthetic
training rows:

* Stage 3a ``train_diffmean_all`` — DiffMean direction per concept.
* Stage 3b ``train_linear_probe_all`` — L1-logreg direction per concept.
* Stage 3c ``write_prompting_descriptions`` — concept_id -> description JSON.
* Stage 3d ``train_lora_all`` — per-concept LoRA adapter.

A shared ``_make_tiny_gpt`` fixture + stub nanochat tokenizer keep
every test under 30 s. The LoRA test monkey-patches the
flash-attention backend to SDPA because FlexAttention can't backward
on CPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
from safetensors.torch import load_file

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.baselines import (
    diffmean as diffmean_mod,
)
from steering.eval.axbench_base.baselines import (
    linear_probe as linear_probe_mod,
)
from steering.eval.axbench_base.baselines.diffmean import train_diffmean_all
from steering.eval.axbench_base.baselines.linear_probe import train_linear_probe_all
from steering.eval.axbench_base.baselines.lora import training as lora_training_mod
from steering.eval.axbench_base.baselines.lora.training import train_lora_all
from steering.eval.axbench_base.baselines.prompting import (
    write_prompting_descriptions,
)
from steering.eval.axbench_base.config import LayerHooksConfig, LoraConfig
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.negative_mode import NEG_TRAINING_SCHEMA
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import AxBenchBaseError, Concept, SteeringMode
from tests.axbench_base import paths as test_paths

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Tiny GPT fixture
# ---------------------------------------------------------------------------


TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_BOS_ID: int = 1
TINY_LAYER: int = 0  # 2-layer fixture, so layer-0 is valid


@pytest.fixture(autouse=True)
def _force_sdpa() -> Iterator[None]:
    """Force SDPA attention backend: FlexAttention can't backward on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _make_tiny_gpt() -> GPT:
    """Tiny CPU GPT in eval mode, deterministic init."""
    t.manual_seed(0xD46F)
    config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


class _StubNanochatTokenizer:
    """Character-hashing nanochat-like tokenizer that matches the real
    BOS-prepend convention.
    """

    def __init__(
        self, vocab_size: int = TINY_VOCAB, bos_id: int = TINY_BOS_ID,
    ) -> None:
        self.vocab_size = vocab_size
        self.bos_id = bos_id

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        base = [(ord(c) % (self.vocab_size - 2)) + 2 for c in text]
        if prepend is not None:
            return [prepend] + base
        return base

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 2) for token_id in token_ids)


def _install_stub_tokenizer_diffmean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diffmean_mod, "get_tokenizer", lambda: _StubNanochatTokenizer(),
    )


def _install_stub_tokenizer_linear_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        linear_probe_mod, "get_tokenizer", lambda: _StubNanochatTokenizer(),
    )


def _install_stub_tokenizer_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lora_training_mod, "get_tokenizer", lambda: _StubNanochatTokenizer(),
    )


# ---------------------------------------------------------------------------
# Row construction + corpus writers
# ---------------------------------------------------------------------------


def _pos_row(
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "category": "positive" if content_class == "concept_rich" else "negative",
        "content_class": content_class,
        "prefix_text": prefix_text,
        "output_text": output_text,
        "output_concept": "stub concept",
        "n_prefix_nanochat_tokens": n_prefix_nanochat_tokens,
        "gen_source": "test:stub",
        "gen_seed": 0,
        "qc_feature_id": 7,
        "qc_seed_percentile": 0.0,
        "qc_output_percentile": 0.0,
        "qc_delta": 0.0,
        "qc_attempts": 1,
        "qc_accepted": True,
    }


def _neg_row(
    concept_id: int,
    content_class: str,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
) -> dict[str, Any]:
    base = _pos_row(
        concept_id, content_class, prefix_text, output_text, n_prefix_nanochat_tokens,
    )
    base.update({
        "prefix_source_shard": 0,
        "prefix_source_doc": 0,
        "prefix_source_seg_idx": 0,
        "qc_zero_activation_verified": content_class == "concept_avoiding",
    })
    return base


def _write_positive_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=schema_with_stage1_continuation_contract(
                schema_with_positive_qc_contract(
                    schema_with_extraction_tokenizer_metadata(
                        TRAINING_SCHEMA,
                        "google/gemma-2-2b",
                    ),
                    accept_threshold=0.3,
                ),
                role="stage1b_positive",
            ),
        ),
        path,
    )


def _write_negative_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=schema_with_stage1_continuation_contract(
                schema_with_extraction_tokenizer_metadata(
                    NEG_TRAINING_SCHEMA,
                    "google/gemma-2-2b",
                ),
                role="stage1c_negative",
            ),
        ),
        path,
    )


def _concept(concept_id: int, feature_index: int = 7) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=feature_index,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"concept {concept_id} description",
        corpus_count=100,
        description_source="neuronpedia",
    )


def _n(tok: _StubNanochatTokenizer, text: str) -> int:
    return len(tok.encode(text, prepend=tok.get_bos_token_id()))


def _build_tiny_positive_rows(
    concepts: list[Concept], tok: _StubNanochatTokenizer,
) -> list[dict[str, Any]]:
    """Build a tiny positive-mode corpus: per-concept rich + avoiding rows.

    Mirrors the post-refactor Stage-1b contract: each core prefix owns BOTH a
    ``concept_rich`` and a paired ``concept_avoiding`` completion at the real
    ``concept_id`` (no shared ``concept_id=-1`` global-negatives pool). The
    avoiding rows are the per-concept negatives consumed by DiffMean/probe.
    """
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        for i in range(4):
            prefix_text = f"POS_c{concept.concept_id}_{i}"
            n_prefix = _n(tok, prefix_text)
            rows.append(
                _pos_row(
                    concept_id=concept.concept_id,
                    content_class="concept_rich",
                    prefix_text=prefix_text,
                    output_text=f"POS OUTPUT_{i}",
                    n_prefix_nanochat_tokens=n_prefix,
                )
            )
            rows.append(
                _pos_row(
                    concept_id=concept.concept_id,
                    content_class="concept_avoiding",
                    prefix_text=prefix_text,
                    output_text=f"AVOID OUTPUT_{i}",
                    n_prefix_nanochat_tokens=n_prefix,
                )
            )
    return rows


def _build_tiny_negative_rows(
    concepts: list[Concept], tok: _StubNanochatTokenizer,
) -> list[dict[str, Any]]:
    """Build a tiny negative-mode corpus: per-concept rich + avoiding rows."""
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        for i in range(4):
            rows.append(
                _neg_row(
                    concept_id=concept.concept_id,
                    content_class="concept_rich",
                    prefix_text=f"RICH_c{concept.concept_id}_{i}",
                    output_text=f"RICH_{i}",
                    n_prefix_nanochat_tokens=_n(
                        tok, f"RICH_c{concept.concept_id}_{i}",
                    ),
                )
            )
            rows.append(
                _neg_row(
                    concept_id=concept.concept_id,
                    content_class="concept_avoiding",
                    prefix_text=f"AVOID_c{concept.concept_id}_{i}",
                    output_text=f"AVOID_{i}",
                    n_prefix_nanochat_tokens=_n(
                        tok, f"AVOID_c{concept.concept_id}_{i}",
                    ),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Stage 3a tests
# ---------------------------------------------------------------------------


def test_stage_3_trainers_reject_synth_parquet_missing_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_missing_tokenizer_metadata",
        device="cpu",
    ).replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
        lora=LoraConfig(min_training_rows=1),
    )
    corpus = tmp_run_root / "missing_metadata.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [_pos_row(0, "concept_rich", "prefix", "output", 1)],
            schema=TRAINING_SCHEMA,
        ),
        corpus,
    )
    concept = _concept(0)
    unused_model = nn.Module()

    with pytest.raises(
        AxBenchBaseError,
        match="Stage 3a DiffMean: .*missing 'extraction_tokenizer_id' metadata",
    ):
        train_diffmean_all(
            cfg=cfg,
            target_model=unused_model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[concept],
            training_corpus_path=corpus,
        )
    with pytest.raises(
        AxBenchBaseError,
        match="Stage 3b LinearProbe: .*missing 'extraction_tokenizer_id' metadata",
    ):
        train_linear_probe_all(
            cfg=cfg,
            target_model=unused_model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[concept],
            training_corpus_path=corpus,
        )
    with pytest.raises(
        AxBenchBaseError,
        match="Stage 3d LoRA: .*missing 'extraction_tokenizer_id' metadata",
    ):
        train_lora_all(
            cfg=cfg,
            target_model=unused_model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[concept],
            training_corpus_path=corpus,
        )


def test_stage_3_trainers_reject_synth_parquet_mismatched_tokenizer_metadata(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_mismatch_tokenizer_metadata",
        device="cpu",
        extraction_tokenizer_id="custom/current-tokenizer",
    ).replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
        lora=LoraConfig(min_training_rows=1),
    )
    corpus = tmp_run_root / "mismatch_metadata.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [_pos_row(0, "concept_rich", "prefix", "output", 1)],
            schema=schema_with_extraction_tokenizer_metadata(
                TRAINING_SCHEMA,
                "custom/stale-tokenizer",
            ),
        ),
        corpus,
    )
    concept = _concept(0)
    unused_model = nn.Module()

    with pytest.raises(
        AxBenchBaseError,
        match="custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        train_diffmean_all(
            cfg=cfg,
            target_model=unused_model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[concept],
            training_corpus_path=corpus,
        )
    with pytest.raises(
        AxBenchBaseError,
        match="custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        train_linear_probe_all(
            cfg=cfg,
            target_model=unused_model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[concept],
            training_corpus_path=corpus,
        )
    with pytest.raises(
        AxBenchBaseError,
        match="custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        train_lora_all(
            cfg=cfg,
            target_model=unused_model,
            target_model_name="decoder_only",
            mode=SteeringMode.POSITIVE,
            concepts=[concept],
            training_corpus_path=corpus,
        )


def test_stage_3_diffmean_produces_one_direction_per_concept(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 3a DiffMean writes one safetensors direction per concept."""
    _install_stub_tokenizer_diffmean(monkeypatch)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_diffmean",
        device="cpu",
    ).replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
    )
    model = _make_tiny_gpt()
    tok = _StubNanochatTokenizer()
    concepts = [_concept(0), _concept(1)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    out_paths = train_diffmean_all(
        cfg=cfg,
        target_model=model,
        target_model_name="decoder_only",
        mode=SteeringMode.POSITIVE,
        concepts=concepts,
        training_corpus_path=corpus,
    )

    assert len(out_paths) == len(concepts)
    for concept, out_path in zip(concepts, out_paths, strict=True):
        expected = test_paths.default_paths("stage3_diffmean").diffmean_direction(
            target_model="decoder_only", mode=SteeringMode.POSITIVE, concept_id=concept.concept_id
        )
        assert out_path == expected
        assert out_path.exists()
        payload = load_file(str(out_path))
        assert payload["direction"].shape == (TINY_N_EMBD,)


def test_stage_3_linear_probe_produces_one_direction_per_concept(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 3a LinearProbe writes direction+coef+intercept per concept."""
    _install_stub_tokenizer_linear_probe(monkeypatch)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_linear_probe",
        device="cpu",
    ).replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
    )
    model = _make_tiny_gpt()
    tok = _StubNanochatTokenizer()
    concepts = [_concept(0), _concept(1)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    out_paths = train_linear_probe_all(
        cfg=cfg,
        target_model=model,
        target_model_name="decoder_only",
        mode=SteeringMode.POSITIVE,
        concepts=concepts,
        training_corpus_path=corpus,
    )

    assert len(out_paths) == len(concepts)
    for concept, out_path in zip(concepts, out_paths, strict=True):
        expected = test_paths.default_paths("stage3_linear_probe").linear_probe_direction(
            target_model="decoder_only", mode=SteeringMode.POSITIVE, concept_id=concept.concept_id
        )
        assert out_path == expected
        assert out_path.exists()
        payload = load_file(str(out_path))
        assert payload["direction"].shape == (TINY_N_EMBD,)
        assert payload["coef"].shape == (1, TINY_N_EMBD)
        assert payload["intercept"].shape == (1,)


def test_stage_3_prompting_produces_descriptions_json(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
) -> None:
    """Stage 3c writes ``{mode}_descriptions.json`` for both modes."""
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_prompting",
        run_negative_mode=True,
    )
    concepts = [
        _concept(0),
        _concept(1),
        _concept(2),
    ]

    out_paths = write_prompting_descriptions(cfg, concepts)

    assert len(out_paths) == 2
    pos_path = test_paths.default_paths("stage3_prompting").prompting_descriptions(SteeringMode.POSITIVE)
    neg_path = test_paths.default_paths("stage3_prompting").prompting_descriptions(SteeringMode.NEGATIVE)
    assert pos_path in out_paths
    assert neg_path in out_paths
    assert pos_path.exists()
    assert neg_path.exists()
    pos_data = json.loads(pos_path.read_text(encoding="utf-8"))
    neg_data = json.loads(neg_path.read_text(encoding="utf-8"))
    assert pos_data == {
        "0": "concept 0 description",
        "1": "concept 1 description",
        "2": "concept 2 description",
    }
    assert neg_data == pos_data


# ---------------------------------------------------------------------------
# Stage 3b tests (LoRA)
# ---------------------------------------------------------------------------


def test_stage_3_lora_produces_one_adapter_per_concept(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 3b LoRA writes one adapter safetensors per concept."""
    _install_stub_tokenizer_lora(monkeypatch)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_lora",
        device="cpu",
    ).replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
        lora=LoraConfig(
            rank=2,
            alpha=4.0,
            lr=1e-3,
            steps=2,
            batch_size=2,
            min_training_rows=2,
        ),
    )
    model = _make_tiny_gpt()
    tok = _StubNanochatTokenizer()
    concepts = [_concept(0), _concept(1)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    out_paths = train_lora_all(
        cfg=cfg,
        target_model=model,
        target_model_name="decoder_only",
        mode=SteeringMode.POSITIVE,
        concepts=concepts,
        training_corpus_path=corpus,
    )

    assert len(out_paths) == len(concepts)
    for concept, out_path in zip(concepts, out_paths, strict=True):
        expected = test_paths.default_paths("stage3_lora").lora_adapter(
            mode=SteeringMode.POSITIVE, concept_id=concept.concept_id
        )
        assert out_path == expected
        assert out_path.exists()
        state = load_file(str(out_path))
        # Four sublayers per block -> 4 * 2 = 8 tensors.
        assert len(state) == 8
        for key in state:
            assert key.startswith(f"layer_{TINY_LAYER}_")


def test_stage_3_all_baselines_run_for_both_modes(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run DiffMean + LinearProbe + Prompting + LoRA for both modes end-to-end.

    Uses a positive-mode corpus for ``mode=SteeringMode.POSITIVE`` and a
    negative-mode corpus for ``mode=SteeringMode.NEGATIVE``. Asserts that every
    expected output file exists after all four baselines have run on
    both corpora.
    """
    _install_stub_tokenizer_diffmean(monkeypatch)
    _install_stub_tokenizer_linear_probe(monkeypatch)
    _install_stub_tokenizer_lora(monkeypatch)

    cfg = make_valid_axbench_cfg(
        run_tag="stage3_all_modes",
        device="cpu",
        run_negative_mode=True,
    ).replace(
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
        lora=LoraConfig(
            rank=2,
            alpha=4.0,
            lr=1e-3,
            steps=2,
            batch_size=2,
            min_training_rows=2,
        ),
    )
    model = _make_tiny_gpt()
    tok = _StubNanochatTokenizer()
    concepts = [_concept(0), _concept(1)]

    pos_corpus = tmp_run_root / "pos_corpus.parquet"
    neg_corpus = tmp_run_root / "neg_corpus.parquet"
    _write_positive_corpus(pos_corpus, _build_tiny_positive_rows(concepts, tok))
    _write_negative_corpus(neg_corpus, _build_tiny_negative_rows(concepts, tok))

    for mode, corpus in (
        (SteeringMode.POSITIVE, pos_corpus),
        (SteeringMode.NEGATIVE, neg_corpus),
    ):
        train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=mode,
            concepts=concepts,
            training_corpus_path=corpus,
        )
        train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=mode,
            concepts=concepts,
            training_corpus_path=corpus,
        )
        train_lora_all(
            cfg=cfg,
            target_model=model,
            target_model_name="decoder_only",
            mode=mode,
            concepts=concepts,
            training_corpus_path=corpus,
        )
    write_prompting_descriptions(cfg, concepts)

    for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
        for concept in concepts:
            dm_path = test_paths.default_paths("stage3_all_modes").diffmean_direction(
                target_model="decoder_only", mode=mode, concept_id=concept.concept_id
            )
            lp_path = test_paths.default_paths("stage3_all_modes").linear_probe_direction(
                target_model="decoder_only", mode=mode, concept_id=concept.concept_id
            )
            lora_path = test_paths.default_paths("stage3_all_modes").lora_adapter(
                mode=mode, concept_id=concept.concept_id
            )
            assert dm_path.exists(), f"missing DiffMean: {dm_path}"
            assert lp_path.exists(), f"missing LinearProbe: {lp_path}"
            assert lora_path.exists(), f"missing LoRA: {lora_path}"
        pr_path = test_paths.default_paths("stage3_all_modes").prompting_descriptions(mode)
        assert pr_path.exists(), f"missing Prompting: {pr_path}"


# ---------------------------------------------------------------------------
# #1249: encdec-target fitting is CONDITIONED, and stamped with its bag
# ---------------------------------------------------------------------------


def _make_tiny_encdec() -> Any:
    """Tiny EncoderDecoderGPT at the harness's OWN vocab (TINY_VOCAB=128).

    ``tests.encdec_test_helpers.build_tiny_model`` pins vocab_size=100, but the
    stub tokenizer here round-trips only ids below ``vocab - 2`` -- at 100 the
    letter 'c' maps to id 3 and decodes to chr(1), tripping the trainers' exact
    boundary contract. Mirroring the helper at 128 keeps one vocab across the
    stub tokenizer, the decoder GPT fixture, and this model.
    """
    from nanochat.gpt import GPTConfig
    from steering.conditional.encoder_decoder.model import (
        CrossAttentionConfig,
        EncoderConfig,
        EncoderDecoderGPT,
    )

    t.manual_seed(1249)
    dec_config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD // 2,
        depth=1,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=(TINY_N_EMBD // TINY_N_HEAD) // 2,
    )
    xattn_config = CrossAttentionConfig(
        frequency=2, n_head=TINY_N_HEAD, n_kv_head=TINY_N_KV_HEAD,
    )
    model = EncoderDecoderGPT(
        dec_config, enc_config, xattn_config,
        n_sae_features=200, num_buckets=4,
    )
    model.init_weights()
    model.eval()
    return model


def test_stage_3_diffmean_encdec_requires_conditioning(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An encdec fit with no bag must raise BEFORE any direction is written.

    This drives the REAL trainer chain (train_diffmean_all ->
    _collect_row_vectors -> _capture_per_row_continuation ->
    _run_capture_forward) on a real tiny EncoderDecoderGPT -- not the unit
    guard in isolation -- so a dropped kwarg anywhere in the chain fails here.
    """
    tok = _StubNanochatTokenizer()
    monkeypatch.setattr(diffmean_mod, "get_tokenizer", lambda: tok)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_encdec_uncond", device="cpu",
    ).replace(layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER))
    model = _make_tiny_encdec()
    concepts = [_concept(0)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    with pytest.raises(AxBenchBaseError, match="1249"):
        train_diffmean_all(
            cfg=cfg,
            target_model=model,
            target_model_name="encdec",
            mode=SteeringMode.POSITIVE,
            concepts=concepts,
            training_corpus_path=corpus,
        )
    out_path = test_paths.default_paths("stage3_encdec_uncond").diffmean_direction(
        target_model="encdec", mode=SteeringMode.POSITIVE, concept_id=0
    )
    assert not out_path.exists(), "no artifact may be written by an unconditioned fit"


def test_stage_3_diffmean_encdec_conditioned_fit_stamps_the_bag_digest(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conditioned fit writes a direction stamped with ITS concept's bag.

    Per-concept bags (not one broadcast bag) are the point: each concept's
    digest must match its own conditioning, so the two artifacts here must
    carry DIFFERENT digests.
    """
    from safetensors import safe_open

    from steering.eval.axbench_base.baselines.conditioning import (
        EncoderConditioning,
    )
    tok = _StubNanochatTokenizer()
    monkeypatch.setattr(diffmean_mod, "get_tokenizer", lambda: tok)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_encdec_cond", device="cpu",
    ).replace(layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER))
    model = _make_tiny_encdec()
    concepts = [_concept(0), _concept(1)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    conditioning = {
        0: EncoderConditioning(entries=((2, 1), (5, 0))),
        1: EncoderConditioning(entries=((7, 3),)),
    }
    out_paths = train_diffmean_all(
        cfg=cfg,
        target_model=model,
        target_model_name="encdec",
        mode=SteeringMode.POSITIVE,
        concepts=concepts,
        training_corpus_path=corpus,
        encoder_conditioning_for=conditioning.__getitem__,
    )

    assert len(out_paths) == 2
    digests: list[str] = []
    for concept, out_path in zip(concepts, out_paths, strict=True):
        assert out_path.exists()
        with safe_open(str(out_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            direction = handle.get_tensor("direction")
        assert direction.ndim == 1
        assert metadata["target_model"] == "encdec"
        assert metadata["encdec_fit_bag_digest"] == (
            conditioning[concept.concept_id].bag_digest
        ), "the artifact must be bound to the bag it was fit under"
        digests.append(metadata["encdec_fit_bag_digest"])
    assert digests[0] != digests[1], (
        "per-concept bags must yield per-concept digests; equal digests mean "
        "one bag was broadcast across concepts"
    )


def test_stage_3_diffmean_decoder_only_has_no_bag_digest(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoder-only artifacts stay exactly as before: no digest key at all."""
    from safetensors import safe_open

    _install_stub_tokenizer_diffmean(monkeypatch)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_dec_nodigest", device="cpu",
    ).replace(layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER))
    model = _make_tiny_gpt()
    tok = _StubNanochatTokenizer()
    concepts = [_concept(0)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    [out_path] = train_diffmean_all(
        cfg=cfg,
        target_model=model,
        target_model_name="decoder_only",
        mode=SteeringMode.POSITIVE,
        concepts=concepts,
        training_corpus_path=corpus,
    )
    with safe_open(str(out_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    assert "encdec_fit_bag_digest" not in metadata


def test_stage_3_linear_probe_encdec_conditioned_fit_stamps_the_bag_digest(
    tmp_run_root: Path,
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LinearProbe shares the #1249 contract: conditioned fit, stamped artifact."""
    from safetensors import safe_open

    from steering.eval.axbench_base.baselines.conditioning import (
        EncoderConditioning,
    )

    tok = _StubNanochatTokenizer()
    monkeypatch.setattr(linear_probe_mod, "get_tokenizer", lambda: tok)
    cfg = make_valid_axbench_cfg(
        run_tag="stage3_probe_encdec_cond", device="cpu",
    ).replace(layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER))
    model = _make_tiny_encdec()
    concepts = [_concept(0)]
    corpus = tmp_run_root / "corpus.parquet"
    _write_positive_corpus(corpus, _build_tiny_positive_rows(concepts, tok))

    conditioning = {0: EncoderConditioning(entries=((3, 2),))}
    with pytest.raises(AxBenchBaseError, match="1249"):
        train_linear_probe_all(
            cfg=cfg,
            target_model=model,
            target_model_name="encdec",
            mode=SteeringMode.POSITIVE,
            concepts=concepts,
            training_corpus_path=corpus,
        )
    [out_path] = train_linear_probe_all(
        cfg=cfg,
        target_model=model,
        target_model_name="encdec",
        mode=SteeringMode.POSITIVE,
        concepts=concepts,
        training_corpus_path=corpus,
        encoder_conditioning_for=conditioning.__getitem__,
    )
    with safe_open(str(out_path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    assert metadata["encdec_fit_bag_digest"] == conditioning[0].bag_digest
