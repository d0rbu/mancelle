"""Regenerate axbench-bpb test fixtures.

Runs two toy model instantiations + init_weights under a fixed seed and
saves checkpoints to ``tests/axbench_bpb/fixtures/tiny_{decoder_only,encdec}/ckpt.pt``.

Run with: ``uv run python -m tests.axbench_bpb.fixtures.regenerate_fixtures``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import safetensors.torch as st
import torch

from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.axbench_base.baselines.lora._attach import (
    _attach_lora_to_model,
    _collect_attached_adapters,
    _detach_lora_from_model,
)
from steering.eval.axbench_base.baselines.lora.training import _save_lora_adapter
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.direction_scale import encode_max_activation
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
    validate_positive_qc_rows,
)
from steering.eval.axbench_base.positive_selection_contract import (
    schema_with_default_positive_selection_contract,
)
from steering.eval.axbench_base.synth.negative_mode import NEG_TRAINING_SCHEMA
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from tests.axbench_bpb_adv.source_run_builders import write_seed_contract

FIXTURES_ROOT = Path(__file__).parent
SEED = 12345
TINY_VOCAB = 128
TINY_N_EMBD = 32
TINY_N_LAYER = 2
TINY_N_HEAD = 2
TINY_N_KV_HEAD = 2
TINY_SEQ_LEN = 64
TINY_HEAD_DIM = 16
TINY_N_SAE_FEATURES = 64
TINY_NUM_BUCKETS = 4

# LoRA fixture constants — minimal but valid
TINY_LORA_LAYER = 1  # tiny model has n_layer=2 → layer 0 or 1
TINY_LORA_RANK = 4
TINY_LORA_ALPHA = 8.0

# Mini axbench-base fixture — 5 concepts, 4 rich + 4 avoid samples each
MINI_FIXTURE_DIR = FIXTURES_ROOT / "mini_axbench_base_run"
MINI_N_CONCEPTS = 5
MINI_SAMPLES_PER_SIDE = 4
MINI_TOP_K = 8                  # concept_features top-K
MINI_DEC_LAYER = 1              # tiny model has 2 layers; layer 1 is the deepest

_CONCEPT_NAMES = ["cats", "dogs", "cars", "books", "music"]


def _make_dec_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )


def build_tiny_decoder_only() -> Path:
    """Build + save a tiny decoder-only GPT checkpoint under ~50KB."""
    torch.manual_seed(SEED)
    config = _make_dec_config()
    with torch.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()

    out_dir = FIXTURES_ROOT / "tiny_decoder_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(config),
            "pad_vocab_size_to": TINY_VOCAB,
        },
        out_path,
    )
    print(f"Wrote {out_path}")
    return out_path


def build_tiny_encdec() -> Path:
    """Build + save a tiny EncoderDecoderGPT checkpoint under ~200KB."""
    torch.manual_seed(SEED)
    dec_config = _make_dec_config()
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(frequency=1)
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config=dec_config,
            enc_config=enc_config,
            xattn_config=xattn_config,
            n_sae_features=TINY_N_SAE_FEATURES,
            num_buckets=TINY_NUM_BUCKETS,
            pad_vocab_size_to=TINY_VOCAB,
        )
    model = model.to_empty(device="cpu")
    model.init_weights()

    out_dir = FIXTURES_ROOT / "tiny_encdec"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "dec_config": vars(dec_config),
            "enc_config": vars(enc_config),
            "xattn_config": vars(xattn_config),
            "n_sae_features": TINY_N_SAE_FEATURES,
            "num_buckets": TINY_NUM_BUCKETS,
            "pad_vocab_size_to": TINY_VOCAB,
        },
        out_path,
    )
    print(f"Wrote {out_path}")
    return out_path


def _build_tiny_lora_adapter_for_concept(
    concept_id: int, *, seed: int, lora_b_std: float = 0.0,
) -> Path:
    """Build a tiny LoRA adapter for ``concept_id`` with a chosen seed.

    Loads the tiny_decoder_only checkpoint, attaches LoRA adapters at
    ``TINY_LORA_LAYER`` with ``LoraConfig(rank=TINY_LORA_RANK,
    alpha=TINY_LORA_ALPHA)``, randomises ``lora_A`` (and optionally
    ``lora_B`` if ``lora_b_std > 0`` so the adapter actually perturbs
    forward output), then saves via ``_save_lora_adapter`` and detaches
    the model in place.
    """
    torch.manual_seed(seed)
    # Load tiny decoder-only checkpoint
    ckpt_path = FIXTURES_ROOT / "tiny_decoder_only" / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**state["config"])
    with torch.device("meta"):
        model = GPT(cfg, pad_vocab_size_to=state["pad_vocab_size_to"])
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.eval()

    # Attach LoRA + collect references
    lora_cfg = LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA)
    original = _attach_lora_to_model(model, layer=TINY_LORA_LAYER, lora_cfg=lora_cfg)
    refs = _collect_attached_adapters(model, layer=TINY_LORA_LAYER)
    # Randomise lora_A so adapter is non-zero (lora_B is zero-init by design)
    for ref in refs:
        torch.nn.init.normal_(ref.adapter.lora_A, mean=0.0, std=0.01)
        if lora_b_std > 0.0:
            # Make the adapter non-identity in forward by also perturbing
            # lora_B, otherwise lora_B=0 means the adapter outputs the
            # base linear's output regardless of lora_A.
            torch.nn.init.normal_(ref.adapter.lora_B, mean=0.0, std=lora_b_std)

    # Save adapter
    fname = f"lora_adapter_concept_{concept_id:04d}.pt"
    out_path = FIXTURES_ROOT / "tiny_decoder_only" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_lora_adapter(
        refs=refs, out_path=out_path,
        metadata={
            "layer": str(TINY_LORA_LAYER),
            "rank": str(TINY_LORA_RANK),
            "concept_id": str(concept_id),
        },
    )

    # Detach to leave the model unmodified for any downstream caller of
    # this script (defensive — main() doesn't reuse `model`).
    _detach_lora_from_model(model, original=original)

    print(f"Wrote {out_path}")
    return out_path


def build_tiny_lora_adapter() -> Path:
    """Build the canonical concept_0000 tiny LoRA adapter file.

    ``lora_B`` is left at its 0-init (the LoRAAdapter constructor sets
    it that way) — this matches axbench-base's training-loop output at
    initialisation. Tests that require the adapter to perturb forward
    output should use the concept_0001 fixture instead.
    """
    return _build_tiny_lora_adapter_for_concept(
        concept_id=0, seed=SEED, lora_b_std=0.0,
    )


def build_tiny_lora_adapter_concept_0001() -> Path:
    """Build a second tiny LoRA adapter with different parameter values.

    Identical shape to ``lora_adapter_concept_0000.pt`` but with a
    different seed and a non-zero ``lora_B`` so the adapter actually
    perturbs forward output. Used by
    ``test_concept_a_then_concept_b_interleaving`` to verify that two
    distinct adapters produce different losses (not just byte-equal
    state after detach).
    """
    return _build_tiny_lora_adapter_for_concept(
        concept_id=1, seed=SEED + 9999, lora_b_std=0.05,
    )


def build_mini_axbench_base_run() -> Path:
    """Build a synthetic 5-concept axbench-base mini run for integration tests.

    Produces every artifact axbench-bpb's SourcePaths resolves; tests can
    exercise the full CLI pipeline against this fixture without needing
    a real axbench-base run. See spec §3.3.
    """
    torch.manual_seed(SEED)
    MINI_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _build_concepts_json()
    _build_synth_parquets()
    write_seed_contract(MINI_FIXTURE_DIR)
    _build_concept_features()
    _build_baseline_directions()
    _build_lora_adapters()
    _build_prompting_descriptions()
    print(f"Wrote mini axbench-base run at {MINI_FIXTURE_DIR}")
    return MINI_FIXTURE_DIR


def _concept_name(cid: int) -> str:
    return _CONCEPT_NAMES[cid]


def _build_concepts_json() -> None:
    """5 concepts: cats, dogs, cars, books, music.

    Emits the FULL ``steering.eval.axbench_base.types.Concept`` field set (not
    just ``concept_id`` + ``description``) so the canonical reader
    ``Concept(**entry)`` -- used by the bpb-adv scorer and the generate-samples
    path -- can load it. The feature/layer/source fields are deterministic
    stand-ins for the tiny fixture; only ``concept_id`` + ``description`` carry
    semantic meaning downstream.
    """
    concepts = [
        {
            "concept_id": cid,
            "feature_index": cid,
            "layer": MINI_DEC_LAYER,
            "sae_source": "tiny-fixture-sae",
            "description": _concept_name(cid),
            "corpus_count": 0,
            "description_source": "fixture",
        }
        for cid in range(MINI_N_CONCEPTS)
    ]
    payload = json.dumps(concepts, indent=2)
    (MINI_FIXTURE_DIR / "concepts.json").write_text(payload)
    (MINI_FIXTURE_DIR / "concepts_stage1c.json").write_text(payload)


def _build_synth_parquets() -> None:
    """5 concepts × (4 rich + 4 avoiding) = 40 rows per parquet.

    Positive mode is now symmetric with negative mode (plan §1.1/§1.4): every
    accepted ``concept_rich`` row owns a paired ``concept_avoiding`` row of the
    SAME prefix (``concept_id == cid``), produced from the seed's natural
    continuation (natural QC tier, ``gen_source = corpus:doc_continuation@…`` and
    ``qc_zero_activation_verified = True``). There is no longer a shared
    global-negative pool and no ``concept_id == -1`` sentinel. Both completions
    of a positive prefix land at ``(cid, positive)`` so ``bpb_aggregation``
    produces a real (finite) positive ``primary_margin``.
    """
    pos_rows: list[dict] = []
    neg_rows: list[dict] = []
    for cid in range(MINI_N_CONCEPTS):
        for s in range(MINI_SAMPLES_PER_SIDE):
            prefix_text = f"This is a piece about {_concept_name(cid)} sample {s}: "
            base = {
                "concept_id": cid, "category": "positive",
                "content_class": "concept_rich",
                "prefix_text": prefix_text,
                "output_text": f"output for concept {cid} sample {s}",
                "output_concept": _concept_name(cid),
                "n_prefix_nanochat_tokens": 12, "gen_source": "fixture",
                "gen_seed": cid * 100 + s, "qc_feature_id": cid + 100,
                "qc_seed_percentile": 10.0, "qc_output_percentile": 40.0,
                "qc_delta": 30.0, "qc_attempts": 1, "qc_accepted": True,
                "qc_zero_activation_verified": False,
            }
            pos_rows.append(dict(base))
            # Paired positive avoiding row: SAME prefix, concept_id == cid,
            # natural-tier (seed's own continuation, feature verified inactive).
            avoid = {
                **base,
                "category": "negative", "content_class": "concept_avoiding",
                "prefix_text": prefix_text,
                "output_text": f"avoid output for concept {cid} sample {s}",
                "gen_source": f"corpus:doc_continuation@0/{cid}",
                "qc_seed_percentile": float("nan"),
                "qc_output_percentile": float("nan"),
                "qc_delta": float("nan"),
                "qc_zero_activation_verified": True,
            }
            pos_rows.append(avoid)

            # Negative mode (Stage 1c): unchanged shape — both completions of a
            # verified prefix with prefix_source_* provenance columns.
            neg_base = {
                k: v for k, v in base.items()
                if k != "qc_zero_activation_verified"
            }
            neg_base.update(
                qc_seed_percentile=0.05,
                qc_output_percentile=0.85,
                qc_delta=0.80,
            )
            neg_rows.append({
                **neg_base,
                "category": "positive",
                "content_class": "concept_rich",
                "prefix_source_shard": 0, "prefix_source_doc": cid,
                "prefix_source_seg_idx": s,
                "qc_zero_activation_verified": False,
            })
            neg_rows.append({
                **neg_base,
                "category": "negative", "content_class": "concept_avoiding",
                "prefix_text": f"Unrelated text about something else, sample {s}: ",
                "output_text": f"avoid output for concept {cid} sample {s}",
                "prefix_source_shard": 0, "prefix_source_doc": cid,
                "prefix_source_seg_idx": s,
                "qc_zero_activation_verified": True,
            })

    pos_path = MINI_FIXTURE_DIR / "synth_training_positive.parquet"
    neg_path = MINI_FIXTURE_DIR / "synth_training_negative.parquet"
    validate_positive_qc_rows(pos_rows, accept_threshold_points=30.0)
    positive_schema = schema_with_stage1_continuation_contract(
        schema_with_default_positive_selection_contract(
            schema_with_positive_qc_contract(
                TRAINING_SCHEMA,
                accept_threshold=0.3,
            ),
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(pa.Table.from_pylist(pos_rows, schema=positive_schema), pos_path)
    negative_schema = schema_with_stage1_continuation_contract(
        NEG_TRAINING_SCHEMA,
        role="stage1c_negative",
    )
    pq.write_table(pa.Table.from_pylist(neg_rows, schema=negative_schema), neg_path)


def _build_concept_features() -> None:
    """5 × MINI_TOP_K int64 + metadata={"mode": "canonical"}.

    Feature ids are arbitrary small distinct ints per concept. The same
    layout is used for both positive (concept_features.safetensors) and
    negative (concept_features_neg.safetensors) — for this fixture we
    use the same tensor for both to keep the schema simple.
    """
    feature_ids = torch.zeros(MINI_N_CONCEPTS, MINI_TOP_K, dtype=torch.int64)
    for cid in range(MINI_N_CONCEPTS):
        for k in range(MINI_TOP_K):
            feature_ids[cid, k] = cid * 100 + k
    metadata = {"mode": "canonical"}
    for fname in ("concept_features.safetensors", "concept_features_neg.safetensors"):
        st.save_file(
            {"feature_ids": feature_ids},
            str(MINI_FIXTURE_DIR / fname),
            metadata=metadata,
        )


def _build_baseline_directions() -> None:
    """Diffmean + linear_probe directions for all (target, mode, concept) triples.

    Per axbench-bpb's SourcePaths:
      baselines/diffmean/{target}/{mode}/concept_{cid:04d}.safetensors
      baselines/linear_probe/{target}/{mode}/concept_{cid:04d}.safetensors

    Each file contains a single tensor named "direction" of shape
    (n_embd,) for the relevant target model. Dimensions:
      - decoder_only: TINY_N_EMBD == 32
      - encdec: also uses the decoder's n_embd (32) for the residual hook
    """
    baselines_root = MINI_FIXTURE_DIR / "baselines"
    for baseline in ("diffmean", "linear_probe"):
        for target in ("decoder_only", "encdec"):
            for mode in ("positive", "negative"):
                target_dir = baselines_root / baseline / target / mode
                target_dir.mkdir(parents=True, exist_ok=True)
                for cid in range(MINI_N_CONCEPTS):
                    direction = torch.randn(TINY_N_EMBD, dtype=torch.float32)
                    out_path = target_dir / f"concept_{cid:04d}.safetensors"
                    # Deterministic synthetic stand-in for the direction-owned
                    # Stage-3 statistic; concept ids equal feature ids (0..4).
                    max_activation = 2.0 + 0.5 * cid
                    st.save_file(
                        {"direction": direction},
                        str(out_path),
                        metadata={
                            "concept_id": str(cid),
                            "target_model": target,
                            "mode": mode,
                            "layer": str(MINI_DEC_LAYER),
                            "artifact_kind": f"{baseline}_direction_v1",
                            "n_pos": "2",
                            "n_neg": "2",
                            "max_activation": encode_max_activation(
                                max_activation
                            ),
                        },
                    )


def _build_lora_adapters() -> None:
    """Generate 10 LoRA adapters: 5 concepts × 2 modes (decoder_only only).

    Reuses _attach_lora_to_model + _save_lora_adapter for each (cid, mode)
    pair. Each adapter has its own random init so multi-concept
    interleaving tests can detect cross-attach contamination.
    """
    ckpt_path = FIXTURES_ROOT / "tiny_decoder_only" / "ckpt.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**state["config"])

    base_dir = MINI_FIXTURE_DIR / "baselines" / "lora" / "decoder_only"
    for mode in ("positive", "negative"):
        mode_dir = base_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for cid in range(MINI_N_CONCEPTS):
            torch.manual_seed(SEED + cid * 7 + (1 if mode == "negative" else 0))
            with torch.device("meta"):
                model = GPT(cfg, pad_vocab_size_to=state["pad_vocab_size_to"])
            model.to_empty(device="cpu")
            model.init_buffers()
            model.load_state_dict(state["model"], strict=True, assign=True)
            model.eval()

            lora_cfg = LoraConfig(
                rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA,
            )
            original = _attach_lora_to_model(
                model, layer=TINY_LORA_LAYER, lora_cfg=lora_cfg,
            )
            refs = _collect_attached_adapters(model, layer=TINY_LORA_LAYER)
            for ref in refs:
                torch.nn.init.normal_(ref.adapter.lora_A, mean=0.0, std=0.01)

            out_path = mode_dir / f"concept_{cid:04d}.pt"
            _save_lora_adapter(
                refs=refs, out_path=out_path,
                metadata={
                    "layer": str(TINY_LORA_LAYER),
                    "rank": str(TINY_LORA_RANK),
                    "concept_id": str(cid), "mode": mode,
                },
            )
            # Defensive detach so the next iteration's load_state_dict gets a clean model
            _detach_lora_from_model(model, original=original)


def _build_prompting_descriptions() -> None:
    """positive_descriptions.json + negative_descriptions.json under
    baselines/prompting/. Each maps str(concept_id) → description string.
    """
    out_dir = MINI_FIXTURE_DIR / "baselines" / "prompting"
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptions = {str(cid): _concept_name(cid) for cid in range(MINI_N_CONCEPTS)}
    for mode in ("positive", "negative"):
        (out_dir / f"{mode}_descriptions.json").write_text(
            json.dumps(descriptions, indent=2)
        )


def main() -> int:
    print(f"Fixtures root: {FIXTURES_ROOT}", file=sys.stderr)
    build_tiny_decoder_only()
    build_tiny_encdec()
    build_tiny_lora_adapter()
    build_tiny_lora_adapter_concept_0001()
    build_mini_axbench_base_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
