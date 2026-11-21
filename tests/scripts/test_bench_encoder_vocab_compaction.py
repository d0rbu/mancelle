"""The compaction benchmark must measure the shape it CLAIMS to measure (#695, FINDING 15).

``scripts/benchmark/bench_encoder_vocab_compaction.py`` asserts — three times, in its own
module docstring — that its default invocation measures "the QUEUED RUN's actual shape".
It did not. It hard-coded an encoder depth ratio of ``0.75``, a cross-attention frequency
of ``4`` and a decoder vocabulary of ``65,536``, while
``configs/training/cluster_fp/
conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml`` resolves to
``1.0``, ``1`` and ``32,768``.

None of those is cosmetic — every one of them moves the number the benchmark exists to
report:

* **encoder depth** 0.75 ⇒ 9 layers ⇒ ``has_ve`` puts a value embedding on 5 of them;
  1.0 ⇒ 12 layers ⇒ 6. The feature-indexed tables are ``feature_embed`` PLUS one row per
  SAE feature in every encoder value embedding, so the benchmark was sizing SIX
  feature-indexed tables where the queued run has SEVEN — 302.0M parameters of dead weight
  instead of 352.3M, a 14% understatement of the very thing compaction deletes.
* **cross-attention frequency** 4 ⇒ 3 injection sites; 1 ⇒ 12. That is 7.1M vs 28.3M of
  cross-attention parameters, and it moves MODEL TOTAL — the denominator of the saving.
* **decoder vocab** 65,536 vs the tokenizer's real 32,768 doubles ``wte``, ``lm_head`` and
  every decoder value embedding, inflating the denominator again.

So this is not a docstring nit: a benchmark that overstates the model and understates the
compacted tables reports a **saving fraction that the queued run will not see**.

The assertion below is deliberately not a comparison of magic numbers against other magic
numbers. It builds the benchmark's model and the PRODUCTION model — through
``build_encdec_meta_model_from_training_cfg``, the same function the trainer calls — from
the real YAML, and compares the resulting shapes. A docstring edit cannot make it pass.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import cast

import torch as t
from omegaconf import DictConfig

from scripts.benchmark.bench_encoder_vocab_compaction import (
    DEC_VOCAB_SIZE,
    N_SAE,
    QUEUED_CONFIG_NAME,
    SEQ_LEN,
    build_meta_model,
    make_batch,
    measure,
)
from scripts.train.conditional_encdec_train import (
    build_encdec_meta_model_from_training_cfg,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_DIR = REPO_ROOT / "configs" / "training"

# The nanochat tokenizer's vocabulary — what ``conditional_encdec_train.py`` passes as
# ``base_vocab`` (``tokenizer.get_vocab_size()``). ``configs/data/tok_train.yaml`` pins
# ``vocab_size: 32768``, and ``tests/steering/test_all_training_configs_build.py`` already
# hard-codes the same constant for the same reason.
NANOCHAT_VOCAB_SIZE = 32768

# The SAE the queued run conditions on: gemma-3-4b-pt / l22 / 65k.
QUEUED_N_SAE = 65536


def _queued_cfg(hydra_compose) -> DictConfig:
    return hydra_compose(str(TRAINING_CONFIG_DIR), QUEUED_CONFIG_NAME, None)


def _production_model(cfg: DictConfig, vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    """The model the TRAINER builds from the queued config. The reference, by definition."""
    with t.device("meta"):
        return build_encdec_meta_model_from_training_cfg(
            cfg,
            base_vocab=NANOCHAT_VOCAB_SIZE,
            n_sae_features=QUEUED_N_SAE,
            feature_vocab=vocab,
        )


class TestTheBenchMeasuresTheQueuedShape:
    def test_the_bench_targets_the_config_it_names(self) -> None:
        assert (TRAINING_CONFIG_DIR / f"{QUEUED_CONFIG_NAME}.yaml").is_file(), (
            "the benchmark names a queued config that does not exist"
        )

    def test_the_SAE_width_matches(self, hydra_compose) -> None:
        cfg = _queued_cfg(hydra_compose)
        assert str(cfg.sae.width) == "65k"
        assert N_SAE == QUEUED_N_SAE

    def test_the_DECODER_config_is_IDENTICAL_to_the_production_one(self, hydra_compose) -> None:
        """Catches the hard-coded ``vocab_size=65536`` (the real one is 32,768), and any
        drift in depth / n_embd / heads / window pattern / sequence length."""
        cfg = _queued_cfg(hydra_compose)
        prod = _production_model(cfg, None)
        with t.device("meta"):
            bench = build_meta_model(None, SEQ_LEN)

        assert bench.dec_config == prod.dec_config

    def test_the_ENCODER_config_is_IDENTICAL_to_the_production_one(self, hydra_compose) -> None:
        """Catches the hard-coded encoder ``depth=0.75`` (the queued run is ``1.0``)."""
        cfg = _queued_cfg(hydra_compose)
        prod = _production_model(cfg, None)
        with t.device("meta"):
            bench = build_meta_model(None, SEQ_LEN)

        assert bench.encoder.config == prod.encoder.config

    def test_the_number_of_FEATURE_INDEXED_tables_matches(self, hydra_compose) -> None:
        """The headline. ``feature_embed`` + one value embedding per ``has_ve`` encoder
        layer. A 9-layer encoder has 5 of them, a 12-layer encoder 6 — so the benchmark was
        deleting dead rows from SIX tables where the queued run has SEVEN."""
        cfg = _queued_cfg(hydra_compose)
        prod = _production_model(cfg, None)
        with t.device("meta"):
            bench = build_meta_model(None, SEQ_LEN)

        assert len(bench.encoder.value_embeds) == len(prod.encoder.value_embeds)
        assert len(prod.encoder.value_embeds) == 6, (
            "the queued encoder is 12 layers deep — six value embeddings, seven "
            "feature-indexed tables in total"
        )

    def test_the_CROSS_ATTENTION_sites_match(self, hydra_compose) -> None:
        """Catches the hard-coded ``frequency=4`` (the queued run injects at EVERY layer)."""
        cfg = _queued_cfg(hydra_compose)
        prod = _production_model(cfg, None)
        with t.device("meta"):
            bench = build_meta_model(None, SEQ_LEN)

        assert sorted(bench.cross_attn_layers) == sorted(prod.cross_attn_layers)
        assert len(prod.cross_attn_layers) == int(cfg.depth), (
            "cross_attention.frequency: 1 ⇒ one site per decoder layer"
        )

    def test_EVERY_PARAMETER_COUNT_matches_the_production_model(self, hydra_compose) -> None:
        """The non-gameable one: same encoder, same cross-attention, same decoder, same
        totals. If any of the three hard-coded numbers is wrong, this fails."""
        cfg = _queued_cfg(hydra_compose)
        prod = _production_model(cfg, None)
        with t.device("meta"):
            bench = build_meta_model(None, SEQ_LEN)

        assert bench.num_scaling_params() == prod.num_scaling_params()

    def test_exact_full_and_compact_parameter_counts_are_pinned(self, hydra_compose) -> None:
        """Protect the published analytical counts from another dimension-factor error."""
        cfg = _queued_cfg(hydra_compose)
        vocab = FeatureVocab(
            n_sae_features=QUEUED_N_SAE,
            original_ids=tuple(range(2074)),
        )
        full = _production_model(cfg, None)
        compact = _production_model(cfg, vocab)

        def feature_indexed_params(model: EncoderDecoderGPT) -> int:
            return model.encoder.feature_embed.weight.numel() + sum(
                parameter.numel() for parameter in model.encoder.value_embeds.parameters()
            )

        assert feature_indexed_params(full) == 352_321_536
        assert feature_indexed_params(compact) == 11_149_824
        assert full.num_scaling_params() == {
            "wte": 25_165_824,
            "value_embeds": 150_994_944,
            "lm_head": 25_165_824,
            "transformer_matrices": 84_935_808,
            "scalars": 24,
            "encoder": 437_258_880,
            "cross_attn": 28_311_552,
            "feature_pred_head": 50_397_184,
            "total": 802_230_040,
        }
        assert compact.num_scaling_params() == {
            "wte": 25_165_824,
            "value_embeds": 150_994_944,
            "lm_head": 25_165_824,
            "transformer_matrices": 84_935_808,
            "scalars": 24,
            "encoder": 96_087_168,
            "cross_attn": 28_311_552,
            "feature_pred_head": 1_594_906,
            "total": 412_256_050,
        }

    def test_the_queued_run_has_the_production_feature_prediction_head(
        self, hydra_compose
    ) -> None:
        cfg = _queued_cfg(hydra_compose)
        assert dict(cfg.feature_prediction) == {
            "block": "mlp",
            "head_layers": 0,
            "weight": 0.1,
            "layer": -1,
            "loss_mode": "max_pool",
            "loss_fn": "bce",
        }
        with t.device("meta"):
            head = build_meta_model(None, SEQ_LEN).feature_pred_head
        assert head is not None
        assert head.out_features == QUEUED_N_SAE

    def test_the_COMPACTED_model_also_matches_the_production_compacted_model(
        self, hydra_compose
    ) -> None:
        """Compaction must be applied to the same tables in both."""
        cfg = _queued_cfg(hydra_compose)
        vocab = FeatureVocab(
            n_sae_features=QUEUED_N_SAE, original_ids=tuple(range(0, QUEUED_N_SAE, 2))
        )
        prod = _production_model(cfg, vocab)
        with t.device("meta"):
            bench = build_meta_model(vocab, SEQ_LEN)

        assert bench.num_scaling_params() == prod.num_scaling_params()
        assert bench.encoder.feature_embed.weight.shape[0] == vocab.size
        assert bench.feature_pred_head is not None
        assert bench.feature_pred_head.out_features == vocab.size

    def test_the_seq_len_constant_matches_the_configs_max_seq_len(self, hydra_compose) -> None:
        assert SEQ_LEN == int(_queued_cfg(hydra_compose).max_seq_len)


class TestTheBenchmarkCanActuallyRun:
    """This module had 10 tests and none of them called into ``measure()``.

    So when the round-3 "shape fix" lowered ``DEC_VOCAB_SIZE`` 65,536 -> 32,768 and left the
    batch drawing token ids from a hardcoded ``[0, 65536)``, nothing noticed: every forward
    indexed a 32,768-row ``wte`` out of range, the benchmark died in its first warmup step, and
    the PR's *only* performance evidence — every VRAM, step-time and parameter figure it quotes
    — was unreproducible. The literal was the SAE width; the comment on ``DEC_VOCAB_SIZE``
    warns about exactly that conflation, four lines above the code that commits it.

    ``measure()`` is CUDA-only end to end, which is why it was untested. The batch it feeds is
    not, so that is now a separate function and these tests exercise it on CPU.
    """

    def test_the_batch_ids_are_in_range_for_the_MODELS_token_table(self) -> None:
        rows = int(
            cast(
                t.nn.Embedding,
                build_meta_model(None, SEQ_LEN).transformer.wte,
            ).num_embeddings
        )
        idx, tgt, *_ = make_batch(None, "cpu", B=4, T=64, n_seg=2, F=8)
        assert int(idx.max()) < rows, f"token id {int(idx.max())} >= wte rows {rows}"
        assert int(tgt.max()) < rows, f"target id {int(tgt.max())} >= wte rows {rows}"

    def test_the_token_bound_is_the_PARAMETER_not_a_literal(self) -> None:
        """Deterministic, where the test above is only probabilistic.

        With the bug restored, ``idx.max()`` merely *tends* to exceed 32,768 — a small draw
        could pass by luck. Passing a tiny ``vocab_size`` removes the luck: a hardcoded bound
        overshoots it every single time.
        """
        idx, tgt, *_ = make_batch(None, "cpu", B=8, T=64, n_seg=2, F=8, vocab_size=8)
        assert int(idx.max()) < 8, "make_batch ignored vocab_size — the bound is hardcoded"
        assert int(tgt.max()) < 8, "make_batch ignored vocab_size — the bound is hardcoded"

    def test_DEC_VOCAB_SIZE_is_tied_to_the_model_it_feeds(self) -> None:
        """The two cannot drift again: the constant IS the table's row count."""
        rows = int(
            cast(
                t.nn.Embedding,
                build_meta_model(None, SEQ_LEN).transformer.wte,
            ).num_embeddings
        )
        assert DEC_VOCAB_SIZE == rows
        assert DEC_VOCAB_SIZE != N_SAE, "the token vocabulary is not the SAE width"

    def test_the_feature_ids_stay_inside_the_compacted_vocabulary(self) -> None:
        """The other half of the batch: conditioning ids must be ORIGINAL SAE ids drawn from
        the vocab, or the encoder's OOV guard fires on the benchmark's own data."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=tuple(range(0, 2048, 4)))
        _, _, fids, _, mask, _ = make_batch(vocab, "cpu", B=2, T=32, n_seg=2, F=8)
        live = fids[mask]
        assert live.numel() > 0, "no conditioning ids — the assertion below would be vacuous"
        assert set(live.tolist()) <= set(vocab.original_ids)

    def test_each_arm_releases_its_final_graph_and_cuda_batch(self) -> None:
        """The compact arm must not inherit live tensors from the full arm."""
        tree = ast.parse(inspect.getsource(measure))
        deleted = {
            node.id
            for delete in ast.walk(tree)
            if isinstance(delete, ast.Delete)
            for target in delete.targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name)
        }
        assert {
            "m",
            "opt",
            "result",
            "loss",
            "idx",
            "tgt",
            "fids",
            "mags",
            "mask",
            "seg_map",
            "fp_targets",
            "seg_doc_id",
        } <= deleted
