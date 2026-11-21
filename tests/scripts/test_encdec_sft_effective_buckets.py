"""Conditioned SFT must encode positives into the bucket space PRETRAINING used.

A checkpoint pretrained with ``negative_selection`` discretised positives into
``effective_buckets = num_buckets - 1`` and reserved bucket ``num_buckets - 1``
as the "feature inactive" row (``steering/packed_batch_io.py`` writes
``enc_mags[...] = effective_buckets`` for negatives).  Conditioned SFT built its
boundaries from the RAW ``num_buckets``, so high-magnitude positives landed in
the inactive row — silent conditioning corruption, invisible in the loss.

These tests pin, in order:

* the regression: with ``negative_selection`` the SFT bucket vector equals the
  hand-computed pretraining vector, not the raw-``num_buckets`` one;
* the no-op: without ``negative_selection`` the bucket vector is bitwise the
  frozen pre-fix golden;
* the model/data split: the encoder's magnitude embedding stays sized
  ``num_buckets`` (reducing it breaks the strict state-dict load);
* the HARDCODED delta policy (splice_counterfactual warns, the cross-segment
  modes raise) and the chain-provenance metadata that keeps a chained SFT run
  reading the same contract.

**Why the bucket tests drive SFT's literal call site.**  ``build_feature_pipeline``
is the shared production owner; calling it directly with
``negative_selection=<the checkpoint's>`` proves nothing about SFT, because the
bug was SFT *not passing* that argument.  An earlier draft of this suite did
exactly that and SURVIVED the mutation.  So ``_sft_call_site_pipeline`` lifts the
real ``build_feature_pipeline(...)`` call out of
``scripts/train/conditional_encdec_sft.py`` by AST and evaluates it against a
fixture namespace: reverting the reduction at the call site turns these tests
red.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import torch as t

from scripts.train import conditional_encdec_sft as sft
from steering.conditional.checkpoint_conditioning import (
    PRETRAIN_CONDITIONING_META_KEY,
    CheckpointConditioning,
    assert_conditioning_supported,
    assert_pretrain_conditioning_recoverable,
    checkpoint_negative_selection,
    read_checkpoint_conditioning,
)
from steering.conditional.delta import DeltaMode
from steering.conditional.discretization import (
    build_boundaries_from_stats,
    discretize_activations,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline,
)
from steering.conditional.sampling import (
    parse_feature_selection,
    parse_negative_selection,
)
from steering.stats_types import PerFeatureStat
from tests.encdec_test_helpers import build_tiny_model

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tests" / "steering" / "goldens" / "sft_effective_buckets_golden.json"
SFT_SOURCE = REPO_ROOT / "scripts" / "train" / "conditional_encdec_sft.py"

SFT_CONFIG_NAMES = [
    "conditional_encdec_sft",
    "conditional_encdec_sft_4090",
    "conditional_encdec_sft_2xh200",
    "conditional_encdec_sft_4xh200",
    "conditional_encdec_sft_2xb200",
    "conditional_encdec_sft_profile_4090",
]

# Fixture magnitudes, one per feature. Every feature's percentile pN has value
# float(N), so the pN bucket boundary IS N and every expected vector below can
# be read off by hand.
MAGNITUDES = [10.0, 25.0, 30.0, 50.0, 60.0, 80.0, 99.0]
N_FEATURES = len(MAGNITUDES)

NEGATIVE_SELECTION_CFG = {
    "distribution": "geometric",
    "mean": 10.0,
    "min_k": 0,
    "max_k": 40,
}


def _stats() -> dict[str, PerFeatureStat]:
    return {
        str(fid): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=50.0,
            mean_mag_incl_zeros=25.0,
            std_mag=10.0,
            min_mag=1.0,
            max_mag=99.0,
            median_mag=50.0,
            percentiles={f"p{p}": float(p) for p in range(1, 100)},
        )
        for fid in range(N_FEATURES)
    }


def _fids_and_mags() -> tuple[t.Tensor, t.Tensor]:
    return (
        t.arange(N_FEATURES, dtype=t.int32),
        t.tensor(MAGNITUDES, dtype=t.float32),
    )


def _call_node(call_name: str) -> ast.Call:
    """The first ``call_name(...)`` call in the SFT training script."""
    tree = ast.parse(SFT_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name == call_name:
            return node
    raise AssertionError(f"no call to {call_name} in {SFT_SOURCE}")


def _call_site_kwargs(call_name: str) -> dict[str, str]:
    return {
        kw.arg: ast.unparse(kw.value)
        for kw in _call_node(call_name).keywords
        if kw.arg is not None
    }


def _sft_call_site_pipeline(**namespace: Any) -> FeaturePipelineConfig:
    """Evaluate SFT's OWN ``build_feature_pipeline(...)`` call, verbatim.

    Not a re-implementation: the expression comes straight out of the training
    script's source, so any change to which arguments SFT passes changes what
    this returns.
    """
    source = ast.unparse(_call_node("build_feature_pipeline"))
    return eval(  # noqa: S307 - source is this repo's own training script
        source,
        {
            "build_feature_pipeline": build_feature_pipeline,
            "SFT_FEATURE_PIPELINE_SEED": sft.SFT_FEATURE_PIPELINE_SEED,
        },
        namespace,
    )


def _sft_buckets(
    *,
    num_buckets: int,
    negative_selection: dict | None,
    selection: str = "threshold",
) -> tuple[list[int], int, tuple[int, ...]]:
    """Drive conditioned SFT's OWN pipeline resolution (its literal call site)."""
    feature_selection = parse_feature_selection(
        {"threshold_type": "segment_topk", "threshold": N_FEATURES}
        if selection == "threshold"
        else {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "min_k": 1,
            "max_k": 7,
            "mode": "top_k_percentile",
        }
    )
    pipeline = _sft_call_site_pipeline(
        feature_selection=feature_selection,
        pretrain_negative_selection=parse_negative_selection(negative_selection),
        per_feature_stats=_stats(),
        n_sae_features=N_FEATURES,
        num_buckets=num_buckets,
        ddp_rank=0,
        _sft_feature_mask=None,
    )
    fids, fmags = _fids_and_mags()
    buckets = discretize_activations(fids, fmags, pipeline.boundaries)
    return (
        buckets.tolist(),
        pipeline.effective_buckets,
        tuple(pipeline.boundaries.shape),
    )


# ---------------------------------------------------------------------------
# 0. The harness itself must be bound to the production call site
# ---------------------------------------------------------------------------


class TestCallSiteHarnessIsNonVacuous:
    def test_the_evaluated_expression_is_the_scripts_own_call(self) -> None:
        source = ast.unparse(_call_node("build_feature_pipeline"))
        assert source.startswith("build_feature_pipeline(")
        assert "negative_selection=pretrain_negative_selection" in source
        assert "num_buckets=num_buckets" in source
        assert source in ast.unparse(ast.parse(SFT_SOURCE.read_text(encoding="utf-8")))

    def test_harness_would_notice_a_reverted_reduction(self) -> None:
        """Mutation surrogate: the same fixture with the reduction dropped
        produces a DIFFERENT bucket space, so the tests below cannot pass
        vacuously."""
        reverted = build_feature_pipeline(
            feature_selection=parse_feature_selection(
                {"threshold_type": "segment_topk", "threshold": N_FEATURES}
            ),
            negative_selection=None,
            per_feature_stats=_stats(),
            n_sae_features=N_FEATURES,
            num_buckets=4,
            seed=42,
            ddp_rank=0,
            feature_delta=None,
            feature_mask=None,
        )
        _, effective, _ = _sft_buckets(
            num_buckets=4,
            negative_selection=NEGATIVE_SELECTION_CFG,
        )
        assert reverted.effective_buckets == 4
        assert effective == 3


# ---------------------------------------------------------------------------
# 1. Regression: negatives-trained checkpoints reduce to effective_buckets
# ---------------------------------------------------------------------------


class TestNegativeSelectionReducesBucketSpace:
    def test_three_bucket_positives_match_hand_computed_pretraining_vector(self) -> None:
        """num_buckets=4 + negative_selection => effective_buckets=3.

        Hand derivation: 3 buckets use boundary keys p33, p66, whose fixture
        values are 33.0 and 66.0.  ``discretize_activations`` counts boundaries
        <= mag (searchsorted right=True), so
          10 -> 0, 25 -> 0, 30 -> 0, 50 -> 1, 60 -> 1, 80 -> 2, 99 -> 2.
        Bucket 3 (num_buckets - 1) is the reserved INACTIVE row and must never
        appear among positives.
        """
        expected = [0, 0, 0, 1, 1, 2, 2]
        buckets, effective, shape = _sft_buckets(
            num_buckets=4,
            negative_selection=NEGATIVE_SELECTION_CFG,
        )
        assert effective == 3
        assert shape == (N_FEATURES, 2)
        assert buckets == expected
        assert max(buckets) < 3, "positive must never occupy the inactive bucket"

    def test_binary_mode_puts_every_positive_in_bucket_zero(self) -> None:
        """num_buckets=2 + negative_selection => effective_buckets=1.

        This is the production d12/l22 configuration.  One effective bucket has
        no boundaries at all, so every positive is bucket 0 and bucket 1 is the
        inactive marker.  Pre-fix SFT produced [0, 0, 0, 1, 1, 1, 1] here — four
        of seven positives written into the inactive row.
        """
        expected = [0] * N_FEATURES
        buckets, effective, shape = _sft_buckets(
            num_buckets=2,
            negative_selection=NEGATIVE_SELECTION_CFG,
        )
        assert effective == 1
        assert shape == (N_FEATURES, 0), "binary mode carries zero boundary columns"
        assert buckets == expected

    def test_sampling_selection_reduces_identically(self) -> None:
        """The reduction is a property of the checkpoint, not of the selection
        mode; the sampling path (d12/l22 use ``dist_geometric``) reduces too."""
        expected = [0] * N_FEATURES
        buckets, effective, _ = _sft_buckets(
            num_buckets=2,
            negative_selection=NEGATIVE_SELECTION_CFG,
            selection="sampling",
        )
        assert effective == 1
        assert buckets == expected

    def test_raw_num_buckets_encoding_is_what_the_bug_produced(self) -> None:
        """Non-vacuity: the pre-fix encoding really does differ, and really does
        occupy the inactive bucket.  Hand-computed from boundary p50 = 50.0."""
        fids, fmags = _fids_and_mags()
        pre_fix = discretize_activations(
            fids,
            fmags,
            build_boundaries_from_stats(_stats(), N_FEATURES, 2),
        ).tolist()
        assert pre_fix == [0, 0, 0, 1, 1, 1, 1]
        fixed, _, _ = _sft_buckets(num_buckets=2, negative_selection=NEGATIVE_SELECTION_CFG)
        assert fixed != pre_fix
        assert sum(1 for b in pre_fix if b == 1) == 4

    def test_num_buckets_one_with_negatives_raises_rather_than_emptying_the_space(
        self,
    ) -> None:
        """Degenerate config: ``effective_buckets == 0`` leaves no positive
        bucket at all.  It must fail loudly at pipeline construction, not
        silently produce a zero-width space."""
        with pytest.raises(
            ValueError,
            match=r"leaves effective_buckets=0 — an empty positive bucket space",
        ):
            _sft_buckets(num_buckets=1, negative_selection=NEGATIVE_SELECTION_CFG)

    def test_num_buckets_one_without_negatives_is_untouched(self) -> None:
        """The guard is gated on ``negative_selection``: an unconditioned
        lineage with a single bucket keeps whatever behaviour it had."""
        buckets, effective, _ = _sft_buckets(num_buckets=1, negative_selection=None)
        assert effective == 1
        assert buckets == [0] * N_FEATURES


# ---------------------------------------------------------------------------
# 2. No-op pin: no negative_selection => bitwise pre-change output
# ---------------------------------------------------------------------------


class TestNoNegativeSelectionIsByteIdentical:
    @pytest.mark.parametrize("num_buckets", [2, 3, 4])
    def test_buckets_match_frozen_pre_change_golden(self, num_buckets: int) -> None:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        assert golden["fixture"]["magnitudes"] == MAGNITUDES
        entry = golden["no_negative_selection"][str(num_buckets)]

        buckets, effective, shape = _sft_buckets(
            num_buckets=num_buckets,
            negative_selection=None,
        )
        assert effective == num_buckets
        assert shape == (N_FEATURES, num_buckets - 1)
        assert buckets == entry["buckets"]

    @pytest.mark.parametrize("num_buckets", [2, 3, 4])
    def test_boundaries_match_frozen_pre_change_golden(self, num_buckets: int) -> None:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        entry = golden["no_negative_selection"][str(num_buckets)]
        pipeline = _sft_call_site_pipeline(
            feature_selection=parse_feature_selection(
                {"threshold_type": "segment_topk", "threshold": N_FEATURES}
            ),
            pretrain_negative_selection=None,
            per_feature_stats=_stats(),
            n_sae_features=N_FEATURES,
            num_buckets=num_buckets,
            ddp_rank=0,
            _sft_feature_mask=None,
        )
        assert pipeline.boundaries[0].tolist() == entry["boundaries_row0"]
        assert pipeline.negative_bundle is None, (
            "an unconditioned lineage must produce no negative bundle, so the "
            "generator emits no negatives and stays byte-identical"
        )


# ---------------------------------------------------------------------------
# 3. Model/data split: the embedding table keeps the RAW num_buckets
# ---------------------------------------------------------------------------


class TestModelKeepsRawNumBuckets:
    def test_magnitude_embedding_is_sized_num_buckets_not_effective_buckets(self) -> None:
        model = build_tiny_model(n_sae_features=N_FEATURES, num_buckets=2)
        assert model.encoder.magnitude_embed.num_embeddings == 2

    def test_strict_load_succeeds_at_num_buckets_and_fails_at_effective_buckets(
        self,
    ) -> None:
        """Getting the split backwards — sizing the model from
        ``effective_buckets`` — must break the strict state-dict load, which is
        exactly the failure this test exists to catch."""
        checkpoint = build_tiny_model(n_sae_features=N_FEATURES, num_buckets=2)
        state = checkpoint.state_dict()

        same_width = build_tiny_model(n_sae_features=N_FEATURES, num_buckets=2)
        same_width.load_state_dict(state, strict=True)

        reduced = build_tiny_model(n_sae_features=N_FEATURES, num_buckets=1)
        with pytest.raises(RuntimeError, match="magnitude_embed.weight"):
            reduced.load_state_dict(state, strict=True)

    def test_sft_source_builds_the_model_from_raw_num_buckets(self) -> None:
        """Source pin: the model construction call must pass ``num_buckets``,
        while the pipeline call must pass the checkpoint's negative_selection
        (which is what reduces the DATA encoding)."""
        model_kwargs = _call_site_kwargs("EncoderDecoderGPT")
        pipeline_kwargs = _call_site_kwargs("build_feature_pipeline")

        assert model_kwargs.get("num_buckets") == "num_buckets", model_kwargs
        assert pipeline_kwargs.get("num_buckets") == "num_buckets"
        assert pipeline_kwargs.get("negative_selection") == "pretrain_negative_selection"

    def test_sft_source_no_longer_hand_rolls_boundaries(self) -> None:
        """The reduction has exactly one production owner."""
        source = SFT_SOURCE.read_text(encoding="utf-8")
        assert "build_boundaries_from_stats" not in source
        assert "build_feature_pipeline(" in source

    def test_the_wrapper_is_gone_and_the_reasoning_survived_as_a_comment(self) -> None:
        """The owner asked for the pass-through wrapper to be inlined.  The
        reasoning it carried is the reason the bug cannot silently return, so it
        must still be at the call site."""
        source = SFT_SOURCE.read_text(encoding="utf-8")
        assert "build_sft_feature_pipeline" not in source
        assert not hasattr(sft, "build_sft_feature_pipeline")
        assert "effective_buckets = num_buckets - 1" in source
        assert "packed_batch_io.py" in source
        assert "inactive row" in source

    def test_reduced_boundaries_are_what_reach_the_data_generators(self) -> None:
        """Closes the loop: the pipeline's tensor is assigned to
        ``feature_boundaries`` and that is what both the train and the
        validation ``make_sft_data_generator`` calls condition on, so the
        reduction actually reaches ``enc_mags``."""
        source = SFT_SOURCE.read_text(encoding="utf-8")
        assert "feature_boundaries = feature_pipeline.boundaries" in source
        assert source.count("boundaries=feature_boundaries") == 2


# ---------------------------------------------------------------------------
# 4. The HARDCODED delta policy (no config knob)
# ---------------------------------------------------------------------------


NEGATIVES_ONLY = CheckpointConditioning(
    negative_selection=NEGATIVE_SELECTION_CFG,
    feature_delta=None,
    delta_weights=None,
)
SPLICE_COUNTERFACTUAL = CheckpointConditioning(
    negative_selection=NEGATIVE_SELECTION_CFG,
    feature_delta={"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
    delta_weights="percentile_delta",
)
UNCONDITIONED = CheckpointConditioning(
    negative_selection=None,
    feature_delta=None,
    delta_weights=None,
)


class TestHardcodedConditioningPolicy:
    def test_splice_counterfactual_warns_and_proceeds(self, caplog) -> None:
        """The l22 production configuration.  Pretraining derived negatives
        from a spliced counterfactual completion; SFT samples them randomly from
        the checkpoint's own negative_selection instead.  The positive-ranking
        difference (delta_weights) is folded into the SAME message, not a second
        mechanism."""
        with caplog.at_level("WARNING"):
            assert_conditioning_supported(
                SPLICE_COUNTERFACTUAL,
                checkpoint_label="base checkpoint /ckpt",
            )
        text = caplog.text
        assert "splice_counterfactual" in text
        assert "percentile_delta" in text
        assert "samples them randomly" in text
        assert "plain activation percentile" in text
        assert "never reads delta_weights" in text
        assert "Proceeding." in text

    @pytest.mark.parametrize(
        "mode", [DeltaMode.prev_segment.value, DeltaMode.cum_max.value]
    )
    def test_cross_segment_delta_modes_raise(self, mode: str) -> None:
        conditioning = CheckpointConditioning(
            negative_selection=NEGATIVE_SELECTION_CFG,
            feature_delta={"mode": mode},
            delta_weights="percentile_delta",
        )
        with pytest.raises(
            ValueError,
            match=(
                rf"(?s)feature_delta\.mode='{mode}', which conditioned SFT has "
                r"no analogue for.*difference against another segment"
            ),
        ):
            assert_conditioning_supported(
                conditioning,
                checkpoint_label="base checkpoint /ckpt",
            )

    def test_negatives_only_checkpoint_is_silent(self, caplog) -> None:
        """Negative EMISSION is reproduced now, so a negatives-only checkpoint
        has nothing left to warn about."""
        with caplog.at_level("WARNING"):
            assert_conditioning_supported(
                NEGATIVES_ONLY,
                checkpoint_label="base checkpoint /ckpt",
            )
        assert caplog.text == ""

    def test_unconditioned_checkpoint_is_silent(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            assert_conditioning_supported(
                UNCONDITIONED,
                checkpoint_label="base checkpoint /ckpt",
            )
        assert caplog.text == ""

    def test_delta_weights_without_feature_delta_still_warns(self, caplog) -> None:
        """The positive-ranking axis is folded into the same message, so it is
        reported even when the delta block itself is absent."""
        conditioning = CheckpointConditioning(
            negative_selection=None,
            feature_delta=None,
            delta_weights="percentile_delta",
        )
        with caplog.at_level("WARNING"):
            assert_conditioning_supported(
                conditioning,
                checkpoint_label="base checkpoint /ckpt",
            )
        assert "percentile_delta" in caplog.text

    def test_unreadable_delta_mode_raises_rather_than_defaulting_to_no_delta(
        self,
    ) -> None:
        conditioning = CheckpointConditioning(
            negative_selection=None,
            feature_delta={"mode": "some_future_mode"},
            delta_weights=None,
        )
        with pytest.raises(
            ValueError,
            match=r"feature_delta\.mode='some_future_mode' is not a known delta mode",
        ):
            assert_conditioning_supported(
                conditioning,
                checkpoint_label="base checkpoint /ckpt",
            )

    def test_non_mapping_feature_delta_raises(self) -> None:
        conditioning = CheckpointConditioning(
            negative_selection=None,
            feature_delta="splice_counterfactual",
            delta_weights=None,
        )
        with pytest.raises(
            ValueError, match=r"checkpoint feature_delta must be a mapping, got str"
        ):
            assert_conditioning_supported(
                conditioning,
                checkpoint_label="base checkpoint /ckpt",
            )

    def test_the_gate_runs_after_the_universe_guards_and_before_external_work(
        self,
    ) -> None:
        """Ordering contract. A base checkpoint whose feature UNIVERSE disagrees
        with the run is a more fundamental mismatch than how that universe was
        conditioned, so the mask/cluster/vocab guard must win; but the gate must
        still precede W&B, profile and tokenizer work.  Behaviourally covered by
        ``tests/steering/test_feature_clusters_unsupported_surfaces.py``; pinned
        here so the ordering is visible where the gate is defined."""
        source = SFT_SOURCE.read_text(encoding="utf-8")
        universe_guard = source.index("_assert_sft_universe_matches_checkpoint_vocab(\n")
        gate = source.index("    assert_conditioning_supported(\n")
        wandb = source.index("wandb_run = create_wandb_run(")
        tokenizer = source.index("tokenizer = get_tokenizer()")
        assert universe_guard < gate < wandb
        assert gate < tokenizer

    def test_policy_is_not_configurable(self) -> None:
        """No knob anywhere: not in the module, not in any shipped SFT config,
        not in the training script."""
        import steering.conditional.checkpoint_conditioning as cc

        source = SFT_SOURCE.read_text(encoding="utf-8")
        assert "acknowledged_conditioning_shifts" not in source
        assert not hasattr(cc, "ConditioningShift")
        assert not hasattr(cc, "parse_acknowledged_conditioning_shifts")
        assert not hasattr(cc, "assert_conditioning_shifts_acknowledged")
        assert "assert_conditioning_supported(" in source


# ---------------------------------------------------------------------------
# 5. Contract discovery and chain provenance
# ---------------------------------------------------------------------------


class TestCheckpointConditioningReader:
    def test_pretrain_meta_is_read_from_user_config(self) -> None:
        meta = {
            "user_config": {
                "negative_selection": NEGATIVE_SELECTION_CFG,
                "feature_delta": {"mode": "splice_counterfactual"},
                "feature_selection": {"delta_weights": "percentile_delta"},
            }
        }
        conditioning = read_checkpoint_conditioning(meta)
        assert conditioning.negative_selection == NEGATIVE_SELECTION_CFG
        assert conditioning.delta_weights == "percentile_delta"
        assert conditioning.delta_mode() is DeltaMode.splice_counterfactual

    def test_legacy_reader_behaviour_is_unchanged_for_existing_metas(self) -> None:
        """``checkpoint_negative_selection`` kept its exact semantics through the
        re-home: user_config when present, else the meta itself, else None."""
        assert checkpoint_negative_selection(None) is None
        assert checkpoint_negative_selection({}) is None
        assert (
            checkpoint_negative_selection({"negative_selection": NEGATIVE_SELECTION_CFG})
            == NEGATIVE_SELECTION_CFG
        )
        assert (
            checkpoint_negative_selection(
                {"user_config": {"negative_selection": NEGATIVE_SELECTION_CFG}}
            )
            == NEGATIVE_SELECTION_CFG
        )
        assert (
            checkpoint_negative_selection(
                {"user_config": {}, "negative_selection": NEGATIVE_SELECTION_CFG}
            )
            is None
        ), "user_config wins when present, exactly as before the re-home"

    def test_sft_provenance_block_round_trips_for_a_chained_run(self) -> None:
        pretrain_meta = {"user_config": {"negative_selection": NEGATIVE_SELECTION_CFG}}
        inherited = read_checkpoint_conditioning(pretrain_meta)
        sft_meta = {
            "user_config": {"dataset": "safety"},
            "encoder_config": {},
            **inherited.metadata_dict(),
        }
        assert PRETRAIN_CONDITIONING_META_KEY in sft_meta
        chained = read_checkpoint_conditioning(sft_meta)
        assert chained.negative_selection == NEGATIVE_SELECTION_CFG
        assert chained.delta_mode() is None

    def test_unconditioned_lineage_adds_no_metadata_key(self) -> None:
        assert UNCONDITIONED.metadata_dict() == {}

    def test_pre_fix_sft_base_is_rejected_not_read_as_unconditioned(self) -> None:
        legacy_sft_meta = {
            "user_config": {"dataset": "safety"},
            "encoder_config": {},
            "xattn_config": {},
            "num_buckets": 2,
        }
        assert read_checkpoint_conditioning(legacy_sft_meta).negative_selection is None
        with pytest.raises(
            ValueError,
            match=(
                r"(?s)conditioned-SFT checkpoint with no 'pretrain_conditioning' "
                r"metadata.*"
                r"absence must not be read as 'no negative_selection'"
            ),
        ):
            assert_pretrain_conditioning_recoverable(
                legacy_sft_meta,
                checkpoint_label="base checkpoint /ckpt",
            )

    def test_pretrain_base_and_stamped_sft_base_both_pass(self) -> None:
        assert_pretrain_conditioning_recoverable(
            {"user_config": {"encoder": {}}},
            checkpoint_label="base checkpoint /ckpt",
        )
        assert_pretrain_conditioning_recoverable(
            {
                "user_config": {"dataset": "safety"},
                "encoder_config": {},
                PRETRAIN_CONDITIONING_META_KEY: {"negative_selection": None},
            },
            checkpoint_label="base checkpoint /ckpt",
        )


class TestRealCheckpointMetadata:
    """The production artifacts this bug actually affected.

    Skips when the local checkpoint inventory is absent (CI); on a machine that
    has it, this is the non-synthetic proof that the affected population is
    exactly the negatives-trained encoder-decoder checkpoints.
    """

    @staticmethod
    def _pretrain_metas() -> list[Path]:
        root = Path.home() / ".cache" / "nanochat" / "conditional_encdec_checkpoints"
        if not root.is_dir():
            return []
        return [
            path
            for path in sorted(root.rglob("meta_*.json"))
            if "sft" not in path.parent.name
        ]

    def test_every_local_pretrain_checkpoint_reduces_and_is_policed(self) -> None:
        metas = self._pretrain_metas()
        if not metas:
            pytest.skip("no local conditional_encdec_checkpoints inventory")

        seen_negatives = False
        for path in metas:
            meta = json.loads(path.read_text(encoding="utf-8"))
            conditioning = read_checkpoint_conditioning(meta)
            if conditioning.negative_selection is None:
                continue
            seen_negatives = True
            user_config = meta.get("user_config", {})
            num_buckets = int(user_config.get("num_buckets", 2))
            parsed = parse_negative_selection(dict(conditioning.negative_selection))
            pipeline = _sft_call_site_pipeline(
                feature_selection=parse_feature_selection(
                    {"threshold_type": "segment_topk", "threshold": N_FEATURES}
                ),
                pretrain_negative_selection=parsed,
                per_feature_stats=_stats(),
                n_sae_features=N_FEATURES,
                num_buckets=num_buckets,
                ddp_rank=0,
                _sft_feature_mask=None,
            )
            assert pipeline.effective_buckets == num_buckets - 1
            assert pipeline.negative_bundle is not None, (
                "a negatives-trained checkpoint must now yield a negative "
                "bundle so SFT emits negatives"
            )
            # Every local checkpoint's delta mode is either absent or
            # splice_counterfactual, so the policy must let all of them run.
            assert_conditioning_supported(
                conditioning,
                checkpoint_label=str(path.parent),
            )
        assert seen_negatives, (
            "local inventory exists but no pretrain checkpoint carried "
            "negative_selection; the fixture assumption changed"
        )


# ---------------------------------------------------------------------------
# 6. Config surface: the knob is gone, and nothing reintroduces it
# ---------------------------------------------------------------------------


class TestConfigSurface:
    @pytest.mark.parametrize("config_name", SFT_CONFIG_NAMES)
    def test_no_acknowledgement_key_in_any_sft_config(
        self, hydra_compose, config_name: str
    ) -> None:
        cfg = hydra_compose(str(REPO_ROOT / "configs" / "training"), config_name)
        assert "acknowledged_conditioning_shifts" not in cfg

    def test_shipped_config_text_carries_no_knob(self) -> None:
        text = (
            REPO_ROOT / "configs" / "training" / "conditional_encdec_sft.yaml"
        ).read_text(encoding="utf-8")
        assert "acknowledged_conditioning_shifts" not in text
        assert "assert_conditioning_supported" in text, (
            "the config should point operators at the hardcoded policy's owner"
        )
