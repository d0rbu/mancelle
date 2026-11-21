"""Pin the real checkpoint-metadata shapes the chat TUI setup loader relies on.

These are ground-truth pins over verbatim fixture copies of the production SFT leaf
(``meta_000749.json``) and its pretraining ancestor (``meta_026500.json``). Task 5's
``load_chat_setup`` argues from exactly these facts; if a future checkpoint generation
changes shape, re-pin deliberately (see ``tests/fixtures/chat_tui/README.md``).

Key facts deliberately pinned because they DIVERGE from naive expectations
(plan §Addenda A1):

- The leaf's ``user_config.sae`` is the unrelated config DEFAULT (gemma-2-2b l8 16k), not
  the conditioning identity — SAE identity must come from the ancestor conditioning meta.
- The leaf records NO seed key and NO stats binding — both resolve via the ancestor.
- The leaf's ``user_config.negative_selection`` is ``None`` — the ``pretrain_conditioning``
  fallback is the required path, not a backstop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.conditional.checkpoint_conditioning import (
    read_checkpoint_conditioning,
)
from steering.conditional.delta import DeltaMode
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
    parse_feature_selection,
    parse_negative_selection,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "chat_tui"

EXPECTED_STATS_SHA = "dadf6d4f8d3b56ba1b40e0f49f4bb92bfb635c64f78a530f04f624b1d5773b5d"
EXPECTED_VOCAB_SHA = "8c183369ce4a8ff1e5f153a749e9eb71f9def65283d0a22b4d33bf5ff2d6f16c"
EXPECTED_L22_SAE = {
    "release": "gemmascope_2",
    "model": "gemma-3-4b-pt",
    "activation_type": "residual",
    "layer": 22,
    "width": "65k",
    "l0": "medium",
}


@pytest.fixture(scope="module")
def leaf_meta() -> dict:
    return json.loads((FIXTURES / "meta_000749.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ancestor_meta() -> dict:
    return json.loads((FIXTURES / "meta_026500.json").read_text(encoding="utf-8"))


class TestLeafShape:
    def test_leaf_is_sft_shaped(self, leaf_meta: dict) -> None:
        assert leaf_meta["step"] == 749
        assert "encoder_config" in leaf_meta
        assert "encoder" not in leaf_meta["user_config"]
        assert "conditioning" not in leaf_meta["user_config"]

    def test_leaf_sae_block_is_the_unrelated_config_default(self, leaf_meta: dict) -> None:
        """The SFT run never overrode ``sae``; reading it as identity would be wrong."""
        sae = leaf_meta["user_config"]["sae"]
        assert sae["model"] == "gemma-2-2b"
        assert sae["width"] == "16k"
        # It contradicts the leaf's own vocabulary width, which is the tell:
        assert leaf_meta["feature_vocab"]["n_sae_features"] == 65_536

    def test_leaf_records_no_seed_and_no_stats_binding(self, leaf_meta: dict) -> None:
        user_config = leaf_meta["user_config"]
        seed_keys = [key for key in user_config if "seed" in key.lower()]
        assert seed_keys == []
        stats_keys = [
            key for key in user_config if "stats" in key.lower() or "sha256" in key.lower()
        ]
        assert stats_keys == []

    def test_leaf_output_selection_parses_delta_free(self, leaf_meta: dict) -> None:
        selection = parse_feature_selection(leaf_meta["user_config"]["feature_selection"])
        assert isinstance(selection, SamplingSelection)
        assert selection.mode is SamplingMode.top_k_percentile
        assert selection.delta_weights is None
        assert float(selection.mean) == 12.0
        assert int(selection.min_k) == 1
        assert int(selection.max_k) == 49

    def test_leaf_negatives_come_only_from_pretrain_conditioning(self, leaf_meta: dict) -> None:
        # The key is entirely ABSENT on the leaf (not null): `read_checkpoint_conditioning`
        # treats absent and None identically, but the pin records the real shape.
        assert "negative_selection" not in leaf_meta["user_config"]
        conditioning = read_checkpoint_conditioning(leaf_meta)
        negative = parse_negative_selection(conditioning.negative_selection)
        assert isinstance(negative, NegativeSamplingSelection)
        assert float(negative.mean) == 12.0
        assert int(negative.min_k) == 0
        assert int(negative.max_k) == 49
        assert conditioning.delta_mode() is DeltaMode.splice_counterfactual
        assert conditioning.delta_weights == "percentile_delta"

    def test_leaf_head_and_bucket_contract(self, leaf_meta: dict) -> None:
        fp_config = leaf_meta["feature_prediction_config"]
        assert fp_config["loss_mode"] == "max_pool"
        assert fp_config["loss_fn"] == "bce"
        assert leaf_meta["num_buckets"] == 2
        assert leaf_meta["user_config"]["num_buckets"] == 2

    def test_leaf_feature_contract_blobs_present(self, leaf_meta: dict) -> None:
        assert leaf_meta["feature_mask_canonical"]["mode"] == "whitelist"
        assert len(leaf_meta["feature_mask_canonical"]["ids"]) == 16_254
        assert "feature_clusters_descriptor" in leaf_meta
        assert "feature_clusters_canonical" in leaf_meta
        assert leaf_meta["feature_vocab"] == {
            "n_sae_features": 65_536,
            "vocab_size": 2_074,
            "sha256": EXPECTED_VOCAB_SHA,
        }

    def test_leaf_sft_render_contract_fields(self, leaf_meta: dict) -> None:
        assert leaf_meta["user_config"]["feature_source_bos_char_len"] == 5
        assert leaf_meta["dataset"] == "standard"
        assert leaf_meta["sft_loss_mask"] == "full_sequence"
        assert leaf_meta["missing_percentile_policy"] == "lowest"


class TestAncestorShape:
    def test_ancestor_is_pretrain_shaped(self, ancestor_meta: dict) -> None:
        assert "conditioning" in ancestor_meta["user_config"]
        assert "encoder" in ancestor_meta["user_config"]

    def test_ancestor_conditioning_block_pins(self, ancestor_meta: dict) -> None:
        conditioning = ancestor_meta["user_config"]["conditioning"]
        assert conditioning["stats_source_sha256"] == EXPECTED_STATS_SHA
        assert conditioning["seed"] == 42
        assert conditioning["n_sae_features"] == 65_536
        assert conditioning["num_buckets"] == 2

    def test_ancestor_sae_is_the_l22_identity(self, ancestor_meta: dict) -> None:
        assert ancestor_meta["user_config"]["sae"] == EXPECTED_L22_SAE

    def test_ancestor_prompt_contract_is_internally_consistent(self, ancestor_meta: dict) -> None:
        """The intentional ambiguity check compares these two pairs; they must agree."""
        user_config = ancestor_meta["user_config"]
        conditioning = user_config["conditioning"]
        assert conditioning["feature_mask"] == user_config["feature_mask"]
        assert conditioning["feature_clusters"] == user_config["feature_clusters"]

    def test_ancestor_prompt_selection_parses(self, ancestor_meta: dict) -> None:
        conditioning = ancestor_meta["user_config"]["conditioning"]
        selection = parse_feature_selection(conditioning["feature_selection"])
        assert isinstance(selection, SamplingSelection)
        assert selection.mode is SamplingMode.top_k_percentile
        negative = parse_negative_selection(conditioning["negative_selection"])
        assert isinstance(negative, NegativeSamplingSelection)

    def test_ancestor_and_leaf_share_the_feature_vocabulary(
        self, ancestor_meta: dict, leaf_meta: dict
    ) -> None:
        """Prompt-vocab == output-vocab holds across the lineage by construction."""
        assert ancestor_meta["feature_vocab"] == leaf_meta["feature_vocab"]

    def test_ancestor_target_policy_matches_axbench_requirements(self, ancestor_meta: dict) -> None:
        """`load_method_setup` on this meta must clear `_validate_target_policy`."""
        user_config = ancestor_meta["user_config"]
        assert user_config["feature_delta"]["mode"] == "splice_counterfactual"
        selection = parse_feature_selection(user_config["feature_selection"])
        assert isinstance(selection, SamplingSelection)
        assert selection.delta_weights is not None
        assert selection.delta_weights.value == "percentile_delta"
