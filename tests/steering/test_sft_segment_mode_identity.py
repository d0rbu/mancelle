"""Resume identity for conditioned-SFT feature segmentation."""

from __future__ import annotations

import pytest

from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import (
    ONLINE_SAE_LINEAGE_META_KEY,
    ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY,
    SFT_SEGMENT_MODE_META_KEY,
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    assert_resume_sft_segment_mode_matches,
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
    segmentation_identity_drift,
)


def _online_lineage_mode(mode: SegmentMode | None) -> dict[str, object]:
    lineage: dict[str, object] = {"schema_version": 1}
    if mode is not None:
        lineage[ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY] = mode.value
    return lineage


def _sft_selfcond_meta(
    mode: SegmentMode,
    *,
    cadence: object = 128,
) -> dict[str, object]:
    """A self-conditioning meta whose cadence sits where post-training writes it."""
    return {
        TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: cadence,
        "selfcond": {"data_source": "sft"},
        "selfcond_data": {SFT_SEGMENT_MODE_META_KEY: mode.value},
    }


def _witness_meta(witness: str, mode: SegmentMode) -> dict[str, object]:
    """Metadata whose only segment-mode witness is ``witness``."""
    if witness == "top":
        return {SFT_SEGMENT_MODE_META_KEY: mode.value}
    if witness == "lineage":
        return {ONLINE_SAE_LINEAGE_META_KEY: _online_lineage_mode(mode)}
    return _sft_selfcond_meta(mode)


@pytest.mark.parametrize("meta", [None, {"step": 1}], ids=["absent-meta", "absent-key"])
def test_absent_witness_uses_legacy_fixed_mode(meta: dict[str, object] | None) -> None:
    assert_resume_sft_segment_mode_matches(
        SegmentMode.fixed_tokens,
        meta,
        checkpoint_dir="/checkpoints/legacy",
    )


# kills: dropping any one witness read (top-level, online-SAE lineage, or
# nested selfcond_data) from resolve_checkpoint_sft_segment_mode, and
# narrowing assert_resume_sft_segment_mode_matches to accept only fixed_tokens.
@pytest.mark.parametrize("witness", ["top", "lineage", "selfcond"])
def test_each_witness_alone_resolves_the_turn_mode(witness: str) -> None:
    turn = SegmentMode.turn_delimited_fixed_tokens
    meta = _witness_meta(witness, turn)
    assert (witness == "top") is (SFT_SEGMENT_MODE_META_KEY in meta)
    assert_resume_sft_segment_mode_matches(
        turn,
        meta,
        checkpoint_dir="/checkpoints/witness",
    )


@pytest.mark.parametrize(
    ("configured", "recorded"),
    [
        (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
        (SegmentMode.turn_delimited_fixed_tokens, SegmentMode.fixed_tokens),
    ],
)
def test_mismatch_names_checkpoint_and_both_modes(
    configured: SegmentMode,
    recorded: SegmentMode,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"/checkpoints/mismatch.*{recorded.value}.*{configured.value}.*"
            r"recorded segmentation mode.*fresh SFT run"
        ),
    ):
        assert_resume_sft_segment_mode_matches(
            configured,
            {SFT_SEGMENT_MODE_META_KEY: recorded.value},
            checkpoint_dir="/checkpoints/mismatch",
        )


def test_unknown_recorded_mode_raises() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"/checkpoints/corrupt.*unknown.*sft_segment_mode.*bogus.*"
            r"must be one of.*fixed_tokens.*turn_delimited_fixed_tokens"
        ),
    ):
        assert_resume_sft_segment_mode_matches(
            SegmentMode.fixed_tokens,
            {SFT_SEGMENT_MODE_META_KEY: "bogus"},
            checkpoint_dir="/checkpoints/corrupt",
        )


def test_legacy_online_lineage_omission_is_a_fixed_mode_witness() -> None:
    meta = {ONLINE_SAE_LINEAGE_META_KEY: _online_lineage_mode(None)}
    assert (
        resolve_checkpoint_sft_segment_mode(
            meta,
            checkpoint_dir="/checkpoints/lineage",
        )
        is SegmentMode.fixed_tokens
    )


def test_all_consistent_witnesses_are_accepted() -> None:
    mode = SegmentMode.turn_delimited_fixed_tokens
    meta = _sft_selfcond_meta(mode)
    meta[SFT_SEGMENT_MODE_META_KEY] = mode.value
    meta[ONLINE_SAE_LINEAGE_META_KEY] = _online_lineage_mode(mode)
    assert (
        resolve_checkpoint_sft_segment_mode(
            meta,
            checkpoint_dir="/checkpoints/consistent",
        )
        is mode
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("top", "lineage"),
        ("top", "selfcond"),
        ("lineage", "selfcond"),
    ],
)
def test_conflicting_witnesses_fail_closed(left: str, right: str) -> None:
    meta = _witness_meta(left, SegmentMode.fixed_tokens)
    meta.update(_witness_meta(right, SegmentMode.turn_delimited_fixed_tokens))

    with pytest.raises(ValueError, match=r"conflicting segment-mode witnesses"):
        resolve_checkpoint_sft_segment_mode(
            meta,
            checkpoint_dir="/checkpoints/conflict",
        )


@pytest.mark.parametrize(
    ("raw", "rendered"),
    [(None, "None"), ("unknown", "'unknown'")],
)
def test_top_level_mode_requires_a_known_string(raw: object, rendered: str) -> None:
    # kills: swallowing the offending value (or the witness name) out of the
    # message -- the bare key appears in it either way, so only the rendered
    # raw value distinguishes a real report from a generic one.
    with pytest.raises(
        ValueError,
        match=rf"sft_segment_mode has unknown sft_segment_mode={rendered}; must be one of",
    ):
        resolve_checkpoint_sft_segment_mode(
            {SFT_SEGMENT_MODE_META_KEY: raw},
            checkpoint_dir="/checkpoints/malformed",
        )


@pytest.mark.parametrize(
    "lineage",
    [
        None,
        [],
        {},
        {"schema_version": 2},
        {
            "schema_version": 1,
            ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY: "unknown",
        },
    ],
)
def test_malformed_online_lineage_mode_witness_is_rejected(lineage: object) -> None:
    with pytest.raises(ValueError, match=ONLINE_SAE_LINEAGE_META_KEY):
        resolve_checkpoint_sft_segment_mode(
            {ONLINE_SAE_LINEAGE_META_KEY: lineage},
            checkpoint_dir="/checkpoints/malformed-lineage",
        )


# kills: dropping the `selfcond_data` Mapping guard (a list satisfies the `in`
# test and would reach the parser as a beartype crash), and collapsing the
# nested witness's error into another branch's message. The checkpoint_dir
# deliberately contains none of the matched fragments.
@pytest.mark.parametrize(
    ("selfcond_data", "match"),
    [
        ([SFT_SEGMENT_MODE_META_KEY], r"selfcond_data must be a mapping, got list"),
        (
            {SFT_SEGMENT_MODE_META_KEY: "bogus"},
            r"selfcond_data\.sft_segment_mode has unknown sft_segment_mode='bogus'",
        ),
    ],
)
def test_malformed_selfcond_data_witness_is_rejected(
    selfcond_data: object,
    match: str,
) -> None:
    checkpoint_dir = "/checkpoints/corrupt-block"
    assert "selfcond" not in checkpoint_dir
    with pytest.raises(ValueError, match=match):
        resolve_checkpoint_sft_segment_mode(
            {"selfcond_data": selfcond_data},
            checkpoint_dir=checkpoint_dir,
        )


def _main_minted_legacy_sft_selfcond_meta() -> dict[str, object]:
    """Reproduce origin/main's ``build_selfcond_checkpoint_meta`` output shape.

    main's trainer never wrote ``sft_segment_mode`` anywhere: neither at the
    top level nor inside ``selfcond_data`` (git show
    origin/main:scripts/train/conditional_encdec_selfcond.py:2002-2026,
    2793-2817), and its ``SegmentMode`` StrEnum had only ``fixed_tokens``.
    """
    return {
        "model_config": {"sequence_len": 2048},
        "encoder_config": {"depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_prediction_config": {"loss_mode": "max_pool"},
        "user_config": {"segment_size": 128},
        "base_checkpoint_dir": "/checkpoints/base",
        "selfcond": {
            "action_ranking_policy": "fp_logit",
            "train_scope": "head_only",
            "ste_form": "hard",
            "pool_mode": "max_pool",
            "positive_counts": {"mean": 1, "std": 1, "min_k": 1, "max_k": 2},
            "negative_counts": None,
            "seed": 42,
            "original_head_loss_mode": "max_pool",
            "data_source": "sft",
        },
        "loop_state": {"step": 10},
        "selfcond_data": {
            "dataset": "smoltalk",
            "train_parquet_dir": "/data/train",
            "train_feature_dir": "/features/train",
            "val_parquet_dir": "/data/val",
            "val_feature_dir": "/features/val",
            "sft_loss_mask": "assistant_only",
            "min_loss_tokens": 8,
            "feature_source_bos_char_len": 0,
            "tokenizer_checksum": "a" * 64,
            "percentile_checksum": "b" * 64,
            "feature_selection": {"mode": "top_k_percentile"},
            "negative_selection": None,
            "missing_percentile_policy": "error",
        },
    }


# kills: re-adding the `selfcond_data.sft_segment_mode is mandatory` rejection,
# which would make every main-minted conditioned-SFT self-conditioning
# checkpoint unloadable by all ~15 fixed-mode inference/eval surfaces; and
# widening that legacy acceptance into acceptance of the same shape carrying a
# real turn witness.
def test_legacy_sft_selfcond_without_nested_mode_resolves_fixed() -> None:
    meta = _main_minted_legacy_sft_selfcond_meta()
    assert SFT_SEGMENT_MODE_META_KEY not in meta
    selfcond_data = meta["selfcond_data"]
    assert isinstance(selfcond_data, dict)
    assert SFT_SEGMENT_MODE_META_KEY not in selfcond_data

    assert (
        resolve_checkpoint_sft_segment_mode(
            meta,
            checkpoint_dir="/checkpoints/main-minted",
        )
        is SegmentMode.fixed_tokens
    )
    assert (
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/main-minted",
        )
        is None
    )
    # kills: fixed guard rejecting a valid legacy checkpoint (success returns None).
    assert require_fixed_checkpoint_segment_mode(
        meta,
        surface="chat TUI",
        checkpoint_dir="/checkpoints/main-minted",
    ) is None

    turn_meta = _main_minted_legacy_sft_selfcond_meta()
    turn_meta[SFT_SEGMENT_MODE_META_KEY] = SegmentMode.turn_delimited_fixed_tokens.value
    with pytest.raises(
        ValueError,
        match=r"chat TUI does not support checkpoint .*turn_delimited_fixed_tokens",
    ):
        require_fixed_checkpoint_segment_mode(
            turn_meta,
            surface="chat TUI",
            checkpoint_dir="/checkpoints/turn",
        )


# kills: dropping the `else SegmentMode.fixed_tokens` arm that turns a v1
# lineage block WITHOUT source_segment_mode into an explicit fixed witness.
# Without it both metas below silently resolve turn_delimited_fixed_tokens.
@pytest.mark.parametrize("turn_witness", ["top", "selfcond"])
def test_lineage_without_source_segment_mode_conflicts_with_turn(
    turn_witness: str,
) -> None:
    meta: dict[str, object] = {ONLINE_SAE_LINEAGE_META_KEY: _online_lineage_mode(None)}
    meta.update(_witness_meta(turn_witness, SegmentMode.turn_delimited_fixed_tokens))

    with pytest.raises(
        ValueError,
        match=(
            r"conflicting segment-mode witnesses.*"
            r"structured_sft_online_sae_lineage_v1\.source_segment_mode='fixed_tokens'"
        ),
    ):
        resolve_checkpoint_sft_segment_mode(
            meta,
            checkpoint_dir="/checkpoints/lineage-omission",
        )


# kills: returning a cadence (or raising) for a fixed checkpoint or for one
# with no metadata at all, both of which every legacy consumer still loads.
@pytest.mark.parametrize("meta", [None, {}], ids=["absent-meta", "fixed-meta"])
def test_fixed_and_absent_metadata_have_no_turn_cadence(
    meta: dict[str, object] | None,
) -> None:
    assert (
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/legacy",
        )
        is None
    )


def test_turn_cadence_is_read_only_from_top_level_metadata() -> None:
    meta = _sft_selfcond_meta(
        SegmentMode.turn_delimited_fixed_tokens,
        cadence=37,
    )
    # A different source-tokenizer geometry must have no effect.
    meta["user_config"] = {"segment_size": 4096}
    assert (
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/turn-selfcond",
        )
        == 37
    )


# kills: reading the cadence from the rollout-policy `selfcond` block (its old
# home) instead of the metadata top level, which would silently accept a
# checkpoint minted before the cadence became a measured top-level value and
# feed inference a value the reader no longer validates.
def test_a_cadence_nested_under_selfcond_is_not_a_witness() -> None:
    meta: dict[str, object] = {
        SFT_SEGMENT_MODE_META_KEY: SegmentMode.turn_delimited_fixed_tokens.value,
        "selfcond": {
            "data_source": "sft",
            TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: 37,
        },
    }
    with pytest.raises(
        ValueError,
        match=r"require top-level post-training metadata",
    ):
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/nested-cadence",
        )


def test_turn_cadence_cannot_be_inferred_from_source_segment_size() -> None:
    meta = _sft_selfcond_meta(SegmentMode.turn_delimited_fixed_tokens)
    del meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY]
    meta["user_config"] = {"segment_size": 128}
    with pytest.raises(
        ValueError,
        match=r"turn_self_cond_segment_size_rustbpe.*cannot be inferred",
    ):
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/missing-cadence",
        )


@pytest.mark.parametrize("raw", [True, False, 0, -3, 1.5, "128", None])
def test_turn_cadence_requires_a_strict_positive_integer(raw: object) -> None:
    meta = _sft_selfcond_meta(
        SegmentMode.turn_delimited_fixed_tokens,
        cadence=raw,
    )
    with pytest.raises(ValueError, match=r"positive integer"):
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/bad-cadence",
        )


def test_fixed_checkpoint_rejects_inapplicable_turn_cadence() -> None:
    meta = _sft_selfcond_meta(SegmentMode.fixed_tokens, cadence=128)
    with pytest.raises(ValueError, match=r"valid only for turn-delimited"):
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/fixed-with-cadence",
        )


# kills: reintroducing a `selfcond.data_source` precondition on the cadence
# read, which the measured contract deliberately dropped -- the top-level key
# is written only by SFT turn post-training, and re-adding the guard would
# reject every checkpoint whose rollout block records a different source.
def test_turn_cadence_read_does_not_consult_the_selfcond_block() -> None:
    meta: dict[str, object] = {
        SFT_SEGMENT_MODE_META_KEY: SegmentMode.turn_delimited_fixed_tokens.value,
        TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: 64,
        "selfcond": "not-a-mapping",
    }
    assert (
        resolve_turn_self_cond_segment_size_rustbpe(
            meta,
            checkpoint_dir="/checkpoints/turn-opaque-selfcond",
        )
        == 64
    )


@pytest.mark.parametrize(
    ("loaded", "setup", "expected"),
    [
        ({}, {}, []),
        (
            {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: 37},
            {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: 37},
            [],
        ),
        (
            {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: None},
            {},
            [TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY],
        ),
        (
            {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: 37.0},
            {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: 37},
            [TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY],
        ),
        (
            {SFT_SEGMENT_MODE_META_KEY: "fixed_tokens", ONLINE_SAE_LINEAGE_META_KEY: {"a": 1}},
            {
                SFT_SEGMENT_MODE_META_KEY: "turn_delimited_fixed_tokens",
                ONLINE_SAE_LINEAGE_META_KEY: {"a": 1},
            },
            [SFT_SEGMENT_MODE_META_KEY],
        ),
    ],
    ids=["both-empty", "equal", "absent-vs-null", "int-vs-float", "one-key-drifts"],
)
def test_segmentation_identity_drift_separates_absent_null_and_numeric_types(
    loaded, setup, expected
):
    # kills: dict.get / raw != comparisons (absent == null, 37 == 37.0).
    assert segmentation_identity_drift(loaded, setup) == expected
