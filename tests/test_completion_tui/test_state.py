from __future__ import annotations

import pytest

from scripts.completion_tui.state import (
    FeatureCondition,
    GenerationSnapshot,
    Segment,
    SessionState,
    deep_copy_segments,
    display_magnitude_range,
    display_to_internal,
    internal_to_display,
)

# ---------------------------------------------------------------------------
# display_magnitude_range
# ---------------------------------------------------------------------------


class TestDisplayMagnitudeRange:
    def test_11_buckets_no_negatives(self) -> None:
        assert display_magnitude_range(11, has_negatives=False) == (0, 10)

    def test_11_buckets_with_negatives(self) -> None:
        assert display_magnitude_range(11, has_negatives=True) == (-1, 9)

    def test_2_buckets_with_negatives(self) -> None:
        assert display_magnitude_range(2, has_negatives=True) == (-1, 0)

    def test_2_buckets_no_negatives(self) -> None:
        assert display_magnitude_range(2, has_negatives=False) == (0, 1)

    def test_1_bucket_no_negatives(self) -> None:
        assert display_magnitude_range(1, has_negatives=False) == (0, 0)


# ---------------------------------------------------------------------------
# display_to_internal / internal_to_display
# ---------------------------------------------------------------------------


class TestMagnitudeMapping:
    def test_negative_display_maps_to_null_bucket(self) -> None:
        # -1 display → last bucket (null/negative)
        assert display_to_internal(-1, 11, has_negatives=True) == 10

    def test_internal_null_bucket_maps_to_negative_display(self) -> None:
        assert internal_to_display(10, 11, has_negatives=True) == -1

    def test_display_zero_with_negatives(self) -> None:
        # display 0 → internal 0 (first positive bucket)
        assert display_to_internal(0, 2, has_negatives=True) == 0

    def test_without_negatives_identity(self) -> None:
        # Without negatives, display == internal
        assert display_to_internal(5, 11, has_negatives=False) == 5

    def test_without_negatives_identity_reverse(self) -> None:
        assert internal_to_display(5, 11, has_negatives=False) == 5

    def test_round_trip_with_negatives_11_buckets(self) -> None:
        """All display values through internal and back == identity."""
        lo, hi = display_magnitude_range(11, has_negatives=True)
        for d in range(lo, hi + 1):
            internal = display_to_internal(d, 11, has_negatives=True)
            back = internal_to_display(internal, 11, has_negatives=True)
            assert back == d, f"Round-trip failed for display={d}"

    def test_round_trip_without_negatives_11_buckets(self) -> None:
        lo, hi = display_magnitude_range(11, has_negatives=False)
        for d in range(lo, hi + 1):
            internal = display_to_internal(d, 11, has_negatives=False)
            back = internal_to_display(internal, 11, has_negatives=False)
            assert back == d, f"Round-trip failed for display={d}"

    def test_round_trip_2_buckets_with_negatives(self) -> None:
        lo, hi = display_magnitude_range(2, has_negatives=True)
        for d in range(lo, hi + 1):
            internal = display_to_internal(d, 2, has_negatives=True)
            back = internal_to_display(internal, 2, has_negatives=True)
            assert back == d, f"Round-trip failed for display={d}"

    def test_round_trip_2_buckets_without_negatives(self) -> None:
        lo, hi = display_magnitude_range(2, has_negatives=False)
        for d in range(lo, hi + 1):
            internal = display_to_internal(d, 2, has_negatives=False)
            back = internal_to_display(internal, 2, has_negatives=False)
            assert back == d, f"Round-trip failed for display={d}"

    def test_display_to_internal_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="outside valid display range"):
            display_to_internal(11, 11, has_negatives=False)

    def test_display_to_internal_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="outside valid display range"):
            display_to_internal(-1, 11, has_negatives=False)

    def test_display_to_internal_out_of_range_low_with_neg(self) -> None:
        with pytest.raises(ValueError, match="outside valid display range"):
            display_to_internal(-2, 11, has_negatives=True)

    def test_internal_to_display_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="outside valid internal range"):
            internal_to_display(11, 11, has_negatives=False)

    def test_internal_to_display_out_of_range_negative(self) -> None:
        with pytest.raises(ValueError, match="outside valid internal range"):
            internal_to_display(-1, 11, has_negatives=False)


# ---------------------------------------------------------------------------
# Segment UUID uniqueness
# ---------------------------------------------------------------------------


class TestSegmentId:
    def test_unique_ids(self) -> None:
        s1 = Segment(
            segment_idx=0,
            char_start=0,
            char_end=10,
            color_idx=0,
            features=[],
        )
        s2 = Segment(
            segment_idx=1,
            char_start=10,
            char_end=20,
            color_idx=1,
            features=[],
        )
        assert s1.id != s2.id

    def test_id_is_8_hex_chars(self) -> None:
        s = Segment(
            segment_idx=0,
            char_start=0,
            char_end=5,
            color_idx=0,
            features=[],
        )
        assert len(s.id) == 8
        # Should be valid hex
        int(s.id, 16)

    def test_many_segments_all_unique(self) -> None:
        ids = {
            Segment(
                segment_idx=i,
                char_start=i * 10,
                char_end=(i + 1) * 10,
                color_idx=i % 4,
                features=[],
            ).id
            for i in range(100)
        }
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# deep_copy_segments
# ---------------------------------------------------------------------------


class TestDeepCopySegments:
    def test_preserves_ids(self) -> None:
        segs = [
            Segment(
                segment_idx=0,
                char_start=0,
                char_end=10,
                color_idx=0,
                features=[FeatureCondition(feature_id=42, magnitude=3)],
            ),
        ]
        copied = deep_copy_segments(segs)
        assert copied[0].id == segs[0].id

    def test_modifying_copy_does_not_affect_original(self) -> None:
        segs = [
            Segment(
                segment_idx=0,
                char_start=0,
                char_end=10,
                color_idx=0,
                features=[FeatureCondition(feature_id=42, magnitude=3)],
            ),
        ]
        copied = deep_copy_segments(segs)
        copied[0].features[0].magnitude = 999
        assert segs[0].features[0].magnitude == 3

    def test_features_list_is_independent(self) -> None:
        feat = FeatureCondition(feature_id=1, magnitude=0)
        segs = [
            Segment(
                segment_idx=0,
                char_start=0,
                char_end=5,
                color_idx=0,
                features=[feat],
            ),
        ]
        copied = deep_copy_segments(segs)
        copied[0].features.append(FeatureCondition(feature_id=2, magnitude=1))
        assert len(segs[0].features) == 1

    def test_empty_list(self) -> None:
        assert deep_copy_segments([]) == []

    def test_preserves_all_fields(self) -> None:
        seg = Segment(
            segment_idx=3,
            char_start=100,
            char_end=200,
            color_idx=2,
            features=[FeatureCondition(feature_id=7, magnitude=5, enabled=False, description="test")],
            enabled=False,
            is_generation_target=True,
            target_token_count=64,
        )
        copied = deep_copy_segments([seg])
        c = copied[0]
        assert c.segment_idx == 3
        assert c.char_start == 100
        assert c.char_end == 200
        assert c.color_idx == 2
        assert c.enabled is False
        assert c.is_generation_target is True
        assert c.target_token_count == 64
        assert c.features[0].feature_id == 7
        assert c.features[0].magnitude == 5
        assert c.features[0].enabled is False
        assert c.features[0].description == "test"


# ---------------------------------------------------------------------------
# FeatureCondition defaults
# ---------------------------------------------------------------------------


class TestFeatureCondition:
    def test_defaults(self) -> None:
        fc = FeatureCondition(feature_id=10, magnitude=2)
        assert fc.enabled is True
        assert fc.description is None

    def test_explicit_values(self) -> None:
        fc = FeatureCondition(feature_id=5, magnitude=-1, enabled=False, description="hello")
        assert fc.feature_id == 5
        assert fc.magnitude == -1
        assert fc.enabled is False
        assert fc.description == "hello"


# ---------------------------------------------------------------------------
# GenerationSnapshot
# ---------------------------------------------------------------------------


class TestGenerationSnapshot:
    def test_creation(self) -> None:
        seg = Segment(
            segment_idx=0,
            char_start=0,
            char_end=5,
            color_idx=0,
            features=[],
        )
        snap = GenerationSnapshot(
            document_text="hello",
            segments=[seg],
            generated_text=" world",
            generated_token_ids=[100, 200],
            seed=42,
        )
        assert snap.document_text == "hello"
        assert snap.generated_text == " world"
        assert snap.generated_token_ids == [100, 200]
        assert snap.seed == 42
        assert len(snap.segments) == 1

    def test_is_frozen(self) -> None:
        snap = GenerationSnapshot(
            document_text="hi",
            segments=[],
            generated_text="",
            generated_token_ids=[],
            seed=0,
        )
        field_name = "document_text"
        with pytest.raises(AttributeError):
            setattr(snap, field_name, "changed")

    def test_segments_are_deep_copied(self) -> None:
        """Mutating the source segment after snapshot creation must not affect the snapshot."""
        seg = Segment(
            segment_idx=0,
            char_start=0,
            char_end=5,
            color_idx=0,
            features=[FeatureCondition(feature_id=1, magnitude=3)],
        )
        snap = GenerationSnapshot(
            document_text="hello",
            segments=[seg],
            generated_text="",
            generated_token_ids=[],
            seed=0,
        )
        # Mutate the original segment
        seg.features[0].magnitude = 999
        seg.enabled = False
        assert snap.segments[0].features[0].magnitude == 3
        assert snap.segments[0].enabled is True

    def test_token_ids_are_copied(self) -> None:
        """Mutating the source token list after snapshot creation must not affect the snapshot."""
        token_ids = [10, 20, 30]
        snap = GenerationSnapshot(
            document_text="x",
            segments=[],
            generated_text="",
            generated_token_ids=token_ids,
            seed=0,
        )
        token_ids.append(40)
        assert snap.generated_token_ids == [10, 20, 30]

    def test_segments_list_is_copied(self) -> None:
        """Appending to the source list after snapshot creation must not affect the snapshot."""
        segs: list[Segment] = []
        snap = GenerationSnapshot(
            document_text="x",
            segments=segs,
            generated_text="",
            generated_token_ids=[],
            seed=0,
        )
        segs.append(
            Segment(segment_idx=0, char_start=0, char_end=1, color_idx=0, features=[])
        )
        assert len(snap.segments) == 0


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_defaults(self) -> None:
        state = SessionState(
            document_text="test",
            segments=[],
            num_buckets=11,
            has_negatives=True,
        )
        assert state.cursor_position == 0
        assert state.seed == 42
        assert state.global_conditioning is True
        assert state.conditioning_memory == {}
        assert state.generation_history == []
        assert state.generation_redo_stack == []

    def test_no_takeover_method(self) -> None:
        """SessionState must NOT have a takeover method — it lives in segment_ops.py."""
        assert not hasattr(SessionState, "takeover")
        assert not hasattr(SessionState, "compute_takeover")
