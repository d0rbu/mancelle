"""Hypothesis: sample_id is deterministic + distinguishing under all realistic inputs."""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_bpb.sample_loader import make_sample_id
from steering.eval.axbench_bpb.types import SteeringMode


@given(
    concept_id=st.integers(min_value=-1, max_value=10000),
    content_class=st.sampled_from(["concept_rich", "concept_avoiding"]),
    prefix_text=st.text(min_size=0, max_size=1000),
    output_text=st.text(min_size=0, max_size=1000),
)
def test_sample_id_deterministic(concept_id, content_class, prefix_text, output_text):
    """Same inputs → same sample_id, always."""
    a = make_sample_id(
        concept_id=concept_id, mode=SteeringMode.POSITIVE,
        content_class=content_class,
        prefix_text=prefix_text, output_text=output_text,
    )
    b = make_sample_id(
        concept_id=concept_id, mode=SteeringMode.POSITIVE,
        content_class=content_class,
        prefix_text=prefix_text, output_text=output_text,
    )
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


@given(
    concept_id_1=st.integers(min_value=-1, max_value=100),
    concept_id_2=st.integers(min_value=-1, max_value=100),
)
def test_sample_id_distinguishes_concept_id(concept_id_1, concept_id_2):
    """Different concept_id → different sample_id (with astronomically high prob)."""
    if concept_id_1 == concept_id_2:
        return  # trivially equal
    a = make_sample_id(
        concept_id=concept_id_1, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="x", output_text="y",
    )
    b = make_sample_id(
        concept_id=concept_id_2, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="x", output_text="y",
    )
    assert a != b, f"hash collision: concept_id_1={concept_id_1}, concept_id_2={concept_id_2}"


@given(
    prefix_a=st.text(min_size=1, max_size=100),
    prefix_b=st.text(min_size=1, max_size=100),
)
def test_sample_id_distinguishes_prefix(prefix_a, prefix_b):
    """Different prefix_text → different sample_id."""
    if prefix_a == prefix_b:
        return
    a = make_sample_id(
        concept_id=1, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_a, output_text="y",
    )
    b = make_sample_id(
        concept_id=1, mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=prefix_b, output_text="y",
    )
    assert a != b


@given(
    mode_pair=st.sampled_from([
        (SteeringMode.POSITIVE, SteeringMode.NEGATIVE),
    ]),
)
def test_sample_id_distinguishes_mode(mode_pair):
    """POSITIVE vs NEGATIVE mode produce different sample_ids."""
    mode_a, mode_b = mode_pair
    a = make_sample_id(
        concept_id=1, mode=mode_a, content_class="concept_rich",
        prefix_text="x", output_text="y",
    )
    b = make_sample_id(
        concept_id=1, mode=mode_b, content_class="concept_rich",
        prefix_text="x", output_text="y",
    )
    assert a != b
