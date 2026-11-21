"""Property-based tests for feature_paths — slice 01 Phase 1 audit.

Pins map invariant: `unsanitize_model_name(sanitize_model_name(x)) == x`
for every non-empty `str` `x`.

Also tests the build/parse round-trip across the full Cartesian product of
canonical inputs.

Hypothesis settings: derandomize=True, max_examples=200 (per design spec §7).
"""

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    parse_feature_dir,
    sanitize_model_name,
    unsanitize_model_name,
)
from steering.sae_resolver import ActivationType, SAEWidth

# Strategy: arbitrary non-empty model-name-like strings (printable ASCII
# excluding path separator newlines). We deliberately include `_`, `/`, `S`
# to stress the escape encoding.
_model_name_chars = st.characters(
    whitelist_categories=("L", "N"),
    whitelist_characters="_/-Sabc",
)


@settings(derandomize=True, max_examples=200)
@given(name=st.text(alphabet=_model_name_chars, min_size=1, max_size=64))
def test_sanitize_round_trip(name):
    """sanitize/unsanitize round-trip is a bijection."""
    assert unsanitize_model_name(sanitize_model_name(name)) == name


# Strategy for canonical extraction inputs
_segment_size = st.integers(min_value=1, max_value=4096)
_layer = st.integers(min_value=0, max_value=30)
_activation_type = st.sampled_from(list(ActivationType))
_pooling = st.sampled_from(list(PoolingMode))
_width = st.sampled_from(list(SAEWidth))
# only "safe" model strings for the path round-trip (no `\0`, no `/`-only,
# no whitespace-only — those produce undecidable parses).
_model_str = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="-/_",
    ),
    min_size=1,
    max_size=32,
).filter(lambda s: s.strip() != "")
_phase = st.sampled_from(list(DatasetPhase))
_ci_flag = st.booleans()


@settings(derandomize=True, max_examples=200)
@given(
    segment_size=_segment_size,
    layer=_layer,
    activation_type=_activation_type,
    pooling=_pooling,
    width=_width,
    model=_model_str,
    phase=_phase,
    ci_flag=_ci_flag,
)
def test_build_parse_roundtrip_property(
    segment_size,
    layer,
    activation_type,
    pooling,
    width,
    model,
    phase,
    ci_flag,
):
    """parse_feature_dir(build_feature_dir(...)) reconstructs exact inputs."""
    root = Path("/tmp/test_feature_root")
    path = build_feature_dir(
        root=root,
        segment_size=segment_size,
        layer=layer,
        activation_type=activation_type,
        pooling=pooling,
        width=width,
        model=model,
        dataset_phase=phase,
        context_independent_features=ci_flag,
    )
    parsed = parse_feature_dir(path)
    assert parsed.segment_size == segment_size
    assert parsed.layer == layer
    assert parsed.activation_type == activation_type
    assert parsed.pooling == pooling
    assert parsed.width == width
    assert parsed.model == model
    assert parsed.dataset_phase == phase
    assert parsed.context_independent_features is ci_flag
