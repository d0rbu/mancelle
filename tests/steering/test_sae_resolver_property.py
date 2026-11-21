"""Property tests for sae_resolver — slice 01 Phase 1 audit.

Hypothesis: derandomize=True, max_examples=200 per design spec §7.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
)

_layer = st.integers(min_value=0, max_value=25)


@settings(derandomize=True, max_examples=200)
@given(
    activation_type=st.sampled_from(list(ActivationType)),
    layer=_layer,
    width=st.sampled_from([SAEWidth.W_16K, SAEWidth.W_65K]),
)
def test_neuronpedia_roundtrip_2b(activation_type, layer, width):
    """For valid 2b combos (residual/mlp/attention with 16k/65k widths) the
    round-trip parse(resolve(...)) returns the original tuple."""
    # 16k/65k are universally valid for gemma-2-2b across all activations.
    _, source = resolve_neuronpedia_ids(
        model="gemma-2-2b",
        release=SAERelease.gemmascope,
        activation_type=activation_type,
        layer=layer,
        width=width,
    )
    parsed = parse_neuronpedia_source(source)
    assert parsed == (SAERelease.gemmascope, activation_type, layer, width, None)
