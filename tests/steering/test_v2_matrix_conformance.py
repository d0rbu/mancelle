"""NETWORK-gated: gemma-scope-2 validation-matrix ↔ SAELens registry parity.

For every (model, activation) the production matrix advertises, the set of
`(width, tier, layer)` triples must be *bijective* with the post-injection
SAELens `saes_map` for the corresponding release. This single test catches:

  * a future SAELens-registry change for any v2 release, AND
  * a typo / drift in GEMMASCOPE_2_VALID_COMBOS.

Skipped by default (`-m 'not very_slow'`). Run manually:
    uv run pytest tests/steering/test_v2_matrix_conformance.py -m very_slow -v
"""

import re

import pytest
from sae_lens.loading.pretrained_saes_directory import (
    get_pretrained_saes_directory,
)

# Side-effect import: registers the 8 unregistered v2 PT releases so the
# directory below contains every release our matrix advertises.
import steering  # noqa: F401
from steering.sae_resolver import ActivationType
from steering.sae_resolver_v2_data import (
    GEMMASCOPE_2_MODEL_SIZES,
    GEMMASCOPE_2_VALID_COMBOS,
)

pytestmark = pytest.mark.very_slow

_ACT_SUFFIX: dict[ActivationType, str] = {
    ActivationType.residual: "res",
    ActivationType.mlp: "mlp",
    ActivationType.attention: "att",
}

_KEY_RE = re.compile(
    r"^layer_(?P<layer>\d+)_width_(?P<width>\w+?)_l0_(?P<tier>\w+)$"
)


def _matrix_triples(model: str) -> dict[ActivationType, set[tuple[str, str, int]]]:
    """{activation: {(width_value, tier_value, layer)}} from the matrix."""
    out: dict[ActivationType, set[tuple[str, str, int]]] = {}
    for (act, width), tier_layers in GEMMASCOPE_2_VALID_COMBOS[model].items():
        bucket = out.setdefault(act, set())
        for tier, layers in tier_layers.items():
            for layer in layers:
                bucket.add((width.value, tier.value, layer))
    return out


def _registry_triples(release: str) -> set[tuple[str, str, int]]:
    directory = get_pretrained_saes_directory()
    rec = directory[release]
    triples: set[tuple[str, str, int]] = set()
    for key in rec.saes_map:
        m = _KEY_RE.match(key)
        assert m is not None, f"unexpected saes_map key {key!r} in {release}"
        triples.add((m.group("width"), m.group("tier"), int(m.group("layer"))))
    return triples


@pytest.mark.parametrize("model", sorted(GEMMASCOPE_2_VALID_COMBOS))
def test_v2_matrix_is_bijective_with_registry(model: str):
    size, variant = GEMMASCOPE_2_MODEL_SIZES[model]
    by_act = _matrix_triples(model)
    for activation, expected in by_act.items():
        release = f"gemma-scope-2-{size}-{variant}-{_ACT_SUFFIX[activation]}"
        actual = _registry_triples(release)
        assert actual == expected, (
            f"{release}: matrix vs registry differ.\n"
            f"  matrix-only:   {sorted(expected - actual)[:8]}\n"
            f"  registry-only: {sorted(actual - expected)[:8]}"
        )
