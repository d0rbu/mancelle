"""Roundtrip metric for the AxBench upstream multi-method eval.

Given a selected merged steering frame (the ten-method default derived from
``tuple(METHOD_FACTORS)`` or the explicit seven-method NL product; see
``scripts/eval/axbench_upstream/merge_methods.py``), this module re-encodes each
steered generation through the concept's own gemma-scope SAE and checks
whether the concept feature "fires" on the text the steered method produced
(a roundtrip: steer with the concept, then measure whether the concept
re-activates in the output). This isolates whether a method's steered text
is legibly "about" the concept, independent of the judge model used
elsewhere in the eval.

The probe identity is fixed: every concept's ``sae_link`` in the merged
frame must resolve to the same gemma-2-2b layer-20 gemmascope-res-16k SAE
release (only the feature index varies per concept); ``validate_probe_identity``
enforces this and later stages fail loudly on any other release/model.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import math
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch as t
from loguru import logger

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from scripts.eval.axbench_upstream.merge_baseline_project_methods import (
    BASELINE_PROJECT_NL_METHODS as NL_METHODS,
)
from scripts.eval.axbench_upstream.merge_methods import (
    INPUTS_PER_CONCEPT,
    METHOD_FACTORS,
    SUPPORTED_METHOD_FACTORS,
    read_canonical_merged_frame,
    require_fresh_output_dir,
)
from steering.eval.axbench_base.concept_feature_mapping import _batched_dense_sae_max_pool
from steering.eval.axbench_base.extraction import _infer_n_sae_features
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.roundtrip_steering import resolve_probe_device
from steering.extract_features import Backend, build_nnterp_model, load_sae_and_model
from steering.immutable_io import stream_hash_file_descriptor_bound
from steering.sae_resolver import ResolvedSAE, parse_neuronpedia_source, resolve_sae

ROUNDTRIP_BATCH_SIZE = 8  # module constant, not a knob
ROUNDTRIP_SCHEMA_VERSION = 1
ROUNDTRIP_DATA_FILENAME = "roundtrip_data.parquet"
ROUNDTRIP_SUMMARY_FILENAME = "roundtrip_summary.parquet"
ROUNDTRIP_MANIFEST_FILENAME = "roundtrip_manifest.json"
EXPECTED_PROBE_MODEL_ID = "gemma-2-2b"
EXPECTED_PROBE_SOURCE = "20-gemmascope-res-16k"
PROBE_SAE_LAYER = 20
PROBE_TOKENIZER_ID = "google/gemma-2-2b"
METHODS = tuple(METHOD_FACTORS)


def _require_methods(methods: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(methods)
    if selected not in {METHODS, NL_METHODS}:
        raise ValueError(
            "roundtrip methods must be exactly the ten-method default or "
            f"the seven-method NL tuple, got {selected!r}"
        )
    return selected

#: The merged-frame columns the sidecar carries over verbatim, in order: the
#: three key columns plus the per-concept probe identity, so the sidecar joins
#: back to (and is auditable against) the frame it scored on its own.
_COPIED_COLUMNS = ("concept_id", "input_id", "factor", "sae_link", "sae_id")

#: :func:`build_summary` output columns, in order.
_SUMMARY_COLUMNS = (
    "method",
    "factor",
    "n_declared",
    "n_blank",
    "n_activated",
    "activation_rate",
    "concept_macro_sem",
)

#: ``(release, activation_type, layer, width, l0_tier)`` for the roundtrip
#: probe, DERIVED from :data:`EXPECTED_PROBE_SOURCE` at import rather than
#: written out a second time. Deriving it is the guard: the source string has
#: exactly one decoding, so there is no way for a hand-copied release tuple to
#: drift away from the string the concepts are checked against, and the loader
#: in a later stage resolves the SAE from this same decoding.
#: ``test_expected_probe_source_decodes_to_the_probe_release`` pins the value
#: this derivation must produce.
_EXPECTED_PROBE_RELEASE = parse_neuronpedia_source(EXPECTED_PROBE_SOURCE)

#: Both probe loaders are fp32: ``load_sae_and_model`` builds a
#: ``HookedTransformer`` and a SAELens SAE at TransformerLens' default fp32,
#: and :func:`build_nnterp_model` defaults to ``model_dtype=torch.float32``.
#: The roundtrip stage never overrides either, so the manifest records fp32
#: for both; a future dtype knob would have to change this with them.
_PROBE_DTYPE = "float32"

#: Packages whose installed version the manifest records: the two that define
#: the probe's numerics (SAELens' SAE implementation, TransformerLens' model),
#: plus the two they in turn sit on (transformers, torch). A roundtrip number
#: is only reproducible against a known stack, so these are provenance, not
#: configuration.
_MANIFEST_PACKAGES = ("sae-lens", "transformer-lens", "transformers", "torch")

#: The device types the roundtrip probe may score on. CUDA is the accelerated
#: nnterp path and CPU is the deliberate TransformerLens one; every other
#: accelerator (mps, xpu, ...) has no nnterp backend here, so it would score
#: through a third numeric path and produce numbers that are not comparable
#: with any other run's. :func:`load_roundtrip_probe` refuses them.
_SUPPORTED_PROBE_DEVICE_TYPES = ("cuda", "cpu")


@dataclass(frozen=True)
class SaeLinkIdentity:
    """The (model, SAE release, feature index) triple encoded in a Neuronpedia URL."""

    model_id: str
    source: str
    feature_index: int


def parse_sae_link(url: str) -> SaeLinkIdentity:
    """Parse a Neuronpedia ``sae_link`` URL into its identity components.

    Expects exactly ``https://www.neuronpedia.org/<model_id>/<source>/<feature_index>``
    and nothing else: a query string or fragment is refused rather than
    ignored, because a link carrying either was produced by something other
    than the plain concept-identity convention this metric reads, and silently
    dropping it would score a concept against a probe nobody checked.

    The feature index must be DECIMAL: ``str.isdecimal`` (not ``isdigit``)
    rejects the superscripts and other non-decimal digit characters that
    ``int()`` itself refuses, so the check and the conversion below agree on
    exactly one alphabet.

    Raises ``ValueError`` naming the offending URL on any malformed input.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError(f"sae_link must use the https scheme: {url!r}")
    if parsed.netloc != "www.neuronpedia.org":
        raise ValueError(f"sae_link has an unexpected host: {url!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"sae_link must have no query string or fragment: {url!r}")
    segments = [part for part in parsed.path.split("/") if part]
    if len(segments) != 3:
        raise ValueError(
            f"sae_link path must have exactly 3 path segments (model/source/index): {url!r}"
        )
    model_id, source, index_str = segments
    if not index_str.isdecimal():
        raise ValueError(f"sae_link feature index must be a non-negative integer: {url!r}")
    return SaeLinkIdentity(model_id=model_id, source=source, feature_index=int(index_str))


def read_merged_frame(
    input_path: Path,
    *,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Read the merged steering Parquet the roundtrip metric scores.

    Its default ten-method order is derived from ``tuple(METHOD_FACTORS)``;
    the explicit NL product uses :data:`NL_METHODS`.

    The roundtrip stage's name for
    :func:`~scripts.eval.axbench_upstream.merge_methods.read_canonical_merged_frame`
    -- the SAME reader the vendored judge's launcher goes through -- so the
    roundtrip metric provably scores the same rows, in the same order, under
    the same contract as the judge. ``surface="roundtrip"`` only selects the
    filename check's wording.
    """
    return read_canonical_merged_frame(
        input_path,
        surface="roundtrip",
        methods=_require_methods(methods),
    )


def validate_probe_identity(frame: pd.DataFrame) -> dict[int, int]:
    """Check every concept names the one fixed roundtrip probe; return its features.

    The roundtrip metric loads a SINGLE SAE and scores every concept's
    generations with it, so a frame whose concepts do not all name that SAE
    would be scored against the wrong feature space. Per concept (in
    ``concept_id`` order) this checks, in order:

    1. the concept's ``sae_link`` parses as a Neuronpedia URL;
    2. its ``(model_id, source)`` matches the first concept's -- i.e. all
       concepts share one probe identity;
    3. that pair is exactly
       (:data:`EXPECTED_PROBE_MODEL_ID`, :data:`EXPECTED_PROBE_SOURCE`);
    4. the link's feature index equals EVERY ``sae_id`` value for the concept,
       which catches the vendored ``sae_id = 0`` fallback.

    Check 3 pins the source STRING, so which SAE release/layer/width that
    string denotes is not a property of any frame -- it is a property of the
    module constants, guarded instead by deriving
    :data:`_EXPECTED_PROBE_RELEASE` from :data:`EXPECTED_PROBE_SOURCE`.

    Per-concept constancy of ``sae_link``/``sae_id`` is already enforced by
    ``validate_steering_frame``, so checks 1-3 read the concept's first link.

    A frame with zero rows is refused outright rather than answered with an
    empty map: every per-concept check above would pass vacuously, and an empty
    map is indistinguishable downstream from a fully-checked one, so a caller
    that reached here without :func:`read_merged_frame`'s own non-emptiness
    check would score nothing and report success.

    Returns:
        ``{concept_id: feature_index}`` for every concept in the frame.

    Raises:
        ValueError: the frame is empty, or on any violation above, naming the
            offending concept id.
    """
    if frame.empty:
        raise ValueError(
            "merged frame is empty: validate_probe_identity has no concept to check, "
            "and an empty feature map would let the roundtrip stage score nothing and "
            "still report a successful run"
        )
    feature_by_concept: dict[int, int] = {}
    shared: tuple[int, tuple[str, str]] | None = None
    for raw_concept_id in sorted(frame["concept_id"].unique()):
        concept_id = int(raw_concept_id)
        rows = frame.loc[frame["concept_id"].eq(concept_id)]
        link = str(rows["sae_link"].iloc[0])
        try:
            identity = parse_sae_link(link)
        except ValueError as exc:
            raise ValueError(f"concept {concept_id}: {exc}") from exc

        pair = (identity.model_id, identity.source)
        if shared is None:
            shared = (concept_id, pair)
        else:
            first_concept_id, first_pair = shared
            if pair != first_pair:
                raise ValueError(
                    f"concept {concept_id}: merged frame does not name a single probe "
                    f"identity; concept {first_concept_id} names "
                    f"(model_id={first_pair[0]!r}, source={first_pair[1]!r}) but concept "
                    f"{concept_id} names (model_id={pair[0]!r}, source={pair[1]!r})"
                )
        if pair != (EXPECTED_PROBE_MODEL_ID, EXPECTED_PROBE_SOURCE):
            raise ValueError(
                f"concept {concept_id}: sae_link names (model_id={pair[0]!r}, "
                f"source={pair[1]!r}), not the expected roundtrip probe "
                f"(model_id={EXPECTED_PROBE_MODEL_ID!r}, "
                f"source={EXPECTED_PROBE_SOURCE!r}): {link}"
            )

        sae_ids = rows["sae_id"]
        if not sae_ids.eq(identity.feature_index).all():
            concept_sae_ids = sorted({int(value) for value in sae_ids.unique()})
            raise ValueError(
                f"concept {concept_id}: sae_link feature index "
                f"{identity.feature_index} does not match every sae_id value "
                f"{concept_sae_ids} for that concept"
            )
        feature_by_concept[concept_id] = identity.feature_index
    return feature_by_concept


@dataclass(frozen=True)
class _ScoredCell:
    """One DECLARED sidecar cell.

    Off-grid cells have no ``_ScoredCell`` at all -- their absence is what
    :func:`score_frame` turns into the ``NaN``/``pd.NA``/``pd.NA`` triple.
    """

    max_act: float  # NaN for a blank generation: nothing was probed
    activated: bool
    blank: bool


def _score_concept(
    *,
    concept_id: int,
    feature_index: int,
    positions: Sequence[int],
    factors: Sequence[float],
    generations: Mapping[str, Sequence[object]],
    dense_max_pool_fn: Callable[[list[str]], t.Tensor],
    count_tokens_fn: Callable[[str], int],
    n_ctx: int,
    n_sae_features: int,
    methods: tuple[str, ...],
) -> tuple[dict[tuple[str, int], _ScoredCell], list[str]]:
    """Score every declared cell of ONE concept with a single dense SAE pass.

    ``positions`` are the concept's row positions in frame order; ``factors``
    and ``generations`` are whole-frame columns indexed by those positions.

    Walks methods in the explicitly selected order, then rows, collecting each
    declared cell's generation. Blank generations are recorded immediately and
    never probed. The remaining texts are deduplicated in first-occurrence
    order -- a method that produced the same text twice (or two methods that
    agreed) costs one encode, not two -- and the whole batch goes through
    ``dense_max_pool_fn`` exactly once, INCLUDING when it is empty (the
    production primitive returns a ``(0, n_features)`` tensor for an empty
    batch, so an all-blank concept keeps the one-call-per-concept shape rather
    than needing a skip branch).

    The scoring pass' two preconditions are checked at deliberately different
    scopes. The feature index is checked for EVERY concept before the first
    pass runs (in :func:`score_frame`), because a probe/frame mismatch is a
    whole-run error worth catching in the first second. The length guard below
    instead fires per concept, as that concept's texts are collected: it is
    unreachable for real AxBench generations, which are capped far below the
    probe's context window, so there is nothing for a whole-frame hoist to
    catch earlier. Its cost is a second encode of every unique probed text
    (``count_tokens_fn`` re-tokenizes what the pool tokenizes again for the
    forward pass) -- roughly a millisecond per text against a full SAE forward,
    accepted rather than optimized away, since sharing one encode would mean
    reaching into the pool primitive's tokenization.

    Returns:
        ``({(method, row_position): cell}, unique_probed_texts)``.

    Raises:
        ValueError: a text plus the pool's injected BOS would not fit ``n_ctx``
            (the pool would silently front-clamp it, scoring a truncation); the
            returned activation matrix is not exactly
            ``(len(unique_texts), n_sae_features)``, or holds a non-finite or
            negative value.
    """
    cells: dict[tuple[str, int], _ScoredCell] = {}
    unique_texts: list[str] = []
    index_by_text: dict[str, int] = {}
    probed: list[tuple[tuple[str, int], int]] = []

    for method in methods:
        declared_factors = SUPPORTED_METHOD_FACTORS[method]
        column = generations[method]
        for position in positions:
            if factors[position] not in declared_factors:
                continue
            text = str(column[position])
            if not text.strip():
                cells[(method, position)] = _ScoredCell(
                    max_act=math.nan, activated=False, blank=True
                )
                continue
            text_index = index_by_text.get(text)
            if text_index is None:
                n_tokens = count_tokens_fn(text)
                if n_tokens + 1 > n_ctx:
                    raise ValueError(
                        f"concept {concept_id}: a steered generation tokenizes to "
                        f"{n_tokens} tokens, which with the pool's injected BOS exceeds "
                        f"the probe context window (n_tokens + 1 > n_ctx={n_ctx}); "
                        "the dense max-pool would silently front-clamp it and the "
                        f"roundtrip score would be of a truncation: {text[:80]!r}"
                    )
                text_index = len(unique_texts)
                index_by_text[text] = text_index
                unique_texts.append(text)
            probed.append(((method, position), text_index))

    activations = dense_max_pool_fn(unique_texts)
    expected_shape = (len(unique_texts), n_sae_features)
    # Equality subsumes the rank check: no 1-D (or 3-D) tensor can match a
    # 2-tuple, so a flattened or stacked return is rejected here too.
    if tuple(activations.shape) != expected_shape:
        raise ValueError(
            f"concept {concept_id}: dense SAE max-pool returned shape "
            f"{tuple(activations.shape)}, expected {expected_shape} "
            f"(one row per unique probed text, one column per SAE feature)"
        )
    # Finiteness is checked BEFORE non-negativity so that -inf is reported as
    # the numerical failure it is rather than as a sign error (and so a NaN,
    # which compares False against every bound, cannot slip past both checks).
    if not bool(t.isfinite(activations).all()):
        raise ValueError(
            f"concept {concept_id}: dense SAE max-pool returned non-finite "
            "activations (NaN/inf); the roundtrip probe is unusable"
        )
    if bool((activations < 0.0).any()):
        raise ValueError(
            f"concept {concept_id}: dense SAE max-pool returned negative activations; "
            "JumpReLU post-activations are non-negative, so this is the wrong hook "
            "or the wrong tensor"
        )

    feature_acts = activations[:, feature_index].tolist()
    for key, text_index in probed:
        max_act = float(feature_acts[text_index])
        cells[key] = _ScoredCell(max_act=max_act, activated=max_act > 0.0, blank=False)
    return cells, unique_texts


def score_frame(
    frame: pd.DataFrame,
    feature_by_concept: Mapping[int, int],
    *,
    dense_max_pool_fn: Callable[[list[str]], t.Tensor],
    count_tokens_fn: Callable[[str], int],
    n_ctx: int,
    n_sae_features: int,
    methods: tuple[str, ...] = METHODS,
) -> tuple[pd.DataFrame, int]:
    """Score every declared generation in ``frame`` through the concept's feature.

    ``frame`` must be a :func:`read_merged_frame` result and
    ``feature_by_concept`` a :func:`validate_probe_identity` result, so the
    frame contract (columns, declared/null grid, canonical order,
    non-emptiness) and the probe identity are already established and are NOT
    re-checked here.

    Concepts are scored one at a time, in ascending ``concept_id``, with one
    ``dense_max_pool_fn`` call each (see :func:`_score_concept`); the SAE and
    its feature space are the caller's, injected so this function is pure and
    testable without a GPU.

    Every cell of the returned sidecar is one of exactly three kinds, as
    ``(max_act, activated, blank)``:

    * **off-grid** -- the method does not declare this factor:
      ``(NaN, pd.NA, pd.NA)``.
    * **blank** -- a declared generation that is empty once stripped: it is
      NOT probed, and scores ``(NaN, False, True)``. A method that emitted
      nothing did not put the concept back into its output, so this is a
      real miss, not missing data.
    * **probed** -- ``(feature activation, activation > 0.0, False)``.

    ``NaN`` (not ``pd.NA``) is used for both unscored ``max_act`` kinds so the
    column stays plain ``float64`` and mean/quantile aggregation skips them;
    the two boolean columns use ``pd.NA`` so an undeclared cell is
    distinguishable from a declared one that did not fire -- summing
    ``activated`` over a method must never silently count off-grid rows as
    misses.

    Because an unrecorded cell renders as off-grid, a cell this function failed
    to visit would look undeclared instead of raising. The final accounting
    check exists for exactly that: per method, the number of non-null
    ``activated`` cells must equal the declared population
    (``INPUTS_PER_CONCEPT * n_concepts * len(SUPPORTED_METHOD_FACTORS[method])``).

    Returns:
        ``(sidecar, n_unique_texts)`` -- the sidecar has the input's row count
        and order, the columns in :data:`_COPIED_COLUMNS` copied verbatim, and
        three columns per selected method in its declared order (five copied
        plus three per method: 35 columns by default, 26 for the NL product);
        ``n_unique_texts`` is how many texts were actually encoded across all
        concepts (the probe's real workload, for the manifest).

    Raises:
        ValueError: a concept's feature index falls outside the probe SAE's
            width, or any :func:`_score_concept` failure.
        RuntimeError: the accounting check found a method whose scored-cell
            count disagrees with its declared population.
    """
    methods = _require_methods(methods)
    factors = [float(value) for value in frame["factor"]]
    generations = {method: frame[f"{method}_steered_generation"].tolist() for method in methods}
    positions_by_concept: dict[int, list[int]] = {}
    for position, raw_concept_id in enumerate(frame["concept_id"]):
        positions_by_concept.setdefault(int(raw_concept_id), []).append(position)

    # Checked for ALL concepts before the first (expensive) SAE pass: a feature
    # index the loaded SAE cannot hold is a probe/frame mismatch that would
    # otherwise surface hours in, on whichever concept happens to carry it.
    for concept_id in positions_by_concept:
        feature_index = feature_by_concept[concept_id]
        if feature_index >= n_sae_features:
            raise ValueError(
                f"concept {concept_id}: sae feature index {feature_index} is outside "
                f"the loaded probe SAE (n_sae_features={n_sae_features})"
            )

    cells: dict[tuple[str, int], _ScoredCell] = {}
    n_unique_texts = 0
    for concept_id, positions in sorted(positions_by_concept.items()):
        concept_cells, unique_texts = _score_concept(
            concept_id=concept_id,
            feature_index=feature_by_concept[concept_id],
            positions=positions,
            factors=factors,
            generations=generations,
            dense_max_pool_fn=dense_max_pool_fn,
            count_tokens_fn=count_tokens_fn,
            n_ctx=n_ctx,
            n_sae_features=n_sae_features,
            methods=methods,
        )
        cells.update(concept_cells)
        n_unique_texts += len(unique_texts)

    n_rows = len(frame)
    data = frame.loc[:, list(_COPIED_COLUMNS)].reset_index(drop=True)
    for method in methods:
        max_acts = np.full(n_rows, np.nan, dtype=np.float64)
        activated: list[bool | None] = [None] * n_rows
        blank: list[bool | None] = [None] * n_rows
        for position in range(n_rows):
            cell = cells.get((method, position))
            if cell is None:
                continue  # off-grid: this method does not declare this factor
            max_acts[position] = cell.max_act
            activated[position] = cell.activated
            blank[position] = cell.blank
        data[f"{method}_roundtrip_max_act"] = max_acts
        data[f"{method}_roundtrip_activated"] = pd.array(activated, dtype="boolean")
        data[f"{method}_roundtrip_blank"] = pd.array(blank, dtype="boolean")

    n_concepts = len(positions_by_concept)
    for method in methods:
        n_scored = int(data[f"{method}_roundtrip_activated"].notna().sum())
        n_declared = INPUTS_PER_CONCEPT * n_concepts * len(SUPPORTED_METHOD_FACTORS[method])
        if n_scored != n_declared:
            raise RuntimeError(
                f"roundtrip accounting failed for {method}: scored {n_scored} cells, but "
                f"{n_concepts} concept(s) declare {n_declared} "
                f"({INPUTS_PER_CONCEPT} inputs x "
                f"{len(SUPPORTED_METHOD_FACTORS[method])} factors "
                "each). The scored frame is not the declared merged grid."
            )
    return data, n_unique_texts


def build_summary(
    data: pd.DataFrame,
    *,
    max_concepts: int,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Aggregate a :func:`score_frame` sidecar into one row per declared (method, factor).

    ``data`` must be a :func:`score_frame` result (or an equivalent sidecar
    with the same selected-method column contract) covering exactly ``max_concepts``
    concepts; this is NOT re-validated here beyond the per-cell accounting
    check below.

    Rows follow the selected method order, and within each method the ascending
    factor order in :data:`SUPPORTED_METHOD_FACTORS`. The ten-method default
    produces 76 rows; the seven-method NL product produces 62. Off-grid (undeclared)
    (method, factor) pairs get no row at all.

    Per (method, factor):

    * ``n_declared`` is the SIZE of the declared population,
      ``INPUTS_PER_CONCEPT * max_concepts`` -- NOT a count read off ``data``,
      so a sidecar that silently dropped or duplicated rows cannot inflate or
      shrink its own denominator.
    * that population size is cross-checked against how many of this
      (method, factor)'s ``{method}_roundtrip_activated`` cells are non-null;
      a mismatch means ``data`` is not a complete, correctly-declared sidecar
      for this ``max_concepts``, and raises ``RuntimeError`` rather than
      silently summarizing a partial grid.
    * ``n_blank`` / ``n_activated`` count the ``True`` cells of
      ``{method}_roundtrip_blank`` / ``{method}_roundtrip_activated`` (a
      blank generation is still a DECLARED cell, so it counts in
      ``n_declared`` and lowers ``activation_rate`` rather than being
      excluded from it).
    * ``activation_rate = n_activated / n_declared`` -- the pooled rate over
      every declared cell, regardless of concept. There is deliberately NO
      macro (mean-of-per-concept-rates) counterpart column, and that relies on
      TWO checks, not one: the accounting check just above only pins the
      (method, factor) TOTAL -- that ``n_activated + n_blank``-worth of
      non-null cells equals ``INPUTS_PER_CONCEPT * max_concepts`` -- it says
      nothing about how those cells are distributed across concepts. Per-concept
      balance -- every concept contributing exactly ``INPUTS_PER_CONCEPT``
      cells at this (method, factor) -- instead comes from upstream, from
      :func:`validate_steering_frame`'s exact key product check on the input
      frame (every ``(concept_id, input_id, factor)`` combination the grid
      declares is present, so each concept contributes exactly
      ``INPUTS_PER_CONCEPT`` rows per factor by construction). It is the two
      checks TOGETHER -- a correct total here, on a per-concept-balanced input
      there -- that make the mean of the per-concept rates algebraically equal
      to the pooled rate. A macro-rate column could therefore never differ
      from ``activation_rate`` by so much as a rounding step, so it was a
      second spelling of one number rather than a second measurement.
    * ``concept_macro_sem`` is the cross-concept signal that survives that
      collapse: the SPREAD of the per-concept rates (each concept's own mean
      of ``activated`` over its declared cells at this factor) -- their sample
      standard deviation (``ddof=1``) divided by ``sqrt(max_concepts)``. It is
      undefined (``NaN``) when ``max_concepts < 2``, since a sample standard
      deviation needs at least two observations. This is what says whether a
      pooled rate reflects a uniform effect or a couple of dominant concepts.

    Returns:
        The selected-method summary with columns :data:`_SUMMARY_COLUMNS`
        (76 rows by default, 62 for the NL product).

    Raises:
        RuntimeError: a (method, factor)'s non-null ``activated`` count does
            not equal ``INPUTS_PER_CONCEPT * max_concepts``.
    """
    rows: list[dict[str, object]] = []
    n_declared = INPUTS_PER_CONCEPT * max_concepts
    for method in _require_methods(methods):
        for factor in SUPPORTED_METHOD_FACTORS[method]:
            subset = data.loc[data["factor"].eq(factor)]
            activated_col = subset[f"{method}_roundtrip_activated"]
            n_scored = int(activated_col.notna().sum())
            if n_scored != n_declared:
                raise RuntimeError(
                    f"roundtrip summary accounting failed for {method} factor {factor}: "
                    f"{n_scored} scored cells at that factor, but max_concepts={max_concepts} "
                    f"declares {n_declared} ({INPUTS_PER_CONCEPT} inputs x {max_concepts} "
                    "concepts). The sidecar is not a complete declared grid for this max_concepts."
                )

            n_blank = int(subset[f"{method}_roundtrip_blank"].sum())
            n_activated = int(activated_col.sum())
            # The per-concept groupby survives ONLY for the SEM: the mean of
            # these rates is the pooled rate on the balanced grid the check
            # above enforces, but their spread is not recoverable from it.
            per_concept_rate = (
                subset.assign(_roundtrip_activated_rate=activated_col.astype("float64"))
                .groupby("concept_id")["_roundtrip_activated_rate"]
                .mean()
            )
            concept_macro_sem = (
                float(per_concept_rate.std(ddof=1) / math.sqrt(max_concepts))
                if max_concepts >= 2
                else math.nan
            )
            rows.append(
                {
                    "method": method,
                    "factor": float(factor),
                    "n_declared": n_declared,
                    "n_blank": n_blank,
                    "n_activated": n_activated,
                    "activation_rate": n_activated / n_declared,
                    "concept_macro_sem": concept_macro_sem,
                }
            )
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


def _require_scoreable_device(device: t.device) -> None:
    """Refuse every device the roundtrip probe cannot score comparable numbers on.

    Two checks, both pure and both cheap enough to run before anything is
    resolved, loaded or read:

    1. the device TYPE must be one of :data:`_SUPPORTED_PROBE_DEVICE_TYPES`;
    2. a ``cuda`` device must actually be AVAILABLE, since CUDA is the one
       device type where the nnterp backend is required, and a box without a
       working CUDA runtime would otherwise fail deep inside the loader with a
       torch-level message instead of a fixable one.

    Check 2 is not redundant with ``resolve_probe_device``: that helper only
    consults ``cuda.is_available()`` for ``device_arg="auto"``, so an explicit
    ``--device cuda`` on a CPU-only box sails past it.

    Raises:
        RuntimeError: on either violation, naming the fix.
    """
    if device.type not in _SUPPORTED_PROBE_DEVICE_TYPES:
        raise RuntimeError(
            f"{device} is not a supported roundtrip probe device: only "
            f"{list(_SUPPORTED_PROBE_DEVICE_TYPES)} are. A non-CUDA accelerator has no "
            "nnterp backend here, so it would score through TransformerLens on a third "
            "numeric path and its roundtrip numbers would not be comparable with any "
            "other run's; cpu remains supported as the deliberate TransformerLens path."
        )
    if device.type == "cuda" and not t.cuda.is_available():
        raise RuntimeError(
            f"the roundtrip probe was asked for {device}, but torch.cuda.is_available() "
            "is False: there is no CUDA runtime here to load the probe onto. Pass "
            "--device cpu to score through TransformerLens on this box, or run the "
            "roundtrip stage on a CUDA machine."
        )


def load_roundtrip_probe(device: t.device) -> tuple[Any, Any, ResolvedSAE, str]:
    """Load the one fixed roundtrip probe: gemma-2-2b and its layer-20 SAE.

    The probe identity is not a parameter -- it is the decoding of
    :data:`EXPECTED_PROBE_SOURCE` held in :data:`_EXPECTED_PROBE_RELEASE`,
    unpacked and handed to ``resolve_sae`` here rather than re-typed as enum
    literals. That is what makes the identity ``validate_probe_identity``
    checks every concept's ``sae_link`` against and the identity actually
    loaded provably the same tuple: there is no second spelling of it to
    drift. :data:`PROBE_SAE_LAYER` (the layer the forward pass reads) is
    cross-checked against the derived layer for the same reason.

    The device is checked FIRST, before anything is resolved or loaded, through
    :func:`_require_scoreable_device` -- the same pure helper
    :func:`execute_roundtrip` runs in its own preflight. Repeating it here is
    deliberate rather than redundant: it costs nothing, and it keeps this
    loader safe for a caller that reaches it directly (the GPU suite does).

    On CUDA the nnterp backend is REQUIRED. It is attached as
    ``model.nnterp_model``, the attribute ``steering.extract_features``'
    activation getter dispatches on, and a failed build raises instead of
    falling back to TransformerLens -- the repo's accelerated-backend
    fail-fast convention, since that fallback is far slower and numerically
    divergent, which would make these scores incomparable to every other
    run's. On CPU there is no nnterp to build: TransformerLens is the intended
    path there and is reported as such.

    Returns:
        ``(model, sae, resolved, backend)`` -- ``backend`` is the
        ``Backend`` value that produced the activations.

    Raises:
        RuntimeError: :func:`_require_scoreable_device` refuses ``device``, the
            derived layer disagrees with :data:`PROBE_SAE_LAYER`, or nnterp
            could not be built on CUDA.
    """
    _require_scoreable_device(device)
    release, activation_type, layer, width, l0 = _EXPECTED_PROBE_RELEASE
    if layer != PROBE_SAE_LAYER:
        raise RuntimeError(
            f"roundtrip probe constants disagree: {EXPECTED_PROBE_SOURCE!r} decodes to "
            f"layer {layer}, but PROBE_SAE_LAYER is {PROBE_SAE_LAYER}. The forward pass "
            "reads PROBE_SAE_LAYER while the SAE is resolved from the decoded layer, so "
            "this would score one layer's activations through another layer's SAE."
        )
    resolved = resolve_sae(
        model=EXPECTED_PROBE_MODEL_ID,
        release=release,
        activation_type=activation_type,
        layer=layer,
        width=width,
        l0=l0,
    )
    model, sae, _tokenizer = load_sae_and_model(
        model_name=EXPECTED_PROBE_MODEL_ID,
        layer=PROBE_SAE_LAYER,
        device=str(device),
        resolved=resolved,
    )
    if device.type != "cuda":
        return model, sae, resolved, Backend.transformerlens.value

    nnterp_model = build_nnterp_model(model, device=str(device))
    if nnterp_model is None:
        raise RuntimeError(
            f"build_nnterp_model returned None for the roundtrip probe on {device}: the "
            "nnterp accelerated backend is required on CUDA. Refusing to fall back to "
            "TransformerLens silently (far slower and numerically divergent, so the "
            "roundtrip activations would not be comparable with other runs); fix the "
            "nnterp backend, or score on CPU to use TransformerLens intentionally."
        )
    model.nnterp_model = nnterp_model
    return model, sae, resolved, Backend.nnterp.value


def sae_weights_sha256(sae: Any) -> str:
    """Fingerprint the loaded probe SAE's weights for the manifest.

    Hashes ``sae.state_dict()`` in sorted-key order -- each entry's name bytes
    then its value bytes -- so the digest is a property of the weights
    themselves and not of the order PyTorch happened to register them in.
    Values are normalized to contiguous CPU float32 first, so the digest
    recorded on a GPU run compares equal to the same weights hashed on CPU.

    This is a provenance fingerprint, not a serialization: it covers parameter
    names and values, not shapes or dtypes. Its job is to make "was this
    scored with the same SAE?" answerable across runs, which a value+name
    digest settles; two different SAEs sharing every weight value under every
    weight name are the same SAE for that purpose.
    """
    digest = hashlib.sha256()
    state_dict = sae.state_dict()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].detach().to("cpu", t.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def resolve_manifest_versions() -> dict[str, str]:
    """Read the installed version of every :data:`_MANIFEST_PACKAGES` entry.

    Split out of :func:`build_manifest` so :func:`execute_roundtrip` can resolve
    the versions BEFORE the scoring pass: a package that is missing from the
    environment must fail the run in the first second, not after hours of SAE
    forward passes whose numbers would then have nowhere to be recorded.

    Raises:
        importlib.metadata.PackageNotFoundError: a recorded package is not
            installed.
    """
    return {name: importlib.metadata.version(name) for name in _MANIFEST_PACKAGES}


def build_manifest(
    *,
    input_path: Path,
    input_sha256: str,
    max_concepts: int,
    data: pd.DataFrame,
    n_unique_texts: int,
    resolved: ResolvedSAE,
    backend: str,
    device: t.device,
    n_ctx: int,
    n_sae_features: int,
    sae_sha256: str,
    versions: Mapping[str, str],
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    """Describe one roundtrip run completely enough to reproduce or refute it.

    The manifest is the sidecar's provenance record: which bytes were scored
    (``input.sha256``), which weights scored them (``probe``, including
    ``sae_weights_sha256``), under which stack (``versions``), with which
    fixed policy (``scoring``), and what came out (``accounting``). Nothing
    here is a knob -- every value is either read off the run or a module
    constant -- so two manifests that differ anywhere identify exactly what
    changed between two sets of numbers.

    The probe's layer/activation/width/l0 come from
    :data:`_EXPECTED_PROBE_RELEASE`, the same derivation the loader resolves
    from, while ``release``/``saelens_sae_id``/``hook_name`` come from the
    ``ResolvedSAE`` the run actually loaded -- so the record reflects the
    load, not a re-derivation of it. ``saelens_sae_id`` carries its prefix
    because SAELens' id is a STRING path into a release, while the sidecar's
    ``sae_id`` COLUMN is the int64 Neuronpedia feature index; naming both
    ``sae_id`` in one artifact set is exactly the confusion this avoids.

    ``versions`` is recorded verbatim -- :func:`execute_roundtrip` resolves it
    through :func:`resolve_manifest_versions` before scoring starts, so this
    function never touches the environment.

    ``accounting`` is READ OFF ``data`` rather than recomputed from the
    declared grid. It has two keys of uniform shape: ``per_method`` maps each
    method to its ``{n_declared, n_blank, n_scored}`` object -- ``n_declared``
    counts its non-null ``activated`` cells, ``n_blank`` its blank
    generations, and ``n_scored`` is the difference (the cells that reached the
    SAE) -- and ``n_unique_texts_per_concept_sum`` is the run-level count
    below. Keeping the per-method objects under their own key means a reader
    can iterate the accounting without knowing which keys are method names and
    which are run-level scalars. It therefore reports what was actually
    written; :func:`score_frame`'s own accounting check is what ties those
    counts to the declared population.

    ``n_unique_texts_per_concept_sum`` is named for exactly what
    :func:`score_frame` returns: the sum over concepts of each concept's
    unique probed texts. Deduplication is per concept (each concept is scored
    through its own feature in its own batch), so one text appearing under two
    concepts is encoded -- and counted -- twice. The number is the probe's
    encode workload, not a global distinct-text count, and the key says so.
    """
    _release, activation_type, layer, width, l0 = _EXPECTED_PROBE_RELEASE
    per_method: dict[str, dict[str, int]] = {}
    methods = _require_methods(methods)
    for method in methods:
        n_declared = int(data[f"{method}_roundtrip_activated"].notna().sum())
        n_blank = int(data[f"{method}_roundtrip_blank"].sum())
        per_method[method] = {
            "n_declared": n_declared,
            "n_blank": n_blank,
            "n_scored": n_declared - n_blank,
        }
    return {
        "schema_version": ROUNDTRIP_SCHEMA_VERSION,
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "n_rows": len(data),
            "max_concepts": max_concepts,
        },
        "probe": {
            "model_name": EXPECTED_PROBE_MODEL_ID,
            # The HuggingFace repo the probe's weights and tokenizer both come
            # from -- i.e. what the TransformerLens alias in ``model_name``
            # actually loads -- which is exactly :data:`PROBE_TOKENIZER_ID`.
            "hf_model_id": PROBE_TOKENIZER_ID,
            "release": resolved.release,
            # ``saelens_*``-prefixed because this is SAELens' STRING id for the
            # loaded SAE ("layer_20/width_16k/canonical"); the sidecar's own
            # ``sae_id`` column is the int64 Neuronpedia FEATURE INDEX, a
            # completely different quantity that a bare ``sae_id`` key here
            # would read as.
            "saelens_sae_id": resolved.sae_id,
            "hook_name": resolved.hook_name,
            "layer": layer,
            "activation": activation_type.value,
            "width": width.value,
            "n_sae_features": n_sae_features,
            "l0": l0,
            "model_dtype": _PROBE_DTYPE,
            "sae_dtype": _PROBE_DTYPE,
            "backend": backend,
            "device": str(device),
            "sae_weights_sha256": sae_sha256,
            "n_ctx": n_ctx,
        },
        "versions": dict(versions),
        "scoring": {
            **({"methods": list(methods)} if methods != METHODS else {}),
            "batch_size": ROUNDTRIP_BATCH_SIZE,
            "activation_threshold": "post_act > 0",
            "blank_policy": "count_as_not_activated",
        },
        "accounting": {
            "per_method": per_method,
            "n_unique_texts_per_concept_sum": n_unique_texts,
        },
    }


def _require_disjoint_input_and_output(input_path: Path, output_dir: Path) -> None:
    """Keep the sidecar out of (and off of) the merged run's own directory.

    The merged input's directory is what the vendored evaluator is pointed at
    (``--overwrite_inference_dump_dir``), and the roundtrip sidecar's data
    Parquet is a DIFFERENT frame than the ``steering_data.parquet`` that lives
    there. Writing inside it would put a second Parquet in a directory the
    evaluator scans; writing a parent of it would make the freshness check
    above reject the merged run's own files as debris (or, worse, accept the
    directory and later have the roundtrip's artifacts read as merged input).

    ``Path.is_relative_to`` is reflexive, so the equal-paths case is caught by
    the first clause.
    """
    input_dir = input_path.parent
    if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise ValueError(
            "roundtrip output directory and the merged input's directory must be "
            f"path-disjoint: output_dir={output_dir}, input directory={input_dir}"
        )


def _count_probe_tokens(tokenizer: Any, text: str) -> int:
    """Count ``text``'s CONTENT tokens under the probe's tokenizer.

    ``add_special_tokens=False`` matches how
    :func:`_batched_dense_sae_max_pool` tokenizes -- it injects exactly one BOS
    of its own afterwards -- so :func:`score_frame`'s ``n_tokens + 1 > n_ctx``
    budget counts that BOS exactly once. Letting the tokenizer add its own
    would count it twice and reject texts that actually fit.

    That injected BOS is also why :func:`score_frame`'s budget is
    ``n_tokens + 1 > n_ctx`` rather than ``n_tokens > n_ctx``: the pool
    front-clamps its own BOS-prepended input to ``n_ctx`` SILENTLY, so a text
    of exactly ``n_ctx`` content tokens would be scored as a truncation instead
    of refused.

    The ``len()`` wrapping is load-bearing, not incidental: ``score_frame``
    takes a ``str -> int`` callable, and this is where a tokenizer's token-id
    list becomes the count that callable promises.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def execute_roundtrip(
    *,
    input_path: Path,
    output_dir: Path,
    device_arg: str = "auto",
    include_nl_steering: bool = False,
) -> None:
    """Score one merged steering Parquet end to end and write the three artifacts.

    The whole stage in order: resolve and check the scoring device, refuse a
    non-fresh or overlapping output directory, resolve the manifest's package
    versions (every check that can fail in seconds happens before the pass that
    takes hours), read and validate the merged frame and its probe identity,
    fingerprint the exact bytes scored, load the one fixed probe, score every
    declared generation through each concept's own feature, aggregate, and
    write ``roundtrip_data.parquet``, ``roundtrip_summary.parquet`` and
    ``roundtrip_manifest.json`` -- the manifest LAST, so its presence is the
    completion marker for the run.

    Nothing is written before scoring succeeds: the output directory is not
    even created until every value is in hand (the atomic writers create it on
    the first artifact), so a failed run leaves either no directory at all or
    (only if a write itself fails) an incomplete one without a manifest.

    Args:
        input_path: the merged ``steering_data.parquet`` in the selected
            default or explicit NL method order.
        output_dir: absent, or an existing empty directory, disjoint from the
            input's own directory.
        device_arg: ``"auto"`` or an explicit torch device string, resolved by
            ``resolve_probe_device``.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if type(include_nl_steering) is not bool:
        raise TypeError("include_nl_steering must be a boolean")
    methods = NL_METHODS if include_nl_steering else METHODS
    # The device is settled and checked before ANY other work: an unscoreable
    # or unavailable device is the one preflight failure that would otherwise
    # surface only inside the loader, after the frame has been read and hashed.
    device = resolve_probe_device(device_arg)
    _require_scoreable_device(device)
    _require_disjoint_input_and_output(input_path, output_dir)
    require_fresh_output_dir(output_dir, what="roundtrip")
    # Resolved up front, with the other preflight checks: a package missing
    # from the environment is a manifest the run could never write, and that
    # must surface now rather than after a multi-hour scoring pass.
    versions = resolve_manifest_versions()

    frame = read_merged_frame(input_path, methods=methods)
    feature_by_concept = validate_probe_identity(frame)
    input_sha256 = stream_hash_file_descriptor_bound(input_path).sha256
    max_concepts = len(feature_by_concept)

    logger.info("Roundtrip scoring {} concept(s) from {} on {}", max_concepts, input_path, device)
    model, sae, resolved, backend = load_roundtrip_probe(device)
    sae_sha256 = sae_weights_sha256(sae)
    # Plain Python ints: these reach the manifest, and ``atomic_json_dump``
    # cannot serialize a numpy (or torch) scalar.
    n_sae_features = int(_infer_n_sae_features(sae))
    n_ctx = int(model.cfg.n_ctx)
    tokenizer = get_extraction_tokenizer(PROBE_TOKENIZER_ID)

    data, n_unique_texts = score_frame(
        frame,
        feature_by_concept,
        # Bound HERE, at call time, so the module global is what gets partially
        # applied -- which is also the seam the driver suite monkeypatches.
        # ``texts`` stays the sole positional argument ``score_frame`` supplies;
        # everything else is a module constant, not a knob.
        dense_max_pool_fn=functools.partial(
            _batched_dense_sae_max_pool,
            base_model=model,
            base_sae=sae,
            sae_layer=PROBE_SAE_LAYER,
            tokenizer_id=PROBE_TOKENIZER_ID,
            batch_size=ROUNDTRIP_BATCH_SIZE,
        ),
        count_tokens_fn=functools.partial(_count_probe_tokens, tokenizer),
        n_ctx=n_ctx,
        n_sae_features=n_sae_features,
        methods=methods,
    )
    summary = build_summary(data, max_concepts=max_concepts, methods=methods)
    manifest = build_manifest(
        input_path=input_path,
        input_sha256=input_sha256,
        max_concepts=max_concepts,
        data=data,
        n_unique_texts=n_unique_texts,
        resolved=resolved,
        backend=backend,
        device=device,
        n_ctx=n_ctx,
        n_sae_features=n_sae_features,
        sae_sha256=sae_sha256,
        versions=versions,
        methods=methods,
    )

    # No mkdir: every ``nanochat.atomic_io`` writer does
    # ``final.parent.mkdir(parents=True, exist_ok=True)`` itself, so the first
    # artifact below creates the directory -- and only once there is one to
    # write, which is the "no directory at all on failure" property.
    atomic_write_parquet(
        pa.Table.from_pandas(data, preserve_index=False),
        output_dir / ROUNDTRIP_DATA_FILENAME,
    )
    atomic_write_parquet(
        pa.Table.from_pandas(summary, preserve_index=False),
        output_dir / ROUNDTRIP_SUMMARY_FILENAME,
    )
    atomic_json_dump(manifest, output_dir / ROUNDTRIP_MANIFEST_FILENAME)
    logger.info("Roundtrip artifacts written to {}", output_dir)
