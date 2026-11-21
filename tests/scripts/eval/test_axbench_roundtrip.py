"""Unit tests for the axbench_upstream roundtrip metric logic."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch as t

from scripts.eval.axbench_upstream.merge_methods import (
    _SOURCE_METHODS,
    FULL_FACTORS,
    INPUT_COLUMNS,
    METHOD_FACTORS,
    FrameSource,
    method_columns,
)
from steering.eval.axbench_upstream import roundtrip
from steering.eval.axbench_upstream.roundtrip import (
    _EXPECTED_PROBE_RELEASE,
    EXPECTED_PROBE_MODEL_ID,
    EXPECTED_PROBE_SOURCE,
    METHODS,
    NL_METHODS,
    PROBE_SAE_LAYER,
    PROBE_TOKENIZER_ID,
    ROUNDTRIP_BATCH_SIZE,
    ROUNDTRIP_SCHEMA_VERSION,
    SaeLinkIdentity,
    build_manifest,
    build_summary,
    load_roundtrip_probe,
    parse_sae_link,
    read_merged_frame,
    resolve_manifest_versions,
    sae_weights_sha256,
    score_frame,
    validate_probe_identity,
)
from steering.sae_resolver import (
    ActivationType,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
    parse_neuronpedia_source,
    resolve_sae,
)
from tests.scripts.eval.test_axbench_evaluation_launcher import _merged_frame

VALID_LINK = "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/16038"
EXPECTED_PROBE_RELEASE = (
    SAERelease.gemmascope,
    ActivationType.residual,
    PROBE_SAE_LAYER,
    SAEWidth.W_16K,
    None,
)

#: Scoring-test probe geometry: gemma-scope 16k width, gemma-2-2b context, and
#: the one feature index :class:`_StubDense` writes a non-zero activation into.
N_SAE_FEATURES = 16384
N_CTX = 8192
PROBE_FEATURE = 7
DEFAULT_FEATURE_BY_CONCEPT: Mapping[int, int] = MappingProxyType({0: PROBE_FEATURE})

#: Six full-grid methods and four factor-1.0 methods declare 760 generations/concept.
DECLARED_PER_CONCEPT = 760

#: The sidecar schema, spelled out rather than derived from ``METHODS`` so that
#: reordering the production method tuple fails this pin instead of following it.
EXPECTED_SIDECAR_COLUMNS = [
    "concept_id",
    "input_id",
    "factor",
    "sae_link",
    "sae_id",
    "DiffMean_roundtrip_max_act",
    "DiffMean_roundtrip_activated",
    "DiffMean_roundtrip_blank",
    "LinearProbe_roundtrip_max_act",
    "LinearProbe_roundtrip_activated",
    "LinearProbe_roundtrip_blank",
    "PromptSteering_roundtrip_max_act",
    "PromptSteering_roundtrip_activated",
    "PromptSteering_roundtrip_blank",
    "LoRA_roundtrip_max_act",
    "LoRA_roundtrip_activated",
    "LoRA_roundtrip_blank",
    "EncDecDiffMean_roundtrip_max_act",
    "EncDecDiffMean_roundtrip_activated",
    "EncDecDiffMean_roundtrip_blank",
    "EncDecLinearProbe_roundtrip_max_act",
    "EncDecLinearProbe_roundtrip_activated",
    "EncDecLinearProbe_roundtrip_blank",
    "EncDecPromptSteering_roundtrip_max_act",
    "EncDecPromptSteering_roundtrip_activated",
    "EncDecPromptSteering_roundtrip_blank",
    "EncDecLoRA_roundtrip_max_act",
    "EncDecLoRA_roundtrip_activated",
    "EncDecLoRA_roundtrip_blank",
    "SelfConditioning_roundtrip_max_act",
    "SelfConditioning_roundtrip_activated",
    "SelfConditioning_roundtrip_blank",
    "SelfConditioningMasked_roundtrip_max_act",
    "SelfConditioningMasked_roundtrip_activated",
    "SelfConditioningMasked_roundtrip_blank",
]
COPIED_COLUMNS = ["concept_id", "input_id", "factor", "sae_link", "sae_id"]

#: The summary schema, spelled out rather than derived so that reordering the
#: production columns fails this pin instead of following it. There is NO
#: ``concept_macro_rate``: on the balanced grid ``build_summary`` enforces, the
#: mean of the per-concept rates is algebraically ``activation_rate``, so the
#: column was a second spelling of one number rather than a second measurement
#: (``test_build_summary_pooled_rate_is_the_macro_rate_on_the_balanced_grid``
#: is what holds that identity, so the removal stays justified).
EXPECTED_SUMMARY_COLUMNS = [
    "method",
    "factor",
    "n_declared",
    "n_blank",
    "n_activated",
    "activation_rate",
    "concept_macro_sem",
]

#: 76 rows = six full grids of 12 factors plus four single-factor grids, method-major in
#: ``METHODS`` order, factors ascending -- spelled out from ``FULL_FACTORS``
#: rather than from ``METHOD_FACTORS`` so that reordering the production method
#: tuple, or its per-method factor grids, fails this pin instead of following it.
EXPECTED_SUMMARY_METHOD_FACTOR_ORDER = [
    *(("DiffMean", f) for f in FULL_FACTORS),
    *(("LinearProbe", f) for f in FULL_FACTORS),
    ("PromptSteering", 1.0),
    ("LoRA", 1.0),
    *(("EncDecDiffMean", f) for f in FULL_FACTORS),
    *(("EncDecLinearProbe", f) for f in FULL_FACTORS),
    ("EncDecPromptSteering", 1.0),
    ("EncDecLoRA", 1.0),
    *(("SelfConditioning", f) for f in FULL_FACTORS),
    *(("SelfConditioningMasked", f) for f in FULL_FACTORS),
]


#: The ``ResolvedSAE`` the fixed probe identity resolves to, spelled out from
#: the live SAELens registry (verified) rather than recomputed from the module
#: constants, so a retargeted probe fails these pins instead of following them.
PROBE_RESOLVED = ResolvedSAE(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_20/width_16k/canonical",
    hook_name="blocks.20.hook_resid_post",
)

#: Manifest inputs that are opaque pass-throughs: build_manifest records them,
#: it does not recompute or validate them.
MANIFEST_INPUT_PATH = Path("/runs/axbench/merged/steering_data.parquet")
MANIFEST_INPUT_SHA = "a" * 64
MANIFEST_SAE_SHA = "b" * 64

EXPECTED_MANIFEST_KEYS = ["schema_version", "input", "probe", "versions", "scoring", "accounting"]
EXPECTED_MANIFEST_INPUT_KEYS = ["path", "sha256", "n_rows", "max_concepts"]
EXPECTED_MANIFEST_PROBE_KEYS = [
    "model_name",
    "hf_model_id",
    "release",
    # ``saelens_sae_id``, not ``sae_id``: this is SAELens' STRING id for the
    # loaded SAE ("layer_20/width_16k/canonical"), which would otherwise read
    # as the sidecar's int64 Neuronpedia ``sae_id`` feature-index column.
    "saelens_sae_id",
    "hook_name",
    "layer",
    "activation",
    "width",
    "n_sae_features",
    "l0",
    "model_dtype",
    "sae_dtype",
    "backend",
    "device",
    "sae_weights_sha256",
    "n_ctx",
]
EXPECTED_MANIFEST_VERSION_KEYS = ["sae-lens", "transformer-lens", "transformers", "torch"]
EXPECTED_MANIFEST_SCORING_KEYS = ["batch_size", "activation_threshold", "blank_policy"]


class _StubDense:
    def __init__(self, act_by_text: dict[str, float], n_features: int) -> None:
        self.act_by_text = act_by_text
        self.n_features = n_features
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> t.Tensor:
        self.calls.append(list(texts))
        out = t.zeros((len(texts), self.n_features), dtype=t.float32)
        for i, text in enumerate(texts):
            out[i, :] = 0.0
            out[i, PROBE_FEATURE] = self.act_by_text[text]
        return out


class _EmptyBatchDense:
    """Dense stub for a concept with nothing to probe: a ``(0, n_features)`` matrix.

    This is what the production primitive returns for an empty batch, so a
    concept whose every declared generation is blank must still make exactly
    one call -- with an empty text list -- and get a zero-row matrix back.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> t.Tensor:
        self.calls.append(list(texts))
        return t.zeros((0, N_SAE_FEATURES), dtype=t.float32)


def _count_words(text: str) -> int:
    return len(text.split())


def _roundtrip_frame(
    *, max_concepts: int = 1, feature_ids: tuple[int, ...] | None = None
) -> pd.DataFrame:
    frame = _merged_frame(max_concepts=max_concepts)
    for concept_id in range(max_concepts):
        fid = feature_ids[concept_id] if feature_ids is not None else 16038 + concept_id
        mask = frame["concept_id"].eq(concept_id)
        frame.loc[mask, "sae_link"] = (
            f"https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/{fid}"
        )
        frame.loc[mask, "sae_id"] = fid
    frame["sae_id"] = frame["sae_id"].astype("int64")
    return frame


def _nl_roundtrip_frame(
    *, max_concepts: int = 1, feature_ids: tuple[int, ...] | None = None
) -> pd.DataFrame:
    """Return the explicit seven-method frame scored by the NL roundtrip route."""
    frame = _roundtrip_frame(max_concepts=max_concepts, feature_ids=feature_ids)
    encdec_methods = (
        "EncDecDiffMean",
        "EncDecLinearProbe",
        "EncDecPromptSteering",
        "EncDecLoRA",
    )
    frame = frame.drop(columns=list(method_columns(encdec_methods)))
    frame["NLSteering_steered_generation"] = frame["factor"].map(
        lambda factor: f"NL retrieval generated a healthy answer at {factor}"
    )
    frame["NLSteering_strength"] = frame["factor"].astype("float64")
    return frame.loc[:, [*INPUT_COLUMNS, *method_columns(NL_METHODS)]]


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _relink(
    frame: pd.DataFrame,
    concept_id: int,
    *,
    model_id: str,
    source: str,
    feature_index: int,
) -> None:
    """Rewrite one concept's ``sae_link`` without touching its ``sae_id``."""
    frame.loc[frame["concept_id"].eq(concept_id), "sae_link"] = (
        f"https://www.neuronpedia.org/{model_id}/{source}/{feature_index}"
    )


def _distinct_acts(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, float]:
    """Give every declared, non-blank generation its own positive activation.

    Values are multiples of ``0.25`` so they survive the stub's float32 tensor
    exactly and a cell scored from the wrong text cannot coincide with the
    right one.
    """
    acts: dict[str, float] = {}
    for method in methods:
        column = frame[f"{method}_steered_generation"]
        for factor, text in zip(frame["factor"], column, strict=True):
            key = str(text)
            if (
                float(factor) in roundtrip.SUPPORTED_METHOD_FACTORS[method]
                and key.strip()
                and key not in acts
            ):
                acts[key] = 0.5 + 0.25 * len(acts)
    return acts


def _blank_every_declared_generation(frame: pd.DataFrame) -> pd.DataFrame:
    """Whitespace-fill every DECLARED generation, leaving off-grid cells null."""
    blanked = frame.copy()
    for method in METHODS:
        column = blanked.columns.get_loc(f"{method}_steered_generation")
        declared = blanked["factor"].isin(METHOD_FACTORS[method]).to_numpy()
        for position in np.flatnonzero(declared):
            blanked.iloc[int(position), column] = "  \n "
    return blanked


def _set_generation(frame: pd.DataFrame, pos: int, method: str, text: str) -> None:
    """Overwrite one declared generation cell by ROW POSITION (not label)."""
    frame.iloc[pos, frame.columns.get_loc(f"{method}_steered_generation")] = text


def _position_of(frame: pd.DataFrame, *, concept_id: int, input_id: int, factor: float) -> int:
    """Row position (for ``.iloc``) of the one row matching this key, by content.

    Avoids hard-coding the canonical row order's arithmetic in tests: callers
    that want to target a specific ``(concept_id, input_id, factor)`` cell
    look it up instead of computing its offset by hand.
    """
    mask = (
        frame["concept_id"].eq(concept_id)
        & frame["input_id"].eq(input_id)
        & frame["factor"].eq(factor)
    )
    positions = np.flatnonzero(mask.to_numpy())
    assert len(positions) == 1, (
        f"expected exactly one row for concept={concept_id} input={input_id} "
        f"factor={factor}, found {len(positions)}"
    )
    return int(positions[0])


def _two_concept_diffmean_activation_fixture() -> tuple[pd.DataFrame, int]:
    """A 2-concept sidecar with a hand-chosen DiffMean-factor-1.0 activation split.

    Concept 0 activates on exactly 3 of its 10 DiffMean factor-1.0
    generations; concept 1 activates on exactly 7 of its 10. Every other
    declared, non-blank cell gets a distinct positive activation (via
    :func:`_distinct_acts`), so every other row is unambiguously activated
    and non-blank -- this isolates the summary's rate/SEM computation to the
    one (method, factor) row under test.
    """
    max_concepts = 2
    frame = _roundtrip_frame(max_concepts=max_concepts, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    acts = _distinct_acts(frame)
    active_inputs = {0: {0, 1, 2}, 1: {0, 1, 2, 3, 4, 5, 6}}
    for concept_id, active in active_inputs.items():
        for input_id in range(10):
            if input_id in active:
                continue
            text = f"DiffMean healthy generated answer {concept_id}/{input_id}/1.0"
            acts[text] = 0.0
    stub = _StubDense(acts, N_SAE_FEATURES)
    data, _ = _score(frame, stub, feature_by_concept={0: PROBE_FEATURE, 1: PROBE_FEATURE})
    return data, max_concepts


def _score(
    frame: pd.DataFrame,
    dense_max_pool_fn: Callable[[list[str]], t.Tensor],
    *,
    feature_by_concept: Mapping[int, int] = DEFAULT_FEATURE_BY_CONCEPT,
    count_tokens_fn: Callable[[str], int] = _count_words,
    n_ctx: int = N_CTX,
    n_sae_features: int = N_SAE_FEATURES,
) -> tuple[pd.DataFrame, int]:
    return score_frame(
        frame,
        feature_by_concept,
        dense_max_pool_fn=dense_max_pool_fn,
        count_tokens_fn=count_tokens_fn,
        n_ctx=n_ctx,
        n_sae_features=n_sae_features,
    )


def _expected_cells(
    frame: pd.DataFrame, acts: Mapping[str, float]
) -> dict[str, tuple[list[float], list[bool | None], list[bool | None]]]:
    """Build the expected ``(max_act, activated, blank)`` columns per method.

    Written against the spec (declared grid -> blank convention -> activation)
    and driven by the fixture, so it is an oracle for the whole sidecar; the
    focused tests below pin each rule on its own.
    """
    expected: dict[str, tuple[list[float], list[bool | None], list[bool | None]]] = {}
    for method in METHODS:
        max_acts: list[float] = []
        activated: list[bool | None] = []
        blank: list[bool | None] = []
        column = frame[f"{method}_steered_generation"]
        for factor, text in zip(frame["factor"], column, strict=True):
            key = str(text)
            if float(factor) not in METHOD_FACTORS[method]:
                max_acts.append(math.nan)
                activated.append(None)
                blank.append(None)
            elif not key.strip():
                max_acts.append(math.nan)
                activated.append(False)
                blank.append(True)
            else:
                max_acts.append(acts[key])
                activated.append(acts[key] > 0.0)
                blank.append(False)
        expected[method] = (max_acts, activated, blank)
    return expected


def _assert_sidecar_matches(
    data: pd.DataFrame, frame: pd.DataFrame, acts: Mapping[str, float]
) -> None:
    """Assert the full sidecar (schema, dtypes, every cell) against the oracle."""
    assert list(data.columns) == EXPECTED_SIDECAR_COLUMNS
    assert len(data) == len(frame)
    pd.testing.assert_frame_equal(
        data.loc[:, COPIED_COLUMNS], frame.loc[:, COPIED_COLUMNS].reset_index(drop=True)
    )
    for method, (max_acts, activated, blank) in _expected_cells(frame, acts).items():
        pd.testing.assert_series_equal(
            data[f"{method}_roundtrip_max_act"],
            pd.Series(np.asarray(max_acts, dtype=np.float64), name=f"{method}_roundtrip_max_act"),
        )
        pd.testing.assert_series_equal(
            data[f"{method}_roundtrip_activated"],
            pd.Series(pd.array(activated, dtype="boolean"), name=f"{method}_roundtrip_activated"),
        )
        pd.testing.assert_series_equal(
            data[f"{method}_roundtrip_blank"],
            pd.Series(pd.array(blank, dtype="boolean"), name=f"{method}_roundtrip_blank"),
        )


def test_methods_is_the_merged_frames_own_method_order() -> None:
    """``METHODS`` and the merge contract's ``merged`` tuple are one order.

    ``METHODS`` is derived from ``METHOD_FACTORS`` while the merged frame's
    columns are built from ``_SOURCE_METHODS[FrameSource.MERGED]``; the sidecar's column
    order, the summary's method-major order, and the manifest's accounting all
    follow the former while the frame being scored follows the latter. Pinning
    them equal turns any future divergence between the two order sources into a
    loud failure here.
    """
    assert METHODS == _SOURCE_METHODS[FrameSource.MERGED]


def test_explicit_nl_roundtrip_reads_scores_and_serializes_all_three_columns(
    tmp_path: Path,
) -> None:
    frame = _nl_roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_frame(frame, tmp_path / "merged" / "steering_data.parquet")

    loaded = read_merged_frame(input_path, methods=NL_METHODS)
    acts: dict[str, float] = {}
    for method in NL_METHODS:
        for factor, text in zip(
            loaded["factor"], loaded[f"{method}_steered_generation"], strict=True
        ):
            if factor in roundtrip.SUPPORTED_METHOD_FACTORS[method] and str(text).strip():
                acts.setdefault(str(text), 0.5 + 0.25 * len(acts))
    data, n_unique = score_frame(
        loaded,
        {0: PROBE_FEATURE},
        dense_max_pool_fn=_StubDense(acts, N_SAE_FEATURES),
        count_tokens_fn=_count_words,
        n_ctx=N_CTX,
        n_sae_features=N_SAE_FEATURES,
        methods=NL_METHODS,
    )
    summary = build_summary(data, max_concepts=1, methods=NL_METHODS)
    manifest = build_manifest(
        input_path=input_path,
        input_sha256="a" * 64,
        max_concepts=1,
        data=data,
        n_unique_texts=n_unique,
        resolved=PROBE_RESOLVED,
        backend="nnterp",
        device=t.device("cpu"),
        n_ctx=N_CTX,
        n_sae_features=N_SAE_FEATURES,
        sae_sha256="b" * 64,
        versions=dict.fromkeys(EXPECTED_MANIFEST_VERSION_KEYS, "test"),
        methods=NL_METHODS,
    )

    assert list(data.columns[-3:]) == [
        "NLSteering_roundtrip_max_act",
        "NLSteering_roundtrip_activated",
        "NLSteering_roundtrip_blank",
    ]
    assert list(summary["method"].drop_duplicates()) == list(NL_METHODS)
    assert len(summary) == 62
    assert manifest["scoring"]["methods"] == list(NL_METHODS)
    assert list(manifest["accounting"]["per_method"]) == list(NL_METHODS)
    with pytest.raises(ValueError, match="unknown method result columns.*NLSteering"):
        read_merged_frame(input_path)


def test_parse_sae_link_valid() -> None:
    identity = parse_sae_link(VALID_LINK)
    assert identity == SaeLinkIdentity(
        model_id="gemma-2-2b", source="20-gemmascope-res-16k", feature_index=16038
    )
    assert identity.model_id == EXPECTED_PROBE_MODEL_ID
    assert identity.source == EXPECTED_PROBE_SOURCE


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/1", "scheme"),
        ("https://example.com/gemma-2-2b/20-gemmascope-res-16k/1", "host"),
        ("https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k", "path segments"),
        ("https://www.neuronpedia.org/a/b/c/d", "path segments"),
        ("https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/x", "feature index"),
        ("https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/-3", "feature index"),
        ("https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/", "path segments"),
        ("", "scheme"),
    ],
)
def test_parse_sae_link_rejects(url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_sae_link(url)


def test_parse_sae_link_rejects_a_nondecimal_digit_feature_index() -> None:
    """``str.isdigit`` accepts superscripts; ``int()`` does not -- so ``isdecimal`` guards.

    ``"16038²".isdigit()`` is ``True`` (U+00B2 is a digit character), so the
    old check let this URL through and the conversion right after it died with
    ``invalid literal for int() with base 10`` -- a ``ValueError`` that names
    neither ``sae_link`` nor the offending URL, from a line whose guard had
    just passed. ``isdecimal`` and ``int`` agree on one alphabet, so the URL is
    now refused by the guard, with the URL in the message.
    """
    url = "https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/16038²"
    assert "16038²".isdigit()

    with pytest.raises(ValueError, match="feature index") as excinfo:
        parse_sae_link(url)
    assert url in str(excinfo.value)


@pytest.mark.parametrize(
    "suffix",
    ["?foo=bar", "?", "#anchor", "#", "?foo=bar#anchor"],
    ids=["query", "empty_query", "fragment", "empty_fragment", "both"],
)
def test_parse_sae_link_rejects_query_and_fragment(suffix: str) -> None:
    """A link carrying anything past the three path segments is refused, not trimmed.

    ``urlsplit`` strips a query/fragment out of ``.path``, so the segment and
    index checks would BOTH pass on such a URL and the extra text would be
    silently dropped -- scoring a concept against a probe identity nobody
    checked. ``?``/``#`` alone parse to empty query/fragment strings and so are
    accepted, which is the non-vacuity for the truthiness test in production:
    those two ids must NOT raise.
    """
    url = f"{VALID_LINK}{suffix}"
    if suffix in {"?", "#"}:
        assert parse_sae_link(url) == parse_sae_link(VALID_LINK)
        return

    with pytest.raises(ValueError, match="query string or fragment") as excinfo:
        parse_sae_link(url)
    assert url in str(excinfo.value)


def test_read_merged_frame_accepts_canonical_merged_input(tmp_path: Path) -> None:
    frame = _roundtrip_frame(max_concepts=2)
    path = _write_frame(frame, tmp_path / "merged" / "steering_data.parquet")

    pd.testing.assert_frame_equal(read_merged_frame(path), frame)


def test_public_roundtrip_reader_rejects_a_malformed_explicit_method_tuple(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be exactly the ten-method default"):
        read_merged_frame(
            tmp_path / "steering_data.parquet",
            methods=(*METHODS, "NLSteering"),
        )


def test_read_merged_frame_rejects_wrong_filename(tmp_path: Path) -> None:
    path = _write_frame(_roundtrip_frame(), tmp_path / "merged.parquet")

    with pytest.raises(ValueError, match="must be named steering_data.parquet"):
        read_merged_frame(path)


def test_read_merged_frame_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="merged steering Parquet not found"):
        read_merged_frame(tmp_path / "steering_data.parquet")


def test_read_merged_frame_rejects_empty_frame(tmp_path: Path) -> None:
    path = _write_frame(_roundtrip_frame().iloc[:0], tmp_path / "steering_data.parquet")

    with pytest.raises(ValueError, match="must not be empty"):
        read_merged_frame(path)


def test_read_merged_frame_names_the_missing_concept_id_column(tmp_path: Path) -> None:
    """A frame without ``concept_id`` is reported as such, not as a bad argument.

    ``max_concepts`` is INFERRED from that column, so before the explicit check
    a frame missing it inferred ``max_concepts = 0`` and surfaced from deep
    inside ``validate_steering_frame`` as "max_concepts must be a positive
    integer" -- naming an argument the caller never passed. The assertion below
    is what keeps the fix honest: the old message must be gone.
    """
    frame = _roundtrip_frame().drop(columns=["concept_id"])
    path = _write_frame(frame, tmp_path / "steering_data.parquet")

    with pytest.raises(ValueError, match="concept_id") as excinfo:
        read_merged_frame(path)
    assert "max_concepts must be a positive integer" not in str(excinfo.value)


def test_read_merged_frame_rejects_noncanonical_row_order(tmp_path: Path) -> None:
    frame = _roundtrip_frame(max_concepts=2)
    swapped = frame.iloc[[1, 0, *range(2, len(frame))]].reset_index(drop=True)
    path = _write_frame(swapped, tmp_path / "steering_data.parquet")

    with pytest.raises(ValueError, match="canonical.*concept_id, input_id, factor.*order"):
        read_merged_frame(path)


def test_read_merged_frame_rejects_evaluated_frame(tmp_path: Path) -> None:
    """An already-judged frame carries judge columns the merge contract forbids."""
    frame = _roundtrip_frame()
    frame["DiffMean_LMJudgeEvaluator"] = 1.0
    path = _write_frame(frame, tmp_path / "steering_data.parquet")

    with pytest.raises(ValueError, match="columns do not match the contract") as excinfo:
        read_merged_frame(path)
    assert "DiffMean_LMJudgeEvaluator" in str(excinfo.value)


def test_validate_probe_identity_returns_feature_index_per_concept() -> None:
    assert validate_probe_identity(_roundtrip_frame(max_concepts=2)) == {0: 16038, 1: 16039}


def test_validate_probe_identity_rejects_an_empty_frame() -> None:
    """Non-vacuity for the empty-frame guard: zero rows must raise, not return ``{}``.

    An empty feature map is indistinguishable downstream from a frame whose
    concepts were all checked, so a caller that skipped
    :func:`read_merged_frame`'s own non-emptiness check would score nothing and
    report success. The test above is the other half: a populated frame does
    return a per-concept map.
    """
    with pytest.raises(ValueError, match="empty"):
        validate_probe_identity(_roundtrip_frame().iloc[:0])


def test_validate_probe_identity_rejects_unparseable_link() -> None:
    """Non-vacuity for the parse conjunct: the raw fixture's dummy links must not pass."""
    with pytest.raises(ValueError, match=r"concept 0:.*host"):
        validate_probe_identity(_merged_frame())


def test_validate_probe_identity_rejects_sae_id_mismatch() -> None:
    """Non-vacuity for the sae_id conjunct: the vendored ``sae_id = 0`` fallback."""
    frame = _roundtrip_frame(max_concepts=2)
    frame.loc[frame["concept_id"].eq(1), "sae_id"] = 0

    with pytest.raises(ValueError, match=r"concept 1:.*sae_id") as excinfo:
        validate_probe_identity(frame)
    assert "16039" in str(excinfo.value)


def test_validate_probe_identity_rejects_mixed_identities() -> None:
    """Non-vacuity for the shared-identity conjunct.

    Concept 1 keeps a matching ``sae_id`` and a well-formed v1 source, so the
    parse and ``sae_id`` conjuncts both pass; only the cross-concept identity
    differs. (A frame CANNOT fail this conjunct alone: two concepts that each
    matched the expected pair would be identical, so any split necessarily
    also puts one concept off the expected pair -- the shared-identity check
    runs first so the message names the split rather than one concept.)
    """
    frame = _roundtrip_frame(max_concepts=2)
    _relink(
        frame,
        1,
        model_id=EXPECTED_PROBE_MODEL_ID,
        source="12-gemmascope-res-16k",
        feature_index=16039,
    )

    with pytest.raises(ValueError, match=r"concept 1:.*single probe identity") as excinfo:
        validate_probe_identity(frame)
    assert "12-gemmascope-res-16k" in str(excinfo.value)


def test_validate_probe_identity_rejects_wrong_model_id() -> None:
    """Non-vacuity for the model half of the expected-pair conjunct."""
    frame = _roundtrip_frame()
    _relink(frame, 0, model_id="gemma-2-9b", source=EXPECTED_PROBE_SOURCE, feature_index=16038)

    with pytest.raises(ValueError, match=r"concept 0:.*expected roundtrip probe") as excinfo:
        validate_probe_identity(frame)
    assert EXPECTED_PROBE_MODEL_ID in str(excinfo.value)


def test_validate_probe_identity_rejects_v2_source_grammar() -> None:
    """Non-vacuity for the source half: a GemmaScope-2 source at the same layer."""
    frame = _roundtrip_frame()
    _relink(
        frame,
        0,
        model_id=EXPECTED_PROBE_MODEL_ID,
        source="20-gemmascope-2-res-16k",
        feature_index=16038,
    )

    with pytest.raises(ValueError, match=r"concept 0:.*expected roundtrip probe") as excinfo:
        validate_probe_identity(frame)
    assert EXPECTED_PROBE_SOURCE in str(excinfo.value)


def test_expected_probe_source_decodes_to_the_probe_release() -> None:
    """Pin the module-constants relationship ``validate_probe_identity`` relies on.

    Which SAE ``EXPECTED_PROBE_SOURCE`` denotes is a property of the module
    constants, not of any frame: the per-concept checks pin the source STRING,
    and the release tuple is DERIVED from that string at import
    (``_EXPECTED_PROBE_RELEASE``) rather than written out twice. This test is
    what holds the derived value to the intended release -- the gemma-scope v1
    layer-20 residual 16k SAE with no L0 tier, spelled out here independently
    -- so editing either constant cannot silently retarget the probe. The
    second assertion keeps the pin non-vacuous by exhibiting a source that
    decodes to something else at the same layer and width (the v2 release).
    """
    assert _EXPECTED_PROBE_RELEASE == EXPECTED_PROBE_RELEASE
    assert parse_neuronpedia_source("20-gemmascope-2-res-16k") != EXPECTED_PROBE_RELEASE


def test_score_frame_scores_every_declared_cell() -> None:
    """Happy path: exact schema, dtypes, per-cell values, and off-grid nulls."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    acts = _distinct_acts(frame)
    assert len(acts) == DECLARED_PER_CONCEPT
    stub = _StubDense(acts, N_SAE_FEATURES)

    data, n_unique_texts = _score(frame, stub)

    _assert_sidecar_matches(data, frame, acts)
    assert n_unique_texts == DECLARED_PER_CONCEPT
    assert stub.calls == [list(acts)]
    for method in METHODS:
        assert data[f"{method}_roundtrip_max_act"].dtype == "float64"
        assert data[f"{method}_roundtrip_activated"].dtype == "boolean"
        assert data[f"{method}_roundtrip_blank"].dtype == "boolean"


def test_score_frame_leaves_off_grid_cells_null() -> None:
    """All four single-factor methods are undeclared outside factor 1.0."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    acts = _distinct_acts(frame)

    data, _ = _score(frame, _StubDense(acts, N_SAE_FEATURES))

    off_grid = data.loc[data["factor"].ne(1.0)]
    on_grid = data.loc[data["factor"].eq(1.0)]
    assert len(off_grid) == 11 * 10  # 11 undeclared factors x 10 inputs
    assert len(on_grid) == 10
    for method in ("PromptSteering", "LoRA", "EncDecPromptSteering", "EncDecLoRA"):
        assert off_grid[f"{method}_roundtrip_max_act"].isna().all()
        assert off_grid[f"{method}_roundtrip_activated"].isna().all()
        assert off_grid[f"{method}_roundtrip_blank"].isna().all()
        assert off_grid[f"{method}_roundtrip_activated"].iloc[0] is pd.NA
        assert off_grid[f"{method}_roundtrip_blank"].iloc[0] is pd.NA
        # Non-vacuity: the same columns ARE populated on the declared factor.
        assert on_grid[f"{method}_roundtrip_activated"].notna().all()
        assert on_grid[f"{method}_roundtrip_blank"].notna().all()


def test_score_frame_treats_whitespace_generation_as_blank_and_never_probes_it() -> None:
    """A blank cell is declared-but-unscorable: False/True/NaN, and never probed."""
    blank_text = "  \n "
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    _set_generation(frame, 0, "DiffMean", blank_text)
    acts = _distinct_acts(frame)
    assert len(acts) == DECLARED_PER_CONCEPT - 1
    stub = _StubDense(acts, N_SAE_FEATURES)

    data, n_unique_texts = _score(frame, stub)

    _assert_sidecar_matches(data, frame, acts)
    assert bool(data["DiffMean_roundtrip_activated"].iloc[0]) is False
    assert bool(data["DiffMean_roundtrip_blank"].iloc[0]) is True
    assert math.isnan(data["DiffMean_roundtrip_max_act"].iloc[0])
    assert all(blank_text not in call for call in stub.calls)
    assert n_unique_texts == DECLARED_PER_CONCEPT - 1
    # The blank cell still counts as scored, so accounting must not have fired.
    assert data["DiffMean_roundtrip_activated"].notna().sum() == 120


def test_score_frame_scores_an_all_blank_concept_with_one_empty_probe_call() -> None:
    """A concept with nothing to probe keeps the one-call-per-concept shape.

    Every one of the concept's 760 declared generations is whitespace, so the
    deduplicated batch is empty. ``_score_concept`` promises it still calls the
    pool exactly once (the production primitive answers an empty batch with a
    ``(0, n_features)`` tensor, so there is no skip branch); this pins that
    promise, along with every declared cell scoring as a blank and the
    accounting check still passing on a frame where nothing reached the SAE.
    """
    frame = _blank_every_declared_generation(_roundtrip_frame(feature_ids=(PROBE_FEATURE,)))
    stub = _EmptyBatchDense()

    data, n_unique_texts = _score(frame, stub)

    assert stub.calls == [[]]
    assert n_unique_texts == 0
    _assert_sidecar_matches(data, frame, {})
    for method in METHODS:
        declared = data["factor"].isin(METHOD_FACTORS[method])
        assert int(declared.sum()) == 10 * len(METHOD_FACTORS[method])
        assert data.loc[declared, f"{method}_roundtrip_activated"].eq(False).all()
        assert data.loc[declared, f"{method}_roundtrip_blank"].eq(True).all()
        assert data.loc[declared, f"{method}_roundtrip_max_act"].isna().all()
        assert data.loc[~declared, f"{method}_roundtrip_activated"].isna().all()
        assert data.loc[~declared, f"{method}_roundtrip_blank"].isna().all()
        assert data.loc[~declared, f"{method}_roundtrip_max_act"].isna().all()


def test_score_frame_activation_is_strictly_positive() -> None:
    """A feature that never fires is not activated; any positive max-act is."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    acts = _distinct_acts(frame)
    silent = str(frame["DiffMean_steered_generation"].iloc[0])
    faint = str(frame["DiffMean_steered_generation"].iloc[1])
    acts[silent] = 0.0
    acts[faint] = 1e-30

    data, _ = _score(frame, _StubDense(acts, N_SAE_FEATURES))

    assert data["DiffMean_roundtrip_max_act"].iloc[0] == 0.0
    assert bool(data["DiffMean_roundtrip_activated"].iloc[0]) is False
    assert bool(data["DiffMean_roundtrip_blank"].iloc[0]) is False
    assert data["DiffMean_roundtrip_max_act"].iloc[1] == pytest.approx(1e-30)
    assert bool(data["DiffMean_roundtrip_activated"].iloc[1]) is True


def test_score_frame_probes_a_repeated_generation_once() -> None:
    """Identical text in two methods' cells is encoded once and scored the same."""
    shared = "identical steered text shared by two methods"
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    _set_generation(frame, 0, "DiffMean", shared)
    _set_generation(frame, 0, "LinearProbe", shared)
    acts = _distinct_acts(frame)
    stub = _StubDense(acts, N_SAE_FEATURES)

    data, n_unique_texts = _score(frame, stub)

    assert sum(call.count(shared) for call in stub.calls) == 1
    assert n_unique_texts == DECLARED_PER_CONCEPT - 1
    assert data["DiffMean_roundtrip_max_act"].iloc[0] == acts[shared]
    assert data["LinearProbe_roundtrip_max_act"].iloc[0] == acts[shared]


def test_score_frame_makes_one_dense_call_per_concept() -> None:
    """Concepts are batched one at a time: one SAE pass each, no cross-concept mixing."""
    frame = _roundtrip_frame(max_concepts=2, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    acts = _distinct_acts(frame)
    stub = _StubDense(acts, N_SAE_FEATURES)

    data, n_unique_texts = _score(
        frame, stub, feature_by_concept={0: PROBE_FEATURE, 1: PROBE_FEATURE}
    )

    assert len(stub.calls) == 2
    assert [len(call) for call in stub.calls] == [DECLARED_PER_CONCEPT, DECLARED_PER_CONCEPT]
    # Fixture generations embed "<concept_id>/<input_id>/<factor>".
    assert all(" 0/" in text for text in stub.calls[0])
    assert all(" 1/" in text for text in stub.calls[1])
    assert n_unique_texts == 2 * DECLARED_PER_CONCEPT
    _assert_sidecar_matches(data, frame, acts)


@pytest.mark.parametrize("n_tokens", [9000, N_CTX])
def test_score_frame_rejects_text_that_would_be_front_clamped(n_tokens: int) -> None:
    """The dense pool silently front-clamps; refuse rather than score a truncation.

    ``n_tokens == N_CTX`` is the boundary case: the pool injects one BOS, so a
    text of exactly ``n_ctx`` content tokens no longer fits.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    stub = _StubDense(_distinct_acts(frame), N_SAE_FEATURES)

    with pytest.raises(ValueError, match="n_ctx"):
        _score(frame, stub, count_tokens_fn=lambda _text: n_tokens)
    assert stub.calls == []


def test_score_frame_accepts_text_that_exactly_fills_n_ctx_with_bos() -> None:
    """Non-vacuity for the BOS budget: ``n_ctx - 1`` content tokens still fit."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    stub = _StubDense(_distinct_acts(frame), N_SAE_FEATURES)

    _, n_unique_texts = _score(frame, stub, count_tokens_fn=lambda _text: N_CTX - 1)

    assert n_unique_texts == DECLARED_PER_CONCEPT


@pytest.mark.parametrize(
    "shape_fn",
    [
        lambda n: (n, N_SAE_FEATURES - 1),
        lambda n: (n + 1, N_SAE_FEATURES),
        lambda n: (N_SAE_FEATURES,),
    ],
    ids=["narrow", "extra_row", "one_dimensional"],
)
def test_score_frame_rejects_wrongly_shaped_activations(
    shape_fn: Callable[[int], tuple[int, ...]],
) -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))

    def dense(texts: list[str]) -> t.Tensor:
        return t.zeros(shape_fn(len(texts)), dtype=t.float32)

    with pytest.raises(ValueError, match="shape"):
        _score(frame, dense)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_score_frame_rejects_non_finite_activations(bad: float) -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))

    def dense(texts: list[str]) -> t.Tensor:
        out = t.zeros((len(texts), N_SAE_FEATURES), dtype=t.float32)
        out[0, PROBE_FEATURE] = bad
        return out

    with pytest.raises(ValueError, match="finite"):
        _score(frame, dense)


def test_score_frame_rejects_negative_activations() -> None:
    """JumpReLU post-activations are non-negative; a negative one means wrong hook."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))

    def dense(texts: list[str]) -> t.Tensor:
        out = t.zeros((len(texts), N_SAE_FEATURES), dtype=t.float32)
        out[0, PROBE_FEATURE] = -0.5
        return out

    with pytest.raises(ValueError, match="negative"):
        _score(frame, dense)


def test_score_frame_rejects_feature_index_outside_probe_width() -> None:
    """A feature index the loaded SAE cannot hold must not be silently clipped."""
    frame = _roundtrip_frame(feature_ids=(N_SAE_FEATURES,))
    stub = _StubDense(_distinct_acts(frame), N_SAE_FEATURES)

    with pytest.raises(ValueError, match="n_sae_features"):
        _score(frame, stub, feature_by_concept={0: N_SAE_FEATURES})
    assert stub.calls == []


def test_score_frame_rejects_frame_missing_a_declared_row() -> None:
    """Non-vacuity for the final accounting check: a short frame must not pass.

    ``score_frame`` fills any cell it did not score as off-grid; the accounting
    check is what turns "quietly unscored" into a loud failure, so this drops a
    row (concept 0 / input 0 / factor 0.0) and requires it to fire.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,)).iloc[1:]
    stub = _StubDense(_distinct_acts(frame), N_SAE_FEATURES)

    with pytest.raises(RuntimeError, match="accounting") as excinfo:
        _score(frame, stub)
    assert "DiffMean" in str(excinfo.value)


def test_build_summary_computes_hand_calculated_activation_rate_and_sem() -> None:
    """The brief's hand-computed fixture: 3/10 and 7/10 -> rate 0.5, SEM 0.2.

    Per-concept rates are 0.3 and 0.7: their mean is 0.5 (matching the pooled
    10/20 rate exactly, since both concepts declare the same 10 inputs), sample
    std (ddof=1) = sqrt(((0.3-0.5)**2 + (0.7-0.5)**2) / 1) = 0.2 * sqrt(2),
    divided by sqrt(2) concepts = 0.2.

    Both numbers are pinned UNCHANGED across the removal of the
    ``concept_macro_rate`` column: dropping a duplicate of ``activation_rate``
    must not move ``activation_rate`` or ``concept_macro_sem`` by a bit.
    """
    data, max_concepts = _two_concept_diffmean_activation_fixture()

    summary = build_summary(data, max_concepts=max_concepts)

    row = summary.loc[(summary["method"] == "DiffMean") & (summary["factor"] == 1.0)].iloc[0]
    assert row["n_declared"] == 20
    assert row["n_blank"] == 0
    assert row["n_activated"] == 10
    assert row["activation_rate"] == pytest.approx(0.5)
    assert row["concept_macro_sem"] == pytest.approx(0.2)


def test_build_summary_pooled_rate_is_the_macro_rate_on_the_balanced_grid() -> None:
    """The removed ``concept_macro_rate`` could not have differed from the pooled rate.

    ``build_summary`` refuses any (method, factor) whose non-null cell count is
    not ``INPUTS_PER_CONCEPT * max_concepts``, so every concept contributes the
    SAME number of cells and the mean of the per-concept rates is the pooled
    rate identically. This recomputes the dropped column from the sidecar --
    including on the deliberately lopsided 3/10-vs-7/10 concept split, where an
    unbalanced grid would have separated them -- and requires exact agreement
    on all 76 rows, which is the justification for having only one of the two.
    """
    data, max_concepts = _two_concept_diffmean_activation_fixture()

    summary = build_summary(data, max_concepts=max_concepts)

    macro_rates: list[float] = []
    for method, factor in zip(summary["method"], summary["factor"], strict=True):
        subset = data.loc[data["factor"].eq(factor)]
        per_concept = (
            subset.assign(rate=subset[f"{method}_roundtrip_activated"].astype("float64"))
            .groupby("concept_id")["rate"]
            .mean()
        )
        assert len(per_concept) == max_concepts
        macro_rates.append(float(per_concept.mean()))
    # Non-vacuity: the fixture really does contain a non-degenerate rate.
    assert 0.5 in macro_rates
    assert macro_rates == list(summary["activation_rate"])


def test_build_summary_has_76_rows_in_method_major_factor_ascending_order() -> None:
    data, max_concepts = _two_concept_diffmean_activation_fixture()

    summary = build_summary(data, max_concepts=max_concepts)

    assert list(summary.columns) == EXPECTED_SUMMARY_COLUMNS
    assert len(summary) == 76
    assert list(zip(summary["method"], summary["factor"], strict=True)) == (
        EXPECTED_SUMMARY_METHOD_FACTOR_ORDER
    )


def test_build_summary_column_dtypes() -> None:
    data, max_concepts = _two_concept_diffmean_activation_fixture()

    summary = build_summary(data, max_concepts=max_concepts)

    assert pd.api.types.is_string_dtype(summary["method"].dtype)
    assert summary["factor"].dtype == "float64"
    assert summary["n_declared"].dtype == "int64"
    assert summary["n_blank"].dtype == "int64"
    assert summary["n_activated"].dtype == "int64"
    assert summary["activation_rate"].dtype == "float64"
    assert summary["concept_macro_sem"].dtype == "float64"


def test_build_summary_counts_blank_in_denominator() -> None:
    """A blank generation is still a declared cell: it lowers the rate, not the count."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    pos = _position_of(frame, concept_id=0, input_id=0, factor=1.0)
    _set_generation(frame, pos, "DiffMean", "  \n ")
    acts = _distinct_acts(frame)
    data, _ = _score(frame, _StubDense(acts, N_SAE_FEATURES))

    summary = build_summary(data, max_concepts=1)

    row = summary.loc[(summary["method"] == "DiffMean") & (summary["factor"] == 1.0)].iloc[0]
    assert row["n_declared"] == 10
    assert row["n_blank"] == 1
    assert row["n_activated"] == 9
    assert row["activation_rate"] == pytest.approx(0.9)


def test_build_summary_sem_is_nan_when_max_concepts_is_one() -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    acts = _distinct_acts(frame)
    # Deactivate 7/10 DiffMean factor-1.0 cells so the rate columns are NOT
    # all degenerate -- otherwise an all-1.0 activation rate everywhere would
    # make the all-NaN SEM assertion below vacuous (indistinguishable from a
    # bug that zeroed the SEM column outright).
    for input_id in range(3, 10):
        text = f"DiffMean healthy generated answer 0/{input_id}/1.0"
        acts[text] = 0.0
    data, _ = _score(frame, _StubDense(acts, N_SAE_FEATURES))

    summary = build_summary(data, max_concepts=1)

    assert len(summary) == 76
    assert summary["concept_macro_sem"].isna().all()
    assert summary["activation_rate"].nunique() > 1


def test_build_summary_single_factor_methods_have_rows_only_at_factor_one() -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    acts = _distinct_acts(frame)
    data, _ = _score(frame, _StubDense(acts, N_SAE_FEATURES))

    summary = build_summary(data, max_concepts=1)

    for method in ("PromptSteering", "LoRA", "EncDecPromptSteering", "EncDecLoRA"):
        rows = summary.loc[summary["method"] == method]
        assert list(rows["factor"]) == [1.0]


def test_build_summary_raises_on_accounting_mismatch() -> None:
    """Non-vacuity for the accounting conjunct: a corrupted sidecar must not pass silently.

    Forces one declared DiffMean-factor-1.0 cell's ``activated`` value back to
    ``pd.NA``, mimicking a sidecar row that was never actually scored; the
    non-NA count for that (method, factor) now falls one short of
    ``n_declared`` and must raise rather than silently under-counting.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    acts = _distinct_acts(frame)
    data, _ = _score(frame, _StubDense(acts, N_SAE_FEATURES))
    pos = _position_of(frame, concept_id=0, input_id=0, factor=1.0)
    data.iloc[pos, data.columns.get_loc("DiffMean_roundtrip_activated")] = pd.NA

    with pytest.raises(RuntimeError, match="accounting") as excinfo:
        build_summary(data, max_concepts=1)
    assert "DiffMean" in str(excinfo.value)


class _FakeSae(t.nn.Module):
    """A tiny stand-in for a SAELens SAE: just a ``state_dict`` of parameters.

    Parameters are registered in call order, so passing the same weights under
    a different keyword order produces a differently-ordered ``state_dict`` --
    which is what the sorted-by-key pin below needs.
    """

    def __init__(self, **weights: t.Tensor) -> None:
        super().__init__()
        for name, tensor in weights.items():
            setattr(self, name, t.nn.Parameter(tensor, requires_grad=False))


class _FakeProbeModel:
    """Stand-in for the TransformerLens model ``load_sae_and_model`` returns."""


class _RecordingLoader:
    """Fake ``load_sae_and_model``: records its kwargs, returns fixed objects.

    Keyword-only, so a caller that passes the model name or layer positionally
    fails here instead of silently loading a differently-shaped probe.
    """

    def __init__(self, model: object, sae: object) -> None:
        self.model = model
        self.sae = sae
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, *, model_name: str, layer: int, device: str, resolved: ResolvedSAE
    ) -> tuple[object, object, object]:
        self.calls.append(
            {"model_name": model_name, "layer": layer, "device": device, "resolved": resolved}
        )
        return self.model, self.sae, object()


class _RecordingNnterp:
    """Fake ``build_nnterp_model``: records its calls, returns a fixed result."""

    def __init__(self, result: object | None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tl_model: object, *, device: str) -> object | None:
        self.calls.append({"tl_model": tl_model, "device": device})
        return self.result


def _install_fake_loaders(
    monkeypatch: pytest.MonkeyPatch, *, nnterp: object | None
) -> tuple[_RecordingLoader, _RecordingNnterp]:
    """Replace both real loaders: no weights are downloaded and no CUDA is touched."""
    loader = _RecordingLoader(_FakeProbeModel(), object())
    builder = _RecordingNnterp(nnterp)
    monkeypatch.setattr(roundtrip, "load_sae_and_model", loader)
    monkeypatch.setattr(roundtrip, "build_nnterp_model", builder)
    return loader, builder


def _scored_sidecar(*, blank_cells: int = 0) -> tuple[pd.DataFrame, int]:
    """A single-concept scored sidecar, with ``blank_cells`` blank DiffMean cells."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    for position in range(blank_cells):
        _set_generation(frame, position, "DiffMean", "   ")
    acts = _distinct_acts(frame)
    return _score(frame, _StubDense(acts, N_SAE_FEATURES))


def _manifest(
    data: pd.DataFrame,
    n_unique_texts: int,
    *,
    max_concepts: int = 1,
    backend: str = "nnterp",
    device: str = "cuda",
    versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a manifest, resolving real package versions unless given a mapping."""
    return build_manifest(
        input_path=MANIFEST_INPUT_PATH,
        input_sha256=MANIFEST_INPUT_SHA,
        max_concepts=max_concepts,
        data=data,
        n_unique_texts=n_unique_texts,
        resolved=PROBE_RESOLVED,
        backend=backend,
        device=t.device(device),
        n_ctx=N_CTX,
        n_sae_features=N_SAE_FEATURES,
        sae_sha256=MANIFEST_SAE_SHA,
        versions=resolve_manifest_versions() if versions is None else versions,
    )


def _fake_sae_weights() -> dict[str, t.Tensor]:
    """Values exactly representable in float16, so the dtype pin is unambiguous."""
    return {
        "W_enc": t.tensor([[0.5, -0.25], [1.5, 2.0]], dtype=t.float32),
        "b_enc": t.tensor([0.75, -1.0], dtype=t.float32),
    }


def test_sae_weights_sha256_is_stable_across_calls() -> None:
    sae = _FakeSae(**_fake_sae_weights())

    assert sae_weights_sha256(sae) == sae_weights_sha256(sae)
    assert len(sae_weights_sha256(sae)) == 64


def test_sae_weights_sha256_changes_when_a_weight_changes() -> None:
    weights = _fake_sae_weights()
    before = sae_weights_sha256(_FakeSae(**weights))
    weights["W_enc"] = weights["W_enc"].clone()
    weights["W_enc"][1, 1] = 2.25

    assert sae_weights_sha256(_FakeSae(**weights)) != before


def test_sae_weights_sha256_changes_when_a_parameter_is_renamed() -> None:
    """Non-vacuity for hashing the KEY: identical tensors under a different name differ."""
    weights = _fake_sae_weights()
    renamed = {"W_dec": weights["W_enc"], "b_enc": weights["b_enc"]}

    assert sae_weights_sha256(_FakeSae(**renamed)) != sae_weights_sha256(_FakeSae(**weights))


def test_sae_weights_sha256_ignores_state_dict_insertion_order() -> None:
    """Non-vacuity for sorting by key: registration order must not move the digest."""
    weights = _fake_sae_weights()
    forward = _FakeSae(W_enc=weights["W_enc"], b_enc=weights["b_enc"])
    reversed_registration = _FakeSae(b_enc=weights["b_enc"], W_enc=weights["W_enc"])
    assert list(forward.state_dict()) != list(reversed_registration.state_dict())

    assert sae_weights_sha256(forward) == sae_weights_sha256(reversed_registration)


def test_sae_weights_sha256_normalizes_dtype_to_float32() -> None:
    """A float16 SAE with identical values hashes identically to its float32 twin.

    The digest is a weight-identity fingerprint, not a byte-for-byte
    serialization: it must not move because the probe was loaded in a
    different precision (every value here is exact in float16).
    """
    weights = _fake_sae_weights()
    half = {name: tensor.to(t.float16) for name, tensor in weights.items()}

    assert sae_weights_sha256(_FakeSae(**half)) == sae_weights_sha256(_FakeSae(**weights))


def test_load_roundtrip_probe_resolves_the_derived_probe_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loaded SAE is exactly the identity ``_EXPECTED_PROBE_RELEASE`` denotes.

    ``validate_probe_identity`` pins every concept's ``sae_link`` to the source
    STRING, and the loader resolves from that same string's decoding -- so the
    identity validated in the frame and the identity loaded are one tuple, and
    this pins that tuple to the registry's resolution of it.
    """
    loader, _builder = _install_fake_loaders(monkeypatch, nnterp=None)
    release, activation_type, layer, width, l0 = _EXPECTED_PROBE_RELEASE

    _model, _sae, resolved, _backend = load_roundtrip_probe(t.device("cpu"))

    assert resolved == resolve_sae(
        model=EXPECTED_PROBE_MODEL_ID,
        release=release,
        activation_type=activation_type,
        layer=layer,
        width=width,
        l0=l0,
    )
    assert resolved == PROBE_RESOLVED
    assert loader.calls == [
        {
            "model_name": EXPECTED_PROBE_MODEL_ID,
            "layer": PROBE_SAE_LAYER,
            "device": "cpu",
            "resolved": PROBE_RESOLVED,
        }
    ]


def test_load_roundtrip_probe_follows_the_module_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity for "derived, not re-typed": retarget the derivation, the load follows.

    Swapping the derived activation site to MLP (same model, layer and width,
    so the layer consistency guard still holds) must change the resolved SAE.
    A loader that re-typed ``ActivationType.residual`` as a literal would
    resolve the residual SAE here and fail.
    """
    loader, _builder = _install_fake_loaders(monkeypatch, nnterp=None)
    monkeypatch.setattr(
        roundtrip,
        "_EXPECTED_PROBE_RELEASE",
        (SAERelease.gemmascope, ActivationType.mlp, PROBE_SAE_LAYER, SAEWidth.W_16K, None),
    )

    _model, _sae, resolved, _backend = load_roundtrip_probe(t.device("cpu"))

    assert resolved.hook_name == "blocks.20.hook_mlp_out"
    assert resolved.release == "gemma-scope-2b-pt-mlp-canonical"
    assert loader.calls[0]["resolved"] == resolved


def test_load_roundtrip_probe_rejects_a_layer_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity for the layer guard: the derivation and PROBE_SAE_LAYER must agree.

    ``PROBE_SAE_LAYER`` is the layer the probe forwards to; the derived tuple
    is what its SAE is resolved from. If the two ever disagreed the roundtrip
    would read one layer's activations through another layer's SAE.
    """
    loader, _builder = _install_fake_loaders(monkeypatch, nnterp=None)
    monkeypatch.setattr(
        roundtrip,
        "_EXPECTED_PROBE_RELEASE",
        (SAERelease.gemmascope, ActivationType.residual, 12, SAEWidth.W_16K, None),
    )

    with pytest.raises(RuntimeError, match="PROBE_SAE_LAYER"):
        load_roundtrip_probe(t.device("cpu"))
    assert loader.calls == []


def test_load_roundtrip_probe_on_cpu_uses_transformerlens_and_never_builds_nnterp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, builder = _install_fake_loaders(monkeypatch, nnterp=object())

    model, sae, _resolved, backend = load_roundtrip_probe(t.device("cpu"))

    assert backend == "transformerlens"
    assert builder.calls == []
    assert not hasattr(model, "nnterp_model")
    assert model is loader.model
    assert sae is loader.sae


def test_load_roundtrip_probe_on_cuda_attaches_the_nnterp_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity for the CPU branch above: on CUDA the accelerated backend IS built.

    ``cuda.is_available`` is forced True so this exercises the CUDA branch on
    any box; the availability guard itself is pinned separately below.
    """
    nnterp = object()
    loader, builder = _install_fake_loaders(monkeypatch, nnterp=nnterp)
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)

    model, _sae, _resolved, backend = load_roundtrip_probe(t.device("cuda"))

    assert backend == "nnterp"
    assert model.nnterp_model is nnterp
    assert builder.calls == [{"tl_model": loader.model, "device": "cuda"}]
    assert loader.calls[0]["device"] == "cuda"


def test_load_roundtrip_probe_rejects_an_unsupported_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``cuda`` and ``cpu`` are scoreable; ``mps`` (and friends) are refused.

    Non-vacuity for the whitelist: the two tests above show ``cpu`` and
    ``cuda`` both load, so this fails ONLY because ``mps`` is off the list --
    and it must fail before either loader runs, since a third numeric path
    (no nnterp backend, so silently TransformerLens on an accelerator) would
    produce roundtrip numbers that are not comparable with any other run's.
    """
    loader, builder = _install_fake_loaders(monkeypatch, nnterp=object())

    with pytest.raises(RuntimeError, match=r"mps.*not a supported roundtrip probe device"):
        load_roundtrip_probe(t.device("mps"))
    assert loader.calls == []
    assert builder.calls == []


def test_load_roundtrip_probe_rejects_cuda_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CUDA device on a CPU-only box is refused before anything loads.

    ``resolve_probe_device`` only consults ``cuda.is_available()`` for
    ``"auto"``, so ``--device cuda`` on a machine without a CUDA runtime used
    to reach ``load_sae_and_model`` and die inside torch. The completed
    preflight names the fix instead, and must fire before either loader runs.
    ``test_load_roundtrip_probe_on_cuda_attaches_the_nnterp_model`` is the
    non-vacuity: with availability left alone, the very same call loads.
    """
    loader, builder = _install_fake_loaders(monkeypatch, nnterp=object())
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"cuda.*is_available\(\) is False") as excinfo:
        load_roundtrip_probe(t.device("cuda"))

    assert "--device cpu" in str(excinfo.value)
    assert loader.calls == []
    assert builder.calls == []


def test_load_roundtrip_probe_raises_when_nnterp_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No silent TransformerLens fallback on CUDA (accelerated-backend fail-fast)."""
    _install_fake_loaders(monkeypatch, nnterp=None)
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="nnterp"):
        load_roundtrip_probe(t.device("cuda"))


def test_build_manifest_reports_the_declared_accounting() -> None:
    data, n_unique_texts = _scored_sidecar()

    manifest = _manifest(data, n_unique_texts)

    assert n_unique_texts == DECLARED_PER_CONCEPT
    assert manifest["accounting"] == {
        "per_method": {
            "DiffMean": {"n_declared": 120, "n_blank": 0, "n_scored": 120},
            "LinearProbe": {"n_declared": 120, "n_blank": 0, "n_scored": 120},
            "PromptSteering": {"n_declared": 10, "n_blank": 0, "n_scored": 10},
            "LoRA": {"n_declared": 10, "n_blank": 0, "n_scored": 10},
            "EncDecDiffMean": {"n_declared": 120, "n_blank": 0, "n_scored": 120},
            "EncDecLinearProbe": {"n_declared": 120, "n_blank": 0, "n_scored": 120},
            "EncDecPromptSteering": {"n_declared": 10, "n_blank": 0, "n_scored": 10},
            "EncDecLoRA": {"n_declared": 10, "n_blank": 0, "n_scored": 10},
            "SelfConditioning": {"n_declared": 120, "n_blank": 0, "n_scored": 120},
            "SelfConditioningMasked": {"n_declared": 120, "n_blank": 0, "n_scored": 120},
        },
        "n_unique_texts_per_concept_sum": DECLARED_PER_CONCEPT,
    }


def test_build_manifest_accounts_blank_generations_as_declared_but_unscored() -> None:
    """Non-vacuity for ``n_blank``/``n_scored``: blanks stay declared and leave ``n_scored``."""
    data, n_unique_texts = _scored_sidecar(blank_cells=3)

    manifest = _manifest(data, n_unique_texts)

    assert manifest["accounting"]["per_method"]["DiffMean"] == {
        "n_declared": 120,
        "n_blank": 3,
        "n_scored": 117,
    }
    assert manifest["accounting"]["per_method"]["LinearProbe"]["n_blank"] == 0
    assert manifest["accounting"]["n_unique_texts_per_concept_sum"] == DECLARED_PER_CONCEPT - 3


def test_build_manifest_names_the_unique_text_count_a_per_concept_sum() -> None:
    """The key says what the number is: a SUM of per-concept unique-text counts.

    The same generation under two concepts is encoded (and counted) twice --
    the count is the probe's encode workload, not a global distinct-text
    count -- so the bare name ``n_unique_texts`` must not appear.
    """
    frame = _roundtrip_frame(max_concepts=2, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    shared = "the very same steered text under both concepts"
    for concept_id in (0, 1):
        position = _position_of(frame, concept_id=concept_id, input_id=0, factor=1.0)
        _set_generation(frame, position, "LoRA", shared)
    acts = _distinct_acts(frame)
    data, n_unique_texts = _score(
        frame,
        _StubDense(acts, N_SAE_FEATURES),
        feature_by_concept={0: PROBE_FEATURE, 1: PROBE_FEATURE},
    )

    manifest = _manifest(data, n_unique_texts, max_concepts=2)

    accounting = manifest["accounting"]
    assert list(accounting) == ["per_method", "n_unique_texts_per_concept_sum"]
    assert "n_unique_texts" not in accounting
    # Each concept still encodes all 760 of its own texts: the shared text is
    # deduplicated WITHIN a concept only, so the sum stays 2 x 760 -- exactly
    # the double-count the key name is there to declare.
    assert accounting["n_unique_texts_per_concept_sum"] == 2 * DECLARED_PER_CONCEPT


def test_build_manifest_records_the_input_probe_and_scoring_policy() -> None:
    data, n_unique_texts = _scored_sidecar()

    manifest = _manifest(data, n_unique_texts)

    assert list(manifest) == EXPECTED_MANIFEST_KEYS
    assert manifest["schema_version"] == ROUNDTRIP_SCHEMA_VERSION
    assert list(manifest["input"]) == EXPECTED_MANIFEST_INPUT_KEYS
    assert manifest["input"] == {
        "path": str(MANIFEST_INPUT_PATH),
        "sha256": MANIFEST_INPUT_SHA,
        "n_rows": len(data),
        "max_concepts": 1,
    }
    assert list(manifest["probe"]) == EXPECTED_MANIFEST_PROBE_KEYS
    assert manifest["probe"] == {
        "model_name": EXPECTED_PROBE_MODEL_ID,
        "hf_model_id": PROBE_TOKENIZER_ID,
        "release": PROBE_RESOLVED.release,
        "saelens_sae_id": PROBE_RESOLVED.sae_id,
        "hook_name": PROBE_RESOLVED.hook_name,
        "layer": PROBE_SAE_LAYER,
        "activation": "residual",
        "width": "16k",
        "n_sae_features": N_SAE_FEATURES,
        "l0": None,
        "model_dtype": "float32",
        "sae_dtype": "float32",
        "backend": "nnterp",
        "device": "cuda",
        "sae_weights_sha256": MANIFEST_SAE_SHA,
        "n_ctx": N_CTX,
    }
    assert manifest["probe"]["hf_model_id"] == "google/gemma-2-2b"
    # The probe's SAELens id is a STRING path into a release; the sidecar's own
    # ``sae_id`` column is an int64 Neuronpedia feature index. The manifest key
    # carries the ``saelens_`` prefix so the two are not read as one quantity,
    # and the bare name must be gone.
    assert manifest["probe"]["saelens_sae_id"] == "layer_20/width_16k/canonical"
    assert "sae_id" not in manifest["probe"]
    assert list(manifest["scoring"]) == EXPECTED_MANIFEST_SCORING_KEYS
    assert manifest["scoring"] == {
        "batch_size": ROUNDTRIP_BATCH_SIZE,
        "activation_threshold": "post_act > 0",
        "blank_policy": "count_as_not_activated",
    }
    assert list(manifest["versions"]) == EXPECTED_MANIFEST_VERSION_KEYS
    assert manifest["versions"] == resolve_manifest_versions()
    for package, installed in manifest["versions"].items():
        assert isinstance(installed, str) and installed, package


def test_resolve_manifest_versions_reads_the_four_recorded_packages() -> None:
    """The hoisted seam resolves exactly the manifest's four packages, installed."""
    versions = resolve_manifest_versions()

    assert list(versions) == EXPECTED_MANIFEST_VERSION_KEYS
    for package, installed in versions.items():
        assert isinstance(installed, str) and installed, package


def test_build_manifest_records_the_versions_it_was_given() -> None:
    """Non-vacuity for the hoisted seam: the manifest records the caller's mapping.

    ``execute_roundtrip`` resolves the versions before scoring so that a
    package missing from the environment fails the run in seconds rather than
    after a multi-hour pass; that only works if ``build_manifest`` records what
    it is handed instead of re-reading the environment itself.
    """
    data, n_unique_texts = _scored_sidecar()
    supplied = {package: f"{package}-0.0.0-test" for package in EXPECTED_MANIFEST_VERSION_KEYS}

    manifest = _manifest(data, n_unique_texts, versions=supplied)

    assert manifest["versions"] == supplied
    assert manifest["versions"] is not supplied


@pytest.mark.parametrize(("backend", "device"), [("nnterp", "cuda"), ("transformerlens", "cpu")])
def test_build_manifest_records_the_runtime_backend_and_device(backend: str, device: str) -> None:
    """Non-vacuity for backend/device: they are recorded, not hard-coded."""
    data, n_unique_texts = _scored_sidecar()

    manifest = _manifest(data, n_unique_texts, backend=backend, device=device)

    assert manifest["probe"]["backend"] == backend
    assert manifest["probe"]["device"] == device


def test_build_manifest_is_json_serializable() -> None:
    """The manifest is written with ``atomic_json_dump``: numpy scalars would break it."""
    data, n_unique_texts = _scored_sidecar(blank_cells=1)

    manifest = _manifest(data, n_unique_texts)

    assert json.loads(json.dumps(manifest)) == manifest
