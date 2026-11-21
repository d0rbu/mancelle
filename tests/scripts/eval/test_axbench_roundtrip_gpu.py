"""Real-weights (GPU) tests for the axbench_upstream roundtrip metric.

Everything else in the roundtrip suite runs against stubs: the unit tests pin
the scoring rules and the driver tests pin the orchestration, both with the
probe seam faked out. That leaves one class of claim unproven -- that the
metric, wired to the ACTUAL gemma-2-2b layer-20 gemmascope-res-16k probe,
measures the thing it says it measures. These tests close that gap on real
weights and real AxBench data:

* separation -- a concept's own feature fires on that concept's positive
  examples far more often than on its negatives (the metric has signal);
* bitwise equality -- the production ``dense_max_pool_fn`` composition is the
  documented primitive, not a re-implementation of it;
* determinism -- the same texts score identically twice in one process;
* cross-check -- our number stands in the documented relation to the OTHER
  probe primitive in this repo (``probe_text_against_features``), whose only
  difference is that it pools the injected BOS row that we exclude -- shown
  both by the ``max(ours, BOS)`` identity and by a feature where the two
  genuinely disagree;
* end to end -- the CLI, on the GPU, recovers planted concept text.

Gates (both explicit skips, never silent):

* CUDA must be available -- the probe is loaded on ``cuda`` and the nnterp
  backend is REQUIRED there, which is exactly the configuration production
  scores in;
* ``AXBENCH_ROUNDTRIP_PREPARED_DIR`` must point at a prepared AxBench run
  directory, whose ``generate/latent_eval_data.parquet`` supplies the real
  per-concept positive/negative texts. That directory is READ-ONLY here.

Every test lists ``probe`` FIRST, so the CUDA gate is the primary one: a
machine without a GPU skips on the hardware it does not have, rather than
raising over a prepared-data path it was never going to use.

The probe is loaded ONCE for the whole session (a fp32 gemma-2-2b plus its
nnterp companion is ~10 GB of weights each); the end-to-end test reuses that
one load rather than putting a second copy on the same card.
"""

from __future__ import annotations

import functools
import gc
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch as t
from loguru import logger

from scripts.eval.axbench_upstream import run_roundtrip
from steering.eval.axbench_base.concept_feature_mapping import _batched_dense_sae_max_pool
from steering.eval.axbench_base.extraction import (
    _bos_prepended_model_input,
    _infer_n_sae_features,
    _run_batched_extractor,
    _sae_hook_name,
)
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_upstream import roundtrip
from steering.eval.axbench_upstream.roundtrip import (
    PROBE_SAE_LAYER,
    PROBE_TOKENIZER_ID,
    ROUNDTRIP_BATCH_SIZE,
    ROUNDTRIP_DATA_FILENAME,
    ROUNDTRIP_MANIFEST_FILENAME,
    ROUNDTRIP_SUMMARY_FILENAME,
    load_roundtrip_probe,
)
from steering.eval.probe_model import ProbeModel
from steering.eval.roundtrip_steering import probe_text_against_features
from steering.sae_resolver import ResolvedSAE
from tests.scripts.eval.test_axbench_roundtrip import (
    _position_of,
    _roundtrip_frame,
    _set_generation,
    _write_frame,
)

#: Environment variable naming a prepared AxBench run directory. Same gating
#: pattern as the adv CUDA suites: absent means "this box has no prepared data",
#: which is a SKIP with a reason, not a failure.
PREPARED_DIR_ENV = "AXBENCH_ROUNDTRIP_PREPARED_DIR"

#: The file inside it these tests read (never write).
LATENT_EVAL_RELPATH = Path("generate") / "latent_eval_data.parquet"

#: The two concepts the separation test measures, and how many texts per side.
#: Concepts 0 and 1 are the first two in the prepared file and carry DIFFERENT
#: features, so a pass is not a property of one lucky latent. Which features
#: those are is read off the prepared data, never assumed here.
SEPARATION_CONCEPTS = (0, 1)
SEPARATION_N_TEXTS = 20

#: Marker every separation measurement is logged under. Asserted present in
#: ``caplog`` so the rates stay recoverable from a PASSING run -- see
#: ``test_real_positive_negative_separation``'s docstring for the recovery
#: command. A refactor that drops the logging fails that assertion.
SEPARATION_LOG_MARKER = "roundtrip separation:"

#: Marker the cross-check logs its BOS-exclusion counterexample under, for the
#: same reason: the divergent feature is the suite's only concrete evidence
#: that excluding the BOS row changes an answer, and a passing run would
#: otherwise discard it.
CROSSCHECK_LOG_MARKER = "roundtrip bos-divergence:"

#: Separation thresholds, pinned here rather than inline so the numbers this
#: suite defends are stated once. Both are now set from OBSERVED real-weights
#: rates -- see ``test_real_positive_negative_separation``'s docstring for the
#: measurements and for why the margin deviates from the headroom rule.
SEPARATION_MARGIN = 0.3
POSITIVE_RATE_FLOOR = 0.7

#: How many texts the bitwise-equality and determinism tests score, and how
#: many the cross-check probes. Kept small: each is a real forward pass.
EQUALITY_N_TEXTS = 6
CROSSCHECK_N_TEXTS = 3

#: Relative tolerance the cross-check compares the two probe primitives at.
#: They are not the same computation -- one runs SAELens' full ``sae.encode``,
#: the other recomputes JumpReLU for a single feature column -- so they agree
#: to float error, not exactly.
CROSSCHECK_REL_TOL = 1e-3

#: How far past :data:`CROSSCHECK_REL_TOL` a BOS-vs-content gap must sit before
#: the divergence demonstration will use that feature. Without this the
#: exhibited "the two primitives disagree here" could be a gap SMALLER than the
#: tolerance the agreement above is asserted at -- i.e. indistinguishable from
#: float noise. Requiring an order of magnitude of headroom makes the
#: divergence structural: it cannot be an artifact of the comparison's own
#: slack, whatever the observed activations turn out to be.
CROSSCHECK_DIVERGENCE_FACTOR = 10.0

#: The end-to-end frame plants 5 of the first separation concept's own
#: positives and 5 unrelated texts into DiffMean's ten factor-1.0 cells.
E2E_N_PLANTED_POSITIVE = 5
E2E_N_PLANTED_TOTAL = 10

#: The end-to-end run's floor on DiffMean's factor-1.0 activation rate.
#: DERIVED, not chosen: exactly half of that (method, factor)'s ten declared
#: cells hold that concept's positives, so :data:`POSITIVE_RATE_FLOOR` -- the rate
#: those texts must clear on their own feature -- applies to half the
#: population. Tightening the separation floor tightens this with it.
E2E_ACTIVATION_RATE_FLOOR = POSITIVE_RATE_FLOOR * E2E_N_PLANTED_POSITIVE / E2E_N_PLANTED_TOTAL

#: Texts with no relationship to the AxBench concept space (concept 0 of the
#: reference prepared run is "references to rental services and associated
#: equipment"), written here rather than sampled so the negative half of the
#: end-to-end fixture is unambiguous.
E2E_UNRELATED_TEXTS = (
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "The Pythagorean theorem relates the squares of a right triangle's three side lengths.",
    "Beethoven completed his ninth symphony in 1824, years after he had lost his hearing.",
    "Antarctic ice cores preserve a record of atmospheric composition spanning many millennia.",
    "A binary search halves the remaining interval at every step of the lookup.",
)


@dataclass(frozen=True)
class _RealProbe:
    """The one real probe load, plus the geometry read off it."""

    model: Any
    sae: Any
    resolved: ResolvedSAE
    backend: str
    n_sae_features: int
    n_ctx: int

    def dense_max_pool(self, texts: list[str]) -> t.Tensor:
        """The production ``dense_max_pool_fn``, wired exactly as the run wires it.

        Spelled out as the same ``functools.partial`` over the same primitive
        that ``execute_roundtrip`` builds -- so the bitwise-equality test below
        compares the RUN's composition against the bare primitive, not a helper
        against itself.
        """
        return functools.partial(
            _batched_dense_sae_max_pool,
            base_model=self.model,
            base_sae=self.sae,
            sae_layer=PROBE_SAE_LAYER,
            tokenizer_id=PROBE_TOKENIZER_ID,
            batch_size=ROUNDTRIP_BATCH_SIZE,
        )(texts)

    def feature_max_acts(self, texts: list[str], feature: int) -> t.Tensor:
        """One concept feature's column of :meth:`dense_max_pool`."""
        return self.dense_max_pool(texts)[:, feature]


def _prepared_dir() -> Path:
    """The prepared AxBench directory, or an explicit skip naming what is missing."""
    raw = os.environ.get(PREPARED_DIR_ENV)
    if not raw:
        pytest.skip(
            f"{PREPARED_DIR_ENV} is unset: these tests score REAL AxBench "
            f"generations from <prepared>/{LATENT_EVAL_RELPATH}. Point it at a "
            "prepared AxBench run directory to enable them."
        )
    return Path(raw).expanduser()


@pytest.fixture(scope="session")
def prepared_frame() -> pd.DataFrame:
    """The prepared ``latent_eval_data`` frame (read-only input).

    A directory that IS configured but does not hold the file raises rather
    than skipping: the operator asked for this data, so a missing file is a
    misconfiguration to fix, not a reason to quietly report success.
    """
    path = _prepared_dir() / LATENT_EVAL_RELPATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{PREPARED_DIR_ENV} is set but {path} does not exist; the roundtrip "
            "GPU suite needs the prepared AxBench latent-eval generations."
        )
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def probe() -> Iterator[_RealProbe]:
    """Load the real roundtrip probe ONCE, on CUDA, and pin its precision.

    Torn down explicitly at session end: the fp32 gemma-2-2b plus its nnterp
    companion is ~20 GB of VRAM held for the whole session, and a session
    fixture's objects otherwise stay referenced until interpreter shutdown --
    long enough to make a subsequent CUDA suite in the same process fail to
    allocate. Dropping the references, collecting, and emptying torch's caching
    allocator returns the card before that happens.

    The fp32 assertions are not decoration: ``build_manifest`` records
    ``model_dtype``/``sae_dtype`` from the module constant ``_PROBE_DTYPE``,
    which is a CLAIM about what the two loaders produce. Checking the loaded
    weights here turns that recorded constant into evidence -- including for
    the nnterp companion, which is what actually runs the forward pass on
    CUDA while the TransformerLens model is parked on the CPU.

    The freeze below mirrors a PRODUCTION PRECONDITION. Every real consumer of
    ``probe_text_against_features`` builds its ``ProbeModel`` through
    ``load_probe_model`` (``steering/eval/probe_model.py:208-212``), which ends
    with ``base_model.eval()`` and ``requires_grad_(False)`` over all model and
    SAE parameters. The cross-check test below hand-builds its ``ProbeModel``
    from these objects instead -- deliberately, so a second fp32 gemma-2-2b is
    never resident on the card -- and therefore has to establish that
    precondition itself. Skipping it is not a harmless difference: with
    grad-enabled SAE parameters, that primitive records autograd over the
    inference tensors its own forward produces and raises
    ``RuntimeError: Inference tensors cannot be saved for backward``. That the
    precondition is implicit and unenforced is filed as a latent-fragility
    issue against the primitive; this fixture's job is to reproduce production,
    not to work around it.
    """
    if not t.cuda.is_available():
        pytest.skip(
            "torch.cuda.is_available() is False: the roundtrip probe is loaded on "
            "cuda (where the nnterp backend is required), so these real-weights "
            "tests need a GPU."
        )
    device = t.device("cuda")
    model, sae, resolved, backend = load_roundtrip_probe(device)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in sae.parameters():
        parameter.requires_grad_(False)

    assert backend == "nnterp"
    assert {parameter.dtype for parameter in model.parameters()} == {t.float32}
    assert sae.W_enc.dtype is t.float32
    nnterp_module = getattr(model.nnterp_model, "model", model.nnterp_model)
    assert {parameter.dtype for parameter in nnterp_module.parameters()} == {t.float32}
    # The pool reads the hook off the SAE cfg while the ProbeModel below is
    # built from the resolver's; the cross-check only compares like with like
    # if those name one activation site.
    assert _sae_hook_name(sae) == resolved.hook_name

    loaded = _RealProbe(
        model=model,
        sae=sae,
        resolved=resolved,
        backend=backend,
        # The same helper production reads ``n_sae_features`` through, so the
        # manifest assertion in the end-to-end test compares like with like.
        n_sae_features=int(_infer_n_sae_features(sae)),
        n_ctx=int(model.cfg.n_ctx),
    )
    yield loaded

    # The nnterp companion is reachable only through the attribute the loader
    # attaches, so it is dropped explicitly before the model it hangs off.
    if hasattr(model, "nnterp_model"):
        del model.nnterp_model
    del loaded, model, sae
    gc.collect()
    t.cuda.empty_cache()


def _concept_rows(frame: pd.DataFrame, concept_id: int) -> pd.DataFrame:
    rows = frame.loc[frame["concept_id"].eq(concept_id)]
    assert not rows.empty, f"prepared latent-eval data has no concept {concept_id}"
    return rows


def _concept_feature(frame: pd.DataFrame, concept_id: int) -> int:
    """The concept's own SAE feature index, taken from the prepared data itself."""
    sae_ids = _concept_rows(frame, concept_id)["sae_id"].unique()
    assert len(sae_ids) == 1, f"concept {concept_id} names {len(sae_ids)} sae_ids"
    return int(sae_ids[0])


def _category_texts(frame: pd.DataFrame, concept_id: int, category: str, count: int) -> list[str]:
    """The first ``count`` ``output`` texts of one concept/category, in file order."""
    rows = _concept_rows(frame, concept_id)
    texts = [str(text) for text in rows.loc[rows["category"].eq(category), "output"].head(count)]
    assert len(texts) == count, (
        f"concept {concept_id} has only {len(texts)} {category!r} rows, need {count}"
    )
    assert all(text.strip() for text in texts)
    return texts


def _activation_rate(probe: _RealProbe, texts: list[str], feature: int) -> float:
    """Fraction of ``texts`` whose max-pooled activation of ``feature`` is positive.

    This is the metric's own activation rule (``max_act > 0.0``, see
    ``score_frame``) applied to one batch of texts.
    """
    return float((probe.feature_max_acts(texts, feature) > 0.0).float().mean())


def _document_activations(probe: _RealProbe, text: str) -> tuple[list[int], t.Tensor]:
    """The raw ``[seq_len, n_features]`` SAE post-activations for one text.

    Composed from the SAME two primitives ``_batched_dense_sae_max_pool``
    composes -- ``_bos_prepended_model_input`` then ``_run_batched_extractor``
    with no per-item reduction -- so row 0 is the injected BOS the pool
    excludes and rows 1+ are the span it pools. Having the whole matrix is
    what lets the cross-check below reach the BOS row at all.
    """
    tokenizer = get_extraction_tokenizer(PROBE_TOKENIZER_ID)
    model_input = _bos_prepended_model_input(
        text,
        tokenizer=tokenizer,
        max_tokens=probe.n_ctx,
        n_ctx=probe.n_ctx,
        context="roundtrip GPU cross-check",
    )
    (document,) = _run_batched_extractor(
        model=probe.model,
        sae=probe.sae,
        token_id_lists=[model_input],
        layer=PROBE_SAE_LAYER,
        hook_name=probe.resolved.hook_name,
        per_item_reduce=None,
    )
    assert tuple(document.shape) == (len(model_input), probe.n_sae_features)
    return model_input, document


def _post_act(text: str, feature: int, probe_model: ProbeModel) -> float:
    """``probe_text_against_features``' post-activation for ONE feature of ONE text.

    ``post_act_mean`` is a MEAN over the requested features, so it equals that
    feature's masked token-max -- the quantity the cross-check compares against
    -- only when exactly one feature id was requested. ``feature_ids`` is
    deduplicated and sorted inside the primitive, so the count is asserted on
    the RESULT rather than trusted from the call site: the single-feature
    invariant this helper's whole meaning rests on is checked, not assumed.
    """
    scores = probe_text_against_features(
        prompt_text="",
        response_text=text,
        feature_ids=(feature,),
        probe_model=probe_model,
        percentile_table=None,
    )
    assert scores.n_features == 1, scores
    return scores.post_act_mean


def _build_probe_model(probe: _RealProbe) -> ProbeModel:
    """Wrap the ALREADY-LOADED probe as a ``ProbeModel``; never load a second one."""
    return ProbeModel(
        model=probe.model,
        sae=probe.sae,
        tokenizer=probe.model.tokenizer,
        hook_name=probe.resolved.hook_name,
        layer=PROBE_SAE_LAYER,
        nnterp_model=getattr(probe.model, "nnterp_model", None),
    )


@pytest.mark.slow
def test_real_positive_negative_separation(
    probe: _RealProbe, prepared_frame: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """The concept's own feature fires on its positives, not on its negatives.

    This is the metric's reason to exist: if a concept's feature fired equally
    on text that is and is not about the concept, every roundtrip number
    downstream would be noise. Measured on real AxBench generations, per
    concept, over 20 positives and 20 negatives each.

    OBSERVED, 2026-08-16, RTX 4090, gemma-2-2b + gemmascope layer-20 16k
    canonical, 20 texts per side. Identical to the float repr across two
    consecutive runs (one cold with the weight download, one warm), so these
    are stable measurements rather than one sample:

    ======= ======= ======== ======== ======
    concept feature pos_rate neg_rate margin
    ======= ======= ======== ======== ======
    0       16038   0.95     0.05     0.90
    1       3409    1.00     0.60     0.40
    ======= ======= ======== ======== ======

    ``POSITIVE_RATE_FLOOR`` is set to **0.7** from this: the minimum observed
    ``pos_rate`` is 0.95, so 0.7 is the largest 0.1-grid value clearing it by
    more than the required 0.2 (headroom 0.25). 0.75 would leave exactly 0.20,
    which is thin for a rate whose 20-sample resolution is 0.05.

    ``SEPARATION_MARGIN`` STAYS AT **0.3**, which is a DELIBERATE DEVIATION
    from the "largest round value with >= 0.2 headroom" rule, ruled on after
    seeing these numbers. The rule, applied mechanically to the minimum
    observed margin of 0.40, would permit at most 0.2 -- i.e. it would force
    LOOSENING an already-passing bound, inverting its own purpose (the rule
    exists to stop over-tightening, not to mandate weakening). The tighter
    proven bound is kept instead, with its cost stated plainly: **on concept 1
    the headroom is only 0.10**, so two more of its twenty negatives firing
    (0.60 -> 0.70) would fail this test.

    That concept-1 number is not noise and should not be read as one. Feature
    3409 fires on 60% of concept 1's own negatives -- expected for a concept as
    broad as "scientific terms related to research findings and their
    implications", whose negatives are same-genre text. The practical
    consequence is that the roundtrip metric's discriminative power is strongly
    CONCEPT-DEPENDENT (0.90 margin vs 0.40 across two concepts from one
    prepared run), so a pooled activation_rate across concepts should not be
    read as a uniform signal.

    RECOVERING THE RATES FROM A PASSING RUN. pytest discards captured output
    for tests that pass, so a bare ``-q`` run tells you nothing about how much
    headroom the bounds have. Two mediums carry the numbers out instead:

    * live logs -- ``uv run pytest tests/scripts/eval/test_axbench_roundtrip_gpu.py
      -m slow --log-cli-level=INFO -k separation``. This works because the test
      requests ``caplog``, and only that fixture installs the repo's
      loguru-to-stdlib bridge (``tests/conftest.py``); the plain ``logger.info``
      of an earlier revision went to stderr and was swallowed. ``-s`` also
      works, unfiltered.
    * the assertion messages below, which carry the whole ``observed`` mapping,
      so a run that FAILS a bound reports every rate rather than only the one
      that tripped.

    The ``caplog`` assertion at the end of the body is what keeps the first
    medium honest: delete the logging and this test fails.
    """
    observed: dict[int, tuple[float, float]] = {}
    for concept_id in SEPARATION_CONCEPTS:
        feature = _concept_feature(prepared_frame, concept_id)
        positive = _category_texts(prepared_frame, concept_id, "positive", SEPARATION_N_TEXTS)
        negative = _category_texts(prepared_frame, concept_id, "negative", SEPARATION_N_TEXTS)
        observed[concept_id] = (
            _activation_rate(probe, positive, feature),
            _activation_rate(probe, negative, feature),
        )
        logger.info(
            "{} concept {} feature {} pos_rate={} neg_rate={} (n={} per side)",
            SEPARATION_LOG_MARKER,
            concept_id,
            feature,
            observed[concept_id][0],
            observed[concept_id][1],
            SEPARATION_N_TEXTS,
        )

    for concept_id, (positive_rate, negative_rate) in observed.items():
        assert positive_rate > POSITIVE_RATE_FLOOR, (
            f"concept {concept_id} pos_rate={positive_rate} does not clear "
            f"POSITIVE_RATE_FLOOR={POSITIVE_RATE_FLOOR}; observed "
            f"{{concept: (pos_rate, neg_rate)}} = {observed}"
        )
        assert positive_rate >= negative_rate + SEPARATION_MARGIN, (
            f"concept {concept_id} margin={positive_rate - negative_rate} does not "
            f"clear SEPARATION_MARGIN={SEPARATION_MARGIN}; observed "
            f"{{concept: (pos_rate, neg_rate)}} = {observed}"
        )

    # Every measurement reached the loguru bridge, so --log-cli-level=INFO (or
    # -s) recovers the rates from a run where the bounds all held. Without this
    # the documented tightening procedure would silently stop working the first
    # time someone removed the logging call.
    assert caplog.text.count(SEPARATION_LOG_MARKER) == len(SEPARATION_CONCEPTS), caplog.text
    for concept_id in SEPARATION_CONCEPTS:
        assert f"pos_rate={observed[concept_id][0]}" in caplog.text


@pytest.mark.slow
def test_real_bitwise_equality_vs_oracle(probe: _RealProbe, prepared_frame: pd.DataFrame) -> None:
    """The production composition IS the documented primitive, bit for bit.

    ``execute_roundtrip`` scores through a ``functools.partial`` that binds
    ``_batched_dense_sae_max_pool``'s model, SAE, layer, tokenizer and batch
    size, leaving ``texts`` as its only free argument; the metric's contract is
    that this equals the bare primitive called with those same values.
    :meth:`_RealProbe.dense_max_pool` builds that partial identically, so on
    real weights the two must agree EXACTLY -- an approximate agreement would
    mean the composition changed something (a dtype, a batch boundary, a hook)
    that the stub suites cannot see.
    """
    feature = _concept_feature(prepared_frame, SEPARATION_CONCEPTS[0])
    texts = [
        *_category_texts(prepared_frame, SEPARATION_CONCEPTS[0], "positive", EQUALITY_N_TEXTS // 2),
        *_category_texts(prepared_frame, SEPARATION_CONCEPTS[0], "negative", EQUALITY_N_TEXTS // 2),
    ]

    ours = probe.feature_max_acts(texts, feature)
    oracle = _batched_dense_sae_max_pool(
        texts,
        probe.model,
        probe.sae,
        sae_layer=PROBE_SAE_LAYER,
        tokenizer_id=PROBE_TOKENIZER_ID,
        batch_size=ROUNDTRIP_BATCH_SIZE,
    )[:, feature]

    assert t.equal(ours, oracle)
    # Non-vacuity: a column of zeros would satisfy t.equal above.
    assert bool((ours > 0.0).any())


@pytest.mark.slow
def test_real_determinism(probe: _RealProbe, prepared_frame: pd.DataFrame) -> None:
    """The same texts score identically twice in one process.

    Roundtrip numbers are compared across runs and recorded in a manifest that
    claims reproducibility; a probe whose GPU forward drifted between calls
    would make every such comparison meaningless.
    """
    feature = _concept_feature(prepared_frame, SEPARATION_CONCEPTS[0])
    texts = [
        *_category_texts(prepared_frame, SEPARATION_CONCEPTS[0], "positive", EQUALITY_N_TEXTS // 2),
        *_category_texts(prepared_frame, SEPARATION_CONCEPTS[0], "negative", EQUALITY_N_TEXTS // 2),
    ]

    first = probe.feature_max_acts(texts, feature)
    second = probe.feature_max_acts(texts, feature)

    assert t.equal(first, second)
    assert bool((first > 0.0).any())


@pytest.mark.slow
def test_real_crosscheck_vs_probe_text_primitive(
    probe: _RealProbe, prepared_frame: pd.DataFrame
) -> None:
    """Our number relates to the repo's OTHER probe primitive exactly as documented.

    ``probe_text_against_features`` max-pools over EVERY token of its input,
    and with ``prompt_text=""`` its completion mask is all-True -- including
    the tokenizer's leading BOS. The roundtrip metric deliberately excludes
    that row (``acts[1:]``, the ``extract_features`` convention: BOS is an
    attention-sink outlier). So the two are not independent measurements of
    the same quantity; they stand in an exact relation::

        old_value == max(our_max_act, bos_post_act)

    which is what this test asserts, per text, from the raw per-token matrix
    (so ``our_max_act`` is simultaneously re-verified against the raw
    ``_bos_prepended_model_input`` + ``_run_batched_extractor`` composition
    that the production pool is built from).

    The relation is only meaningful if a feature exists where the two
    genuinely disagree -- otherwise ``max(ours, bos)`` would just be ``ours``
    and the test would pass on a metric that never excluded BOS at all. The
    second half therefore exhibits one: a feature whose BOS activation exceeds
    everything in the pooled span, where the old primitive reports the BOS
    value and ours reports the smaller content value. That feature is picked
    by a gap that clears :data:`CROSSCHECK_REL_TOL` by
    :data:`CROSSCHECK_DIVERGENCE_FACTOR`, so "these two disagree" cannot be
    satisfied by a difference smaller than the tolerance the agreement itself
    is asserted at.
    """
    concept_id = SEPARATION_CONCEPTS[0]
    feature = _concept_feature(prepared_frame, concept_id)
    texts = _category_texts(prepared_frame, concept_id, "positive", CROSSCHECK_N_TEXTS)
    probe_model = _build_probe_model(probe)

    divergences: list[tuple[int, float, float]] = []
    for text in texts:
        model_input, document = _document_activations(probe, text)
        # The relation below is an identity over ONE token sequence, so the
        # two primitives must be reading the same one: ours injects a BOS
        # after an ``add_special_tokens=False`` encode, the old one lets the
        # tokenizer prepend it.
        assert probe_model.tokenizer.encode(text) == model_input

        bos_post_act = float(document[0, feature])
        our_max_act = float(document[1:, feature].max())
        assert our_max_act == float(probe.feature_max_acts([text], feature)[0])

        old_value = _post_act(text, feature, probe_model)
        assert old_value == pytest.approx(max(our_max_act, bos_post_act), rel=CROSSCHECK_REL_TOL)

        # A feature whose BOS row dominates its whole pooled span: the case
        # the BOS exclusion exists for. The gap must clear the tolerance the
        # agreement above is asserted at, by CROSSCHECK_DIVERGENCE_FACTOR, so
        # the disagreement exhibited below is structurally larger than the
        # comparison's own slack rather than merely observed to be.
        content_max = document[1:].max(dim=0).values
        gap = document[0] - content_max
        tolerance_band = (
            CROSSCHECK_DIVERGENCE_FACTOR
            * CROSSCHECK_REL_TOL
            * t.maximum(document[0].abs(), content_max.abs())
        )
        candidates = t.nonzero(gap > tolerance_band).flatten()
        if candidates.numel() == 0:
            continue
        divergent = int(candidates[int(t.argmax(gap[candidates]))])
        bos_value = float(document[0, divergent])
        content_value = float(content_max[divergent])
        divergences.append((divergent, bos_value, content_value))
        old_divergent = _post_act(text, divergent, probe_model)
        assert old_divergent == pytest.approx(bos_value, rel=CROSSCHECK_REL_TOL)
        assert old_divergent > content_value, (divergent, old_divergent, content_value)
        # Logged for the same reason the separation rates are: this is the one
        # concrete BOS-exclusion counterexample the suite produces, and a
        # passing run would otherwise discard it. Recover with ``-s`` /
        # ``--capture=no``: this test does not request ``caplog``, so the
        # loguru->stdlib bridge is absent and ``--log-cli-level`` shows nothing.
        logger.info(
            "{} feature {} bos_post_act={} content_max={} gap={} old_value={} text={!r}",
            CROSSCHECK_LOG_MARKER,
            divergent,
            bos_value,
            content_value,
            bos_value - content_value,
            old_divergent,
            text[:60],
        )

    assert divergences, (
        "no feature had a BOS activation exceeding its whole pooled span by more "
        f"than {CROSSCHECK_DIVERGENCE_FACTOR} x {CROSSCHECK_REL_TOL}, so the "
        "max(ours, bos) relation above never distinguished BOS-exclusion from "
        "plain max-pooling"
    )


@pytest.mark.slow
def test_real_end_to_end_cli(
    probe: _RealProbe,
    prepared_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI, on the GPU, recovers planted concept text through the real probe.

    The concept's ten DiffMean factor-1.0 generations are replaced by five of
    its own positive examples and five texts about photosynthesis, geometry,
    Beethoven, ice cores and binary search. The merged fixture frame is built
    around the feature the PREPARED DATA names for that concept, rather than
    around the helper's default index, so the run scores those ten texts
    through the latent they were actually written for on whatever prepared run
    is configured -- and so a prepared run with a different feature map
    exercises this test instead of failing an incidental identity check.

    The one real probe load is reused rather than repeated: a second fp32
    gemma-2-2b plus nnterp companion would not co-reside with this one on a
    24 GB card. Everything else -- argument parsing, path policy, reading and
    validating the merged frame, the per-concept scoring passes, the summary,
    and all three artifacts -- runs for real.
    """
    concept_id = SEPARATION_CONCEPTS[0]
    feature = _concept_feature(prepared_frame, concept_id)
    frame = _roundtrip_frame(max_concepts=1, feature_ids=(feature,))

    planted = [
        *_category_texts(prepared_frame, concept_id, "positive", E2E_N_PLANTED_POSITIVE),
        *E2E_UNRELATED_TEXTS,
    ]
    assert len(planted) == E2E_N_PLANTED_TOTAL
    for input_id, text in enumerate(planted):
        position = _position_of(frame, concept_id=0, input_id=input_id, factor=1.0)
        _set_generation(frame, position, "DiffMean", text)
    input_path = _write_frame(frame, tmp_path / "merged" / "steering_data.parquet")
    output_dir = tmp_path / "roundtrip"

    load_devices: list[t.device] = []

    def reuse_session_probe(device: t.device) -> tuple[Any, Any, ResolvedSAE, str]:
        load_devices.append(device)
        return probe.model, probe.sae, probe.resolved, probe.backend

    monkeypatch.setattr(roundtrip, "load_roundtrip_probe", reuse_session_probe)

    exit_code = run_roundtrip.main(
        ["--input", str(input_path), "--output-dir", str(output_dir), "--device", "cuda"]
    )

    assert exit_code == 0
    assert load_devices == [t.device("cuda")]
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        (ROUNDTRIP_DATA_FILENAME, ROUNDTRIP_MANIFEST_FILENAME, ROUNDTRIP_SUMMARY_FILENAME)
    )

    manifest = json.loads((output_dir / ROUNDTRIP_MANIFEST_FILENAME).read_text())
    assert manifest["probe"]["backend"] == "nnterp"
    assert manifest["probe"]["device"] == "cuda"
    assert manifest["probe"]["model_dtype"] == "float32"
    assert manifest["probe"]["sae_dtype"] == "float32"
    assert manifest["probe"]["n_sae_features"] == probe.n_sae_features

    summary = pd.read_parquet(output_dir / ROUNDTRIP_SUMMARY_FILENAME)
    row = summary.loc[summary["method"].eq("DiffMean") & summary["factor"].eq(1.0)].iloc[0]
    assert int(row["n_declared"]) == E2E_N_PLANTED_TOTAL
    assert int(row["n_blank"]) == 0
    assert float(row["activation_rate"]) >= E2E_ACTIVATION_RATE_FLOOR

    # WHICH cells fired, not just how many: a probe that fired on every text
    # would clear the pooled rate bound above while measuring nothing. The two
    # halves are compared at the same separation margin the dedicated
    # separation test uses, so this fails on an indiscriminate probe.
    data = pd.read_parquet(output_dir / ROUNDTRIP_DATA_FILENAME)
    activated = {
        int(input_id): bool(flag)
        for input_id, factor, flag in zip(
            data["input_id"], data["factor"], data["DiffMean_roundtrip_activated"], strict=True
        )
        if float(factor) == 1.0
    }
    assert len(activated) == E2E_N_PLANTED_TOTAL
    planted_rate = sum(activated[i] for i in range(E2E_N_PLANTED_POSITIVE)) / (
        E2E_N_PLANTED_POSITIVE
    )
    unrelated_rate = sum(
        activated[i] for i in range(E2E_N_PLANTED_POSITIVE, E2E_N_PLANTED_TOTAL)
    ) / (E2E_N_PLANTED_TOTAL - E2E_N_PLANTED_POSITIVE)
    logger.info(
        "roundtrip end-to-end: DiffMean factor 1.0 activation_rate={} planted={} unrelated={}",
        float(row["activation_rate"]),
        planted_rate,
        unrelated_rate,
    )
    assert planted_rate >= unrelated_rate + SEPARATION_MARGIN, (planted_rate, unrelated_rate)
