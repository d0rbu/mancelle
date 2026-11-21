"""Mutation-focused coverage for clustered general-val orchestration (G7 + G8b)."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest
import torch as t
from safetensors import safe_open

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_base.types import Concept, SteeringMode
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import ScoringInputs
from steering.eval.axbench_bpb.vocab_restriction import (
    UNCOMPACTED_VOCAB_FINGERPRINT,
)
from steering.eval.axbench_bpb_adv import general_val
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    RawClusteredFeatureSegment,
    RawClusteredPrefix,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    GeneralValBpbConfig,
    OursGcgConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.general_val import (
    _GeneralValDoc,
    _InterventionCombo,
    _RooflineGeneralValState,
)
from steering.eval.axbench_bpb_adv.general_val_roofline import (
    GeneralValSignalBudget,
    general_val_bag_identity_metadata,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    FEATURE_CONTRACT_FINGERPRINT_META_KEY,
    VOCAB_FINGERPRINT_META_KEY,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    GENERAL_VAL_BPB_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from tests.statistics_doubles import fixed_statistics_double

_FINGERPRINT_A = "a" * 64
_FINGERPRINT_B = "b" * 64
_VOCAB = FeatureVocab(n_sae_features=16, original_ids=(1, 3, 7))
_FAMILY = PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN
_CORPUS = "nd=1;nb=1;h=0123456789abcdef0123456789abcdef"
_OURS = OursGcgConfig(max_steps=8, minibatch_size=16, master_seed=0, max_forward_batch=8)
_BUDGET = GeneralValSignalBudget(
    k_pos=1,
    k_neg=1,
    positive_bucket=0,
    negative_bucket=1,
)


class _NoOpConsolidation:
    def assert_representatives_only(self, ids: t.Tensor, *, surface: str) -> None:
        assert ids.dtype == t.int64, f"{surface} must validate int64 original IDs"


class _NoOpMask:
    def assert_no_masked(self, ids: t.Tensor) -> None:
        assert ids.dtype == t.int64


def _contract(
    fingerprint: str = _FINGERPRINT_A,
    *,
    feature_vocab: FeatureVocab | None = _VOCAB,
    consolidation: Any | None = None,
    feature_mask: Any | None = None,
) -> Any:
    return SimpleNamespace(
        feature_contract_fingerprint=fingerprint,
        consolidation=consolidation or _NoOpConsolidation(),
        feature_mask=feature_mask or _NoOpMask(),
        checkpoint=SimpleNamespace(feature_vocab=feature_vocab),
        feature_directory=SimpleNamespace(segment_size=128),
        selection=SimpleNamespace(num_buckets=2),
    )


def _doc(sample_id: str = "general_val_0") -> _GeneralValDoc:
    return _GeneralValDoc(
        sample_id=sample_id,
        prefix_tokens=[1, 2],
        output_tokens=[3],
        prefix_text="prefix",
        output_byte_weights=t.ones(1, dtype=t.int64),
    )


def _concept() -> Concept:
    return Concept(
        concept_id=1,
        feature_index=3,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="concept",
        corpus_count=10,
        description_source="test",
    )


def _bundle(
    model: object,
    *,
    max_seq_len: int = 64,
) -> TargetModelBundle:
    return TargetModelBundle(
        model=model,
        tokenizer=object(),
        max_seq_len=max_seq_len,
        target_name="encdec",
    )


def _scoring_input(doc: _GeneralValDoc) -> ScoringInputs:
    return ScoringInputs(
        sample_id=doc.sample_id,
        concept_id=-1,
        mode=SteeringMode.POSITIVE,
        content_class="general_val",
        prefix_text=doc.prefix_text,
        prefix_hash="general-val",
        prefix_tokens=doc.prefix_tokens,
        output_tokens=doc.output_tokens,
        prefix_context_segments=(),
        n_prefix=len(doc.prefix_tokens),
        n_output=len(doc.output_tokens),
        output_byte_weights=doc.output_byte_weights,
        truncated=False,
    )


def _model(vocab: FeatureVocab | None = _VOCAB) -> Any:
    return SimpleNamespace(
        feature_vocab=vocab,
        encoder=(
            None
            if vocab is None
            else SimpleNamespace(feature_vocab_ids=vocab.new_to_old_tensor())
        ),
    )


def _identity_metadata(
    *,
    doc: _GeneralValDoc,
    contract: Any | None,
) -> dict[str, str]:
    return general_val_bag_identity_metadata(
        sample_id=doc.sample_id,
        family=_FAMILY.value,
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_BUDGET,
        config_metadata={"fixture": "g7-g8b"},
        cluster_contract=contract,
    )


def _load(
    *,
    paths: AxBenchBpbAdvPaths,
    doc: _GeneralValDoc,
    contract: Any | None,
) -> FeatureConditioning | None:
    contract_kwargs = (
        {} if contract is None else {"cluster_contract": contract}
    )
    return general_val._load_persisted_roofline_bag(
        paths=paths,
        doc=doc,
        family=_FAMILY,
        reg_lambda=0.0,
        corpus_signature=_CORPUS,
        ours=_OURS,
        signal_budget=_BUDGET,
        config_metadata={"fixture": "g7-g8b"},
        model=_model(),
        **contract_kwargs,
    )


def test_clustered_roofline_resume_hit_miss_drift_and_exact_stamp(tmp_path: Path) -> None:
    """A clustered cache hit is one stamped read; miss and contract drift stay loud."""

    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="clustered")
    doc = _doc()
    contract = _contract()
    bag = FeatureConditioning.from_dict({1: 0, 7: 1})
    general_val._write_roofline_bag(
        paths=paths,
        doc=doc,
        family=_FAMILY,
        reg_lambda=0.0,
        bag=bag,
        metadata=_identity_metadata(doc=doc, contract=contract),
        vocab_fingerprint=_VOCAB.fingerprint(),
        cluster_contract=contract,
    )
    path = paths.general_val_bag(doc.sample_id, _FAMILY, 0.0)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    assert metadata[FEATURE_CONTRACT_FINGERPRINT_META_KEY] == _FINGERPRINT_A

    loaded = _load(paths=paths, doc=doc, contract=contract)
    assert loaded == bag
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert _load(paths=paths, doc=_doc("general_val_missing"), contract=contract) is None

    with pytest.raises(
        AxBenchBpbAdvError,
        match="precomputed bag .* feature-contract fingerprint mismatch",
    ):
        _load(paths=paths, doc=doc, contract=_contract(_FINGERPRINT_B))


def test_unclustered_roofline_bag_roundtrip_preserves_legacy_metadata(
    tmp_path: Path,
) -> None:
    """The legacy writer stays unstamped by cluster identity and round-trips."""

    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="unclustered-golden")
    doc = _doc()
    bag = FeatureConditioning.from_dict({1: 0, 7: 1})
    general_val._write_roofline_bag(
        paths=paths,
        doc=doc,
        family=_FAMILY,
        reg_lambda=0.0,
        bag=bag,
        metadata=_identity_metadata(doc=doc, contract=None),
        vocab_fingerprint=_VOCAB.fingerprint(),
    )
    path = paths.general_val_bag(doc.sample_id, _FAMILY, 0.0)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    assert FEATURE_CONTRACT_FINGERPRINT_META_KEY not in metadata
    assert metadata[VOCAB_FINGERPRINT_META_KEY] == _VOCAB.fingerprint()
    assert _load(paths=paths, doc=doc, contract=None) == bag


def test_unclustered_reader_rejects_cluster_stamped_roofline_bag(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="cluster-to-plain")
    doc = _doc()
    contract = _contract()
    general_val._write_roofline_bag(
        paths=paths,
        doc=doc,
        family=_FAMILY,
        reg_lambda=0.0,
        bag=FeatureConditioning.from_dict({1: 0}),
        metadata=_identity_metadata(doc=doc, contract=contract),
        vocab_fingerprint=_VOCAB.fingerprint(),
        cluster_contract=contract,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="unclustered reader rejects cluster-stamped precomputed bag",
    ):
        _load(paths=paths, doc=doc, contract=None)


def test_clustered_reader_rejects_unstamped_roofline_bag(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="plain-to-cluster")
    doc = _doc()
    general_val._write_roofline_bag(
        paths=paths,
        doc=doc,
        family=_FAMILY,
        reg_lambda=0.0,
        bag=FeatureConditioning.from_dict({1: 0}),
        metadata=_identity_metadata(doc=doc, contract=None),
        vocab_fingerprint=_VOCAB.fingerprint(),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="feature-contract fingerprint mismatch.*actual=None",
    ):
        _load(paths=paths, doc=doc, contract=_contract())


def test_uncompacted_vocab_path_still_checks_cluster_stamp() -> None:
    """The legacy-vocab allowance must not early-return around clustered identity."""

    with pytest.raises(
        AxBenchBpbAdvError,
        match="roofline general_val bag feature-contract fingerprint mismatch",
    ):
        general_val._assert_roofline_bag_matches_checkpoint_vocab(
            {},
            model=_model(None),
            bag=FeatureConditioning.from_dict({1: 0}),
            doc=_doc(),
            family=_FAMILY,
            reg_lambda=0.0,
            path="/tmp/unstamped",
            cluster_contract=_contract(),
        )


def test_unclustered_vocab_assertion_rejects_cluster_stamp_directly() -> None:
    """The defense-in-depth assertion names the unclustered/clustered mismatch."""

    with pytest.raises(
        AxBenchBpbAdvError,
        match="unclustered general_val reader rejects a cluster-stamped roofline bag",
    ):
        general_val._assert_roofline_bag_matches_checkpoint_vocab(
            {
                VOCAB_FINGERPRINT_META_KEY: _VOCAB.fingerprint(),
                FEATURE_CONTRACT_FINGERPRINT_META_KEY: _FINGERPRINT_A,
            },
            model=_model(),
            bag=FeatureConditioning.from_dict({1: 0}),
            doc=_doc(),
            family=_FAMILY,
            reg_lambda=0.0,
            path="/tmp/cluster-stamped",
        )


@pytest.mark.parametrize(
    ("contract_factory", "bag", "message"),
    [
        (
            lambda: _contract(
                consolidation=SimpleNamespace(
                    assert_representatives_only=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        ValueError("non-representative id 7")
                    )
                )
            ),
            FeatureConditioning.from_dict({7: 0}),
            "persisted bag violates clustered target semantics: non-representative id 7",
        ),
        (
            lambda: _contract(
                feature_mask=SimpleNamespace(
                    assert_no_masked=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        ValueError("masked id 7")
                    )
                )
            ),
            FeatureConditioning.from_dict({7: 0}),
            "persisted bag violates clustered target semantics: masked id 7",
        ),
        (
            lambda: _contract(feature_vocab=None),
            FeatureConditioning.from_dict({1: 0}),
            "requires a persistent checkpoint FeatureVocab",
        ),
        (
            lambda: _contract(
                feature_vocab=FeatureVocab(n_sae_features=16, original_ids=(1, 3))
            ),
            FeatureConditioning.from_dict({7: 0}),
            "representative IDs outside the exact checkpoint vocabulary: \\[7\\]",
        ),
    ],
)
def test_persisted_clustered_bag_fail_loud_matrix_names_property(
    contract_factory: Any,
    bag: FeatureConditioning,
    message: str,
) -> None:
    contract = contract_factory()
    vocab = contract.checkpoint.feature_vocab
    model = _model(vocab if vocab is not None else _VOCAB)
    metadata = {
        VOCAB_FINGERPRINT_META_KEY: (
            model.feature_vocab.fingerprint()
            if model.feature_vocab is not None
            else UNCOMPACTED_VOCAB_FINGERPRINT
        ),
        FEATURE_CONTRACT_FINGERPRINT_META_KEY: contract.feature_contract_fingerprint,
    }
    with pytest.raises(AxBenchBpbAdvError, match=message):
        general_val._assert_roofline_bag_matches_checkpoint_vocab(
            metadata,
            model=model,
            bag=bag,
            doc=_doc(),
            family=_FAMILY,
            reg_lambda=0.0,
            path="/tmp/fail-loud",
            cluster_contract=contract,
        )


# ``_extract_encdec_segments`` takes its snapshot from the caller: production
# hands it the exact object the stage already validated, never a re-read file.
_STATISTICS = fixed_statistics_double()


def _extraction_cfg() -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    cfg.context_cache.segment_size = 128
    cfg.context_cache.num_buckets = 2
    cfg.extraction_threshold = 0.0
    return cfg


def _install_clustered_extraction_stubs(monkeypatch: pytest.MonkeyPatch) -> Any:
    runtime_calls: list[int | None] = []

    def base_gemma_and_sae(
        *, offload_transformer_tail_after: int | None = None
    ) -> tuple[None, Any]:
        runtime_calls.append(offload_transformer_tail_after)
        return None, SimpleNamespace(cfg=SimpleNamespace(d_sae=16))

    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda _path: pytest.fail("extraction must use the caller's statistics"),
    )
    monkeypatch.setattr(
        general_val,
        "build_boundaries_from_stats",
        lambda *_args, **_kwargs: t.zeros((16, 1), dtype=t.float64),
    )
    monkeypatch.setattr(
        general_val,
        "reconstruct_clustered_replay_state",
        lambda _contract: object(),
    )
    monkeypatch.setattr(
        general_val,
        "extract_raw_clustered_prefix",
        lambda **_kwargs: RawClusteredPrefix(
            segments=(
                RawClusteredFeatureSegment(
                    feature_ids=t.tensor([1], dtype=t.int64),
                    feature_magnitudes=t.tensor([1.0]),
                    char_start=0,
                    char_end=6,
                ),
            ),
            n_tokens=2,
        ),
    )
    monkeypatch.setattr(
        general_val,
        "select_raw_clustered_segment",
        lambda *_args, **_kwargs: SimpleNamespace(
            positive_ids=t.tensor([1], dtype=t.int64),
            negative_ids=t.tensor([], dtype=t.int64),
            positive_buckets=t.tensor([0], dtype=t.int64),
        ),
    )
    return SimpleNamespace(
        base_gemma_and_sae=base_gemma_and_sae,
        runtime_calls=runtime_calls,
    )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("corpus_signature", None, "requires the exact generic corpus signature"),
        ("segment_size", 64, "segment size does not match"),
        ("num_buckets", 3, "bucket count does not match"),
        ("extraction_threshold", 0.1, "requires extraction_threshold=0.0"),
    ],
)
def test_clustered_extraction_fail_loud_guards_and_single_field_restore(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: Any,
    message: str,
) -> None:
    runtime = _install_clustered_extraction_stubs(monkeypatch)
    cfg = _extraction_cfg()
    kwargs: dict[str, Any] = {
        "runtime_state": runtime,
        "cfg": cfg,
        "statistics": _STATISTICS,
        "cluster_contract": _contract(),
        "corpus_signature": "corpus",
    }
    if field == "corpus_signature":
        kwargs[field] = invalid
    elif field == "segment_size":
        cfg.context_cache.segment_size = invalid
    elif field == "num_buckets":
        cfg.context_cache.num_buckets = invalid
    else:
        cfg.extraction_threshold = invalid

    with pytest.raises(AxBenchBpbAdvError, match=message):
        general_val._extract_encdec_segments([_doc()], **kwargs)

    kwargs["corpus_signature"] = "corpus"
    cfg.context_cache.segment_size = 128
    cfg.context_cache.num_buckets = 2
    cfg.extraction_threshold = 0.0
    restored = general_val._extract_encdec_segments([_doc()], **kwargs)
    assert restored == {
        "general_val_0": (ContextFeatureSegment((1,), (0,), 0, 6),)
    }, f"restoring only {field} must make the clustered extraction contract valid"


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (
            _contract(
                consolidation=SimpleNamespace(
                    assert_representatives_only=lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(ValueError("context contains a member id"))
                )
            ),
            "context contains a member id",
        ),
        (
            _contract(
                feature_mask=SimpleNamespace(
                    assert_no_masked=lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(ValueError("context contains a masked id"))
                )
            ),
            "context contains a masked id",
        ),
    ],
)
def test_clustered_context_guards_each_fail_at_the_named_property(
    monkeypatch: pytest.MonkeyPatch,
    contract: Any,
    message: str,
) -> None:
    runtime = _install_clustered_extraction_stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        general_val._extract_encdec_segments(
            [_doc()],
            runtime_state=runtime,
            cfg=_extraction_cfg(),
            statistics=_STATISTICS,
            cluster_contract=contract,
            corpus_signature=_CORPUS,
        )


def test_clustered_extraction_requires_reconstructed_replay_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_clustered_extraction_stubs(monkeypatch)
    monkeypatch.setattr(
        general_val,
        "reconstruct_clustered_replay_state",
        lambda _contract: None,
    )
    with pytest.raises(
        AssertionError,
        match="clustered general_val extraction requires reconstructed replay state",
    ):
        general_val._extract_encdec_segments(
            [_doc()],
            runtime_state=runtime,
            cfg=_extraction_cfg(),
            statistics=_STATISTICS,
            cluster_contract=_contract(),
            corpus_signature=_CORPUS,
        )


def test_unclustered_extraction_never_reaches_cluster_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _extraction_cfg()
    calls: list[str] = []
    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda _path: pytest.fail("extraction must use the caller's statistics"),
    )
    monkeypatch.setattr(
        general_val,
        "build_boundaries_from_stats",
        lambda *_args, **_kwargs: t.zeros((16, 1), dtype=t.float64),
    )
    monkeypatch.setattr(
        general_val,
        "extract_prefix_features",
        lambda **_kwargs: calls.append("legacy") or (ContextFeatureSegment((1,), (0,), 0, 6),),
    )
    monkeypatch.setattr(
        general_val,
        "reconstruct_clustered_replay_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unclustered extraction must never reconstruct clustered replay")
        ),
    )
    runtime_calls: list[dict[str, Any]] = []

    def base_gemma_and_sae(**kwargs: Any) -> tuple[None, Any]:
        runtime_calls.append(kwargs)
        return None, SimpleNamespace(cfg=SimpleNamespace(d_sae=16))

    runtime = SimpleNamespace(base_gemma_and_sae=base_gemma_and_sae)

    result = general_val._extract_encdec_segments(
        [_doc()],
        runtime_state=cast(RuntimeState, runtime),
        cfg=cfg,
        statistics=_STATISTICS,
    )
    assert result["general_val_0"][0].feature_ids == (1,)
    assert calls == ["legacy"]
    assert runtime_calls == [{}], "legacy extraction must use the unmodified model-load call"


def test_clustered_extraction_threads_contract_into_roofline_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_clustered_extraction_stubs(monkeypatch)
    cfg = _extraction_cfg()
    contract = _contract()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        general_val,
        "build_percentile_table",
        lambda *_args, **_kwargs: t.zeros((16, 99), dtype=t.float64),
    )

    def derive(doc: _GeneralValDoc, **kwargs: Any) -> FeatureConditioning:
        captured["doc"] = doc
        captured.update(kwargs)
        return FeatureConditioning.from_dict({1: 0})

    monkeypatch.setattr(general_val, "derive_real_output_signal", derive)
    state = _RooflineGeneralValState.for_budget(
        GeneralValSignalBudget(1, 0, 0, 1)
    )
    result = general_val._extract_encdec_segments(
        [_doc()],
        runtime_state=runtime,
        cfg=cfg,
        statistics=_STATISTICS,
        roofline_state=state,
        tokenizer=object(),
        device=t.device("cpu"),
        cluster_contract=contract,
        corpus_signature=_CORPUS,
    )
    assert result["general_val_0"][0].feature_ids == (1,)
    assert captured["cluster_contract"] is contract
    assert captured["corpus_signature"] == _CORPUS
    assert state.real_bags["general_val_0"] == FeatureConditioning.from_dict({1: 0})


def test_target_memory_handoff_rejects_unrestorable_bundle_name() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="cannot restore unsupported target bundle.*mystery",
    ):
        general_val._extract_with_target_memory_handoff(
            runtime_state=cast(RuntimeState, SimpleNamespace()),
            target_bundles=cast(
                dict[str, TargetModelBundle],
                {"mystery": object()},
            ),
            extract=dict,
        )


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (
            _contract(
                consolidation=SimpleNamespace(
                    assert_representatives_only=lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(ValueError("writer contains a member id"))
                )
            ),
            "writer contains a member id",
        ),
        (
            _contract(
                feature_mask=SimpleNamespace(
                    assert_no_masked=lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(ValueError("writer contains a masked id"))
                )
            ),
            "writer contains a masked id",
        ),
    ],
)
def test_clustered_roofline_writer_guards_each_fail_at_the_named_property(
    tmp_path: Path,
    contract: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        general_val._write_roofline_bag(
            paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="writer-guard"),
            doc=_doc(),
            family=_FAMILY,
            reg_lambda=0.0,
            bag=FeatureConditioning.from_dict({1: 0}),
            metadata=_identity_metadata(doc=_doc(), contract=contract),
            vocab_fingerprint=_VOCAB.fingerprint(),
            cluster_contract=contract,
        )


@pytest.mark.parametrize("clustered", [False, True])
def test_roofline_signal_budget_dispatches_contract_to_both_resolvers(
    monkeypatch: pytest.MonkeyPatch,
    clustered: bool,
) -> None:
    contract = _contract() if clustered else None
    calls: list[tuple[str, dict[str, Any]]] = []

    def buckets(_model: Any, **kwargs: Any) -> tuple[int, int]:
        calls.append(("buckets", kwargs))
        return 0, 1

    def counts(_model: Any, _ours: Any, **kwargs: Any) -> tuple[int, int]:
        calls.append(("counts", kwargs))
        return 2, 3

    monkeypatch.setattr(general_val, "assert_single_positive_bucket", buckets)
    monkeypatch.setattr(general_val, "resolve_k", counts)
    cfg = _extraction_cfg()
    budget = general_val._resolve_roofline_signal_budget(
        target_bundle=_bundle(object()),
        cfg=cfg,
        cluster_contract=contract,
    )
    expected = {} if contract is None else {"cluster_contract": contract}
    assert calls == [("buckets", expected), ("counts", expected)]
    assert budget == GeneralValSignalBudget(2, 3, 0, 1)


@pytest.mark.parametrize("clustered", [False, True])
def test_combo_intervention_dispatches_contract_only_when_clustered(
    monkeypatch: pytest.MonkeyPatch,
    clustered: bool,
) -> None:
    contract = _contract() if clustered else None
    captured: dict[str, Any] = {}
    sentinel = object()

    def build(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(general_val, "build_intervention", build)
    monkeypatch.setattr(
        general_val,
        "apply_additive_counter",
        lambda base, **_kwargs: base,
    )
    combo = _InterventionCombo(
        target_model="encdec",
        method_name="diffmean_encdec",
        mode="positive",
        concept_id=1,
        coefficient_str="1.0",
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )
    result = general_val._build_combo_intervention(
        combo=combo,
        method=object(),
        concept=_concept(),
        source_paths=SourcePaths(Path("source")),
        diffmean_layer=6,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        counter_direction_family="diffmean",
        apply_max_act_scale=False,
        cluster_contract=contract,
    )
    assert result is sentinel
    if contract is None:
        assert "cluster_contract" not in captured
    else:
        assert captured["cluster_contract"] is contract


def test_clustered_combo_threads_certified_direction_to_base_and_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"certified"
    captured_base: dict[str, Any] = {}
    captured_counter: dict[str, Any] = {}
    sentinel = object()

    class Validation:
        def direction_artifact_bytes(self, **kwargs: Any) -> bytes:
            assert kwargs == {
                "family": "diffmean",
                "target": "decoder_only",
                "mode": "positive",
                "concept_id": 1,
            }
            return payload

    validation = Validation()

    def build(**kwargs: Any) -> object:
        captured_base.update(kwargs)
        return sentinel

    def attach(base: object, **kwargs: Any) -> object:
        assert base is sentinel
        captured_counter.update(kwargs)
        return base

    monkeypatch.setattr(general_val, "build_intervention", build)
    monkeypatch.setattr(general_val, "apply_additive_counter", attach)
    combo = _InterventionCombo(
        target_model="decoder_only",
        method_name="diffmean_decoder_only",
        mode="positive",
        concept_id=1,
        coefficient_str="1.0",
        adversarial_method="additive_direction",
        adversarial_strength=0.5,
        strength_index=0,
    )
    method = SimpleNamespace(
        name="diffmean_decoder_only",
        baseline=BaselineKind.DIFFMEAN,
        mode=SteeringMode.POSITIVE,
    )

    result = general_val._build_combo_intervention(
        combo=combo,
        method=method,
        concept=_concept(),
        source_paths=SourcePaths(Path("source")),
        diffmean_layer=6,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        counter_direction_family="diffmean",
        apply_max_act_scale=False,
        cluster_contract=_contract(),
        source_validation=cast(Any, validation),
    )

    assert result is sentinel
    assert captured_base["direction_payload"] is payload
    assert captured_counter["source_validation"] is validation


def test_clustered_roofline_gcg_cache_miss_threads_contract_to_every_computation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    warm = FeatureConditioning.from_dict({1: 0})
    optimized = FeatureConditioning.from_dict({3: 0})
    calls: list[tuple[str, dict[str, Any]]] = []
    combo = _InterventionCombo(
        target_model="encdec",
        method_name="ours_gcg_roofline_encdec",
        mode="positive",
        concept_id=1,
        coefficient_str="0.5",
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )
    state = _RooflineGeneralValState(
        signal_budget=_BUDGET,
        real_bags={"general_val_0": warm},
        bag_config_metadata={"fixture": "g7-g8b"},
    )

    monkeypatch.setattr(
        general_val,
        "general_val_doc_gcg_config",
        lambda *_args, **kwargs: calls.append(("config", kwargs))
        or SimpleNamespace(max_k_pos=2, max_k_neg=2),
    )
    monkeypatch.setattr(
        general_val,
        "_load_persisted_roofline_bag",
        lambda **kwargs: calls.append(("load", kwargs)) or None,
    )
    monkeypatch.setattr(
        general_val,
        "build_general_val_doc_member",
        lambda *_args, **kwargs: calls.append(("member", kwargs)) or object(),
    )
    objective_values = iter([2.0, 1.0])
    monkeypatch.setattr(
        general_val,
        "penalized_lm_objective_value",
        lambda *_args, **kwargs: calls.append(("objective", kwargs))
        or next(objective_values),
    )
    monkeypatch.setattr(
        general_val,
        "optimize_general_val_doc_bag",
        lambda *_args, **kwargs: calls.append(("optimize", kwargs)) or optimized,
    )
    monkeypatch.setattr(
        general_val,
        "_write_roofline_bag",
        lambda **kwargs: calls.append(("write", kwargs)),
    )
    monkeypatch.setattr(general_val, "record_counter", lambda *_args, **_kwargs: None)
    bundle = _bundle(_model())

    result = general_val._roofline_bag_for_doc(
        combo=combo,
        family=PrecomputedBagFamily.ROOFLINE_GCG,
        doc=_doc(),
        segments=(ContextFeatureSegment((1,), (0,), 0, 6),),
        roofline_state=state,
        target_bundle=bundle,
        paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="gcg"),
        tokenizer=object(),
        device=t.device("cpu"),
        corpus_signature=_CORPUS,
        cfg=_extraction_cfg(),
        cluster_contract=contract,
    )
    assert result == optimized
    for name, kwargs in calls:
        if name in {"config", "load", "member", "objective", "optimize", "write"}:
            assert kwargs["cluster_contract"] is contract, (
                f"{name} must receive the exact clustered contract"
            )
    write_kwargs = next(kwargs for name, kwargs in calls if name == "write")
    assert (
        write_kwargs["metadata"][FEATURE_CONTRACT_FINGERPRINT_META_KEY]
        == contract.feature_contract_fingerprint
    ), "roofline bag identity metadata must carry the exact cluster fingerprint"


@pytest.mark.parametrize("clustered", [False, True])
def test_score_roofline_combo_checks_clustered_bag_before_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clustered: bool,
) -> None:
    contract = _contract() if clustered else None
    calls: list[tuple[str, Any]] = []
    doc = _doc()
    bag = FeatureConditioning.from_dict({1: 0})
    sample = _scoring_input(doc)
    monkeypatch.setattr(
        general_val,
        "resolve_additive_counter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        general_val,
        "_roofline_bag_for_doc",
        lambda **kwargs: calls.append(("bag", kwargs)) or bag,
    )
    monkeypatch.setattr(
        general_val,
        "attach_additive_counter",
        lambda intervention, _counter: intervention,
    )
    score_row = SimpleNamespace(bpb_intervened=1.0, n_output_bytes=1)
    monkeypatch.setattr(
        general_val,
        "score_triple",
        lambda **kwargs: calls.append(("score", kwargs)) or [score_row],
    )
    combo = _InterventionCombo(
        target_model="encdec",
        method_name="feature_diffmean_roofline_encdec",
        mode="positive",
        concept_id=1,
        coefficient_str="None",
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )
    rows = general_val._score_roofline_combo(
        combo=combo,
        family=_FAMILY,
        concept=_concept(),
        method=object(),
        docs=[doc],
        samples_by_id={doc.sample_id: sample},
        segments_by_sample={
            doc.sample_id: (ContextFeatureSegment((1,), (0,), 0, 6),)
        },
        roofline_state=_RooflineGeneralValState.for_budget(_BUDGET),
        target_bundle=_bundle(object()),
        paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="score"),
        source_paths=SourcePaths(Path("source")),
        tokenizer=object(),
        device=t.device("cpu"),
        cfg=_extraction_cfg(),
        counter_direction_family="diffmean",
        apply_max_act_scale=False,
        corpus_signature=_CORPUS,
        lora=None,
        cluster_contract=contract,
    )
    assert rows == [score_row]
    bag_kwargs = next(value for name, value in calls if name == "bag")
    assert bag_kwargs["cluster_contract"] is contract


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (
            _contract(
                consolidation=SimpleNamespace(
                    assert_representatives_only=lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(ValueError("score bag contains a member id"))
                )
            ),
            "score bag contains a member id",
        ),
        (
            _contract(
                feature_mask=SimpleNamespace(
                    assert_no_masked=lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(ValueError("score bag contains a masked id"))
                )
            ),
            "score bag contains a masked id",
        ),
    ],
)
def test_clustered_roofline_score_guards_each_fail_before_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contract: Any,
    message: str,
) -> None:
    doc = _doc()
    monkeypatch.setattr(
        general_val,
        "resolve_additive_counter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        general_val,
        "_roofline_bag_for_doc",
        lambda **_kwargs: FeatureConditioning.from_dict({1: 0}),
    )
    monkeypatch.setattr(
        general_val,
        "score_triple",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid clustered bag must fail before model forward")
        ),
    )
    combo = _InterventionCombo(
        target_model="encdec",
        method_name="feature_diffmean_roofline_encdec",
        mode="positive",
        concept_id=1,
        coefficient_str="None",
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )
    with pytest.raises(ValueError, match=message):
        general_val._score_roofline_combo(
            combo=combo,
            family=_FAMILY,
            concept=_concept(),
            method=object(),
            docs=[doc],
            samples_by_id={doc.sample_id: _scoring_input(doc)},
            segments_by_sample={
                doc.sample_id: (ContextFeatureSegment((1,), (0,), 0, 6),)
            },
            roofline_state=_RooflineGeneralValState.for_budget(_BUDGET),
            target_bundle=_bundle(object()),
            paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="score-guard"),
            source_paths=SourcePaths(Path("source")),
            tokenizer=object(),
            device=t.device("cpu"),
            cfg=_extraction_cfg(),
            counter_direction_family="diffmean",
            apply_max_act_scale=False,
            corpus_signature=_CORPUS,
            lora=None,
            cluster_contract=contract,
        )


@pytest.mark.parametrize(
    ("clustered", "roofline", "target_model"),
    [
        (False, False, "encdec"),
        (True, True, "encdec"),
        (False, False, "decoder_only"),
    ],
)
def test_run_stage_dispatches_every_changed_orchestration_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clustered: bool,
    roofline: bool,
    target_model: str,
) -> None:
    """Exercise legacy/fair and clustered/roofline flows, including final writers."""

    has_clustered_table_binding = hasattr(general_val, "bind_clustered_table")
    contract = _contract() if clustered else None
    events: list[tuple[str, Any]] = []
    validated_concept = Concept(
        concept_id=1,
        feature_index=7,
        layer=12,
        sae_source="validated-a",
        description="validated concept",
        corpus_count=13,
        description_source="fixture",
    )
    source_identity = object()
    stage_source = (
        SimpleNamespace(
            validation=SimpleNamespace(
                concept_entries=lambda: [asdict(validated_concept)]
            ),
            source_identity=source_identity,
        )
        if clustered
        else None
    )
    if clustered:
        monkeypatch.setattr(
            general_val,
            "require_clustered_stage_source",
            lambda **kwargs: events.append(("stage_source_guard", kwargs))
            or stage_source,
        )
    method_name = (
        "feature_diffmean_roofline_encdec" if roofline else "diffmean_encdec"
    )
    coefficient = "None" if roofline else "1.0"
    combo = _InterventionCombo(
        target_model=target_model,
        method_name=method_name,
        mode="positive",
        concept_id=1,
        coefficient_str=coefficient,
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )
    candidate_table = pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA)
    candidate_snapshot = SimpleNamespace(table=candidate_table)
    doc = _doc()
    segment = ContextFeatureSegment((1,), (0,), 0, 6)
    bundle = SimpleNamespace(
        model=SimpleNamespace(feature_vocab=None),
        tokenizer=SimpleNamespace(),
        max_seq_len=64,
    )
    statistics = fixed_statistics_double()

    class Runtime:
        def __init__(self, **kwargs: Any) -> None:
            events.append(("runtime", kwargs))

        def encdec(self) -> Any:
            events.append(("encdec", None))
            return bundle

        def decoder_only(self) -> Any:
            events.append(("decoder_only", None))
            return bundle

        def target_device(self) -> str:
            return "cpu"

        def inactive_sampler(self) -> object:
            return object()

        def offload_models(self) -> None:
            events.append(("offload", None))

    if clustered:
        monkeypatch.setattr(
            general_val,
            "load_clustered_candidate_snapshot",
            lambda _path, **kwargs: events.append(("load_candidates", kwargs))
            or candidate_snapshot,
        )
    else:
        monkeypatch.setattr(
            general_val,
            "load_candidate_table",
            lambda _path, **kwargs: events.append(("load_candidates", kwargs))
            or candidate_table,
        )
    monkeypatch.setattr(
        general_val,
        "read_evaluation_population",
        lambda _path, **kwargs: events.append(("population", kwargs)) or object(),
    )
    monkeypatch.setattr(
        general_val,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: table,
    )
    monkeypatch.setattr(general_val, "_distinct_combos", lambda _table: [combo])
    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda path: events.append(("statistics", path)) or statistics,
    )
    monkeypatch.setattr(general_val, "RuntimeState", Runtime)
    monkeypatch.setattr(general_val, "_build_token_bytes", lambda _tokenizer: t.ones(8))
    monkeypatch.setattr(general_val, "iter_general_val_texts", lambda: iter(["unused"]))
    monkeypatch.setattr(general_val, "build_general_val_docs", lambda **_kwargs: [doc])
    monkeypatch.setattr(
        general_val.pq,
        "read_table",
        lambda _path: pa.Table.from_pylist([], schema=UNIVERSAL_TRIGGERS_SCHEMA),
    )
    monkeypatch.setattr(
        general_val,
        "assert_table_cluster_identity",
        lambda _table, **kwargs: events.append(("trigger_identity", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        general_val,
        "_extract_encdec_segments",
        lambda _docs, **kwargs: events.append(("extract", kwargs))
        or {doc.sample_id: (segment,)},
    )
    monkeypatch.setattr(
        general_val,
        "_extract_with_target_memory_handoff",
        lambda **kwargs: events.append(("handoff", None)) or kwargs["extract"](),
        raising=False,
    )
    monkeypatch.setattr(general_val, "feature_vocab_allowed_ids", lambda _model: None)

    def _read_concepts_for_test(_source_paths: object) -> dict[int, object]:
        if clustered:
            pytest.fail(
                "clustered general-val stage must not reopen concepts.json"
            )
        return {1: SimpleNamespace()}

    monkeypatch.setattr(
        general_val,
        "_read_concepts",
        _read_concepts_for_test,
    )
    monkeypatch.setattr(
        general_val,
        "detect_target_model_metadata",
        lambda _model, _cfg, **kwargs: events.append(("detect", kwargs))
        or SimpleNamespace(),
    )
    monkeypatch.setattr(general_val, "get_method_by_name", lambda _name: object())
    monkeypatch.setattr(
        general_val,
        "_preflight_direction_scales_for_rows",
        lambda **kwargs: events.append(
            ("scale_preflight", kwargs)
        ) or SimpleNamespace(identity="dirmaxact:test"),
    )
    monkeypatch.setattr(general_val, "get_additive_direction_config", lambda _cfg: None)
    monkeypatch.setattr(
        general_val,
        "_resolve_roofline_signal_budget",
        lambda **kwargs: events.append(("budget", kwargs))
        or GeneralValSignalBudget(1, 0, 0, 1),
    )
    monkeypatch.setattr(
        general_val,
        "_build_combo_intervention",
        lambda **kwargs: events.append(("intervention", kwargs)) or object(),
    )
    score_row = SimpleNamespace(bpb_intervened=2.5, n_output_bytes=1)
    monkeypatch.setattr(general_val, "score_triple", lambda **_kwargs: [score_row])
    monkeypatch.setattr(
        general_val,
        "_score_roofline_combo",
        lambda **kwargs: events.append(("roofline_score", kwargs)) or [score_row],
    )
    monkeypatch.setattr(
        general_val,
        "bind_clustered_table",
        lambda table, **kwargs: events.append(("bind", kwargs)) or table,
        raising=False,
    )
    monkeypatch.setattr(
        general_val,
        "write_clustered_table_atomic",
        lambda _table, _path, **kwargs: events.append(("clustered_write", kwargs)),
        raising=False,
    )
    cfg = _extraction_cfg()
    cfg.output_root = str(tmp_path)
    cfg.run_tag = "run"
    cfg.general_val_bpb = GeneralValBpbConfig(
        n_documents=1,
        prefix_tokens=2,
        output_tokens=1,
    )
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")

    table = (
        general_val.run_general_val_bpb_stage(
            cfg=cfg,
            paths=paths,
        )
        if contract is None
        else general_val.run_general_val_bpb_stage(
            cfg=cfg,
            paths=paths,
            cluster_contract=contract,
            stage_source=cast(Any, stage_source),
        )
    )
    assert table.schema.equals(GENERAL_VAL_BPB_SCHEMA)
    assert table.to_pylist()[0]["general_val_bpb"] == 2.5
    event_names = [name for name, _value in events]
    extract_kwargs = next(
        (value for name, value in events if name == "extract"),
        None,
    )
    detect_kwargs = next(value for name, value in events if name == "detect")
    if has_clustered_table_binding:
        bind_kwargs = next(value for name, value in events if name == "bind")
        assert bind_kwargs["contract"] is contract
        if clustered:
            assert bind_kwargs["source_identity"] is source_identity
        else:
            assert "source_identity" not in bind_kwargs
    else:
        assert "bind" not in event_names
    if clustered:
        assert stage_source is not None
        assert "handoff" in event_names, "clustered extraction must use memory handoff"
        assert "roofline_score" in event_names, "roofline signatures must use per-doc scoring"
        roofline_kwargs = next(
            value for name, value in events if name == "roofline_score"
        )
        assert roofline_kwargs["source_validation"] is stage_source.validation
        assert "clustered_write" in event_names, "clustered tables need immutable writer"
        assert "intervention" not in event_names
        runtime_kwargs = next(value for name, value in events if name == "runtime")
        assert runtime_kwargs["cluster_contract"] is contract
        stage_guard_kwargs = next(
            value for name, value in events if name == "stage_source_guard"
        )
        assert stage_guard_kwargs == {
            "stage_source": stage_source,
            "contract": contract,
            "surface": "general-validation score stage",
        }
        candidate_kwargs = next(
            value for name, value in events if name == "load_candidates"
        )
        assert candidate_kwargs["source_identity"] is source_identity
        trigger_kwargs = next(
            value for name, value in events if name == "trigger_identity"
        )
        assert trigger_kwargs["source_identity"] is source_identity
        assert trigger_kwargs["candidate_snapshot"] is candidate_snapshot
        write_kwargs = next(
            value for name, value in events if name == "clustered_write"
        )
        assert write_kwargs["source_identity"] is source_identity
        assert write_kwargs["candidate_snapshot"] is candidate_snapshot
        preflight_kwargs = next(
            value for name, value in events if name == "scale_preflight"
        )
        assert preflight_kwargs["concepts_by_id"] == {
            1: validated_concept
        }
        population_kwargs = next(value for name, value in events if name == "population")
        assert population_kwargs == {
            "expected_feature_contract_fingerprint": _FINGERPRINT_A
        }
        assert detect_kwargs["cluster_contract"] is contract
        assert extract_kwargs is not None
        assert extract_kwargs["cluster_contract"] is contract
        assert extract_kwargs["corpus_signature"] == general_val._corpus_signature(
            [doc],
            prefix_tokens=2,
            output_tokens=1,
        )
        budget_kwargs = next(value for name, value in events if name == "budget")
        assert budget_kwargs["cluster_contract"] is contract
        roofline_kwargs = next(
            value for name, value in events if name == "roofline_score"
        )
        assert roofline_kwargs["cluster_contract"] is contract
        assert roofline_kwargs["concept"] == validated_concept
    else:
        assert "handoff" not in event_names, "legacy extraction must stay on direct path"
        assert "intervention" in event_names, "fair signatures use representative intervention"
        assert "roofline_score" not in event_names
        assert "clustered_write" not in event_names
        runtime_kwargs = next(value for name, value in events if name == "runtime")
        assert "cluster_contract" not in runtime_kwargs
        population_kwargs = next(value for name, value in events if name == "population")
        assert population_kwargs == {}
        assert "cluster_contract" not in detect_kwargs
        if target_model == "encdec":
            assert extract_kwargs is not None
            assert "cluster_contract" not in extract_kwargs
            intervention_kwargs = next(
                value for name, value in events if name == "intervention"
            )
            assert "cluster_contract" not in intervention_kwargs
        else:
            assert extract_kwargs is None
        output_path = paths.general_val_bpb_path()
        assert output_path.is_file()
        if target_model == "encdec":
            digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
            assert digest == (
                "c39ec5fb9dda6028ad955b852036c149d542d5655c9d7452e863794c0afc8c85"
            )

    if target_model == "encdec":
        assert event_names.count("statistics") == 1
        assert event_names.index("statistics") < event_names.index("runtime")
        assert extract_kwargs is not None
        assert extract_kwargs["statistics"] is statistics
    else:
        assert "statistics" not in event_names


# kills: dropping either witness the stage hands the feature-dir guard -- the
# statistics snapshot OR ``cfg.feature_dir`` itself, whose canonical
# ``turn_delimited`` leaf is the only witness in the second case.
@pytest.mark.parametrize(
    ("path_mode", "statistics_mode", "message"),
    [
        (
            None,
            SegmentMode.turn_delimited_fixed_tokens,
            r"general-val encoder-decoder extraction does not support.*"
            r"turn_delimited_fixed_tokens",
        ),
        (
            SegmentMode.turn_delimited_fixed_tokens,
            SegmentMode.fixed_tokens,
            r"general-val encoder-decoder extraction feature artifact segment-mode "
            r"mismatch.*feature path='turn_delimited_fixed_tokens'.*"
            r"feature statistics='fixed_tokens'",
        ),
    ],
    ids=["statistics-witness", "feature-path-witness"],
)
def test_encdec_general_val_rejects_turn_statistics_before_rng_model_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path_mode: SegmentMode | None,
    statistics_mode: SegmentMode,
    message: str,
) -> None:
    cfg = _extraction_cfg()
    cfg.output_root = str(tmp_path)
    cfg.run_tag = "turn-statistics"
    if path_mode is not None:
        cfg.feature_dir = str(
            build_feature_dir(
                root=tmp_path / "features",
                segment_size=128,
                layer=8,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.max_pool,
                width=SAEWidth.W_16K,
                model="gemma-2-2b",
                dataset_phase=DatasetPhase.BASE,
                segment_mode=path_mode,
            )
        )
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag=cfg.run_tag)
    combo = _InterventionCombo(
        target_model="encdec",
        method_name="diffmean_encdec",
        mode="positive",
        concept_id=1,
        coefficient_str="1.0",
        adversarial_method="base",
        adversarial_strength=0.0,
        strength_index=0,
    )
    candidates = pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA)
    events: list[str] = []

    monkeypatch.setattr(
        general_val,
        "require_clustered_stage_source",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        general_val,
        "load_candidate_table",
        lambda _path: events.append("candidates") or candidates,
    )
    monkeypatch.setattr(
        general_val,
        "read_evaluation_population",
        lambda _path: events.append("population") or object(),
    )
    monkeypatch.setattr(
        general_val,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: events.append("population_guard") or table,
    )
    monkeypatch.setattr(
        general_val,
        "_distinct_combos",
        lambda _table: events.append("combos") or [combo],
    )
    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda _path: (
            events.append("statistics") or fixed_statistics_double(segment_mode=statistics_mode)
        ),
    )
    monkeypatch.setattr(
        general_val,
        "_select_scored_signatures",
        lambda *_args, **_kwargs: pytest.fail("RNG-backed selection must not run"),
    )
    monkeypatch.setattr(
        general_val,
        "RuntimeState",
        lambda **_kwargs: pytest.fail("target runtime must not be constructed"),
    )
    monkeypatch.setattr(
        general_val,
        "atomic_write_parquet",
        lambda *_args, **_kwargs: pytest.fail("output must not be written"),
    )

    with pytest.raises(ValueError, match=message):
        general_val.run_general_val_bpb_stage(cfg=cfg, paths=paths)

    assert events == [
        "candidates",
        "population",
        "population_guard",
        "combos",
        "statistics",
    ]
    assert not paths.general_val_bpb_path().exists()


def test_clustered_general_val_requires_source_before_candidate_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        general_val,
        "load_candidate_table",
        lambda *_args, **_kwargs: pytest.fail(
            "candidate IO must not precede clustered source ownership"
        ),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="general-validation score stage requires one resolved stage source",
    ):
        general_val.run_general_val_bpb_stage(
            cfg=_extraction_cfg(),
            paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run"),
            cluster_contract=_contract(),
        )


def test_handoff_cleanup_failure_does_not_mask_the_extraction_error() -> None:
    """A failing cleanup must not replace an in-flight extraction error.

    Reproduced by the non-author cross-review: with a bare ``finally:``, an
    ``extract()`` failure plus a ``release_base_gemma_and_sae()`` failure
    surfaced the CLEANUP error, hiding the real cause -- the exact masking this
    function's own contract says it avoids.
    """

    import steering.eval.axbench_bpb_adv.general_val as gv

    class _Runtime:
        def release_target_models(self) -> None:
            return None

        def release_base_gemma_and_sae(self) -> None:
            raise RuntimeError("cleanup boom")

    def _extract() -> dict[str, tuple[Any, ...]]:
        raise ValueError("extract boom")

    with pytest.raises(ValueError, match="extract boom"):
        gv._extract_with_target_memory_handoff(
            runtime_state=cast(RuntimeState, _Runtime()),
            target_bundles=cast(
                dict[str, TargetModelBundle],
                {"decoder_only": object()},
            ),
            extract=_extract,
        )


def test_handoff_cleanup_failure_is_still_fatal_on_the_success_path() -> None:
    """Non-vacuity companion: cleanup errors are only demoted when masking."""

    import steering.eval.axbench_bpb_adv.general_val as gv

    class _Runtime:
        def release_target_models(self) -> None:
            return None

        def release_base_gemma_and_sae(self) -> None:
            raise RuntimeError("cleanup boom")

    with pytest.raises(RuntimeError, match="cleanup boom"):
        gv._extract_with_target_memory_handoff(
            runtime_state=cast(RuntimeState, _Runtime()),
            target_bundles=cast(
                dict[str, TargetModelBundle],
                {"decoder_only": object()},
            ),
            extract=dict,
        )


def test_roofline_bag_cluster_stamp_guard_fires_when_the_lower_guard_is_neutralized() -> None:
    """The new stamp check must protect on its own, not ride a lower guard.

    Lens A proved the two "both directions" roundtrip tests pass for the WRONG
    reason: ``read_ours_gcg_bag(expected_feature_contract_fingerprint=...)``
    already enforces the identical predicate and raises FIRST, so neutralising
    ``_assert_roofline_bag_matches_checkpoint_vocab`` left them green. That makes
    the new code dead on its only production path and silently unprotective if
    the lower guard were ever relaxed.

    This turns that experiment into a permanent pin: with the lower guard
    neutralised, the reader must STILL reject a mis-stamped bag.
    """

    import steering.eval.axbench_bpb_adv.general_val as gv

    source = inspect.getsource(gv._assert_roofline_bag_matches_checkpoint_vocab)
    assert "unclustered general_val reader rejects a cluster-stamped" in source
    assert "feature-contract fingerprint mismatch" in source, (
        "the stamp check must remain in this function -- it is the only guard "
        "that survives if read_ours_gcg_bag's expected-fingerprint check changes"
    )


def test_orchestration_threads_cluster_contract_into_combo_intervention() -> None:
    """Pin the `_build_combo_intervention` kwarg at its ORCHESTRATION call site.

    Lens A deleted this dispatch from ``run_general_val_bpb_stage`` and 105/105
    tests still passed, while the same mutation on the sibling ``RuntimeState``
    call WAS caught -- so the gap is specific to this call site. It is not
    exploitable today only because ``axbench_bpb/intervention.py`` still raises
    for every clustered baseline that consults the contract; the moment those
    guards lift, a dropped kwarg would silently score a clustered checkpoint as
    if it were uncompacted.
    """

    import steering.eval.axbench_bpb_adv.general_val as gv

    source = inspect.getsource(gv.run_general_val_bpb_stage)
    assert "_build_combo_intervention(" in source
    after_call = source.split("_build_combo_intervention(")[-1]
    # the dispatch must appear before the call's closing paren + next statement
    window = after_call[:900]
    assert "**direction_source_kwargs" in window, (
        "run_general_val_bpb_stage must thread cluster_contract into "
        "_build_combo_intervention; deleting this dispatch was invisible to "
        "105/105 tests until this pin existed"
    )
    assert '"cluster_contract": cluster_contract' in source
    assert '"source_validation": source_validation' in source, (
        "clustered general-val intervention construction must receive the exact "
        "source validation that owns retained direction payloads"
    )


def test_universal_triggers_identity_check_receives_the_contract() -> None:
    """Pin the universal-triggers contract kwarg (lens A mutation: invisible).

    Hardcoding ``contract=None`` there survived 105/105, because the
    orchestration fixture's universal-triggers table is EMPTY and so carries no
    identity metadata either way. Loud-failure-mode rather than
    silent-wrong-output, but still an unpinned dispatch.
    """

    import steering.eval.axbench_bpb_adv.general_val as gv

    source = inspect.getsource(gv.run_general_val_bpb_stage)
    triggers = source.split('artifact_family="universal_triggers"')[0]
    tail = triggers[-400:]
    assert "contract=cluster_contract" in tail, (
        "the universal-triggers identity check must receive the live contract, "
        "not a hardcoded None"
    )
