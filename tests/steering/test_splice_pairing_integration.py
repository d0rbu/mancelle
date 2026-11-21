"""Pairing semantics through real shard I/O and unchanged downstream consumers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest
import torch as t

import steering.feature_shard_splice as splice_core
from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering._roundtrip_seam import is_roundtrip_stable
from steering.compute_statistics import compute_statistics
from steering.conditional.percentile_table import build_percentile_table, magnitude_to_percentile
from steering.conditional.splice_provenance import SpliceProvenanceIndex
from steering.feature_io import glob_features, load_features, save_features, shard_stem_from_path
from steering.feature_shard_pairing import OppositePercentilePairing
from steering.feature_shard_splice import (
    PoolExhaustedError,
    load_source_segments,
    recombination_seed,
    recombine_segments,
)
from steering.models import resolve_sae_max_context, resolve_tokenizer
from steering.parquet_io import read_parquet_texts
from steering.shard_validation import validate_shard_data
from tests.steering._splice_fixtures import doc_seg_ranges, make_base_shard
from tests.steering._splice_pairing_fixtures import (
    PAIRING_TOKENIZER,
    PAIRING_WIDTH,
    build_pairing_corpus,
    pairing_run_config,
    snapshot_files,
)
from tests.steering._splice_reference import reference_recombine
from tests.steering._splice_test_tokenizers import CharMergeTokenizer, SpanMergeTokenizer


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return PAIRING_TOKENIZER

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", lambda _model: 8192)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def _dense_percentiles(shard: dict, table: t.Tensor) -> t.Tensor:
    result = t.zeros(int(shard["n_segments"]), table.shape[0], dtype=t.float64)
    for row in range(len(result)):
        lo, hi = int(shard["offsets"][row]), int(shard["offsets"][row + 1])
        ids, mags = shard["feature_ids"][lo:hi], shard["feature_mags"][lo:hi]
        result[row, ids.long()] = magnitude_to_percentile(ids, mags, table)
    return result


def _counterfactual_score(shard: dict, sidecar: dict, table: t.Tensor) -> float:
    dense = _dense_percentiles(shard, table)
    ranges = doc_seg_ranges(shard)
    total = 0.0
    for doc in sidecar["docs"]:
        for splice in doc["splices"]:
            incoming = ranges[doc["doc_id"]][0] + splice["pos"]
            reference_doc, reference_pos = splice["whf_derived"]
            reference = ranges[reference_doc][0] + reference_pos
            total += float((dense[incoming] - dense[reference]).abs().sum())
    return total


@pytest.mark.parametrize("compact_provenance", [False, True])
def test_opposite_pairing_improves_replaced_continuations_and_preserves_features_and_references(
    tmp_path: Path, compact_provenance: bool
) -> None:
    corpus = build_pairing_corpus(tmp_path)
    assert corpus.statistics is not None
    source_before = snapshot_files(corpus.feature_dir)
    table = build_percentile_table(corpus.statistics.per_feature_stats, PAIRING_WIDTH)
    random_dir, opposite_dir = tmp_path / "random", tmp_path / "opposite"
    sfs.run(pairing_run_config(corpus, random_dir, pairing_mode="random"))
    sfs.run(pairing_run_config(corpus, opposite_dir))
    index = SpliceProvenanceIndex(opposite_dir, compact=compact_provenance)
    score_random, score_opposite = 0.0, 0.0
    for source_path, random_path, opposite_path in zip(
        glob_features(corpus.feature_dir),
        glob_features(random_dir),
        glob_features(opposite_dir),
        strict=True,
    ):
        source, random_shard, opposite = map(
            load_features, (source_path, random_path, opposite_path)
        )
        assert validate_shard_data(opposite).valid
        stem = shard_stem_from_path(source_path)
        sidecar = json.loads((opposite_dir / sfs.PROVENANCE_DIRNAME / f"{stem}.json").read_text())
        source_ranges = doc_seg_ranges(source)
        derived_ranges = doc_seg_ranges(opposite)
        assert sidecar["params"]["n_redrawn"] == 0
        assert sidecar["params"]["n_loops"] == 0
        assert sidecar["params"]["n_splits"] == 0
        assert index.shard_n_splice_positions(stem) == 8
        placed: list[tuple[int, int]] = []
        texts = read_parquet_texts(opposite_dir / sfs.PARQUET_DIRNAME / f"{stem}.parquet")
        for doc, (lo, hi), text in zip(sidecar["docs"], derived_ranges, texts, strict=True):
            assert hi - lo == 2
            ids = tuple(PAIRING_TOKENIZER(text)["input_ids"])
            assert is_roundtrip_stable(ids, PAIRING_TOKENIZER)
            assert len(ids) == 8
            assert opposite["segment_positions"][lo:hi].tolist() == [0, 4]
            for row, (_, source_doc, source_ordinal) in zip(
                range(lo, hi), doc["provenance"], strict=True
            ):
                placed.append((source_doc, source_ordinal))
                source_row = source_ranges[source_doc][0] + source_ordinal
                source_lo, source_hi = (
                    int(source["offsets"][j]) for j in (source_row, source_row + 1)
                )
                derived_lo, derived_hi = (int(opposite["offsets"][j]) for j in (row, row + 1))
                for key in ("feature_ids", "feature_mags"):
                    assert t.equal(
                        source[key][source_lo:source_hi], opposite[key][derived_lo:derived_hi]
                    )
            splices = index.doc_splices(stem, doc["doc_id"])
            assert set(splices) == {1}
            ref = splices[1]
            prefix = doc["provenance"][0]
            assert ref.whf_base == (prefix[0], prefix[1], prefix[2] + 1)
            whf_doc, whf_pos = ref.whf_derived
            assert tuple(sidecar["docs"][whf_doc]["provenance"][whf_pos]) == ref.whf_base
        assert sorted(placed) == [(doc, ordinal) for doc in range(8) for ordinal in range(2)]
        random_sidecar = json.loads(
            (random_dir / sfs.PROVENANCE_DIRNAME / f"{stem}.json").read_text()
        )
        score_random += _counterfactual_score(random_shard, random_sidecar, table)
        score_opposite += _counterfactual_score(opposite, sidecar, table)
    assert (
        score_opposite > score_random
    )  # non-vacuous semantic improvement, not merely valid output
    assert snapshot_files(corpus.feature_dir) == source_before


def test_pairing_opposes_replaced_continuations_even_when_prefixes_favor_the_reverse() -> None:
    texts = [char * 7 for char in "abcd"]
    shard = make_base_shard(texts, PAIRING_TOKENIZER, 4, context_window=32)
    source, _ = load_source_segments(
        shard,
        texts,
        PAIRING_TOKENIZER,
        segment_size=4,
        context_window=32,
        sae_max_context=None,
        shard_name="counterfactual",
    )
    source = [
        [
            replace(
                doc[0],
                feature_ids=t.tensor([1 - d // 2], dtype=t.int32),
                feature_mags=t.tensor([1.0]),
            ),
            replace(
                doc[1], feature_ids=t.tensor([d // 2], dtype=t.int32), feature_mags=t.tensor([1.0])
            ),
        ]
        for d, doc in enumerate(source)
    ]
    pairing = OppositePercentilePairing(t.ones(2, 99, dtype=t.float64), candidate_pool_size=4)
    original = [1, 0, 3, 2]
    optimized = pairing.improve_pairing(source, [(d, 0) for d in range(4)], original)
    assert original == [1, 0, 3, 2]
    assert optimized == [3, 2, 1, 0]
    assert all(
        int(source[d][1].feature_ids[0]) != int(source[target][1].feature_ids[0])
        for d, target in enumerate(optimized)
    )
    assert all(
        int(source[d][0].feature_ids[0]) == int(source[target][1].feature_ids[0])
        for d, target in enumerate(optimized)
    )


def test_opposite_recomputed_statistics_equal_base_statistics(tmp_path: Path) -> None:
    corpus = build_pairing_corpus(tmp_path)
    assert corpus.statistics is not None
    output = tmp_path / "opposite"
    sfs.run(pairing_run_config(corpus, output))
    derived = compute_statistics(
        str(output),
        str(output / "feature_statistics.safetensors"),
        n_sae_features=PAIRING_WIDTH,
        device="cpu",
        exclude_last_shard=True,
        allow_degenerate_statistics=True,
    )
    assert derived.per_feature_stats == corpus.statistics.per_feature_stats
    assert t.equal(
        build_percentile_table(derived.per_feature_stats, PAIRING_WIDTH),
        build_percentile_table(corpus.statistics.per_feature_stats, PAIRING_WIDTH),
    )


@pytest.mark.parametrize("fraction", [0.0, 0.125])
def test_identity_cut_count_never_queries_pairing_or_changes_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fraction: float
) -> None:
    corpus = build_pairing_corpus(tmp_path)

    def unexpected_pairing(*_args, **_kwargs):
        pytest.fail("fewer than two cuts must bypass pairing")

    monkeypatch.setattr(OppositePercentilePairing, "improve_pairing", unexpected_pairing)
    output = tmp_path / "identity"
    sfs.run(pairing_run_config(corpus, output, cut_fraction=fraction))
    for source_path, output_path in zip(
        glob_features(corpus.feature_dir), glob_features(output), strict=True
    ):
        source, derived = load_features(source_path), load_features(output_path)
        for key in (
            "offsets",
            "feature_ids",
            "feature_mags",
            "doc_ids",
            "segment_indices",
            "segment_positions",
        ):
            assert t.equal(source[key], derived[key])
        payload = json.loads(
            (
                output / sfs.PROVENANCE_DIRNAME / f"{shard_stem_from_path(source_path)}.json"
            ).read_text()
        )
        assert payload["params"]["n_cuts_final"] == 0
        assert payload["params"]["pairing_mode"] == "opposite_percentile"


@pytest.mark.parametrize(
    "texts,tokenizer,fraction,seed,limit,expected_counter,branch",
    [
        (
            ["az by", "bx ya", "cz dy", "wa vb"],
            CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")]),
            0.5,
            23,
            2048,
            "n_redrawn",
            None,
        ),
        (["abcdefghi"], CharMergeTokenizer(bos_id=500), 0.5, 0, 2048, "n_loops", None),
        (
            ["aaaa bbbb cccc", "dddd eeee ffff", "gggg hhhh iiii"],
            CharMergeTokenizer(bos_id=500),
            0.5,
            3,
            6,
            "n_splits",
            None,
        ),
        (
            ["epqrfs", "gpXqrYsZ", "kpmqrnso", "apbqrcsd", "wpxqrysz", "pqXsrp"],
            SpanMergeTokenizer(bos_id=500, spans=["pqrs"]),
            0.7,
            56,
            2048,
            "n_redrawn",
            "_first_cut_slot_in_piece",
        ),
        (
            ["wAB12x", "yCD45z", "kEF78m"],
            SpanMergeTokenizer(bos_id=500, spans=["BCDE", "DEFA", "FABC"]),
            0.8,
            1355,
            2048,
            "n_redrawn",
            "_lowest_cut_slot_in_cycle",
        ),
    ],
    ids=[
        "seam-redraw",
        "loop-break",
        "overflow-split",
        "full-document-redraw",
        "loop-culprit-redraw",
    ],
)
def test_tied_pairing_preserves_exact_reference_recovery_behavior(
    monkeypatch: pytest.MonkeyPatch,
    texts,
    tokenizer,
    fraction,
    seed,
    limit,
    expected_counter,
    branch,
) -> None:
    shard = make_base_shard(texts, tokenizer, 2, context_window=2048)
    source, _ = load_source_segments(
        shard,
        texts,
        tokenizer,
        segment_size=2,
        context_window=2048,
        sae_max_context=None,
        shard_name="recovery",
    )
    # All vectors identical: pairing really runs, but a correct tie rule makes
    # no swap. Existing nontrivial recovery fixtures can therefore use the
    # independent legacy graph oracle without predicting their internals.
    source = [
        [
            replace(seg, feature_ids=t.tensor([0], dtype=t.int32), feature_mags=t.tensor([1.0]))
            for seg in doc
        ]
        for doc in source
    ]
    pairing = OppositePercentilePairing(t.ones(1, 99, dtype=t.float64), candidate_pool_size=8)
    calls = []
    if branch is not None:
        original = getattr(splice_core, branch)

        def recording(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(splice_core, branch, recording)
    shard_seed = recombination_seed(seed, 7, fraction)
    result = recombine_segments(
        source,
        tokenizer,
        cut_fraction=fraction,
        shard_seed=shard_seed,
        shard_name="recovery",
        limit=limit,
        pairing=pairing,
    )
    docs, counters = reference_recombine(
        source, tokenizer, cut_fraction=fraction, shard_seed=shard_seed, limit=limit
    )
    assert [(doc.segment_refs, doc.new_started) for doc in result.docs] == docs
    assert {key: getattr(result, key) for key in counters} == counters
    assert getattr(result, expected_counter) > 0
    if branch is not None:
        assert calls, "the pinned rare recovery branch must actually execute"
    assert sorted(ref for doc in result.docs for ref in doc.segment_refs) == [
        (d, s) for d, doc in enumerate(source) for s in range(len(doc))
    ]
    for doc in result.docs:
        ids = tuple(tid for d, s in doc.segment_refs for tid in source[d][s].tokens)
        if doc.new_started:
            ids = (500, *ids)
        assert len(ids) <= limit
        assert is_roundtrip_stable(ids, tokenizer)


def test_opposite_keeps_pool_exhaustion_fail_loud() -> None:
    tokenizer = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    texts = ["az", "xb"]
    shard = make_base_shard(texts, tokenizer, 2)
    source, _ = load_source_segments(
        shard,
        texts,
        tokenizer,
        segment_size=2,
        context_window=2048,
        sae_max_context=None,
        shard_name="empty_pool",
    )
    pairing = OppositePercentilePairing(t.ones(4096, 99, dtype=t.float64))
    with pytest.raises(PoolExhaustedError):
        recombine_segments(
            source,
            tokenizer,
            cut_fraction=1.0,
            shard_seed=recombination_seed(1, 0, 1.0),
            shard_name="empty_pool",
            limit=2048,
            pairing=pairing,
        )


@pytest.mark.slow
def test_real_gemma_tokenizer_pairing_preserves_tokens_features_and_counterfactual_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU-only real-tokenizer path; use the cached Gemma tokenizer, no model weights."""
    tokenizer = resolve_tokenizer("gemma-2-2b")
    sae_limit = resolve_sae_max_context("gemma-2-2b")
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return tokenizer

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", resolve_sae_max_context)
    texts = [
        "The quick brown fox jumps over the lazy dog while the sun rises slowly over the mountains.",
        "Machine learning models process streams of tokens to produce high-dimensional embeddings for downstream tasks.",
        "Sparse autoencoders decompose hidden layer activations into interpretable parts using a large overcomplete dictionary.",
        "Researchers study how features distribute across many different documents in a large natural language corpus.",
    ]
    feature_dir, data_dir, output = tmp_path / "base", tmp_path / "parquets", tmp_path / "derived"
    feature_dir.mkdir()
    data_dir.mkdir()
    for index in range(2):
        shard = make_base_shard(texts, tokenizer, 4, context_window=2048, n_features=8)
        count = int(shard["n_segments"])
        shard["offsets"] = t.arange(count + 1, dtype=t.int64)
        shard["feature_ids"] = shard["doc_ids"] % 2
        shard["feature_mags"] = (1 + shard["doc_ids"] // 2).float()
        save_features(shard, feature_dir / f"shard_{index:05d}")
        atomic_write_parquet(pa.table({"text": texts}), data_dir / f"shard_{index:05d}.parquet")
    stats = compute_statistics(
        str(feature_dir),
        str(feature_dir / "feature_statistics.safetensors"),
        n_sae_features=8,
        device="cpu",
        exclude_last_shard=True,
        allow_degenerate_statistics=True,
    )
    source_snapshot = snapshot_files(feature_dir)
    optimize = OppositePercentilePairing.improve_pairing
    changed = []

    def observed_pairing(self, source_segments, cuts, sigma):
        optimized = optimize(self, source_segments, cuts, sigma)
        changed.append(optimized != sigma)
        return optimized

    monkeypatch.setattr(OppositePercentilePairing, "improve_pairing", observed_pairing)
    sfs.run(
        sfs.load_splice_config(
            {
                "feature_dir": str(feature_dir),
                "data_dir": str(data_dir),
                "output_dir": str(output),
                "model": "gemma-2-2b",
                "segment_size": 4,
                "context_window": 2048,
                "cut_fraction": 0.5,
                "pairing_mode": "opposite_percentile",
                "pairing_candidate_pool_size": 64,
            }
        )
    )
    assert len(changed) == 2 and any(changed), (
        "real run must execute and change partner assignments"
    )
    index = SpliceProvenanceIndex(output)
    for source_path, derived_path in zip(
        glob_features(feature_dir), glob_features(output), strict=True
    ):
        source_shard, derived = load_features(source_path), load_features(derived_path)
        stem = shard_stem_from_path(source_path)
        segments, _ = load_source_segments(
            source_shard,
            texts,
            tokenizer,
            segment_size=4,
            context_window=2048,
            sae_max_context=sae_limit,
            shard_name=stem,
        )
        assert validate_shard_data(derived).valid
        sidecar = json.loads((output / sfs.PROVENANCE_DIRNAME / f"{stem}.json").read_text())
        derived_texts = read_parquet_texts(output / sfs.PARQUET_DIRNAME / f"{stem}.parquet")
        ranges = doc_seg_ranges(derived)
        placed = []
        seen_splices = 0
        for doc, text, (start, stop) in zip(sidecar["docs"], derived_texts, ranges, strict=True):
            refs = [(d, s) for _shard, d, s in doc["provenance"]]
            placed.extend(refs)
            expected_ids = tuple(tid for d, s in refs for tid in segments[d][s].tokens)
            if refs[0][1] != 0:
                expected_ids = (tokenizer.bos_token_id, *expected_ids)
            assert tuple(tokenizer(text)["input_ids"]) == expected_ids
            assert is_roundtrip_stable(expected_ids, tokenizer)
            assert len(expected_ids) <= 2048
            for row, (d, s) in zip(range(start, stop), refs, strict=True):
                lo, hi = int(derived["offsets"][row]), int(derived["offsets"][row + 1])
                assert t.equal(derived["feature_ids"][lo:hi], segments[d][s].feature_ids)
                assert t.equal(derived["feature_mags"][lo:hi], segments[d][s].feature_mags)
            for position, reference in index.doc_splices(stem, doc["doc_id"]).items():
                seen_splices += 1
                previous = doc["provenance"][position - 1]
                assert reference.whf_base == (previous[0], previous[1], previous[2] + 1)
                whf_doc, whf_position = reference.whf_derived
                assert (
                    tuple(sidecar["docs"][whf_doc]["provenance"][whf_position])
                    == reference.whf_base
                )
        assert sorted(placed) == [(d, s) for d, doc in enumerate(segments) for s in range(len(doc))]
        assert seen_splices == index.shard_n_splice_positions(stem) > 0
    assert snapshot_files(feature_dir) == source_snapshot
    recomputed = compute_statistics(
        str(output),
        str(output / "feature_statistics.safetensors"),
        n_sae_features=8,
        device="cpu",
        exclude_last_shard=True,
        allow_degenerate_statistics=True,
    )
    assert recomputed.per_feature_stats == stats.per_feature_stats
