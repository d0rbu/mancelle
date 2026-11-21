"""Fail-loud gradient-overlay producer identity preflights (GA-001..010)."""

# ruff: noqa: F811 - imported pytest fixtures are intentionally requested by name.

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import scripts.preprocess.precompute_gradient_attribution as mod
from errors import GradientAttributionError
from steering.feature_io import (
    CorruptFeatureShardError,
    glob_features,
    load_features,
    save_features,
    shard_stem_from_path,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.overlay_io import (
    OverlayKind,
    OverlayStaleError,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.sae_resolver import (
    ActivationType,
    ResolvedSAE,
    SAEWidth,
)
from tests.steering.test_gradient_attribution_producer_integration import (  # noqa: F401, F811
    _PCT_CHECKSUM,
    CONTEXT_WINDOW,
    SAE_MAX_CONTEXT,
    SEG_SIZE,
    TINY_LAYER,
    TinyTok,
    _cfg,
    _corpus_template,
    corpus,
    patched_builders,
)

_DELETE = object()


@pytest.fixture()
def construction_spy(monkeypatch, patched_builders):
    calls: Counter[str] = Counter()
    original_tokenizer = mod.resolve_tokenizer
    original_sae_loader = mod.SAE.from_pretrained
    original_model = mod.build_gradient_model

    def tokenizer(*args, **kwargs):
        calls["tokenizer"] += 1
        return original_tokenizer(*args, **kwargs)

    def sae_loader(*args, **kwargs):
        calls["sae"] += 1
        return original_sae_loader(*args, **kwargs)

    def model(*args, **kwargs):
        calls["model"] += 1
        return original_model(*args, **kwargs)

    monkeypatch.setattr(mod, "resolve_tokenizer", tokenizer)
    monkeypatch.setattr(mod.SAE, "from_pretrained", sae_loader)
    monkeypatch.setattr(mod, "build_gradient_model", model)
    return calls, patched_builders


def _assert_preflight_rejection(corpus: dict, calls: Counter[str]) -> None:
    assert calls == Counter()
    assert not (corpus["feature_dir"] / "grad").exists()


def _refresh_percentile_manifest(feature_dir: Path) -> None:
    write_manifest(
        overlay_dir(feature_dir, OverlayKind.PERCENTILE, _PCT_CHECKSUM),
        _PCT_CHECKSUM,
        OverlayKind.PERCENTILE,
        base_shard_paths=glob_features(feature_dir),
    )


def _rewrite_metadata(
    feature_dir: Path,
    field: str,
    value: object,
    *,
    shard_index: int = 1,
    refresh_percentile_manifest: bool = True,
) -> None:
    path = glob_features(feature_dir)[shard_index]
    data = load_features(path)
    if value is _DELETE:
        data.pop(field, None)
    else:
        data[field] = value
    save_features(data, path)
    if refresh_percentile_manifest:
        _refresh_percentile_manifest(feature_dir)


def _copy_corpus_to(corpus: dict, feature_dir: Path) -> dict:
    shutil.copytree(corpus["feature_dir"], feature_dir)
    return {"feature_dir": feature_dir, "data_dir": corpus["data_dir"]}


class TestPercentileOverlayPreflight:
    def test_stale_percentile_manifest_is_refused_before_construction(
        self, corpus, construction_spy
    ):
        calls, _ = construction_spy
        base = glob_features(corpus["feature_dir"])[0]
        data = load_features(base)
        data["feature_mags"] = data["feature_mags"].clone()
        data["feature_mags"][0] += 1.0
        save_features(data, base)
        # Same tensor geometry, different base content; the old pct manifest is stale.
        with pytest.raises(OverlayStaleError):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)

    @pytest.mark.parametrize("coverage", ["missing", "extra"])
    def test_percentile_coverage_must_be_exact(
        self, corpus, construction_spy, coverage
    ):
        calls, _ = construction_spy
        pct_dir = overlay_dir(
            corpus["feature_dir"], OverlayKind.PERCENTILE, _PCT_CHECKSUM
        )
        if coverage == "missing":
            overlay_shard_path(pct_dir, "shard_00001").unlink()
        else:
            save_overlay_shard(
                {"feature_percentiles": t.ones(1, dtype=t.uint8)},
                overlay_shard_path(pct_dir, "shard_99999"),
            )
        with pytest.raises(OverlayStaleError, match="coverage"):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)

    @pytest.mark.parametrize("corruption", ["keys", "rank", "dtype", "length"])
    def test_each_percentile_payload_is_structurally_validated(
        self, corpus, construction_spy, corruption
    ):
        calls, _ = construction_spy
        base = glob_features(corpus["feature_dir"])[0]
        stem = shard_stem_from_path(base)
        n = len(load_features(base)["feature_ids"])
        values = t.ones(n, dtype=t.uint8)
        tensors = {"feature_percentiles": values}
        if corruption == "keys":
            tensors["unexpected"] = t.ones(1, dtype=t.uint8)
        elif corruption == "rank":
            tensors["feature_percentiles"] = values.reshape(1, -1)
        elif corruption == "dtype":
            tensors["feature_percentiles"] = values.to(t.int32)
        else:
            tensors["feature_percentiles"] = values[:-1]
        save_overlay_shard(
            tensors,
            overlay_shard_path(
                overlay_dir(
                    corpus["feature_dir"], OverlayKind.PERCENTILE, _PCT_CHECKSUM
                ),
                stem,
            ),
        )
        with pytest.raises(CorruptFeatureShardError):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)


class TestBatchSizeAndCoverage:
    @pytest.mark.parametrize("bad", [-1, 0, True, 1.5, "2"])
    def test_run_rejects_non_positive_or_non_integer_batch_size(
        self, corpus, construction_spy, bad
    ):
        calls, _ = construction_spy
        with pytest.raises(GradientAttributionError, match="batch_size"):
            mod.run(_cfg(corpus, batch_size=bad))
        _assert_preflight_rejection(corpus, calls)

    @pytest.mark.parametrize("bad", [-1, 0, True, 1.5])
    def test_precompute_shard_rejects_bad_batch_before_loading(
        self, monkeypatch, tmp_path, bad
    ):
        monkeypatch.setattr(
            mod,
            "load_features",
            lambda *_args, **_kwargs: pytest.fail("base shard loaded before batch guard"),
        )
        with pytest.raises(GradientAttributionError, match="batch_size"):
            mod.precompute_shard(
                tmp_path / "base.safetensors.zst",
                tmp_path / "source.parquet",
                object(),  # ty: ignore[invalid-argument-type]
                TinyTok(),
                context_window=64,
                sae_max_context=256,
                segment_size=8,
                batch_size=bad,
                output_dir=tmp_path / "grad",
                shard_stem="shard_00000",
                pct_overlay_dir=tmp_path / "pct",
            )

    def test_precompute_shard_rejects_turn_delimited_base_shard(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(
            mod,
            "load_features",
            lambda *_args, **_kwargs: {
                "segment_mode": "turn_delimited_fixed_tokens"
            },
        )
        with pytest.raises(
            GradientAttributionError,
            match="gradient-attribution precompute supports fixed_tokens base shards only",
        ):
            mod.precompute_shard(
                tmp_path / "base.safetensors.zst",
                tmp_path / "source.parquet",
                object(),  # ty: ignore[invalid-argument-type]
                TinyTok(),
                context_window=64,
                sae_max_context=256,
                segment_size=8,
                batch_size=1,
                output_dir=tmp_path / "grad",
                shard_stem="shard_00000",
                pct_overlay_dir=tmp_path / "pct",
            )

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 16])
    def test_positive_batch_sizes_process_every_document_exactly_once(
        self, corpus, patched_builders, monkeypatch, batch_size
    ):
        real_collector = mod.GradientAttributionCollector
        seen: Counter[tuple[int, ...]] = Counter()

        class RecordingCollector:
            def __init__(self, *args, **kwargs):
                self.delegate = real_collector(*args, **kwargs)

            def score_documents(self, token_ids, *args, **kwargs):
                seen.update(tuple(ids) for ids in token_ids)
                return self.delegate.score_documents(token_ids, *args, **kwargs)

        monkeypatch.setattr(mod, "GradientAttributionCollector", RecordingCollector)
        assert mod.run(_cfg(corpus, batch_size=batch_size)) == 0
        assert sum(seen.values()) == 8
        assert set(seen.values()) == {1}


class TestBaseMetadataPreflight:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("layer", TINY_LAYER + 1),
            ("context_window", 63),
            ("context_independent_features", True),
            ("pooling", "mean_pool"),
            ("segment_size", SEG_SIZE + 1),
        ],
    )
    def test_cross_shard_identity_mismatch_is_rejected(
        self, corpus, construction_spy, field, value
    ):
        calls, _ = construction_spy
        _rewrite_metadata(corpus["feature_dir"], field, value)
        with pytest.raises(GradientAttributionError, match=field):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)

    @pytest.mark.parametrize(
        "field",
        ["layer", "context_window", "context_independent_features", "pooling", "segment_size"],
    )
    def test_required_identity_metadata_may_not_be_missing(
        self, corpus, construction_spy, field
    ):
        calls, _ = construction_spy
        _rewrite_metadata(corpus["feature_dir"], field, _DELETE)
        with pytest.raises(GradientAttributionError, match=field):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("layer", "not-an-int"),
            ("context_window", "not-an-int"),
            ("context_independent_features", "TRUE"),
            ("pooling", "not-a-pool"),
            ("segment_size", "not-an-int"),
        ],
    )
    def test_required_identity_metadata_may_not_be_malformed(
        self, corpus, construction_spy, field, value
    ):
        calls, _ = construction_spy
        _rewrite_metadata(corpus["feature_dir"], field, value)
        with pytest.raises(GradientAttributionError, match=field):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)


class TestBaseDocumentIdentityPreflight:
    def test_reordered_equal_geometry_doc_blocks_fail_before_construction(
        self, corpus, construction_spy
    ):
        calls, _ = construction_spy
        base = glob_features(corpus["feature_dir"])[0]
        data = load_features(base)
        data["doc_ids"] = data["doc_ids"].flip(0)
        save_features(data, base)
        _refresh_percentile_manifest(corpus["feature_dir"])

        with pytest.raises(GradientAttributionError, match="canonical"):
            mod.run(_cfg(corpus))
        _assert_preflight_rejection(corpus, calls)

    def test_equal_geometry_token_drift_fails_for_missing_identity_witness(
        self, corpus, tmp_path
    ):
        """Different token ids can retain every stored alignment dimension."""
        base_path = glob_features(corpus["feature_dir"])[0]
        stem = shard_stem_from_path(base_path)
        parquet_path = corpus["data_dir"] / f"{stem}.parquet"
        table = pq.read_table(parquet_path)
        texts = table.column("text").to_pylist()
        replacement = "q" * len(texts[0])
        tokenizer = TinyTok()
        assert tokenizer.encode(replacement) != tokenizer.encode(texts[0])
        assert len(tokenizer.encode(replacement)) == len(tokenizer.encode(texts[0]))
        pq.write_table(
            table.set_column(0, "text", pa.array([replacement, *texts[1:]])),
            parquet_path,
        )

        class ScoringMustNotStart:
            def score_documents(self, *_args, **_kwargs):
                pytest.fail("gradient scoring started without a token identity witness")

        with pytest.raises(
            NotImplementedError,
            match=(
                "verifying base-shard token-stream identity is not implemented.*"
                "base-shard identity witness.*feature_selection.gradient"
            ),
        ):
            mod.precompute_shard(
                base_path,
                parquet_path,
                ScoringMustNotStart(),  # ty: ignore[invalid-argument-type]
                tokenizer,
                context_window=CONTEXT_WINDOW,
                sae_max_context=SAE_MAX_CONTEXT,
                segment_size=SEG_SIZE,
                batch_size=3,
                output_dir=tmp_path / "grad",
                shard_stem=stem,
                pct_overlay_dir=overlay_dir(
                    corpus["feature_dir"], OverlayKind.PERCENTILE, _PCT_CHECKSUM
                ),
                dry_run=True,
            )


class TestCanonicalFeaturePathIdentity:
    def test_unmarked_explicit_copy_is_rejected(self, corpus, construction_spy, tmp_path):
        calls, _ = construction_spy
        relocated = _copy_corpus_to(corpus, tmp_path / "custom" / "s8")
        with pytest.raises(GradientAttributionError, match="canonical feature path"):
            mod.run(_cfg(relocated))
        _assert_preflight_rejection(relocated, calls)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("activation_type", ActivationType.mlp),
            ("model", "gemma-2-9b"),
            ("layer", TINY_LAYER + 1),
            ("width", SAEWidth.W_32K),
            ("pooling", PoolingMode.mean_pool),
            ("segment_size", SEG_SIZE + 1),
        ],
    )
    def test_canonical_path_component_must_match_run(
        self, corpus, construction_spy, tmp_path, field, value
    ):
        calls, _ = construction_spy
        params = {
            "segment_size": SEG_SIZE,
            "layer": TINY_LAYER,
            "activation_type": ActivationType.residual,
            "pooling": PoolingMode.max_pool,
            "width": SAEWidth.W_16K,
            "model": "gemma-2-2b",
        }
        params[field] = value
        path = build_feature_dir(
            tmp_path / "relocated",
            dataset_phase=DatasetPhase.BASE,
            **params,  # ty: ignore[invalid-argument-type]
        )
        relocated = _copy_corpus_to(corpus, path)
        with pytest.raises(GradientAttributionError, match=field):
            mod.run(_cfg(relocated))
        _assert_preflight_rejection(relocated, calls)


class TestL0TierPreflight:
    def _stamp_all(self, corpus: dict, tiers: list[str]) -> None:
        for path, tier in zip(glob_features(corpus["feature_dir"]), tiers, strict=True):
            data = load_features(path)
            data["l0_tier"] = tier
            save_features(data, path)
        _refresh_percentile_manifest(corpus["feature_dir"])

    def test_matching_v2_tier_is_accepted(
        self, corpus, patched_builders, monkeypatch
    ):
        self._stamp_all(corpus, ["medium", "medium"])
        monkeypatch.setattr(
            mod,
            "resolve_sae",
            lambda **_kwargs: ResolvedSAE(
                release="tiny",
                sae_id="layer_2_width_16k_l0_medium",
                hook_name=f"blocks.{TINY_LAYER}.hook_resid_post",
            ),
        )
        assert mod.run(
            _cfg(corpus, sae={"release": "gemmascope_2", "l0": "medium"})
        ) == 0

    @pytest.mark.parametrize("tiers", [["small", "small"], ["medium", "small"]])
    def test_wrong_or_mixed_v2_tier_is_rejected(
        self, corpus, construction_spy, monkeypatch, tiers
    ):
        calls, _ = construction_spy
        self._stamp_all(corpus, tiers)
        monkeypatch.setattr(
            mod,
            "resolve_sae",
            lambda **_kwargs: ResolvedSAE(
                release="tiny",
                sae_id="layer_2_width_16k_l0_medium",
                hook_name=f"blocks.{TINY_LAYER}.hook_resid_post",
            ),
        )
        with pytest.raises(GradientAttributionError, match="l0_tier"):
            mod.run(
                _cfg(corpus, sae={"release": "gemmascope_2", "l0": "medium"})
            )
        _assert_preflight_rejection(corpus, calls)


class TestObsoleteIgnoreBosConfig:
    @pytest.mark.parametrize("stale", [True, False, None, "malformed"])
    def test_any_supplied_value_is_rejected(
        self, corpus, construction_spy, stale
    ):
        calls, _ = construction_spy
        with pytest.raises(GradientAttributionError, match="remove.*ignore_bos_features"):
            mod.run(_cfg(corpus, ignore_bos_features=stale))
        _assert_preflight_rejection(corpus, calls)


class TestLoadedSaeHookPreflight:
    @pytest.mark.parametrize("kind", ["resid_pre", "wrong_layer", "missing", "conflicting"])
    def test_loaded_hook_mismatch_never_builds_model_or_writes_manifest(
        self, corpus, construction_spy, kind
    ):
        calls, sae = construction_spy
        expected = f"blocks.{TINY_LAYER}.hook_resid_post"
        if kind == "resid_pre":
            sae.cfg.hook_name = f"blocks.{TINY_LAYER}.hook_resid_pre"
        elif kind == "wrong_layer":
            sae.cfg.hook_name = f"blocks.{TINY_LAYER + 1}.hook_resid_post"
        elif kind == "missing":
            del sae.cfg.hook_name
        else:
            sae.cfg.hook_name = expected
            sae.cfg.metadata.hook_name = f"blocks.{TINY_LAYER + 1}.hook_resid_post"

        with pytest.raises(GradientAttributionError, match="hook"):
            mod.run(_cfg(corpus))
        assert calls["sae"] == 1
        assert calls["model"] == 0
        assert not (corpus["feature_dir"] / "grad").exists()
