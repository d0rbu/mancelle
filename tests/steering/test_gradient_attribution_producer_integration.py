"""Integration tests for scripts.preprocess.precompute_gradient_attribution.

End-to-end on a tiny corpus: a real parquet shard pair → REAL base extraction
(``process_shard`` with a tiny TL-style model + a real sae_lens tiny SAE, so
the producer's re-tokenization/alignment contract is exercised honestly) →
the gradient overlay producer with its heavy builders monkeypatched to tiny
equivalents. Covers: overlay layout + self-contained join tensors, manifest
(schema v2 + descriptor), score parity vs a direct collector call,
determinism, resume/force, checksum sensitivity, alignment-drift and
metadata-mismatch refusals, consume-time staleness, and DDP striping.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import yaml
from omegaconf import OmegaConf

import scripts.preprocess.precompute_gradient_attribution as mod
import steering.overlay_io as overlay_io
from errors import GradientAttributionError
from scripts.analysis.gradient_vs_magnitude_examples import build_examples
from steering.conditional.freq_blacklist import (
    BlacklistResult,
    blacklist_paths,
    write_blacklist,
)
from steering.extract_features import _truncate_to_training_window, process_shard
from steering.feature_io import (
    glob_features,
    load_features,
    save_features,
    shard_stem_from_path,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.gradient_attribution import GradientAttributionCollector
from steering.gradient_overlay import (
    GRAD_DOC_SEG_OFFSETS_KEY,
    GRAD_OFFSETS_KEY,
    GRAD_SCORES_KEY,
    GradientAttributionMode,
)
from steering.overlay_io import (
    OverlayKind,
    OverlayStaleError,
    discover_overlay,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    write_manifest,
)
from steering.refined_types import PositiveInt
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode, segment_tokens
from tests.reference_gradient_attribution import (
    TINY_BOS,
    TINY_D_SAE,
    TINY_LAYER,
    TINY_VOCAB,
    make_tiny_sae,
    tiny_gemma2,
)

SEG_SIZE = 8
CONTEXT_WINDOW = 64
SAE_MAX_CONTEXT = 256
D_MODEL = 32


class TinyTok:
    """Deterministic char-level tokenizer with ids < TINY_VOCAB and BOS=2.

    Satisfies every contract the extraction + producer paths touch:
    ``__call__(text) -> {"input_ids": [...]}``, ``decode`` (progressive,
    1 char per token — monotone for char offsets), ``bos_token_id``.
    Module-level and stateless => picklable.
    """

    bos_token_id = TINY_BOS

    def __call__(self, text, padding: bool = False, **_kw):
        # Mirror an HF tokenizer: a LIST input returns one id-list per text
        # (each encoded independently — what batched tokenization relies on).
        if isinstance(text, list):
            return {"input_ids": [self.encode(t) for t in text]}
        return {"input_ids": self.encode(text)}

    def encode(self, text: str) -> list[int]:
        return [TINY_BOS] + [3 + (ord(c) % (TINY_VOCAB - 3)) for c in text]

    def decode(self, token_ids) -> str:
        return "x" * len(list(token_ids))


class TinyResidModel:
    """Minimal TransformerLens-shaped model for base extraction: activations
    are a seeded embedding lookup at the hook point (differentiability not
    needed for extraction)."""

    def __init__(self):
        self.tokenizer = TinyTok()
        self.cfg = SimpleNamespace(n_ctx=SAE_MAX_CONTEXT, model_name="gemma-2-2b")
        gen = t.Generator().manual_seed(123)
        self._embedding = t.randn(TINY_VOCAB, D_MODEL, generator=gen) * 0.5
        self._param = t.nn.Parameter(t.zeros(1))

    def parameters(self):
        yield self._param

    def run_with_cache(self, tokens, names_filter, attention_mask=None, stop_at_layer=None):
        del attention_mask, stop_at_layer
        acts = self._embedding[tokens.clamp(0, TINY_VOCAB - 1)]
        cache = dict.fromkeys(names_filter, acts)
        return None, cache


def _extraction_sae():
    """The base-extraction SAE: every feature fires everywhere (positive
    bias) so no segment is ever empty (issue #522 invariant)."""
    sae = make_tiny_sae("standard", seed=77)
    with t.no_grad():
        sae.b_enc.abs_().add_(0.05)
    return sae


TEXTS_A = [
    "the quick brown fox jumps over the lazy dog and keeps going",
    "a" * 90,  # > context_window tokens => head-truncated (n_full > kept)
    "tiny",  # 5 tokens incl BOS => single short segment
    "exactly fifteen!",  # 17 tokens => 8 + 8 + 1: len-1 final segment
    "z",  # 2 tokens
]
TEXTS_B = [
    "another document with several segments worth of text in it ok",
    "short one",
    "mid sized document right here",
]


# Deterministic percentile overlay for the corpus — the grad producer now bakes
# pct * sensitivity, so it requires one. uint8 in [1, 255] (never 0, so the
# score>0 eligibility set matches the raw sensitivity).
_PCT_CHECKSUM = "pct" + "0" * 61  # 64 chars; the dir leaf is the [:16] prefix


def _canonical_feature_dir(root: Path) -> Path:
    return build_feature_dir(
        root / "features",
        segment_size=SEG_SIZE,
        layer=TINY_LAYER,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )


def _write_synthetic_pct_overlay(feature_dir: Path) -> None:
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, _PCT_CHECKSUM)
    base_shards = glob_features(feature_dir)
    for base in base_shards:
        n = int(len(load_features(base)["feature_ids"]))
        pcts = (((t.arange(n, dtype=t.int64) * 37 + 13) % 255) + 1).to(t.uint8)
        save_overlay_shard(
            {"feature_percentiles": pcts},
            overlay_shard_path(pct_dir, shard_stem_from_path(base)),
        )
    write_manifest(
        pct_dir, _PCT_CHECKSUM, OverlayKind.PERCENTILE, base_shard_paths=base_shards
    )


def _write_freq_blacklist(feature_dir: Path, ids: tuple[int, ...] = ()) -> None:
    """Materialize the activation-frequency blacklist artifact the analysis tool
    reads (it ranks over the production candidate set and fails loud if absent).
    Empty ``ids`` => the tool keeps every feature (preserves pre-filter output)."""
    txt, meta = blacklist_paths(feature_dir, 0.0, 0.99)
    txt.parent.mkdir(parents=True, exist_ok=True)
    write_blacklist(
        txt, meta,
        BlacklistResult(
            ids=ids, n_sae_features=4096, n_segments=8, n_active_features=8,
            freq_min=0.0, freq_max=0.99, stats_generated_at="test",
        ),
    )


def _pct_weights_for_shard(feature_dir: Path, stem: str) -> t.Tensor:
    """The corpus pct overlay's [0,1] weights for ``stem`` — exactly what the
    producer multiplies the gradient sensitivity by."""
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, _PCT_CHECKSUM)
    u8 = load_overlay_shard(overlay_shard_path(pct_dir, stem))["feature_percentiles"]
    return u8.to(t.float32) / 255.0


@pytest.fixture(scope="module")
def _corpus_template(tmp_path_factory) -> dict:
    """Parquet shards + real base extraction (built once; copied per test)."""
    root = tmp_path_factory.mktemp("grad_producer_template")
    data_dir = root / "parquet"
    data_dir.mkdir()
    feature_dir = _canonical_feature_dir(root)
    feature_dir.mkdir(parents=True)

    model = TinyResidModel()
    sae = _extraction_sae()
    for stem, texts in [("shard_00000", TEXTS_A), ("shard_00001", TEXTS_B)]:
        pq.write_table(pa.table({"text": texts}), data_dir / f"{stem}.parquet")
        process_shard(
            str(data_dir / f"{stem}.parquet"),
            cast(Any, model),
            sae,
            layer=TINY_LAYER,
            output_dir=str(feature_dir),
            context_window=CONTEXT_WINDOW,
            pooling=PoolingMode.max_pool,
            segment_size=SEG_SIZE,
            hook_name=f"blocks.{TINY_LAYER}.hook_resid_post",
            activation_type=ActivationType.residual,
            pipeline_post=False,
        )
    _write_synthetic_pct_overlay(feature_dir)
    return {"data_dir": data_dir, "feature_dir": feature_dir}


@pytest.fixture()
def corpus(_corpus_template, tmp_path) -> dict:
    """A fresh per-test copy (tests write grad/ overlays into feature_dir;
    the full canonical path is required by the producer identity preflight)."""
    import shutil

    data_dir = tmp_path / "parquet"
    feature_dir = _canonical_feature_dir(tmp_path)
    shutil.copytree(_corpus_template["data_dir"], data_dir)
    shutil.copytree(_corpus_template["feature_dir"], feature_dir)
    return {"data_dir": data_dir, "feature_dir": feature_dir}


def _cfg(corpus: dict, **overrides):
    base = {
        "feature_dir": str(corpus["feature_dir"]),
        "data_dir": str(corpus["data_dir"]),
        "dataset_phase": None,
        "model": "gemma-2-2b",
        "activation_type": "residual",
        "layer": TINY_LAYER,
        "pooling": "max_pool",
        "segment_size": SEG_SIZE,
        "sae_width": "16k",
        "feature_root": "unused",
        "sae": {"release": "gemmascope", "l0": None},
        "context_window": CONTEXT_WINDOW,
        "attribution_mode": "future_half_split",
        "min_final_segment_tokens": 4,
        "device": "cpu",
        "model_dtype": "float32",
        "batch_size": 3,
        "resume": False,
        "force": False,
        "dry_run": False,
        "zstd_level": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


@pytest.fixture()
def patched_builders(monkeypatch):
    """Swap the heavy production builders for tiny equivalents."""
    grad_sae = make_tiny_sae("jumprelu", seed=11)
    grad_sae.cfg.hook_name = f"blocks.{TINY_LAYER}.hook_resid_post"
    monkeypatch.setattr(mod, "resolve_tokenizer", lambda _name: TinyTok())
    monkeypatch.setattr(
        mod, "build_gradient_model", lambda _n, _d, _dt: tiny_gemma2(softcap=True)
    )
    monkeypatch.setattr(
        mod,
        "SAE",
        SimpleNamespace(from_pretrained=lambda release, sae_id, device: grad_sae),
    )
    monkeypatch.setattr(
        mod,
        "resolve_sae",
        lambda **_kw: SimpleNamespace(
            release="tiny", sae_id="tiny", hook_name=f"blocks.{TINY_LAYER}.hook_resid_post"
        ),
    )
    monkeypatch.setattr(
        mod,
        "AutoConfig",
        SimpleNamespace(
            from_pretrained=lambda _id: SimpleNamespace(
                max_position_embeddings=SAE_MAX_CONTEXT
            )
        ),
    )
    # The tiny SAE has d_sae=TINY_D_SAE; map the config's width knob to it so
    # the producer's loaded-width guard passes (the guard itself is exercised
    # in test_width_mismatch_refused).
    monkeypatch.setattr(
        mod, "SAE_WIDTH_TO_NUM_FEATURES", {SAEWidth("16k"): TINY_D_SAE}
    )
    # These tests pin the quarantined scoring math and artifact contracts. The
    # production-only GA-001 stop has its own equal-geometry drift regression.
    monkeypatch.setattr(
        mod, "_raise_for_unwitnessed_base_token_stream_identity", lambda: None
    )
    return grad_sae


def _expected_scores_for_shard(
    corpus: dict, stem: str, grad_sae, mode: GradientAttributionMode, min_final: int,
    *, ignore_bos: bool = False, apply_pct: bool = True,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Direct collector invocation on the same inputs the producer derives.

    ``apply_pct`` (default) multiplies the collector's gradient sensitivity by
    the percentile weight, matching the producer's stored ``pct * sensitivity``;
    set False to get the bare sensitivity (for non-vacuity checks)."""
    tok = TinyTok()
    data = load_features(corpus["feature_dir"] / f"{stem}_features.safetensors.zst")
    table = pq.read_table(corpus["data_dir"] / f"{stem}.parquet")
    texts = table.column("text").to_pylist()
    offsets = data["offsets"].to(t.int64)
    fids = data["feature_ids"].to(t.int64)
    doc_ids = data["doc_ids"].to(t.int64)
    n_docs = int(data["n_docs"])

    collector = GradientAttributionCollector(
        tiny_gemma2(softcap=True),
        grad_sae,
        layer=TINY_LAYER,
        mode=mode,
        min_final_segment_tokens=min_final,
        bos_token_id=TINY_BOS if ignore_bos else None,
        ignore_bos_features=ignore_bos,
    )
    flat = t.zeros(len(fids), dtype=t.float16)
    counts = t.bincount(doc_ids, minlength=n_docs)
    doc_seg_offsets = t.cat([t.zeros(1, dtype=t.int64), counts.cumsum(0)])
    for d in range(n_docs):
        kept, _ = _truncate_to_training_window(
            texts[d], tok, CONTEXT_WINDOW, SAE_MAX_CONTEXT
        )
        segs = segment_tokens(
            kept,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(SEG_SIZE),
        )
        lo, hi = int(doc_seg_offsets[d]), int(doc_seg_offsets[d + 1])
        fid_rows = [fids[int(offsets[s]) : int(offsets[s + 1])] for s in range(lo, hi)]
        rows = collector.score_documents([kept], [segs], [fid_rows])[0]
        for s, row in zip(range(lo, hi), rows, strict=True):
            flat[int(offsets[s]) : int(offsets[s + 1])] = row
    if apply_pct:
        # The producer bakes in the percentile weight: stored = pct * sensitivity
        # (fp32 multiply then fp16, matching precompute_shard exactly).
        flat = (
            flat.to(t.float32) * _pct_weights_for_shard(corpus["feature_dir"], stem)
        ).to(t.float16)
    return flat, offsets, doc_seg_offsets.to(t.int32)


class TestProducerHappyPath:
    def test_end_to_end(self, corpus, patched_builders):
        grad_sae = patched_builders
        cfg = _cfg(corpus)
        assert mod.run(cfg) == 0

        grad_dirs = list((corpus["feature_dir"] / "grad").iterdir())
        assert len(grad_dirs) == 1
        overlay = grad_dirs[0]
        assert len(overlay.name) == 16

        manifest = read_manifest(overlay)
        assert manifest is not None
        assert manifest["kind"] == "grad"
        assert manifest["schema_version"] == 2
        assert manifest["checksum"].startswith(overlay.name)
        assert manifest["metadata"]["attribution_mode"] == "future_half_split"
        assert manifest["metadata"]["min_final_segment_tokens"] == 4
        assert manifest["metadata"]["score_reduction"] == "pct_relu_neg_max"
        assert manifest["metadata"]["percentile_checksum"] == _PCT_CHECKSUM
        assert "base_shard_content_hash" in manifest

        shards = glob_overlay_shards(overlay)
        assert [p.name for p in shards] == [
            "shard_00000.safetensors.zst",
            "shard_00001.safetensors.zst",
        ]

        for base_path in glob_features(corpus["feature_dir"]):
            stem = shard_stem_from_path(base_path)
            got = load_overlay_shard(overlay / f"{stem}.safetensors.zst")
            base = load_features(base_path)
            assert got[GRAD_SCORES_KEY].dtype == t.float16
            assert got[GRAD_OFFSETS_KEY].dtype == t.int64
            assert got[GRAD_DOC_SEG_OFFSETS_KEY].dtype == t.int32
            assert len(got[GRAD_SCORES_KEY]) == len(base["feature_ids"])
            assert t.equal(got[GRAD_OFFSETS_KEY], base["offsets"].to(t.int64))
            expected, offsets, dso = _expected_scores_for_shard(
                corpus, stem, grad_sae,
                GradientAttributionMode.future_half_split, 4,
            )
            assert t.equal(got[GRAD_DOC_SEG_OFFSETS_KEY], dso)
            assert t.equal(got[GRAD_SCORES_KEY], expected), (
                f"{stem}: producer scores differ from a direct collector call"
            )
            assert bool((got[GRAD_SCORES_KEY].float() >= 0).all())

    def test_percentile_weighting_is_nonvacuous(self, corpus, patched_builders):
        """The stored score is pct * sensitivity, NOT bare sensitivity: the
        percentile weight (uint8/255 <= 1) demotes scores, so the overlay
        differs from — and is bounded above by — the raw gradient sensitivity."""
        grad_sae = patched_builders
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        got = load_overlay_shard(overlay / "shard_00000.safetensors.zst")[
            GRAD_SCORES_KEY
        ]
        m = GradientAttributionMode.future_half_split
        weighted, _, _ = _expected_scores_for_shard(corpus, "shard_00000", grad_sae, m, 4)
        raw, _, _ = _expected_scores_for_shard(
            corpus, "shard_00000", grad_sae, m, 4, apply_pct=False
        )
        assert t.equal(got, weighted)  # producer bakes in the percentile weight
        assert not t.equal(weighted, raw), "percentile weighting had no effect"
        assert bool((weighted.float() <= raw.float() + 1e-6).all())  # pct in [0,1]

    def test_determinism_and_force(self, corpus, patched_builders):
        cfg = _cfg(corpus)
        assert mod.run(cfg) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        before = {
            p.name: p.read_bytes() for p in glob_overlay_shards(overlay)
        }
        assert mod.run(_cfg(corpus, force=True)) == 0
        after = {p.name: p.read_bytes() for p in glob_overlay_shards(overlay)}
        assert before == after, "force rewrite must be byte-identical"

    def test_frozen_manifest_and_compressed_shard_byte_golden(
        self, corpus, patched_builders, monkeypatch
    ):
        monkeypatch.setattr(
            overlay_io.time,
            "strftime",
            lambda *_args, **_kwargs: "2000-01-01T00:00:00Z",
        )
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())

        def _artifact_bytes() -> dict[str, bytes]:
            paths = [overlay / "manifest.json", *glob_overlay_shards(overlay)]
            return {path.name: path.read_bytes() for path in paths}

        before = _artifact_bytes()
        assert mod.run(_cfg(corpus, force=True)) == 0
        after = _artifact_bytes()
        assert after == before
        expected_manifest = {
            "checksum": (
                "69ffe042f88d05871c9c15111cb42f82608034918185ca315767f3e26671c25f"
            ),
            "kind": "grad",
            "created_at": "2000-01-01T00:00:00Z",
            "checksum_prefix": "69ffe042f88d0587",
            "metadata": {
                "attribution_mode": "future_half_split",
                "score_reduction": "pct_relu_neg_max",
                "split_policy_version": 2,
                "min_final_segment_tokens": 4,
                "model": "gemma-2-2b",
                "model_dtype": "float32",
                "layer": 2,
                "sae_release": "gemmascope",
                "sae_width": "16k",
                "sae_l0": None,
                "activation_type": "residual",
                "ignore_bos_features": False,
                "percentile_checksum": _PCT_CHECKSUM,
                "segment_size": SEG_SIZE,
                "context_window": CONTEXT_WINDOW,
            },
            "base_shard_content_hash": overlay_io.compute_base_shard_content_hash(
                glob_features(corpus["feature_dir"])
            ),
            "schema_version": 2,
        }
        assert after["manifest.json"] == (
            json.dumps(expected_manifest, indent=2) + "\n"
        ).encode()
        digests = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in after.items()
            if name != "manifest.json"
        }
        # Frozen from the pre-PR3 HEAD producer under the same fixed timestamp;
        # the manifest is asserted directly above because its base-shard hash
        # legitimately includes each fixture's source-path metadata.
        assert digests == {
            "shard_00000.safetensors.zst": (
                "cde55eb3f7b76eefed9365a9a8c3958a3ca61ad28510f1841bf2cf3295bf03d9"
            ),
            "shard_00001.safetensors.zst": (
                "3c74cb0dbb29bcd05a1a5b01db3a771201cef8946de7d0a92de7a3e7eec2eea5"
            ),
        }

    def test_short_circuit_and_resume(self, corpus, patched_builders):
        cfg = _cfg(corpus)
        assert mod.run(cfg) == 0
        # Without resume/force the second run is a no-op short-circuit.
        assert mod.run(cfg) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        victim = overlay / "shard_00001.safetensors.zst"
        keeper = overlay / "shard_00000.safetensors.zst"
        keeper_bytes = keeper.read_bytes()
        victim.unlink()
        assert mod.run(_cfg(corpus, resume=True)) == 0
        assert victim.exists()
        assert keeper.read_bytes() == keeper_bytes

    def test_mode_changes_overlay_dir_and_scores(self, corpus, patched_builders):
        assert mod.run(_cfg(corpus)) == 0
        assert mod.run(_cfg(corpus, attribution_mode="self")) == 0
        grad_root = corpus["feature_dir"] / "grad"
        dirs = sorted(p.name for p in grad_root.iterdir())
        assert len(dirs) == 2, "different modes must land in different overlay dirs"
        a, b = (
            load_overlay_shard(grad_root / d / "shard_00000.safetensors.zst")[
                GRAD_SCORES_KEY
            ]
            for d in dirs
        )
        assert not t.equal(a, b), "future vs self scores must differ"

    def test_dry_run_writes_nothing(self, corpus, patched_builders):
        assert mod.run(_cfg(corpus, dry_run=True)) == 0
        grad_root = corpus["feature_dir"] / "grad"
        for d in grad_root.iterdir() if grad_root.is_dir() else []:
            assert glob_overlay_shards(d) == []

    def test_consume_time_staleness(self, corpus, patched_builders):
        cfg = _cfg(corpus)
        assert mod.run(cfg) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        manifest = read_manifest(overlay)
        assert manifest is not None
        checksum = manifest["checksum"]
        base_paths = glob_features(corpus["feature_dir"])
        # Matching base shards: discovery succeeds.
        assert (
            discover_overlay(
                corpus["feature_dir"], OverlayKind.GRADIENT, checksum,
                base_shard_paths=base_paths,
            )
            is not None
        )
        # Tamper one base shard (re-save with a mutated magnitude).
        victim = base_paths[0]
        original = victim.read_bytes()
        try:
            data = load_features(victim)
            data["feature_mags"] = data["feature_mags"].clone()
            data["feature_mags"][0] += 1.0
            save_features(data, victim)
            with pytest.raises(OverlayStaleError):
                discover_overlay(
                    corpus["feature_dir"], OverlayKind.GRADIENT, checksum,
                    base_shard_paths=glob_features(corpus["feature_dir"]),
                )
        finally:
            victim.write_bytes(original)


class TestProducerRefusals:
    def test_alignment_drift_fails_loud(self, corpus, patched_builders, tmp_path):
        """A base shard whose recorded doc_token_counts disagree with
        re-tokenization must abort naming the shard and doc."""
        drift_dir = _canonical_feature_dir(tmp_path / "drift")
        drift_dir.mkdir(parents=True)
        src = corpus["feature_dir"] / "shard_00000_features.safetensors.zst"
        data = load_features(src)
        data["doc_token_counts"] = data["doc_token_counts"].clone()
        data["doc_token_counts"][0] += 3
        save_features(data, drift_dir / "shard_00000_features.safetensors.zst")
        _write_synthetic_pct_overlay(drift_dir)  # producer needs one to reach the loop
        cfg = _cfg(corpus, feature_dir=str(drift_dir))
        with pytest.raises(GradientAttributionError, match="re-tokenized"):
            mod.run(cfg)

    def test_context_independent_refused(self, corpus, patched_builders, tmp_path):
        ci_dir = _canonical_feature_dir(tmp_path / "context_independent") / "context_independent"
        ci_dir.mkdir(parents=True)
        src = corpus["feature_dir"] / "shard_00000_features.safetensors.zst"
        data = load_features(src)
        data["context_independent_features"] = True
        save_features(data, ci_dir / "shard_00000_features.safetensors.zst")
        _write_synthetic_pct_overlay(ci_dir)
        with pytest.raises(GradientAttributionError, match="context_independent_features"):
            mod.run(_cfg(corpus, feature_dir=str(ci_dir)))

    def test_stale_ignore_bos_config_key_is_fatal(self, corpus, patched_builders):
        """The obsolete key must be removed, regardless of its stale value."""
        with pytest.raises(GradientAttributionError, match="remove ignore_bos_features"):
            mod.run(_cfg(corpus, ignore_bos_features=True))
        assert not (corpus["feature_dir"] / "grad").exists()

    def test_non_residual_refused(self, corpus, patched_builders):
        with pytest.raises(GradientAttributionError, match="residual-only"):
            mod.run(_cfg(corpus, activation_type="mlp"))

    def test_missing_parquet_fails_loud(self, corpus, patched_builders, tmp_path):
        empty = tmp_path / "no_parquet"
        empty.mkdir()
        with pytest.raises(GradientAttributionError, match="source parquet"):
            mod.run(_cfg(corpus, data_dir=str(empty)))

    def test_segment_size_mismatch_refused(self, corpus, patched_builders):
        with pytest.raises(Exception, match="segment_size"):
            mod.run(_cfg(corpus, segment_size=16))


class TestDDPStriping:
    def test_two_ranks_cover_disjoint_shards(
        self, corpus, patched_builders, monkeypatch
    ):
        monkeypatch.setenv("WORLD_SIZE", "2")
        # Rank 1 first: writes its stripe, no manifest.
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        assert read_manifest(overlay) is None, "non-zero ranks must not write manifests"
        rank1_files = {p.name for p in glob_overlay_shards(overlay)}
        assert rank1_files == {"shard_00001.safetensors.zst"}
        # Rank 0: manifest + the other stripe.
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert mod.run(_cfg(corpus)) == 0
        assert read_manifest(overlay) is not None
        assert {p.name for p in glob_overlay_shards(overlay)} == {
            "shard_00000.safetensors.zst",
            "shard_00001.safetensors.zst",
        }

    def test_late_rank_with_partial_overlay_fails_loud(
        self, corpus, patched_builders, monkeypatch
    ):
        """The first-run race regression: rank 0 stamps the manifest BEFORE
        computing, so a rank arriving after rank 0 FINISHED its stripe finds
        manifest + partial shards. It must NOT silently exit 0 (dropping its
        stripe); it exits non-zero pointing at resume=true, which then fills
        exactly the missing stripe."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert mod.run(_cfg(corpus)) == 0  # manifest + stripe 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        assert {p.name for p in glob_overlay_shards(overlay)} == {
            "shard_00000.safetensors.zst"
        }
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert mod.run(_cfg(corpus)) == 1, (
            "a late rank seeing a partial overlay must fail loud, not skip"
        )
        assert mod.run(_cfg(corpus, resume=True)) == 0
        assert {p.name for p in glob_overlay_shards(overlay)} == {
            "shard_00000.safetensors.zst",
            "shard_00001.safetensors.zst",
        }

    def test_fresh_manifest_zero_shards_proceeds(
        self, corpus, patched_builders, monkeypatch
    ):
        """A manifest with ZERO shards is the concurrent-fresh-launch window
        (rank 0 stamps before any stripe computes): a flagless run proceeds
        as fresh instead of short-circuiting."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        for p in glob_overlay_shards(overlay):
            p.unlink()
        assert mod.run(_cfg(corpus)) == 0
        assert len(glob_overlay_shards(overlay)) == 2


class TestProducerHardening:
    def test_partial_overlay_flagless_rerun_exits_nonzero(
        self, corpus, patched_builders
    ):
        """Exit code 0 for a partial overlay would read as success to
        orchestration; a flagless rerun over an interrupted run must exit 1
        with resume/force guidance."""
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        (overlay / "shard_00001.safetensors.zst").unlink()
        assert mod.run(_cfg(corpus)) == 1
        # And the complete overlay short-circuits with 0.
        assert mod.run(_cfg(corpus, resume=True)) == 0
        assert mod.run(_cfg(corpus)) == 0

    def test_resume_over_reextracted_bases_refused(self, corpus, patched_builders):
        """The resume-launders-stale-overlay regression: after base shards
        are re-extracted in place, resume must raise OverlayStaleError BEFORE
        re-stamping the manifest — never re-certify old scores against the
        new bases."""
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        manifest_before = read_manifest(overlay)
        assert manifest_before is not None
        victim = glob_features(corpus["feature_dir"])[0]
        data = load_features(victim)
        data["feature_mags"] = data["feature_mags"].clone()
        data["feature_mags"][0] += 1.0
        save_features(data, victim)
        with pytest.raises(OverlayStaleError):
            mod.run(_cfg(corpus, resume=True))
        manifest_after = read_manifest(overlay)
        assert manifest_after is not None
        assert (
            manifest_after["base_shard_content_hash"]
            == manifest_before["base_shard_content_hash"]
        ), "the stale manifest must not be re-stamped"

    def test_resume_does_not_restamp_manifest(self, corpus, patched_builders):
        """A no-change resume neither rewrites shards nor the manifest."""
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        manifest_path = overlay / "manifest.json"
        before = manifest_path.read_bytes()
        assert mod.run(_cfg(corpus, resume=True)) == 0
        assert manifest_path.read_bytes() == before

    def test_width_mismatch_refused(self, corpus, patched_builders, monkeypatch):
        """The loaded SAE's d_sae must match the width knob that resolved the
        feature dir — scoring with a different-width SAE would persist
        silently wrong scores."""
        monkeypatch.setattr(
            mod, "SAE_WIDTH_TO_NUM_FEATURES", {SAEWidth("16k"): 16384}
        )
        with pytest.raises(GradientAttributionError, match="width mismatch"):
            mod.run(_cfg(corpus))

    def test_length_sorted_batching_matches_per_doc_scoring(
        self, corpus, patched_builders
    ):
        """Batches are length-sorted (pad-waste reduction); scores must still
        equal a direct per-doc collector run (the absolute-slice write-back
        and pad-invariance contract make processing order free)."""
        grad_sae = patched_builders
        assert mod.run(_cfg(corpus, batch_size=2)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        for base_path in glob_features(corpus["feature_dir"]):
            stem = shard_stem_from_path(base_path)
            got = load_overlay_shard(overlay / f"{stem}.safetensors.zst")
            expected, _, _ = _expected_scores_for_shard(
                corpus, stem, grad_sae,
                GradientAttributionMode.future_half_split, 4,
            )
            assert t.equal(got[GRAD_SCORES_KEY], expected)


class TestMaxPositionEmbeddings:
    """gemma-3 multimodal configs (4b/12b/27b) nest text params under
    text_config; the producer must read max_position_embeddings from there."""

    def test_flat_config_gemma2_style(self):
        cfg = SimpleNamespace(max_position_embeddings=8192)
        assert mod.max_position_embeddings(cfg) == 8192

    def test_nested_config_gemma3_multimodal_style(self):
        # Gemma3Config: no top-level max_position_embeddings; it lives under
        # text_config (the exact shape that crashed the real gemma-3-4b run).
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=131072),
            vision_config=SimpleNamespace(),
        )
        assert mod.max_position_embeddings(cfg) == 131072

    def test_producer_handles_nested_config(self, corpus, patched_builders, monkeypatch):
        """End-to-end: with a gemma-3-multimodal-style AutoConfig (nested
        text_config, NO top-level max_position_embeddings) the producer runs
        clean instead of AttributeError-ing at the sae_max_context read."""
        monkeypatch.setattr(
            mod,
            "AutoConfig",
            SimpleNamespace(
                from_pretrained=lambda _id: SimpleNamespace(
                    text_config=SimpleNamespace(
                        max_position_embeddings=SAE_MAX_CONTEXT
                    ),
                    vision_config=SimpleNamespace(),
                )
            ),
        )
        assert mod.run(_cfg(corpus)) == 0
        assert list((corpus["feature_dir"] / "grad").iterdir())


class TestActivationTypeGuard:
    """activation_type isn't in shard metadata, so the producer guards against
    scoring mlp/attention features with a residual SAE via the path's
    a_{label} component (gradient attribution is residual-only)."""

    def test_residual_path_ok(self):
        from pathlib import Path

        mod._assert_residual_feature_dir(
            Path("out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base")
        )

    def test_unmarked_path_refused(self):
        from pathlib import Path

        with pytest.raises(GradientAttributionError, match="canonical feature path"):
            mod._assert_residual_feature_dir(Path("/tmp/features/s8"))

    def test_mlp_and_attention_paths_refused(self):
        from pathlib import Path

        for marker in ("a_mlp", "a_att"):
            with pytest.raises(GradientAttributionError, match="residual-only"):
                mod._assert_residual_feature_dir(
                    Path(f"out/data/features/s128/l8/{marker}/p_max_pool/f16k/"
                         f"m_gemma-2-2b/base")
                )


class TestIgnoreBosEndToEnd:
    """The ignore_bos path through the PRODUCER (config flag -> collector
    keep-mask), end to end on the tiny corpus. The collector's keep-mask
    correctness is pinned separately in test_gradient_attribution_unit; this
    pins the producer PLUMBING + that BOS exclusion actually changes scores."""

    def _flag_no_bos(self, corpus):
        """Rewrite the base shards' metadata to ignore_bos_features=true (the
        producer INHERITS this flag; the tiny corpus is otherwise BOS-included)."""
        for base_path in glob_features(corpus["feature_dir"]):
            data = load_features(base_path)
            data["ignore_bos_features"] = True
            save_features(data, base_path)
        _write_synthetic_pct_overlay(corpus["feature_dir"])

    def test_producer_matches_collector_with_ignore_bos(self, corpus, patched_builders):
        grad_sae = patched_builders
        self._flag_no_bos(corpus)
        # No config knob: the producer INHERITS ignore_bos=true from the base
        # shards' metadata. Scores must match the collector's ignore_bos=True.
        assert mod.run(_cfg(corpus)) == 0
        overlay = next((corpus["feature_dir"] / "grad").iterdir())
        for base_path in glob_features(corpus["feature_dir"]):
            stem = shard_stem_from_path(base_path)
            got = load_overlay_shard(overlay / f"{stem}.safetensors.zst")
            expected, _, _ = _expected_scores_for_shard(
                corpus, stem, grad_sae,
                GradientAttributionMode.future_half_split, 4, ignore_bos=True,
            )
            assert t.equal(got[GRAD_SCORES_KEY], expected), (
                f"{stem}: producer ignore_bos scores != collector ignore_bos"
            )

    def test_stale_false_config_is_fatal_even_when_shards_are_true(
        self, corpus, patched_builders
    ):
        self._flag_no_bos(corpus)
        with pytest.raises(GradientAttributionError, match="remove ignore_bos_features"):
            mod.run(_cfg(corpus, ignore_bos_features=False))
        assert not (corpus["feature_dir"] / "grad").exists()

    def test_ignore_bos_changes_scores(self, corpus, patched_builders):
        """BOS exclusion is non-vacuous: scoring the SAME shards with vs
        without the keep-mask differs (segment 0 carries the BOS row)."""
        grad_sae = patched_builders
        without = _expected_scores_for_shard(
            corpus, "shard_00000", grad_sae,
            GradientAttributionMode.future_half_split, 4, ignore_bos=False,
        )[0]
        with_mask = _expected_scores_for_shard(
            corpus, "shard_00000", grad_sae,
            GradientAttributionMode.future_half_split, 4, ignore_bos=True,
        )[0]
        assert not t.equal(without, with_mask), (
            "ignore_bos had no effect — the BOS keep-mask is not reaching the "
            "score reduction"
        )


class TestGradientVsMagnitudeExamples:
    """The qualitative magnitude-vs-gradient YAML tool consumes the producer's
    real overlay (scripts.analysis.gradient_vs_magnitude_examples) — exercises
    the char-offset slicing, the grad gather, the two rankings, and YAML I/O."""

    def test_emits_comparable_examples(self, corpus, patched_builders):
        assert mod.run(_cfg(corpus)) == 0  # write the grad overlay first
        _write_freq_blacklist(corpus["feature_dir"])  # tool reads it (fails loud if absent)

        result = build_examples(
            corpus["feature_dir"],
            None,  # auto-resolve the single grad/<hash> overlay
            shard="shard_00000",
            num_examples=3,
            top_k=3,
            min_features=1,
            prefix_chars=50,
            suffix_chars=50,
            seed=0,
            data_dir=corpus["data_dir"],
            dataset_phase=None,
        )

        meta = result["meta"]
        assert meta["attribution_mode"] == "future_half_split"
        assert meta["model"] == "gemma-2-2b"
        assert meta["shard"] == "shard_00000"
        # cache-only (no neuronpedia_cache.yaml in the tmp corpus) => 0 found,
        # reported as "<found>/<displayed>".
        found, _, displayed = meta["descriptions_found"].partition("/")
        assert found == "0" and int(displayed) > 0
        assert result["examples"], "no non-final segment with features to compare"

        for ex in result["examples"]:
            assert ex["n_features"] >= 1
            assert isinstance(ex["segment"], str) and ex["segment"]
            assert 1 <= len(ex["by_magnitude"]) <= 3
            assert len(ex["by_gradient"]) == len(ex["by_magnitude"])
            # each ranking is sorted by its own channel, descending
            mags = [float(ln.split("mag=")[1].split()[0]) for ln in ex["by_magnitude"]]
            grads = [float(ln.split("grad=")[1].split()[0]) for ln in ex["by_gradient"]]
            assert mags == sorted(mags, reverse=True)
            assert grads == sorted(grads, reverse=True)
            assert 0.0 <= ex["top_k_jaccard"] <= 1.0
            # both lines carry both signals + a Neuronpedia link
            assert all("mag=" in ln and "grad=" in ln for ln in ex["by_gradient"])

        # the whole structure round-trips as YAML
        assert yaml.safe_load(yaml.safe_dump(result, allow_unicode=True))

    def test_percentile_baseline_when_pct_overlay_present(self, corpus, patched_builders):
        """A pct overlay makes the baseline rank by PERCENTILE (the real
        top_k_percentile criterion); absent it falls back to raw magnitude."""
        assert mod.run(_cfg(corpus)) == 0  # grad overlay
        _write_freq_blacklist(corpus["feature_dir"])

        # Synthesize a percentile overlay: uint8 feature_percentiles, 1:1 with
        # the base feature_ids (exactly what precompute_percentiles writes).
        fdir = corpus["feature_dir"]
        pct_dir = fdir / "pct" / "synthpct"
        pct_dir.mkdir(parents=True)
        for base in glob_features(fdir):
            stem = shard_stem_from_path(base)
            n = int(len(load_features(base)["feature_ids"]))
            pcts_u8 = (t.arange(n, dtype=t.int64) * 7 % 256).to(t.uint8)
            save_overlay_shard(
                {"feature_percentiles": pcts_u8}, overlay_shard_path(pct_dir, stem)
            )

        result = build_examples(
            fdir, None, shard="shard_00000", num_examples=3, top_k=3,
            min_features=1, prefix_chars=50, suffix_chars=50, seed=0,
            data_dir=corpus["data_dir"], dataset_phase=None,
        )
        assert result["meta"]["baseline"] == "percentile"
        for ex in result["examples"]:
            assert "by_percentile" in ex
            # by_percentile is sorted by pct descending; every line carries pct=.
            pcts = [float(ln.split("pct=")[1].split()[0]) for ln in ex["by_percentile"]]
            assert pcts == sorted(pcts, reverse=True)
            assert all("pct=" in ln for ln in ex["by_magnitude"])
            assert all("pct=" in ln for ln in ex["by_gradient"])

    def test_frequency_whitelist_drops_blacklisted(self, corpus, patched_builders):
        """The tool ranks over the production candidate set: a blacklisted
        feature is dropped from EVERY list (read from the on-disk artifact)."""
        assert mod.run(_cfg(corpus)) == 0
        fdir = corpus["feature_dir"]
        common = {
            "shard": "shard_00000", "num_examples": 3, "top_k": 5, "min_features": 1,
            "prefix_chars": 50, "suffix_chars": 50, "seed": 0,
            "data_dir": corpus["data_dir"], "dataset_phase": None,
        }

        def _shown(res):
            return {
                int(ln.split()[0][1:])
                for ex in res["examples"]
                for key, lines in ex.items()
                if key.startswith("by_")
                for ln in lines
            }

        _write_freq_blacklist(fdir, ids=())  # empty: every feature kept
        shown = _shown(build_examples(fdir, None, **common))  # ty: ignore[invalid-argument-type]
        assert shown, "no features displayed to blacklist"
        victim = min(shown)

        # Re-materialize the SAME-bounds artifact (one file) with victim masked.
        _write_freq_blacklist(fdir, ids=(victim,))
        res = build_examples(fdir, None, **common)  # ty: ignore[invalid-argument-type]
        assert victim not in _shown(res), f"blacklisted f{victim} still displayed"
        assert res["meta"]["frequency_filter"].startswith("blacklist/")
