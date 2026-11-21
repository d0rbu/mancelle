"""Task 5 (#695 PR-1): checkpoint-``V``-safe precomputed-bag cache.

A derived DiffMean/GCG bag is restricted to the checkpoint's ``V`` (compacted
feature vocabulary), so it is CHECKPOINT-SPECIFIC. The on-disk bag cache keys
by ``(concept_id, mode, reg_lambda, family, prefix_id)`` under an
operator-chosen ``run_tag`` — no ``V``/checkpoint in the key. So reusing a
``run_tag`` across checkpoints with a different ``V`` would silently serve a
WRONG bag. This file pins:

  1. ``checkpoint_vocab_fingerprint`` — the loaded-model -> fingerprint helper.
  2. ``write_ours_gcg_bag`` stamps it; ``read_ours_gcg_bag`` returns it.
  3. The REAL production read path (``build_intervention`` /
     ``_build_ours_intervention`` in ``steering.eval.axbench_bpb.intervention``)
     runs the compare — not just a unit test of the primitive in isolation.
"""

from __future__ import annotations

import random
from hashlib import sha256
from itertools import islice, permutations
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import (
    _assert_bag_matches_checkpoint_vocab,
    build_intervention,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention
from steering.eval.axbench_bpb.vocab_restriction import (
    UNCOMPACTED_VOCAB_FINGERPRINT,
    checkpoint_vocab_fingerprint,
)
from steering.eval.axbench_bpb_adv import precomputed_bag_artifacts as bag_artifacts
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    FEATURE_CONTRACT_FINGERPRINT_META_KEY,
    VOCAB_FINGERPRINT_META_KEY,
    read_ours_gcg_bag,
    write_ours_gcg_bag,
)

# ---------------------------------------------------------------------------
# Real EncoderDecoderGPT fixtures (mirrors tests/steering/test_vocab_restriction.py's
# _build_model — proves the helper matches the production shape, not just an
# assumption about its attributes).
# ---------------------------------------------------------------------------

N_SAE = 64
VOCAB_A = FeatureVocab(n_sae_features=N_SAE, original_ids=(0, 3, 7, 11))
VOCAB_B = FeatureVocab(n_sae_features=N_SAE, original_ids=(1, 2, 5, 9))

DEC_KW = {
    "sequence_len": 16,
    "vocab_size": 65536,
    "n_layer": 2,
    "n_head": 2,
    "n_kv_head": 2,
    "n_embd": 64,
    "window_pattern": "L",
}
ENC_KW = {"n_embd": 32, "depth": 2, "n_head": None, "n_kv_head": None, "head_dim": 16}


def _build_model(vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=16,
                vocab_size=65536,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="L",
            ),
            EncoderConfig(
                n_embd=32,
                depth=2,
                n_head=None,
                n_kv_head=None,
                head_dim=16,
            ),
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=4,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    m.init_buffers()
    return m


def _make_concept(concept_id: int = 1) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=0,
        layer=8,
        sae_source="dummy",
        description="test concept",
        corpus_count=10,
        description_source="manual",
    )


def _write_legacy_bag(path: Path, bag: dict[int, int]) -> None:
    """Write a bag with NO vocab_fingerprint key -- pre-#695-PR1 shape.

    Bypasses ``write_ours_gcg_bag`` (which now REQUIRES the key) to construct
    exactly the artifact shape a pre-existing on-disk cache has: no
    ``VOCAB_FINGERPRINT_META_KEY`` entry at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_fids = sorted(bag.keys())
    tensors = {
        "feature_ids": t.tensor(sorted_fids, dtype=t.int64),
        "buckets": t.tensor([bag[k] for k in sorted_fids], dtype=t.int64),
    }
    save_file(tensors, str(path))


# ---------------------------------------------------------------------------
# 1. checkpoint_vocab_fingerprint
# ---------------------------------------------------------------------------


class TestCheckpointVocabFingerprint:
    def test_compacted_model_returns_the_feature_vocabs_own_fingerprint(self):
        model = _build_model(VOCAB_A)
        assert checkpoint_vocab_fingerprint(model) == VOCAB_A.fingerprint()

    def test_legacy_model_returns_the_uncompacted_sentinel(self):
        model = _build_model(None)
        assert checkpoint_vocab_fingerprint(model) == UNCOMPACTED_VOCAB_FINGERPRINT

    def test_different_vocabs_fingerprint_differently(self):
        # Non-vacuous: the two fixture vocabs used throughout this file must
        # actually disagree, or every mismatch test below would be vacuous.
        assert VOCAB_A.fingerprint() != VOCAB_B.fingerprint()

    def test_duck_typed_stub(self):
        class _Stub:
            feature_vocab = None

        assert checkpoint_vocab_fingerprint(_Stub()) == UNCOMPACTED_VOCAB_FINGERPRINT


# ---------------------------------------------------------------------------
# 2. write_ours_gcg_bag stamps / read_ours_gcg_bag returns it
# ---------------------------------------------------------------------------


class TestWriteReadStamp:
    def test_write_stamps_the_fingerprint_and_read_returns_it(self, tmp_path: Path):
        bag = FeatureConditioning.from_dict({3: 0, 7: 1})
        path = tmp_path / "bag.safetensors"
        write_ours_gcg_bag(
            path, bag, metadata={"mode": "positive"}, vocab_fingerprint="fingerprint-A"
        )
        loaded_bag, metadata = read_ours_gcg_bag(path)
        assert loaded_bag == {3: 0, 7: 1}
        assert metadata[VOCAB_FINGERPRINT_META_KEY] == "fingerprint-A"
        assert metadata["mode"] == "positive"  # caller metadata preserved alongside the stamp

    def test_caller_supplied_fingerprint_key_collision_raises(self, tmp_path: Path):
        bag = FeatureConditioning.from_dict({1: 0})
        path = tmp_path / "bag.safetensors"
        with pytest.raises(AxBenchBpbAdvError, match=VOCAB_FINGERPRINT_META_KEY):
            write_ours_gcg_bag(
                path,
                bag,
                metadata={VOCAB_FINGERPRINT_META_KEY: "sneaky"},
                vocab_fingerprint="real",
            )

    def test_clustered_bag_bytes_are_canonical_across_metadata_order(
        self,
        tmp_path: Path,
    ) -> None:
        bag = FeatureConditioning.from_dict({7: 1, 3: 0})
        fingerprint = "a" * 64
        metadata_items = [
            (FEATURE_CONTRACT_FINGERPRINT_META_KEY, fingerprint),
            ("mode", "positive"),
            ("concept_id", "1"),
            ("source_contract_sha256", "b" * 64),
        ]
        paths: list[Path] = []
        for index, metadata_order in enumerate(
            islice(permutations(metadata_items), 8)
        ):
            path = tmp_path / f"bag-{index}.safetensors"
            write_ours_gcg_bag(
                path,
                bag,
                metadata=dict(metadata_order),
                vocab_fingerprint="fingerprint-A",
            )
            paths.append(path)

        rendered = [path.read_bytes() for path in paths]
        assert len(set(rendered)) == 1
        assert sha256(rendered[0]).hexdigest() == (
            "6e0eba9405012a107ed1178b36e16aff1614fbe00070a341aee87021ec46fece"
        )
        loaded = [
            read_ours_gcg_bag(
                path,
                expected_feature_contract_fingerprint=fingerprint,
            )
            for path in paths
        ]
        assert all(item == loaded[0] for item in loaded)

    def test_clustered_bag_resume_is_exact_or_fails_without_overwrite(
        self,
        tmp_path: Path,
    ) -> None:
        fingerprint = "a" * 64
        path = tmp_path / "clustered-resume.safetensors"
        bag = FeatureConditioning.from_dict({3: 0, 7: 1})
        metadata = {
            FEATURE_CONTRACT_FINGERPRINT_META_KEY: fingerprint,
            "mode": "positive",
        }

        def write_identity() -> tuple[int, int, int, int]:
            stat = path.stat()
            return (
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )

        # Fresh install.
        write_ours_gcg_bag(
            path,
            bag,
            metadata=metadata,
            vocab_fingerprint="fingerprint-A",
        )
        original = path.read_bytes()
        original_identity = write_identity()

        # Exact replay is a no-op, including inode and write timestamps.
        write_ours_gcg_bag(
            path,
            bag,
            metadata=metadata,
            vocab_fingerprint="fingerprint-A",
        )
        assert path.read_bytes() == original
        assert write_identity() == original_identity

        # Changed entries or load-bearing metadata fail before committed bytes
        # or their filesystem identity are touched.
        with pytest.raises(AxBenchBpbAdvError, match="content changed on resume"):
            write_ours_gcg_bag(
                path,
                FeatureConditioning.from_dict({3: 0, 11: 1}),
                metadata=metadata,
                vocab_fingerprint="fingerprint-A",
            )
        assert path.read_bytes() == original
        assert write_identity() == original_identity

        with pytest.raises(AxBenchBpbAdvError, match="content changed on resume"):
            write_ours_gcg_bag(
                path,
                bag,
                metadata={**metadata, "mode": "negative"},
                vocab_fingerprint="fingerprint-A",
            )
        assert path.read_bytes() == original
        assert write_identity() == original_identity

    def test_unclustered_bag_keeps_exact_legacy_writer_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        legacy_calls: list[tuple[dict[str, t.Tensor], Path, dict[str, str]]] = []
        install_calls: list[tuple[bytes, Path, str]] = []

        def record_legacy(
            tensors: dict[str, t.Tensor],
            path: Path,
            metadata: dict[str, str],
        ) -> None:
            legacy_calls.append((tensors, path, metadata))

        def record_install(payload: bytes, path: Path, *, surface: str) -> bool:
            install_calls.append((payload, path, surface))
            return True

        monkeypatch.setattr(bag_artifacts, "atomic_safetensors_save", record_legacy)
        monkeypatch.setattr(bag_artifacts, "install_bytes_once", record_install)
        path = tmp_path / "legacy-path.safetensors"
        write_ours_gcg_bag(
            path,
            FeatureConditioning.from_dict({3: 0, 7: 1}),
            metadata={"mode": "positive"},
            vocab_fingerprint="fingerprint-A",
        )

        assert len(legacy_calls) == 1
        assert install_calls == []
        tensors, actual_path, metadata = legacy_calls[0]
        assert actual_path == path
        assert tensors["feature_ids"].tolist() == [3, 7]
        assert tensors["buckets"].tolist() == [0, 1]
        assert list(metadata.items()) == [
            ("mode", "positive"),
            (VOCAB_FINGERPRINT_META_KEY, "fingerprint-A"),
        ]


# ---------------------------------------------------------------------------
# 3. Direct primitive: _assert_bag_matches_checkpoint_vocab
# ---------------------------------------------------------------------------


class TestAssertBagMatchesCheckpointVocab:
    def test_same_fingerprint_returns_none_no_raise(self):
        model = _build_model(VOCAB_A)
        _assert_bag_matches_checkpoint_vocab(
            {VOCAB_FINGERPRINT_META_KEY: VOCAB_A.fingerprint()},
            model=model,
            method_name="ours_gcg_encdec",
            concept_id=1,
            bag_path="irrelevant",
        )  # no raise

    def test_mismatched_fingerprint_raises_naming_both(self):
        model_b = _build_model(VOCAB_B)
        fp_a = VOCAB_A.fingerprint()
        with pytest.raises(AxBenchBpbError) as exc_info:
            _assert_bag_matches_checkpoint_vocab(
                {VOCAB_FINGERPRINT_META_KEY: fp_a},
                model=model_b,
                method_name="ours_gcg_encdec",
                concept_id=1,
                bag_path="irrelevant",
            )
        message = str(exc_info.value)
        assert fp_a in message
        assert VOCAB_B.fingerprint() in message

    def test_legacy_stamped_bag_against_legacy_checkpoint_loads(self):
        model = _build_model(None)
        _assert_bag_matches_checkpoint_vocab(
            {VOCAB_FINGERPRINT_META_KEY: UNCOMPACTED_VOCAB_FINGERPRINT},
            model=model,
            method_name="ours_gcg_encdec",
            concept_id=1,
            bag_path="irrelevant",
        )  # no raise

    def test_legacy_stamped_bag_against_compacted_checkpoint_raises(self):
        """An uncompacted-derived bag is NOT valid for a now-compacted checkpoint."""
        model = _build_model(VOCAB_A)
        with pytest.raises(AxBenchBpbError) as exc_info:
            _assert_bag_matches_checkpoint_vocab(
                {VOCAB_FINGERPRINT_META_KEY: UNCOMPACTED_VOCAB_FINGERPRINT},
                model=model,
                method_name="ours_gcg_encdec",
                concept_id=1,
                bag_path="irrelevant",
            )
        message = str(exc_info.value)
        assert UNCOMPACTED_VOCAB_FINGERPRINT in message
        assert VOCAB_A.fingerprint() in message

    def test_no_fingerprint_key_against_compacted_checkpoint_raises(self):
        """Back-compat DECISION: a pre-existing (un-stamped) bag cannot prove it
        was derived for this V, so a COMPACTED checkpoint must RAISE."""
        model = _build_model(VOCAB_A)
        with pytest.raises(AxBenchBpbError, match="no.*vocab_fingerprint"):
            _assert_bag_matches_checkpoint_vocab(
                {"mode": "positive"},  # no VOCAB_FINGERPRINT_META_KEY at all
                model=model,
                method_name="ours_gcg_encdec",
                concept_id=1,
                bag_path="irrelevant",
            )

    def test_no_fingerprint_key_against_legacy_checkpoint_loads(self):
        """Back-compat DECISION: a pre-existing (un-stamped) bag against a LEGACY
        checkpoint loads unchanged -- every pre-#695 bag was implicitly derived
        against the full (uncompacted) vocabulary, exactly what a legacy
        checkpoint has."""
        model = _build_model(None)
        _assert_bag_matches_checkpoint_vocab(
            {"mode": "positive"},
            model=model,
            method_name="ours_gcg_encdec",
            concept_id=1,
            bag_path="irrelevant",
        )  # no raise


# ---------------------------------------------------------------------------
# 4. END-TO-END: the REAL production read path (_build_ours_intervention /
#    build_intervention), not just the primitive in isolation.
# ---------------------------------------------------------------------------


def _build(
    method_name: str,
    concept: Concept,
    coefficient: float,
    adv_paths: AxBenchBpbAdvPaths | None,
    *,
    source_run_dir: Path,
    model: Any,
    core_prefix_id: str | None = None,
) -> Intervention:
    return build_intervention(
        method=get_method_by_name(method_name),
        concept=concept,
        coefficient=coefficient,
        source_paths=SourcePaths(source_run_dir=source_run_dir),
        diffmean_layer=8,
        target_model_meta=None,
        inactive_sampler=None,
        tokenizer_nanochat=None,
        rng=random.Random(0),
        adv_paths=adv_paths,
        core_prefix_id=core_prefix_id,
        model=model,
        apply_max_act_scale=False,
    )


class TestEndToEndReadVerify:
    """Drives the REAL ``build_intervention`` dispatcher -> ``_build_ours_intervention``
    -> ``read_ours_gcg_bag`` -> ``_assert_bag_matches_checkpoint_vocab`` chain against a
    bag file actually written to disk and a REAL ``EncoderDecoderGPT``. This is the
    proof that the compare fires on the PRODUCTION read path, not merely that the
    primitive raises when called directly (covered above)."""

    def test_same_checkpoint_loads_the_bag_unchanged(self, tmp_path: Path):
        model = _build_model(VOCAB_A)
        concept = _make_concept(concept_id=1)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run1")
        bag_path = paths.ours_gcg_bag(1, "positive", 0.0, "fair")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({3: 0, 7: 1}),
            metadata={},
            vocab_fingerprint=checkpoint_vocab_fingerprint(model),
        )

        result = _build(
            "ours_gcg_encdec", concept, 0.0, paths, source_run_dir=tmp_path, model=model
        )

        assert result == Intervention.ours(feature_conditions={3: 0, 7: 1})

    def test_correct_fingerprint_with_corrupt_oov_payload_raises(
        self,
        tmp_path: Path,
    ):
        model = _build_model(VOCAB_A)
        concept = _make_concept(concept_id=8)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run8")
        bag_path = paths.ours_gcg_bag(8, "positive", 0.0, "fair")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({3: 0, 5: 1}),
            metadata={},
            vocab_fingerprint=checkpoint_vocab_fingerprint(model),
        )

        with pytest.raises(
            AxBenchBpbError,
            match="producer invariant violation.*5",
        ):
            _build(
                "ours_gcg_encdec",
                concept,
                0.0,
                paths,
                source_run_dir=tmp_path,
                model=model,
            )

    def test_bag_backed_path_never_loads_concept_top_k(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        model = _build_model(VOCAB_A)
        concept = _make_concept(concept_id=9)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run9")
        bag_path = paths.ours_gcg_bag(9, "positive", 0.0, "fair")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({3: 0}),
            metadata={},
            vocab_fingerprint=checkpoint_vocab_fingerprint(model),
        )
        monkeypatch.setattr(
            intervention_module,
            "_load_top_k_concept_fids",
            lambda *_args, **_kwargs: pytest.fail(
                "bag-backed methods must not load concept top-K"
            ),
        )

        assert _build(
            "ours_gcg_encdec",
            concept,
            0.0,
            paths,
            source_run_dir=tmp_path,
            model=model,
        ) == Intervention.ours(feature_conditions={3: 0})

    def test_different_checkpoint_v_raises_naming_both_fingerprints(self, tmp_path: Path):
        """The motivating scenario: a run_tag reused across checkpoints with a
        DIFFERENT V must fail loud, not silently serve the wrong bag."""
        derive_model = _build_model(VOCAB_A)
        score_model = _build_model(VOCAB_B)
        concept = _make_concept(concept_id=2)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="reused_run_tag")
        bag_path = paths.ours_gcg_bag(2, "positive", 0.0, "fair")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({3: 0}),
            metadata={},
            vocab_fingerprint=checkpoint_vocab_fingerprint(derive_model),
        )

        with pytest.raises(AxBenchBpbError) as exc_info:
            _build(
                "ours_gcg_encdec",
                concept,
                0.0,
                paths,
                source_run_dir=tmp_path,
                model=score_model,
            )
        message = str(exc_info.value)
        assert VOCAB_A.fingerprint() in message
        assert VOCAB_B.fingerprint() in message

    def test_legacy_bag_against_legacy_checkpoint_loads(self, tmp_path: Path):
        legacy_model = _build_model(None)
        concept = _make_concept(concept_id=3)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run3")
        bag_path = paths.ours_gcg_bag(3, "positive", 0.0, "fair")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({9: 0}),
            metadata={},
            vocab_fingerprint=checkpoint_vocab_fingerprint(legacy_model),
        )

        result = _build(
            "ours_gcg_encdec",
            concept,
            0.0,
            paths,
            source_run_dir=tmp_path,
            model=legacy_model,
        )

        assert result == Intervention.ours(feature_conditions={9: 0})

    def test_backcompat_unstamped_bag_against_compacted_checkpoint_raises(
        self, tmp_path: Path
    ):
        """A bag written before this feature existed (no vocab_fingerprint key at
        all) read against a COMPACTED checkpoint must RAISE end-to-end."""
        model = _build_model(VOCAB_A)
        concept = _make_concept(concept_id=4)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run4")
        bag_path = paths.ours_gcg_bag(4, "positive", 0.0, "fair")
        _write_legacy_bag(bag_path, {5: 0})

        with pytest.raises(AxBenchBpbError, match="no.*vocab_fingerprint"):
            _build(
                "ours_gcg_encdec", concept, 0.0, paths, source_run_dir=tmp_path, model=model
            )

    def test_backcompat_unstamped_bag_against_legacy_checkpoint_loads(self, tmp_path: Path):
        """The SAME unstamped bag against a LEGACY checkpoint loads unchanged --
        the back-compat decision, proven end-to-end (not just the primitive)."""
        legacy_model = _build_model(None)
        concept = _make_concept(concept_id=5)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run5")
        bag_path = paths.ours_gcg_bag(5, "positive", 0.0, "fair")
        _write_legacy_bag(bag_path, {6: 1})

        result = _build(
            "ours_gcg_encdec",
            concept,
            0.0,
            paths,
            source_run_dir=tmp_path,
            model=legacy_model,
        )

        assert result == Intervention.ours(feature_conditions={6: 1})

    def test_model_not_provided_for_bag_backed_method_raises(self, tmp_path: Path):
        """A caller that forgets model= for a bag-backed method must fail loud,
        not silently skip the checkpoint-V check."""
        concept = _make_concept(concept_id=6)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run6")
        bag_path = paths.ours_gcg_bag(6, "positive", 0.0, "fair")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({1: 0}),
            metadata={},
            vocab_fingerprint="fingerprint-x",
        )

        with pytest.raises(AxBenchBpbError, match="model"):
            _build(
                "ours_gcg_encdec", concept, 0.0, paths, source_run_dir=tmp_path, model=None
            )

    def test_roofline_family_also_verified_end_to_end(self, tmp_path: Path):
        """The compare is not fair-family-specific -- roofline bags go through
        the SAME read path (precomputed_bag_method_family classifies both)."""
        derive_model = _build_model(VOCAB_A)
        score_model = _build_model(VOCAB_B)
        concept = _make_concept(concept_id=7)
        paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run7")
        bag_path = paths.ours_gcg_bag(7, "positive", 0.0, "roofline", prefix_id="pfx")
        write_ours_gcg_bag(
            bag_path,
            FeatureConditioning.from_dict({2: 0}),
            metadata={},
            vocab_fingerprint=checkpoint_vocab_fingerprint(derive_model),
        )

        with pytest.raises(AxBenchBpbError):
            _build(
                "ours_gcg_roofline_encdec",
                concept,
                0.0,
                paths,
                source_run_dir=tmp_path,
                model=score_model,
                core_prefix_id="pfx",
            )
