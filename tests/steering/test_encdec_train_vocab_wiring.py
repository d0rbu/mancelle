"""The train script must resolve the feature vocabulary ONCE (#695).

``conditional_encdec_train`` derives THREE artifacts from the ``feature_mask``, and a
disagreement between any two of them is silent, unrecoverable corruption:

1. the encoder's embedding tables (sized by ``FeatureVocab.size``),
2. the checkpoint directory's ``bl_``/``wl_`` leaf (``append_mask_leaf``),
3. the ``feature_vocab`` fingerprint written into ``meta_*.json``.

But the mask is resolved at THREE separate sites — ``build_shared_state`` (line ~1414,
which sizes the model), the discovery selection-hash (~1500), and ``train_one_config``
(~2259, which builds the leaf and the meta). The fast sweep hands ``train_one_config`` a
FRESHLY COMPOSED per-config ``DictConfig``, so those sites read different objects. If they
ever disagreed, the run would train the SHARED model's dictionary while filing the weights
under the PER-CONFIG mask's leaf and stamping the per-config mask into the meta — and
because a same-size-different-ids vocab produces IDENTICAL tensor shapes, ``load_state_dict``
would never notice (``TestTheSplitBrainWouldOtherwiseBeSilent`` proves exactly this).

``train_one_config`` line 2268 is the one thing standing in the way. Nothing tested it.

This module drives the REAL ``train_one_config`` against a REAL ``EncoderDecoderGPT``, a
REAL ``SharedState`` and a REAL Hydra-composed ``conditional_encdec_train`` config, and it
drives the REAL frequency-band producer against a REAL ``feature_statistics.safetensors``
so that the band -> mask -> vocab chain that EVERY default run now executes is exercised
end to end, rather than stood in for by a synthetic blacklist.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import scripts.train.conditional_encdec_train as train_mod
from errors import FeatureMaskError, FeatureVocabError
from nanochat.gpt import GPTConfig
from scripts.train.conditional_encdec_train import (
    SharedState,
    build_encdec_meta_model_from_training_cfg,
    train_one_config,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    append_mask_leaf,
    canonical_dict,
    resolve_feature_mask,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.freq_blacklist import (
    DEFAULT_FREQ_MAX,
    DEFAULT_FREQ_MIN,
    maybe_apply_frequency_blacklist,
)
from steering.refined_types import PositiveInt
from tests.steering._freq_blacklist_fixtures import write_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_DIR = REPO_ROOT / "configs" / "training"

N_SAE: PositiveInt = PositiveInt.parse(64)
NUM_BUCKETS = 4

# Vocab A and vocab B have the SAME SIZE and DIFFERENT IDS. That is the whole point:
# identical shapes ⇒ every tensor lines up ⇒ PyTorch raises nothing ⇒ only an explicit
# vocabulary comparison can catch the swap.
IDS_A = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)
IDS_B = (1, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 62)
VOCAB_A = FeatureVocab(n_sae_features=N_SAE, original_ids=IDS_A)
VOCAB_B = FeatureVocab(n_sae_features=N_SAE, original_ids=IDS_B)

MASK_A = FeatureMask(mode=FeatureMaskMode.whitelist, ids=IDS_A)
MASK_B = FeatureMask(mode=FeatureMaskMode.whitelist, ids=IDS_B)
# The SAME allowed universe as MASK_A, spelled as a blacklist of the complement. The guard
# compares VOCABULARIES, not masks, so this must be accepted — a guard that compared masks
# would reject it and would be wrong.
BLACKLIST_A = FeatureMask(
    mode=FeatureMaskMode.blacklist,
    ids=tuple(i for i in range(N_SAE) if i not in IDS_A),
)

DEC_KW = {
    "sequence_len": 16,
    "vocab_size": 128,
    "n_layer": 2,
    "n_head": 2,
    "n_kv_head": 2,
    "n_embd": 64,
    "window_pattern": "L",
}
ENC_KW = {"n_embd": 32, "depth": 2, "n_head": None, "n_kv_head": None, "head_dim": 16}

TRAIN_SOURCE = Path(train_mod.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Real models, real shared state
# ---------------------------------------------------------------------------


def _model(vocab: FeatureVocab | None, *, seed: int = 7) -> EncoderDecoderGPT:
    """A REAL EncoderDecoderGPT, built exactly the way production builds it.

    meta -> to_empty -> seed -> init_weights. The meta-device step is load-bearing (a
    compact ``nn.Embedding`` would otherwise draw fewer randoms at construction and
    diverge the RNG stream), and it is what the production builder does.
    """
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(**DEC_KW),  # ty: ignore[invalid-argument-type]
            EncoderConfig(**ENC_KW),  # ty: ignore[invalid-argument-type]
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=NUM_BUCKETS,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    t.manual_seed(seed)
    m.init_weights()
    return m


# The fields ``train_one_config`` actually READS before the vocabulary guard at line 2268:
#   n_sae_features, feature_dir, orig_model, device_type, ddp_rank
# Everything else is unpacked into a local and never touched on the way to the guard, so it
# is passed as None rather than faked into something that looks meaningful but is not. This
# is a REAL SharedState dataclass holding a REAL model — not a mock.
_UNUSED_BEFORE_THE_GUARD = {
    "tokenizer": None,
    "token_bytes": None,
    "base_vocab": 0,
    "feature_index": None,
    "feature_stats": None,
    "per_feature_stats": None,
    "resolved_feature_clusters": None,
    "feature_clusters": None,
    "percentile_checksum": None,
    "cached_context_provider": None,
    "device": t.device("cpu"),
    "ddp": False,
    "ddp_local_rank": 0,
    "ddp_world_size": 1,
    "extraction_segment_size": 128,
    "gpu_peak_flops": 1.0,
    "boundaries": t.zeros(N_SAE, 0, dtype=t.float64),
    "d_ref": 1.0,
    "b_ref": 1,
    "tokenizer_checksum": None,
    "steering_eval_cfg": None,
    "steering_probe_model": None,
    "packed_row_dir": None,
    "selection_dir": None,
    "original_val_feature_dir": None,
    "original_val_packed_row_dir": None,
    "original_val_selection_dir": None,
    "original_val_feature_index": None,
    "spliced_val_feature_index": None,
    "val_packed_row_dir": None,
    "val_selection_dir": None,
    "max_seg_size": None,
    "pad_F": None,
    "live_pad_n_seg": None,
    "fp_target_layout": None,
    "fp_target_source_certificate_digest": None,
}


def _shared(
    orig_model: EncoderDecoderGPT,
    cfg: DictConfig,
    feature_dir: str,
    *,
    resolved_feature_clusters: ResolvedFeatureClusters | None = None,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> SharedState:
    unused = {
        **_UNUSED_BEFORE_THE_GUARD,
        "resolved_feature_clusters": resolved_feature_clusters,
        "feature_clusters": feature_clusters,
    }
    return SharedState(
        n_sae_features=N_SAE,
        feature_dir=feature_dir,
        model=orig_model,
        orig_model=orig_model,
        device_type="cpu",
        ddp_rank=0,
        cfg=cfg,
        **unused,  # ty: ignore[invalid-argument-type]
    )


def _cfg(hydra_compose, mask: FeatureMask | None) -> DictConfig:
    """The REAL shipped ``conditional_encdec_train`` config, with an explicit mask.

    An explicit ``feature_mask`` makes ``maybe_apply_frequency_blacklist`` a no-op
    (SKIPPED_USER_MASK — the user mask always wins), which is exactly how the standalone
    and subprocess-sweep paths reach ``train_one_config``.

    ``mask=None`` additionally NULLS the frequency band. That is not a convenience: since
    #695 the shipped ``(1e-5, 0.5)`` band resolves a mask on EVERY default run, so
    nulling both bounds is the *only* way to get an unmasked run — and it is precisely the
    workaround the FP-head ``NotImplementedError`` warns operators away from. An operator
    who takes it anyway, against a compacted model, lands in the guard below.
    """
    cfg = hydra_compose(str(TRAINING_CONFIG_DIR), "conditional_encdec_train", None)
    OmegaConf.update(
        cfg,
        "feature_mask",
        None if mask is None else canonical_dict(mask),
        merge=False,
        force_add=True,
    )
    if mask is None:
        OmegaConf.update(cfg, "feature_mask_freq_min", None, merge=False, force_add=True)
        OmegaConf.update(cfg, "feature_mask_freq_max", None, merge=False, force_add=True)
    return cfg


@pytest.fixture(autouse=True)
def _base_dir(tmp_path, monkeypatch):
    """``train_one_config`` calls ``get_base_dir()`` (which mkdirs) before the guard."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))


# ===========================================================================
# 1. The split-brain guard
# ===========================================================================


class TestTheSplitBrainGuardFires:
    """``train_one_config`` line 2268-2279.

    The model was sized by the SHARED config's mask. If the per-config config resolves a
    DIFFERENT vocabulary, the run must die — before it computes a checkpoint leaf, before
    it loads anything, before it trains a single step.
    """

    def test_a_different_per_config_vocab_is_REFUSED(self, hydra_compose, tmp_path):
        model = _model(VOCAB_A)  # the shared model carries vocab A
        cfg = _cfg(hydra_compose, MASK_B)  # this config resolves vocab B
        shared = _shared(model, cfg, str(tmp_path))

        with pytest.raises(FeatureVocabError) as exc:
            train_one_config(cfg, shared, wandb_run=None)

        msg = str(exc.value)
        assert "different encoder vocabulary" in msg
        # An operator must be told WHAT to do, not merely that something is wrong.
        assert "run_encdec_sweep.py" in msg, (
            "the error must name the escape hatch (the subprocess sweep, which builds a "
            f"fresh model per config): {msg!r}"
        )

    def test_the_IDENTICAL_mask_is_accepted_by_the_guard(self, hydra_compose, tmp_path):
        """The guard must not be a blanket refusal: the normal path has to get through it.

        It cannot reach the end of ``train_one_config`` (no data, no tokenizer), so the
        proof is that it dies LATER and for a DIFFERENT reason than the vocabulary.
        """
        model = _model(VOCAB_A)
        cfg = _cfg(hydra_compose, MASK_A)
        shared = _shared(model, cfg, str(tmp_path))

        with pytest.raises(Exception) as exc:  # noqa: PT011 — any non-vocab failure will do
            train_one_config(cfg, shared, wandb_run=None)

        assert not isinstance(exc.value, FeatureVocabError), (
            "a config whose mask EQUALS the one the model was built with was refused by "
            f"the vocabulary guard: {exc.value!r}"
        )

    def test_the_SAME_universe_spelled_as_a_BLACKLIST_is_accepted(self, hydra_compose, tmp_path):
        """The trap this guard could easily fall into: comparing MASKS instead of VOCABS.

        A blacklist of the complement of A has the IDENTICAL allowed universe as a
        whitelist of A. The encoder is shaped by the universe alone, so this run is
        perfectly valid and must NOT be refused. A guard written as
        ``if feature_mask != shared_feature_mask`` would reject it — passing every other
        test in this class for entirely the wrong reason.
        """
        assert FeatureVocab.from_mask(BLACKLIST_A, N_SAE) == VOCAB_A
        assert BLACKLIST_A.mode is not MASK_A.mode

        model = _model(VOCAB_A)
        cfg = _cfg(hydra_compose, BLACKLIST_A)
        shared = _shared(model, cfg, str(tmp_path))

        with pytest.raises(Exception) as exc:  # noqa: PT011
            train_one_config(cfg, shared, wandb_run=None)

        assert not isinstance(exc.value, FeatureVocabError), (
            "the guard compared mask MODES, not the resolved vocabulary: a blacklist with "
            f"the same allowed universe was refused ({exc.value!r})"
        )

    def test_an_UNMASKED_config_against_a_COMPACTED_model_is_refused(self, hydra_compose, tmp_path):
        """The asymmetric split-brain: ``None`` vs a vocab. ``from_mask(None) is None``, so
        a naive ``if per_config_vocab and per_config_vocab != ...`` would wave it through
        and train a compacted model as if it were full width."""
        model = _model(VOCAB_A)
        cfg = _cfg(hydra_compose, None)
        shared = _shared(model, cfg, str(tmp_path))

        with pytest.raises(FeatureVocabError, match="different encoder vocabulary"):
            train_one_config(cfg, shared, wandb_run=None)

    def test_a_MASKED_config_against_a_FULL_WIDTH_model_is_refused(self, hydra_compose, tmp_path):
        """The mirror image: the model is full width, the config wants compaction."""
        model = _model(None)
        cfg = _cfg(hydra_compose, MASK_A)
        shared = _shared(model, cfg, str(tmp_path))

        with pytest.raises(FeatureVocabError, match="different encoder vocabulary"):
            train_one_config(cfg, shared, wandb_run=None)

    def test_active_clusters_reach_the_per_config_effective_vocab_guard_directly(
        self,
        hydra_compose,
        tmp_path,
    ):
        """The fast-sweep entrypoint rejects clusters; exercise this guard directly."""
        cluster_path = tmp_path / "clusters.json"
        cluster_path.write_text(
            json.dumps({"schema_version": 1, "clusters": [[3, 7, 10]]}),
            encoding="utf-8",
        )
        resolved = resolve_feature_clusters(
            {"feature_clusters": {"path": str(cluster_path)}},
            N_SAE,
        )
        assert resolved is not None
        shared_clusters = FeatureClusterConsolidation.build(resolved, MASK_A)
        shared_vocab = effective_feature_vocab(shared_clusters, N_SAE)
        assert shared_vocab is not None
        model = _model(shared_vocab)

        cfg = _cfg(hydra_compose, MASK_B)
        OmegaConf.update(
            cfg,
            "feature_clusters",
            {"path": str(cluster_path)},
            merge=False,
            force_add=True,
        )
        for key in ("core_metric_every", "sample_every", "steering_eval_every"):
            OmegaConf.update(cfg, key, 0, merge=False, force_add=True)
        OmegaConf.update(cfg, "feature_prediction", None, merge=False, force_add=True)
        OmegaConf.update(cfg, "use_precomputed_batches", True, merge=False, force_add=True)
        OmegaConf.update(cfg, "feature_delta", None, merge=False, force_add=True)
        OmegaConf.update(
            cfg,
            "feature_selection",
            {"threshold_type": "feature_percentile", "threshold": 0.8},
            merge=False,
            force_add=True,
        )
        OmegaConf.update(
            cfg,
            "next_segment_conditioning_offset",
            False,
            merge=False,
            force_add=True,
        )
        shared = _shared(
            model,
            cfg,
            str(tmp_path),
            resolved_feature_clusters=resolved,
            feature_clusters=shared_clusters,
        )

        with pytest.raises(FeatureVocabError) as exc:
            train_one_config(cfg, shared, wandb_run=None)
        assert "feature_mask and/or feature_clusters" in str(exc.value)


class TestTheSplitBrainWouldOtherwiseBeSilent:
    """Why the guard above is not decoration.

    If it were deleted, nothing downstream would catch a same-size vocabulary swap. These
    tests establish that premise directly, so the guard's tests cannot pass vacuously.
    """

    def test_two_different_vocabs_produce_IDENTICAL_tensor_shapes(self):
        assert VOCAB_A.size == VOCAB_B.size
        assert VOCAB_A != VOCAB_B

        a, b = _model(VOCAB_A), _model(VOCAB_B)

        assert a.encoder.feature_embed.weight.shape == b.encoder.feature_embed.weight.shape

    def test_a_vocab_A_checkpoint_loads_into_a_vocab_B_model_WITHOUT_ERROR(self):
        """``strict=True`` is happy. Every embedding row now means a different feature."""
        state = _model(VOCAB_A).state_dict()
        wrong = _model(VOCAB_B)

        wrong.load_state_dict(state, strict=True, assign=True)  # no exception!

        # And the corruption is real: the model's own idea of its vocabulary has been
        # replaced by the checkpoint's, silently.
        assert wrong.encoder.feature_vocab_ids.tolist() == list(IDS_A)

    def test_the_two_masks_file_the_weights_under_DIFFERENT_checkpoint_leaves(self, tmp_path):
        """The other half of the catastrophe: the leaf is built from the PER-CONFIG mask.

        A split-brain run would train vocab A's tables and save them under mask B's leaf —
        so the directory name would forever lie about which dictionary the weights encode.
        """
        leaf_a = append_mask_leaf(str(tmp_path), MASK_A)
        leaf_b = append_mask_leaf(str(tmp_path), MASK_B)

        assert leaf_a != leaf_b


# ===========================================================================
# 2. Ordering — the guard must run BEFORE anything acts on the mask
# ===========================================================================


def _train_one_config_lines() -> list[str]:
    return inspect.getsource(train_one_config).splitlines()


def _first_line_matching(needle: str) -> int:
    for i, line in enumerate(_train_one_config_lines()):
        if needle in line and not line.strip().startswith("#"):
            return i
    raise AssertionError(f"{needle!r} not found in train_one_config")


class TestGuardOrdering:
    """A guard that fires too late is a guard that fires after the damage."""

    def test_the_guard_precedes_the_checkpoint_leaf(self):
        """If the leaf were computed first, a resume would already have been pointed at
        the WRONG directory — and an existing checkpoint there would look legitimate."""
        assert _first_line_matching("per_config_vocab") < _first_line_matching("append_mask_leaf")

    def test_the_guard_precedes_the_resume_load(self):
        assert _first_line_matching("per_config_vocab") < _first_line_matching("load_checkpoint(")

    def test_the_guard_follows_the_frequency_band_producer(self):
        """``maybe_apply_frequency_blacklist`` MUTATES ``cfg.feature_mask``. Resolving the
        mask before it ran would compare the guard against a mask the run does not use."""
        assert _first_line_matching("maybe_apply_frequency_blacklist") < (
            _first_line_matching("resolve_feature_mask")
        )
        assert _first_line_matching("resolve_feature_mask") < _first_line_matching(
            "per_config_vocab"
        )


# ===========================================================================
# 3. One vocab, three consumers — the model, the leaf, the meta
# ===========================================================================


def _main_fn(name: str) -> ast.FunctionDef:
    tree = ast.parse(TRAIN_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in conditional_encdec_train")


class TestOneVocabThreeConsumers:
    """The model, the checkpoint leaf and the meta fingerprint must trace to ONE vocab."""

    def test_EVERY_model_build_uses_the_SHARED_resolve(self):
        """``build_shared_state`` must hand the builder the vocab it just resolved — not
        re-resolve it, and not pass ``None``.

        There are TWO builds: the training model (line ~1418) and the ``d12_ref``
        scaling-law reference (~1993). Both must take the same vocab. Asserting "exactly
        one call" would have been a lie; asserting "every call" is the real invariant, and
        it also covers a third build added tomorrow.
        """
        ctors = [
            n
            for n in ast.walk(_main_fn("build_shared_state"))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "build_encdec_meta_model_from_training_cfg"
        ]
        assert len(ctors) == 2, (
            f"expected the 2 known model builds in build_shared_state, found {len(ctors)}"
        )
        for ctor in ctors:
            kwargs = {kw.arg: kw.value for kw in ctor.keywords}
            assert "feature_vocab" in kwargs, (
                f"the build at line {ctor.lineno} omits feature_vocab — that model would "
                "silently be full width"
            )
            assert isinstance(kwargs["feature_vocab"], ast.Name)
            assert kwargs["feature_vocab"].id == "shared_feature_vocab", (
                f"the build at line {ctor.lineno} does not use the shared resolve "
                f"({ast.unparse(kwargs['feature_vocab'])!r}) — a second resolve is a "
                "second chance to disagree"
            )

    def test_the_effective_vocab_is_derived_once_after_consolidation_before_build(self):
        """Resolve mask/clusters, consolidate, derive once, then size every model."""
        fn = _main_fn("build_shared_state")
        resolves = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "resolve_feature_mask"
        ]
        builds = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "build_encdec_meta_model_from_training_cfg"
        ]
        derives = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "effective_feature_vocab"
        ]
        consolidations = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "build"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "FeatureClusterConsolidation"
        ]

        assert len(derives) == 1, "the shared effective vocab must be derived exactly once"
        assert len(consolidations) == 1, "the shared consolidation must be built exactly once"
        assert min(resolves) < consolidations[0] < derives[0] < min(builds), (
            "the mask must be resolved, applied to clusters, turned into an effective "
            "vocab, and only THEN handed to the builder "
            f"(resolves={resolves}, consolidations={consolidations}, "
            f"derives={derives}, builds={builds})"
        )

    def test_the_meta_fingerprint_comes_from_the_MODEL_not_a_re_resolve(self):
        """``_save_meta['feature_vocab']`` must be ``orig_model.feature_vocab``.

        Re-deriving it from ``cfg`` at save time would reintroduce the split brain at the
        one moment it becomes permanent — the meta would describe a vocabulary the weights
        do not have.
        """
        # The write is keyed by the FEATURE_VOCAB_META_KEY alias, so pin the alias too:
        # if it ever stopped meaning "feature_vocab", the resume guard (which reads that
        # literal key) would silently stop finding what training writes.
        assert train_mod.FEATURE_VOCAB_META_KEY == "feature_vocab"

        writes = [
            n
            for n in ast.walk(_main_fn("train_one_config"))
            if isinstance(n, ast.Assign)
            and any(
                isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "_save_meta"
                and ast.unparse(tgt.slice) == "FEATURE_VOCAB_META_KEY"
                for tgt in n.targets
            )
        ]
        assert len(writes) == 1, "expected exactly one feature_vocab meta write"

        src = ast.unparse(writes[0].value)
        assert "orig_model.feature_vocab" in src, (
            "the meta fingerprint is not taken from the model that actually holds the "
            f"embedding tables: {src!r}"
        )
        assert "metadata_dict" in src

    def test_the_leaf_is_built_from_the_resolved_mask(self):
        calls = [
            n
            for n in ast.walk(_main_fn("train_one_config"))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "append_mask_leaf"
        ]
        assert len(calls) == 1
        assert isinstance(calls[0].args[1], ast.Name)
        assert calls[0].args[1].id == "feature_mask"

    def test_the_three_agree_for_every_mask_the_guard_admits(self):
        """The invariant the guard buys, stated directly: once it passes, the vocab that
        sized the tables IS the vocab the leaf's mask resolves to IS the vocab the meta
        fingerprints. Checked for both spellings of the same universe."""
        model = _model(VOCAB_A)

        for mask in (MASK_A, BLACKLIST_A):
            resolved = resolve_feature_mask({"feature_mask": canonical_dict(mask)}, N_SAE)
            vocab = effective_feature_vocab(resolved, N_SAE)

            assert vocab == model.feature_vocab  # the guard's condition
            assert model.encoder.feature_embed.weight.shape[0] == vocab.size  # ty: ignore[unresolved-attribute]
            assert model.feature_vocab.metadata_dict()["sha256"] == vocab.fingerprint()  # ty: ignore[unresolved-attribute]


# ===========================================================================
# 4. The REAL band -> mask -> vocab chain (what every default run executes)
# ===========================================================================


def _band_cfg(hydra_compose, *, fmin: float | None, fmax: float | None) -> DictConfig:
    cfg = hydra_compose(str(TRAINING_CONFIG_DIR), "conditional_encdec_train", None)
    OmegaConf.update(cfg, "feature_mask", None, merge=False, force_add=True)
    OmegaConf.update(cfg, "feature_mask_freq_min", fmin, merge=False, force_add=True)
    OmegaConf.update(cfg, "feature_mask_freq_max", fmax, merge=False, force_add=True)
    return cfg


def _production_chain(cfg: DictConfig, feature_dir: Path) -> FeatureVocab | None:
    """The exact no-cluster three-line producer/resolve/effective chain."""
    maybe_apply_frequency_blacklist(cfg, str(feature_dir), N_SAE)
    mask = resolve_feature_mask(cfg, N_SAE)
    return effective_feature_vocab(mask, N_SAE)


class TestTheRealBandProducesTheRealVocab:
    """``test_all_training_configs_build.py`` stands the band's mask in with a synthetic
    3-id blacklist because it has no ``feature_statistics`` on disk. That is honest but it
    leaves the actual producer — the thing every default run depends on — unexercised from
    band to vocabulary. A real (tiny) statistics artifact closes the loop.

    ``counts`` below, against ``n_segments=100_000`` and the SHIPPED ``(1e-5, 0.5)`` band
    (KEEP iff ``min < freq <= max`` — lower bound EXCLUSIVE-of-keep, upper INCLUSIVE-of-keep):

      * count 0 (absent)  -> freq 0.0     -> killed by freq_min (dead feature)
      * count 1           -> freq 1e-5    -> killed: freq == freq_min, and KEEP needs `>`
      * count 2           -> freq 2e-5    -> KEPT
      * count 50_000      -> freq 0.5     -> KEPT: freq == freq_max, and KEEP allows `<=`
      * count 50_001+     -> freq > 0.5   -> killed by freq_max

    ``n_segments`` is 100_000, not 10_000, on purpose: at 10_000 the smallest non-zero
    frequency is 1e-4, which is a decade ABOVE the 1e-5 floor — so no live feature could
    reach the lower bound and the inclusive-boundary case would go untested. The two counts
    that sit EXACTLY on the bounds are the point of this fixture.
    """

    def test_the_shipped_band_compacts_a_real_statistics_artifact(self, hydra_compose, tmp_path):
        counts = {0: 1, 1: 2, 2: 50_000, 3: 50_001, 4: 99_999}  # 5..63 absent ⇒ dead
        write_stats(tmp_path, n_segments=100_000, counts=counts, n_sae_features=N_SAE)
        cfg = _band_cfg(hydra_compose, fmin=DEFAULT_FREQ_MIN, fmax=DEFAULT_FREQ_MAX)

        vocab = _production_chain(cfg, tmp_path)

        assert vocab is not None, "the shipped band did not compact anything"
        assert vocab.original_ids == (1, 2), (
            "the band kept the wrong features: id 0 (freq == freq_min exactly, and KEEP "
            "needs freq > min) and ids 3/4 (freq > freq_max) must die, the 59 dead features "
            "must die, and id 2 (freq == freq_max exactly) must SURVIVE"
        )
        assert vocab.n_sae_features == N_SAE  # the ORIGINAL width survives

    def test_the_producer_leaves_a_mask_the_MODEL_can_actually_be_built_from(
        self, hydra_compose, tmp_path
    ):
        """The full production sequence, ending in the real builder: band -> generated
        blacklist .txt -> cfg.feature_mask -> vocab -> compacted encoder."""
        write_stats(
            tmp_path,
            n_segments=100_000,
            counts={0: 1, 1: 2, 2: 50_000, 3: 50_001, 4: 99_999},
            n_sae_features=N_SAE,
        )
        cfg = _band_cfg(hydra_compose, fmin=DEFAULT_FREQ_MIN, fmax=DEFAULT_FREQ_MAX)

        vocab = _production_chain(cfg, tmp_path)
        # The producer really did write a blacklist and point cfg at it.
        assert cfg.feature_mask.mode == "blacklist"
        assert Path(cfg.feature_mask.path).is_file()

        model = build_encdec_meta_model_from_training_cfg(
            cfg, base_vocab=128, n_sae_features=N_SAE, feature_vocab=vocab
        )

        assert model.encoder.feature_embed.weight.shape[0] == 2
        assert model.encoder.config.n_sae_features == N_SAE
        for key, ve in model.encoder.value_embeds.items():
            assert ve.weight.shape[0] == 2, f"value_embeds[{key}] was not compacted"  # ty: ignore[not-subscriptable]

    def test_a_band_that_kills_EVERY_feature_fails_loudly_and_actionably(
        self, hydra_compose, tmp_path
    ):
        """A perfectly legal band (``resolve_blacklist_bounds`` accepts it: 0.4 < 0.5) can
        still blacklist the entire dictionary on a given corpus. The encoder would have
        ZERO embedding rows. That must be a loud startup failure that names the fix — not
        a zero-row ``nn.Embedding`` discovered on the first forward pass.

        Nothing tested this reachable path; the empty-vocab guard was only ever exercised
        by hand-built masks.
        """
        # Every feature's frequency lands outside (0.4, 0.5]: 4 are far too rare, the rest
        # are dead. The band therefore blacklists all 64.
        write_stats(
            tmp_path,
            n_segments=10_000,
            counts={0: 1, 1: 2, 2: 3, 3: 4},
            n_sae_features=N_SAE,
        )
        cfg = _band_cfg(hydra_compose, fmin=0.4, fmax=0.5)

        with pytest.raises(FeatureVocabError) as exc:
            _production_chain(cfg, tmp_path)

        msg = str(exc.value)
        assert "zero features" in msg
        assert "Widen the mask" in msg, f"the error must name the fix: {msg!r}"

    def test_an_inverted_band_is_rejected_before_it_can_reach_the_vocab(
        self, hydra_compose, tmp_path
    ):
        write_stats(tmp_path, n_segments=100, counts={0: 50}, n_sae_features=N_SAE)
        cfg = _band_cfg(hydra_compose, fmin=0.6, fmax=0.5)

        with pytest.raises(FeatureMaskError, match="strictly less than"):
            _production_chain(cfg, tmp_path)

    def test_nulling_BOTH_bounds_disables_compaction_entirely(self, hydra_compose, tmp_path):
        """The documented escape hatch (the one the FP-head error tells operators NOT to
        take). It must genuinely produce a full-width encoder — if it did not, the
        NotImplementedError's advice would be wrong.
        """
        write_stats(
            tmp_path,
            n_segments=10_000,
            counts={0: 1, 1: 2, 2: 5_000},
            n_sae_features=N_SAE,
        )
        cfg = _band_cfg(hydra_compose, fmin=None, fmax=None)

        vocab = _production_chain(cfg, tmp_path)

        assert vocab is None
        assert cfg.feature_mask is None, "the producer mutated cfg on the disabled path"


class TestClusterAwareProductionWiring:
    def test_shared_and_per_config_derivations_receive_their_consolidations(self):
        shared_calls = [
            node
            for node in ast.walk(_main_fn("build_shared_state"))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "effective_feature_vocab"
        ]
        per_config_calls = [
            node
            for node in ast.walk(_main_fn("train_one_config"))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "effective_feature_vocab"
        ]
        assert len(shared_calls) == len(per_config_calls) == 1
        # Union signature: ONE universe argument — the consolidation when clusters
        # are configured, else the plain mask — plus the SAE width. The conditional
        # expression is the pin: a regression to passing the bare mask alongside an
        # available consolidation (or dropping the fallback) changes this unparse.
        assert [ast.unparse(arg) for arg in shared_calls[0].args] == [
            "feature_clusters if feature_clusters is not None else shared_feature_mask",
            "n_sae_features",
        ]
        assert [ast.unparse(arg) for arg in per_config_calls[0].args] == [
            "per_config_clusters if per_config_clusters is not None else feature_mask",
            "n_sae_features",
        ]

    def test_per_config_consolidation_is_built_before_effective_derivation(self):
        fn = _main_fn("train_one_config")
        builds = [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "FeatureClusterConsolidation"
        ]
        derives = [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "effective_feature_vocab"
        ]
        assert len(builds) == len(derives) == 1
        assert builds[0] < derives[0]

    def test_resume_diagnostics_keep_cluster_then_migration_then_generic_order(self):
        cluster_guard = _first_line_matching("assert_resume_feature_clusters_match")
        migration_guard = _first_line_matching("_assert_not_pre_effective_compaction_checkpoint")
        generic_guard = _first_line_matching("assert_checkpoint_vocab_compatible")
        assert cluster_guard < migration_guard < generic_guard

    def test_effective_vocab_does_not_feed_hash_or_path_identity(self):
        for relative_path in (
            "steering/conditional/feature_mask.py",
            "steering/conditional/feature_clusters.py",
            "steering/packed_batch_io.py",
        ):
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            assert "effective_feature_vocab" not in source, relative_path
            assert "FeatureVocab" not in source, relative_path
