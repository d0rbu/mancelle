"""SFT inherits the encoder's feature vocabulary from the CHECKPOINT (#695 PR B).

``conditional_encdec_sft.py:1595`` was the SECOND site (after
``checkpoint_manager.py:523``) that inferred ``n_sae_features`` from the feature
table's ROW COUNT::

    n_sae_features = base_model_data["encoder.feature_embed.weight"].shape[0]

Once the encoder's dictionary is compacted that row count is **V**, not ``n_sae``. The
old line would therefore have silently redefined ``n_sae_features`` to ``V`` for the
whole SFT run — corrupting every consumer that bounds-checks ORIGINAL feature ids
against ``model.encoder.config.n_sae_features`` (``eval_adapter``,
``target_conditioning.validate_feature_dict``, ``model_detection``,
``harmfulness_eval``), the frequency-blacklist producer, and the resolved
``feature_mask`` itself.

Three separable contracts are pinned here:

1. **Inference** — the vocab comes from the checkpoint's state dict ALONE, and
   ``n_sae_features`` stays the ORIGINAL width (real checkpoints, real models).
2. **The vocab is a property of the CHECKPOINT, not of the SFT config.** SFT can
   neither create nor widen one. When the base checkpoint is compacted, the SFT run's
   resolved allowed-universe must **EQUAL** the checkpoint's vocab — strict equality,
   not a subset (owner decision). A pre-compaction masked base is accepted only when
   its canonical mask metadata exactly matches; uncompacted clusters remain invalid.
3. **Wiring** — ``main()`` is a 700-line body that needs a Hydra config, a tokenizer,
   DDP init, SFT parquet and a feature dir, so it cannot be driven in a unit test.
   ``TestSftWiring`` reads its AST instead and proves the call sequence above is
   actually present, in order. Without it, every functional test below could pass
   against a ``main()`` that still reads ``.shape[0]``.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

import scripts.train.conditional_encdec_sft as sft
from errors import FeatureVocabError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import GPTConfig
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
)
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    assert_resume_feature_vocab_matches,
    feature_vocab_from_state_dict,
)

N_SAE = 64
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)

# The whitelist whose allowed universe IS ``VOCAB``.
WHITELIST = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED)
# The blacklist whose allowed universe is ALSO ``VOCAB`` — same vocab, different mode.
# The guard compares VOCABULARIES, never mask modes.
BLACKLIST_SAME_VOCAB = FeatureMask(
    mode=FeatureMaskMode.blacklist,
    ids=tuple(i for i in range(N_SAE) if i not in ALLOWED),
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
NUM_BUCKETS = 4

SFT_SOURCE = Path(sft.__file__).read_text(encoding="utf-8")
SFT_TREE = ast.parse(SFT_SOURCE)


# ---------------------------------------------------------------------------
# Helpers — REAL models on disk. No mocks.
# ---------------------------------------------------------------------------


def _build(
    vocab: FeatureVocab | None,
    *,
    fp: FeaturePredictionConfig | None = None,
    seed: int = 3,
) -> EncoderDecoderGPT:
    """Build a real EncoderDecoderGPT.

    The meta-device ordering is LOAD-BEARING: ``nn.Embedding`` draws ``numel()``
    randoms in ``reset_parameters()`` at CONSTRUCTION on a real device, so building a
    compact model directly on CPU would consume a different RNG stream than the
    reference. ``meta`` draws nothing; ``to_empty`` allocates without drawing; the
    seed is set immediately before ``init_weights()``. This mirrors both production
    (``conditional_encdec_train.py``) and ``tests/nanochat/test_encdec_vocab_checkpoint.py``.
    """
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(**DEC_KW),  # ty: ignore[invalid-argument-type]
            EncoderConfig(**ENC_KW),  # ty: ignore[invalid-argument-type]
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=NUM_BUCKETS,
            feature_pred_config=fp,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    t.manual_seed(seed)
    m.init_weights()
    with t.no_grad():  # de-degenerate: init_weights zero-inits c_proj
        g = t.Generator().manual_seed(seed + 1)
        for p in m.parameters():
            p.copy_(t.randn(p.shape, generator=g) * 0.05)
    m.encoder.init_vocab_buffers()  # restore buffers clobbered by the randomisation
    return m


def _save(
    tmp_path,
    model: EncoderDecoderGPT,
    *,
    step: int = 5,
    mask: FeatureMask | None = None,
    resolved_clusters: ResolvedFeatureClusters | None = None,
) -> str:
    """Write a real pretrain-shaped encdec checkpoint the SFT loader can read."""
    user_config: dict[str, Any] = {
        "encoder": dict(ENC_KW),
        "cross_attention": {"frequency": 1, "n_head": None, "n_kv_head": None},
        "num_buckets": NUM_BUCKETS,
    }
    meta: dict[str, Any] = {
        "step": step,
        "model_config": dict(DEC_KW),
        "user_config": user_config,
    }
    if mask is not None:
        meta["feature_mask_canonical"] = canonical_dict(mask)
    if resolved_clusters is not None:
        meta.update(feature_clusters_checkpoint_metadata(resolved_clusters))
        user_config["feature_clusters"] = {"path": str(resolved_clusters.source_path)}
    if model.feature_vocab is not None:
        meta["feature_vocab"] = model.feature_vocab.metadata_dict()
    save_checkpoint(str(tmp_path), step, model.state_dict(), None, meta, rank=0)
    return str(tmp_path)


def _resolved_clusters(
    tmp_path: Path,
    groups: tuple[tuple[int, ...], ...],
) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=groups)
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=N_SAE,
    )
    return ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=descriptor,
    )


def _identity_cluster_consolidation(tmp_path: Path) -> FeatureClusterConsolidation:
    """Build the normalized-identity guard state, independent of parser reachability."""
    resolved = _resolved_clusters(tmp_path, ((1, 2),))
    ordinary = FeatureClusterConsolidation.build(resolved, None)
    return replace(
        ordinary,
        rep_of=t.arange(N_SAE, dtype=t.int64),
        consolidated_away=t.zeros(N_SAE, dtype=t.bool),
        allowed_universe=t.ones(N_SAE, dtype=t.bool),
        cluster_allowed_universe=t.ones(N_SAE, dtype=t.bool),
        is_identity_after_mask=True,
    )


def _sft_load_and_build(
    checkpoint_dir: str,
    *,
    step: int = 5,
    fp: FeaturePredictionConfig | None = None,
) -> tuple[EncoderDecoderGPT, int, FeatureVocab | None]:
    """Replay the SFT script's exact model-build sequence against a real checkpoint.

    ``load_checkpoint`` → ``feature_vocab_from_state_dict`` → meta-device construction
    with ``feature_vocab=`` → ``to_empty`` → ``init_buffers`` → strict/assign load →
    ``assert_vocab_consistent``.  ``TestSftWiring`` pins that ``main()`` performs the
    same sequence.
    """
    model_data, _, _ = load_checkpoint(checkpoint_dir, step, t.device("cpu"))
    n_sae_features, vocab = feature_vocab_from_state_dict(model_data)
    with t.device("meta"):
        model = EncoderDecoderGPT(
            GPTConfig(**DEC_KW),  # ty: ignore[invalid-argument-type]
            EncoderConfig(**ENC_KW),  # ty: ignore[invalid-argument-type]
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=n_sae_features,
            num_buckets=NUM_BUCKETS,
            feature_pred_config=fp,
            feature_vocab=vocab,
        )
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.assert_vocab_consistent()
    return model, n_sae_features, vocab


def _fwd(model: EncoderDecoderGPT) -> t.Tensor:
    model.eval()
    idx = t.arange(16).unsqueeze(0) % 128
    seg_map = (t.arange(16) // 8).unsqueeze(0)
    fids = t.zeros(1, 2, 5, dtype=t.int64)
    mags = t.zeros(1, 2, 5, dtype=t.int64)
    mask = t.zeros(1, 2, 5, dtype=t.bool)
    for s in range(2):
        fids[0, s, :3] = t.tensor([10, 20, 30])
        mags[0, s, :3] = t.tensor([1, 2, 3])
        mask[0, s, :3] = True
    with t.no_grad():
        return model(idx, None, fids, mags, mask, seg_map)


# ---------------------------------------------------------------------------
# AST helpers for the wiring tests
# ---------------------------------------------------------------------------


def _sft_function(name: str) -> ast.FunctionDef:
    for node in SFT_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"conditional_encdec_sft.{name} not found")


def _main_fn() -> ast.FunctionDef:
    return _sft_function("main")


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    """Every call to ``name`` anywhere under ``node`` (bare or as a method)."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if (isinstance(fn, ast.Name) and fn.id == name) or (
            isinstance(fn, ast.Attribute) and fn.attr == name
        ):
            out.append(sub)
    return out


def _is_model_load(node: ast.AST) -> bool:
    """``model.load_state_dict(...)`` — NOT ``optimizer.load_state_dict(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
    )


def _own_nodes(stmt: ast.stmt):
    """Nodes belonging to ``stmt`` itself, not to any statement nested inside it.

    Without this a containing ``if`` would "own" the calls in its branches, and an
    ordering assertion over a block would silently pass (or fail) for the wrong reason.
    """
    stack = [c for c in ast.iter_child_nodes(stmt) if not isinstance(c, ast.stmt)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, ast.stmt))


def _blocks(fn: ast.AST):
    """Every statement list (``body`` / ``orelse`` / ``finalbody``) under ``fn``."""
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


# ===========================================================================
# 1. Inference — the vocab comes from the checkpoint, not from a row count
# ===========================================================================


class TestCheckpointVocabInference:
    def test_n_sae_features_is_the_ORIGINAL_width_not_V(self, tmp_path):
        """THE trap. The compacted table has 12 rows; n_sae_features is still 64."""
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)
        model_data, _, _ = load_checkpoint(d, 5, t.device("cpu"))

        n_sae_features, vocab = feature_vocab_from_state_dict(model_data)

        assert n_sae_features == N_SAE == 64
        assert vocab == VOCAB
        assert vocab is not None and vocab.size == 12

    def test_the_row_count_would_have_LIED(self, tmp_path):
        """Pin the bug the old line had: shape[0] is V, and V != n_sae_features.

        If this ever stops being true the compaction has silently stopped working and
        the rest of this file would pass vacuously.
        """
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)
        model_data, _, _ = load_checkpoint(d, 5, t.device("cpu"))

        row_count = int(model_data["encoder.feature_embed.weight"].shape[0])

        assert row_count == 12
        assert row_count != N_SAE, "compaction is not actually compacting"

    def test_vocab_is_recovered_from_the_state_dict_ALONE(self, tmp_path):
        """No training config, no sidecar, no meta (requirement 1)."""
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)
        model_data = t.load(Path(d) / "model_000005.pt", weights_only=True, map_location="cpu")

        _, vocab = feature_vocab_from_state_dict(model_data)

        assert vocab is not None
        assert vocab.original_ids == ALLOWED

    def test_legacy_checkpoint_yields_no_vocab_and_the_row_count_is_correct(self, tmp_path):
        """Requirement 5: an uncompacted checkpoint keeps working, unchanged."""
        d = _save(tmp_path, _build(None))
        model_data, _, _ = load_checkpoint(d, 5, t.device("cpu"))

        n_sae_features, vocab = feature_vocab_from_state_dict(model_data)

        assert vocab is None
        assert n_sae_features == N_SAE
        assert int(model_data["encoder.feature_embed.weight"].shape[0]) == N_SAE


# ===========================================================================
# 2. The SFT model build — real checkpoints, real strict loads
# ===========================================================================


class TestSftModelBuild:
    def test_compacted_checkpoint_loads_strict_and_stays_consistent(self, tmp_path):
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)

        model, n_sae_features, vocab = _sft_load_and_build(d)

        assert n_sae_features == N_SAE
        assert vocab == VOCAB
        assert model.encoder.config.n_sae_features == N_SAE
        assert model.encoder.feature_embed.weight.shape[0] == 12
        assert model.feature_vocab == VOCAB
        model.assert_vocab_consistent()

    def test_logits_survive_the_SFT_load_bitwise(self, tmp_path):
        m = _build(VOCAB)
        before = _fwd(m)
        d = _save(tmp_path, m, mask=WHITELIST)

        loaded, _, _ = _sft_load_and_build(d)

        assert t.equal(before, _fwd(loaded))

    def test_the_remap_is_correct_after_the_assign_load(self, tmp_path):
        """``assign=True`` REPLACES the persistent ``feature_vocab_ids`` tensor with the
        checkpoint's. ``_to_compact`` derives the remap per-forward from that buffer (no stored
        map), so the loaded model must map every ORIGINAL id to its compact row."""
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)

        model, _, _ = _sft_load_and_build(d)

        ids = t.tensor([list(ALLOWED)], dtype=t.int64)
        mask = t.ones(1, len(ALLOWED), dtype=t.bool)
        got = model.encoder._to_compact(ids, mask)
        assert got.tolist() == [list(range(len(ALLOWED)))]

    # ``test_a_STALE_derived_map_is_caught`` was deleted: there is no longer a separately-stored
    # derived map to go stale. The only thing ``assert_vocab_consistent`` can catch is a
    # ``feature_vocab_ids`` buffer that disagrees with the configured vocab, which is exactly
    # ``test_a_checkpoint_whose_ids_differ_is_caught`` below.

    def test_a_checkpoint_whose_ids_differ_is_caught(self, tmp_path):
        """Same V, different ids ⇒ shapes agree, ``strict=True`` is happy, and every
        embedding row silently means a different feature. ``assert_vocab_consistent``
        is the only thing standing between that and a corrupt run."""
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)
        model_data, _, _ = load_checkpoint(d, 5, t.device("cpu"))
        other = FeatureVocab(
            n_sae_features=N_SAE, original_ids=tuple(i + 1 for i in ALLOWED[:-1]) + (2,)
        )
        assert other.size == VOCAB.size and other != VOCAB

        with t.device("meta"):
            model = EncoderDecoderGPT(
                GPTConfig(**DEC_KW),  # ty: ignore[invalid-argument-type]
                EncoderConfig(**ENC_KW),  # ty: ignore[invalid-argument-type]
                CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
                n_sae_features=N_SAE,
                num_buckets=NUM_BUCKETS,
                feature_vocab=other,
            )
        model.to_empty(device="cpu")
        model.init_buffers()
        model.load_state_dict(model_data, strict=True, assign=True)  # no shape error!

        with pytest.raises(FeatureVocabError, match="(?i)disagree|reinterpret"):
            model.assert_vocab_consistent()

    def test_legacy_checkpoint_builds_full_width(self, tmp_path):
        d = _save(tmp_path, _build(None))

        model, n_sae_features, vocab = _sft_load_and_build(d)

        assert vocab is None
        assert n_sae_features == N_SAE
        assert model.feature_vocab is None
        assert model.encoder.feature_embed.weight.shape[0] == N_SAE
        assert not hasattr(model.encoder, "feature_vocab_ids")

    def test_legacy_logits_survive_the_SFT_load_bitwise(self, tmp_path):
        m = _build(None)
        before = _fwd(m)
        d = _save(tmp_path, m)

        loaded, _, _ = _sft_load_and_build(d)

        assert t.equal(before, _fwd(loaded))

    def test_compacted_base_plus_an_fp_head_builds_the_compact_shape(self, tmp_path):
        """The model primitive reconstructs the compact checkpoint shape.

        The SFT entry point separately rejects compact-FP checkpoints before
        data/model work until compact SFT targets are implemented.
        """
        fp = FeaturePredictionConfig(weight=0.1, layer=-1)
        d = _save(tmp_path, _build(VOCAB, fp=fp), mask=WHITELIST)

        model, _, _ = _sft_load_and_build(d, fp=fp)
        assert model.feature_pred_head is not None
        assert model.feature_pred_head.out_features == VOCAB.size


# ===========================================================================
# 3. The startup guard — STRICT effective-universe equality on every checkpoint
# ===========================================================================


class TestSftMaskEqualityGuard:
    """``_assert_sft_universe_matches_checkpoint_vocab`` (#1279 D5).

    The vocab is a property of the CHECKPOINT. SFT inherits it and can neither create
    nor widen/narrow it. Equality covers the mask-only and clustered effective
    universes, including full-width legacy checkpoints.
    """

    def test_an_equal_mask_passes(self):
        sft._assert_sft_universe_matches_checkpoint_vocab(
            WHITELIST,
            WHITELIST,
            VOCAB,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={"feature_mask_canonical": canonical_dict(WHITELIST)},
        )

    def test_a_blacklist_with_the_SAME_allowed_universe_passes(self):
        """Modes are irrelevant — only the resolved vocabulary is compared."""
        assert BLACKLIST_SAME_VOCAB.mode is not WHITELIST.mode

        sft._assert_sft_universe_matches_checkpoint_vocab(
            BLACKLIST_SAME_VOCAB,
            BLACKLIST_SAME_VOCAB,
            VOCAB,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={"feature_mask_canonical": canonical_dict(BLACKLIST_SAME_VOCAB)},
        )

    def test_a_SUBSET_mask_RAISES(self):
        """Owner decision: strict equality, NOT a subset. Structurally a subset would
        be safe; this is policy, and it must not silently relax into a subset check."""
        subset = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED[:5])

        with pytest.raises(
            FeatureVocabError,
            match=(
                r"allows 5 features.*trained on 12.*0 allowed.*e\.g\. \[\].*"
                r"7 in the checkpoint.*e\.g\. \[20, 21, 30, 31, 40, 55, 63\]"
            ),
        ):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                subset,
                subset,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_a_SUPERSET_mask_raises(self):
        superset = FeatureMask(mode=FeatureMaskMode.whitelist, ids=ALLOWED + (1, 2, 4, 5))

        with pytest.raises(
            FeatureVocabError,
            match=(
                r"allows 16 features.*trained on 12.*4 allowed.*"
                r"e\.g\. \[1, 2, 4, 5\].*0 in the checkpoint.*e\.g\. \[\]"
            ),
        ):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                superset,
                superset,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_a_disjoint_mask_raises(self):
        other = FeatureMask(
            mode=FeatureMaskMode.whitelist,
            ids=tuple(i for i in range(N_SAE) if i not in ALLOWED)[:12],
        )

        with pytest.raises(
            FeatureVocabError,
            match=(
                r"allows 12 features.*trained on 12.*12 allowed.*"
                r"e\.g\. \[1, 2, 4, 5, 6, 8, 9, 12\].*12 in the checkpoint.*"
                r"e\.g\. \[0, 3, 7, 10, 11, 20, 21, 30\]"
            ),
        ):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                other,
                other,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_NO_mask_against_a_compacted_checkpoint_raises(self):
        """``feature_mask: null`` must RAISE — the checkpoint's vocab is never
        silently adopted as the eval/SFT universe (§4.9.4)."""
        with pytest.raises(
            FeatureVocabError,
            match=r"no feature_mask is configured.*never silently adopted",
        ):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                None,
                None,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_a_whitelist_of_EVERYTHING_raises(self):
        """An all-inclusive whitelist resolves to the identity universe, i.e. NO
        compaction — so it cannot match a compacted checkpoint. ``from_mask``
        normalises identity to ``None``; the guard must not read that as "no mask
        configured, skip"."""
        everything = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(range(N_SAE)))
        assert FeatureVocab.from_mask(everything, N_SAE) is None

        with pytest.raises(FeatureVocabError, match="allows ALL"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                everything,
                everything,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_the_error_names_the_dump_feature_mask_recovery_route(self):
        with pytest.raises(FeatureVocabError, match="dump_feature_mask") as exc:
            sft._assert_sft_universe_matches_checkpoint_vocab(
                None,
                None,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt/d12",
                base_meta={},
            )

        msg = str(exc.value)
        assert "scripts.analysis.dump_feature_mask" in msg
        assert "/ckpt/d12" in msg

    def test_a_legacy_checkpoint_with_a_compacting_mask_and_absent_metadata_raises(self):
        weird = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 2, 3))

        with pytest.raises(
            FeatureVocabError,
            match=r"cannot be verified.*pre-compaction.*dump_feature_mask",
        ):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                weird,
                weird,
                None,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_a_legacy_masked_checkpoint_with_equal_canonical_metadata_passes(self):
        messages: list[str] = []
        sink = logger.add(messages.append, level="INFO", format="{message}")
        try:
            sft._assert_sft_universe_matches_checkpoint_vocab(
                WHITELIST,
                WHITELIST,
                None,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={"feature_mask_canonical": canonical_dict(WHITELIST)},
            )
        finally:
            logger.remove(sink)

        matching = [
            message
            for message in messages
            if "pre-compaction masked base; canonical mask metadata matches the configured mask"
            in message
        ]
        assert len(matching) == 1

    def test_a_legacy_masked_checkpoint_with_differing_canonical_metadata_raises(self):
        with pytest.raises(FeatureVocabError, match=r"disagrees.*configured"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                WHITELIST,
                WHITELIST,
                None,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={"feature_mask_canonical": canonical_dict(BLACKLIST_SAME_VOCAB)},
            )

    def test_a_legacy_checkpoint_with_no_mask_passes(self):
        sft._assert_sft_universe_matches_checkpoint_vocab(
            None,
            None,
            None,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={},
        )

    def test_a_legacy_checkpoint_with_an_allows_all_whitelist_passes(self):
        everything = FeatureMask(FeatureMaskMode.whitelist, tuple(range(N_SAE)))

        sft._assert_sft_universe_matches_checkpoint_vocab(
            everything,
            everything,
            None,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={},
        )

    def test_a_legacy_checkpoint_with_compacting_clusters_raises(self, tmp_path):
        consolidation = FeatureClusterConsolidation.build(
            _resolved_clusters(tmp_path, ((1, 2),)),
            None,
        )

        with pytest.raises(FeatureVocabError, match="configured feature_clusters"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                consolidation,
                None,
                None,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_legacy_null_mask_metadata_agreeing_with_configured_null_passes(self):
        sft._assert_sft_universe_matches_checkpoint_vocab(
            None,
            None,
            None,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={"feature_mask_canonical": None},
        )

    def test_legacy_mask_metadata_disagreeing_with_configured_null_raises(self):
        with pytest.raises(FeatureVocabError, match="disagrees.*configured"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                None,
                None,
                None,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={"feature_mask_canonical": canonical_dict(WHITELIST)},
            )

    def test_compacted_base_with_agreeing_mask_metadata_passes(self):
        sft._assert_sft_universe_matches_checkpoint_vocab(
            WHITELIST,
            WHITELIST,
            VOCAB,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={"feature_mask_canonical": canonical_dict(WHITELIST)},
        )

    def test_compacted_base_with_disagreeing_mask_metadata_raises(self):
        with pytest.raises(FeatureVocabError, match="despite an equal state-dict vocabulary"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                WHITELIST,
                WHITELIST,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={"feature_mask_canonical": canonical_dict(BLACKLIST_SAME_VOCAB)},
            )

    def test_compacted_base_without_mask_metadata_uses_state_dict_authority(self):
        messages: list[str] = []
        sink = logger.add(messages.append, level="INFO", format="{message}")
        try:
            sft._assert_sft_universe_matches_checkpoint_vocab(
                WHITELIST,
                WHITELIST,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )
        finally:
            logger.remove(sink)

        assert any(
            "state-dict feature vocabulary as the authority" in message for message in messages
        )

    def test_clustered_effective_vocab_equal_to_checkpoint_passes(self, tmp_path):
        cluster_mask = FeatureMask(
            FeatureMaskMode.whitelist,
            tuple(sorted((*ALLOWED, 1))),
        )
        resolved = _resolved_clusters(tmp_path, ((0, 1),))
        consolidation = FeatureClusterConsolidation.build(resolved, cluster_mask)
        assert effective_feature_vocab(consolidation, N_SAE) == VOCAB

        sft._assert_sft_universe_matches_checkpoint_vocab(
            consolidation,
            cluster_mask,
            VOCAB,
            N_SAE,
            checkpoint_dir="/ckpt",
            base_meta={"feature_mask_canonical": canonical_dict(cluster_mask)},
        )

    def test_clustered_effective_vocab_with_one_extra_id_raises_samples(self, tmp_path):
        cluster_mask = FeatureMask(
            FeatureMaskMode.whitelist,
            tuple(sorted((*ALLOWED, 1, 2))),
        )
        resolved = _resolved_clusters(tmp_path, ((0, 1),))
        consolidation = FeatureClusterConsolidation.build(resolved, cluster_mask)

        with pytest.raises(FeatureVocabError, match="feature_clusters descriptor") as exc:
            sft._assert_sft_universe_matches_checkpoint_vocab(
                consolidation,
                cluster_mask,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

        assert "e.g. [2]" in str(exc.value)
        assert "e.g. []" in str(exc.value)

    def test_identity_effective_cluster_universe_vs_legacy_raises(self, tmp_path):
        """Pin the normalized-None guard cell independently of parser/build shape."""
        identity = _identity_cluster_consolidation(tmp_path)
        assert effective_feature_vocab(identity, N_SAE) is None

        with pytest.raises(FeatureVocabError, match="configured feature_clusters"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                identity,
                None,
                None,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_identity_effective_cluster_universe_vs_compacted_base_raises(self, tmp_path):
        identity = _identity_cluster_consolidation(tmp_path)

        with pytest.raises(FeatureVocabError, match="feature_clusters descriptor=.*identity"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                identity,
                None,
                VOCAB,
                N_SAE,
                checkpoint_dir="/ckpt",
                base_meta={},
            )

    def test_the_guard_runs_against_a_REAL_compacted_checkpoint(self, tmp_path):
        """End-to-end: the vocab the guard compares against is the one the SFT script
        actually reads off disk."""
        d = _save(tmp_path, _build(VOCAB), mask=WHITELIST)
        model_data, _, meta = load_checkpoint(d, 5, t.device("cpu"))
        n_sae_features, vocab = feature_vocab_from_state_dict(model_data)

        sft._assert_sft_universe_matches_checkpoint_vocab(
            WHITELIST,
            WHITELIST,
            vocab,
            n_sae_features,
            checkpoint_dir=d,
            base_meta=meta,
        )
        with pytest.raises(FeatureVocabError, match="never silently adopted"):
            sft._assert_sft_universe_matches_checkpoint_vocab(
                None,
                None,
                vocab,
                n_sae_features,
                checkpoint_dir=d,
                base_meta=meta,
            )

    def test_REAL_clustered_checkpoint_round_trip_reaches_equal_guard(self, tmp_path):
        cluster_mask = FeatureMask(
            FeatureMaskMode.whitelist,
            tuple(sorted((*ALLOWED, 1))),
        )
        resolved = _resolved_clusters(tmp_path, ((0, 1),))
        consolidation = FeatureClusterConsolidation.build(resolved, cluster_mask)
        clustered_vocab = effective_feature_vocab(consolidation, N_SAE)
        assert clustered_vocab == VOCAB
        checkpoint_dir = _save(
            tmp_path,
            _build(clustered_vocab),
            mask=cluster_mask,
            resolved_clusters=resolved,
        )
        model_data, _, meta = load_checkpoint(checkpoint_dir, 5, t.device("cpu"))
        n_sae_features, checkpoint_vocab = feature_vocab_from_state_dict(model_data)

        sft._assert_sft_universe_matches_checkpoint_vocab(
            consolidation,
            cluster_mask,
            checkpoint_vocab,
            n_sae_features,
            checkpoint_dir=checkpoint_dir,
            base_meta=meta,
        )


# ===========================================================================
# 4. Resume metadata
# ===========================================================================


class TestSftResumeVocabMeta:
    def test_the_meta_SFT_writes_satisfies_the_resume_guard(self):
        """The write side (``vocab.metadata_dict()``) and the read side
        (``assert_resume_feature_vocab_matches``) must agree on the dict shape, or
        every compacted SFT resume raises on a checkpoint it wrote itself."""
        meta = {"step": 5, "feature_vocab": VOCAB.metadata_dict()}

        assert_resume_feature_vocab_matches(VOCAB, meta, checkpoint_dir="/sft")

    def test_a_resume_meta_without_the_vocab_key_is_FINE(self):
        """ABSENCE is fine; only DISAGREEMENT is a corruption signal.

        The persistent ``feature_vocab_ids`` / ``feature_vocab_n_sae`` buffers ARE the record,
        and the resume path validates them before this runs (``assert_checkpoint_vocab_compatible``
        → ``load_state_dict(strict=True)`` → ``assert_vocab_consistent()``). So an absent meta
        block leaves nothing unverified. Raising on it made the sidecar MANDATORY on every
        resume — throwing away checkpoints that are provably recoverable, since metas routinely
        get thinned — and contradicted the contract that ``checkpoint_manager`` and
        ``dump_feature_mask`` both state outright.
        """
        assert_resume_feature_vocab_matches(VOCAB, {"step": 5}, checkpoint_dir="/sft")
        assert_resume_feature_vocab_matches(VOCAB, None, checkpoint_dir="/sft")

    def test_a_resume_meta_with_a_DIFFERENT_vocab_raises(self):
        other = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED[:-1])
        meta = {"step": 5, "feature_vocab": other.metadata_dict()}

        with pytest.raises(FeatureVocabError, match="(?i)mismatch"):
            assert_resume_feature_vocab_matches(VOCAB, meta, checkpoint_dir="/sft")

    def test_no_vocab_means_the_guard_is_a_noop(self):
        """I2/I5: a legacy SFT run writes no ``feature_vocab`` key and resumes exactly
        as it does today."""
        assert_resume_feature_vocab_matches(None, {"step": 5}, checkpoint_dir="/sft")
        assert_resume_feature_vocab_matches(None, None, checkpoint_dir="/sft")


# ===========================================================================
# 5. Wiring — main() really does all of the above
# ===========================================================================


class TestSftWiring:
    def test_the_row_count_inference_is_GONE(self):
        """The exact line this PR exists to delete.

        ``main()`` must not reach into the feature table at all: any expression naming
        ``feature_embed`` is, by construction, either the old ``.shape[0]`` inference
        or a new way to reinvent it."""
        body = ast.unparse(_main_fn())

        assert "feature_embed" not in body, (
            "conditional_encdec_sft.main() still reaches into encoder.feature_embed. "
            "Under compaction its row count is V, not n_sae_features — inferring "
            "n_sae_features from it silently redefines the whole feature space."
        )
        assert "feature_embed_key" not in SFT_SOURCE

    def test_main_calls_feature_vocab_from_state_dict_on_the_BASE_state_dict(self):
        calls = _calls_named(_main_fn(), "feature_vocab_from_state_dict")

        assert len(calls) == 1, (
            "main() must resolve (n_sae_features, feature_vocab) from the base "
            "checkpoint's state dict exactly once"
        )
        (arg,) = calls[0].args
        assert isinstance(arg, ast.Name) and arg.id == "base_model_data"

    def test_main_passes_feature_vocab_to_EncoderDecoderGPT(self):
        ctors = _calls_named(_main_fn(), "EncoderDecoderGPT")

        assert ctors, "main() no longer constructs an EncoderDecoderGPT"
        for ctor in ctors:
            kwargs = {kw.arg for kw in ctor.keywords}
            assert "feature_vocab" in kwargs, (
                "EncoderDecoderGPT is built without feature_vocab — the SFT model "
                "would be full-width while the checkpoint is compacted, and "
                "load_state_dict would fail with PyTorch's bare 'size mismatch'."
            )

    def test_every_model_load_state_dict_is_followed_by_assert_vocab_consistent(self):
        """``assign=True`` replaces the persistent vocab buffer but NOT the derived
        map. Every model load must be followed by the consistency check, in the same
        block. (``optimizer.load_state_dict`` is deliberately not matched.)"""
        checked = 0
        for block in _blocks(_main_fn()):
            for i, stmt in enumerate(block):
                if not any(_is_model_load(n) for n in _own_nodes(stmt)):
                    continue
                assert any(_calls_named(s, "assert_vocab_consistent") for s in block[i + 1 :]), (
                    f"model.load_state_dict at line {stmt.lineno} is not followed by "
                    "model.assert_vocab_consistent() in the same block — a checkpoint "
                    "whose vocabulary differs from the configured one would load "
                    "silently."
                )
                checked += 1
        assert checked == 2, (
            f"expected the 2 known model-load sites (resume + base), found {checked}"
        )

    def test_main_wires_the_resume_vocab_guard(self):
        assert _calls_named(_main_fn(), "assert_resume_feature_vocab_matches"), (
            "assert_resume_feature_vocab_matches is not called — a compacted SFT "
            "resume would not verify the checkpoint's recorded vocabulary"
        )

    def test_main_wires_the_startup_equality_guard(self):
        assert _calls_named(_main_fn(), "_assert_sft_universe_matches_checkpoint_vocab"), (
            "the SFT effective-universe/vocab equality guard is defined but never called"
        )

    def test_generator_fold_wires_representatives_only_assertion(self):
        calls = _calls_named(
            _sft_function("make_sft_data_generator"),
            "assert_representatives_only",
        )
        assert len(calls) == 1, (
            "make_sft_data_generator must assert representative-only IDs exactly once "
            f"after clustered selection, found {len(calls)} calls"
        )
        assert ast.unparse(calls[0].func) == "consolidation.assert_representatives_only"

    @pytest.mark.parametrize(
        "call_name",
        [
            "resolve_feature_clusters",
            "FeatureClusterConsolidation.build",
            "append_feature_clusters_leaf",
            "feature_clusters_checkpoint_metadata",
        ],
    )
    def test_main_wires_the_cluster_contract(self, call_name: str):
        terminal_name = call_name.rsplit(".", 1)[-1]
        calls = _calls_named(_main_fn(), terminal_name)
        assert any(ast.unparse(call.func) == call_name for call in calls), (
            f"main() does not call {call_name}; SFT clustering is only partially wired"
        )

    @pytest.mark.parametrize(
        "call_name",
        [
            "assert_resume_feature_clusters_match",
            "reject_feature_clusters_metadata",
        ],
    )
    def test_main_wires_both_base_and_resume_cluster_metadata_gates(
        self,
        call_name: str,
    ):
        calls = _calls_named(_main_fn(), call_name)
        assert len(calls) == 2, (
            f"main() must call {call_name} once for the base checkpoint and once "
            f"for the SFT resume checkpoint, found {len(calls)}"
        )

    def test_main_writes_the_feature_vocab_into_the_save_meta(self):
        writes = [
            n
            for n in ast.walk(_main_fn())
            if isinstance(n, ast.Assign)
            and any(
                isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "_sft_save_meta"
                and isinstance(tgt.slice, ast.Constant)
                and tgt.slice.value == "feature_vocab"
                for tgt in n.targets
            )
        ]

        assert len(writes) == 1, (
            "_sft_save_meta['feature_vocab'] is never written — a compacted SFT "
            "checkpoint could not be resumed (assert_resume_feature_vocab_matches "
            "demands the key)"
        )
        assert _calls_named(writes[0].value, "metadata_dict"), (
            "the feature_vocab meta must be vocab.metadata_dict() — the exact shape "
            "assert_resume_feature_vocab_matches compares against"
        )

    def test_the_vocab_meta_is_written_only_when_compacted(self):
        """I2/I5: a no-mask SFT run must write a byte-identical meta to pre-#695."""
        guarded = False
        for node in ast.walk(_main_fn()):
            if not isinstance(node, ast.If):
                continue
            writes_vocab = any(
                isinstance(n, ast.Assign)
                and any(
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value == "feature_vocab"
                    for tgt in n.targets
                )
                for n in ast.walk(node)
            )
            if writes_vocab and "is not None" in ast.unparse(node.test):
                guarded = True
        assert guarded, "the feature_vocab meta write is not guarded by an `is not None` check"

    def test_the_vocab_imports_are_at_MODULE_scope(self):
        """Hard project rule: no function-local imports, no exceptions."""
        for node in ast.walk(_main_fn()):
            assert not isinstance(node, ast.Import | ast.ImportFrom), (
                f"function-local import in main() at line {node.lineno}: {ast.unparse(node)}"
            )
        for name in (
            "FeatureVocab",
            "feature_vocab_from_state_dict",
            "assert_resume_feature_vocab_matches",
            "FeatureVocabError",
        ):
            assert hasattr(sft, name), f"{name} is not imported at module scope"


class TestSftLoadsAreGuardedLikeEveryOtherEncdecLoad:
    """SFT's two ``load_state_dict(strict=True, assign=True)`` calls were the ONLY production
    encdec loads not preceded by ``assert_checkpoint_vocab_compatible``.

    ``assert_vocab_consistent()`` runs AFTER the load — but a compacted-vs-full-width
    mismatch blows up INSIDE ``load_state_dict`` first, as PyTorch's bare::

        RuntimeError: Error(s) in loading state_dict for EncoderDecoderGPT:
          Unexpected key(s) in state_dict: "encoder.feature_vocab_ids", ...
          size mismatch for encoder.feature_embed.weight: copying a param with shape
          torch.Size([V, d]) from checkpoint, the shape in current model is torch.Size([65536, d])

    — the exact unactionable error the guard exists to eliminate. The operator gets no
    mention of compaction and no ``dump_feature_mask`` remedy.

    Reachable: the ``bl_``/``wl_`` leaf hashes only ``(mode, ids)``, so a pre-#695 full-width
    run and a post-#695 compacted run with the SAME mask land in the same base dir — hence
    the same ``<dataset>_sft_conditioned_checkpoints/`` dir.
    """

    def _model_load_functions(self) -> list[ast.FunctionDef]:
        tree = ast.parse(Path(sft.__file__).read_text(encoding="utf-8"))
        out = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "load_state_dict"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "model"
                for n in ast.walk(fn)
            ):
                out.append(fn)
        return out

    def test_every_model_load_is_followed_by_assert_vocab_consistent(self):
        """AST, not text. The universal invariant: after ANY load, the encoder's persistent ids
        buffer must be checked against the Python vocab the model was built with.

        ``load_state_dict(assign=True)`` REBINDS the ids buffer to the checkpoint's, so this is
        the only thing standing between a wrong checkpoint and every embedding row meaning a
        different feature.
        """
        fns = self._model_load_functions()
        assert fns, "no model.load_state_dict found — this test is stale"
        for fn in fns:
            checks = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "assert_vocab_consistent"
            ]
            loads = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "load_state_dict"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "model"
            ]
            assert len(checks) >= len(loads), (
                f"{fn.name}: {len(loads)} model.load_state_dict call(s) but only "
                f"{len(checks)} assert_vocab_consistent() call(s)."
            )

    def test_the_RESUME_load_is_guarded_by_assert_checkpoint_vocab_compatible(self):
        """The resume load reads a DIFFERENT checkpoint (``sft_model_data``) than the one the
        model's vocabulary came from, so the two genuinely can disagree — a legacy base (model
        full-width) resuming a compacted SFT checkpoint from the same directory. That is what
        the guard is for, and PyTorch's bare "size mismatch" is what it replaces.
        """
        fns = self._model_load_functions()
        guarded = [
            fn
            for fn in fns
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "assert_checkpoint_vocab_compatible"
                for n in ast.walk(fn)
            )
        ]
        assert guarded, (
            "no function loads the model under assert_checkpoint_vocab_compatible — the "
            "resume path lost its guard"
        )

    def test_the_guard_on_the_BASE_load_would_be_TAUTOLOGICAL(self):
        """Why the base load carries no such guard, pinned so it is not "restored" later.

        The model's vocabulary IS the base checkpoint's: SFT reads it with
        ``feature_vocab_from_state_dict(base_model_data)`` and hands it straight to the
        constructor. So a guard there would re-run the same pure function on the same dict and
        compare the result with itself — every branch unreachable, at the cost of re-parsing V
        ids (61,835 for the shipped band) on every start. A dead guard on the most dangerous
        load in the file is worse than none, because it reads as protection.

        Demonstrated over both shapes: compacted, and legacy full-width.
        """
        for vocab in (FeatureVocab(n_sae_features=N_SAE, original_ids=(1, 4, 9)), None):
            sd: dict = {FEATURE_EMBED_KEY: t.zeros(vocab.size if vocab else N_SAE, 8)}
            if vocab is not None:
                sd[VOCAB_IDS_KEY] = t.tensor(vocab.original_ids, dtype=t.int64)
                sd[VOCAB_N_SAE_KEY] = t.tensor(N_SAE, dtype=t.int64)

            _, derived = feature_vocab_from_state_dict(sd)
            assert derived == vocab, "fixture is wrong: the vocab must come FROM the state dict"
            # Exactly what SFT would do: guard the model's own vocabulary against the very
            # dict it was derived from. It cannot fail, for any input.
            assert_checkpoint_vocab_compatible(derived, sd, checkpoint_dir="/base")

    def test_a_COMPACTED_sft_checkpoint_into_a_FULL_WIDTH_model_is_actionable(self):
        """The reachable case: a legacy base (so the SFT model is full-width) resuming from
        a compacted SFT checkpoint. Must name compaction and the recovery command."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(1, 4, 9))
        compacted_sd = {
            FEATURE_EMBED_KEY: t.zeros(vocab.size, 8),
            VOCAB_IDS_KEY: t.tensor(vocab.original_ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(N_SAE, dtype=t.int64),
        }
        with pytest.raises(FeatureVocabError, match="COMPACTED") as exc:
            assert_checkpoint_vocab_compatible(
                None, compacted_sd, checkpoint_dir="/ckpt/sft/bl_deadbeefdeadbeef"
            )
        msg = str(exc.value)
        assert "COMPACTED" in msg
        assert "dump_feature_mask" in msg
        assert "size mismatch" not in msg, "must not be PyTorch's bare error"
