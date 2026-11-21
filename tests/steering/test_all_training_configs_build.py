"""EVERY shipped encoder-decoder training config must build the production model (#695).

The RED this module was written against: ``conditional_encdec_d12_fp_4xh100``,
``conditional_encdec_d12_fp10_4xh100`` and ``conditional_encdec_d12_fp_s32`` all raised
``NotImplementedError`` at model construction before compact FP support. Each ``defaults:``-inherits
``conditional_encdec_train`` — and with it the ``(1e-5, 0.5)`` frequency band that
resolves a ``feature_mask`` on EVERY default run — while ALSO setting
``feature_prediction``. Since #695 a ``feature_mask`` SIZES the encoder's feature-indexed
embedding tables. The head now uses the same effective vocabulary, and this matrix keeps
every shipped pretraining config on that supported construction path.

Nothing COULD catch it. ``test_encdec_train_vocab.py`` drives the same builder, but with a
synthetic ``OmegaConf.create({...})``; ``test_encdec_memory.py`` composes exactly ONE
config (``conditional_encdec_4090``); ``test_freq_blacklist_integration.py::
test_train_yaml_has_the_fp_head_OFF_by_default`` reads exactly ONE yaml. Not one of them
looks at the config matrix that actually ships. This module enumerates the configs from
DISK, so a config file added later is covered the moment it lands.

The families are partitioned by NAME, and the partition is **fail-closed**: an
unrecognised config MUST build. Both exceptional families are then verified from the
config's own structure, so neither allow-list can rot into a silent skip:

* ``conditional_encdec_sft*`` — consumed by ``scripts/train/conditional_encdec_sft.py``,
  which reads the encoder/cross-attention shape out of the BASE CHECKPOINT's
  metadata. Bucket counts default to checkpoint values; matched recipes may pin
  that same count. They carry no encoder architecture, which is asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig

from scripts.train.conditional_encdec_train import (
    build_encdec_meta_model_from_training_cfg,
)
from steering.conditional import freq_blacklist as fb
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    resolve_feature_mask,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_DIR = REPO_ROOT / "configs" / "training"
PRECOMPUTE_BLACKLIST_YAML = (
    REPO_ROOT / "configs" / "extraction" / "precompute_feature_blacklist.yaml"
)
NANOCHAT_VOCAB_SIZE = 32768

WONTFIX_SWEEP_CONFIGS: frozenset[str] = frozenset()
SFT_CONFIG_PREFIX = "conditional_encdec_sft"
SELFCOND_CONFIG_PREFIX = "conditional_encdec_selfcond"
MATCHED_HEAD_SFT_CONFIGS = frozenset({
    "conditional_encdec_sft_3b_residual_head",
    "conditional_encdec_sft_3b_transformer_head",
})

# The band-derived blacklist can only be computed from a real
# ``feature_statistics.safetensors`` under ``cfg.feature_dir`` (``null`` in every shipped
# config, and a multi-TB directory in practice), so it is NOT required on disk here. Its
# effect on model construction is fully characterised without it: the band yields a
# non-empty blacklist, i.e. a NON-IDENTITY ``FeatureVocab`` — and the builder uses only
# the vocabulary identity and ``vocab.size`` to determine the feature-indexed table and
# FP-head heights. Any non-identity blacklist reproduces that exactly. The real ones,
# measured on the real corpus: 3,701 of 65,536 under the shipped ``(1e-5, 0.5)`` band.
SYNTHETIC_BAND_BLACKLIST_IDS = (0, 1, 2)
ISSUE768_PRODUCTION_CONFIG = (
    "cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000"
)
ISSUE768_PRODUCTION_RUN = "splice_cf_gemma3pt_8xpro6000_cluster_fp"
ISSUE768_EFFECTIVE_V = 2_074


def _config_names() -> list[str]:
    names = sorted(p.stem for p in TRAINING_CONFIG_DIR.glob("conditional_encdec_*.yaml"))
    if not names:
        raise AssertionError(
            f"no conditional_encdec_*.yaml under {TRAINING_CONFIG_DIR} — the glob that "
            "makes this suite self-extending matched NOTHING, which would silently pass "
            "every test below."
        )
    production_cluster_fp = TRAINING_CONFIG_DIR / f"{ISSUE768_PRODUCTION_CONFIG}.yaml"
    if not production_cluster_fp.is_file():
        raise AssertionError(
            "the canonical issue-768 clustered-FP config is missing: "
            f"{production_cluster_fp}"
        )
    return sorted([*names, ISSUE768_PRODUCTION_CONFIG])


ALL_CONFIG_NAMES = _config_names()
WONTFIX_CONFIGS = sorted(n for n in ALL_CONFIG_NAMES if n in WONTFIX_SWEEP_CONFIGS)
SFT_CONFIGS = sorted(
    n
    for n in ALL_CONFIG_NAMES
    if n.startswith(SFT_CONFIG_PREFIX) and n not in WONTFIX_SWEEP_CONFIGS
)
SELFCOND_CONFIGS = sorted(
    n
    for n in ALL_CONFIG_NAMES
    if n.startswith(SELFCOND_CONFIG_PREFIX) and n not in WONTFIX_SWEEP_CONFIGS
)
# Fail-closed default: anything not explicitly excused MUST build.
PRETRAIN_CONFIGS = sorted(
    set(ALL_CONFIG_NAMES) - set(WONTFIX_CONFIGS) - set(SFT_CONFIGS) - set(SELFCOND_CONFIGS)
)


def _compose(hydra_compose, config_name: str) -> DictConfig:
    return hydra_compose(str(TRAINING_CONFIG_DIR), config_name, None)


@pytest.fixture
def matched_head_feature_artifacts(config_name, tmp_path, monkeypatch):
    """Resolve each matched recipe's explicit artifacts without production paths."""
    if config_name not in MATCHED_HEAD_SFT_CONFIGS:
        return None
    mask = tmp_path / "whitelist.txt"
    mask.write_text("1\n4\n9\n12\n20\n")
    clusters = tmp_path / "clusters.yaml"
    clusters.write_text(yaml.safe_dump({
        "schema_version": 1,
        "clusters": [[0, 1, 4], [9, 12]],
    }))
    monkeypatch.setenv("RESIDUAL_HEAD_MASK", str(mask))
    monkeypatch.setenv("RESIDUAL_HEAD_CLUSTERS", str(clusters))
    # Masked-out cluster member 0 cannot become a representative; unclustered
    # allowed member 20 survives. These IDs independently pin mask precedence
    # and clustering, rather than substituting a vocabulary for missing files.
    return (1, 9, 20)


def _n_sae_features(cfg: DictConfig) -> int:
    """The SAE dictionary width the config's ``sae.width`` names.

    Synthetic in the sense that no feature directory is read — but it is the SAME number
    ``build_shared_state`` resolves for that width, so the encoder tables are the real
    production shape (16,384 / 65,536 rows). Meta-device construction makes that free.
    """
    return SAE_WIDTH_TO_NUM_FEATURES[SAEWidth(str(cfg.sae.width))]


def _production_feature_vocab(cfg: DictConfig, n_sae_features: int) -> FeatureVocab | None:
    """The ``feature_vocab`` ``build_shared_state`` would hand the builder for this config.

    Mirrors ``conditional_encdec_train.build_shared_state`` exactly, including precedence:

        maybe_apply_frequency_blacklist(cfg, feature_dir, n_sae)  # may set cfg.feature_mask
        mask  = resolve_feature_mask(cfg, n_sae)
        vocab = FeatureVocab.from_mask(mask, n_sae)

    An explicit ``feature_mask`` wins over the band (the producer skips with a warning),
    so it is resolved first and used verbatim. Otherwise the band decides, via the
    production predicate ``resolve_blacklist_bounds`` — which additionally VALIDATES every
    shipped band (it raises on an inverted or out-of-range pair) — and the resulting mask
    is stood in for (see ``SYNTHETIC_BAND_BLACKLIST_IDS``).
    """
    feature_mask_cfg = cfg.get("feature_mask")
    feature_clusters_cfg = cfg.get("feature_clusters")
    missing_external_path = any(
        block is not None
        and block.get("path") is not None
        and not Path(str(block.path)).is_file()
        for block in (feature_mask_cfg, feature_clusters_cfg)
    )
    if missing_external_path:
        if str(cfg.run) != ISSUE768_PRODUCTION_RUN:
            raise AssertionError(
                "a shipped config references an unavailable external mask/cluster "
                f"artifact without a pinned self-contained shape oracle: run={cfg.run!r}"
            )
        # The ordinary matrix remains self-contained in CI/worktrees. The opt-in
        # real-artifact gate independently resolves the exact original IDs and
        # fingerprint; this stand-in exercises the exact production table height.
        return FeatureVocab(
            n_sae_features=n_sae_features,
            original_ids=tuple(range(ISSUE768_EFFECTIVE_V)),
        )

    feature_mask = resolve_feature_mask(cfg, n_sae_features)
    if feature_mask is None and fb.resolve_blacklist_bounds(cfg) is not None:
        feature_mask = FeatureMask(
            mode=FeatureMaskMode.blacklist,
            ids=SYNTHETIC_BAND_BLACKLIST_IDS,
        )
    resolved_clusters = resolve_feature_clusters(cfg, n_sae_features)
    if resolved_clusters is not None:
        return effective_feature_vocab(
            FeatureClusterConsolidation.build(resolved_clusters, feature_mask),
            n_sae_features,
        )
    return effective_feature_vocab(feature_mask, n_sae_features)


def _build(cfg: DictConfig, vocab: FeatureVocab | None, n_sae_features: int):
    return build_encdec_meta_model_from_training_cfg(
        cfg,
        base_vocab=NANOCHAT_VOCAB_SIZE,
        n_sae_features=n_sae_features,
        feature_vocab=vocab,
    )


class TestEveryPretrainingConfigBuilds:
    """The regression suite for review finding 4."""

    @pytest.mark.parametrize("config_name", PRETRAIN_CONFIGS)
    def test_it_builds_the_production_model(self, hydra_compose, config_name: str) -> None:
        cfg = _compose(hydra_compose, config_name)
        n_sae = _n_sae_features(cfg)
        vocab = _production_feature_vocab(cfg, n_sae)

        model = _build(cfg, vocab, n_sae)

        # n_sae_features stays the ORIGINAL dictionary width; only the tables shrink.
        assert model.encoder.config.n_sae_features == n_sae
        expected_rows = n_sae if vocab is None else vocab.size
        assert model.encoder.feature_embed.weight.shape[0] == expected_rows
        for key, value_embed in model.encoder.value_embeds.items():
            assert value_embed.weight.shape[0] == expected_rows, (
                f"{config_name}: value_embeds[{key}] was not compacted with feature_embed"
            )
        if cfg.get("feature_prediction") is not None:
            assert model.feature_pred_head is not None
            assert model.feature_pred_head.out_features == expected_rows

    @pytest.mark.parametrize("config_name", PRETRAIN_CONFIGS)
    def test_the_name_based_routing_agrees_with_the_config_structure(
        self, hydra_compose, config_name: str
    ) -> None:
        """A config routed here really is a pretraining config: it carries the encoder
        architecture ``build_encdec_meta_model_from_training_cfg`` reads."""
        cfg = _compose(hydra_compose, config_name)
        assert "encoder" in cfg
        assert "cross_attention" in cfg
        assert cfg.get("num_buckets") is not None
        assert "sae" in cfg


class TestTheMatrixIsNotVacuous:
    """A green matrix proves nothing if no config exercises the interesting paths."""

    def test_the_shipped_band_really_does_compact_the_encoder(self, hydra_compose) -> None:
        """If the band ever stops resolving a mask, every build above goes green while
        testing the pre-#695 code path only."""
        cfg = _compose(hydra_compose, "conditional_encdec_train")
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=fb.DEFAULT_FREQ_MIN, max=fb.DEFAULT_FREQ_MAX
        )
        vocab = _production_feature_vocab(cfg, _n_sae_features(cfg))
        assert vocab is not None
        assert vocab.size < _n_sae_features(cfg)

    def test_both_branches_of_the_compaction_rule_are_covered(self, hydra_compose) -> None:
        compacted: list[str] = []
        fp_headed: list[str] = []
        for config_name in PRETRAIN_CONFIGS:
            cfg = _compose(hydra_compose, config_name)
            if _production_feature_vocab(cfg, _n_sae_features(cfg)) is not None:
                compacted.append(config_name)
            if cfg.get("feature_prediction") is not None:
                fp_headed.append(config_name)

        assert compacted, (
            "no shipped pretraining config resolves a mask — the COMPACTED build path is "
            "untested by this module."
        )
        assert fp_headed, (
            "no shipped pretraining config carries the FP head — nothing exercises "
            "the compact prediction table."
        )


class TestTheSftFamilyIsNotBuiltHere:
    """Checked, not merely asserted in prose."""

    @pytest.mark.parametrize("config_name", SFT_CONFIGS)
    def test_it_carries_no_encoder_architecture(self, hydra_compose, config_name: str) -> None:
        """SFT configs are consumed by ``scripts/train/conditional_encdec_sft.py``, which
        reads encoder / cross-attention geometry out of the BASE CHECKPOINT's
        metadata. A bucket override may reproduce the checkpoint's encoding,
        but these configs cannot go through the pretraining builder.

        If someone ever gives an SFT config an ``encoder:`` block, this fails and forces
        the routing question, instead of letting a pretrain-shaped config skip the build
        matrix because of its name.
        """
        cfg = _compose(hydra_compose, config_name)
        assert "encoder" not in cfg
        assert "cross_attention" not in cfg
        if config_name in MATCHED_HEAD_SFT_CONFIGS:
            pretrained = _compose(hydra_compose, "full3b_clustered_c2048")
            assert cfg.num_buckets == pretrained.num_buckets == 2
        else:
            assert cfg.get("num_buckets") is None

    @pytest.mark.parametrize("config_name", SFT_CONFIGS)
    def test_its_frequency_band_still_resolves_a_valid_mask(
        self, hydra_compose, config_name: str, matched_head_feature_artifacts,
    ) -> None:
        """SFT resolves a mask too (``conditional_encdec_sft.py`` → ``FeatureVocab
        .from_mask``); an explicit whitelist/cluster policy takes precedence over
        the default frequency band and must resolve through the same readers."""
        cfg = _compose(hydra_compose, config_name)
        n_sae = _n_sae_features(cfg)
        vocab = _production_feature_vocab(cfg, n_sae)
        assert vocab is not None
        assert vocab.size < n_sae
        if matched_head_feature_artifacts is not None:
            assert vocab.original_ids == matched_head_feature_artifacts


class TestTheFamilyPartition:
    def test_every_shipped_config_lands_in_exactly_one_family(self) -> None:
        routed = PRETRAIN_CONFIGS + SFT_CONFIGS + SELFCOND_CONFIGS + WONTFIX_CONFIGS
        assert len(routed) == len(set(routed)), "a config was routed to two families"
        assert sorted(routed) == ALL_CONFIG_NAMES, (
            "the family partition does not cover every conditional_encdec_*.yaml on disk"
        )

    def test_a_new_config_defaults_to_MUST_BUILD(self) -> None:
        """The fail-closed default, pinned. A config file added tomorrow is built by
        ``TestEveryPretrainingConfigBuilds`` unless someone explicitly excuses it."""
        assert "conditional_encdec_train" in PRETRAIN_CONFIGS
        assert ISSUE768_PRODUCTION_CONFIG in PRETRAIN_CONFIGS
        assert not set(PRETRAIN_CONFIGS) & WONTFIX_SWEEP_CONFIGS
        assert not any(n.startswith(SFT_CONFIG_PREFIX) for n in PRETRAIN_CONFIGS)
        assert not any(n.startswith(SELFCOND_CONFIG_PREFIX) for n in PRETRAIN_CONFIGS)


class TestTheSelfcondFamilyIsCheckpointShaped:
    def test_the_expected_config_is_present(self) -> None:
        assert SELFCOND_CONFIGS == ["conditional_encdec_selfcond"]

    @pytest.mark.parametrize("config_name", SELFCOND_CONFIGS)
    def test_it_composes_with_the_pinned_schema(self, hydra_compose, config_name: str) -> None:
        cfg = _compose(hydra_compose, config_name)
        assert cfg.model_tag is None
        assert "encoder" not in cfg
        assert "cross_attention" not in cfg
        assert cfg.selfcond.train_scope == "head_only"
        assert cfg.selfcond.ste_form == "dense"
        assert cfg.selfcond.positive_counts is None
        assert cfg.selfcond.negative_counts is None


class TestTheDefaultBandIsShippedInLockstep:
    """Review finding 5: a default-band flip is easy to ship NON-lockstep.

    ``freq_blacklist.DEFAULT_FREQ_MIN``/``MAX`` are only a FALLBACK for an ABSENT key, so
    a config that ships the keys explicitly overrides them —
    ``configs/extraction/precompute_feature_blacklist.yaml`` kept shipping the OLD band
    while the docs already documented the new one. Re-running the standalone producer
    with the config's own defaults would then generate a DIFFERENT blacklist from the one
    training generates, i.e. a different encoder shape.

    ``tests/steering/test_freq_blacklist_integration.py::
    test_training_yamls_have_threshold_default`` pins the two TRAINING yamls and is the
    natural home for this too; it is asserted here because this change is scoped not to
    touch that module.
    """

    def test_the_standalone_producer_ships_the_same_default_band(self) -> None:
        data = yaml.safe_load(PRECOMPUTE_BLACKLIST_YAML.read_text())
        assert data["feature_mask_freq_min"] == fb.DEFAULT_FREQ_MIN == 1e-5
        assert data["feature_mask_freq_max"] == fb.DEFAULT_FREQ_MAX == 0.5
