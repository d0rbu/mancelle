"""Every #695 error message promises a recovery route. This proves the route WORKS.

Four separate guards refuse to run and tell the operator, verbatim::

    Recover it with: uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>

That promise is only worth anything if the whole loop closes:

    compacted checkpoint  ->  dump_feature_mask  ->  paste into a config
                          ->  resolve_feature_mask  ->  FeatureVocab.from_mask
                          ->  the SAME vocabulary, and the guard now PASSES.

``tests/scripts/test_dump_feature_mask.py`` tests the tool against HAND-WRITTEN state
dicts, and it stops at ``FeatureVocab`` equality. It never (a) reads a checkpoint written
by the production save path, (b) feeds the recovered mask back into the guards that sent
the operator there, or (c) reproduces the situation the module exists for.

That situation is the ``.txt`` hostage. A default run's mask is a *generated* artifact —
``{feature_dir}/blacklist/freq_min*_max*.txt`` — referenced by PATH. Move the
feature dir, recompute statistics, or clean a scratch disk, and the config that trained
the checkpoint can no longer be resolved at all. Every test below runs with that file
DELETED, which is the only honest way to test a tool whose reason to exist is that the
file is gone.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import torch as t
import yaml
from omegaconf import DictConfig, OmegaConf

from errors import FeatureMaskError, FeatureVocabError
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import GPTConfig
from scripts.analysis.dump_feature_mask import (
    FORMAT_TXT,
    FORMAT_YAML,
    SOURCE_META,
    SOURCE_VOCAB,
    load_dumped_mask,
    render,
)
from scripts.analysis.dump_feature_mask import main as dump_main
from scripts.train.conditional_encdec_sft import (
    _assert_sft_universe_matches_checkpoint_vocab,
    _build_sft_checkpoint_dir,
)
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    append_mask_leaf,
    canonical_dict,
    feature_mask_dirname,
    resolve_feature_mask,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    feature_vocab_from_state_dict,
)
from steering.conditional.freq_blacklist import (
    DEFAULT_FREQ_MAX,
    DEFAULT_FREQ_MIN,
    maybe_apply_frequency_blacklist,
)
from tests.steering._freq_blacklist_fixtures import write_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_DIR = REPO_ROOT / "configs" / "training"

N_SAE = 64
NUM_BUCKETS = 4
STEP = 42

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

# n_segments=100_000 with the SHIPPED (1e-5, 0.5) band (KEEP iff min < freq <= max —
# lower bound EXCLUSIVE-of-keep, upper INCLUSIVE-of-keep):
#   id 0: count 1      -> freq 1e-5   -> KILLED: freq == freq_min, and KEEP needs `>`
#   id 1: count 2      -> freq 2e-5   -> kept
#   id 2: count 40_000 -> freq 0.4    -> kept
#   id 5: count 50_000 -> freq 0.5    -> KEPT: freq == freq_max, and KEEP allows `<=`
#   id 3: count 50_001 -> freq >0.5   -> killed by freq_max
#   id 4: count 99_999 -> freq ~1.0   -> killed by freq_max
#   id 9: count 170    -> freq 0.0017 -> kept
#   absent             -> freq 0      -> killed (dead)
# So of 64 features exactly {1, 2, 5, 9} survive -> a genuine, producer-derived vocab.
#
# n_segments is 100_000, not 10_000, ON PURPOSE: at 10_000 the smallest non-zero frequency
# is 1e-4, a DECADE above the 1e-5 floor, so id 0 would survive and the inclusive-lower
# boundary would go untested. Ids 0 and 5 sit EXACTLY on the two bounds; that is the point.
N_SEGMENTS = 100_000
COUNTS = {0: 1, 1: 2, 2: 40_000, 3: 50_001, 4: 99_999, 5: 50_000, 9: 170}
SURVIVORS = (1, 2, 5, 9)

# The command every guard tells the operator to run.
DUMP_CMD_RE = re.compile(r"uv run python -m\s+scripts\.analysis\.dump_feature_mask\s+(\S+)")


# ---------------------------------------------------------------------------
# A checkpoint built the way conditional_encdec_train actually builds one
# ---------------------------------------------------------------------------


class Trained:
    """A real compacted checkpoint on disk, plus the feature dir that is about to die."""

    def __init__(
        self,
        ckpt_dir: Path,
        feature_dir: Path,
        mask: FeatureMask,
        vocab: FeatureVocab,
        blacklist_txt: Path,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.feature_dir = feature_dir
        self.mask = mask
        self.vocab = vocab
        self.blacklist_txt = blacklist_txt

    def state_dict(self) -> dict:
        model_data, _, _ = load_checkpoint(str(self.ckpt_dir), STEP, t.device("cpu"))
        return model_data


def _model(vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=16,
                vocab_size=128,
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
            num_buckets=NUM_BUCKETS,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    t.manual_seed(11)
    m.init_weights()
    return m


def _train_cfg(hydra_compose, feature_dir: Path) -> DictConfig:
    cfg = hydra_compose(str(TRAINING_CONFIG_DIR), "conditional_encdec_train", None)
    OmegaConf.update(cfg, "feature_mask", None, merge=False, force_add=True)
    OmegaConf.update(cfg, "feature_mask_freq_min", DEFAULT_FREQ_MIN, merge=False, force_add=True)
    OmegaConf.update(cfg, "feature_mask_freq_max", DEFAULT_FREQ_MAX, merge=False, force_add=True)
    return cfg


def _train(hydra_compose, tmp_path: Path, *, record_mask_in_meta: bool = True) -> Trained:
    """Reproduce ``build_shared_state`` + ``train_one_config``'s save, for real.

    band -> generated blacklist .txt -> cfg.feature_mask={mode,path} -> resolve -> vocab
    -> compacted model -> save_checkpoint with the meta ``train_one_config`` writes.
    """
    feature_dir = tmp_path / "features"
    write_stats(feature_dir, n_segments=N_SEGMENTS, counts=COUNTS, n_sae_features=N_SAE)

    cfg = _train_cfg(hydra_compose, feature_dir)
    outcome = maybe_apply_frequency_blacklist(cfg, str(feature_dir), N_SAE)
    assert outcome.status.value == "applied", outcome
    # The mask is a PATH into the feature dir. That is the hostage.
    blacklist_txt = Path(cfg.feature_mask.path)
    assert blacklist_txt.is_file()

    mask = resolve_feature_mask(cfg, N_SAE)
    assert mask is not None
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    assert vocab is not None and vocab.original_ids == SURVIVORS

    model = _model(vocab)
    # Production appends the MASK LEAF (conditional_encdec_train.py, via append_mask_leaf):
    # .../ckpt/bl_<16hex>/. Saving to a bare directory made every test in this file blind to
    # the fact that the leaf name is the ONLY on-disk record of the mask's MODE.
    ckpt_dir = Path(append_mask_leaf(str(tmp_path / "ckpt"), mask))

    # The exact meta shape conditional_encdec_train.py:2912-2937 writes.
    meta: dict = {
        "step": STEP,
        "val_bpb": 1.23,
        "model_config": dict(DEC_KW),
        "user_config": OmegaConf.to_container(cfg, resolve=True),
    }
    if record_mask_in_meta:
        meta["feature_mask_canonical"] = canonical_dict(mask)
    model_vocab = model.feature_vocab
    assert model_vocab is not None
    meta["feature_vocab"] = model_vocab.metadata_dict()

    save_checkpoint(str(ckpt_dir), STEP, model.state_dict(), None, meta, rank=0)
    return Trained(ckpt_dir, feature_dir, mask, vocab, blacklist_txt)


@pytest.fixture()
def orphaned(hydra_compose, tmp_path) -> Trained:
    """A compacted checkpoint whose feature dir — and thus its mask .txt — is GONE."""
    trained = _train(hydra_compose, tmp_path)
    shutil.rmtree(trained.feature_dir)
    assert not trained.blacklist_txt.exists()
    return trained


@pytest.fixture()
def orphaned_no_meta_mask(hydra_compose, tmp_path) -> Trained:
    """Same, but the meta records no ``feature_mask_canonical`` — so the mask has to be
    RECONSTRUCTED from the weights' vocab buffer alone."""
    trained = _train(hydra_compose, tmp_path, record_mask_in_meta=False)
    shutil.rmtree(trained.feature_dir)
    return trained


def _paste(yaml_text: str) -> dict:
    """What an operator does: paste the tool's stdout into a config file."""
    block = yaml.safe_load(yaml_text)
    assert set(block) == {"feature_mask"}, f"not a pasteable config block: {block!r}"
    return block


# ===========================================================================
# 0. The hostage situation is REAL
# ===========================================================================


class TestTheProblemIsReal:
    """If the original config could still be resolved, this whole module is theatre."""

    def test_the_trained_mask_is_a_PATH_into_the_feature_dir(self, hydra_compose, tmp_path):
        trained = _train(hydra_compose, tmp_path)

        assert trained.blacklist_txt.is_relative_to(trained.feature_dir)

    def test_once_the_feature_dir_is_gone_the_ORIGINAL_config_cannot_be_resolved(self, orphaned):
        """The run's own ``feature_mask: {mode: blacklist, path: ...}`` is now unusable —
        which is precisely why the vocabulary has to be recoverable from the CHECKPOINT."""
        original_cfg = {"feature_mask": {"mode": "blacklist", "path": str(orphaned.blacklist_txt)}}

        with pytest.raises(
            (FeatureMaskError, FileNotFoundError, OSError),
            match="not found|No such file",
        ):
            resolve_feature_mask(original_cfg, N_SAE)

    def test_the_checkpoint_still_knows_its_own_vocabulary(self, orphaned):
        n_sae, vocab = feature_vocab_from_state_dict(orphaned.state_dict())

        assert n_sae == N_SAE
        assert vocab == orphaned.vocab


# ===========================================================================
# 1. The loop closes — dump -> paste -> resolve -> the SAME vocab
# ===========================================================================


class TestTheRecoveryLoopCloses:
    def test_the_dumped_yaml_reproduces_the_vocab_FINGERPRINT(self, orphaned, capsys):
        dump_main([str(orphaned.ckpt_dir)])
        pasted = _paste(capsys.readouterr().out)

        mask = resolve_feature_mask(pasted, N_SAE)
        vocab = FeatureVocab.from_mask(mask, N_SAE)

        assert vocab is not None
        assert vocab == orphaned.vocab
        assert vocab.fingerprint() == orphaned.vocab.fingerprint()
        # And it is the fingerprint the checkpoint's own metadata records.
        assert vocab.metadata_dict() == orphaned.vocab.metadata_dict()

    def test_the_recovered_mask_is_the_TRAINING_mask_not_its_inverse(self, orphaned, capsys):
        """A blacklist run must come back as the BLACKLIST. Emitting the allowed universe
        under ``mode: blacklist`` would invert the mask and mask out exactly the features
        the run was trained on — while still 'round-tripping' if you only compared sizes.
        """
        dump_main([str(orphaned.ckpt_dir)])
        block = _paste(capsys.readouterr().out)["feature_mask"]

        assert block["mode"] == "blacklist"
        assert block["ids"] == list(orphaned.mask.ids)
        assert set(block["ids"]).isdisjoint(SURVIVORS), (
            "the blacklist must list the KILLED features, never the survivors"
        )

    def test_the_dumped_txt_reproduces_the_vocab_through_feature_mask_path(
        self, orphaned, tmp_path
    ):
        """The other documented route: save the ids, reference them with ``path:``."""
        dumped = load_dumped_mask(orphaned.ckpt_dir)
        ids_txt = tmp_path / "recovered_ids.txt"
        ids_txt.write_text(render(dumped, FORMAT_TXT), encoding="utf-8")

        mask = resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "path": str(ids_txt)}}, N_SAE
        )

        assert FeatureVocab.from_mask(mask, N_SAE) == orphaned.vocab

    def test_the_source_is_the_metadata_when_it_has_one(self, orphaned):
        assert load_dumped_mask(orphaned.ckpt_dir).source == SOURCE_META

    def test_it_works_with_NO_recorded_mask_by_reconstructing_from_the_WEIGHTS(
        self, orphaned_no_meta_mask, capsys
    ):
        """Belt and braces: even a checkpoint whose meta lost the mask is recoverable,
        because the load-bearing record is the ``encoder.feature_vocab_ids`` buffer."""
        dumped = load_dumped_mask(orphaned_no_meta_mask.ckpt_dir)
        assert dumped.source == SOURCE_VOCAB

        dump_main([str(orphaned_no_meta_mask.ckpt_dir)])
        block = _paste(capsys.readouterr().out)["feature_mask"]

        # The ORIGINAL mask, mode included — recovered from the checkpoint's leaf name.
        # Emitting a whitelist of the same universe would have been a DIFFERENT mode, hence
        # a different bl_/wl_ leaf hash, hence a config addressing a checkpoint that does
        # not exist. Same universe is NOT sufficient; the leaf has to match.
        assert block["mode"] == "blacklist" == orphaned_no_meta_mask.mask.mode.value
        assert tuple(block["ids"]) == orphaned_no_meta_mask.mask.ids
        mask = resolve_feature_mask({"feature_mask": block}, N_SAE)
        assert FeatureVocab.from_mask(mask, N_SAE) == orphaned_no_meta_mask.vocab
        assert feature_mask_dirname(mask) == orphaned_no_meta_mask.ckpt_dir.name


# ===========================================================================
# 2. The recovered mask actually SATISFIES the guards that sent us here
# ===========================================================================


def _recovered_mask(ckpt_dir: Path, capsys) -> FeatureMask:
    dump_main([str(ckpt_dir)])
    mask = resolve_feature_mask(_paste(capsys.readouterr().out), N_SAE)
    assert mask is not None
    return mask


class TestTheRecoveredMaskSatisfiesTheGuards:
    """Closing the loop where it matters: the operator pastes, re-runs, and it WORKS."""

    def test_the_checkpoint_compat_guard_REFUSES_without_a_mask(self, orphaned):
        """The failure the operator actually hits: a compacted checkpoint, no mask."""
        with pytest.raises(FeatureVocabError, match="COMPACTED") as exc:
            assert_checkpoint_vocab_compatible(
                None, orphaned.state_dict(), checkpoint_dir=str(orphaned.ckpt_dir)
            )

        assert "COMPACTED" in str(exc.value)

    def test_and_ACCEPTS_the_recovered_mask(self, orphaned, capsys):
        mask = _recovered_mask(orphaned.ckpt_dir, capsys)
        vocab = FeatureVocab.from_mask(mask, N_SAE)

        assert_checkpoint_vocab_compatible(
            vocab, orphaned.state_dict(), checkpoint_dir=str(orphaned.ckpt_dir)
        )

    def test_the_recovered_mask_lets_the_model_LOAD_the_checkpoint_strictly(self, orphaned, capsys):
        """The end of the road: a model built from the recovered mask takes the weights."""
        mask = _recovered_mask(orphaned.ckpt_dir, capsys)
        vocab = FeatureVocab.from_mask(mask, N_SAE)

        model = _model(vocab)
        model.load_state_dict(orphaned.state_dict(), strict=True, assign=True)
        model.assert_vocab_consistent()

        assert model.encoder.feature_embed.weight.shape[0] == len(SURVIVORS)

    def test_the_SFT_guard_REFUSES_without_a_mask(self, orphaned):
        with pytest.raises(FeatureVocabError, match="never silently adopted"):
            _assert_sft_universe_matches_checkpoint_vocab(
                None,
                None,
                orphaned.vocab,
                N_SAE,
                checkpoint_dir=str(orphaned.ckpt_dir),
                base_meta={},
            )

    def test_and_the_SFT_guard_ACCEPTS_the_recovered_mask(self, orphaned, capsys):
        mask = _recovered_mask(orphaned.ckpt_dir, capsys)

        _assert_sft_universe_matches_checkpoint_vocab(
            mask,
            mask,
            orphaned.vocab,
            N_SAE,
            checkpoint_dir=str(orphaned.ckpt_dir),
            base_meta={},
        )

    def test_the_RECONSTRUCTED_mask_also_satisfies_the_SFT_guard(
        self, orphaned_no_meta_mask, capsys
    ):
        """The guard compares allowed UNIVERSES, so it was always mode-blind — which is
        exactly why it could not catch the mode being wrong. The RECONSTRUCTED mask must
        satisfy it AND carry the original mode."""
        mask = _recovered_mask(orphaned_no_meta_mask.ckpt_dir, capsys)
        assert mask.mode is FeatureMaskMode.blacklist

        _assert_sft_universe_matches_checkpoint_vocab(
            mask,
            mask,
            orphaned_no_meta_mask.vocab,
            N_SAE,
            checkpoint_dir=str(orphaned_no_meta_mask.ckpt_dir),
            base_meta={},
        )


# ===========================================================================
# 3. The error message is a runnable instruction, not a gesture
# ===========================================================================


# The remedy each guard prints is a RUNNABLE command, not a gesture: these tests parse it
# out of the guard's own error text and execute THAT. A literal '<checkpoint_dir>' used to
# ship in feature_vocab.py:346 (a plain string in a raise whose previous line interpolated
# the real path); it is fixed, and these tests are what keep it fixed.


class TestTheAdvertisedCommandIsRunnable:
    """Take the guard at its word: parse the command out of its own error text, run THAT,
    and check the output fixes THAT error. If the message ever names the wrong directory,
    the wrong module, or a directory that cannot be dumped, this goes red."""

    @pytest.fixture()
    def compat_error(self, orphaned) -> str:
        with pytest.raises(FeatureVocabError, match="COMPACTED") as exc:
            assert_checkpoint_vocab_compatible(
                None, orphaned.state_dict(), checkpoint_dir=str(orphaned.ckpt_dir)
            )
        return str(exc.value)

    @pytest.fixture()
    def sft_error(self, orphaned) -> str:
        with pytest.raises(FeatureVocabError, match="dump_feature_mask") as exc:
            _assert_sft_universe_matches_checkpoint_vocab(
                None,
                None,
                orphaned.vocab,
                N_SAE,
                checkpoint_dir=str(orphaned.ckpt_dir),
                base_meta={},
            )
        return str(exc.value)

    @pytest.mark.parametrize("which", ["compat_error", "sft_error"])
    def test_the_message_states_the_real_checkpoint_dir_SOMEWHERE(self, which, request, orphaned):
        """Scopes the defect above: the operator is TRAPPED, not STRANDED.

        Both guards do name the real directory in prose. Only the copy-pasteable command
        is broken (in one of them). This test passes for both and must keep passing — if
        the path ever vanished from the message entirely, the defect would become fatal.
        """
        assert str(orphaned.ckpt_dir) in request.getfixturevalue(which)

    @pytest.mark.parametrize(
        "which",
        ["compat_error", "sft_error"],
    )
    def test_the_message_names_the_dump_command_and_the_right_directory(
        self, which, request, orphaned
    ):
        msg = request.getfixturevalue(which)

        found = DUMP_CMD_RE.search(msg)
        assert found, f"the error names no recovery command: {msg!r}"
        named = found.group(1).rstrip("`.'\"")
        assert Path(named) == orphaned.ckpt_dir, (
            f"the error points the operator at {named!r}, not at the checkpoint that "
            f"actually failed ({orphaned.ckpt_dir})"
        )

    @pytest.mark.parametrize(
        "which",
        ["compat_error", "sft_error"],
    )
    def test_running_the_command_the_message_names_FIXES_the_error(
        self, which, request, orphaned, capsys
    ):
        msg = request.getfixturevalue(which)
        match = DUMP_CMD_RE.search(msg)
        assert match is not None
        named = match.group(1).rstrip("`.'\"")

        # Run exactly what the operator was told to run.
        assert dump_main([named]) == 0
        mask = resolve_feature_mask(_paste(capsys.readouterr().out), N_SAE)
        vocab = FeatureVocab.from_mask(mask, N_SAE)

        # Both guards that emit this message now pass.
        assert_checkpoint_vocab_compatible(
            vocab, orphaned.state_dict(), checkpoint_dir=str(orphaned.ckpt_dir)
        )
        _assert_sft_universe_matches_checkpoint_vocab(
            mask,
            mask,
            orphaned.vocab,
            N_SAE,
            checkpoint_dir=str(orphaned.ckpt_dir),
            base_meta={},
        )


# ===========================================================================
# 4. Strictness the loop must NOT relax
# ===========================================================================


class TestStrictnessIsNotRelaxedByRecovery:
    def test_a_SUBSET_of_the_recovered_mask_is_still_refused(self, orphaned, capsys):
        """Recovering the mask must not make the guard lenient. Drop ONE blacklisted id
        (widening the allowed universe by one feature) and it must still be refused —
        there is no embedding row for that feature."""
        recovered = _recovered_mask(orphaned.ckpt_dir, capsys)
        widened = FeatureMask(mode=recovered.mode, ids=recovered.ids[1:])
        assert FeatureVocab.from_mask(widened, N_SAE) != orphaned.vocab

        with pytest.raises(FeatureVocabError, match="allowed universe must EQUAL"):
            _assert_sft_universe_matches_checkpoint_vocab(
                widened,
                widened,
                orphaned.vocab,
                N_SAE,
                checkpoint_dir=str(orphaned.ckpt_dir),
                base_meta={},
            )
        with pytest.raises(FeatureVocabError, match="(?i)mismatch|does not match"):
            assert_checkpoint_vocab_compatible(
                FeatureVocab.from_mask(widened, N_SAE),
                orphaned.state_dict(),
                checkpoint_dir=str(orphaned.ckpt_dir),
            )

    def test_the_SAME_ids_against_a_DIFFERENT_n_sae_is_a_DIFFERENT_vocab(self, orphaned):
        """``n_sae_features`` is part of the vocabulary's identity, and the ``bl_``/``wl_``
        checkpoint leaf does NOT cover it (it hashes only mode + ids). Two runs against
        SAE dictionaries of different widths therefore compose the SAME directory — so the
        width has to be compared explicitly, or a 16k checkpoint would silently load into
        a 65k run.

        NOTE: unreachable through ``conditional_encdec_sft.main()``, which takes BOTH the
        width and the vocab from the same ``feature_vocab_from_state_dict`` call. It IS
        reachable in training, where the width comes from ``feature_statistics``.
        """
        wider = FeatureVocab(n_sae_features=N_SAE * 2, original_ids=SURVIVORS)
        assert wider.original_ids == orphaned.vocab.original_ids
        assert wider != orphaned.vocab
        assert wider.fingerprint() != orphaned.vocab.fingerprint()

        with pytest.raises(FeatureVocabError, match="n_sae_features mismatch"):
            assert_checkpoint_vocab_compatible(
                wider, orphaned.state_dict(), checkpoint_dir=str(orphaned.ckpt_dir)
            )

    def test_the_dumped_ids_are_ORIGINAL_ids_not_compact_ones(self, orphaned, capsys):
        """The subtlest possible corruption: emitting COMPACT ids (0..V-1) would produce a
        mask that is perfectly well-formed, resolves without error, and is completely
        wrong. With V=4 and survivors {1,2,5,9}, compact ids would be {0,1,2,3}."""
        dump_main([str(orphaned.ckpt_dir)])
        block = _paste(capsys.readouterr().out)["feature_mask"]

        mask = resolve_feature_mask({"feature_mask": block}, N_SAE)
        vocab = FeatureVocab.from_mask(mask, N_SAE)
        assert vocab is not None
        allowed = set(vocab.original_ids)
        assert allowed == set(SURVIVORS)
        assert allowed != set(range(len(SURVIVORS))), "compact ids leaked into the dump"


# ===========================================================================
# 5. Determinism — the artifact an operator commits must not drift
# ===========================================================================


class TestTheDumpIsStable:
    def test_two_dumps_of_the_same_checkpoint_are_byte_identical(self, orphaned, capsys):
        dump_main([str(orphaned.ckpt_dir)])
        first = capsys.readouterr().out
        dump_main([str(orphaned.ckpt_dir)])
        second = capsys.readouterr().out

        assert first == second

    def test_the_yaml_and_txt_routes_agree_on_the_universe(self, orphaned, tmp_path):
        dumped = load_dumped_mask(orphaned.ckpt_dir)
        assert dumped.mask is not None

        from_yaml = resolve_feature_mask(_paste(render(dumped, FORMAT_YAML)), N_SAE)
        ids_txt = tmp_path / "agree.txt"
        ids_txt.write_text(render(dumped, FORMAT_TXT), encoding="utf-8")
        from_txt = resolve_feature_mask(
            {"feature_mask": {"mode": dumped.mask.mode.value, "path": str(ids_txt)}},
            N_SAE,
        )

        assert FeatureVocab.from_mask(from_yaml, N_SAE) == dumped.vocab
        assert FeatureVocab.from_mask(from_txt, N_SAE) == dumped.vocab


# ===========================================================================
# 5. The recovered mask must address the SAME checkpoint leaf it came from
# ===========================================================================


class TestTheRecoveredMaskAddressesTheSAMECheckpoint:
    """The entire point of the tool: paste its output into a config and you can resume or
    evaluate THE CHECKPOINT YOU RECOVERED IT FROM.

    Reconstructing from the vocab alone recovers the allowed UNIVERSE but not the MODE, and
    ``FeatureMask.dirname()`` hashes ``(mode, ids)``. A whitelist of the vocab and a
    blacklist of its complement describe the identical universe yet produce DIFFERENT leaves
    (``wl_<hex>`` vs ``bl_<hex>``) and a different ``compute_selection_hash``. The default
    frequency band always produces a BLACKLIST, so emitting a whitelist sent the operator to
    a directory that holds no checkpoint — the recovery loop did not close for the exact case
    the tool exists to serve.

    The mode is not in the weights. It IS on disk: production stores the checkpoint under
    ``append_mask_leaf(...)``, so the leaf directory's own NAME carries it.
    """

    def test_the_recovered_mask_reproduces_the_ORIGINAL_leaf(self, orphaned_no_meta_mask):
        trained = orphaned_no_meta_mask
        recovered = load_dumped_mask(trained.ckpt_dir).mask

        assert recovered is not None
        assert feature_mask_dirname(recovered) == trained.ckpt_dir.name, (
            "the recovered mask hashes to a DIFFERENT leaf than the checkpoint it came "
            f"from: {feature_mask_dirname(recovered)} != {trained.ckpt_dir.name}. Pasting "
            "this into a config sends resume/eval to a directory with no checkpoint in it."
        )

    def test_the_recovered_mask_has_the_ORIGINAL_mode_and_ids(self, orphaned_no_meta_mask):
        trained = orphaned_no_meta_mask
        recovered = load_dumped_mask(trained.ckpt_dir).mask
        assert recovered is not None
        assert recovered.mode == trained.mask.mode == FeatureMaskMode.blacklist
        assert recovered.ids == trained.mask.ids

    def test_recovering_the_MODE_does_not_cost_us_the_UNIVERSE(self, orphaned_no_meta_mask):
        trained = orphaned_no_meta_mask
        recovered = load_dumped_mask(trained.ckpt_dir).mask
        assert recovered is not None
        vocab = FeatureVocab.from_mask(recovered, N_SAE)
        assert vocab is not None
        assert vocab.original_ids == SURVIVORS

    def test_a_checkpoint_NOT_under_a_mask_leaf_FAILS_LOUD_instead_of_guessing(
        self, hydra_compose, tmp_path
    ):
        """Without the meta AND without the leaf, the mode is genuinely unknowable. Emitting
        a plausible guess would hand the operator a config addressing a different
        checkpoint — silently. Refuse, and say which two leaves it could be."""
        trained = _train(hydra_compose, tmp_path, record_mask_in_meta=False)
        flat = tmp_path / "no_leaf"
        shutil.copytree(trained.ckpt_dir, flat)

        with pytest.raises(FeatureMaskError, match="(?i)mode"):
            load_dumped_mask(flat)


# ===========================================================================
# 6. An SFT checkpoint — where the mask leaf is the PARENT, not the basename
# ===========================================================================

SFT_DATASET = "smoltalk"


@pytest.fixture()
def orphaned_sft_no_meta_mask(hydra_compose, tmp_path) -> Trained:
    """An SFT checkpoint written where ``conditional_encdec_sft.py`` actually writes one.

    Production composes the path with the PRODUCTION helper used below::

        _build_sft_checkpoint_dir(append_mask_leaf(pretrain_dir, mask), cfg.dataset)
            -> .../{fs}/bl_<16hex>/{dataset}_sft_conditioned_checkpoints/

    so the ``bl_``/``wl_`` leaf is the PARENT of the directory that holds ``model_*.pt`` —
    never its basename. Every other fixture in this file saves AT the mask leaf (that is what
    the pretrain path does when no gradient leaf is configured), which left the whole module
    blind to the nesting: a tool matching only ``checkpoint_dir.name`` dead-ended with
    "cannot determine the MODE" on EVERY SFT checkpoint, i.e. on exactly the artifacts
    ``_assert_sft_universe_matches_checkpoint_vocab`` tells operators to run it against.

    SFT inherits the base checkpoint's dictionary (it can neither widen nor shrink it), so
    the saved encoder carries the same compacted vocab — and, with the meta thinned, the leaf
    NAME is once again the only surviving record of the mode.
    """
    trained = _train(hydra_compose, tmp_path, record_mask_in_meta=False)
    sft_dir = Path(_build_sft_checkpoint_dir(str(trained.ckpt_dir), SFT_DATASET))

    model = _model(trained.vocab)
    model_vocab = model.feature_vocab
    assert model_vocab is not None
    meta: dict = {
        "step": STEP,
        "val_bpb": 1.01,
        "model_config": dict(DEC_KW),
        "feature_vocab": model_vocab.metadata_dict(),
    }
    save_checkpoint(str(sft_dir), STEP, model.state_dict(), None, meta, rank=0)

    shutil.rmtree(trained.feature_dir)
    return Trained(sft_dir, trained.feature_dir, trained.mask, trained.vocab, trained.blacklist_txt)


class TestAnSFTCheckpointRecoversItsMask:
    """The SFT checkpoint is the one operators actually hold, and its mask leaf is upstairs."""

    def test_the_fixture_really_is_nested_BELOW_the_mask_leaf(self, orphaned_sft_no_meta_mask):
        """Guard the guard: if this ever flattens, every test below silently stops testing
        the nesting it exists to test."""
        sft = orphaned_sft_no_meta_mask.ckpt_dir

        assert sft.name == f"{SFT_DATASET}_sft_conditioned_checkpoints"
        assert (sft / f"model_{STEP:06d}.pt").is_file()
        assert feature_mask_dirname(orphaned_sft_no_meta_mask.mask) == sft.parent.name

    def test_the_SFT_checkpoint_recovers_its_BLACKLIST(self, orphaned_sft_no_meta_mask):
        """This raised "cannot determine the MODE" — on the tool's headline use case."""
        recovered = load_dumped_mask(orphaned_sft_no_meta_mask.ckpt_dir)

        assert recovered.source == SOURCE_VOCAB
        assert recovered.mask is not None
        assert recovered.mask.mode == FeatureMaskMode.blacklist
        assert recovered.mask == orphaned_sft_no_meta_mask.mask

    def test_the_recovered_SFT_mask_ADDRESSES_the_leaf_the_checkpoint_lives_under(
        self, orphaned_sft_no_meta_mask, capsys
    ):
        """The whole loop, on the SFT layout: dump -> paste -> resolve -> the same vocab,
        and the pasted mask hashes to the leaf this checkpoint is actually stored under."""
        trained = orphaned_sft_no_meta_mask
        dump_main([str(trained.ckpt_dir)])
        pasted = _paste(capsys.readouterr().out)

        mask = resolve_feature_mask(pasted, N_SAE)

        assert FeatureVocab.from_mask(mask, N_SAE) == trained.vocab
        assert feature_mask_dirname(mask) == trained.ckpt_dir.parent.name

    def test_the_recovered_SFT_mask_PASSES_the_guard_that_sent_us_here(
        self, orphaned_sft_no_meta_mask, capsys
    ):
        """``_assert_sft_universe_matches_checkpoint_vocab`` is the guard whose error message
        names this tool. Its output must satisfy it."""
        trained = orphaned_sft_no_meta_mask
        dump_main([str(trained.ckpt_dir)])
        mask = resolve_feature_mask(_paste(capsys.readouterr().out), N_SAE)
        assert mask is not None

        _assert_sft_universe_matches_checkpoint_vocab(
            mask,
            mask,
            trained.vocab,
            N_SAE,
            checkpoint_dir=str(trained.ckpt_dir),
            base_meta={},
        )

    def test_the_emitted_header_does_not_LIE_about_the_mode(self, orphaned_sft_no_meta_mask):
        """The provenance comments hard-coded "Emitted as an explicit whitelist ..." directly
        above a body reading ``mode: blacklist``. The operator reads the header."""
        dumped = load_dumped_mask(orphaned_sft_no_meta_mask.ckpt_dir)
        assert dumped.mask is not None

        for out in (render(dumped, FORMAT_YAML), render(dumped, FORMAT_TXT)):
            header = "\n".join(ln for ln in out.splitlines() if ln.startswith("#"))
            assert "blacklist" in header
            assert "whitelist" not in header, (
                f"the header calls this a whitelist; the body emits mode={dumped.mask.mode.value}"
            )
