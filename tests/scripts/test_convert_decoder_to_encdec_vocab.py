"""The converter must build a COMPACTED encoder and record its mask (#695 PR B).

``convert_decoder_to_encdec.py`` already resolves a ``feature_mask`` — but only to
compose the ``bl_``/``wl_`` checkpoint leaf. Two things follow:

1. **It never passed the mask to the model.** Post-#695 every masked training run
   builds a compacted encoder, so a converted checkpoint that is full-width under a
   mask leaf is the wrong shape for the run that will consume it.

2. **It never wrote ``feature_mask_canonical`` into the meta** — a pre-existing gap.
   ``assert_resume_feature_mask_matches`` demands that key, so a masked converted
   checkpoint could not be resumed at all, and ``dump_feature_mask`` could not recover
   the mask from it. Compaction makes the gap load-bearing: without the meta mirror,
   ``build_encdec_model`` refuses the checkpoint outright (the meta and the weights'
   vocab buffers must agree).

The converter is a Hydra CLI, so it is exercised the way the existing suite does it —
in a real subprocess against a real tiny decoder checkpoint on disk, with
``NANOCHAT_BASE_DIR`` pointed at a tmp dir. Nothing is mocked; the assertions read the
``.pt`` and ``.json`` the converter actually wrote. Conversions are module-scoped
because each spawns a Python process.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import scripts.preprocess.convert_decoder_to_encdec as converter
from errors import FeatureClustersError, FeatureVocabError, ModelWeightLoadError
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model, save_checkpoint
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    assert_resume_feature_mask_matches,
)
from steering.conditional.feature_vocab import (
    FeatureVocab,
    assert_resume_feature_vocab_matches,
    feature_vocab_from_state_dict,
)
from tests.conftest import TINY_VOCAB, _make_gpt, _make_gpt_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_TAG = "d2"
STEP = 0
SEG = 16
THRESHOLD_DIR = "t2048.9"

N_SAE = 32
WHITELIST_IDS = (0, 3, 7, 10, 11, 20, 21, 30, 31)
WHITELIST = FeatureMask(mode=FeatureMaskMode.whitelist, ids=WHITELIST_IDS)
WHITELIST_VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=WHITELIST_IDS)

BLACKLIST_IDS = (1, 2, 4, 5, 6)
BLACKLIST = FeatureMask(mode=FeatureMaskMode.blacklist, ids=BLACKLIST_IDS)
BLACKLIST_VOCAB = FeatureVocab(
    n_sae_features=N_SAE,
    original_ids=tuple(i for i in range(N_SAE) if i not in BLACKLIST_IDS),
)

# A DIFFERENT whitelist with the SAME V as WHITELIST (9 ids). Every feature-indexed
# tensor therefore has the identical shape, so ``load_state_dict`` is perfectly happy
# and PyTorch sees nothing wrong — which is exactly what makes the encdec-source bug
# silent (see TestAnEncoderCarryingSourceIsRefused).
OTHER_IDS = (0, 3, 7, 10, 11, 20, 21, 29, 30)
OTHER_WHITELIST = FeatureMask(mode=FeatureMaskMode.whitelist, ids=OTHER_IDS)
OTHER_VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=OTHER_IDS)

# The encoder / cross-attention geometry ``_run_converter`` asks the CLI to build. An
# encdec *source* must be built with the same geometry, or the shapes would not line up
# and PyTorch's own "size mismatch" would mask the bug under test.
ENC_KW = {"n_embd": 32, "depth": 1, "n_head": 2, "n_kv_head": 2, "head_dim": 16}
XATTN_KW = {"frequency": 1, "n_head": None, "n_kv_head": None}
NUM_BUCKETS = 4

CONVERTER_SOURCE = Path(converter.__file__).read_text(encoding="utf-8")
CONVERTER_TREE = ast.parse(CONVERTER_SOURCE)

RUNNER_TEMPLATE = textwrap.dedent(
    """
    import sys

    # The converter warm-loads the tokenizer for a log message; the real tokenizer.pkl
    # is irrelevant to (and unavailable for) this test. Same stub the existing
    # converter suite uses.
    import nanochat.tokenizer as _tok_mod

    def _stub_get_tokenizer():
        class _StubTok:
            def get_vocab_size(self):
                return {vocab}
        return _StubTok()

    _tok_mod.get_tokenizer = _stub_get_tokenizer

    sys.argv = ["convert_decoder_to_encdec"] + {overrides!r}
    import runpy
    runpy.run_module("scripts.preprocess.convert_decoder_to_encdec", run_name="__main__")
    """
)


@dataclass(frozen=True)
class Conversion:
    """Everything the converter left on disk for one run."""

    base_dir: Path
    out_dir: Path
    state: dict[str, torch.Tensor]
    meta: dict
    decoder_state: dict[str, torch.Tensor]


def _write_decoder_checkpoint(base_dir: Path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = _make_gpt(device="cpu")
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_checkpoint(
        str(base_dir / "base_checkpoints" / MODEL_TAG),
        STEP,
        sd,
        None,
        {
            "step": STEP,
            "val_bpb": 1.23,
            "model_config": asdict(_make_gpt_config()),
            "user_config": {},
        },
    )
    return sd


def _mask_overrides(mask: FeatureMask | None) -> list[str]:
    """``feature_mask`` is not a key in the converter's YAML, so it is APPENDED."""
    if mask is None:
        return []
    ids = ",".join(str(i) for i in mask.ids)
    return [f"+feature_mask.mode={mask.mode.value}", f"+feature_mask.ids=[{ids}]"]


def _run_converter(
    base_dir: Path, mask: FeatureMask | None, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run the converter CLI in a real subprocess.

    ``check=False`` returns the ``CompletedProcess`` for a run that is EXPECTED to fail,
    so a test can assert on the exit code and the message the operator actually sees.
    """
    overrides = [
        "source=base",
        f"model_tag={MODEL_TAG}",
        f"step={STEP}",
        f"segment_size={SEG}",
        f"encoder.n_embd={ENC_KW['n_embd']}",
        f"encoder.depth={ENC_KW['depth']}",
        f"encoder.n_head={ENC_KW['n_head']}",
        f"encoder.n_kv_head={ENC_KW['n_kv_head']}",
        f"encoder.head_dim={ENC_KW['head_dim']}",
        f"cross_attention.frequency={XATTN_KW['frequency']}",
        f"n_sae_features={N_SAE}",
        f"num_buckets={NUM_BUCKETS}",
        *_mask_overrides(mask),
    ]
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = str(base_dir)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c", RUNNER_TEMPLATE.format(vocab=TINY_VOCAB, overrides=overrides)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Converter subprocess failed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _build_encdec(vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    """A real EncoderDecoderGPT with the geometry ``_run_converter`` asks for."""
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            _make_gpt_config(),
            EncoderConfig(**ENC_KW),
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=NUM_BUCKETS,
            feature_vocab=vocab,
        )
    model.to_empty(device="cpu")
    torch.manual_seed(0)
    model.init_weights()
    return model


def _write_encdec_as_base_checkpoint(
    base_dir: Path, vocab: FeatureVocab | None
) -> dict[str, torch.Tensor]:
    """An ENCDEC checkpoint sitting where the converter expects a DECODER-ONLY one.

    The realistic operator error: pointing ``source=base`` at a tag whose
    ``base_checkpoints/`` dir holds an already-converted (or otherwise
    encoder-carrying) model. The decoder ``model_config`` in the meta is genuine, so
    nothing upstream of the load can tell the difference — the extra ``encoder.*`` /
    ``cross_attn_layers.*`` keys in the state dict are the ONLY signal that the source
    violates the converter's decoder-only contract.
    """
    model = _build_encdec(vocab)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_checkpoint(
        str(base_dir / "base_checkpoints" / MODEL_TAG),
        STEP,
        sd,
        None,
        {
            "step": STEP,
            "val_bpb": 1.23,
            "model_config": asdict(_make_gpt_config()),
            "user_config": {},
        },
    )
    return sd


def _out_dir(base_dir: Path, mask: FeatureMask | None) -> Path:
    d = base_dir / "conditional_encdec_checkpoints" / MODEL_TAG / f"s{SEG}" / THRESHOLD_DIR
    return d if mask is None else d / mask.dirname()


def _convert(tmp_path: Path, mask: FeatureMask | None) -> Conversion:
    dec_sd = _write_decoder_checkpoint(tmp_path)
    _run_converter(tmp_path, mask)
    out = _out_dir(tmp_path, mask)
    model_path = out / f"model_{STEP:06d}.pt"
    assert model_path.exists(), f"Converter did not write {model_path}"
    return Conversion(
        base_dir=tmp_path,
        out_dir=out,
        state=torch.load(model_path, map_location="cpu", weights_only=True),
        meta=json.loads((out / f"meta_{STEP:06d}.json").read_text()),
        decoder_state=dec_sd,
    )


@pytest.fixture(scope="module")
def whitelisted(tmp_path_factory) -> Conversion:
    return _convert(tmp_path_factory.mktemp("wl"), WHITELIST)


@pytest.fixture(scope="module")
def blacklisted(tmp_path_factory) -> Conversion:
    return _convert(tmp_path_factory.mktemp("bl"), BLACKLIST)


@pytest.fixture(scope="module")
def unmasked(tmp_path_factory) -> Conversion:
    return _convert(tmp_path_factory.mktemp("none"), None)


# ===========================================================================
# 1. The encoder is actually compacted
# ===========================================================================


class TestMaskedConversionCompactsTheEncoder:
    def test_feature_embed_has_V_rows_not_n_sae(self, whitelisted):
        rows = whitelisted.state["encoder.feature_embed.weight"].shape[0]

        assert rows == len(WHITELIST_IDS) == 9
        assert rows != N_SAE, "the converter built a full-width encoder under a mask"

    def test_every_value_embed_is_compacted_too(self, whitelisted):
        ve = [
            k
            for k in whitelisted.state
            if k.startswith("encoder.value_embeds.") and k.endswith(".weight")
        ]

        assert ve, "no encoder value_embeds in the converted checkpoint"
        for key in ve:
            assert whitelisted.state[key].shape[0] == len(WHITELIST_IDS), (
                f"{key} is not compacted — value_embeds are the MAJORITY of the "
                "feature-indexed parameters"
            )

    def test_the_vocab_buffers_are_persisted(self, whitelisted):
        ids = whitelisted.state["encoder.feature_vocab_ids"]
        n_sae = whitelisted.state["encoder.feature_vocab_n_sae"]

        assert tuple(int(i) for i in ids.tolist()) == WHITELIST_IDS
        assert int(n_sae.item()) == N_SAE, (
            "n_sae_features is NOT recoverable from a compacted row count — it must "
            "be persisted explicitly"
        )

    def test_the_checkpoint_describes_its_own_vocabulary(self, whitelisted):
        n_sae, vocab = feature_vocab_from_state_dict(whitelisted.state)

        assert n_sae == N_SAE
        assert vocab == WHITELIST_VOCAB

    def test_a_BLACKLIST_compacts_identically(self, blacklisted):
        """The committed default produces a blacklist, not a whitelist. Gating
        compaction on the mode would have excluded the very run that motivated #695."""
        n_sae, vocab = feature_vocab_from_state_dict(blacklisted.state)

        assert n_sae == N_SAE
        assert vocab == BLACKLIST_VOCAB
        assert blacklisted.state["encoder.feature_embed.weight"].shape[0] == (
            N_SAE - len(BLACKLIST_IDS)
        )


# ===========================================================================
# 2. The meta gap this PR closes
# ===========================================================================


class TestConverterWritesTheMaskMeta:
    def test_feature_mask_canonical_is_written(self, whitelisted):
        """The pre-existing gap: the converter composed the mask leaf but never
        recorded the mask, so a masked converted checkpoint could not be resumed."""
        assert whitelisted.meta["feature_mask_canonical"] == {
            "mode": "whitelist",
            "ids": list(WHITELIST_IDS),
        }

    def test_the_written_mask_satisfies_the_RESUME_guard(self, whitelisted):
        """The real function a resume calls. This is the whole point of the key."""
        assert_resume_feature_mask_matches(
            WHITELIST, whitelisted.meta, checkpoint_dir=str(whitelisted.out_dir)
        )

    def test_a_blacklist_mask_is_recorded_too(self, blacklisted):
        assert blacklisted.meta["feature_mask_canonical"] == {
            "mode": "blacklist",
            "ids": list(BLACKLIST_IDS),
        }
        assert_resume_feature_mask_matches(
            BLACKLIST, blacklisted.meta, checkpoint_dir=str(blacklisted.out_dir)
        )

    def test_the_feature_vocab_fingerprint_is_written(self, whitelisted):
        assert whitelisted.meta["feature_vocab"] == WHITELIST_VOCAB.metadata_dict()

    def test_the_written_vocab_satisfies_the_RESUME_guard(self, whitelisted):
        assert_resume_feature_vocab_matches(
            WHITELIST_VOCAB, whitelisted.meta, checkpoint_dir=str(whitelisted.out_dir)
        )

    def test_the_recorded_mask_and_the_recorded_vocab_agree(self, whitelisted):
        """The two records are independent; a converter that derived them from
        different masks would produce a checkpoint ``dump_feature_mask`` refuses."""
        recorded = FeatureMask(
            mode=FeatureMaskMode(whitelisted.meta["feature_mask_canonical"]["mode"]),
            ids=tuple(whitelisted.meta["feature_mask_canonical"]["ids"]),
        )
        _, weights_vocab = feature_vocab_from_state_dict(whitelisted.state)

        assert FeatureVocab.from_mask(recorded, N_SAE) == weights_vocab


# ===========================================================================
# 3. The converted checkpoint is actually loadable
# ===========================================================================


@pytest.fixture
def _fake_tokenizer():
    tok = Mock()
    tok.get_vocab_size.return_value = TINY_VOCAB
    with patch("nanochat.checkpoint_manager.get_tokenizer", return_value=tok):
        yield


class TestConvertedCheckpointLoads:
    def test_build_encdec_model_loads_the_masked_conversion(self, whitelisted, _fake_tokenizer):
        """End-to-end. ``build_encdec_model`` cross-checks the meta's vocab
        fingerprint against the weights' buffers, so this ALSO fails if the converter
        writes a compacted checkpoint without the meta mirror."""
        model, _, _ = build_encdec_model(
            str(whitelisted.out_dir),
            STEP,
            "cpu",
            ModelPhase.TRAIN,
            eval_conditioning=EvalConditioning.not_evaluating(),
        )

        assert model.encoder.config.n_sae_features == N_SAE  # ORIGINAL width, not V
        assert model.encoder.feature_embed.weight.shape[0] == len(WHITELIST_IDS)
        assert model.feature_vocab == WHITELIST_VOCAB
        model.assert_vocab_consistent()

    def test_a_missing_vocab_meta_still_LOADS_because_the_buffers_are_the_record(
        self, whitelisted, _fake_tokenizer
    ):
        """The meta block is a MIRROR, not a sidecar.

        An earlier draft of this test asserted a missing mirror was REFUSED — that was the
        pre-review behaviour, and it made the sidecar mandatory, i.e. requirement 1 ("a
        loaded model is self-describing and does not need ... a sidecar file") was not
        actually delivered. The loader now tolerates ABSENCE (warning + load from the
        persistent buffers, which ARE the record) and raises only on DISAGREEMENT.

        The converter still writes the mirror, and it should: it lets tooling compare a
        checkpoint's vocabulary without torch.load-ing a multi-GB .pt. But its absence must
        never make a checkpoint unreadable.
        """
        meta_path = whitelisted.out_dir / f"meta_{STEP:06d}.json"
        original = meta_path.read_text()
        meta = json.loads(original)
        del meta["feature_vocab"]
        meta_path.write_text(json.dumps(meta))
        try:
            model, _, _ = build_encdec_model(
                str(whitelisted.out_dir),
                STEP,
                "cpu",
                ModelPhase.TRAIN,
                eval_conditioning=EvalConditioning.not_evaluating(),
            )
            assert model.feature_vocab is not None
            model.assert_vocab_consistent()
        finally:
            meta_path.write_text(original)

    def test_but_a_DISAGREEING_vocab_meta_is_still_refused(self, whitelisted, _fake_tokenizer):
        """Absence is fine; disagreement is corruption and must stay fatal."""
        meta_path = whitelisted.out_dir / f"meta_{STEP:06d}.json"
        original = meta_path.read_text()
        meta = json.loads(original)
        meta["feature_vocab"]["sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta))
        try:
            with pytest.raises(FeatureVocabError, match="(?i)disagree"):
                build_encdec_model(
                    str(whitelisted.out_dir),
                    STEP,
                    "cpu",
                    ModelPhase.TRAIN,
                    eval_conditioning=EvalConditioning.not_evaluating(),
                )
        finally:
            meta_path.write_text(original)

    def test_build_encdec_model_loads_the_blacklisted_conversion(
        self, blacklisted, _fake_tokenizer
    ):
        model, _, _ = build_encdec_model(
            str(blacklisted.out_dir),
            STEP,
            "cpu",
            ModelPhase.TRAIN,
            eval_conditioning=EvalConditioning.not_evaluating(),
        )

        assert model.feature_vocab == BLACKLIST_VOCAB
        model.assert_vocab_consistent()


# ===========================================================================
# 4. The unmasked path is untouched (I2 / I5)
# ===========================================================================


class TestUnmaskedConversionIsUnchanged:
    def test_no_vocab_keys_in_the_state_dict(self, unmasked):
        """A no-mask conversion must stay byte-identical to pre-#695, or every
        existing ``strict=True`` load of a converted checkpoint breaks."""
        assert not [k for k in unmasked.state if "feature_vocab" in k]

    def test_the_tables_are_full_width(self, unmasked):
        assert unmasked.state["encoder.feature_embed.weight"].shape[0] == N_SAE

    def test_no_mask_meta_keys(self, unmasked):
        assert "feature_mask_canonical" not in unmasked.meta
        assert "feature_vocab" not in unmasked.meta

    def test_the_output_path_has_no_mask_leaf(self, unmasked):
        assert unmasked.out_dir.name == THRESHOLD_DIR

    def test_it_still_loads(self, unmasked, _fake_tokenizer):
        model, _, _ = build_encdec_model(
            str(unmasked.out_dir),
            STEP,
            "cpu",
            ModelPhase.TRAIN,
            eval_conditioning=EvalConditioning.not_evaluating(),
        )

        assert model.feature_vocab is None
        assert model.encoder.config.n_sae_features == N_SAE


# ===========================================================================
# 5. Regressions the compaction wiring could plausibly introduce
# ===========================================================================


class TestDecoderIsStillPreserved:
    def test_decoder_weights_are_bit_exact_under_a_mask(self, whitelisted):
        """``init_weights()`` must still run BEFORE the ``strict=False`` load. If the
        vocab wiring moved the init after it, every decoder weight would be silently
        replaced by fresh random init — and the conversion would be worthless."""
        mismatched = [
            k
            for k, dec in whitelisted.decoder_state.items()
            if k not in whitelisted.state or not torch.equal(whitelisted.state[k], dec)
        ]

        assert not mismatched, f"decoder weights not preserved: {mismatched}"

    def test_the_mask_leaf_matches_the_mask_that_shaped_the_MODEL(self, whitelisted):
        """The leaf and the vocab must come from the same resolved mask — if they can
        drift, the path says one dictionary and the weights encode another."""
        assert whitelisted.out_dir.name == WHITELIST.dirname()
        assert whitelisted.out_dir.name.startswith("wl_")

        _, vocab = feature_vocab_from_state_dict(whitelisted.state)
        assert vocab == FeatureVocab.from_mask(WHITELIST, N_SAE)


class TestMaskedConversionIsIdempotent:
    def test_two_masked_conversions_are_bit_identical(self, tmp_path):
        """Slice-init draws the FULL (n_sae, d) table before subsetting, so a compacted
        conversion consumes a different amount of RNG than a naive one would. It must
        still be deterministic under the converter's ``manual_seed(0)``."""
        first = _convert(tmp_path, WHITELIST)
        _run_converter(tmp_path, WHITELIST)
        second = torch.load(
            first.out_dir / f"model_{STEP:06d}.pt", map_location="cpu", weights_only=True
        )

        assert set(first.state) == set(second)
        for key in first.state:
            assert torch.equal(first.state[key], second[key]), f"non-idempotent: {key}"


# ===========================================================================
# 6. The source must be DECODER-ONLY (review finding 8)
# ===========================================================================
#
# The converter was the ONE model load in the branch with no vocab consistency check,
# and it downgraded a contract violation to ``logger.warning``. Point it at a source
# that already carries encoder keys (an encdec checkpoint) and ``assign=True`` REPLACES
# ``encoder.feature_embed.weight`` and the persistent ``encoder.feature_vocab_ids`` with
# the SOURCE's tensors — while the meta ``feature_vocab`` is written from the CONFIG's
# freshly-resolved vocab. The conversion "succeeds" and writes a checkpoint whose
# state-dict vocab and meta fingerprint disagree, discovered only much later when
# ``build_encdec_model`` refuses it.
#
# NOTE the trap this walked into: those keys are NOT reported in ``load_state_dict``'s
# ``unexpected`` list. The target IS an EncoderDecoderGPT, so ``encoder.*`` and
# ``cross_attn_layers.*`` are EXPECTED keys — they are consumed silently. A guard that
# inspected ``unexpected`` would be vacuous. The source state dict itself must be
# checked, which is what ``_assert_decoder_only_source`` does.


def _decoder_only_state_dict() -> dict[str, torch.Tensor]:
    return dict(_make_gpt(device="cpu").state_dict())


class TestDecoderOnlySourceGuard:
    """Unit tests for ``convert_decoder_to_encdec._assert_decoder_only_source``."""

    def test_a_decoder_only_state_dict_is_accepted(self):
        converter._assert_decoder_only_source(_decoder_only_state_dict(), checkpoint_dir="/ckpt")

    def test_an_encoder_key_is_refused(self):
        sd = _decoder_only_state_dict()
        sd["encoder.feature_embed.weight"] = torch.zeros(N_SAE, 32)

        with pytest.raises(ModelWeightLoadError, match="encoder.feature_embed.weight"):
            converter._assert_decoder_only_source(sd, checkpoint_dir="/ckpt")

    def test_a_cross_attention_key_is_refused(self):
        sd = _decoder_only_state_dict()
        sd["cross_attn_layers.0.c_q.weight"] = torch.zeros(64, 64)

        with pytest.raises(ModelWeightLoadError, match="cross_attn_layers.0.c_q.weight"):
            converter._assert_decoder_only_source(sd, checkpoint_dir="/ckpt")

    def test_a_bare_feature_vocab_key_is_refused(self):
        """Defensive: a hand-assembled state dict need not nest the buffers under
        ``encoder.`` for them to reinterpret every embedding row."""
        sd = _decoder_only_state_dict()
        sd["feature_vocab_ids"] = torch.tensor([1, 2, 3])

        with pytest.raises(ModelWeightLoadError, match="feature_vocab_ids"):
            converter._assert_decoder_only_source(sd, checkpoint_dir="/ckpt")

    def test_every_offending_key_is_named(self):
        """An operator must not have to re-run the converter once per bad key."""
        sd = _decoder_only_state_dict()
        sd["encoder.feature_embed.weight"] = torch.zeros(N_SAE, 32)
        sd["encoder.feature_vocab_ids"] = torch.tensor([1, 2, 3])
        sd["cross_attn_layers.0.c_kv.weight"] = torch.zeros(128, 32)

        with pytest.raises(ModelWeightLoadError) as exc:
            converter._assert_decoder_only_source(sd, checkpoint_dir="/ckpt/d2")

        msg = str(exc.value)
        for key in (
            "encoder.feature_embed.weight",
            "encoder.feature_vocab_ids",
            "cross_attn_layers.0.c_kv.weight",
        ):
            assert key in msg
        assert "/ckpt/d2" in msg

    def test_the_decoder_keys_are_not_named_as_offenders(self):
        """The message must point at the encoder keys, not drown them in the ~50
        perfectly-legitimate decoder keys."""
        sd = _decoder_only_state_dict()
        sd["encoder.feature_embed.weight"] = torch.zeros(N_SAE, 32)

        with pytest.raises(ModelWeightLoadError) as exc:
            converter._assert_decoder_only_source(sd, checkpoint_dir="/ckpt")

        assert "transformer.wte.weight" not in str(exc.value)


@pytest.fixture(scope="module")
def compacted_encdec_source(tmp_path_factory) -> tuple[subprocess.CompletedProcess, Path]:
    """Convert from a COMPACTED encdec source whose vocab differs from the config's.

    Source carries ``WHITELIST_VOCAB``; the config resolves ``OTHER_WHITELIST`` — the
    same V, so every tensor shape lines up and ``load_state_dict`` raises nothing.
    """
    base = tmp_path_factory.mktemp("encdec_compacted")
    _write_encdec_as_base_checkpoint(base, WHITELIST_VOCAB)
    return _run_converter(base, OTHER_WHITELIST, check=False), base


@pytest.fixture(scope="module")
def uncompacted_encdec_source(tmp_path_factory) -> tuple[subprocess.CompletedProcess, Path]:
    """Convert from an UNCOMPACTED encdec source with no mask configured.

    Both vocabs are ``None``, so ``assert_vocab_consistent()`` passes and the encoder
    weights are silently inherited from the source. Only the source-key guard catches
    this one — which is why fix (b) is not redundant with fix (a).
    """
    base = tmp_path_factory.mktemp("encdec_uncompacted")
    _write_encdec_as_base_checkpoint(base, None)
    return _run_converter(base, None, check=False), base


class TestAnEncoderCarryingSourceIsRefused:
    def test_the_two_masks_have_the_SAME_V(self):
        """Pins the premise: identical shapes ⇒ PyTorch sees nothing wrong. If this
        ever stops holding, the fixture below would be caught by a bare size mismatch
        and would pass for the wrong reason."""
        assert WHITELIST_VOCAB.size == OTHER_VOCAB.size == 9
        assert WHITELIST_VOCAB != OTHER_VOCAB

    def test_a_compacted_encdec_source_exits_nonzero(self, compacted_encdec_source):
        proc, _ = compacted_encdec_source

        assert proc.returncode != 0, (
            "the converter accepted an encdec source: it silently REPLACED the freshly "
            "initialised encoder (and its vocab buffer) with the source's, while writing "
            "the config's vocab into the meta"
        )

    def test_the_failure_names_the_offending_encoder_keys(self, compacted_encdec_source):
        proc, _ = compacted_encdec_source
        stderr = proc.stderr

        assert "ModelWeightLoadError" in stderr
        assert "encoder.feature_embed.weight" in stderr
        assert "encoder.feature_vocab_ids" in stderr

    def test_no_corrupt_checkpoint_is_left_behind(self, compacted_encdec_source):
        """Fail LOUD *and* leave nothing on disk: a half-written checkpoint whose meta
        and weights disagree is exactly what this guard exists to prevent."""
        _, base = compacted_encdec_source
        out = _out_dir(base, OTHER_WHITELIST)

        assert not (out / f"model_{STEP:06d}.pt").exists()

    def test_an_UNCOMPACTED_encdec_source_is_ALSO_refused(self, uncompacted_encdec_source):
        """The case ``assert_vocab_consistent()`` alone cannot catch: no mask on either
        side ⇒ both vocabs are ``None`` ⇒ the consistency check passes while the encoder
        weights are silently inherited from the source instead of freshly initialised."""
        proc, base = uncompacted_encdec_source

        assert proc.returncode != 0
        assert "ModelWeightLoadError" in proc.stderr
        assert "encoder.feature_embed.weight" in proc.stderr
        assert not (_out_dir(base, None) / f"model_{STEP:06d}.pt").exists()

    def test_a_decoder_only_source_still_converts(self, whitelisted):
        """The guard must not have broken the contract it protects: the real
        decoder-only path still produces a compacted, loadable checkpoint."""
        assert whitelisted.state["encoder.feature_embed.weight"].shape[0] == 9


# ===========================================================================
# 7. Wiring — main() really does check what it loads (review finding 8)
# ===========================================================================
#
# ``tests/steering/test_encdec_sft_vocab.py`` has an equivalent AST test, but it only
# scans the SFT script — which is precisely why the converter escaped. This is the
# converter's copy of that guard.


def _converter_main() -> ast.FunctionDef:
    for node in CONVERTER_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("convert_decoder_to_encdec.main not found")


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
    """Nodes belonging to ``stmt`` itself, not to any statement nested inside it."""
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


class TestConverterWiring:
    def test_feature_clusters_config_is_rejected_immediately_and_names_the_surface(self):
        with pytest.raises(
            FeatureClustersError,
            match="convert_decoder_to_encdec does not support static feature clusters",
        ):
            converter.main(["+feature_clusters.path=/not-read-by-the-rejection-guard.yaml"])

    def test_feature_clusters_guard_runs_before_config_logging_or_checkpoint_io(self):
        calls = _calls_named(_converter_main(), "reject_feature_clusters_config")
        assert len(calls) == 1
        guard = calls[0]
        loads = _calls_named(_converter_main(), "load_checkpoint")
        assert len(loads) == 1
        assert guard.lineno < loads[0].lineno

    def test_every_model_load_state_dict_is_followed_by_assert_vocab_consistent(self):
        """The converter was the ONE model load in the branch without this call.

        ``assign=True`` replaces the persistent vocab buffer but NOT the derived map, so
        every model load must be followed by the consistency check, in the same block.
        """
        checked = 0
        for block in _blocks(_converter_main()):
            for i, stmt in enumerate(block):
                if not any(_is_model_load(n) for n in _own_nodes(stmt)):
                    continue
                assert any(_calls_named(s, "assert_vocab_consistent") for s in block[i + 1 :]), (
                    f"model.load_state_dict at line {stmt.lineno} is not followed by "
                    "model.assert_vocab_consistent() in the same block — a source "
                    "checkpoint whose vocabulary differs from the configured one would "
                    "load silently, and the written meta would disagree with the "
                    "written weights."
                )
                checked += 1
        assert checked == 1, (
            f"expected the 1 known model-load site in the converter, found {checked}"
        )

    def test_main_wires_the_decoder_only_source_guard(self):
        assert _calls_named(_converter_main(), "_assert_decoder_only_source"), (
            "_assert_decoder_only_source is defined but never called — an encdec source "
            "would still be accepted"
        )

    def test_the_guard_inspects_the_SOURCE_state_dict_not_the_unexpected_list(self):
        """Regression pin for the trap this finding walked into.

        An encdec source's ``encoder.*`` / ``cross_attn_layers.*`` keys are EXPECTED keys
        of the EncoderDecoderGPT target, so ``load_state_dict`` consumes them silently and
        they NEVER appear in its ``unexpected`` list. A guard that filtered ``unexpected``
        would therefore be vacuous against the very failure it is meant to catch. It has
        to be handed the SOURCE state dict.
        """
        calls = _calls_named(_converter_main(), "_assert_decoder_only_source")

        assert len(calls) == 1
        (arg,) = calls[0].args
        assert isinstance(arg, ast.Name) and arg.id == "model_data", (
            "the decoder-only guard must inspect the SOURCE state dict (model_data); "
            "filtering load_state_dict's `unexpected` list would be vacuous"
        )

    def test_no_function_local_imports(self):
        """Hard project rule: all imports at the top of the file."""
        for node in ast.walk(_converter_main()):
            assert not isinstance(node, ast.Import | ast.ImportFrom), (
                f"function-local import in main() at line {node.lineno}: {ast.unparse(node)}"
            )
