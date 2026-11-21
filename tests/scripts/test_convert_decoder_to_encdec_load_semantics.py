"""A guard over ``load_state_dict``'s ``unexpected`` list would be VACUOUS. Proven.

``convert_decoder_to_encdec._assert_decoder_only_source`` inspects the SOURCE state dict.
Its docstring says the obvious-looking alternative — filtering ``load_state_dict``'s
``unexpected`` list — "would be vacuous against the exact failure this prevents", because
the target *is* an ``EncoderDecoderGPT``, so ``encoder.*`` and ``cross_attn_layers.*`` are
its OWN keys: PyTorch consumes them in silence.

``test_convert_decoder_to_encdec_vocab.py::TestConverterWiring`` pins that claim with an
AST assertion — it proves the guard is *called* with ``model_data``. It does not prove the
claim itself. So the single most important sentence in that guard's rationale rests on
nobody having checked it, and a reviewer who "simplified" the guard into
``if unexpected: raise`` would find every test still green.

This module checks it against real models and the real PyTorch loader:

* a decoder-only source leaves the encoder keys **missing** — which is the only reason
  ``strict=False`` is there at all;
* an encdec source produces ``unexpected == []`` — the keys are consumed, silently;
* and with ``assign=True`` those keys REPLACE the freshly-initialised encoder, which is
  the corruption the real guard exists to stop.

It then runs the hypothetical ``if unexpected: raise`` guard over both sources and shows
it never fires, while the real guard fires on exactly the bad one.
"""

from __future__ import annotations

import pytest
import torch

import scripts.preprocess.convert_decoder_to_encdec as converter
from errors import FeatureVocabError, ModelWeightLoadError
from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab

N_SAE = 32
NUM_BUCKETS = 4

# Same geometry on both sides. If the shapes disagreed, PyTorch's own "size mismatch"
# would abort the load and mask the very silence under test.
DEC_KW = {
    "sequence_len": 16,
    "vocab_size": 128,
    "n_layer": 2,
    "n_head": 2,
    "n_kv_head": 2,
    "n_embd": 64,
    "window_pattern": "L",
}
ENC_KW = {"n_embd": 32, "depth": 1, "n_head": 2, "n_kv_head": 2, "head_dim": 16}
XATTN_KW = {"frequency": 1, "n_head": None, "n_kv_head": None}

IDS_A = (0, 3, 7, 10, 11, 20, 21, 30, 31)
IDS_B = (0, 3, 7, 10, 11, 20, 21, 29, 30)  # SAME V (9), different ids
VOCAB_A = FeatureVocab(n_sae_features=N_SAE, original_ids=IDS_A)
VOCAB_B = FeatureVocab(n_sae_features=N_SAE, original_ids=IDS_B)

ENCODER_PREFIXES = ("encoder.", "cross_attn_layers.")


def _decoder_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )


def _encoder_config() -> EncoderConfig:
    return EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16)


def _cross_attention_config() -> CrossAttentionConfig:
    return CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None)


def _encdec(vocab: FeatureVocab | None, *, seed: int) -> EncoderDecoderGPT:
    with torch.device("meta"):
        m = EncoderDecoderGPT(
            _decoder_config(),
            _encoder_config(),
            _cross_attention_config(),
            n_sae_features=N_SAE,
            num_buckets=NUM_BUCKETS,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    torch.manual_seed(seed)
    m.init_weights()
    return m


def _decoder_only() -> dict[str, torch.Tensor]:
    """A real decoder-only GPT state dict — the converter's legitimate source."""
    with torch.device("meta"):
        g = GPT(_decoder_config(), pad_vocab_size_to=128)
    g.to_empty(device="cpu")
    torch.manual_seed(99)
    g.init_weights()
    return dict(g.state_dict())


def _encoder_keys(sd) -> list[str]:
    return sorted(k for k in sd if k.startswith(ENCODER_PREFIXES))


def _hypothetical_unexpected_guard(unexpected: list[str]) -> None:
    """The guard a reviewer would 'obviously' have written instead.

    Reproduced here verbatim so its vacuity is demonstrated rather than asserted.
    """
    if unexpected:
        raise ModelWeightLoadError(f"source carries unexpected keys: {unexpected}")


# ===========================================================================
# 1. The premise: encoder keys are the TARGET's own keys
# ===========================================================================


class TestTheTargetOwnsTheEncoderKeys:
    def test_the_converter_target_is_an_EncoderDecoderGPT(self):
        """Stated plainly, because everything below follows from it."""
        assert issubclass(EncoderDecoderGPT, torch.nn.Module)
        target = _encdec(None, seed=1)

        assert _encoder_keys(target.state_dict()), "the target has no encoder keys at all"

    def test_an_encdec_sources_encoder_keys_are_a_SUBSET_of_the_targets_keys(self):
        """The mechanical reason ``unexpected`` comes back empty: there is nothing
        unexpected about them."""
        source = _encdec(None, seed=2).state_dict()
        target = _encdec(None, seed=1).state_dict()

        assert set(_encoder_keys(source)) <= set(target.keys())


# ===========================================================================
# 2. The proof
# ===========================================================================


class TestAGuardOverUnexpectedWouldBeVacuous:
    def test_a_DECODER_ONLY_source_leaves_the_encoder_keys_MISSING(self):
        """Why ``strict=False`` exists. ``missing`` is the informative list here — and it
        is informative in exactly the WRONG direction for catching a bad source."""
        target = _encdec(None, seed=1)

        missing, unexpected = target.load_state_dict(
            _decoder_only(), strict=False, assign=True
        )

        assert _encoder_keys(missing), "a decoder-only source must leave encoder keys unset"
        assert list(unexpected) == []

    def test_an_ENCDEC_source_produces_an_EMPTY_unexpected_list(self):
        """THE claim. The bad source is indistinguishable from the good one by
        ``unexpected`` — both give ``[]``."""
        target = _encdec(None, seed=1)
        source = _encdec(None, seed=2).state_dict()
        assert _encoder_keys(source), "the fixture is not actually an encdec source"

        missing, unexpected = target.load_state_dict(source, strict=False, assign=True)

        assert list(unexpected) == [], (
            "PyTorch reported the encoder keys as unexpected — if this ever becomes true, "
            "the guard's rationale changes and _assert_decoder_only_source can be "
            "simplified. Until then it cannot."
        )
        assert list(missing) == [], "an encdec source supplies every key the target has"

    def test_the_hypothetical_guard_fires_on_NEITHER_source(self):
        """Run the alternative implementation. It is a no-op against both inputs — it
        cannot tell the legitimate source from the corrupting one."""
        for name, source in (
            ("decoder-only", _decoder_only()),
            ("encdec", _encdec(None, seed=2).state_dict()),
        ):
            target = _encdec(None, seed=1)
            _missing, unexpected = target.load_state_dict(
                source, strict=False, assign=True
            )

            _hypothetical_unexpected_guard(list(unexpected))  # never raises

            assert list(unexpected) == [], f"{name}: unexpected was not empty"

    def test_but_the_REAL_guard_separates_them_cleanly(self):
        """The contrast, on the same two inputs."""
        converter._assert_decoder_only_source(_decoder_only(), checkpoint_dir="/ckpt")

        with pytest.raises(ModelWeightLoadError, match="NOT decoder-only"):
            converter._assert_decoder_only_source(
                _encdec(None, seed=2).state_dict(), checkpoint_dir="/ckpt"
            )


# ===========================================================================
# 3. The corruption the empty list is hiding
# ===========================================================================


class TestTheSilentlyConsumedKeysCorruptTheEncoder:
    def test_assign_True_REPLACES_the_freshly_initialised_encoder(self):
        """The converter's whole contract is "the encoder is built FROM SCRATCH". With an
        encdec source it is not: the source's encoder wins, silently."""
        target = _encdec(None, seed=1)
        source_model = _encdec(None, seed=2)
        source = source_model.state_dict()

        fresh = target.encoder.feature_embed.weight.detach().clone()
        theirs = source_model.encoder.feature_embed.weight.detach().clone()
        assert not torch.equal(fresh, theirs), "the two fixtures are not distinguishable"

        target.load_state_dict(source, strict=False, assign=True)

        assert torch.equal(target.encoder.feature_embed.weight, theirs), (
            "the source's encoder did NOT overwrite the target's — this test no longer "
            "reproduces the failure it is guarding"
        )

    def test_the_UNCOMPACTED_case_slips_past_assert_vocab_consistent_as_well(self):
        """The case where the source-key guard is the ONLY line of defence.

        Both sides are unmasked, so both vocabs are ``None``, so the post-load consistency
        check has nothing to compare and passes. The encoder has been silently inherited
        and NOTHING notices.
        """
        target = _encdec(None, seed=1)
        source = _encdec(None, seed=2).state_dict()

        _m, unexpected = target.load_state_dict(source, strict=False, assign=True)

        assert list(unexpected) == []
        target.assert_vocab_consistent()  # passes — there is no vocab to disagree about
        assert target.feature_vocab is None

    def test_the_COMPACTED_same_V_case_is_caught_but_only_AFTER_the_load(self):
        """Belt and braces. Here ``assert_vocab_consistent`` DOES catch it — but only once
        the weights are already in the model, and only because a vocab exists to compare.
        The source guard fires first, before anything is overwritten."""
        target = _encdec(VOCAB_B, seed=1)
        source = _encdec(VOCAB_A, seed=2).state_dict()
        assert VOCAB_A.size == VOCAB_B.size and VOCAB_A != VOCAB_B

        _m, unexpected = target.load_state_dict(source, strict=False, assign=True)

        assert list(unexpected) == []  # still nothing unexpected, even here
        with pytest.raises(FeatureVocabError):
            target.assert_vocab_consistent()

    def test_the_source_guard_rejects_that_source_BEFORE_the_load(self):
        source = _encdec(VOCAB_A, seed=2).state_dict()

        with pytest.raises(ModelWeightLoadError) as exc:
            converter._assert_decoder_only_source(source, checkpoint_dir="/ckpt")

        msg = str(exc.value)
        assert "encoder.feature_vocab_ids" in msg, (
            "the vocab buffer is the key whose silent replacement makes the meta and the "
            f"weights describe different models; it must be named: {msg!r}"
        )
        assert "does not need converting" in msg, "the error must tell the operator what to do"
