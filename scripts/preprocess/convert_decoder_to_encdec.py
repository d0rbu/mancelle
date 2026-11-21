"""Convert a decoder-only (GPT) checkpoint to an encoder-decoder (EncoderDecoderGPT) checkpoint.

Loads the decoder weights and builds an EncoderDecoderGPT with the same decoder
config plus user-specified encoder/cross-attention settings. Decoder weights are
preserved bit-exactly (verified in tests/test_convert_decoder_to_encdec.py).
The encoder is initialized from scratch and is untrained after conversion. The
cross-attention layers — including c_proj — are initialized with
`Uniform(+-sqrt(3/dec_n_embd))` (see cross_attention.py::init_weights).

Note: c_proj is NOT zero-initialized. An earlier implementation zeroed c_proj
so the converted model's output at step 0 matched the pure decoder, but all-zero
weights cause FP8 tensorwise scaling to degenerate (amax=0 -> extreme scale
factors -> inf gradients on the first backward). The Uniform init closes that
regression at the cost of the "identical output at step 0" property, which no
longer holds. A deterministic RNG seed below keeps the conversion reproducible.

Usage:
    python -m scripts.preprocess.convert_decoder_to_encdec source=sft
    python -m scripts.preprocess.convert_decoder_to_encdec source=sft n_sae_features=16384 num_buckets=10
    python -m scripts.preprocess.convert_decoder_to_encdec source=base encoder.depth=0.5 cross_attention.frequency=4
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

from errors import ModelWeightLoadError
from nanochat.checkpoint_manager import (
    _fuse_qkv_weights,
    _patch_missing_config_keys,
    _patch_missing_keys,
    find_largest_model,
    find_last_step,
    get_checkpoint_dir,
    load_checkpoint,
    save_checkpoint,
)
from nanochat.common import get_base_dir
from nanochat.gpt import GPTConfig
from nanochat.tokenizer import get_tokenizer
from steering.conditional.compaction_summary import (
    format_compaction_summary,
    no_compaction_reason,
)
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import reject_feature_clusters_config
from steering.conditional.feature_mask import (
    append_mask_leaf,
    canonical_dict,
    resolve_feature_mask,
)
from steering.conditional.feature_pipeline import feature_selection_dirname
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import assert_no_gradient_ranking, parse_feature_selection

# Every key a decoder-only source is forbidden to carry. ``feature_vocab`` is listed
# unprefixed as well: a hand-assembled state dict need not nest the vocab buffers under
# ``encoder.`` for them to reinterpret every embedding row.
_ENCODER_KEY_PREFIXES = ("encoder.", "cross_attn_layers.", "feature_vocab")


def _assert_decoder_only_source(
    model_data: Mapping[str, torch.Tensor], *, checkpoint_dir: str
) -> None:
    """Refuse a source checkpoint that already carries encoder / cross-attention keys.

    The converter's contract is a DECODER-ONLY source: it builds the encoder and the
    cross-attention layers FROM SCRATCH. ``strict=False`` on the load exists to tolerate
    those keys being **missing** from the source — never to tolerate a source that
    already **carries** them.

    This inspects the SOURCE state dict rather than ``load_state_dict``'s ``unexpected``
    list, and that is load-bearing: the target *is* an ``EncoderDecoderGPT``, so
    ``encoder.*`` and ``cross_attn_layers.*`` are **expected** keys — PyTorch consumes
    them in silence and reports ``unexpected == []``. A guard over ``unexpected`` would
    be vacuous against the exact failure this prevents.

    That failure: with ``assign=True`` the source's tensors REPLACE the freshly
    initialised encoder — including the persistent ``encoder.feature_vocab_ids`` buffer —
    while the meta's ``feature_vocab`` fingerprint is still written from the vocabulary
    this run resolved from its OWN config. The conversion "succeeds" and writes a
    checkpoint whose weights and metadata describe different models, which nothing
    notices until ``build_encdec_model`` refuses it much later.
    """
    offending = sorted(k for k in model_data if k.startswith(_ENCODER_KEY_PREFIXES))
    if not offending:
        return
    raise ModelWeightLoadError(
        f"source checkpoint {checkpoint_dir} is NOT decoder-only: it carries "
        f"{len(offending)} encoder/cross-attention key(s): {', '.join(offending)}. "
        "convert_decoder_to_encdec builds the encoder and cross-attention layers from "
        "scratch and requires a decoder-only (base/sft/rl) source. Loading these would "
        "silently overwrite the freshly initialised encoder — and its feature-vocabulary "
        "buffer — with the source's, while the meta records the vocabulary resolved from "
        "THIS run's config, producing a checkpoint whose weights and metadata describe "
        "different models. If the source is already an encoder-decoder checkpoint, it "
        "does not need converting."
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for decoder -> encoder-decoder conversion.

    Slice-12 audit P2 #12 (2026-05-05): the body of this function used
    to live at module-import scope, leaking Hydra's ``GlobalHydra``
    state to any importer (e.g., a test that imported helper functions
    from this file).  Wrapping in ``def main()`` plus
    ``with initialize_config_dir(...)`` makes the module import-safe.

    Args:
        argv: Override list (defaults to ``sys.argv[1:]``).  Tests can
            pass an explicit list to invoke the CLI without polluting
            global ``sys.argv``.
    """
    if argv is None:
        argv = sys.argv[1:]

    config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "serving")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="convert_decoder_to_encdec", overrides=argv)
    reject_feature_clusters_config(cfg, "convert_decoder_to_encdec")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    feature_selection_raw = OmegaConf.to_container(cfg.feature_selection, resolve=True)
    if not isinstance(feature_selection_raw, dict):
        raise TypeError(
            "convert_decoder_to_encdec feature_selection must resolve to a mapping, "
            f"got {type(feature_selection_raw).__name__}"
        )
    feature_selection = parse_feature_selection(cast(dict[str, Any], feature_selection_raw))
    assert_no_gradient_ranking(feature_selection, "convert_decoder_to_encdec live selection")
    # Resolve source checkpoint directory using nested path structure
    valid_sources = ("base", "sft", "rl")
    if cfg.source not in valid_sources:
        raise ValueError(f"source must be one of {list(valid_sources)}, got {cfg.source!r}")

    base_dir = get_base_dir()
    checkpoint_dir = get_checkpoint_dir(cfg.source, model_tag=cfg.model_tag)
    model_tag = cfg.model_tag
    if model_tag is None:
        model_tag = find_largest_model(os.path.join(base_dir, "base_checkpoints"))
        logger.info(f"No model tag provided, guessing: {model_tag}")

    step = cfg.step
    if step is None:
        step = find_last_step(checkpoint_dir)
    logger.info(f"Loading decoder checkpoint from {checkpoint_dir} step {step}")

    # Load decoder checkpoint
    device = torch.device("cpu")
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build decoder config
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    dec_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, dec_config)
    _fuse_qkv_weights(model_data)

    # Build encoder-decoder model
    enc_config = EncoderConfig(
        n_embd=cfg.encoder.n_embd,
        depth=cfg.encoder.depth,
        n_head=cfg.encoder.get("n_head"),
        n_kv_head=cfg.encoder.get("n_kv_head"),
        head_dim=cfg.encoder.get("head_dim", 64),
    )
    xattn_config = CrossAttentionConfig(
        frequency=cfg.cross_attention.frequency,
        n_head=cfg.cross_attention.get("n_head"),
        n_kv_head=cfg.cross_attention.get("n_kv_head"),
    )

    n_sae_features = cfg.n_sae_features
    num_buckets = cfg.num_buckets

    # The feature_mask is resolved ONCE, here, and drives three things that must never
    # disagree: the encoder's compacted dictionary (below), the checkpoint's mask leaf,
    # and the mask recorded in the meta. Resolving it twice invites exactly the drift
    # where the path claims one dictionary and the weights encode another.
    # None ⇒ no compaction, and every downstream use is a strict no-op (I2/I5).
    feature_mask = resolve_feature_mask(cfg, n_sae_features)
    feature_vocab = FeatureVocab.from_mask(feature_mask, n_sae_features)

    logger.info(f"Building EncoderDecoderGPT: {n_sae_features} SAE features, {num_buckets} buckets")
    logger.info(f"Encoder config: {enc_config}")
    logger.info(f"Cross-attention config: {xattn_config}")

    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            feature_vocab=feature_vocab,
        )
    # Logged only now that the model exists: the feature-indexed parameter saving is
    # computed from the RESOLVED encoder config and the real value-embedding count, and
    # a meta model carries both (nothing here reads data).
    logger.info(
        format_compaction_summary(
            model,
            no_vocab_reason=no_compaction_reason(feature_mask, n_sae_features),
        )
    )
    model.to_empty(device=device)
    # Deterministic RNG for reproducible conversion: the encoder and cross-attention
    # weights are freshly initialized (not loaded from the decoder checkpoint), and
    # without a fixed seed two runs of this script would produce bit-different
    # encdec checkpoints. See tests/nanochat/test_convert_decoder_to_encdec.py
    # (test_converter_is_idempotent).
    #
    # ORDER IS LOAD-BEARING: init_weights() must run BEFORE the strict=False load, or
    # it would overwrite the decoder weights we just loaded with fresh random init. It
    # also fills the encoder's vocab buffers (init_weights -> encoder.init_vocab_buffers),
    # which the decoder-only source checkpoint cannot supply.
    torch.manual_seed(0)
    model.init_weights()

    # The source must be DECODER-ONLY. strict=False below tolerates the encoder/xattn keys
    # being MISSING; it must not be read as tolerating a source that CARRIES them, which
    # assign=True would silently splice over the encoder we just initialised.
    _assert_decoder_only_source(model_data, checkpoint_dir=checkpoint_dir)

    # Load decoder weights (strict=False: encoder + cross-attention keys are missing in source)
    missing, unexpected = model.load_state_dict(model_data, strict=False, assign=True)
    logger.info(
        f"Loaded decoder weights: {len(missing)} new keys (encoder/xattn), {len(unexpected)} unexpected"
    )
    if unexpected:
        raise ModelWeightLoadError(
            f"{checkpoint_dir}: the source checkpoint carries {len(unexpected)} weight(s) "
            f"the encoder-decoder model has no slot for: {sorted(unexpected)}.\n"
            "`strict=False` is used here ONLY because a decoder-only source legitimately "
            "LACKS the encoder/cross-attention keys (those show up as `missing`, and are "
            "supplied by init_weights). An UNEXPECTED key is the opposite problem: the "
            "source has a weight nothing consumes, which means the two decoder "
            "architectures disagree. Converting anyway would silently produce a checkpoint "
            "whose corresponding weight is a random init, and nothing downstream would "
            "catch it.\n"
            "This was previously only a warning."
        )
    # assign=True REPLACES the persistent feature_vocab_ids buffer while the derived
    # (non-persistent) old_to_new map keeps whatever init_weights() produced. Every other
    # model load in the codebase verifies they still agree; this one did not, which is how
    # a state-dict vocab could drift from the feature_vocab written into the meta below.
    model.assert_vocab_consistent()

    # Save as encdec checkpoint
    output_tag = cfg.output_tag if cfg.output_tag else model_tag
    # feature_mask=None ⇒ append_mask_leaf returns the path unchanged
    # (I2: byte-identical legacy checkpoint path).
    output_dir = append_mask_leaf(
        os.path.join(
            base_dir,
            "conditional_encdec_checkpoints",
            output_tag,
            f"s{cfg.segment_size}",
            feature_selection_dirname(feature_selection),
        ),
        feature_mask,
    )
    logger.info(f"Saving encdec checkpoint to {output_dir} step {step}")

    # Build user_config matching what conditional_encdec_train.py saves
    user_config = OmegaConf.to_container(cfg, resolve=True)

    checkpoint_meta: dict = {
        "step": step,
        "val_bpb": meta_data.get("val_bpb", -1),
        "model_config": model_config_kwargs,
        "user_config": user_config,
    }
    # Record the RESOLVED mask, exactly as the training scripts do. Without this a
    # masked converted checkpoint composes a bl_/wl_ leaf it cannot justify:
    # assert_resume_feature_mask_matches demands the key, and dump_feature_mask reads
    # it — so the checkpoint could be neither resumed nor interrogated.
    if feature_mask is not None:
        checkpoint_meta["feature_mask_canonical"] = canonical_dict(feature_mask)
    # The compacted dictionary's fingerprint. The load-bearing record is the persistent
    # buffer inside the state dict; build_encdec_model cross-checks this mirror against
    # it and refuses the checkpoint if they disagree. Absent when not compacted ⇒ the
    # meta stays byte-identical to pre-#695.
    if feature_vocab is not None:
        checkpoint_meta["feature_vocab"] = feature_vocab.metadata_dict()

    save_checkpoint(
        output_dir,
        step,
        model.state_dict(),
        None,  # no optimizer state
        checkpoint_meta,
    )
    logger.info("Done! Converted checkpoint saved.")
    _ = get_tokenizer()  # warm-load to surface tokenizer-resolution errors
    logger.info(
        f"To use: python -m scripts.chat.steer_chat_cli source=encdec model_tag={output_tag} step={step}"
    )


if __name__ == "__main__":
    main()
