# Modified by nano-gpt-pretrain-steer: preserve chat-template tokens unless
# they are the tokenizer's actual BOS token, support single-process
# inference without an initialized torch.distributed group, resolve the
# decoder layer list generically for GPT-NeoX-family targets (Pythia), which
# expose it as `.gpt_neox.layers` rather than the Llama/Qwen-style
# `.model.layers`, skip the chat-template suffix probe for base-model
# tokenizers that have none (every caller only consumes its result inside an
# `is_chat_model` branch), unwrap a decoder layer's forward-hook output
# whether or not it is tuple-wrapped -- every architecture tested (GPT-NeoX
# and Llama-family) now returns the hidden-states tensor directly rather
# than `(hidden_states, ...)`, so the previous unconditional `outputs[0]`
# silently indexed into the BATCH dimension instead of unwrapping a tuple,
# corrupting every caller's collected activations (DiffMean, LinearProbe,
# LoRA) with the transformers bump this package now requires for SmolLM3
# support -- and pin the wall-clock date SmolLM3's chat template embeds in
# its default system prompt (`{%- set today = strftime_now("%d %B %Y") -%}`,
# no override kwarg exists for it). Left unpinned, two stages of the SAME
# sweep launched on different calendar days render textually different
# (but semantically identical) prompts for the same concept -- caught when
# a c50 run's main (rendered 2026-08-27) and aux (rendered 2026-08-28)
# stages failed the merge's input-matching check. `render_jinja_template`'s
# compiled Jinja environment binds `strftime_now` to a closure over
# `chat_template_utils.datetime.now()`, which has no public override
# parameter, so this subclasses `datetime.datetime` with a frozen `.now()`
# and substitutes it into that one module's namespace; every other
# `datetime` consumer, and every template that does not call `strftime_now`
# (Qwen's and Pythia's have no chat template that does), is unaffected. See
# ../../VENDORED.md.
#################################
#
# Model utils.
#
#################################
import datetime as _datetime_module
import logging

import torch, einops
from torch import nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class _FrozenChatTemplateDatetime(_datetime_module.datetime):
    """`datetime.datetime` with `.now()` pinned for chat-template rendering."""

    _frozen_now = _datetime_module.datetime(2026, 8, 27, 12, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls._frozen_now


def _pin_chat_template_date() -> None:
    try:
        import transformers.utils.chat_template_utils as _chat_template_utils
    except ImportError:
        logger.warning(
            "transformers.utils.chat_template_utils not found -- the "
            "chat-template date pin did not apply. Any chat template that "
            "calls strftime_now() (e.g. SmolLM3's default system prompt) "
            "will render the live wall-clock date instead of a pinned one, "
            "risking non-reproducible prompts across a multi-day sweep."
        )
        return
    _chat_template_utils.datetime = _FrozenChatTemplateDatetime


_pin_chat_template_date()


def _disable_chat_template_reasoning() -> None:
    """Default enable_thinking=False for every template that knows the flag.

    SmolLM3 opens ~50% of its generations with <think> and never closes it
    inside the 128-token steering budget, so the judge grades a reasoning
    monologue with no answer in it. The steering prompt is assembled in many
    places (inference, train, prompt/reft models), so the default is bound once
    on the tokenizer class instead of at each call site. Templates that never
    mention the flag (Qwen's, Pythia's) get no kwarg and render byte-identically.
    An explicit enable_thinking=... from a caller still wins.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        logger.warning("PreTrainedTokenizerBase not found -- reasoning default not applied")
        return
    original = PreTrainedTokenizerBase.apply_chat_template
    if getattr(original, "_nano_no_reasoning", False):
        return

    def apply_chat_template(self, *args, **kwargs):
        template = kwargs.get("chat_template") or getattr(self, "chat_template", None) or ""
        if isinstance(template, str) and "enable_thinking" in template:
            kwargs.setdefault("enable_thinking", False)
        return original(self, *args, **kwargs)

    apply_chat_template._nano_no_reasoning = True
    PreTrainedTokenizerBase.apply_chat_template = apply_chat_template


_disable_chat_template_reasoning()


def _unwrap_chat_template_tokens(token_ids):
    """Unwrap a possible BatchEncoding from apply_chat_template(tokenize=True).

    As of transformers 5.16.1, apply_chat_template(tokenize=True) always
    returns a BatchEncoding (its return_dict parameter now defaults True),
    not a plain list of token ids. pyproject.toml's transformers constraint
    is a floor (>=4.53.0), not an exact pin, so an older resolved version
    without this default -- or any future caller passing an explicit
    return_dict=False -- can still hand back a plain list; accept both
    rather than assuming either unconditionally.
    """
    if hasattr(token_ids, "get") and "input_ids" in token_ids:
        return token_ids["input_ids"]
    return token_ids


def strip_leading_bos(token_ids, tokenizer):
    """Remove a leading token only when it is the tokenizer's actual BOS."""
    token_ids = _unwrap_chat_template_tokens(token_ids)
    bos_token_id = tokenizer.bos_token_id
    if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
        return token_ids[1:]
    return token_ids


def distributed_rank_or_zero():
    """Return the DDP rank, or zero when no process group is initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def strip_chat_template_tokens(token_ids, tokenizer, suffix_length=0):
    """Trim an emitted BOS and, when present, a known trailing suffix."""
    token_ids = strip_leading_bos(token_ids, tokenizer)
    if suffix_length > 0:
        return token_ids[:-suffix_length]
    return token_ids


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_model_continues(
    model, tokenizer, prompts, max_new_tokens, is_chat_model=True, batch_size=8, include_system_prompt=False, verbose=False
):
    """we ground examples with the model's original generation."""
    tokenizer.padding_side = "left"
    if is_chat_model:
        if include_system_prompt:
            def apply_chat_template(prompt):
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
                nobos = strip_leading_bos(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    ),
                    tokenizer,
                )
                return tokenizer.decode(nobos)
        else:
            def apply_chat_template(prompt):
                messages = [{"role": "user", "content": prompt}]
                nobos = strip_leading_bos(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    ),
                    tokenizer,
                )
                return tokenizer.decode(nobos)
        prompts = [apply_chat_template(prompt) for prompt in prompts]
    
    # Process prompts in batches
    all_generated_texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses", disable=not verbose):
        batch_prompts = prompts[i:i + batch_size]
        encoding = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = generated_ids[:, encoding.input_ids.shape[1]:]
        batch_generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_generated_texts.extend(batch_generated_texts)
    
    return all_generated_texts


def decoder_layers(model):
    """Return the decoder block list regardless of the base model's attribute name."""
    if getattr(model.config, "model_type", None) == "gpt_neox":
        return model.gpt_neox.layers
    return model.model.layers


def gather_residual_activations(model, target_layer, inputs):
  target_act = None
  def gather_target_act_hook(mod, inputs, outputs):
    nonlocal target_act # make sure we can modify the target_act from the outer scope
    target_act = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return outputs
  handle = decoder_layers(model)[target_layer].register_forward_hook(
      gather_target_act_hook, always_call=True)
  _ = model.forward(**inputs)
  handle.remove()
  return target_act


@torch.no_grad()
def set_decoder_norm_to_unit_norm(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."

    eps = torch.finfo(model.proj.weight.dtype).eps
    norm = torch.norm(model.proj.weight.data, dim=1, keepdim=True)
    model.proj.weight.data /= norm + eps


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."
    assert model.proj.weight.grad is not None  # keep pyright happy

    parallel_component = einops.einsum(
        model.proj.weight.grad,
        model.proj.weight.data,
        "d_out d_in, d_out d_in -> d_out",
    )
    model.proj.weight.grad -= einops.einsum(
        parallel_component,
        model.proj.weight.data,
        "d_out, d_out d_in -> d_out d_in",
    )


def calculate_l1_losses(latent, non_topk_latent, labels=None, mask=None):
    """
    Calculate L1 losses with masked mean.
    
    Parameters:
    - latent: latent representation, shape [batch_size, seq_len]
    - non_topk_latent: non-topk latent representation, shape [batch_size, seq_len]
    - labels: labels, shape [batch_size]
    - mask: long mask, shape [batch_size, seq_len]
    """
    if mask is None:
        mask = torch.ones_like(latent, dtype=torch.long)
    
    mask = mask.bool()
    
    valid_counts = mask.sum(dim=-1)  # [batch_size]
    eps = torch.finfo(latent.dtype).eps
    if non_topk_latent is not None:
        masked_non_topk_sum = (non_topk_latent * mask).sum(dim=-1)  # [batch_size]
        mean_non_topk = masked_non_topk_sum / (valid_counts + eps)
        l1_loss = mean_non_topk.mean() # mean across batch
    else:
        masked_sum = (latent * mask).sum(dim=-1)  # [batch_size]
        mean_all = masked_sum / (valid_counts + eps)
        l1_loss = mean_all.mean() # mean across batch
    return l1_loss


def get_prefix_length(tokenizer, common_prefix=None):
    if common_prefix is None:
        message_a = [{"role": "user", "content": "1"}]
        message_b = [{"role": "user", "content": "2"}]
        # apply_chat_template(tokenize=True) can return a BatchEncoding
        # rather than a plain list of token ids -- unwrap it, or zip()/len()
        # below would silently operate on its dict KEYS instead of the
        # actual token ids.
        tokens_a = _unwrap_chat_template_tokens(
            tokenizer.apply_chat_template(message_a, tokenize=True))
        tokens_b = _unwrap_chat_template_tokens(
            tokenizer.apply_chat_template(message_b, tokenize=True))
        print("Detecting sequence a:", tokens_a)
        print("Detecting sequence b:", tokens_b)
        prefix_length = 0
        for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
            if ta != tb:
                prefix_length = i
                break
    else:
        message = [{"role": "user", "content": common_prefix}]
        tokens = _unwrap_chat_template_tokens(tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True))
        prefix_length = len(tokens)
    return prefix_length


def get_suffix_length(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        # Base-model tokenizers with no chat template: every caller only
        # consumes this result inside an `is_chat_model` branch, so an
        # empty suffix is a safe, unused placeholder here.
        return 0, ""
    message_a = [{"role": "user", "content": "1"}]
    message_b = [{"role": "user", "content": "2"}]
    # apply_chat_template(tokenize=True) can return a BatchEncoding rather
    # than a plain list of token ids -- unwrap it so `reversed()` and
    # slicing below work.
    tokens_a = _unwrap_chat_template_tokens(
        tokenizer.apply_chat_template(message_a, tokenize=True))
    tokens_b = _unwrap_chat_template_tokens(
        tokenizer.apply_chat_template(message_b, tokenize=True))
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    return suffix_length, tokenizer.decode(tokens_a[-suffix_length:])
