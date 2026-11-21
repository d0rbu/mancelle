"""Canonical sampling helpers shared by the nanochat engine and steering inference paths.

Previously three near-identical implementations of "sample one next token from
logits with a non-finite guard" existed in nanochat/engine.py,
steering/eval/inference_engine.py, and steering/eval/multiseg_generator.py.
Issue #440 consolidates them here.
"""

import torch
import torch.nn.functional as F


def _assert_finite_logits(logits):
    """Raise immediately if generation receives NaN or Inf logits.

    A NaN logit silently sampled to argmax(NaN) = 0 would pollute eval results;
    the multinomial path can also degenerate. Surface non-finite logits as
    FloatingPointError so the failure points at the producer (numerically
    unstable forward, Muon-trained encdec checkpoint, etc.) rather than at
    the sampler.
    """
    finite = torch.isfinite(logits)
    if finite.all():
        return
    bad_count = int((~finite).sum().item())
    total = logits.numel()
    bad_positions = (~finite).nonzero(as_tuple=False)[:10].tolist()
    raise FloatingPointError(
        "sample_next_token received non-finite logits "
        f"({bad_count}/{total}; first positions={bad_positions})."
    )


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size).

    Returns a (B, 1) int64 tensor.

    * Non-finite logits raise FloatingPointError (see _assert_finite_logits).
    * temperature == 0 -> deterministic argmax.
    * Otherwise top-k filtering + softmax + multinomial sampling.
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    _assert_finite_logits(logits)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)
