"""Shared residual-capture helpers for Stage 3a baselines.

DiffMean (Task 21) and LinearProbe (Task 22) both need to run the
target model on the prefix + output tokens of every training row, snag
the residual-stream activations at layer
``cfg.layer_hooks.diffmean_and_probe_layer``, and reduce the
continuation-token positions to a single per-row vector.

This module owns that machinery once so the two baselines don't
duplicate it. It also owns the label-column constants and the per-row
positive/negative filtering, which are identical across both baselines.

Why a raw ``register_forward_hook`` and not nnterp
--------------------------------------------------
The rest of the codebase's "capture residual activations at layer L"
entry point is ``steering/extract_features.py::_get_activations_nnterp``,
which uses ``nnterp.StandardizedTransformer`` + ``tracer.stop()`` for
~2x faster early-halt extraction. That path is the right one for
Gemma/GPT-2 — they are HuggingFace ``PreTrainedModel`` subclasses, and
``StandardizedTransformer`` (which inherits from nnsight's
``LanguageModel``) is designed to wrap HF models. Its rename table
(``nnterp.rename_utils.get_rename_dict``) is keyed on HF module path
conventions (``gpt_neox``, ``model.layers``, ``self_attention`` etc.).

The ``model`` argument to ``_capture_per_row_continuation`` is a
``nanochat.gpt.GPT`` (or ``nanochat.encoder_decoder_gpt.EncoderDecoderGPT``),
which is a plain ``torch.nn.Module`` with a bespoke architecture (rotary
embeds, QK-norm, ReLU-squared MLP, GQA, ResFormer value embeddings,
per-layer residual/x0 lambda scalars, sliding-window attention). It is
NOT an HF ``PreTrainedModel`` subclass and shares no config/interface
with ``transformers``. ``StandardizedTransformer.__init__`` delegates
to ``LanguageModel.__init__``, which dispatches via ``AutoModel``/HF
loading; handing it a nanochat ``GPT`` bypasses module-path detection
and produces a wrapper where ``layers_output`` / ``mlps`` / etc. are
unbound.

Until nnterp grows first-class support for custom (non-HF) ``nn.Module``
architectures, the natural capture path for nanochat models is a direct
``register_forward_hook`` on ``model.transformer.h[layer]``. This is
NOT reinvention of the wheel: it's the only option for a non-HF model,
and it is kept minimal and local to this single helper. The nnterp
path remains canonical for the HF-backed extraction pipeline in
``steering/extract_features.py`` (Gemma/GPT-2 SAE extraction).

Pieces exported:

  * ``_POS_LABEL`` / ``_NEG_LABEL`` — the ``content_class`` values
    identifying positive / negative training rows.
  * ``_ResidualCaptureHook`` — context manager installing a forward
    hook on ``model.transformer.h[layer]`` that records the block's
    output tensor on each forward pass. See the architecture note
    above for why this is a raw hook and not an nnterp tracer.
  * ``_capture_per_row_continuation`` — single-row driver that runs one
    forward pass and returns both the mean continuation activation and
    continuation-token activation matrix.
  * ``_row_side`` — maps a row dict to ``"pos"`` / ``"neg"`` / ``None``.
  * ``_filter_rows_for_concept`` — returns ``(positive_rows,
    negative_rows)`` for a given concept; both sides are filtered
    per-concept (positive and negative mode corpora are now symmetric,
    so there is no global-negatives ``concept_id == -1`` fallthrough).
  * Callers are responsible for their own per-row collection loops and
    invoke these helpers through this canonical module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.axbench_base.baselines.conditioning import EncoderConditioning
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_base.tokenizer_utils import tokenize_labeled_pair
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
)
from steering.eval.cluster_feature_contract import stream_hash_file_stable

# Aliases of the canonical content_class labels in ``...types`` (#559).
_POS_LABEL: str = CONCEPT_RICH_LABEL
_NEG_LABEL: str = CONCEPT_AVOIDING_LABEL


def _read_cluster_pinned_training_table(path: Path) -> tuple[pa.Table, str]:
    """Read the exact clustered corpus bytes bound into direction metadata.

    A clustered direction is valid only for the synth corpus that produced
    it. Hashing the path after ``pq.read_table`` leaves a replacement window:
    training can consume one file while metadata names another. Hash first,
    copy the file to immutable memory, require that copy to have the same
    digest and length, and parse only that copy. The no-cluster trainers
    deliberately do not call this helper so their legacy read path remains
    unchanged.
    """

    try:
        stable_digest = stream_hash_file_stable(path)
        payload = stable_digest.path.read_bytes()
    except Exception as exc:
        raise AxBenchBaseError(
            "Could not read a stable clustered synth training corpus at "
            f"{path}. Regenerate the clustered source under a fresh run tag."
        ) from exc

    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        len(payload) != stable_digest.byte_length
        or payload_sha256 != stable_digest.sha256
    ):
        raise AxBenchBaseError(
            "Clustered synth training corpus changed between stable hashing "
            f"and immutable read: {path}. Regenerate the clustered source "
            "under a fresh run tag."
        )
    try:
        table = pq.read_table(pa.BufferReader(payload))
    except Exception as exc:
        raise AxBenchBaseError(
            f"Could not parse clustered synth training corpus at {path}. "
            "Regenerate the clustered source under a fresh run tag."
        ) from exc
    return table, payload_sha256


@dataclass
class _ResidualCaptureHook:
    """Context manager that captures ``model.transformer.h[layer]`` output.

    On ``__enter__`` installs a forward hook on the block; on
    ``__exit__`` removes it (even if an exception propagates). Each
    forward pass overwrites ``self.last`` with a detached clone of the
    block's output tensor (shape ``(B, T, C)``) — callers that need
    multiple captures should read ``.last`` after each forward.

    Why capture instead of using ``register_residual_hook``: the Task-20
    helper MODIFIES the residual in-place by adding ``coefficient *
    direction``. DiffMean / LinearProbe training need to READ the
    residual, not modify it. The two use cases share the
    ``transformer.h[layer]`` attribute path but nothing else.

    Why a raw forward hook and not nnterp: see the module-level docstring
    above. tl;dr — nanochat's ``GPT`` is a custom ``nn.Module``, not an
    HF ``PreTrainedModel``, and ``nnterp.StandardizedTransformer`` is
    built to wrap HF models. Until nnterp supports custom architectures,
    ``register_forward_hook`` is the natural path for nanochat models.
    The HF-model capture path lives in
    ``steering/extract_features.py::_get_activations_nnterp``.
    """

    model: nn.Module
    layer: int
    last: t.Tensor | None = field(default=None, init=False)
    _handle: RemovableHandle | None = field(default=None, init=False)

    def __enter__(self) -> _ResidualCaptureHook:
        transformer = cast(nn.ModuleDict, self.model.transformer)
        blocks = cast(nn.ModuleList, transformer["h"])
        # Bounds-check before indexing. `nn.ModuleList` raises a bare
        # `IndexError: index 22 is out of range` that names neither the config
        # key nor the model, which is actively misleading here: the layer is
        # configured against an SAE on one model (gemma-3-4b-pt) while this hook
        # runs on another (the nanochat decoder), so the usual reading of "layer
        # 22" is not the one that applies.
        if not -len(blocks) <= self.layer < len(blocks):
            raise AxBenchBaseError(
                f"cfg.layer_hooks.diffmean_and_probe_layer={self.layer} is out "
                f"of range for this model, which has {len(blocks)} blocks "
                f"(valid: 0..{len(blocks) - 1}). This layer indexes the model "
                f"being hooked -- for the scalar baselines that is the *target* "
                f"model (e.g. the nanochat encdec decoder), not the model the "
                f"SAE features were extracted from. A config that copies the "
                f"SAE's layer (e.g. 22 for gemma-scope-2 layer_22) into this "
                f"field will land here."
            )
        block = blocks[self.layer]

        def _hook(
            module: nn.Module,
            inputs: tuple[t.Tensor, ...],
            output: t.Tensor,
        ) -> None:
            del module, inputs
            # Detach + clone so the tensor lives past the forward and
            # doesn't hold graph memory. Cast to float32 immediately so
            # downstream math (norm, mean) doesn't accumulate bf16 noise.
            self.last = output.detach().to(t.float32).clone()
            return None

        self._handle = block.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        # ``__enter__`` always sets ``self._handle`` — if it's None here,
        # the user constructed the dataclass directly and then called
        # ``__exit__`` without ``__enter__``. That's a programmer error;
        # raise loudly rather than silently no-op.
        if self._handle is None:
            raise AxBenchBaseError(
                "_ResidualCaptureHook.__exit__ called without a prior "
                "__enter__; use this class as a context manager."
            )
        self._handle.remove()
        self._handle = None


def _run_capture_forward(
    model: nn.Module,
    idx: t.Tensor,
    device: t.device,
    *,
    encoder_conditioning: EncoderConditioning | None = None,
) -> None:
    """Dispatch one forward pass for residual capture.

    A decoder-only model takes ``model(idx)``; there is nothing to condition.

    An :class:`EncoderDecoderGPT` REQUIRES ``encoder_conditioning`` -- the
    encoder-side tensors built from the feature bag that this (concept, mode)
    is steered with. This function will not synthesise conditioning of its own.

    #1249: it used to feed the all-masked ``build_no_feature_inputs`` tensors,
    so cross-attention produced NaN that ``nan_to_num`` folded to zero -- by its
    own former docstring, "effectively a decoder-only residual trajectory on the
    encdec backbone". The direction was therefore fit in a regime the model is
    never steered in, and then used to counter-steer a decoder that IS
    conditioned by a real bag. On the l22 clustered source that produced a
    NEGATIVE ``max_activation`` (-6404.02 for concept 2), which the #1228 scale
    guard correctly refused rather than silently flip the steering.

    There is deliberately NO fallback. Omitting the conditioning raises, because
    a fallback is exactly how the defect stayed invisible: it is undetectable in
    the output, and every run at ``magnitudes: [0.0]`` never exercises the
    counter direction at all.

    Callers must install :class:`_ResidualCaptureHook` before invoking
    this helper -- the return value is ignored.
    """
    if isinstance(model, EncoderDecoderGPT):
        if encoder_conditioning is None:
            raise AxBenchBaseError(
                "encoder-decoder residual capture requires encoder_conditioning "
                "(the feature bag this concept/mode is steered with); refusing to "
                "fit a direction under an unconditioned forward -- see #1249. The "
                "resulting direction would describe activations the model never "
                "produces when steered."
            )
        enc_fids, enc_mags, enc_mask = encoder_conditioning.as_tensors(device)
        seg_map = t.zeros(idx.shape, dtype=t.long, device=device)
        model(
            idx,
            targets=None,
            enc_feature_ids=enc_fids,
            enc_magnitudes=enc_mags,
            enc_mask=enc_mask,
            seg_map=seg_map,
        )
        return
    if encoder_conditioning is not None:
        raise AxBenchBaseError(
            "encoder_conditioning was supplied for a non-encoder-decoder model "
            f"({type(model).__name__}); a decoder-only capture has nothing to "
            "condition, so passing a bag here means the caller has the target "
            "models crossed"
        )
    model(idx)


def _capture_per_row_continuation(
    *,
    model: nn.Module,
    layer: int,
    tokenizer: Any,
    prefix_text: str,
    output_text: str,
    n_prefix_nanochat_tokens: int,
    encoder_conditioning: EncoderConditioning | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Run one forward pass and return mean and token continuation activations.

    Tokenizes the labeled pair once, prepends one owned BOS, and requires the
    exact overlapping-token boundary to equal the value committed in synth
    parquet. A mismatch is stale/corrupt training data, not a warning.

    Returns:
        A pair of float32 CPU tensors: the ``(C,)`` continuation mean and the
        ``(T, C)`` continuation-token activation matrix.

    Raises:
        AxBenchBaseError: If the committed synth row has malformed text,
            an invalid exact boundary, or otherwise produces no continuation
            tokens. These are artifact-contract violations and must never be
            converted into silently dropped training examples.
    """
    if not isinstance(prefix_text, str):
        raise AxBenchBaseError(
            "Synth row prefix_text must be a string during residual capture, "
            f"got {type(prefix_text).__name__}."
        )
    if not isinstance(output_text, str):
        raise AxBenchBaseError(
            "Synth row output_text must be a string during residual capture, "
            f"got {type(output_text).__name__}."
        )
    if not output_text:
        raise AxBenchBaseError(
            "Synth row has empty output_text during residual capture. "
            "Stage 1 must never commit an empty continuation; regenerate the "
            "synth parquet under a fresh run tag."
        )
    if (
        not isinstance(n_prefix_nanochat_tokens, int)
        or isinstance(n_prefix_nanochat_tokens, bool)
        or n_prefix_nanochat_tokens <= 0
    ):
        raise AxBenchBaseError(
            "Synth row n_prefix_nanochat_tokens must be a positive integer "
            "during residual capture, got "
            f"{n_prefix_nanochat_tokens!r}."
        )
    with profile_region("stage3.capture_tokenize"):
        full_tokens, slice_start = tokenize_labeled_pair(
            tokenizer, prefix_text, output_text
        )
    if slice_start != n_prefix_nanochat_tokens:
        raise AxBenchBaseError(
            "Synth labeled-pair boundary mismatch during residual capture: "
            f"stored={n_prefix_nanochat_tokens}, exact={slice_start}. "
            "Regenerate the synth parquet under a fresh run tag."
        )

    if len(full_tokens) <= slice_start:
        raise AxBenchBaseError(
            "Synth labeled-pair tokenization produced no continuation tokens "
            "during residual capture: "
            f"total_model_tokens={len(full_tokens)}, output_start={slice_start}. "
            "Regenerate the synth parquet under a fresh run tag."
        )

    device = next(model.parameters()).device
    idx = t.tensor([full_tokens], dtype=t.long, device=device)

    was_training = model.training
    model.eval()
    try:
        with _ResidualCaptureHook(model, layer=layer) as cap:
            with t.no_grad():
                with profile_region(
                    "stage3.capture_forward",
                    layer=layer,
                    token_count=len(full_tokens),
                ):
                    _run_capture_forward(
                        model, idx, device,
                        encoder_conditioning=encoder_conditioning,
                    )
            captured = cap.last
    finally:
        if was_training:
            model.train()

    if captured is None:  # pragma: no cover
        # Unreachable under normal operation: the forward hook installed
        # by _ResidualCaptureHook always fires on a successful forward
        # pass. Present as a defensive guard so a future refactor that
        # changes hook registration surfaces the breakage loudly.
        raise AxBenchBaseError(
            "Residual capture hook did not fire during forward pass; "
            f"model={type(model).__name__}, layer={layer}."
        )

    # captured: (B=1, T, C). Slice continuation tokens and mean along T.
    # The earlier `len(full_tokens) <= slice_start` check above rules
    # out the zero-slice case, so `continuation.shape[0] >= 1` here.
    # BYTE-IDENTITY CONSTRAINT (#1228, load-bearing): the mean MUST stay
    # computed on the capture device from the un-moved slice, exactly as
    # before — continuation.mean(dim=0).cpu(). Computing it on CPU from the
    # moved tensor changes reduction order and breaks direction byte-identity.
    continuation = captured[0, slice_start:, :]
    return continuation.mean(dim=0).cpu(), continuation.cpu()


def _row_side(row: dict[str, Any]) -> str | None:
    """Map a training row to ``"pos"`` / ``"neg"`` / ``None`` (neither).

    See ``diffmean`` / ``linear_probe`` module docstrings for the
    label-column convention.
    """
    label = row.get("content_class")
    if label == _POS_LABEL:
        return "pos"
    if label == _NEG_LABEL:
        return "neg"
    return None


def _filter_rows_for_concept(
    rows: list[dict[str, Any]], concept_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(positive_rows, negative_rows)`` for a given concept.

    Both sides are filtered per-concept: a row joins ``pos``/``neg`` only
    when its ``concept_id`` matches. Positive- and negative-mode corpora
    are now symmetric (positive mode emits a paired per-concept
    ``concept_avoiding`` row per rich row), so there is no shared
    global-negatives pool and no ``concept_id == -1`` fallthrough.
    """
    pos: list[dict[str, Any]] = []
    neg: list[dict[str, Any]] = []
    for row in rows:
        side = _row_side(row)
        if side is None:
            continue
        if int(row["concept_id"]) != concept_id:
            continue
        if side == "pos":
            pos.append(row)
        else:
            neg.append(row)
    return pos, neg
