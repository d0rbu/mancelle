"""Project soft suffixes with real clean prompt conditioning and ordinary decoding."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import torch as t

from nanochat.engine import KVCache
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.eval_adapter import (
    ContextProvider,
    SelfCondPromptSeed,
    TargetAwareEncoderDecoderEvalWrapper,
    seed_self_conditioning_from_prompt,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig, SelfCondRuntime
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_upstream.methods import TEMPERATURE, TOP_K, PreparedPromptContext
from steering.eval.inference_engine import InferenceEngine
from steering.eval.target_conditioning import SelfConditionedTarget, SingleEntryContextProvider


def project_suffix_prefix(
    tokenizer: Any,
    prompt: str,
    clean_tokens: Sequence[int],
    clean_seg_map: t.Tensor,
    seed_text: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Keep clean routing through a separator seam wholly within the final user slot."""

    clean_content = tuple(int(token) for token in clean_tokens[2:-2])
    separator_ids = tuple(int(token) for token in tokenizer.encode(prompt + "\n\n"))
    common = 0
    for clean_token, attacked_token in zip(clean_content, separator_ids, strict=False):
        if clean_token != attacked_token:
            break
        common += 1
    source_slot = int(clean_seg_map[0, -2].item())
    affected_clean_slots = clean_seg_map[0, 2 + common : -1]
    if bool(affected_clean_slots.ne(source_slot).any()):
        raise AxBenchAdvError("project GCG separator retokenizes outside the final real-user slot")
    prompt_prefix = (int(clean_tokens[0]), int(clean_tokens[1]), *separator_ids)
    prompt_prefix_seg = (
        *(int(slot) for slot in clean_seg_map[0, : 2 + common].tolist()),
        *([source_slot] * (len(separator_ids) - common)),
    )
    if len(prompt_prefix_seg) != len(prompt_prefix):
        raise AxBenchAdvError("project GCG attacked-prefix routing length mismatch")
    attacked_content = tuple(int(token) for token in tokenizer.encode(prompt + "\n\n" + seed_text))
    seed_ids = tuple(int(token) for token in tokenizer.encode(seed_text))
    if attacked_content != (*separator_ids, *seed_ids):
        raise AxBenchAdvError("project GCG seed changes the fixed separator tokenization")
    return prompt_prefix, prompt_prefix_seg, seed_ids, source_slot


def validate_project_soft_embeddings(
    embeddings: t.Tensor,
    modules: Sequence[tuple[str, Any]],
    token_count: int,
) -> None:
    """Validate master values and both native lookup and decoder compute casts."""

    width = sum(int(module.weight.shape[1]) for _name, module in modules)
    if (
        not isinstance(embeddings, t.Tensor)
        or embeddings.dtype is not t.float32
        or tuple(embeddings.shape) != (token_count, width)
        or token_count <= 0
        or not bool(t.isfinite(embeddings).all())
    ):
        raise AxBenchAdvError("project soft embeddings must be finite FP32 with shape (k, width)")
    offset = 0
    for name, module in modules:
        stop = offset + int(module.weight.shape[1])
        native = embeddings[:, offset:stop].detach().to(dtype=module.weight.dtype)
        if not bool(t.isfinite(native).all()) or not bool(
            t.isfinite(native.to(dtype=model_module.COMPUTE_DTYPE)).all()
        ):
            raise AxBenchAdvError(f"project soft embedding {name} overflows its native dtype")
        offset = stop


def replace_soft_embedding_rows(
    output: t.Tensor,
    embeddings: t.Tensor,
    starts: Sequence[int],
) -> t.Tensor:
    """Replace only suffix rows, retaining native lookup dtype and independent VJP roots."""

    if output.ndim != 3 or output.shape[0] != len(starts) or output.shape[2] != embeddings.shape[1]:
        raise AxBenchAdvError("project soft embedding lookup shape differs from the prepared span")
    count = embeddings.shape[0]
    if any(start < 0 or start + count > output.shape[1] for start in starts):
        raise AxBenchAdvError("project soft embedding span lies outside the decoder input")
    with t.no_grad():
        result = output.detach().clone()
        native = embeddings.detach().to(device=output.device, dtype=output.dtype)
        for row, start in enumerate(starts):
            result[row, start : start + count] = native
    return result


@contextmanager
def _soft_prefill_embeddings(
    modules: Sequence[tuple[str, Any]],
    embeddings: t.Tensor,
    input_ids: t.Tensor,
    suffix_start: int,
) -> Iterator[None]:
    """Inject once per lookup in this one actual prefix forward, then remove all hooks."""

    handles = []
    calls: dict[str, int] = {}
    offset = 0
    try:
        for name, module in modules:
            stop = offset + int(module.weight.shape[1])
            values = embeddings[:, offset:stop]
            offset = stop
            calls[name] = 0

            def inject(
                _module: Any,
                args: tuple[Any, ...],
                output: Any,
                *,
                name: str = name,
                values: t.Tensor = values,
            ) -> t.Tensor:
                if (
                    len(args) != 1
                    or not isinstance(args[0], t.Tensor)
                    or not t.equal(args[0], input_ids)
                ):
                    raise AxBenchAdvError(
                        "project soft lookup input differs from the attacked prefix"
                    )
                if not isinstance(output, t.Tensor):
                    raise AxBenchAdvError("project soft lookup returned a non-tensor")
                calls[name] += 1
                if calls[name] != 1:
                    raise AxBenchAdvError("project soft lookup ran more than once during prefill")
                return replace_soft_embedding_rows(output, values, (suffix_start,))

            handles.append(module.register_forward_hook(inject))
        yield
        if any(count != 1 for count in calls.values()):
            raise AxBenchAdvError("project soft prefill omitted an embedding lookup")
    finally:
        for handle in reversed(handles):
            handle.remove()


class ProjectSoftPromptWrapper(TargetAwareEncoderDecoderEvalWrapper):
    """Seed from clean text, replay the attacked prefix, then inherit normal state transitions."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        context_provider: ContextProvider,
        self_cond_config: SelfCondConfig,
        clean_token_ids: Sequence[int],
        attacked_token_ids: Sequence[int],
        raw_prompt: str,
        seed_text: str,
        embeddings: t.Tensor,
        embedding_modules: Sequence[tuple[str, Any]],
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=context_provider,
            self_cond_config=self_cond_config,
        )
        validate_project_soft_embeddings(
            embeddings, embedding_modules, len(tokenizer.encode(seed_text))
        )
        self._soft_clean_ids = tuple(clean_token_ids)
        self._soft_attacked_ids = tuple(attacked_token_ids)
        self._soft_raw_prompt = raw_prompt
        self._soft_seed_text = seed_text
        self._soft_embeddings = embeddings.detach().clone()
        self._soft_embedding_modules = tuple(embedding_modules)

    def _self_cond_prompt_seed(
        self, input_ids: t.Tensor, runtime: SelfCondRuntime, kv_cache: KVCache
    ) -> SelfCondPromptSeed:
        if (
            tuple(input_ids.shape) != (1, len(self._soft_attacked_ids))
            or tuple(input_ids[0].tolist()) != self._soft_attacked_ids
        ):
            raise AxBenchAdvError("project soft prefill requires the exact seeded prompt")
        scratch = KVCache(
            batch_size=1,
            num_heads=kv_cache.n_heads,
            seq_len=len(self._soft_clean_ids),
            head_dim=kv_cache.head_dim,
            num_layers=kv_cache.n_layers,
            device=input_ids.device,
            dtype=kv_cache.k_cache.dtype,
        )
        clean_ids = t.tensor([self._soft_clean_ids], dtype=t.long, device=input_ids.device)
        # InferenceEngine has bound the attacked IDs on the outer context. The
        # clean cache lookup must temporarily override those IDs as well.
        with self.conditioning_context(context_token_ids=[self._soft_clean_ids]):
            encoder_out, clean_seg_map = self._build_conditioning(clean_ids, kv_cache=scratch)
            prompt_seed = seed_self_conditioning_from_prompt(
                model=self.model,
                runtime=runtime,
                input_ids=clean_ids,
                encoder_out=encoder_out,
                seg_map=clean_seg_map,
                kv_cache=scratch,
            )
        del scratch
        prefix, prefix_seg, seed_ids, source_slot = project_suffix_prefix(
            self.tokenizer,
            self._soft_raw_prompt,
            self._soft_clean_ids,
            clean_seg_map,
            self._soft_seed_text,
        )
        tail = self._soft_clean_ids[-2:]
        if self._soft_attacked_ids != (*prefix, *seed_ids, *tail):
            raise AxBenchAdvError("project soft prompt changed the fixed prefix or assistant tail")
        attacked_prefix = input_ids[:, :-1]
        seg_map = t.tensor(
            [[*prefix_seg, *([source_slot] * (len(seed_ids) + len(tail) - 1))]],
            dtype=t.long,
            device=input_ids.device,
        )
        with _soft_prefill_embeddings(
            self._soft_embedding_modules, self._soft_embeddings, attacked_prefix, len(prefix)
        ):
            self.model(
                attacked_prefix,
                None,
                seg_map=seg_map,
                kv_cache=kv_cache,
                encoder_out=prompt_seed.state.encoder_out,
                encoder_strict=True,
                logits_to_keep=1,
            )
        if kv_cache.get_pos() != len(self._soft_attacked_ids) - 1:
            raise AxBenchAdvError(
                "project soft prefill cache did not stop before the assistant marker"
            )
        return prompt_seed


def generate_project_soft_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    self_cond_config: SelfCondConfig,
    attacked_token_ids: Sequence[int],
    raw_prompt: str,
    seed_text: str,
    embeddings: t.Tensor,
    embedding_modules: Sequence[tuple[str, Any]],
    seed: int,
    max_new_tokens: int,
    reuse_single_token_mask: bool = False,
) -> str:
    """Use the ordinary engine, sampling recipe, and stops with a scoped soft prefill."""

    provider = SingleEntryContextProvider(
        prepared.token_ids,
        prepared.segments,
        segment_mode=prepared.segment_mode,
        token_assignments=prepared.token_assignments,
    )
    wrapper = ProjectSoftPromptWrapper(
        model=model,
        tokenizer=tokenizer,
        context_provider=provider,
        self_cond_config=self_cond_config,
        clean_token_ids=prepared.token_ids,
        attacked_token_ids=attacked_token_ids,
        raw_prompt=raw_prompt,
        seed_text=seed_text,
        embeddings=embeddings,
        embedding_modules=embedding_modules,
    )
    engine = InferenceEngine(wrapper, tokenizer, model.get_device())
    mask_context = (
        model.reuse_single_token_packed_masks() if reuse_single_token_mask else nullcontext()
    )
    with mask_context:
        generated = list(
            engine.generate_stream(
                list(attacked_token_ids),
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                seed=seed,
                stop_token_ids={int(tokenizer.get_bos_token_id()), prepared.assistant_end_id},
            )
        )
    return tokenizer.decode(generated)
