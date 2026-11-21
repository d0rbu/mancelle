"""Exact full-context SAE assessments of original chats and assistant bodies.

Scoring has one fixed legacy RustBPE renderer, independent of target-model
conditioning. Character offsets select original content after joint extraction
tokenization; the extractor sees the complete causal history, once.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch as t

from steering.eval.axbench_base import extraction
from steering.eval.axbench_negative.source import (
    FeatureTarget,
    SourceCandidate,
    canonical_bytes,
    value_sha256,
)
from steering.eval.dense_feature_targets import (
    load_dense_extraction_stack,
    load_weight_free_gemma3_extraction_stack,
)
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import resolve_bos_token_id

# Bump these policies when rendering or scored-span semantics change.
SCORING_RENDERER = "rustbpe-legacy-original-chat-full-context-v1"
SPAN_POLICY = "positive-character-overlap-original-content-all-bos-excluded-v1"


class ChatQcError(ValueError):
    """A response cannot be measured under the declared extraction contract."""


@dataclass(frozen=True)
class ChatAssessment:
    status: str
    max_activation: float
    active_token_count: int
    scored_token_count: int
    input_token_count: int
    prepared_digest: str
    percentile: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"active", "inactive", "no_tokens"}:
            raise ChatQcError("unknown chat assessment status")
        for name in ("active_token_count", "scored_token_count", "input_token_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ChatQcError(f"{name} must be a nonnegative integer")
        if not 0 <= self.active_token_count <= self.scored_token_count <= self.input_token_count:
            raise ChatQcError("assessment token counts are inconsistent")
        if (
            isinstance(self.max_activation, bool)
            or not isinstance(self.max_activation, (int, float))
            or not math.isfinite(self.max_activation)
        ):
            raise ChatQcError("max_activation must be finite")
        if self.percentile is not None and (
            isinstance(self.percentile, bool)
            or not isinstance(self.percentile, (int, float))
            or not math.isfinite(self.percentile)
            or not 0 <= self.percentile <= 100
        ):
            raise ChatQcError("percentile must be finite in [0,100] or null")
        expected = (
            "no_tokens"
            if self.scored_token_count == 0
            else "active"
            if self.active_token_count
            else "inactive"
        )
        if self.status != expected or (self.max_activation > 0) != bool(self.active_token_count):
            raise ChatQcError("assessment status and raw activation disagree")
        if self.status == "no_tokens" and self.max_activation != 0:
            raise ChatQcError("no_tokens assessment must have zero maximum")
        if not isinstance(self.prepared_digest, str) or not self.prepared_digest:
            raise ChatQcError("prepared_digest must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChatAssessment:
        return cls(**dict(value))


@dataclass(frozen=True)
class RenderedChat:
    text: str
    content_spans: tuple[tuple[int, int], ...]
    native_input_ids: tuple[int, ...]


def render_scoring_prompt(candidate: SourceCandidate, render_tokenizer: Any) -> RenderedChat:
    """Prove spans against the actual uncapped checkpoint renderer.

    Building spans from source string lengths avoids ambiguous marker-like text
    and repeated substrings. The full rendering equality is a mandatory guard
    against future renderer changes and tokenizer normalization.
    """
    candidate.validate()
    messages = [dict(message) for message in candidate.messages]
    ids, _mask = render_tokenizer.render_conversation({"messages": messages}, max_tokens=None)
    if not ids or ids[0] != render_tokenizer.get_bos_token_id():
        raise ChatQcError("scoring renderer must prepend exactly the native BOS")
    ids = [*ids, render_tokenizer.encode_special("<assistant_start>")]
    actual = render_tokenizer.decode(ids[1:])
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    length = 0

    def append(text: str, *, original: bool = False) -> None:
        nonlocal length
        if original:
            spans.append((length, length + len(text)))
        pieces.append(text)
        length += len(text)

    system = messages.pop(0)["content"] if messages[0]["role"] == "system" else None
    for index, message in enumerate(messages):
        role = message["role"]
        append(render_tokenizer.decode([render_tokenizer.encode_special(f"<{role}_start>")]))
        if index == 0 and system is not None:
            append(system, original=True)
            append("\n\n")
        append(message["content"], original=True)
        append(render_tokenizer.decode([render_tokenizer.encode_special(f"<{role}_end>")]))
    append(render_tokenizer.decode([render_tokenizer.encode_special("<assistant_start>")]))
    expected = "".join(pieces)
    if actual != expected:
        raise ChatQcError("legacy renderer no longer preserves the declared original-content spans")
    return RenderedChat(actual, tuple(spans), tuple(ids))


def _encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Read exact character offsets without coercing malformed tokenizer output."""
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ChatQcError("exact full-context assessment requires valid character offsets") from exc
    if not isinstance(encoded, Mapping):
        raise ChatQcError("extraction tokenizer must return offset mappings")
    ids, offsets = encoded.get("input_ids"), encoded.get("offset_mapping")
    if not isinstance(ids, (list, tuple)) or not isinstance(offsets, (list, tuple)):
        raise ChatQcError("extraction tokenizer lacks explicit token/offset lists")
    if len(ids) != len(offsets):
        raise ChatQcError("extraction token and offset counts disagree")
    previous = (0, 0)
    validated_offsets = []
    for token_id, pair in zip(ids, offsets, strict=True):
        if type(token_id) is not int or token_id < 0:
            raise ChatQcError("invalid extraction token ID")
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(type(v) is not int for v in pair)
        ):
            raise ChatQcError("offsets must contain pairs of integers")
        start, end = pair
        if not 0 <= start <= end <= len(text) or start < previous[0] or end < previous[1]:
            raise ChatQcError("extraction offsets are out of bounds or nonmonotonic")
        previous = (start, end)
        validated_offsets.append(previous)
    return list(ids), validated_offsets


@dataclass(frozen=True)
class PreparedChat:
    token_ids: tuple[int, ...]
    score_mask: tuple[bool, ...]
    digest: str


class ChatScorer:
    """Production full-context extractor with explicit injectable loaded assets."""

    def __init__(
        self,
        *,
        model: Any,
        sae: Any,
        render_tokenizer: Any,
        layer: int,
        context_limit: int,
        batch_size: int = 1,
        activation_type: ActivationType = ActivationType.residual,
        identity: Mapping[str, Any],
    ) -> None:
        if type(layer) is not int or layer < 0:
            raise ChatQcError("layer must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ChatQcError("batch_size must be a positive integer")
        n_ctx = getattr(getattr(model, "cfg", None), "n_ctx", None)
        if type(n_ctx) is not int or n_ctx < 1:
            raise ChatQcError("extraction model must declare a positive cfg.n_ctx")
        if type(context_limit) is not int or not 1 <= context_limit <= n_ctx:
            raise ChatQcError(f"context_limit must lie within loaded model context {n_ctx}")
        self.model, self.sae, self.render_tokenizer = model, sae, render_tokenizer
        self.tokenizer = model.tokenizer
        self.layer, self.context_limit, self.batch_size = layer, context_limit, batch_size
        self.activation_type = activation_type
        self.hook_name = extraction._sae_hook_name(sae)
        self.n_features = extraction._infer_n_sae_features(sae)
        if self.n_features < 1:
            raise ChatQcError("SAE feature width must be positive")
        bos_id = resolve_bos_token_id(self.tokenizer)
        if bos_id is None:
            raise ChatQcError("extraction tokenizer must define a BOS ID")
        self.bos_id = bos_id
        self._identity = copy.deepcopy(dict(identity))
        self._identity.update(
            {
                "scoring_renderer": SCORING_RENDERER,
                "span_policy": SPAN_POLICY,
                "threshold": 0.0,
                "layer": layer,
                "hook_name": self.hook_name,
                "activation_type": activation_type.value,
                "context_limit": context_limit,
                "batch_size": batch_size,
                "n_features": self.n_features,
                "bos_id": self.bos_id,
            }
        )
        canonical_bytes(self._identity)

    @property
    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(self._identity)

    def prepare(self, candidate: SourceCandidate, response: str | None = None) -> PreparedChat:
        rendered = render_scoring_prompt(candidate, self.render_tokenizer)
        if response is not None and not isinstance(response, str):
            raise ChatQcError("assistant response must be a string")
        text = rendered.text if response is None else rendered.text + response
        spans = rendered.content_spans if response is None else ((len(rendered.text), len(text)),)
        ids, offsets = _encode_with_offsets(self.tokenizer, text)
        if len(ids) + 1 > self.context_limit:
            raise ChatQcError(
                f"extraction input has {len(ids) + 1} tokens, exceeds context_limit={self.context_limit}"
            )
        token_ids = (int(self.bos_id), *ids)
        score_mask = (
            False,
            *(
                token_id != self.bos_id
                and any(
                    start < span_end and end > span_start and start < end and span_start < span_end
                    for span_start, span_end in spans
                )
                for token_id, (start, end) in zip(ids, offsets, strict=True)
            ),
        )
        digest = value_sha256(
            {
                "renderer": SCORING_RENDERER,
                "span_policy": SPAN_POLICY,
                "text": text,
                "spans": spans,
                "token_ids": token_ids,
                "score_mask": score_mask,
            }
        )
        return PreparedChat(token_ids, score_mask, digest)

    def _assess(
        self,
        prepared: Sequence[PreparedChat],
        targets: Sequence[Sequence[FeatureTarget]],
    ) -> list[dict[int, ChatAssessment]]:
        if len(prepared) != len(targets):
            raise ChatQcError("prepared input and target populations differ")
        result: list[dict[int, ChatAssessment]] = []
        pending: list[int] = []
        for index, (item, item_targets) in enumerate(zip(prepared, targets, strict=True)):
            seen: set[int] = set()
            for target in item_targets:
                if target.feature_id in seen:
                    raise ChatQcError("duplicate target feature ID")
                if target.member_ids[-1] >= self.n_features:
                    raise ChatQcError("target member is outside the loaded SAE feature width")
                seen.add(target.feature_id)
            result.append({})
            if not any(item.score_mask):
                result[index] = {
                    target.feature_id: ChatAssessment(
                        "no_tokens",
                        0.0,
                        0,
                        0,
                        len(item.token_ids),
                        item.digest,
                    )
                    for target in item_targets
                }
            elif item_targets:
                pending.append(index)

        def reduce_item(acts: t.Tensor, batch_index: int) -> t.Tensor:
            index = pending[batch_index]
            item = prepared[index]
            extraction.BosPrependedActivations.from_forward(acts, list(item.token_ids))
            if acts.shape[1] != self.n_features:
                raise ChatQcError("extractor activation width disagrees with the declared SAE")
            if not bool(t.isfinite(acts).all()):
                raise ChatQcError("extractor returned nonfinite activations")
            mask = t.tensor(item.score_mask, dtype=t.bool, device=acts.device)
            for target in targets[index]:
                values = acts[:, list(target.member_ids)][mask]
                maximum = float(values.max().item())
                count = int((values > 0).any(dim=1).sum().item())
                result[index][target.feature_id] = ChatAssessment(
                    "active" if count else "inactive",
                    maximum,
                    count,
                    int(mask.sum().item()),
                    len(item.token_ids),
                    item.digest,
                )
            return t.empty(0)

        if pending:
            reduced = extraction._run_batched_extractor(
                model=self.model,
                sae=self.sae,
                token_id_lists=[list(prepared[i].token_ids) for i in pending],
                layer=self.layer,
                hook_name=self.hook_name,
                activation_type=self.activation_type,
                batch_size=self.batch_size,
                per_item_reduce=reduce_item,
            )
            if len(reduced) != len(pending) or any(not result[i] for i in pending):
                raise ChatQcError("extractor failed to assess the complete input population")
        return result

    def assess_prefix(
        self,
        candidate: SourceCandidate,
        targets: Sequence[FeatureTarget],
    ) -> dict[int, ChatAssessment]:
        return self._assess([self.prepare(candidate)], [targets])[0]

    def assess_completion(
        self,
        candidate: SourceCandidate,
        response: str,
        target: FeatureTarget,
    ) -> ChatAssessment:
        return self.assess_completions([(candidate, response, target)])[0]

    def assess_completions(
        self,
        items: Sequence[tuple[SourceCandidate, str, FeatureTarget]],
    ) -> list[ChatAssessment]:
        prepared = [self.prepare(candidate, response) for candidate, response, _target in items]
        assessments = self._assess(prepared, [[target] for _candidate, _response, target in items])
        return [
            row[target.feature_id]
            for row, (_candidate, _response, target) in zip(assessments, items, strict=True)
        ]


def module_weights_sha256(module: t.nn.Module) -> str:
    """Hash actual loaded tensors, retaining names, shapes and numerical dtypes."""
    digest = hashlib.sha256()
    state = module.state_dict()
    if not state:
        raise ChatQcError("cannot identify an extraction module without tensor state")
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, t.Tensor) or tensor.is_sparse:
            raise ChatQcError("extraction identity requires dense tensor state")
        digest.update(
            canonical_bytes({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        )
        byte_tensor = tensor.detach().cpu().contiguous().reshape(-1).view(t.uint8)
        digest.update(memoryview(byte_tensor.numpy()))
    return digest.hexdigest()


def tokenizer_sha256(tokenizer: Any) -> str:
    """Hash actual tokenizer content, not names or a few diagnostic probe strings."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        return value_sha256(
            {
                "backend": backend.to_str(),
                "special_tokens_map": tokenizer.special_tokens_map,
                "bos_token_id": resolve_bos_token_id(tokenizer),
            }
        )
    encoding = getattr(tokenizer, "enc", None)
    ranks = getattr(encoding, "_mergeable_ranks", None)
    specials = getattr(encoding, "_special_tokens", None)
    pattern = getattr(encoding, "_pat_str", None)
    if (
        not isinstance(ranks, dict)
        or not isinstance(specials, dict)
        or not isinstance(pattern, str)
    ):
        raise ChatQcError("tokenizer cannot expose a complete stable vocabulary identity")
    validated_ranks: list[tuple[bytes, int]] = []
    for piece, rank in ranks.items():
        if not isinstance(piece, bytes) or type(rank) is not int or rank < 0:
            raise ChatQcError(
                "tokenizer mergeable vocabulary requires byte pieces and integer ranks"
            )
        validated_ranks.append((piece, rank))
    if not validated_ranks or len({rank for _piece, rank in validated_ranks}) != len(
        validated_ranks
    ):
        raise ChatQcError("tokenizer mergeable ranks must be nonempty and unique")
    if any(
        not isinstance(token, str) or type(token_id) is not int or token_id < 0
        for token, token_id in specials.items()
    ):
        raise ChatQcError("tokenizer special vocabulary requires strings and integer IDs")
    digest = hashlib.sha256(
        canonical_bytes(
            {
                "pattern": pattern,
                "specials": specials,
                "bos_token_id": resolve_bos_token_id(tokenizer),
            }
        )
    )
    for piece, rank in sorted(validated_ranks, key=lambda item: item[1]):
        digest.update(canonical_bytes({"rank": rank, "bytes": piece.hex()}))
    return digest.hexdigest()


def _config_json(value: Any) -> Any:
    """Retain numerical architecture settings alongside actual tensor hashes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (t.dtype, t.device)):
        return str(value)
    if isinstance(value, Enum):
        return _config_json(value.value)
    if isinstance(value, (tuple, list)):
        return [_config_json(item) for item in value]
    if isinstance(value, Mapping):
        # HF configs contain integer-keyed id2label dictionaries. Preserve key
        # types instead of stringifying (which aliases e.g. 1 and "1"). Tag ALL
        # mappings so a user config cannot alias this canonical representation.
        pairs = []
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ChatQcError("extraction config keys must be strings or integers")
            pairs.append([{"type": type(key).__name__, "value": key}, _config_json(item)])
        return {"mapping": sorted(pairs, key=lambda pair: canonical_bytes(pair[0]))}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _config_json(to_dict())
    if hasattr(value, "__dict__") and not callable(value):
        return _config_json(vars(value))
    raise ChatQcError(f"cannot bind extraction config value {type(value).__name__}")


def _library_versions() -> dict[str, str | None]:
    libraries = {}
    for package in (
        "torch",
        "transformer-lens",
        "sae-lens",
        "transformers",
        "tokenizers",
        "tiktoken",
        "nnterp",
        "nnsight",
    ):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    return libraries


def extraction_runtime_identity(model: Any, sae: Any) -> dict[str, Any]:
    """Bind extraction behavior that tensor and tokenizer hashes do not capture."""
    nnterp = getattr(model, "nnterp_model", None)
    actual_model = nnterp.model if nnterp is not None else model
    return {
        "backend": "nnterp" if nnterp is not None else "transformerlens",
        "model_config_sha256": value_sha256(
            _config_json(getattr(actual_model, "config", model.cfg))
        ),
        "sae_config_sha256": value_sha256(_config_json(sae.cfg)),
        "torch_version": t.__version__,
        "libraries": _library_versions(),
        "numeric_settings": {
            "float32_matmul_precision": t.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": t.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": t.backends.cudnn.allow_tf32,
            "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
            "flash_sdp_enabled": t.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": t.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": t.backends.cuda.math_sdp_enabled(),
            "attention_implementation": getattr(
                getattr(actual_model, "config", None),
                "_attn_implementation",
                None,
            ),
        },
        "model_dtypes": sorted({str(p.dtype) for p in actual_model.parameters()}),
        "sae_dtypes": sorted({str(p.dtype) for p in sae.parameters()}),
    }


def build_chat_scorer(config: Mapping[str, Any], render_tokenizer: Any) -> ChatScorer:
    """Load the established real extraction factory and bind observed content.

    Existing model/SAE loaders do not expose immutable revision arguments. We
    therefore record their requested names separately, hash actual loaded
    tensors/tokenizers, and enforce any expected pins before returning a scorer.
    """
    model_name = config["model"]
    release = SAERelease(config["sae_release"])
    activation_type = ActivationType(config.get("activation_type", "residual"))
    width = SAEWidth(config["sae_width"])
    layer = config["layer"]
    if type(layer) is not int or layer < 0:
        raise ChatQcError("layer must be a nonnegative integer")
    resolved = resolve_sae(
        model_name,
        release,
        activation_type,
        layer,
        width,
        coerce_config_l0(release, config.get("sae_l0")),
    )
    device = str(config.get("device", "cuda"))
    if model_name.startswith("gemma-3"):
        model, sae = load_weight_free_gemma3_extraction_stack(
            sae_model=model_name,
            resolved_sae=resolved,
            device=device,
        )
    else:
        model, sae = load_dense_extraction_stack(
            sae_model=model_name,
            sae_layer=layer,
            resolved_sae=resolved,
            device=device,
        )
    nnterp = getattr(model, "nnterp_model", None)
    actual_model = nnterp.model if nnterp is not None else model
    actual_model.eval()
    sae.eval()
    if extraction._sae_hook_name(sae) != resolved.hook_name:
        raise ChatQcError("loaded SAE hook differs from the requested extraction site")
    if extraction._infer_n_sae_features(sae) != SAE_WIDTH_TO_NUM_FEATURES[width]:
        raise ChatQcError("loaded SAE width differs from the requested dictionary")
    hashes = {
        "model_sha256": module_weights_sha256(actual_model),
        "sae_sha256": module_weights_sha256(sae),
        "extraction_tokenizer_sha256": tokenizer_sha256(model.tokenizer),
        "render_tokenizer_sha256": tokenizer_sha256(render_tokenizer),
    }
    for key, actual in hashes.items():
        expected = config.get(f"expected_{key}")
        if expected is not None and (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or actual != expected
        ):
            raise ChatQcError(f"loaded {key} does not match its expected content pin")
    identity = {
        "format": "axbench-negative-extractor-v1",
        **hashes,
        "requested_model": model_name,
        "requested_sae_release": resolved.release,
        "requested_sae_id": resolved.sae_id,
        "observed_revision": None,
        "device": device,
        **extraction_runtime_identity(model, sae),
    }
    return ChatScorer(
        model=model,
        sae=sae,
        render_tokenizer=render_tokenizer,
        layer=layer,
        activation_type=activation_type,
        context_limit=config["context_limit"],
        batch_size=config.get("batch_size", 1),
        identity=identity,
    )
