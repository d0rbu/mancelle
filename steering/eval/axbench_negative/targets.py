"""Production native and encoder-decoder targets for natural-chat suppression."""

from __future__ import annotations

import gc
import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch as t

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from steering.conditional.encoder_decoder.self_conditioned import SelfCondScoreInterpolation
from steering.eval.axbench_base.baselines.hooks import register_residual_hook
from steering.eval.axbench_negative.admission import conditioning_identity
from steering.eval.axbench_negative.api import strict_object
from steering.eval.axbench_negative.qc import (
    extraction_runtime_identity,
    module_weights_sha256,
    tokenizer_sha256,
)
from steering.eval.axbench_upstream import methods as project
from steering.eval.axbench_upstream.data import (
    load_metadata,
    select_training_pairs,
    select_training_texts,
)
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    render_fit_row,
    template_prefix_length,
)
from steering.eval.axbench_upstream.native_baselines import (
    CAPTURE_PREFIX_LENGTH,
    capture_native_post_prefix_activations,
    resolve_native_model_contract,
)
from steering.eval.axbench_upstream.native_decoder import (
    NativeDecoderAdapter,
    canonical_bytes,
    native_runtime_settings,
)
from steering.eval.chat_generation import ChatGenerationResult, render_messages_for_chat_completion
from steering.eval.dense_feature_targets import load_weight_free_gemma3_extraction_stack
from steering.immutable_io import install_bytes_once, stream_hash_file_descriptor_bound
from steering.refined_types import UnitInterval
from steering.safetensors_io import deterministic_safetensors_bytes, load_safetensors_file_snapshot

METHODS = {"native": ("DiffMean", "LinearProbe"), "encdec": ("SelfConditioning",)}
# Bump the relevant policy for changes to fitting or generation semantics.
TARGET_POLICY = "natural-chat-negative-target-v1"
FIT_POLICY = "natural-chat-negative-canonical-train-fit-v1"


def file_sha256(path: Path) -> str:
    return stream_hash_file_descriptor_bound(Path(path).resolve()).sha256


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _write(path: Path, value: Any) -> None:
    install_bytes_once(canonical_bytes(value), path, surface="negative AxBench target artifact")


def validate_methods(family: str, methods: Sequence[str]) -> tuple[str, ...]:
    methods = tuple(methods)
    if family not in METHODS or not methods or len(set(methods)) != len(methods):
        raise ValueError("negative target requires one family and distinct supported methods")
    if any(method not in METHODS[family] for method in methods):
        raise ValueError(f"unsupported negative method for {family}: {methods!r}")
    return methods


def _training_contract(config: Mapping[str, Any], groups: Sequence[Any]) -> tuple[dict[str, Any], dict[int, int]]:
    root = Path(config["prepared_dir"]).resolve()
    observed = {
        "metadata_sha256": file_sha256(root / "generate/metadata.jsonl"),
        "data_sha256": file_sha256(root / "generate/train_data.parquet"),
    }
    if any(config.get(key) != value for key, value in observed.items()):
        raise ValueError("canonical TRAIN data/metadata hashes differ from the declared fit inputs")
    prefix = config.get("ref_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("https://www.neuronpedia.org/") or prefix.endswith("/"):
        raise ValueError("training.ref_prefix must name the exact Neuronpedia model/SAE source")
    metadata = load_metadata(root / "generate/metadata.jsonl")
    rows = pq.read_table(root / "generate/train_data.parquet", columns=["input", "output"]).to_pylist()
    train_histories = set()
    for row in rows:
        if not isinstance(row["input"], str) or not isinstance(row["output"], str):
            raise ValueError("canonical TRAIN rows must contain text input and output")
        messages = [{"role": "user", "content": row["input"]}]
        train_histories.add(digest(messages))
        train_histories.add(digest([*messages, {"role": "assistant", "content": row["output"]}]))
    for group in groups:
        if any(digest(list(group.messages[:cut])) in train_histories for cut in range(1, len(group.messages) + 1)):
            raise ValueError("canonical TRAIN conversation/prefix overlaps frozen evaluation history")
    if (file_sha256(root / "generate/metadata.jsonl") != observed["metadata_sha256"]
        or file_sha256(root / "generate/train_data.parquet") != observed["data_sha256"]):
        raise ValueError("canonical TRAIN inputs changed while validating")
    mapping: dict[int, int] = {}
    concepts = []
    for group in groups:
        matches = [index for index, row in enumerate(metadata)
                   if row.get("ref") == f"{prefix}/{group.feature_id}" and row.get("concept") == group.description]
        if len(matches) != 1:
            raise ValueError(f"feature {group.feature_id} does not map uniquely to canonical TRAIN concept/SAE metadata")
        previous = mapping.setdefault(group.feature_id, matches[0])
        if previous != matches[0]:
            raise ValueError("one feature maps to conflicting TRAIN concepts")
    for feature_id, concept_id in sorted(mapping.items()):
        concepts.append({"feature_id": feature_id, "concept_id": concept_id, "metadata": metadata[concept_id]})
    return {**observed, "ref_prefix": prefix, "concepts": concepts}, mapping


def _fit_cache(
    root: Path, identity: Mapping[str, Any], build: Callable[[], Mapping[str, FittedDirection]],
) -> tuple[dict[str, FittedDirection], str]:
    """One small authenticated vector cache, shared by both target families."""
    _write(root / "identity.json", identity)
    receipt_path, tensor_path = root / "manifest.json", root / "vectors.safetensors"
    if not receipt_path.exists():
        fitted = dict(build())
        if not fitted:
            raise ValueError("negative fit cache cannot publish an empty vector inventory")
        tensors = {key: value.direction.detach().to(device="cpu", dtype=t.float32).contiguous()
                   for key, value in fitted.items()}
        if any(value.ndim != 1 or value.numel() == 0 or not bool(t.isfinite(value).all()) for value in tensors.values()):
            raise ValueError("negative fit vectors must be finite nonempty float32 vectors")
        payload = deterministic_safetensors_bytes(tensors, metadata={"identity_sha256": digest(identity), "format": FIT_POLICY})
        install_bytes_once(payload, tensor_path, surface="negative AxBench fit vectors")
        receipt = {"format": FIT_POLICY, "identity_sha256": digest(identity), "sha256": file_sha256(tensor_path),
                   "vectors": {key: {"max_activation": value.max_activation, "metadata": value.metadata}
                               for key, value in fitted.items()}}
        _write(receipt_path, receipt)
    receipt_snapshot = stream_hash_file_descriptor_bound(receipt_path, retain_payload=True)
    assert receipt_snapshot.payload is not None
    receipt = strict_object(receipt_snapshot.payload.decode("utf-8"))
    if receipt.get("format") != FIT_POLICY or receipt.get("identity_sha256") != digest(identity):
        raise ValueError("negative fit receipt identity differs")
    if file_sha256(tensor_path) != receipt.get("sha256"):
        raise ValueError("negative fit tensor hash differs")
    tensors, metadata = load_safetensors_file_snapshot(tensor_path)
    if file_sha256(tensor_path) != receipt["sha256"] or metadata != {"identity_sha256": digest(identity), "format": FIT_POLICY}:
        raise ValueError("negative fit tensor identity changed or differs")
    inventory = receipt.get("vectors")
    if not isinstance(inventory, dict) or set(inventory) != set(tensors) or not tensors:
        raise ValueError("negative fit vector inventory differs")
    result = {}
    for key, tensor in tensors.items():
        scale = inventory[key].get("max_activation")
        if tensor.dtype is not t.float32 or tensor.ndim != 1 or tensor.numel() == 0 or not bool(t.isfinite(tensor).all()):
            raise ValueError("invalid cached negative fit tensor")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError("invalid cached positive calibration")
        result[key] = FittedDirection(tensor, float(scale), inventory[key]["metadata"])
    if file_sha256(receipt_path) != receipt_snapshot.sha256:
        raise ValueError("negative fit receipt changed while loading")
    return result, receipt_snapshot.sha256


@contextmanager
def negative_direction(model: Any, *, layer: int, fit: FittedDirection, magnitude: float) -> Iterator[None]:
    """Apply the calibrated negative residual direction, removing it on every exit."""
    if isinstance(magnitude, bool) or not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("suppression magnitude must be finite and nonnegative")
    if magnitude == 0:
        yield
        return
    coefficient = -magnitude * fit.max_activation
    if not math.isfinite(coefficient):
        raise ValueError("calibrated suppression coefficient must be finite")
    handle = register_residual_hook(model, layer, fit.direction, coefficient)
    try:
        yield
    finally:
        handle.remove()


def suppression_policy(setup: Any, direction: t.Tensor | None, magnitude: float) -> SelfCondScoreInterpolation:
    if isinstance(magnitude, bool) or not math.isfinite(magnitude) or not 0 <= magnitude <= 6:
        raise ValueError("SelfConditioning suppression magnitude must be in [0,6]")
    width = setup.n_sae_features if setup.feature_vocab is None else setup.feature_vocab.size
    scores = t.zeros(width, dtype=t.float32) if magnitude == 0 else -direction if direction is not None else None
    if scores is None or scores.shape != (width,):
        raise ValueError("SelfConditioning suppression requires the exact feature-head vector")
    return SelfCondScoreInterpolation(target_scores=scores, negative_selection=setup.output_negative_selection,
                                     weight=UnitInterval.parse(magnitude / 6))


def _admission(bank: Any, family: str, checkpoint: Mapping[str, Any], *, max_new_tokens: int, sequence_len: int) -> None:
    targets = bank.identity.get("targets", [])
    matches = [entry for entry in targets if entry.get("family") == family]
    if len(matches) != 1:
        raise ValueError("frozen bank must bind exactly one admission contract for the target family")
    expected = matches[0]
    keys = ("model_sha256", "meta_sha256", "tokenizer_sha256")
    if any(checkpoint[key] != expected.get(key) for key in keys) or expected.get("sequence_len") != sequence_len:
        raise ValueError("target checkpoint/tokenizer differs from frozen source admission")
    budget = expected.get("max_new_tokens")
    if type(budget) is not int or type(max_new_tokens) is not int or not 0 < max_new_tokens <= budget:
        raise ValueError("evaluation generation budget exceeds frozen source admission")


class NativeNegativeTarget:
    family = "native"

    def __init__(self, config: Mapping[str, Any], bank: Any, groups: Sequence[Any], *, max_new_tokens: int):
        self.methods = validate_methods(self.family, config["methods"])
        training, self.concepts = _training_contract(config["training"], groups)
        self.adapter = NativeDecoderAdapter(Path(config["checkpoint_dir"]), config["step"], config["device"], Path(config["tokenizer_path"]))
        geometry = {key: int(getattr(self.adapter.model.config, key)) for key in ("n_layer", "n_embd", "sequence_len")}
        self.contract = resolve_native_model_contract(geometry)
        if self.contract.parameter_count is not None and sum(parameter.numel() for parameter in self.adapter.model.parameters()) != self.contract.parameter_count:
            raise ValueError("native checkpoint parameter count differs from reviewed architecture")
        if template_prefix_length(self.adapter.tokenizer) != CAPTURE_PREFIX_LENGTH:
            raise ValueError("native TRAIN capture prefix differs from the reviewed fit contract")
        _admission(bank, self.family, self.adapter.identity, max_new_tokens=max_new_tokens, sequence_len=geometry["sequence_len"])
        for group in groups:
            if len(self.adapter.prompt_messages(group.messages)) + max_new_tokens > geometry["sequence_len"]:
                raise ValueError("complete bank conversation exceeds native context admission")
        fit_identity = {"format": FIT_POLICY, "family": self.family, "target": self.adapter.identity,
                        "training": training, "methods": list(self.methods), "seed": config["fit_seed"],
                        "geometry": asdict(self.contract), "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
                        "runtime": native_runtime_settings(full_prefix_device=t.device(config["device"]))}

        def build() -> dict[str, FittedDirection]:
            pairs = select_training_pairs(Path(config["training"]["prepared_dir"]), max_concepts=max(self.concepts.values()) + 1)
            result = {}
            for feature_id, concept_id in sorted(self.concepts.items()):
                classes = []
                for values in (pairs[concept_id].positive, pairs[concept_id].negative):
                    acts = []
                    for input_text, output_text in values:
                        row = render_fit_row(self.adapter.tokenizer, input_text=input_text, output_text=output_text,
                                             sequence_len=geometry["sequence_len"], surface="negative canonical TRAIN")
                        acts.append(capture_native_post_prefix_activations(self.adapter.model, row,
                                    layer=self.contract.intervention_layer, prefix_length=CAPTURE_PREFIX_LENGTH))
                    classes.append(acts)
                positive, negative = classes
                for method in self.methods:
                    result[f"f{feature_id}_{method}"] = (fit_diffmean_direction(positive, negative,
                        intervention_layer=self.contract.intervention_layer, surface="negative canonical TRAIN") if method == "DiffMean"
                        else fit_probe_direction(positive, negative, intervention_layer=self.contract.intervention_layer,
                            surface="negative canonical TRAIN", seed=config["fit_seed"], concept_id=concept_id))
            if _training_contract(config["training"], groups)[0] != training:
                raise ValueError("canonical TRAIN inputs changed during fitting")
            return result

        self.fits, fit_sha = _fit_cache(Path(config["fit_dir"]), fit_identity, build)
        expected_keys = {f"f{feature}_{method}" for feature in self.concepts for method in self.methods}
        if set(self.fits) != expected_keys or any(fit.direction.shape != (geometry["n_embd"],) for fit in self.fits.values()):
            raise ValueError("native fit inventory/width differs from declared methods and concepts")
        self.identity = {"policy": TARGET_POLICY, "family": self.family, "checkpoint": self.adapter.identity,
                         "fit_manifest_sha256": fit_sha, "sequence_len": geometry["sequence_len"],
                         "temperature": 1.0, "top_k": 50, "methods": list(self.methods)}

    def generate(self, group: Any, *, method: str, magnitude: float, seed: int,
                 selection_seed: int, max_new_tokens: int) -> ChatGenerationResult:
        if method == "Control":
            if magnitude != 0:
                raise ValueError("Control requires magnitude zero")
            return self.adapter.generate_messages(group.messages, seed=seed, max_new_tokens=max_new_tokens)
        if method not in self.methods:
            raise ValueError("unsupported native negative method")
        with negative_direction(self.adapter.model, layer=self.contract.intervention_layer,
                                fit=self.fits[f"f{group.feature_id}_{method}"], magnitude=magnitude):
            return self.adapter.generate_messages(group.messages, seed=seed, max_new_tokens=max_new_tokens)


class EncdecNegativeTarget:
    family = "encdec"

    def __init__(self, config: Mapping[str, Any], bank: Any, groups: Sequence[Any], *, max_new_tokens: int):
        self.methods = validate_methods(self.family, config["methods"])
        training, self.concepts = _training_contract(config["training"], groups)
        meta_path = Path(config["meta_path"]).resolve()
        self.setup = project.load_method_setup(meta_path, Path(config["feature_statistics_path"]),
            pretrain_meta_path=None if config.get("pretrain_meta_path") is None else Path(config["pretrain_meta_path"]))
        tokenizer_path = Path(config["tokenizer_path"]).resolve()
        if tokenizer_path != (Path(get_base_dir()) / "tokenizer/tokenizer.pkl").resolve():
            raise ValueError("encoder-decoder tokenizer must equal NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl")
        checkpoint = {"meta_sha256": file_sha256(meta_path),
                      "model_sha256": file_sha256(meta_path.parent / f"model_{self.setup.meta['step']:06d}.pt"),
                      "tokenizer_sha256": file_sha256(tokenizer_path)}
        sequence_len = self.setup.meta["model_config"]["sequence_len"]
        _admission(bank, self.family, checkpoint, max_new_tokens=max_new_tokens, sequence_len=sequence_len)
        self.tokenizer = get_tokenizer()
        prompts = {}
        for group in groups:
            prompt = render_messages_for_chat_completion(self.tokenizer, group.messages)
            if len(prompt.token_ids) + max_new_tokens > sequence_len:
                raise ValueError("complete bank conversation exceeds encoder-decoder context admission")
            prompts.setdefault(tuple(prompt.token_ids), prompt)
        fit_identity = {"format": FIT_POLICY, "family": self.family, "target": checkpoint, "training": training,
                        "statistics_sha256": file_sha256(Path(config["feature_statistics_path"])),
                        "pretrain_meta_sha256": None if config.get("pretrain_meta_path") is None else file_sha256(Path(config["pretrain_meta_path"])),
                        "extraction_batch_size": config["extraction_batch_size"], "methods": list(self.methods)}
        if self.setup.sae_model.startswith("gemma-3"):
            extraction_model, sae = load_weight_free_gemma3_extraction_stack(
                sae_model=self.setup.sae_model, resolved_sae=self.setup.resolved_sae,
                device=config["device"],
            )
        else:
            extraction_model, sae = project.load_extraction_stack(self.setup, config["device"])
        try:
            conditioning = conditioning_identity(self.setup, extraction_model.tokenizer, int(extraction_model.cfg.n_ctx),
                statistics_path=Path(config["feature_statistics_path"]),
                pretrain_meta_path=None if config.get("pretrain_meta_path") is None else Path(config["pretrain_meta_path"]))
            admitted_conditioning = next(item["conditioning"] for item in bank.identity["targets"] if item["family"] == "encdec")
            if conditioning != admitted_conditioning:
                raise ValueError("encoder-decoder conditioning differs from frozen source admission")
            fit_identity["conditioning"] = conditioning
            nnterp = getattr(extraction_model, "nnterp_model", None)
            actual_model = nnterp.model if nnterp is not None else extraction_model
            fit_identity["extraction_assets"] = {
                "requested_model": self.setup.sae_model, "layer": self.setup.sae_layer,
                "model_sha256": module_weights_sha256(actual_model), "sae_sha256": module_weights_sha256(sae),
                "tokenizer_sha256": tokenizer_sha256(extraction_model.tokenizer),
                "device_type": t.device(config["device"]).type,
                **extraction_runtime_identity(extraction_model, sae),
            }
            def build() -> dict[str, FittedDirection]:
                examples = select_training_texts(Path(config["training"]["prepared_dir"]), max_concepts=max(self.concepts.values()) + 1)
                selected = {concept: examples[concept] for concept in sorted(set(self.concepts.values()))}
                fitted = project.fit_concept_dense_diffmeans(self.setup, extraction_model, sae, selected,
                    extraction_batch_size=config["extraction_batch_size"])
                if _training_contract(config["training"], groups)[0] != training:
                    raise ValueError("canonical TRAIN inputs changed during fitting")
                return {f"f{feature}_SelfConditioning": FittedDirection(fitted[concept], 1.0,
                        {"concept_id": concept, "units": "percentile_rank_difference_div_100"})
                        for feature, concept in self.concepts.items()}
            self.fits, fit_sha = _fit_cache(Path(config["fit_dir"]), fit_identity, build)
            width = self.setup.n_sae_features if self.setup.feature_vocab is None else self.setup.feature_vocab.size
            if set(self.fits) != {f"f{feature}_SelfConditioning" for feature in self.concepts} or any(
                fit.direction.shape != (width,) or bool((fit.direction.abs() > 1).any()) for fit in self.fits.values()
            ):
                raise ValueError("encoder-decoder fit inventory/feature vocabulary differs")
            self.prepared = {tokens: project.prepare_prompt_context(self.setup, extraction_model, sae, self.tokenizer,
                token_ids=tokens, assistant_end_id=prompt.assistant_end_id) for tokens, prompt in prompts.items()}
        finally:
            extraction_model = None
            sae = None
            actual_model = None
            nnterp = None
            gc.collect()
            if config["device"].startswith("cuda"):
                t.cuda.empty_cache()
        self.model, self.tokenizer = project.load_target_model(self.setup, config["device"])
        if (int(self.model.dec_config.sequence_len) != sequence_len or file_sha256(meta_path) != checkpoint["meta_sha256"]
            or file_sha256(tokenizer_path) != checkpoint["tokenizer_sha256"]
            or file_sha256(meta_path.parent / f"model_{self.setup.meta['step']:06d}.pt") != checkpoint["model_sha256"]):
            raise ValueError("encoder-decoder checkpoint changed while loading")
        self.identity = {"policy": TARGET_POLICY, "family": self.family, "checkpoint": checkpoint,
                         "feature_prediction": asdict(self.model.feature_pred_config),
                         "fit_manifest_sha256": fit_sha, "sequence_len": sequence_len, "temperature": 1.0,
                         "top_k": 50, "methods": list(self.methods), "segment_mode": self.setup.segment_mode.value,
                         "turn_cadence": self.setup.turn_self_cond_segment_size_rustbpe,
                         "runtime": native_runtime_settings(full_prefix_device=t.device(config["device"]))}

    def generate(self, group: Any, *, method: str, magnitude: float, seed: int,
                 selection_seed: int, max_new_tokens: int) -> ChatGenerationResult:
        if method not in ("Control", *self.methods) or (method == "Control" and magnitude != 0):
            raise ValueError("unsupported encoder-decoder negative method/control")
        direction = None if method == "Control" else self.fits[f"f{group.feature_id}_SelfConditioning"].direction
        policy = suppression_policy(self.setup, direction, magnitude)
        tokens = tuple(render_messages_for_chat_completion(self.tokenizer, group.messages).token_ids)
        return project.generate_self_conditioning_result(self.setup, self.model, self.tokenizer,
            self.prepared[tokens], policy, selection_seed=selection_seed, seed=seed, max_new_tokens=max_new_tokens)


def build_target(config: Mapping[str, Any], bank: Any, groups: Sequence[Any], *, max_new_tokens: int) -> Any:
    family = config["family"]
    validate_methods(family, config["methods"])
    if config["training"]["ref_prefix"] != bank.identity.get("feature_ref_prefix"):
        raise ValueError("canonical TRAIN SAE namespace differs from the frozen bank")
    if not groups:
        raise ValueError("cannot evaluate an empty negative bank split")
    cache_root = Path(config["fit_dir"]).resolve()
    protected = [bank.root.resolve(), Path(config["training"]["prepared_dir"]).resolve()]
    protected.append(Path(config["checkpoint_dir"]).resolve() if family == "native" else Path(config["meta_path"]).resolve().parent)
    if any(cache_root == path or cache_root.is_relative_to(path) or path.is_relative_to(cache_root) for path in protected):
        raise ValueError("fit cache must be disjoint from frozen source, TRAIN inputs and checkpoints")
    if family == "native":
        return NativeNegativeTarget(config, bank, groups, max_new_tokens=max_new_tokens)
    return EncdecNegativeTarget(config, bank, groups, max_new_tokens=max_new_tokens)
