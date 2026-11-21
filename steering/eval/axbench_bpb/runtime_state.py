"""Lazy, reference-counted resource manager for axbench-bpb stages.

Mirrors ``axbench_base.runtime_state.RuntimeState`` shape but thinner —
no OpenRouter clients, no generation/judge caches. Each stage that needs
a resource calls the matching accessor; the first call loads it, later
calls return the cached reference.

All three lazy loaders are wired:
- ``decoder_only`` / ``encdec`` via ``nanochat.checkpoint_manager.load_model``
- ``base_gemma_and_sae`` via the resolver chain:
  ``parse_neuronpedia_source`` → ``resolve_sae`` → ``load_sae_and_model``

After loading, clustered and unclustered extraction SAEs share the same
reported-identity validation. Against real ``sae_lens 6.42.0`` v1 loads, three
direct identity fields are absent, so only ``metadata.model_name`` and
``metadata.neuronpedia_id`` can make the checks fire. The v2 no-false-positive
argument is an offline registry-level proof rather than a loader-execution
proof. An SAE reporting no identity metadata at all remains tolerated by
design.
"""
from __future__ import annotations

import gc
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch
from beartype.roar import BeartypeCallHintParamViolation

from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.eval.axbench_base.config import (
    bare_model_id,
    effective_extraction_model_id,
)
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.runtime_state import _build_inactive_sampler
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.paths import OutputPaths
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import AxBenchBpbError, TargetModel
from steering.eval.axbench_bpb.vocab_restriction import feature_vocab_allowed_ids
from steering.eval.cluster_checkpoint_pin import _mapping_get
from steering.eval.cluster_feature_contract import assert_loaded_cluster_target
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.extract_features import load_sae_and_model
from steering.sae_resolver import (
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
    resolve_sae,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


class _LoadedEncoderDecoderGPT(EncoderDecoderGPT):
    """Encoder-decoder GPT with runtime-attached checkpoint metadata."""

    _loaded_meta_data: dict[str, Any]


def _build_cluster_pre_forward_guard(
    contract: ClusterFeatureContract,
    model: Any,
) -> Callable[[], None]:
    """Return a one-shot full checkpoint rehash for the first model forward."""
    if not isinstance(getattr(model, "_loaded_meta_data", None), Mapping):
        raise AxBenchBpbError(
            "clustered axbench_bpb target is missing loaded checkpoint metadata"
        )
    checked = False

    def _guard() -> None:
        nonlocal checked
        if checked:
            return
        metadata = getattr(model, "_loaded_meta_data", None)
        if not isinstance(metadata, Mapping):
            raise AxBenchBpbError(
                "clustered axbench_bpb target lost loaded checkpoint metadata "
                "before first forward"
            )
        assert_loaded_cluster_target(
            contract,
            model,
            metadata,
            surface="axbench_bpb encdec checkpoint immediately before first forward",
        )
        checked = True

    return _guard


@dataclass
class RuntimeState:
    """Runtime state shared across stage executors."""

    cfg: AxBenchBpbConfig
    ddp_rank: int
    ddp_world_size: int
    cluster_contract: ClusterFeatureContract | None = None

    # Resolved paths (populated in __post_init__)
    _source_paths: SourcePaths = field(init=False)
    _output_paths: OutputPaths = field(init=False)

    # Lazy resources
    _decoder_only: TargetModelBundle | None = field(default=None, init=False)
    _encdec: TargetModelBundle | None = field(default=None, init=False)
    _decoder_only_metadata: dict[str, Any] | None = field(default=None, init=False)
    _encdec_metadata: dict[str, Any] | None = field(default=None, init=False)
    _base_gemma: Any = field(default=None, init=False)
    _base_sae: Any = field(default=None, init=False)
    _tokenizer_nanochat: Any = field(default=None, init=False)
    _tokenizer_extraction: Any = field(default=None, init=False)
    _source_concepts: list[Concept] | None = field(default=None, init=False)
    _base_gemma_tail_offloaded_after: int | None = field(default=None, init=False)

    # Inactive sampler — resolves lazily because it requires both the encdec
    # bundle AND its detected metadata. The ``_inactive_sampler_loaded``
    # sentinel distinguishes "not yet built" from "built and the value is
    # None" (encdec target was not trained with negative feature sampling),
    # so we never re-invoke the builder on the ``None`` branch.
    _inactive_sampler: Callable[..., dict[int, int]] | None = field(
        default=None, init=False,
    )
    _inactive_sampler_loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._source_paths = SourcePaths(
            source_run_dir=Path(self.cfg.source_run_dir),
        )
        run_tag = self.cfg.run_tag or ""
        self._output_paths = OutputPaths(
            output_root=Path(self.cfg.output_root),
            run_tag=run_tag,
        )

    def source_paths(self) -> SourcePaths:
        return self._source_paths

    def output_paths(self) -> OutputPaths:
        return self._output_paths

    def source_concepts(self) -> list[Concept]:
        """Load and validate the final Stage 1c concept authority once."""

        if self._source_concepts is not None:
            return self._source_concepts
        path = self._source_paths.concepts_json()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AxBenchBpbError(
                f"failed to load final source concepts from {path}: {exc}"
            ) from exc
        if not isinstance(raw, list) or not raw:
            raise AxBenchBpbError(
                f"final source concepts at {path} must be a non-empty JSON array"
            )
        concepts: list[Concept] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise AxBenchBpbError(
                    f"final source concepts[{index}] at {path} must be an object"
                )
            item_dict = cast(dict[str, Any], item)
            try:
                concept = Concept(**item_dict)
            except (TypeError, BeartypeCallHintParamViolation) as exc:
                raise AxBenchBpbError(
                    f"final source concepts[{index}] at {path} is invalid: {exc}"
                ) from exc
            if concept.layer < 0:
                raise AxBenchBpbError(
                    f"final source concepts[{index}] at {path} has invalid "
                    f"layer={concept.layer!r}"
                )
            concepts.append(concept)
        concept_ids = [concept.concept_id for concept in concepts]
        if concept_ids != sorted(set(concept_ids)):
            raise AxBenchBpbError(
                f"final source concepts at {path} must preserve unique, "
                "increasing original concept IDs"
            )
        feature_indices = [concept.feature_index for concept in concepts]
        if len(feature_indices) != len(set(feature_indices)):
            raise AxBenchBpbError(
                f"final source concepts at {path} has duplicate feature indices"
            )
        self._source_concepts = concepts
        return concepts

    def source_concept_ids(self) -> frozenset[int]:
        """Return stable IDs from the final Stage 1c concept authority."""

        return frozenset(
            concept.concept_id for concept in self.source_concepts()
        )

    def target_device(self) -> str:
        """Return the device string this rank should load models onto.

        Under DDP with ``cfg.device == "cuda"``, ``"cuda"`` is ambiguous:
        third-party loaders (HookedTransformer / SAELens / HF) often
        resolve ``"cuda"`` to ``cuda:0`` even after ``torch.cuda.set_device``,
        causing rank N to load weights split between cuda:0 and cuda:N
        and tripping a "tensors on different devices" error on the first
        forward pass. We promote the bare ``"cuda"`` to a concrete
        ``"cuda:<local_rank>"`` so every load lands on this rank's GPU.

        When cfg.device is already pinned (``"cuda:0"``, ``"cuda:1"``,
        ``"cpu"``, ``"mps"``) we honor the explicit value — callers that
        intentionally pinned a device shouldn't be overridden.
        """
        device = self.cfg.device
        if device != "cuda" or self.ddp_world_size <= 1:
            return device
        # Honour the per-rank CUDA device assignment that compute_init
        # already made via torch.cuda.set_device(local_rank).
        return f"cuda:{torch.cuda.current_device()}"

    def decoder_only(self) -> TargetModelBundle:
        """Return the decoder-only target model bundle.

        On every access, ensures ``bundle.model`` is on the per-rank
        target device. After ``offload_models()`` pushes the cached
        bundle to CPU, the next access re-uploads it to the device —
        preventing CPU/CUDA mismatches against device-resident input
        tensors on the next forward pass.
        """
        if self._decoder_only is None:
            do_cfg = self.cfg.decoder_only
            model, tokenizer, meta = load_model(
                source=do_cfg["source"], device=self.target_device(),
                phase="eval",
                model_tag=do_cfg["model_tag"], step=do_cfg["step"],
                eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"),
            )
            model = cast(GPT, model)
            reject_feature_clusters_metadata(meta, "axbench_bpb decoder checkpoint")
            self._decoder_only_metadata = meta
            self._decoder_only = TargetModelBundle(
                model=model, tokenizer=tokenizer,
                max_seq_len=model.config.sequence_len,  # GPT stores config directly
                target_name=TargetModel.DECODER_ONLY,
            )
        self._restore_bundle_device(self._decoder_only)
        return self._decoder_only

    def encdec(self) -> TargetModelBundle:
        """Return the encdec target model bundle.

        On every access, ensures ``bundle.model`` is on the per-rank
        target device. After ``offload_models()`` pushes the cached
        bundle to CPU, the next access re-uploads it to the device —
        preventing CPU/CUDA mismatches against device-resident input
        tensors on the next forward pass.
        """
        if self._encdec is None:
            en_cfg = self.cfg.encdec
            model, tokenizer, meta = load_model(
                source=en_cfg["source"], device=self.target_device(),
                phase="eval",
                model_tag=en_cfg["model_tag"], step=en_cfg["step"],
                eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"),
            )
            model = cast(_LoadedEncoderDecoderGPT, model)
            if self.cluster_contract is None:
                reject_feature_clusters_metadata(meta, "axbench_bpb encdec checkpoint")
            else:
                assert_loaded_cluster_target(
                    self.cluster_contract,
                    model,
                    meta,
                    surface="axbench_bpb encdec checkpoint",
                )
            require_fixed_checkpoint_segment_mode(
                meta,
                surface="axbench_bpb",
            )
            model._loaded_meta_data = meta  # required by detect_target_model_metadata
            self._encdec_metadata = meta
            self._encdec = TargetModelBundle(
                model=model, tokenizer=tokenizer,
                # EncoderDecoderGPT stores its decoder GPTConfig as model.dec_config
                # (not model.config). See nanochat/checkpoint_manager.py:394 and
                # steering/conditional/encoder_decoder/model.py for the definition.
                # Decoder-only path (above) still uses model.config because nanochat
                # GPT exposes it directly.
                max_seq_len=model.dec_config.sequence_len,
                target_name=TargetModel.ENCDEC,
                cluster_contract=self.cluster_contract,
                cluster_pre_forward_guard=(
                    None
                    if self.cluster_contract is None
                    else _build_cluster_pre_forward_guard(
                        self.cluster_contract,
                        model,
                    )
                ),
            )
        self._restore_bundle_device(self._encdec)
        return self._encdec

    def loaded_target_metadata(self, target_name: str) -> Mapping[str, Any]:
        """Read-only view of the exact metadata returned by ``load_model``.

        Encoder-decoder loading may recover semantic configuration from ancestor
        metadata. Exposing that already-resolved object here lets checkpoint
        identity bind the loader's effective interpretation without duplicating
        vendored recovery logic.
        """

        if target_name == TargetModel.DECODER_ONLY:
            self.decoder_only()
            metadata = self._decoder_only_metadata
        elif target_name == TargetModel.ENCDEC:
            self.encdec()
            metadata = self._encdec_metadata
        else:
            raise AxBenchBpbError(f"unknown target metadata request {target_name!r}")
        if metadata is None:
            raise AxBenchBpbError(
                f"target metadata for {target_name!r} was not retained after loading"
            )
        # A top-level MappingProxyType alone would still expose the retained
        # object's nested dicts/lists to mutation. Return a defensive snapshot
        # so no caller can alter the metadata that RuntimeState preserves as the
        # loader's exact interpretation.
        return MappingProxyType(deepcopy(metadata))

    def _restore_bundle_device(self, bundle: TargetModelBundle) -> None:
        """If ``bundle.model``'s parameter device differs from
        the per-rank target device, move the model back. Used by both
        lazy accessors to guarantee the cached bundle is device-resident
        even after a previous ``offload_models()`` call has pushed it to
        CPU.

        No-op when the bundle is already on the target device, when the
        model exposes no ``parameters()`` (test stubs that aren't real
        ``nn.Module`` instances), or when it has no parameters at all.
        """
        parameters_fn = getattr(bundle.model, "parameters", None)
        if not callable(parameters_fn):
            return
        first_param = next(iter(parameters_fn()), None)
        if first_param is None:
            return
        device = torch.device(self.target_device())
        if first_param.device != device:
            bundle.model.to(device)

    def base_gemma_and_sae(
        self,
        *,
        offload_transformer_tail_after: int | None = None,
    ) -> tuple[Any, Any]:
        """Return the base Gemma model + matching SAE for feature extraction.

        ``offload_transformer_tail_after`` is an opt-in low-peak-memory mode for
        callers that guarantee every forward stops after that layer.  A cached
        mixed-device model may only be reused under the same explicit contract.
        """
        if (
            self._base_gemma is not None
            and self._base_gemma_tail_offloaded_after
            != offload_transformer_tail_after
        ):
            raise AxBenchBpbError(
                "cached extraction model tail-residency contract mismatch: "
                "loaded with offload_transformer_tail_after="
                f"{self._base_gemma_tail_offloaded_after!r}, requested="
                f"{offload_transformer_tail_after!r}"
            )
        if self._base_gemma is None:
            extraction_model_id = effective_extraction_model_id(
                self.cfg,
            )
            release, activation_type, layer, width, l0 = parse_neuronpedia_source(
                self.cfg.neuronpedia_source
            )
            try:
                resolved = resolve_sae(
                    # Bare-normalize on the RESOLVE side too, not just the
                    # post-load comparison below. SAELens force-prefixes
                    # Gemma-3 ids with ``google/`` and a checkpoint's recorded
                    # extraction id can carry that prefix, while resolve_sae's
                    # registry is keyed on bare ids -- so a real clustered
                    # gemma-scope-2 load raised "Unsupported model
                    # 'google/gemma-3-4b-pt'" against a list that contains
                    # 'gemma-3-4b-pt'. FU-D fixed the comparison at :432 but
                    # left this call site prefixed; only a real GPU load
                    # reaches it, so CPU tests that stub the loader cannot see it.
                    model=bare_model_id(str(extraction_model_id)),
                    release=release,
                    activation_type=activation_type,
                    layer=layer,
                    width=width,
                    # v1 sources parse to l0=None; v2 sources retain their
                    # encoded tier in clustered and unclustered runs alike.
                    l0=l0,
                )
            except ValueError as exc:
                raise AxBenchBpbError(
                    "could not resolve the runtime extraction SAE exactly: "
                    f"{exc}"
                ) from exc
            if self.cluster_contract is not None:
                contract_sae = self.cluster_contract.sae
                configured_identity = (
                    # Compared against the contract's SaeIdentity.model, which
                    # stores the bare id. effective_extraction_model_id's own
                    # docstring requires callers comparing against a bare id to
                    # normalize; the loader below deliberately keeps the full
                    # HF form, so normalize here rather than at the source.
                    bare_model_id(str(extraction_model_id)),
                    activation_type.value,
                    layer,
                    width.value,
                    resolved.release,
                    resolved.sae_id,
                    resolved.hook_name,
                )
                contract_identity = (
                    # BOTH sides normalized. A contract may record either form
                    # depending on which producer wrote it, so normalizing only
                    # the configured side just moves the mismatch instead of
                    # removing it -- which is exactly what the one-sided fix
                    # above this did before a pre-existing test caught it.
                    bare_model_id(str(contract_sae.model)),
                    contract_sae.activation_type,
                    contract_sae.layer,
                    contract_sae.width,
                    contract_sae.release,
                    contract_sae.sae_id,
                    contract_sae.hook_name,
                )
                if configured_identity != contract_identity:
                    raise AxBenchBpbError(
                        "runtime extraction SAE identity does not match the "
                        "cluster feature contract: "
                        f"configured={configured_identity!r}, "
                        f"contract={contract_identity!r}"
                    )
            if (
                offload_transformer_tail_after is not None
                and offload_transformer_tail_after != layer
            ):
                raise AxBenchBpbError(
                    "extraction-model tail may only be offloaded after the exact "
                    f"configured SAE layer {layer}, got "
                    f"{offload_transformer_tail_after}"
                )
            load_kwargs: dict[str, Any] = {
                "model_name": extraction_model_id,
                "layer": layer,
                "device": self.target_device(),
                "resolved": resolved,
            }
            if offload_transformer_tail_after is not None:
                load_kwargs["offload_transformer_tail"] = True
            model, sae, tok = load_sae_and_model(
                **load_kwargs,
            )
            sae_cfg = getattr(sae, "cfg", None)
            if self.cluster_contract is not None:
                loaded_width = getattr(sae_cfg, "d_sae", None)
                loaded_hook = getattr(sae_cfg, "hook_name", None)
                if not loaded_hook:
                    sae_metadata = getattr(sae_cfg, "metadata", None)
                    loaded_hook = (
                        sae_metadata.get("hook_name")
                        if isinstance(sae_metadata, Mapping)
                        else getattr(sae_metadata, "hook_name", None)
                    )
                decoder = getattr(sae, "W_dec", None)
                if (
                    type(loaded_width) is not int
                    or loaded_width != self.cluster_contract.n_sae_features
                    or not isinstance(loaded_hook, str)
                    or not loaded_hook
                    or loaded_hook != self.cluster_contract.sae.hook_name
                    or not isinstance(decoder, torch.Tensor)
                    or decoder.ndim != 2
                    or int(decoder.shape[0]) != self.cluster_contract.n_sae_features
                ):
                    raise AxBenchBpbError(
                        "loaded runtime extraction SAE does not match the cluster "
                        "feature contract: "
                        f"d_sae={loaded_width!r}, hook={loaded_hook!r}, "
                        f"W_dec_shape={getattr(decoder, 'shape', None)!r}"
                    )
            # Codex F1b review (HIGH bug fix): the checks above (d_sae /
            # hook_name / W_dec.shape[0]) are IDENTICAL across GemmaScope-2
            # L0 tiers -- big/medium/small share release, layer, width,
            # hook_name, and dictionary width; only sae_id/neuronpedia_id
            # differ (sae_resolver.py's _resolve_gemmascope_2: ``sae_id =
            # f"layer_{layer}_width_{width.value}_l0_{l0_tier.value}"``).
            # A loader that silently returns the wrong tier (same shape,
            # wrong dictionary) would pass every check above and get
            # cached, then Stage 1 would extract features from the wrong
            # SAE. This mirrors cluster_checkpoint_pin.py's
            # ``_resolve_and_validate_sae`` preflight (same identity
            # fields, same ``_mapping_get`` direct-attribute-or-metadata-
            # dict access, same "validate the loaded object against the
            # resolver output" shape) so this runtime reload cannot
            # silently drop the preflight's guarantee. FU-C extends these
            # same reported-identity checks to the unclustered path.
            #
            # Against real SAELens 6.42.0 v1 loads, cfg.release,
            # cfg.model_name, and metadata.sae_id are absent; only the
            # metadata.model_name and metadata.neuronpedia_id field probes
            # can fire. The v2 no-false-positive argument is an offline
            # registry-level proof rather than a loader-execution proof. An
            # SAE reporting no identity metadata remains tolerated by design.
            #
            # Keep the model_name comparison bare-normalized on BOTH sides.
            # SAELens 6.42.0 loading/pretrained_sae_loaders.py:638-641
            # force-prefixes Gemma-3 model ids with ``google/``; the fetched
            # google/gemma-scope-2-4b-pt
            # resid_post/layer_22_width_16k_l0_medium/config.json reports
            # ``"model_name": "google/gemma-3-4b-pt"``, while resolve_sae
            # accepts bare ``gemma-3-4b-pt``. Raw equality would reject
            # every real clustered GemmaScope-2 load.
            loaded_metadata = _mapping_get(sae_cfg, "metadata", None)
            direct_release = _mapping_get(sae_cfg, "release", None)
            if direct_release and direct_release != resolved.release:
                raise AxBenchBpbError(
                    "loaded runtime extraction SAE cfg.release "
                    f"{direct_release!r} does not match resolved release "
                    f"{resolved.release!r}"
                )
            direct_model_name = _mapping_get(sae_cfg, "model_name", None)
            metadata_model_name = _mapping_get(loaded_metadata, "model_name", None)
            loaded_model_name = direct_model_name or metadata_model_name
            if (
                loaded_model_name is not None
                and bare_model_id(str(loaded_model_name))
                != bare_model_id(str(extraction_model_id))
            ):
                raise AxBenchBpbError(
                    "loaded runtime extraction SAE model_name "
                    f"{loaded_model_name!r} does not match resolved model "
                    f"{str(extraction_model_id)!r}"
                )
            metadata_sae_id = _mapping_get(loaded_metadata, "sae_id", None)
            if metadata_sae_id and metadata_sae_id != resolved.sae_id:
                raise AxBenchBpbError(
                    "loaded runtime extraction SAE cfg.metadata.sae_id "
                    f"{metadata_sae_id!r} does not match resolved sae_id "
                    f"{resolved.sae_id!r} -- likely the wrong L0 tier was "
                    "loaded"
                )
            metadata_neuronpedia_id = _mapping_get(
                loaded_metadata, "neuronpedia_id", None,
            )
            if metadata_neuronpedia_id:
                try:
                    expected_model_id, expected_source = resolve_neuronpedia_ids(
                        model=extraction_model_id,
                        release=release,
                        activation_type=activation_type,
                        layer=layer,
                        width=width,
                        l0=l0,
                    )
                except ValueError:
                    pass  # not published for this combo; cannot validate this way
                else:
                    expected_neuronpedia_id = f"{expected_model_id}/{expected_source}"
                    if metadata_neuronpedia_id != expected_neuronpedia_id:
                        raise AxBenchBpbError(
                            "loaded runtime extraction SAE "
                            "cfg.metadata.neuronpedia_id "
                            f"{metadata_neuronpedia_id!r} does not match "
                            f"resolved Neuronpedia id "
                            f"{expected_neuronpedia_id!r} -- likely the "
                            "wrong L0 tier was loaded"
                        )
            self._base_gemma = model
            self._base_sae = sae
            self._tokenizer_extraction = tok
            self._base_gemma_tail_offloaded_after = offload_transformer_tail_after
        return self._base_gemma, self._base_sae

    def extraction_tokenizer(self) -> Any:
        """Return the Gemma tokenizer used for SAE feature extraction.

        Lazy-loaded alongside base_gemma_and_sae(); calling either populates
        the cache.
        """
        if self._tokenizer_extraction is None:
            self.base_gemma_and_sae()
        return self._tokenizer_extraction

    def release_base_gemma_and_sae(self) -> None:
        """Release extraction-only GPU resources while retaining the tokenizer.

        The clustered context-cache stage must load a nanochat target after the
        Gemma/SAE extraction pass to run tokenizer/BOS compatibility guards. A
        4B extraction model and the target checkpoint do not fit concurrently
        on a 24 GiB device. Clearing both cached resource owners, collecting
        reference cycles, and emptying free allocator blocks makes that stage
        transition explicit. The tokenizer is host-side and remains available
        for the guard; a later extraction request reloads the model and SAE.
        """

        if self._base_gemma is None and self._base_sae is None:
            return
        self._base_gemma = None
        self._base_sae = None
        self._base_gemma_tail_offloaded_after = None
        # Dropping the references is not enough on its own. An nn.Module graph
        # reaches its own submodules, and a registered forward hook whose
        # closure captures the module closes the loop -- as does the nnterp
        # speedup wrapper hung off ``model.nnterp_model``. Anything held in a
        # cycle survives refcounting, so its CUDA blocks are still ALLOCATED
        # when empty_cache() runs and empty_cache() only returns blocks that
        # are already free. The release would then free nothing.
        #
        # release_cuda_iteration_memory() deliberately dropped its collect: it
        # measured 65% of the Stage 1b result-handler thread when called per QC
        # batch. Its docstring names the remedy for callers that do create
        # cycles -- collect "at a coarse cadence" instead. This is that cadence:
        # a once-per-stage phase transition, not a hot loop.
        gc.collect()
        release_cuda_iteration_memory()

    def inactive_sampler(self) -> Callable[..., dict[int, int]] | None:
        """Return the inactive-feature sampler for negative-mode methods.

        ``None`` if the encdec target was not trained with negative
        feature sampling (``encdec_meta.negative_selection is None``).
        Otherwise a callable with the signature
        ``(*, rng: random.Random) -> dict[int, int]`` suitable for
        forwarding to ``build_mode_aware_encoder_input``.

        Mirrors ``axbench_base.runtime_state.RuntimeState.inactive_sampler``:
        on first call, briefly enters ``encdec()`` to detect the target
        metadata, then either short-circuits to ``None`` or builds and
        caches the sampler closure. The ``_inactive_sampler_loaded``
        sentinel ensures the ``None`` branch is cached too — a target
        without negative_selection will never re-enter
        ``detect_target_model_metadata`` or ``_build_inactive_sampler``.
        """
        if not self._inactive_sampler_loaded:
            encdec_bundle = self.encdec()
            detect_kwargs = (
                {}
                if self.cluster_contract is None
                else {"cluster_contract": self.cluster_contract}
            )
            encdec_meta = detect_target_model_metadata(
                encdec_bundle.model,
                self.cfg,
                **detect_kwargs,
            )
            if encdec_meta.negative_selection is None:
                self._inactive_sampler = None
            else:
                allowed_feature_ids = (
                    torch.tensor(
                        self.cluster_contract.representative_universe,
                        dtype=torch.int64,
                    )
                    if self.cluster_contract is not None
                    else feature_vocab_allowed_ids(encdec_bundle.model)
                )
                self._inactive_sampler = _build_inactive_sampler(
                    encdec_meta=encdec_meta,
                    cfg=self.cfg,
                    allowed_feature_ids=allowed_feature_ids,
                )
            self._inactive_sampler_loaded = True
        return self._inactive_sampler

    def offload_models(self) -> None:
        """Push both target models to CPU between stages.

        No-op when ``cfg.offload_between_stages`` is False. Useful for
        single-GPU runs where running e.g. context-cache + scoring in
        sequence would OOM if both target models stayed resident.
        """
        if not self.cfg.offload_between_stages:
            return
        if self._decoder_only is not None:
            self._decoder_only.model.to("cpu")
        if self._encdec is not None:
            self._encdec.model.to("cpu")

    def release_target_models(self) -> None:
        """Drop both target-model owners and release their CUDA storage.

        Clustered generic validation must temporarily load the Gemma-3-4B
        extractor and its 65k SAE.  Merely moving the target modules to CPU is
        not a sufficiently strong ownership boundary for that combination:
        model-attached guards and other bundle state can keep the old module
        graph alive.  Callers must first drop any external ``TargetModelBundle``
        references, then call this method.  A later accessor reloads the exact
        pinned checkpoint normally.

        The inactive sampler is checkpoint-derived state, so it is invalidated
        together with the models rather than surviving a target reload.
        """

        self._decoder_only = None
        self._encdec = None
        self._inactive_sampler = None
        self._inactive_sampler_loaded = False
        release_cuda_iteration_memory()


__all__ = ["RuntimeState"]
