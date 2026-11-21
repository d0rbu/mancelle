"""Harmfulness evaluation CLI — generate responses and/or judge them.

Usage:
    # Generate + judge (saves to model report dir):
    uv run python scripts/harmfulness_eval.py source=safety model_tag=d26

    # Judge only (auto-finds responses in model report dir):
    uv run python scripts/harmfulness_eval.py source=safety model_tag=d26 judge_only=true

    # Judge only (explicit file):
    uv run python scripts/harmfulness_eval.py responses_file=responses.yaml output_file=results.yaml

    # Repository Hugging Face SFT export (one concrete local step):
    uv run python -m scripts.eval.harmfulness_eval \
        hf_checkpoint_dir=/path/to/step_000123 temperature=0 top_k=null
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch as t
import yaml
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from errors import DataLoadError
from nanochat.checkpoint_manager import (
    ModelPhase,
    _find_checkpoint_leaf,
    _is_encdec_root,
    build_encdec_model,
    find_largest_model,
    find_last_step,
    get_checkpoint_dir,
    load_model,
    meta_looks_encdec,
)
from nanochat.common import autodetect_device_type, get_base_dir, load_project_dotenv
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_mask import feature_mask_leaf_from_cfg
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.checkpoint_selection import parse_checkpoint_feature_selection
from steering.eval.compacted_eval_loading import (
    assert_conditioned_eval_model,
    conditioned_eval_conditioning,
    conditioning_provenance,
)
from steering.eval.context_cache_io import (
    SafetensorsContextFeatureProvider,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.harmfulness_artifact import (
    build_harmfulness_payload,
    build_harmfulness_responses_artifact,
    context_cache_manifest_sha256,
    current_harmfulness_payload,
    parse_harmfulness_responses_artifact,
    write_harmfulness_payload,
    write_harmfulness_responses_artifact,
)
from steering.eval.harmfulness_generation import (
    DECODER_ONLY_MODEL_TYPE,
    ENCODER_DECODER_MODEL_TYPE,
    HF_BATCHED_SEED_SCHEDULE,
    SEED_SCHEDULE,
    SELF_CONDITIONED_SEED_SCHEDULE,
    absolute_prompt_batch_seed,
    build_generation_provenance,
    capture_native_artifact_identity,
    native_checkpoint_metadata_path,
    native_model_identity,
    validate_generation_provenance,
    validate_harmfulness_response_record,
)
from steering.eval.harmfulness_judge import (
    JudgeClient,
    compute_compliance_metrics,
    compute_metrics,
)
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.eval.harmfulness_self_conditioning import (
    HarmfulnessConditionName,
    HarmfulnessSelfCondSetup,
    generate_harmfulness_response_self_conditioned,
    harmfulness_completion_provenance,
    parse_harmfulness_safety_settings,
    prepare_harmfulness_self_conditioning,
)
from steering.eval.hf_chat_generation import (
    HuggingFaceChatGenerationEngine,
    RepositoryHFCheckpoint,
    load_repository_hf_checkpoint,
    resolve_repository_hf_checkpoint_dir,
)
from steering.eval.inference_engine import InferenceEngine
from steering.eval.prompt_coverage import (
    PromptCoverageSummary,
    assert_exact_prompt_coverage,
)
from steering.immutable_io import stream_hash_file_descriptor_bound

# Pin dotenv load to the project root so `.env` in ancestor directories
# (e.g. /tmp/.env) cannot inject arbitrary env vars.
load_project_dotenv()

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "harmfulness_eval"
_CONFIG_NAME_FLAG = "--config-name"


_RESULTS_DIR = "harmfulness_results"
_RESPONSES_DIR = "harmfulness_results_responses"

ResponseGenerator = Callable[[list[tuple[int, dict[str, Any]]]], list[str]]
LoadedResponses = tuple[list[dict[str, object]], dict[str, Any], dict[str, Any]]


def _resolve_generation_batch_size(cfg: Mapping[str, Any], *, is_hf: bool) -> int:
    """Resolve the backend-specific outer batch without changing native defaults."""

    def strict_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    native_batch_size = strict_positive_int("batch_size", cfg.get("batch_size"))
    hf_batch_size = strict_positive_int(
        "repository_hf_batch_size",
        cfg.get("repository_hf_batch_size", 64),
    )
    return hf_batch_size if is_hf else native_batch_size


def _validate_harmfulness_generation_config(
    *, feature_configs_configured: bool,
    self_cond_segment_size: int | None, is_encdec: bool | None, batch_size: int,
    safety_steering_configured: bool = False,
    generating: bool = True,
) -> None:
    """Keep decoder-only evaluation and dense self-conditioning unambiguous."""
    if feature_configs_configured:
        raise ValueError(
            "encoder-decoder harmfulness rejects hand-supplied feature_configs; use "
            "self_conditioned.safety_steering={targets: PATH, strength: NUMBER}. "
            "Decoder-only harmfulness requires feature_configs=null."
        )
    if self_cond_segment_size is not None and (
        type(self_cond_segment_size) is not int or self_cond_segment_size <= 0
    ):
        raise ValueError("self_cond_segment_size must be a positive fixed-cadence assertion or null")
    if not generating:
        return  # Model-free assertions are checked against the admitted receipt.
    if is_encdec is False:
        if self_cond_segment_size is not None:
            raise ValueError("decoder-only harmfulness requires self_cond_segment_size=null")
        if safety_steering_configured:
            raise ValueError("safety_steering requires an encoder-decoder checkpoint")
        return
    if is_encdec is True and batch_size != 1:
        raise ValueError("self-conditioned harmfulness requires batch_size=1")


def _resolve_hf_checkpoint_dir(cfg: DictConfig) -> Path | None:
    """Validate selector exclusivity and every HF-incompatible config early."""

    source = cfg.get("source")
    hf_raw = cfg.get("hf_checkpoint_dir")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("source must be null or a non-empty string")
    if hf_raw is not None and (not isinstance(hf_raw, str) or not hf_raw.strip()):
        raise ValueError("hf_checkpoint_dir must be null or a non-empty local path")
    if source is not None and hf_raw is not None:
        raise ValueError("Exactly one of source or hf_checkpoint_dir may be set, not both")
    if source is None and hf_raw is None and cfg.get("responses_file") is None:
        raise ValueError(
            "A model selector is required for generation or automatic judge-only reuse: "
            "set exactly one of source or hf_checkpoint_dir, or set responses_file"
        )

    hf_path = None if hf_raw is None else Path(hf_raw).expanduser()
    if hf_path is not None:
        unsupported: list[str] = []
        for key in (
            "model_tag",
            "step",
            "segment_size",
            "self_cond_segment_size",
            "stats_path",
            "extraction_tokenizer_id",
            "feature_selection",
            "feature_mask",
        ):
            if cfg.get(key) is not None:
                unsupported.append(key)
        if cfg.get("feature_configs") is not None and cfg.get("feature_configs") != []:
            unsupported.append("feature_configs")
        if parse_harmfulness_safety_settings(cfg) is not None:
            unsupported.append("self_conditioned.safety_steering")
        if cfg.get("next_segment_conditioning_offset") not in {None, "auto"}:
            unsupported.append("next_segment_conditioning_offset")
        conditioning = cfg.get("conditioning")
        if conditioning is not None:
            if conditioning.get("context_cache_dir") is not None:
                unsupported.append("conditioning.context_cache_dir")
            identity = conditioning.get("identity")
            if identity is not None:
                unsupported.extend(
                    f"conditioning.identity.{key}"
                    for key, value in identity.items()
                    if value is not None
                )
        if unsupported:
            raise ValueError(
                "hf_checkpoint_dir mode does not support native/conditioning settings: "
                + ", ".join(sorted(unsupported))
            )
        hf_path = resolve_repository_hf_checkpoint_dir(hf_path)
    return hf_path


def _parse_cli_config_and_overrides(argv: list[str]) -> tuple[str, list[str]]:
    """Route this single-process CLI's native Hydra config selector.

    ``harmfulness_eval`` uses the compose API internally so its pipeline remains
    directly unit-testable, but its public CLI follows the repository's
    single-process ``--config-name`` convention.  Consume that native selector
    before forwarding ordinary Hydra overrides to :func:`hydra.compose`.
    """

    config_name = _CONFIG_NAME
    overrides: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == _CONFIG_NAME_FLAG:
            index += 1
            if index >= len(argv) or not argv[index] or argv[index].startswith("-"):
                raise ValueError(
                    "--config-name requires a non-empty config name, for example "
                    "--config-name=harmfulness_eval_l22"
                )
            config_name = argv[index]
        elif arg.startswith(f"{_CONFIG_NAME_FLAG}="):
            config_name = arg.split("=", 1)[1]
            if not config_name:
                raise ValueError(
                    "--config-name requires a non-empty config name, for example "
                    "--config-name=harmfulness_eval_l22"
                )
        elif arg.startswith("config_name="):
            raise ValueError(
                "harmfulness_eval is a single-process Hydra CLI; select a preset "
                "with --config-name=harmfulness_eval_l22 (including the leading "
                "dashes), not config_name=harmfulness_eval_l22"
            )
        elif "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        index += 1

    return config_name, overrides


def _compose_cli_config(argv: list[str]) -> DictConfig:
    config_name, overrides = _parse_cli_config_and_overrides(argv)
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _has_checkpoint_file(path: str) -> bool:
    return any(Path(path).glob("model_*.pt"))


def _encdec_resolution_error(
    *,
    source: str,
    model_tag: str | None,
    segment_size: int | None,
    reason: str,
) -> ValueError:
    return ValueError(
        "encdec source could not resolve the checkpoint/report directory. "
        "Set feature_selection=... with segment_size=... for pinned "
        "s<seg>/<feature_selection>/ layouts, or leave feature_selection unset "
        "only when auto-discovery can find "
        "a model_*.pt checkpoint leaf. "
        f"source={source!r}, model_tag={model_tag!r}, "
        f"segment_size={segment_size!r}. {reason}"
    )


def _encdec_tag_root(model_tag: str | None) -> tuple[str, str]:
    checkpoints_dir = os.path.join(get_base_dir(), "conditional_encdec_checkpoints")
    resolved_tag = model_tag or find_largest_model(checkpoints_dir)
    return os.path.join(checkpoints_dir, resolved_tag), resolved_tag


def _encdec_tail_for_source(
    *,
    source: str,
    model_tag: str | None,
) -> str:
    """Return the source-chain tail relative to a feature-selection root."""
    # Use a synthetic segment size so get_checkpoint_dir resolves the chain
    # without triggering its own auto-discovery. The relative tail is
    # independent of the segment value.
    segment_root = get_checkpoint_dir(
        "encdec",
        model_tag=model_tag,
        segment_size=0,
    )
    source_without_feature_selection = get_checkpoint_dir(
        source,
        model_tag=model_tag,
        segment_size=0,
    )
    tail = os.path.relpath(source_without_feature_selection, segment_root)
    if tail.startswith(".."):
        raise ValueError(
            f"Invalid encdec source chain {source!r}: resolved tail escapes "
            f"segment root {segment_root!r}"
        )
    return tail


def _iter_dirs_depth_first(root_dir: str):
    yield root_dir
    for entry in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path):
            yield from _iter_dirs_depth_first(path)


def _find_chained_checkpoint_leaf(search_root: str, tail: str) -> str:
    """Find a feature-selection leaf whose source-chain tail has checkpoints."""
    for candidate in _iter_dirs_depth_first(search_root):
        chained = os.path.join(candidate, tail)
        if _has_checkpoint_file(chained):
            return chained
    raise DataLoadError(f"No checkpoint files found under {search_root} for source tail {tail!r}")


def _auto_discover_encdec_checkpoint_dir(
    *,
    source: str,
    model_tag: str | None,
    segment_size: int | None,
) -> str:
    """Resolve an encdec checkpoint dir by finding a nested checkpoint leaf."""
    try:
        tag_root, resolved_tag = _encdec_tag_root(model_tag)
    except OSError as exc:
        checkpoints_dir = os.path.join(
            get_base_dir(),
            "conditional_encdec_checkpoints",
        )
        raise _encdec_resolution_error(
            source=source,
            model_tag=model_tag,
            segment_size=segment_size,
            reason=(
                f"Auto-discovery could not inspect encdec checkpoint root {checkpoints_dir!r}."
            ),
        ) from exc

    tail = _encdec_tail_for_source(source=source, model_tag=resolved_tag)

    if segment_size is None:
        search_root = tag_root
    else:
        search_root = os.path.join(tag_root, f"s{segment_size}")

    try:
        if tail == ".":
            return _find_checkpoint_leaf(search_root)
        return _find_chained_checkpoint_leaf(search_root, tail)
    except (DataLoadError, OSError) as exc:
        raise _encdec_resolution_error(
            source=source,
            model_tag=model_tag,
            segment_size=segment_size,
            reason=(
                "Auto-discovery did not find a model_*.pt checkpoint leaf "
                f"under {search_root!r} for source tail {tail!r}."
            ),
        ) from exc


def _resolve_checkpoint_dir_for_harmfulness(
    *,
    source: str,
    model_tag: str | None,
    segment_size: int | None,
    feature_selection,
    mask_leaf: str = "",
) -> str:
    """Resolve the concrete checkpoint dir for harmfulness eval/reporting.

    ``mask_leaf`` is the ``bl_<hash>`` / ``wl_<hash>`` subdirectory appended
    when the encdec checkpoint was trained with a feature blacklist/whitelist.
    Pass ``feature_mask_leaf_from_cfg(cfg)`` (returns ``""`` when no mask is
    configured, preserving byte-identical paths for legacy runs).

    Only the pinned ``feature_selection`` resolution path needs ``mask_leaf``
    because auto-discovery
    (``_auto_discover_encdec_checkpoint_dir``) already recurses into
    ``bl_/wl_`` leaves via ``_find_checkpoint_leaf``.  Non-encdec sources
    (decoder-only, base, safety, rl, …) never have mask leaves.
    """
    if not _is_encdec_root(source):
        return get_checkpoint_dir(source, model_tag=model_tag)

    parsed_feature_selection = parse_checkpoint_feature_selection(
        feature_selection,
        live_context="harmfulness_eval live selection",
    )
    if parsed_feature_selection is not None:
        if segment_size is None:
            raise _encdec_resolution_error(
                source=source,
                model_tag=model_tag,
                segment_size=segment_size,
                reason="feature_selection was provided but segment_size is missing.",
            )
        path = get_checkpoint_dir(
            source,
            model_tag=model_tag,
            segment_size=segment_size,
            feature_selection=parsed_feature_selection,
        )
        return os.path.join(path, mask_leaf) if mask_leaf else path

    return _auto_discover_encdec_checkpoint_dir(
        source=source,
        model_tag=model_tag,
        segment_size=segment_size,
    )


def resolve_harmfulness_paths(
    source: str | None,
    model_tag: str | None,
    output_file: str | None,
    responses_file: str | None,
    *,
    hf_checkpoint_dir: str | Path | None = None,
    config_name: str,
    segment_size: int | None = None,
    feature_selection=None,
    mask_leaf: str = "",
) -> tuple[str, str]:
    """Resolve output paths for the harmfulness evaluation.

    When ``output_file`` is None and ``source`` is set, both paths are
    auto-derived from the model's report directory using a config-name-based
    subdirectory layout::

        {report_dir}/harmfulness_results/{config_name}.yaml
        {report_dir}/harmfulness_results_responses/{config_name}.yaml

    The *config_name* identifies the resolved condition, or preserves the
    recorded condition name when judging an explicit responses file.

    Explicit ``output_file`` / ``responses_file`` values always take
    priority and bypass the directory logic entirely.

    For ``encdec`` sources, report placement follows the resolved checkpoint
    layout: explicit ``feature_selection`` uses
    ``feature_selection_dirname()`` via :func:`get_checkpoint_dir`; when it is
    unset, the checkpoint leaf is auto-discovered. Threshold-trained
    checkpoints remain directly addressable through a threshold-form
    ``feature_selection`` block.

    ``mask_leaf`` is the ``bl_<hash>`` / ``wl_<hash>`` directory component
    produced by :func:`feature_mask_leaf_from_cfg` for mask-trained encdec
    checkpoints.  Pass ``""`` (the default) or omit it for unmasked runs —
    the path is byte-identical to the pre-mask layout.

    Returns:
        (responses_path, output_path)

    Raises:
        ValueError: If neither ``source`` nor ``output_file`` is set.
    """
    # Resolve report dir only when an auto-derived output path needs it.
    if source and output_file is None:
        ckpt_dir = _resolve_checkpoint_dir_for_harmfulness(
            source=source,
            model_tag=model_tag,
            segment_size=segment_size,
            feature_selection=feature_selection,
            mask_leaf=mask_leaf,
        )
        report_dir = os.path.join(ckpt_dir, "report")
    elif hf_checkpoint_dir is not None and output_file is None:
        report_dir = str(Path(hf_checkpoint_dir) / "report")
    else:
        report_dir = None

    # Determine the output (judge results) path.
    if output_file is not None:
        out = output_file
    elif report_dir is not None:
        out = os.path.join(report_dir, _RESULTS_DIR, f"{config_name}.yaml")
    elif responses_file is not None:
        response_path = Path(responses_file)
        out = str(response_path.with_stem(response_path.stem + "_results"))
    else:
        raise ValueError(
            "Cannot determine output path: set output_file, source, hf_checkpoint_dir, "
            "or responses_file"
        )

    # Determine the responses path.
    if responses_file is not None:
        resp = responses_file
    elif report_dir is not None and output_file is None:
        # Auto-derive from report dir (same dir as output).
        resp = os.path.join(report_dir, _RESPONSES_DIR, f"{config_name}.yaml")
    else:
        # Derive from the explicit output_file.
        out_path = Path(out)
        resp = str(out_path.with_stem(out_path.stem + "_responses"))

    return resp, out


def _resolve_native_checkpoint_for_harmfulness(cfg) -> tuple[str, int]:
    """Share source selection and latest-step resolution with model-free judging."""
    checkpoint_dir = _resolve_checkpoint_dir_for_harmfulness(
        source=cfg.source,
        model_tag=cfg.model_tag,
        segment_size=cfg.get("segment_size"),
        feature_selection=cfg.get("feature_selection"),
        mask_leaf=feature_mask_leaf_from_cfg(cfg),
    )
    step = cfg.step
    if step is None:
        step = find_last_step(checkpoint_dir)
    return checkpoint_dir, step


def _load_model_for_harmfulness(cfg, device: t.device):
    """Load one native model and bind it to exact checkpoint/tokenizer bytes."""
    checkpoint_dir, step = _resolve_native_checkpoint_for_harmfulness(cfg)
    artifacts = capture_native_artifact_identity(
        checkpoint_dir,
        step=step,
        tokenizer_dir=Path(get_base_dir()) / "tokenizer",
    )

    encdec_source = _is_encdec_root(cfg.source)
    stage_on_cpu = device.type == "cuda" and encdec_source
    if device.type == "cuda" and not encdec_source:
        # Use the already inventoried metadata to recognize conditioned SFT
        # stored under decoder-shaped source chains; malformed bytes fail loud.
        metadata_entry = next(row for row in artifacts.checkpoint_inventory.entries
            if row.path.suffix == ".json")
        stage_on_cpu = meta_looks_encdec(json.loads(metadata_entry.path.read_bytes()))
    load_device = t.device("cpu") if stage_on_cpu else device
    if encdec_source:
        model, tokenizer, meta = build_encdec_model(
            checkpoint_dir,
            step,
            load_device,
            ModelPhase.EVAL,
            eval_conditioning=conditioned_eval_conditioning("harmfulness_eval"),
        )
        contract = assert_conditioned_eval_model(
            model,
            meta,
            checkpoint_dir=checkpoint_dir,
            eval_name="harmfulness_eval",
        )
        meta["checkpoint_dir"] = checkpoint_dir
        assert contract is not None
    else:
        # `load_model` still auto-detects encoder-decoder checkpoints stored
        # under decoder-shaped SFT source chains.
        model, tokenizer, meta = load_model(
            source=cfg.source,
            device=load_device,
            phase=ModelPhase.EVAL,
            model_tag=cfg.model_tag,
            step=step,
            eval_conditioning=conditioned_eval_conditioning("harmfulness_eval"),
        )
        loaded_dir = meta.get("checkpoint_dir") if isinstance(meta, Mapping) else None
        if loaded_dir != checkpoint_dir:
            raise ValueError(
                "harmfulness_eval loaded a different checkpoint directory than it "
                f"pinned: expected {checkpoint_dir!r}, got {loaded_dir!r}"
            )
        contract = assert_conditioned_eval_model(
            model,
            meta,
            checkpoint_dir=checkpoint_dir,
            eval_name="harmfulness_eval",
        )

    if stage_on_cpu:
        # BF16 transfer avoids placing the loader's temporary FP32 model on
        # CUDA alongside its checkpoint tensors on a shared 24 GB device.
        model = model.to(device=device, dtype=t.bfloat16)
    loaded_checkpoint_dir = meta.get("checkpoint_dir") if isinstance(meta, Mapping) else None
    if not isinstance(loaded_checkpoint_dir, str) or not loaded_checkpoint_dir:
        raise ValueError("harmfulness_eval loader metadata is missing a non-empty checkpoint_dir")
    loaded_root = Path(loaded_checkpoint_dir).resolve()
    if loaded_root != artifacts.checkpoint_dir:
        raise ValueError(
            "harmfulness_eval loader returned a different checkpoint directory "
            f"than the identity preflight: expected={artifacts.checkpoint_dir}, "
            f"loaded={loaded_root}"
        )
    artifacts.assert_unchanged()
    model_identity = native_model_identity(
        artifacts,
        architecture=type(model).__name__,
        model_type=(ENCODER_DECODER_MODEL_TYPE if isinstance(model, EncoderDecoderGPT) else DECODER_ONLY_MODEL_TYPE),
        effective_metadata=meta,
    )
    meta["resolved_step"] = artifacts.step
    meta["loaded_model_sha256"] = next(
        row.sha256 for row in artifacts.checkpoint_inventory.entries if row.path.suffix == ".pt"
    )
    return model, tokenizer, meta, contract, model_identity


def _render_harmfulness_prompt_for_coverage(tokenizer, prompt: str) -> tuple[str, list[int]]:
    """Return the exact chat-rendered prompt text/token ids used for generation."""
    rendered = render_prompt_for_chat_completion(tokenizer, prompt)
    return tokenizer.decode(rendered.token_ids), rendered.token_ids


def _expected_harmfulness_context_records(cfg, tokenizer) -> list[EvalContextRecord]:
    """The exact contexts generation will look up, from generation's own helpers.

    Deliberately derived from ``_load_expected_prompt_pairs`` and
    ``_render_harmfulness_prompt_for_coverage`` rather than from a separate
    enumeration: a second derivation could drift from the first, and a drifted
    coverage check would pass while every lookup missed.
    """
    records: list[EvalContextRecord] = []
    for index, pair in enumerate(_load_expected_prompt_pairs(cfg)):
        prompt_text, token_ids = _render_harmfulness_prompt_for_coverage(
            tokenizer, str(pair["prompt"])
        )
        records.append(
            EvalContextRecord(
                suite="harmfulness",
                task="harmfulness",
                seed=int(cfg.seed),
                example_index=index,
                token_ids=tuple(int(token) for token in token_ids),
                context_text=prompt_text,
                metadata={"pool": pair.get("pool")},
            )
        )
    return records


def _generate_responses(
    cfg,
    responses_path: str,
    *,
    condition_name: str,
    response_generator: ResponseGenerator,
    conditioning: dict[str, Any],
    generation: dict[str, Any],
) -> list[dict[str, object]]:
    """Generate responses through one callback and atomically persist them."""
    logger.info("Loading prompts...")
    all_prompts = _load_expected_prompt_pairs(cfg)
    batch_size = int(generation["batch_size"])

    logger.info(f"Generating responses for {len(all_prompts)} prompts in batches of {batch_size}...")
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        items = [(i + offset, pair) for offset, pair in enumerate(batch)]
        responses = response_generator(items)
        if len(responses) != len(items):
            raise ValueError(
                f"[{condition_name}] response generator returned {len(responses)} "
                f"responses for {len(items)} prompts"
            )
        for (index, row), response in zip(items, responses, strict=True):
            if not isinstance(response, str):
                raise ValueError(
                    f"[{condition_name}] response generator response for prompt "
                    f"{index} must be a string, got {type(response).__name__}"
                )
            all_prompts[index] = {**row, "response": response}

        done = min(i + batch_size, len(all_prompts))
        if done % 100 < batch_size:
            logger.info(f"  Generated {done}/{len(all_prompts)}")

    artifact = build_harmfulness_responses_artifact(
        responses=all_prompts,
        conditioning=conditioning,
        generation=generation,
    )
    write_harmfulness_responses_artifact(artifact, responses_path)

    return all_prompts


def _load_expected_prompt_pairs(cfg) -> list[dict[str, Any]]:
    """Resolve the exact ordered prompt pool selected by the current config.

    This is shared by generation and response-cache validation. Reconstructing
    the pool from the pinned loaders prevents a legacy cache whose prompts were
    merely labelled ``pool: benign`` from being scored as benign compliance.
    """
    pool = resolve_harmfulness_prompt_pool(
        seed=cfg.seed,
        max_harmful=cfg.get("num_harmful"),
        num_benign=cfg.num_benign,
    )
    return [dict(row) for row in pool.rows]


def _load_responses(
    path: str,
    cfg,
    *,
    expected_conditioning: dict[str, Any] | None = None,
    expected_generation: dict[str, Any] | None = None,
) -> LoadedResponses:
    """Load responses bound to prompt identity and generation provenance."""
    try:
        with open(path, encoding="utf-8") as f:
            raw_artifact = yaml.safe_load(f)
        pairs, conditioning, generation = parse_harmfulness_responses_artifact(raw_artifact)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Response cache {path} has invalid generation provenance: {exc}") from exc
    if expected_conditioning is not None and conditioning != expected_conditioning:
        raise ValueError(
            f"Response cache {path} conditioning provenance does not match the "
            "current generation run"
        )
    if expected_generation is not None and generation != expected_generation:
        raise ValueError(
            f"Response cache {path} generation provenance does not match the current generation run"
        )

    expected_pairs = _load_expected_prompt_pairs(cfg)
    validate_generation_provenance(generation, prompt_count=len(expected_pairs))
    if len(pairs) != len(expected_pairs):
        raise ValueError(
            f"Response cache {path} prompt count {len(pairs)} does not match "
            f"the current pinned prompt pool count {len(expected_pairs)}"
        )

    validated: list[dict[str, object]] = []
    for index, (pair, expected) in enumerate(zip(pairs, expected_pairs, strict=True)):
        validated.append(validate_harmfulness_response_record(
            pair,
            expected,
            index=index,
            source=f"Response cache {path}",
            require_exact_identity=False,
        ))
    logger.info(f"Loaded {len(pairs)} pre-generated responses from {path}")
    return validated, conditioning, generation


def _get_responses(
    cfg,
    condition_name: str,
    response_generator: ResponseGenerator | None,
    *,
    generation_conditioning: dict[str, Any] | None = None,
    generation_provenance: dict[str, Any] | None = None,
    loaded_responses: LoadedResponses | None = None,
) -> tuple[str, str, list[dict[str, object]], dict[str, Any], dict[str, Any]]:
    """Get responses for a single config (generate or load)."""
    config_name = condition_name

    responses_path, output_path = resolve_harmfulness_paths(
        source=cfg.source,
        model_tag=cfg.model_tag,
        output_file=cfg.output_file,
        responses_file=cfg.responses_file,
        hf_checkpoint_dir=cfg.get("hf_checkpoint_dir"),
        segment_size=cfg.get("segment_size"),
        config_name=config_name,
        feature_selection=cfg.get("feature_selection"),
        mask_leaf=feature_mask_leaf_from_cfg(cfg),
    )
    logger.info(f"[{config_name}] Responses path: {responses_path}")
    logger.info(f"[{config_name}] Output path: {output_path}")
    if cfg.responses_file:
        pairs, conditioning, generation = (
            loaded_responses if loaded_responses is not None else _load_responses(cfg.responses_file, cfg)
        )
    elif cfg.judge_only:
        if not os.path.exists(responses_path):
            logger.error(
                f"judge_only mode: expected responses at {responses_path} "
                "but file does not exist. Run without judge_only first to generate."
            )
            sys.exit(1)
        pairs, conditioning, generation = _load_responses(responses_path, cfg)
    elif response_generator is not None:
        if generation_conditioning is None or generation_provenance is None:
            raise ValueError(
                "Fresh harmfulness generation requires conditioning and generation provenance "
                "before response-cache reuse or generation"
            )
        # Reuse cached responses if available (enables resume after interruption).
        if os.path.exists(responses_path):
            try:
                pairs, conditioning, generation = _load_responses(
                    responses_path,
                    cfg,
                    expected_conditioning=generation_conditioning,
                    expected_generation=generation_provenance,
                )
            except ValueError as exc:
                logger.warning(
                    f"[{config_name}] Cached responses are stale or invalid; regenerating: {exc}"
                )
                pairs = _generate_responses(
                    cfg,
                    responses_path,
                    condition_name=condition_name,
                    response_generator=response_generator,
                    conditioning=generation_conditioning,
                    generation=generation_provenance,
                )
                conditioning = generation_conditioning
                generation = generation_provenance
            else:
                logger.info(f"[{config_name}] Found current cached responses, skipping generation")
        else:
            pairs = _generate_responses(
                cfg,
                responses_path,
                condition_name=condition_name,
                response_generator=response_generator,
                conditioning=generation_conditioning,
                generation=generation_provenance,
            )
            conditioning = generation_conditioning
            generation = generation_provenance
    else:
        logger.error("Must set a model selector to generate, or responses_file to load")
        sys.exit(1)

    # Judge-only inputs report the artifact's own scientific condition label.
    return generation["condition_name"], output_path, pairs, conditioning, generation


def _judge_and_write(
    cfg,
    config_name: str,
    output_path: str,
    pairs: list[dict[str, object]],
    client: JudgeClient,
    *,
    conditioning: dict[str, Any],
    generation: dict[str, Any],
) -> dict:
    """Judge responses and write results. CPU/API only — no GPU needed."""
    logger.info(f"[{config_name}] Judging {len(pairs)} responses...")
    judge_pairs = cast(list[dict[str, str]], pairs)
    results = client.score_all(judge_pairs)
    if len(results) != len(judge_pairs):
        raise ValueError(
            f"Harmfulness judge returned {len(results)} results for {len(judge_pairs)} prompts"
        )
    metrics = compute_metrics(judge_pairs, results)
    benign_pairs = [pair for pair in judge_pairs if pair["pool"] == "benign"]
    compliance_results = client.score_compliance_all(benign_pairs)
    if len(compliance_results) != len(benign_pairs):
        raise ValueError(
            "Benign compliance judge returned "
            f"{len(compliance_results)} results for {len(benign_pairs)} prompts"
        )
    compliance_metrics = compute_compliance_metrics(
        benign_pairs,
        compliance_results,
    )

    run_config = {
        "source": cfg.source,
        "model_tag": cfg.model_tag,
        "step": cfg.step,
        "judge_model": cfg.judge_model,
        "feature_configs": generation["condition_name"],
        "feature_conditions": generation["feature_conditions"],
        "num_harmful": sum(1 for p in pairs if p["pool"] == "harmful"),
        "num_benign": sum(1 for p in pairs if p["pool"] == "benign"),
        "conditioning": conditioning,
        "generation": generation,
    }
    payload = build_harmfulness_payload(
        metrics=metrics,
        compliance_metrics=compliance_metrics,
        model=str(cfg.judge_model),
        pairs=judge_pairs,
        config=run_config,
    )
    write_harmfulness_payload(payload, output_path)
    logger.info(f"[{config_name}] Results written to {output_path}")

    return payload


def main() -> None:
    """Generate or load one condition, then reuse or compute its judgments."""
    cfg = _compose_cli_config(sys.argv[1:])

    if isinstance(cfg, Mapping) and "threshold" in cfg:
        raise ValueError(
            "harmfulness_eval.threshold was removed; express threshold checkpoints "
            "through feature_selection={threshold_type:segment_topk,threshold:<value>} "
            "with segment_size set"
        )
    generate_only = cfg.get("generate_only", False)
    if type(generate_only) is not bool:
        raise ValueError("generate_only must be a boolean")
    if generate_only and (cfg.get("judge_only") or cfg.get("responses_file")):
        raise ValueError("generate_only cannot be combined with judge_only or responses_file")
    hf_checkpoint_dir = _resolve_hf_checkpoint_dir(cfg)
    is_hf = hf_checkpoint_dir is not None
    generation_batch_size = _resolve_generation_batch_size(cfg, is_hf=is_hf)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    feature_configs_configured = cfg.get("feature_configs") is not None and not is_hf
    safety_settings = parse_harmfulness_safety_settings(cfg)
    _validate_harmfulness_generation_config(
        feature_configs_configured=feature_configs_configured,
        self_cond_segment_size=cfg.get("self_cond_segment_size"),
        # Decoder-shaped source chains can contain conditioned SFT checkpoints.
        # Their architecture is unknown until the loader inspects metadata.
        is_encdec=(
            True
            if safety_settings is not None or (bool(cfg.source) and _is_encdec_root(str(cfg.source)))
            else False if is_hf else None
        ),
        batch_size=generation_batch_size,
        safety_steering_configured=safety_settings is not None,
        generating=not (cfg.judge_only or cfg.responses_file),
    )
    # Load model once if we need to generate.
    model = None
    tokenizer = None
    device = None
    checkpoint_meta: dict | None = None
    hf_loaded: RepositoryHFCheckpoint | None = None
    context_provider: SafetensorsContextFeatureProvider | None = None
    prompt_coverage: PromptCoverageSummary | None = None
    checkpoint_contract = None
    checkpoint_identity: dict[str, Any] | None = None
    self_cond_setup: HarmfulnessSelfCondSetup | None = None

    if (cfg.source or hf_checkpoint_dir) and not cfg.judge_only and not cfg.responses_file:
        device_type = autodetect_device_type()
        device = t.device(device_type)
        if is_hf:
            assert hf_checkpoint_dir is not None
            hf_loaded = load_repository_hf_checkpoint(
                hf_checkpoint_dir,
                device=device,
            )
            model = hf_loaded.model
            checkpoint_identity = hf_loaded.model_identity
        else:
            model, tokenizer, meta, checkpoint_contract, checkpoint_identity = (
                _load_model_for_harmfulness(cfg, device)
            )
            checkpoint_meta = meta
        if device.type == "cuda" and not is_hf and not isinstance(model, EncoderDecoderGPT):
            # AUTOCAST_OFFPATTERN: harmfulness_eval aligns model parameter dtype with the
            # InferenceEngine's hardcoded bf16 KV cache on CUDA. Without this explicit
            # cast, flash_attn raises a dtype mismatch error when the pre-allocated bf16
            # cache meets fp32 KV tensors. See docs/architecture.md "Mixed Precision
            # Contract" + tests/test_autocast_canonical.py.
            model = cast(Any, model).to(dtype=t.bfloat16)

        if isinstance(model, EncoderDecoderGPT):
            assert tokenizer is not None and checkpoint_meta is not None
            assert checkpoint_contract is not None and checkpoint_identity is not None
            self_cond_setup = prepare_harmfulness_self_conditioning(
                cfg, model=model, tokenizer=tokenizer, device=device, meta=checkpoint_meta,
                feature_contract=checkpoint_contract,
                prompt_token_ids=[_render_harmfulness_prompt_for_coverage(tokenizer, str(pair["prompt"]))[1]
                    for pair in _load_expected_prompt_pairs(cfg)],
            )
            if not isinstance(self_cond_setup.context_provider, SafetensorsContextFeatureProvider):
                raise ValueError(
                    "harmfulness_eval loaded an encoder-decoder model but resolved no "
                    "context-feature provider. A persisted context cache is required "
                    "to authenticate prompt coverage and generation provenance."
                )
            context_provider = self_cond_setup.context_provider

    loaded_responses: LoadedResponses | None = None
    if cfg.responses_file:
        loaded_responses = _load_responses(cfg.responses_file, cfg)
        is_encdec = loaded_responses[2]["completion"] is not None
    elif model is not None:
        is_encdec = isinstance(model, EncoderDecoderGPT)
    elif is_hf:
        is_encdec = False
    elif _is_encdec_root(str(cfg.source)):
        is_encdec = True
    else:
        checkpoint_dir, step = _resolve_native_checkpoint_for_harmfulness(cfg)
        metadata_path = native_checkpoint_metadata_path(checkpoint_dir, step)
        try:
            metadata = json.loads(metadata_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read judge-only checkpoint metadata: {metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"judge-only checkpoint metadata must be a mapping: {metadata_path}")
        is_encdec = meta_looks_encdec(metadata)
    _validate_harmfulness_generation_config(
        feature_configs_configured=feature_configs_configured,
        self_cond_segment_size=cfg.get("self_cond_segment_size"),
        is_encdec=is_encdec,
        batch_size=generation_batch_size,
        safety_steering_configured=safety_settings is not None,
        generating=model is not None,
    )
    if loaded_responses is not None:
        condition_name = loaded_responses[2]["condition_name"]
    elif is_encdec:
        condition_name = (
            HarmfulnessConditionName.SELF_CONDITIONED
            if safety_settings is None else HarmfulnessConditionName.SAFETY_STEERED
        ).value
    else:
        condition_name = HarmfulnessConditionName.UNCONDITIONAL.value
    logger.info(f"Resolved condition: {condition_name}")
    if context_provider is not None:
        assert tokenizer is not None
        prompt_coverage = assert_exact_prompt_coverage(
            context_provider,
            _expected_harmfulness_context_records(cfg, tokenizer),
            surface="harmfulness_eval",
        )
    generation_conditioning = (
        conditioning_provenance(
            model,
            context_provider,
            prompt_coverage,
            eval_name="harmfulness_eval",
        )
        if model is not None
        else None
    )
    batch_engine: InferenceEngine | HuggingFaceChatGenerationEngine | None = None
    if hf_loaded is not None:
        batch_engine = HuggingFaceChatGenerationEngine(hf_loaded)
    elif model is not None and self_cond_setup is None:
        assert tokenizer is not None and device is not None
        batch_engine = InferenceEngine(model, tokenizer, device)
    response_generator: ResponseGenerator | None = None
    completion = (
        None if self_cond_setup is None else harmfulness_completion_provenance(self_cond_setup)
    )
    run_generation: dict[str, Any] | None = None
    if completion is not None:
        generation_seed_schedule = SELF_CONDITIONED_SEED_SCHEDULE
    elif is_hf:
        generation_seed_schedule = HF_BATCHED_SEED_SCHEDULE
    else:
        generation_seed_schedule = SEED_SCHEDULE
    if batch_engine is not None:

        def response_generator(
            batch: list[tuple[int, dict[str, Any]]],
            *,
            engine: InferenceEngine | HuggingFaceChatGenerationEngine = batch_engine,
        ) -> list[str]:
            batch_start_index = batch[0][0]
            return engine.generate_batch(
                [str(pair["prompt"]) for _, pair in batch],
                max_new_tokens=int(cfg.max_new_tokens),
                temperature=float(cfg.temperature),
                top_k=None if cfg.top_k is None else int(cfg.top_k),
                seed=absolute_prompt_batch_seed(cfg.seed, batch_start_index),
                render_as_chat=True,
            )

    elif self_cond_setup is not None:
        setup = self_cond_setup

        def response_generator(
            batch: list[tuple[int, dict[str, Any]]],
            *,
            prompt_setup: HarmfulnessSelfCondSetup = setup,
        ) -> list[str]:
            if len(batch) != 1:
                raise ValueError("self-conditioned harmfulness requires batch_size=1")
            stable_index, pair = batch[0]
            responses = [
                generate_harmfulness_response_self_conditioned(
                    setup=prompt_setup,
                    stable_pool_index=stable_index,
                    prompt=str(pair["prompt"]),
                    pool=str(pair["pool"]),
                    category=str(pair["category"]),
                    base_seed=int(cfg.seed),
                    max_new_tokens=int(cfg.max_new_tokens),
                    temperature=float(cfg.temperature),
                    top_k=None if cfg.top_k is None else int(cfg.top_k),
                )
            ]
            prompt_setup.assert_target_unchanged()
            return responses

    if model is not None:
        assert checkpoint_identity is not None
        assert device is not None
        run_generation = build_generation_provenance(
            condition_name=condition_name,
            model=checkpoint_identity,
            device_type=cast(Literal["cpu", "cuda", "mps"], device.type),
            inference_dtype=("bfloat16" if is_hf or device.type == "cuda" else "float32"),
            feature_conditions=None,
            target_conditioning_mode=("next" if completion is not None else "none"),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            batch_size=generation_batch_size,
            seed=cfg.seed,
            prompt_count=len(_load_expected_prompt_pairs(cfg)),
            seed_schedule=generation_seed_schedule,
            context_cache_manifest_sha256=(
                None
                if context_provider is None
                else context_cache_manifest_sha256(context_provider.manifest)
            ),
            completion=completion,
        )

    # Generate (GPU-bound).
    config_name, output_path, pairs, run_conditioning, loaded_generation = _get_responses(
        cfg,
        condition_name,
        response_generator,
        generation_conditioning=generation_conditioning,
        generation_provenance=run_generation,
        **({"loaded_responses": loaded_responses} if loaded_responses is not None else {}),
    )

    if self_cond_setup is not None:
        self_cond_setup.assert_target_unchanged()
    if safety_settings is not None:
        expected_safety = {
            "targets_sha256": stream_hash_file_descriptor_bound(safety_settings.targets).sha256,
            "strength": safety_settings.strength,
        }
        loaded_completion = loaded_generation["completion"]
        if loaded_completion is None or loaded_completion.get("safety_steering") != expected_safety:
            raise ValueError("response safety steering receipt differs from the requested bundle or strength")
    if model is None:
        loaded_completion = loaded_generation["completion"]
        if (loaded_completion is not None) != is_encdec:
            raise ValueError("response receipt model family differs from the selected checkpoint")
        if loaded_completion is not None and loaded_generation["model"]["model_type"] != ENCODER_DECODER_MODEL_TYPE:
            raise ValueError("self-conditioned response receipt requires an encoder-decoder model")
        cadence_assertion = cfg.get("self_cond_segment_size")
        if cadence_assertion is not None:
            if loaded_completion is None:
                raise ValueError("decoder-only harmfulness requires self_cond_segment_size=null")
            if cadence_assertion != loaded_completion["self_cond_segment_size"]:
                raise ValueError("self_cond_segment_size differs from the response receipt cadence")
    if generate_only:
        logger.info(f"[{config_name}] Published {len(pairs)} responses; generation-only run complete")
        return

    # Resume only when the result matches the exact current judgment
    # contract and ordered response pairs.
    current_payload, resume_reason = current_harmfulness_payload(
        output_path,
        model=str(cfg.judge_model),
        pairs=pairs,
        expected_conditioning=run_conditioning,
        expected_generation=loaded_generation,
    )
    if current_payload is not None:
        logger.info(f"[{config_name}] Current results already exist, skipping judging")
        return
    if os.path.exists(output_path):
        logger.warning(
            f"[{config_name}] Existing results are stale or invalid; rejudging: {resume_reason}"
        )

    client = JudgeClient(
        model=cfg.judge_model,
        cache_path=cfg.cache_path or None,
        requests_per_minute=cfg.requests_per_minute,
    )
    metrics = _judge_and_write(
        cfg,
        config_name,
        output_path,
        pairs,
        client,
        conditioning=run_conditioning,
        generation=loaded_generation,
    )
    logger.info(f"[{config_name}] Results Summary")
    logger.info(f"  Overall mean score: {metrics['overall']['mean_score']:.3f}")
    logger.info(f"  Overall harmfulness rate: {metrics['overall']['harmfulness_rate']:.3f}")
    logger.info(f"  Overall safe response rate: {metrics['overall']['safe_response_rate']:.3f}")
    if "asr" in metrics.get("harmful_prompts", {}):
        logger.info(f"  ASR (harmful prompts): {metrics['harmful_prompts']['asr']:.3f}")


if __name__ == "__main__":
    main()
