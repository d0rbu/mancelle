"""Run one vendored AxBench baseline stage in its isolated uv environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "vendor" / "axbench"
DEFAULT_PREPARED_DIR = REPO_ROOT / "out" / "axbench" / "prepared"
AXBENCH_UPSTREAM_COMMIT = "41c8332543e5a631f9a8c0a9df38799893ace758"
REQUIRED_PREPARED_PATHS = (
    Path("generate/metadata.jsonl"),
    Path("generate/train_data.parquet"),
    Path("generate/latent_eval_data.parquet"),
    Path("data/alpaca_eval.json"),
    Path("data/seed_sentences"),
    Path("data/seed_instructions"),
    Path("stanza/resources.json"),
    Path("prepare_manifest.json"),
)
RUN_SETTING_KEYS = (
    "schema_version",
    "vendored_axbench_upstream_commit",
    "target_model",
    "dataset_revision",
    "seed",
    "max_concepts",
    "nproc_per_node",
    "prepared_dir",
    "prepared_manifest",
    "config_name",
    "config",
    "package_lock_sha256",
)

# These mirror the control flow in vendored axbench/scripts/inference.py. Models
# skipped by steering, or excluded from latent inference, do not consume the
# merged latent parquet during steering.
STEERING_SKIPPED_MODELS = frozenset(
    {"IntegratedGradients", "InputXGradients", "PromptDetection", "BoW"}
)
LATENT_EXCLUDED_MODELS = frozenset(
    {"PromptSteering", "PromptBaseline", "DiReFT", "LoReFT", "LoRA", "SFT", "HyperSteer"}
)
LM_BACKED_STEERING_DATASETS = frozenset(
    {"AlpacaEvalSuppress", "AttackMultiShot", "AttackOverwrite", "AttackOverwritePrefix"}
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return loaded


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        assert temporary_path is not None
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_config(config_path: Path) -> dict:
    loaded = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a YAML mapping: {config_path}")
    for section in ("train", "generate", "inference"):
        if not isinstance(loaded.get(section), dict):
            raise ValueError(f"config is missing a {section!r} mapping: {config_path}")

    train_models = loaded["train"].get("models", [])
    if not isinstance(train_models, (dict, list)):
        raise ValueError(
            f"config 'train.models' must be a YAML mapping or list of model names: {config_path}"
        )
    train_model_names = list(train_models) if isinstance(train_models, dict) else train_models
    if not all(isinstance(name, str) and name for name in train_model_names):
        raise ValueError(f"config 'train.models' contains an invalid model name: {config_path}")
    if isinstance(train_models, dict) and not all(
        isinstance(settings, dict) for settings in train_models.values()
    ):
        raise ValueError(
            f"config 'train.models' values must be YAML mappings of model settings: {config_path}"
        )

    _string_list(loaded["inference"].get("models"), "inference.models", config_path)
    _string_list(
        loaded["inference"].get("steering_datasets", []),
        "inference.steering_datasets",
        config_path,
    )
    return json.loads(json.dumps(loaded, allow_nan=False))


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _git_worktree_state() -> dict:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        text=True,
    )
    entries = output.splitlines()
    tracked_changes = [entry for entry in entries if not entry.startswith("?? ")]
    untracked_paths = [entry[3:] for entry in entries if entry.startswith("?? ")]
    return {
        "dirty": bool(entries),
        "tracked_changes": tracked_changes,
        "untracked_paths": untracked_paths,
    }


def _string_list(value: object, field: str, config_path: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(
            f"config '{field}' must be a YAML list of non-empty strings: {config_path}"
        )
    return [str(item) for item in value]


def _prepared_concept_count(prepared_dir: Path) -> int:
    metadata_path = prepared_dir / "generate" / "metadata.jsonl"
    with metadata_path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _validate_requested_concepts(prepared_dir: Path, max_concepts: int) -> None:
    prepared_count = _prepared_concept_count(prepared_dir)
    if max_concepts > prepared_count:
        raise ValueError(
            f"--max-concepts={max_concepts} exceeds the {prepared_count} concepts in prepared "
            f"metadata at {prepared_dir / 'generate' / 'metadata.jsonl'}"
        )


def _lsreft_uses_synergy(config: dict, config_path: Path) -> bool:
    inference_models = _string_list(
        config["inference"].get("models"), "inference.models", config_path
    )
    if "LsReFT" not in inference_models:
        return False

    train_models = config["train"].get("models", [])
    if "LsReFT" not in train_models:
        raise ValueError(
            f"config 'inference.models' selects LsReFT but 'train.models' does not: {config_path}"
        )
    use_synergy = config["train"].get("use_synergy")
    if isinstance(train_models, dict):
        use_synergy = train_models["LsReFT"].get("use_synergy", use_synergy)
    if use_synergy is not None and not isinstance(use_synergy, bool):
        raise ValueError(
            f"config 'train.models.LsReFT.use_synergy' must be a boolean: {config_path}"
        )
    return use_synergy is True


def _steering_provider_reasons(config: dict, config_path: Path) -> list[str]:
    models = _string_list(config["inference"].get("models"), "inference.models", config_path)
    datasets = _string_list(
        config["inference"].get("steering_datasets", []),
        "inference.steering_datasets",
        config_path,
    )
    reasons = []
    if "PromptSteering" in models:
        reasons.append("PromptSteering")
    if _lsreft_uses_synergy(config, config_path):
        reasons.append("LsReFT with use_synergy=true")
    lm_datasets = sorted(set(datasets) & LM_BACKED_STEERING_DATASETS)
    if lm_datasets:
        reasons.append(f"LM-backed steering datasets {lm_datasets}")
    return reasons


def _steering_models_requiring_latent(config: dict, config_path: Path) -> list[str]:
    models = _string_list(config["inference"].get("models"), "inference.models", config_path)
    return sorted(
        {
            model
            for model in models
            if model not in STEERING_SKIPPED_MODELS and model not in LATENT_EXCLUDED_MODELS
        }
    )


def _validate_steering_latent_output(config: dict, config_path: Path, dump_dir: Path) -> None:
    required_by = _steering_models_requiring_latent(config, config_path)
    if not required_by:
        return
    latent_path = dump_dir / "inference" / "latent_data.parquet"
    if not latent_path.is_file():
        raise FileNotFoundError(
            f"steering methods {required_by} require max_act values from the latent stage, but "
            f"{latent_path} is missing; run the latent stage with this config and dump directory "
            "before steering"
        )


def validate_prepared_dir(prepared_dir: Path) -> dict:
    prepared_dir = prepared_dir.resolve()
    missing = [str(path) for path in REQUIRED_PREPARED_PATHS if not (prepared_dir / path).exists()]
    if missing:
        prepare_command = shlex.join(
            [
                "uv",
                "run",
                "--project",
                "vendor/axbench",
                "--frozen",
                "python",
                "vendor/axbench/prepare.py",
                "--prepared-dir",
                str(prepared_dir),
            ]
        )
        raise FileNotFoundError(
            f"prepared AxBench inputs are incomplete under {prepared_dir}: {missing}; "
            f"run `{prepare_command}`"
        )
    return _load_json(prepared_dir / "prepare_manifest.json")


def build_command(
    stage: str,
    *,
    config_path: Path,
    dump_dir: Path,
    prepared_dir: Path,
    max_concepts: int,
    nproc_per_node: int,
) -> list[str]:
    if stage not in {"train", "latent", "steering"}:
        raise ValueError(f"unsupported AxBench stage: {stage}")
    if max_concepts <= 0 or nproc_per_node <= 0:
        raise ValueError("max_concepts and nproc_per_node must be positive")

    config_path = config_path.resolve()
    dump_dir = dump_dir.resolve()
    prepared_dir = prepared_dir.resolve()
    script = (
        VENDOR_ROOT / "axbench" / "scripts" / ("train.py" if stage == "train" else "inference.py")
    )
    command = [
        "uv",
        "run",
        "--project",
        str(VENDOR_ROOT),
        "--frozen",
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(script),
        "--config",
        str(config_path),
    ]
    if stage == "train":
        command.extend(
            [
                "--dump_dir",
                str(dump_dir),
                "--overwrite_data_dir",
                str(prepared_dir / "generate"),
                "--max_concepts",
                str(max_concepts),
            ]
        )
    else:
        command.extend(
            [
                "--mode",
                stage,
                "--dump_dir",
                str(dump_dir),
                "--overwrite_metadata_dir",
                str(prepared_dir / "generate"),
                "--overwrite_inference_data_dir",
                str(prepared_dir / "generate"),
                "--master_data_dir",
                str(prepared_dir / "data"),
                "--max_concepts",
                str(max_concepts),
            ]
        )
    return command


def ensure_run_manifest(
    dump_dir: Path,
    *,
    stage: str,
    config_path: Path,
    prepared_dir: Path,
    max_concepts: int,
    nproc_per_node: int,
) -> Path:
    if stage not in {"train", "latent", "steering"}:
        raise ValueError(f"unsupported AxBench stage: {stage}")
    dump_dir = dump_dir.resolve()
    config_path = config_path.resolve()
    prepared_dir = prepared_dir.resolve()
    config = _load_config(config_path)
    prepared_manifest = validate_prepared_dir(prepared_dir)
    _validate_requested_concepts(prepared_dir, max_concepts)
    source_provenance: dict[str, Any] = {
        "repository_git_commit": _git_commit(),
        "repository_git_worktree": _git_worktree_state(),
    }
    manifest = {
        "schema_version": 1,
        **source_provenance,
        "vendored_axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
        "target_model": config["train"].get("model_name"),
        "dataset_revision": prepared_manifest.get("benchmark", {}).get("revision"),
        "seed": config["train"].get("seed"),
        "max_concepts": max_concepts,
        "nproc_per_node": nproc_per_node,
        "prepared_dir": str(prepared_dir),
        "prepared_manifest": prepared_manifest,
        "config_name": config_path.name,
        "config": config,
        "package_lock_sha256": _sha256(VENDOR_ROOT / "uv.lock"),
        "stage_provenance": {stage: source_provenance},
    }

    dump_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dump_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        changed_settings = [
            key for key in RUN_SETTING_KEYS if key not in existing or existing[key] != manifest[key]
        ]
        if changed_settings:
            raise RuntimeError(
                f"run settings {changed_settings} differ from the existing manifest at "
                f"{manifest_path}; "
                "use a new --dump-dir"
            )
        manifest_changed = False
        if "stage_provenance" not in existing:
            creation_commit = existing.get("repository_git_commit")
            creation_worktree = existing.get("repository_git_worktree")
            if not isinstance(creation_commit, str) or (
                creation_worktree is not None and not isinstance(creation_worktree, dict)
            ):
                raise RuntimeError(
                    f"existing run manifest has invalid creation provenance: {manifest_path}"
                )
            stage_provenance_value = {
                "legacy_creation": {
                    "repository_git_commit": creation_commit,
                    "repository_git_worktree": creation_worktree,
                }
            }
            existing["stage_provenance"] = stage_provenance_value
            manifest_changed = True
        else:
            stage_provenance_value = existing["stage_provenance"]
        if not isinstance(stage_provenance_value, dict):
            raise RuntimeError(
                f"existing run manifest has invalid stage provenance: {manifest_path}"
            )
        stage_provenance: dict[str, Any] = stage_provenance_value
        if stage_provenance.get(stage) != source_provenance:
            stage_provenance[stage] = source_provenance
            manifest_changed = True
        if manifest_changed:
            _write_json_atomic(manifest_path, existing)
    else:
        _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "latent", "steering"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--max-concepts", type=_positive_int, required=True)
    parser.add_argument("--nproc-per-node", type=_positive_int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = args.config.resolve()
    dump_dir = args.dump_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"AxBench config not found: {config_path}")

    config = _load_config(config_path)
    if args.stage == "steering":
        provider_reasons = _steering_provider_reasons(config, config_path)
        if provider_reasons and not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for " + ", ".join(provider_reasons)
            )
        _validate_steering_latent_output(config, config_path, dump_dir)

    ensure_run_manifest(
        dump_dir,
        stage=args.stage,
        config_path=config_path,
        prepared_dir=prepared_dir,
        max_concepts=args.max_concepts,
        nproc_per_node=args.nproc_per_node,
    )
    command = build_command(
        args.stage,
        config_path=config_path,
        dump_dir=dump_dir,
        prepared_dir=prepared_dir,
        max_concepts=args.max_concepts,
        nproc_per_node=args.nproc_per_node,
    )
    env = os.environ.copy()
    env["STANZA_RESOURCES_DIR"] = str(prepared_dir / "stanza")
    logger.info("Running vendored AxBench: {}", " ".join(command))
    subprocess.run(command, cwd=VENDOR_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
