"""Download the ordinary data/resources used by the vendored AxBench runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
import stanza
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = REPO_ROOT / "out" / "axbench" / "prepared"

AXBENCH_UPSTREAM_COMMIT = "41c8332543e5a631f9a8c0a9df38799893ace758"
AXBENCH_METADATA_URL = (
    "https://raw.githubusercontent.com/stanfordnlp/axbench/"
    f"{AXBENCH_UPSTREAM_COMMIT}/axbench/concept500/prod_2b_l20_v1/"
    "generate/metadata.jsonl"
)
AXBENCH_METADATA_SHA256 = "435aad14e202402b0989d62b279f63dfc63aa2aff0e488cf62652c0700a1efdc"
METADATA_DOWNLOAD_TIMEOUT_SECONDS = 60
OWNERSHIP_MARKER_NAME = ".axbench-prepared"
OWNERSHIP_MARKER_CONTENT = "nano-gpt-pretrain-steer AxBench prepared data\n"

BENCHMARK_REPO = "pyvene/axbench-concept500"
BENCHMARK_REVISION = "ad8a5d60c4616b599c24dd6689f05f696ec610f3"
BENCHMARK_FILES = {
    "2b/l20/train/data.parquet": "train_data.parquet",
    "2b/l20/test/data.parquet": "latent_eval_data.parquet",
}

ALPACA_REPO = "tatsu-lab/alpaca_eval"
ALPACA_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
ALPACA_FILE = "alpaca_eval.json"

SEED_REVISIONS = {
    "zhengxuanzenwu/wikitext-2-split-128": "817f68fefc4d740360dded88d91f53089f21c10d",
    "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "qwedsacf/competition_math": "e839825f9ec5c6cfa585c654a59610969ec13993",
    "christopher/rosetta-code": "11d8b38cbd90b0ae5bde304ae2d4ba5ace10af05",
    "databricks/databricks-dolly-15k": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    "iamtarun/python_code_instructions_18k_alpaca": ("7cae181e29701a8663a07a3ea43c8e105b663ba1"),
}

STANZA_VERSION = "1.14.0"
STANZA_RESOURCES_COMMIT = "c543b0df3e1aa5703300318202f2f9b0e7b058a1"
STANZA_MODELS_COMMIT = "ae692a7443f4e746bd3da8baf82c003bf92a3585"
STANZA_RESOURCES_URL = (
    f"https://raw.githubusercontent.com/stanfordnlp/stanza-resources/{STANZA_RESOURCES_COMMIT}"
)
STANZA_MODEL_URL = (
    "https://huggingface.co/stanfordnlp/stanza-en/resolve/"
    f"{STANZA_MODELS_COMMIT}/models/{{filename}}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(repo_id: str, revision: str, filename: str, destination: Path) -> None:
    cached = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)


def _download_metadata(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(  # noqa: S310
        AXBENCH_METADATA_URL,
        timeout=METADATA_DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        destination.write_bytes(response.read())
    actual = _sha256(destination)
    if actual != AXBENCH_METADATA_SHA256:
        raise RuntimeError(
            f"AxBench metadata digest mismatch: expected {AXBENCH_METADATA_SHA256}, got {actual}"
        )


def _seed_sentences(destination: Path) -> None:
    wiki = load_dataset(
        "zhengxuanzenwu/wikitext-2-split-128",
        revision=SEED_REVISIONS["zhengxuanzenwu/wikitext-2-split-128"],
    )
    ag_news = load_dataset("fancyzhx/ag_news", revision=SEED_REVISIONS["fancyzhx/ag_news"])
    gsm8k = load_dataset("openai/gsm8k", "main", revision=SEED_REVISIONS["openai/gsm8k"])
    competition_math = load_dataset(
        "qwedsacf/competition_math",
        revision=SEED_REVISIONS["qwedsacf/competition_math"],
    )
    rosetta = load_dataset(
        "christopher/rosetta-code",
        revision=SEED_REVISIONS["christopher/rosetta-code"],
    )

    code = [row["code"][:500] for row in rosetta["train"] if len(row["code"]) > 500]
    midpoint = len(code) // 2
    dataset = DatasetDict(
        {
            "text_train": Dataset.from_dict(
                {"input": [row["text"] for row in wiki["train"]][:1000]}
            ),
            "text_test": Dataset.from_dict(
                {"input": [row["text"] for row in ag_news["train"]][:1000]}
            ),
            "math_train": Dataset.from_dict(
                {"input": [row["answer"] for row in gsm8k["train"]][:1000]}
            ),
            "math_test": Dataset.from_dict(
                {"input": [row["problem"] for row in competition_math["train"]][:1000]}
            ),
            "code_train": Dataset.from_dict({"input": code[:midpoint][:1000]}),
            "code_test": Dataset.from_dict({"input": code[midpoint:][:1000]}),
        }
    )
    dataset.save_to_disk(destination)


def _seed_instructions(destination: Path) -> None:
    dolly = load_dataset(
        "databricks/databricks-dolly-15k",
        revision=SEED_REVISIONS["databricks/databricks-dolly-15k"],
    )
    gsm8k = load_dataset("openai/gsm8k", "main", revision=SEED_REVISIONS["openai/gsm8k"])
    python_instructions = load_dataset(
        "iamtarun/python_code_instructions_18k_alpaca",
        revision=SEED_REVISIONS["iamtarun/python_code_instructions_18k_alpaca"],
    )

    text = [
        row["instruction"]
        for row in dolly["train"]
        if row["category"] == "open_qa" and row["context"] == "" and len(row["instruction"]) < 500
    ]
    math = [row["question"] for row in gsm8k["train"] if len(row["question"]) < 500]
    code = [
        row["instruction"]
        for row in python_instructions["train"]
        if row["input"] == "" and len(row["instruction"]) < 500
    ]
    dataset = DatasetDict(
        {
            "text_train": Dataset.from_dict({"input": text[:1000]}),
            "text_test": Dataset.from_dict({"input": text[1000:2000]}),
            "math_train": Dataset.from_dict({"input": math[:1000]}),
            "math_test": Dataset.from_dict({"input": math[1000:2000]}),
            "code_train": Dataset.from_dict({"input": code[:1000]}),
            "code_test": Dataset.from_dict({"input": code[1000:2000]}),
        }
    )
    dataset.save_to_disk(destination)


def _seed_summary(data_dir: Path) -> dict[str, dict[str, int]]:
    summary = {}
    for name in ("seed_sentences", "seed_instructions"):
        dataset = load_from_disk(data_dir / name)
        split_counts = {split: len(dataset[split]) for split in sorted(dataset.keys())}
        if not split_counts or any(count <= 0 for count in split_counts.values()):
            raise RuntimeError(f"prepared seed dataset is empty: {name}")
        summary[name] = split_counts
    return summary


def _validate(prepared_dir: Path) -> dict:
    manifest_path = prepared_dir / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported prepare manifest")
    if manifest.get("axbench_upstream_commit") != AXBENCH_UPSTREAM_COMMIT:
        raise RuntimeError("prepared AxBench revision does not match this checkout")
    if manifest.get("benchmark", {}).get("revision") != BENCHMARK_REVISION:
        raise RuntimeError("prepared benchmark revision does not match this checkout")
    if manifest.get("alpaca_eval", {}).get("revision") != ALPACA_REVISION:
        raise RuntimeError("prepared AlpacaEval revision does not match this checkout")
    if manifest.get("seed_revisions") != SEED_REVISIONS:
        raise RuntimeError("prepared seed revisions do not match this checkout")
    if manifest.get("stanza") != {
        "version": STANZA_VERSION,
        "resources_commit": STANZA_RESOURCES_COMMIT,
        "models_commit": STANZA_MODELS_COMMIT,
        "processors": ["tokenize", "pos"],
    }:
        raise RuntimeError("prepared Stanza resources do not match this checkout")

    generate_dir = prepared_dir / "generate"
    metadata_path = generate_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    if len(rows) != 500:
        raise RuntimeError(f"prepared metadata has {len(rows)} rows, expected 500")
    if _sha256(metadata_path) != AXBENCH_METADATA_SHA256:
        raise RuntimeError("prepared metadata digest does not match the selected AxBench snapshot")

    for name in ("train_data.parquet", "latent_eval_data.parquet"):
        if pd.read_parquet(generate_dir / name).empty:
            raise RuntimeError(f"prepared parquet is empty: {name}")

    alpaca = json.loads((prepared_dir / "data" / ALPACA_FILE).read_text())
    if not isinstance(alpaca, list) or not alpaca:
        raise RuntimeError("prepared AlpacaEval data is empty or malformed")
    seed_summary = _seed_summary(prepared_dir / "data")

    stanza_dir = prepared_dir / "stanza"
    if not (stanza_dir / "resources.json").is_file():
        raise RuntimeError("prepared Stanza resources are missing")
    if not (stanza_dir / "en" / "tokenize").is_dir():
        raise RuntimeError("prepared Stanza tokenizer is missing")
    if not (stanza_dir / "en" / "pos").is_dir():
        raise RuntimeError("prepared Stanza POS model is missing")

    if manifest.get("seed_rows") != seed_summary:
        raise RuntimeError("prepared seed datasets do not match the manifest")
    return manifest


def _resolve_prepared_dir(prepared_dir: Path) -> Path:
    unresolved = prepared_dir.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    return unresolved.resolve()


def _validate_force_removal_target(prepared_dir: Path, target: Path) -> None:
    unresolved = prepared_dir.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    current = Path(unresolved.anchor)
    for part in unresolved.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"refusing symlinked prepared directory path: {current}")

    repo_root = REPO_ROOT.resolve()
    repository_output_root = repo_root / "out" / "axbench"
    home = Path.home().resolve()
    cwd = Path.cwd().resolve()
    protected = {Path(target.anchor), home, cwd}
    target_contains_repo = repo_root == target or repo_root.is_relative_to(target)
    target_contains_home = home.is_relative_to(target)
    target_contains_cwd = cwd.is_relative_to(target)
    target_is_repo_source = target.is_relative_to(repo_root) and (
        target == repository_output_root or not target.is_relative_to(repository_output_root)
    )
    if (
        target in protected
        or target.parent == Path(target.anchor)
        or target_contains_repo
        or target_contains_home
        or target_contains_cwd
        or target_is_repo_source
    ):
        raise ValueError(f"refusing broad or source-tree prepared directory: {target}")


def _prepared_dir_has_rebuildable_contents(prepared_dir: Path) -> bool:
    if not prepared_dir.is_dir():
        raise ValueError(f"prepared directory target is not a directory: {prepared_dir}")
    if next(prepared_dir.iterdir(), None) is None:
        return False
    if prepared_dir == PREPARED_DIR.resolve():
        return True

    marker = prepared_dir / OWNERSHIP_MARKER_NAME
    if marker.is_file() and not marker.is_symlink():
        if marker.read_text() == OWNERSHIP_MARKER_CONTENT:
            return True

    manifest_path = prepared_dir / "prepare_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
        if (
            isinstance(manifest, dict)
            and manifest.get("schema_version") == 1
            and manifest.get("axbench_upstream_commit") == AXBENCH_UPSTREAM_COMMIT
        ):
            return True

    raise ValueError(
        f"refusing to rebuild nonempty custom directory without AxBench ownership: {prepared_dir}"
    )


def prepare(prepared_dir: Path = PREPARED_DIR, *, force: bool = False) -> Path:
    requested_prepared_dir = prepared_dir
    prepared_dir = _resolve_prepared_dir(prepared_dir)
    manifest_path = prepared_dir / "prepare_manifest.json"
    if prepared_dir.exists() and not force:
        if not manifest_path.is_file():
            raise RuntimeError(f"partial prepared data at {prepared_dir}; rerun with --force")
        try:
            _validate(prepared_dir)
        except Exception as exc:
            raise RuntimeError(
                f"prepared data at {prepared_dir} is invalid; rerun with --force"
            ) from exc
        print(f"prepared data already valid: {prepared_dir}")
        return manifest_path

    if force and prepared_dir.exists():
        if not prepared_dir.is_dir():
            raise ValueError(f"prepared directory target is not a directory: {prepared_dir}")
        if next(prepared_dir.iterdir(), None) is not None:
            _validate_force_removal_target(requested_prepared_dir, prepared_dir)
            if _prepared_dir_has_rebuildable_contents(prepared_dir):
                shutil.rmtree(prepared_dir)

    generate_dir = prepared_dir / "generate"
    data_dir = prepared_dir / "data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / OWNERSHIP_MARKER_NAME).write_text(OWNERSHIP_MARKER_CONTENT)
    generate_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _download_metadata(generate_dir / "metadata.jsonl")
    for source_name, destination_name in BENCHMARK_FILES.items():
        _download_file(
            BENCHMARK_REPO,
            BENCHMARK_REVISION,
            source_name,
            generate_dir / destination_name,
        )
    _download_file(
        ALPACA_REPO,
        ALPACA_REVISION,
        ALPACA_FILE,
        data_dir / ALPACA_FILE,
    )
    _seed_sentences(data_dir / "seed_sentences")
    _seed_instructions(data_dir / "seed_instructions")
    stanza.download(
        "en",
        model_dir=str(prepared_dir / "stanza"),
        processors="tokenize,pos",
        resources_url=STANZA_RESOURCES_URL,
        resources_version=STANZA_VERSION,
        model_url=STANZA_MODEL_URL,
    )

    seed_summary = _seed_summary(data_dir)
    manifest = {
        "schema_version": 1,
        "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
        "benchmark": {
            "repo": BENCHMARK_REPO,
            "revision": BENCHMARK_REVISION,
            "files": BENCHMARK_FILES,
        },
        "alpaca_eval": {
            "repo": ALPACA_REPO,
            "revision": ALPACA_REVISION,
            "file": ALPACA_FILE,
        },
        "seed_revisions": SEED_REVISIONS,
        "seed_rows": seed_summary,
        "stanza": {
            "version": STANZA_VERSION,
            "resources_commit": STANZA_RESOURCES_COMMIT,
            "models_commit": STANZA_MODELS_COMMIT,
            "processors": ["tokenize", "pos"],
        },
        "artifacts": {
            str(path.relative_to(prepared_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(generate_dir.iterdir())
            if path.is_file()
        }
        | {
            f"data/{ALPACA_FILE}": {
                "bytes": (data_dir / ALPACA_FILE).stat().st_size,
                "sha256": _sha256(data_dir / ALPACA_FILE),
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _validate(prepared_dir)
    print(f"prepared AxBench data: {prepared_dir}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete and rebuild the selected prepared directory",
    )
    args = parser.parse_args(argv)
    prepare(args.prepared_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
