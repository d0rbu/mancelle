"""Audit whether SFT parquets/features align to train or validation tasks.

Conditioned SFT loaders re-key extracted feature rows by the parquet
``original_index`` column, then look up features by the current task index. This
is correct only when the parquet was converted from the same task split that the
loader is iterating. This audit compares each task row against the
full-nanochat-rendered ``text`` stored in the parquet, using the same rendering
helper as ``scripts.preprocess.convert_sft_to_parquet``.

Usage:

  uv run python -m scripts.analysis.sft_parquet_alignment_audit \\
    --dataset capabilities --max-examples 100

The validation audit reads the converter's ``<train_dir>_val`` sibling unless
``--val-parquet-dir`` overrides it. A missing val conversion is recorded as
``not_converted`` and exits nonzero unless ``--allow-missing-val`` is passed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from beartype import beartype

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.preprocess.convert_sft_to_parquet import (
    render_conversation_to_full_nanochat_text,
)
from scripts.train.conditional_encdec_sft import _resolve_sft_parquet_dir
from steering.refined_types import PositiveInt, PositiveIntLike
from steering.sft_schema import SFTSplit, legacy_sft_columns, sft_split_dir
from tasks.mixtures import (
    SFTMixture,
    build_task_mixtures,
    sft_val_family,
    steered_mixture_overrides,
)

DEFAULT_OUT = Path("out/analysis/sft_parquet_alignment_audit")
MISMATCH_PREVIEW_LIMIT = 5


class AuditStatus(StrEnum):
    """Report status for one audited SFT split."""

    AUDITED = "audited"
    NOT_CONVERTED = "not_converted"


@dataclass(frozen=True)
class SplitAlignment:
    split: SFTSplit
    checked: int
    matched: int
    skipped_invalid_task_rows: int
    missing_parquet_rows: int
    mismatched: int
    first_mismatches: list[dict[str, Any]]
    status: AuditStatus = AuditStatus.AUDITED
    recovery_command: str | None = None


@dataclass(frozen=True)
class AlignmentAudit:
    dataset: str
    parquet_dir: str
    val_parquet_dir: str | None
    max_examples: PositiveIntLike
    train: SplitAlignment
    val: SplitAlignment | None


def _load_parquet_text_by_original_index(
    parquet_dir: Path,
    indices: set[int],
) -> dict[int, str]:
    rows: dict[int, str] = {}
    for path in sorted(parquet_dir.glob("shard_*.parquet")):
        schema_names = set(pq.read_schema(path).names)
        legacy_cols = legacy_sft_columns(schema_names)
        if legacy_cols or "text" not in schema_names or "original_index" not in schema_names:
            raise ValueError(
                f"{path} has columns {sorted(schema_names)}; expected the "
                "full_nanochat_text schema with 'text' and 'original_index' "
                "(current artifacts also carry 'turn_char_starts'). "
                "Regenerate with scripts.preprocess.convert_sft_to_parquet."
            )
        table = pq.read_table(path, columns=["original_index", "text"])
        original_indices = table.column("original_index").to_pylist()
        texts = table.column("text").to_pylist()
        for original_index, text in zip(original_indices, texts, strict=True):
            idx = int(original_index)
            if idx not in indices:
                continue
            rows[idx] = str(text)
    return rows


@beartype
def compare_task_to_parquet(
    *,
    task,
    split: SFTSplit,
    parquet_rows: dict[int, str],
    max_examples: PositiveIntLike,
    tokenizer,
) -> SplitAlignment:
    checked = min(len(task), max_examples)
    matched = 0
    skipped_invalid = 0
    missing = 0
    mismatched = 0
    first_mismatches: list[dict[str, Any]] = []
    for idx in range(checked):
        expected, skip_reason = render_conversation_to_full_nanochat_text(
            tokenizer,
            task[idx],
        )
        actual = parquet_rows.get(idx)
        if expected is None:
            if actual is None:
                skipped_invalid += 1
                continue
            mismatched += 1
            if len(first_mismatches) < MISMATCH_PREVIEW_LIMIT:
                first_mismatches.append(
                    {
                        "index": idx,
                        "reason": "invalid_task_row_present_in_parquet",
                        "skip_reason": skip_reason,
                        "parquet_text_prefix": actual[:200],
                    }
                )
            continue
        if actual is None:
            missing += 1
            if len(first_mismatches) < MISMATCH_PREVIEW_LIMIT:
                first_mismatches.append({"index": idx, "reason": "missing_parquet_row"})
            continue
        if expected == actual:
            matched += 1
            continue
        mismatched += 1
        if len(first_mismatches) < MISMATCH_PREVIEW_LIMIT:
            first_mismatches.append(
                {
                    "index": idx,
                    "reason": "rendered_text_mismatch",
                    "task_text_prefix": expected[:200],
                    "parquet_text_prefix": actual[:200],
                }
            )
    return SplitAlignment(
        split=split,
        checked=checked,
        matched=matched,
        skipped_invalid_task_rows=skipped_invalid,
        missing_parquet_rows=missing,
        mismatched=mismatched,
        first_mismatches=first_mismatches,
    )


def _derive_val_parquet_dir(train_parquet_dir: Path, dataset: str, val_dataset: str) -> Path:
    """Return the ``_val`` sibling of the conversion this dataset validates on.

    A steered mixture validates on its FAMILY's conversion (standard), so the
    sibling is renamed to that family; every other mixture keeps the train
    directory's own name, which may be an arbitrary operator override.
    """
    if val_dataset != dataset:
        train_parquet_dir = train_parquet_dir.with_name(val_dataset)
    return sft_split_dir(train_parquet_dir, SFTSplit.VAL)


def _not_converted_val(dataset: str) -> SplitAlignment:
    command = (
        "uv run python -m scripts.preprocess.convert_sft_to_parquet "
        f"mixture={dataset} split={SFTSplit.VAL.value}"
    )
    return SplitAlignment(
        split=SFTSplit.VAL,
        checked=0,
        matched=0,
        skipped_invalid_task_rows=0,
        missing_parquet_rows=0,
        mismatched=0,
        first_mismatches=[],
        status=AuditStatus.NOT_CONVERTED,
        recovery_command=command,
    )


@beartype
def run_audit(
    *,
    dataset: str,
    parquet_dir: Path,
    val_parquet_dir: Path | None = None,
    steered_jsonl: str | None = None,
    max_examples: PositiveIntLike,
) -> AlignmentAudit:
    try:
        dataset_mixture = SFTMixture(dataset)
    except ValueError:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {sorted(m.value for m in SFTMixture)}"
        ) from None
    val_dataset = sft_val_family(dataset_mixture).value
    train_task, val_task = build_task_mixtures(
        dataset_mixture,
        **steered_mixture_overrides(dataset_mixture, steered_jsonl),
    )
    indices = set(range(max_examples))
    parquet_rows = _load_parquet_text_by_original_index(parquet_dir, indices)
    tokenizer = get_tokenizer()
    train = compare_task_to_parquet(
        task=train_task,
        split=SFTSplit.TRAIN,
        parquet_rows=parquet_rows,
        max_examples=max_examples,
        tokenizer=tokenizer,
    )
    resolved_val_parquet_dir = (
        _derive_val_parquet_dir(parquet_dir, dataset, val_dataset)
        if val_parquet_dir is None
        else val_parquet_dir
    )
    val: SplitAlignment | None = None
    if val_task is not None:
        if not resolved_val_parquet_dir.is_dir():
            val = _not_converted_val(val_dataset)
        else:
            val_rows = _load_parquet_text_by_original_index(
                resolved_val_parquet_dir,
                indices,
            )
            val = compare_task_to_parquet(
                task=val_task,
                split=SFTSplit.VAL,
                parquet_rows=val_rows,
                max_examples=max_examples,
                tokenizer=tokenizer,
            )
    return AlignmentAudit(
        dataset=dataset,
        parquet_dir=str(parquet_dir),
        val_parquet_dir=(str(resolved_val_parquet_dir) if val_task is not None else None),
        max_examples=max_examples,
        train=train,
        val=val,
    )


def _write_summary(path: Path, audit: AlignmentAudit) -> None:
    lines = [
        "# SFT Parquet Alignment Audit",
        "",
        f"- Dataset: `{audit.dataset}`",
        f"- Parquet dir: `{audit.parquet_dir}`",
        f"- Val parquet dir: `{audit.val_parquet_dir}`",
        f"- Max examples per split: `{audit.max_examples}`",
        "",
        "| split | status | checked | matched | skipped invalid task rows | missing parquet rows | mismatched |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in [audit.train, audit.val]:
        if split is None:
            continue
        lines.append(
            f"| {split.split.value} | {split.status.value} | {split.checked} | {split.matched} | "
            f"{split.skipped_invalid_task_rows} | {split.missing_parquet_rows} | "
            f"{split.mismatched} |"
        )
    lines.append("")
    if audit.val is not None and audit.val.status is AuditStatus.NOT_CONVERTED:
        lines.append("- val: not_converted")
        lines.append(f"- Convert: `{audit.val.recovery_command}`")
        lines.append("")
    lines.append("## First Mismatches")
    for split in [audit.train, audit.val]:
        if split is None or not split.first_mismatches:
            continue
        lines.append("")
        lines.append(f"### {split.split.value}")
        lines.append("```json")
        lines.append(json.dumps(split.first_mismatches, indent=2))
        lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parquet-dir", default=None)
    parser.add_argument("--val-parquet-dir", default=None)
    parser.add_argument("--steered-jsonl", default=None)
    parser.add_argument("--allow-missing-val", action="store_true")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    max_examples = PositiveInt.parse(args.max_examples)

    parquet_dir = Path(
        _resolve_sft_parquet_dir(
            args.parquet_dir,
            get_base_dir(),
            args.dataset,
        )
    )
    audit = run_audit(
        dataset=args.dataset,
        parquet_dir=parquet_dir,
        val_parquet_dir=(Path(args.val_parquet_dir) if args.val_parquet_dir is not None else None),
        steered_jsonl=args.steered_jsonl,
        max_examples=max_examples,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.dataset
    json_path = args.out_dir / f"{stem}.json"
    md_path = args.out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(asdict(audit), indent=2) + "\n", encoding="utf-8")
    _write_summary(md_path, audit)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    if (
        audit.val is not None
        and audit.val.status is AuditStatus.NOT_CONVERTED
        and not args.allow_missing_val
    ):
        print(audit.val.recovery_command)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
