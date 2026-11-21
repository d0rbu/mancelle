"""Audit effective full-sequence SFT supervision for encoder-decoder SFT.

The current SFT path trains only in the conditioned mode and supervises every
non-padding next-token target. This script replays the rendered ``text`` SFT
parquets used for feature extraction and quantifies how many supervised tokens
survive truncation, ``min_loss_tokens``, and optional feature-coverage checks.

Examples:

  uv run python -m scripts.analysis.sft_supervision_audit \
      --dataset capabilities --max-examples 50000

  uv run python -m scripts.analysis.sft_supervision_audit \
      --dataset capabilities --max-examples 50000 --load-features
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer
from steering.feature_io import glob_features, load_features, shard_stem_from_path
from steering.sft_schema import legacy_sft_columns

DEFAULT_SFT_ROOT = Path("nanochat_cache/sft_data")
DEFAULT_FEATURE_ROOT = Path("out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b")
DEFAULT_OUT = Path("out/analysis/sft_supervision_audit")


@dataclass
class FeatureCoverage:
    n_feature_rows: int
    n_feature_docs: int
    char_end_by_original_index: dict[int, int]


@dataclass
class SupervisionStats:
    dataset: str
    parquet_dir: str
    feature_dir: str | None
    max_examples: int | None
    row_capacity: int
    min_loss_tokens: int
    scanned_examples: int = 0
    tokenizer_errors: int = 0
    truncated_examples: int = 0
    full_sequence_eligible_examples: int = 0
    feature_present_examples: int = 0
    feature_missing_examples: int = 0
    feature_coverage_gap_examples: int = 0
    feature_coverage_excess_examples: int = 0
    conditioned_effective_examples: int = 0
    full_sequence_loss_positions: int = 0
    conditioned_effective_loss_positions: int = 0


def _iter_parquet_rows(parquet_dir: Path) -> Iterator[tuple[int, str]]:
    for path in sorted(parquet_dir.glob("shard_*.parquet")):
        schema_names = set(pq.read_schema(path).names)
        legacy_cols = legacy_sft_columns(schema_names)
        if legacy_cols:
            raise ValueError(
                f"{path} uses the legacy SFT parquet schema "
                f"({sorted(legacy_cols)}); regenerate with "
                "scripts.preprocess.convert_sft_to_parquet so rows contain "
                "'text', 'original_index', and 'turn_char_starts'."
            )
        if "text" not in schema_names or "original_index" not in schema_names:
            raise ValueError(
                f"{path} has columns {sorted(schema_names)}; expected 'text' "
                "and 'original_index' (current artifacts also carry "
                "'turn_char_starts')."
            )
        table = pq.read_table(path, columns=["text", "original_index"])
        texts = table.column("text").to_pylist()
        original_indices = table.column("original_index").to_pylist()
        for original_index, text in zip(original_indices, texts, strict=True):
            yield int(original_index), str(text)


def _load_feature_coverage(feature_dir: Path, parquet_dir: Path) -> FeatureCoverage:
    char_end_by_original_index: dict[int, int] = {}
    n_feature_rows = 0
    for feat_path in glob_features(feature_dir):
        stem = shard_stem_from_path(feat_path)
        pq_path = parquet_dir / f"{stem}.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(
                f"Missing parquet shard {pq_path} for feature shard {feat_path}"
            )
        original_indices = pq.read_table(
            pq_path,
            columns=["original_index"],
        ).column("original_index").to_pylist()
        data = load_features(feat_path)
        doc_ids = data["doc_ids"]
        char_ends = data["segment_char_ends"]
        n_segments = int(data["n_segments"])
        n_feature_rows += n_segments
        for seg_i in range(n_segments):
            local_doc_id = int(doc_ids[seg_i].item())
            original_index = int(original_indices[local_doc_id])
            char_end = int(char_ends[seg_i].item())
            prev = char_end_by_original_index.get(original_index)
            if prev is None or char_end > prev:
                char_end_by_original_index[original_index] = char_end
    return FeatureCoverage(
        n_feature_rows=n_feature_rows,
        n_feature_docs=len(char_end_by_original_index),
        char_end_by_original_index=char_end_by_original_index,
    )


def _encode_rendered_text(tokenizer, text: str) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    try:
        return list(tokenizer.encode(text, prepend=bos))
    except TypeError:
        return [bos, *list(tokenizer.encode(text))]


def _decoded_source_chars(tokenizer, ids: list[int], fallback_text: str) -> int:
    source_ids = ids[1:] if ids else []
    if hasattr(tokenizer, "decode"):
        try:
            return len(tokenizer.decode(source_ids))
        except Exception:
            return len(fallback_text)
    return len(fallback_text)


def audit_supervision(
    *,
    dataset: str,
    parquet_dir: Path,
    feature_dir: Path | None,
    max_examples: int | None,
    row_capacity: int,
    min_loss_tokens: int,
    load_features_flag: bool,
) -> tuple[SupervisionStats, FeatureCoverage | None]:
    tokenizer = get_tokenizer()
    coverage = (
        _load_feature_coverage(feature_dir, parquet_dir)
        if load_features_flag and feature_dir is not None
        else None
    )
    stats = SupervisionStats(
        dataset=dataset,
        parquet_dir=str(parquet_dir),
        feature_dir=str(feature_dir) if feature_dir is not None else None,
        max_examples=max_examples,
        row_capacity=row_capacity,
        min_loss_tokens=min_loss_tokens,
    )

    for original_index, text in _iter_parquet_rows(parquet_dir):
        if max_examples is not None and stats.scanned_examples >= max_examples:
            break
        stats.scanned_examples += 1
        try:
            ids = _encode_rendered_text(tokenizer, text)
        except Exception:
            stats.tokenizer_errors += 1
            continue

        original_len = len(ids)
        ids = ids[:row_capacity]
        if original_len > row_capacity:
            stats.truncated_examples += 1

        n_loss_tokens = max(0, len(ids) - 1)
        if n_loss_tokens < min_loss_tokens:
            continue

        stats.full_sequence_eligible_examples += 1
        stats.full_sequence_loss_positions += n_loss_tokens

        feature_ok = True
        coverage_gap_ok = True
        coverage_excess_ok = True
        if coverage is not None:
            feature_char_end = coverage.char_end_by_original_index.get(original_index)
            if feature_char_end is None:
                stats.feature_missing_examples += 1
                feature_ok = False
            else:
                stats.feature_present_examples += 1
                target_chars = _decoded_source_chars(tokenizer, ids, text)
                if target_chars - feature_char_end > 20:
                    stats.feature_coverage_gap_examples += 1
                    coverage_gap_ok = False
                if feature_char_end - target_chars > 20:
                    stats.feature_coverage_excess_examples += 1
                    coverage_excess_ok = False

        if feature_ok and coverage_gap_ok and coverage_excess_ok:
            stats.conditioned_effective_examples += 1
            stats.conditioned_effective_loss_positions += n_loss_tokens

    return stats, coverage


def _pct(numer: int | float, denom: int | float) -> str:
    if denom == 0:
        return "n/a"
    return f"{100.0 * float(numer) / float(denom):.2f}%"


def _write_summary(path: Path, stats: SupervisionStats, coverage: FeatureCoverage | None) -> None:
    lines = [
        f"# SFT Supervision Audit: {stats.dataset}",
        "",
        f"- Parquet dir: `{stats.parquet_dir}`",
        f"- Feature dir: `{stats.feature_dir}`",
        f"- Scanned examples: `{stats.scanned_examples}`",
        f"- Row capacity: `{stats.row_capacity}`",
        f"- `min_loss_tokens`: `{stats.min_loss_tokens}`",
        "",
        "## Example Counts",
        "",
        "| quantity | count | fraction of scanned |",
        "|---|---:|---:|",
        f"| tokenizer errors | {stats.tokenizer_errors} | {_pct(stats.tokenizer_errors, stats.scanned_examples)} |",
        f"| truncated examples | {stats.truncated_examples} | {_pct(stats.truncated_examples, stats.scanned_examples)} |",
        f"| full-sequence eligible examples | {stats.full_sequence_eligible_examples} | {_pct(stats.full_sequence_eligible_examples, stats.scanned_examples)} |",
        f"| conditioned effective examples | {stats.conditioned_effective_examples} | {_pct(stats.conditioned_effective_examples, stats.scanned_examples)} |",
    ]
    if coverage is not None:
        lines.extend(
            [
                f"| feature-present examples | {stats.feature_present_examples} | {_pct(stats.feature_present_examples, stats.scanned_examples)} |",
                f"| feature-missing examples | {stats.feature_missing_examples} | {_pct(stats.feature_missing_examples, stats.scanned_examples)} |",
                f"| feature coverage-gap examples | {stats.feature_coverage_gap_examples} | {_pct(stats.feature_coverage_gap_examples, stats.scanned_examples)} |",
                f"| feature coverage-excess examples | {stats.feature_coverage_excess_examples} | {_pct(stats.feature_coverage_excess_examples, stats.scanned_examples)} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Loss Positions",
            "",
            "| quantity | count | fraction of full-sequence loss positions |",
            "|---|---:|---:|",
            f"| full-sequence loss positions | {stats.full_sequence_loss_positions} | 100.00% |",
            f"| conditioned effective loss positions | {stats.conditioned_effective_loss_positions} | {_pct(stats.conditioned_effective_loss_positions, stats.full_sequence_loss_positions)} |",
        ]
    )
    if coverage is not None:
        lines.extend(
            [
                "",
                "## Feature Coverage",
                "",
                f"- Feature segment rows: `{coverage.n_feature_rows}`",
                f"- Feature-covered original indices: `{coverage.n_feature_docs}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["capabilities", "safety"], default="capabilities")
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-examples", type=int, default=50000)
    parser.add_argument("--row-capacity", type=int, default=1025)
    parser.add_argument("--min-loss-tokens", type=int, default=8)
    parser.add_argument("--load-features", action="store_true")
    args = parser.parse_args()

    parquet_dir = args.sft_root / args.dataset
    feature_dir = args.feature_root / f"{args.dataset}_sft"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Missing parquet dir: {parquet_dir}")
    if args.load_features and not feature_dir.exists():
        raise FileNotFoundError(f"Missing feature dir: {feature_dir}")

    stats, coverage = audit_supervision(
        dataset=args.dataset,
        parquet_dir=parquet_dir,
        feature_dir=feature_dir if args.load_features else None,
        max_examples=args.max_examples,
        row_capacity=args.row_capacity,
        min_loss_tokens=args.min_loss_tokens,
        load_features_flag=args.load_features,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "with_features" if args.load_features else "metadata_only"
    json_path = args.out_dir / f"{args.dataset}_{suffix}.json"
    summary_path = args.out_dir / f"{args.dataset}_{suffix}.md"
    payload = {"stats": asdict(stats), "coverage": asdict(coverage) if coverage else None}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_summary(summary_path, stats, coverage)
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
