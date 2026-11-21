"""Resume identity for live AxBench-BPB ``ours`` vocabulary semantics.

The BPB cache key does not encode intervention implementation semantics or the
loaded checkpoint's compacted feature vocabulary. This run-level marker keeps
pre-fix and post-fix score rows from being silently mixed when a ``run_tag`` is
reused.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.atomic_io import atomic_write_text
from steering.eval.axbench_base.method_registry import ALL_METHODS, BaselineKind
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.axbench_bpb.vocab_restriction import (
    UNCOMPACTED_VOCAB_FINGERPRINT,
    checkpoint_vocab_fingerprint,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    precomputed_bag_method_family,
)

LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME = ".live_ours_vocab_contract"
LIVE_OURS_VOCAB_CONTRACT_VERSION = 1
_IDENTITY_PREFIX = f"axbench-bpb-live-ours-v{LIVE_OURS_VOCAB_CONTRACT_VERSION}:"
_IDENTITY_RE = re.compile(
    rf"{re.escape(_IDENTITY_PREFIX)}"
    rf"(?:{UNCOMPACTED_VOCAB_FINGERPRINT}|[0-9a-f]{{64}})\Z"
)

AFFECTED_LIVE_OURS_METHODS: frozenset[str] = frozenset(
    method.name
    for method in ALL_METHODS
    if method.baseline == BaselineKind.OURS and precomputed_bag_method_family(method.name) is None
)


def live_ours_vocab_contract_identity(model: Any) -> str:
    """Return the versioned live-``ours`` identity for ``model``."""
    return f"{_IDENTITY_PREFIX}{checkpoint_vocab_fingerprint(model)}"


def _required_identity(model: Any | None) -> str:
    if model is None:
        raise AxBenchBpbError(
            "the loaded encoder-decoder model is required when live "
            "non-bag-backed ours methods or durable rows make the "
            "AxBench-BPB vocabulary contract relevant."
        )
    return live_ours_vocab_contract_identity(model)


def _read_marker(marker_path: Path) -> str | None:
    if not marker_path.exists():
        return None
    try:
        return marker_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AxBenchBpbError(
            f"failed to read AxBench-BPB live-ours vocabulary contract "
            f"marker at {marker_path}: {exc}"
        ) from exc


def _contains_affected_method_names(method_names: Iterable[Any]) -> bool:
    return any(str(name) in AFFECTED_LIVE_OURS_METHODS for name in method_names)


def _durable_scores_contain_affected_rows(scores_dir: Path) -> bool:
    """Scan only ``method_name`` from durable score shards under all ranks."""
    for shard in sorted(scores_dir.rglob("*.parquet")):
        try:
            method_names = pq.read_table(
                shard,
                columns=["method_name"],
            ).column("method_name")
        except Exception as exc:
            raise AxBenchBpbError(
                "cannot establish the AxBench-BPB live-ours vocabulary "
                f"contract because score shard {shard} could not be read "
                f"through its method_name column: {exc}"
            ) from exc
        if _contains_affected_method_names(method_names.to_pylist()):
            return True
    return False


def _resume_rejection(
    *,
    marker_path: Path,
    stored_identity: str | None,
    expected_identity: str,
) -> AxBenchBpbError:
    if stored_identity is None:
        provenance = "the contract marker is missing"
    else:
        provenance = (
            "the contract marker is malformed or does not match the loaded "
            f"checkpoint (stored={stored_identity!r}, expected={expected_identity!r})"
        )
    return AxBenchBpbError(
        "existing AxBench-BPB score shards contain live non-bag-backed "
        f"ours rows, but {provenance}. Their provenance cannot prove they "
        "were produced under the corrected checkpoint-V restriction "
        f"contract. Use a fresh run_tag, or clear the old BPB score/cache "
        f"directory before scoring again. No files were changed "
        f"(marker_path={marker_path})."
    )


def enforce_live_ours_vocab_contract(
    *,
    scores_dir: Path,
    methods: Iterable[Any],
    model: Any | None,
) -> None:
    """Guard score resume before cache reconciliation or row skipping.

    The marker is relevant only when the current method set or durable rows
    contain one of :data:`AFFECTED_LIVE_OURS_METHODS`. Existing affected rows
    are accepted only with an exact identity match. A missing/stale marker is
    stamped only when no affected rows already exist.
    """
    marker_path = scores_dir / LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME
    stored_identity = _read_marker(marker_path)
    current_has_affected_method = _contains_affected_method_names(
        getattr(method, "name", method) for method in methods
    )

    if not current_has_affected_method:
        existing_has_affected_rows = _durable_scores_contain_affected_rows(scores_dir)
        if not existing_has_affected_rows:
            return
        expected_identity = _required_identity(model)
        # Another DDP rank can establish the marker and flush its first shard
        # while this rank is scanning. Re-read after the scan so that newly
        # visible rows are judged against the peer's current identity rather
        # than the stale pre-scan snapshot.
        stored_identity = _read_marker(marker_path)
        if stored_identity == expected_identity:
            return
        raise _resume_rejection(
            marker_path=marker_path,
            stored_identity=stored_identity,
            expected_identity=expected_identity,
        )

    expected_identity = _required_identity(model)
    if stored_identity == expected_identity:
        return
    existing_has_affected_rows = _durable_scores_contain_affected_rows(scores_dir)
    if existing_has_affected_rows:
        # The marker is always written before the first affected row. If a peer
        # made both visible during our scan, its exact identity proves this is
        # valid same-run DDP progress rather than an unmarked legacy shard.
        stored_identity = _read_marker(marker_path)
        if stored_identity == expected_identity:
            return
        raise _resume_rejection(
            marker_path=marker_path,
            stored_identity=stored_identity,
            expected_identity=expected_identity,
        )

    atomic_write_text(f"{expected_identity}\n", marker_path)


def assert_aggregate_live_ours_vocab_contract(
    *,
    scores_table: pa.Table,
    scores_dir: Path,
) -> None:
    """Require a well-formed marker before aggregating affected score rows."""
    if "method_name" not in scores_table.column_names:
        raise AxBenchBpbError("AxBench-BPB score table is missing required method_name column.")
    if not _contains_affected_method_names(scores_table.column("method_name").to_pylist()):
        return

    marker_path = scores_dir / LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME
    stored_identity = _read_marker(marker_path)
    if stored_identity is not None and _IDENTITY_RE.fullmatch(stored_identity):
        return
    raise AxBenchBpbError(
        "AxBench-BPB aggregation found live non-bag-backed ours score rows "
        "without a well-formed post-fix vocabulary contract marker. Rerun "
        "scoring under a fresh run_tag, or clear the old BPB score/cache "
        f"directory and score it again (marker_path={marker_path}, "
        f"stored={stored_identity!r})."
    )


__all__ = [
    "AFFECTED_LIVE_OURS_METHODS",
    "LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME",
    "LIVE_OURS_VOCAB_CONTRACT_VERSION",
    "assert_aggregate_live_ours_vocab_contract",
    "enforce_live_ours_vocab_contract",
    "live_ours_vocab_contract_identity",
]
