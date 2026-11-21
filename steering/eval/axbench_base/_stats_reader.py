"""Private safetensors header reader for the axbench_base package.

Exposes :func:`read_n_sae_features_from_stats` which returns the SAE
feature count stored in a feature-statistics safetensors file's metadata
header, without allocating any tensors. Used by two independent modules
in the package:

- :mod:`steering.eval.axbench_base.runtime_state`, when
  :class:`RuntimeState` lazy-loads the concepts list and needs
  ``n_sae_features`` to pass into
  :func:`load_and_validate_concepts`;
- :mod:`steering.eval.axbench_base.dag`, when the Stage 4 per-concept
  dep check enumerates training artifacts and needs to load the
  concept list without constructing a full :class:`RuntimeState`.

Factored out so ``dag.py`` can share the helper without pulling in
``runtime_state.py`` (which transitively imports torch + HuggingFace
weight-loading code). The underscore-prefixed module name signals the
function is package-private.

The implementation mirrors :func:`steering.stats_io._read_metadata`
but is kept local to avoid coupling to a non-public symbol in another
package.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from steering.eval.axbench_base.types import AxBenchBaseError

#: Upper bound for the safetensors header size, in bytes. Real headers
#: from the precompute pipeline are a few KBs at most (a JSON map of
#: tensor names to dtypes + shapes + offsets plus a ``__metadata__``
#: dict); 10 MiB is a generous ceiling that still catches corrupted or
#: truncated files that contain a garbage int64 in the first 8 bytes.
#: Without this bound, a file whose first 8 bytes happen to parse as
#: a massive integer would trigger a multi-gigabyte ``f.read`` and
#: exhaust memory before the JSON decoder ever sees the input.
_MAX_HEADER_BYTES: int = 10 * 1024 * 1024


def read_n_sae_features_from_stats(stats_path: Path) -> int:
    """Return ``n_sae_features`` from a safetensors statistics file.

    Reads only the safetensors header metadata (no tensor allocation).

    Args:
        stats_path: Path to a safetensors file whose ``__metadata__``
            header carries an ``n_sae_features`` key (written by the
            precompute pipeline's statistics stage).

    Returns:
        The integer ``n_sae_features`` value.

    Raises:
        AxBenchBaseError: if the file is missing, truncated, malformed,
            does not carry an ``n_sae_features`` metadata entry, or
            declares a header size outside the sane range
            ``[1, _MAX_HEADER_BYTES]``.
    """
    try:
        with open(stats_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) != 8:
                raise AxBenchBaseError(
                    f"feature statistics file at {stats_path} is truncated; "
                    "cannot read safetensors header"
                )
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
                raise AxBenchBaseError(
                    f"feature statistics file at {stats_path} declares "
                    f"header size {header_size} bytes, outside the valid "
                    f"range [1, {_MAX_HEADER_BYTES}]. The file may be "
                    "corrupt or truncated -- a stray/garbage int64 in "
                    "the first 8 bytes can produce a huge header_size "
                    "that would otherwise trigger a multi-gigabyte read. "
                    "Re-run the precompute pipeline's statistics stage "
                    "to regenerate this file."
                )
            header_bytes = f.read(header_size)
    except FileNotFoundError as exc:
        raise AxBenchBaseError(
            f"feature_statistics_path {stats_path} does not exist. "
            "Run the precompute pipeline's statistics stage first."
        ) from exc
    except OSError as exc:
        raise AxBenchBaseError(
            f"Could not read feature statistics file at {stats_path}: {exc}"
        ) from exc
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise AxBenchBaseError(
            f"feature statistics file at {stats_path} has a malformed "
            f"safetensors header: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise AxBenchBaseError(
            f"feature statistics file at {stats_path} has a non-object "
            f"safetensors header ({type(header).__name__})"
        )
    raw_metadata: Any = header.get("__metadata__", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict):
        raise AxBenchBaseError(
            f"feature statistics file at {stats_path} has non-object "
            f"'__metadata__' ({type(raw_metadata).__name__})"
        )
    metadata: dict[str, Any] = raw_metadata
    raw = metadata.get("n_sae_features")
    if raw is None:
        raise AxBenchBaseError(
            f"feature statistics file at {stats_path} does not carry an "
            "'n_sae_features' metadata key. Re-run the statistics stage "
            "with an up-to-date pipeline so it writes the SAE feature "
            "count into the header."
        )
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"feature statistics file at {stats_path} has a non-integer "
            f"'n_sae_features' metadata value {raw!r}"
        ) from exc
