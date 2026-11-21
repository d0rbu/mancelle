"""Cross-version overlay byte-identity snapshot harness (issue #530).

Runs the overlay producers it can on a prepared ``feature_dir`` (which must already contain base
shards + parquet + a real ``feature_statistics.safetensors``), then emits a JSON *snapshot* of the
DECOMPRESSED overlay content plus every manifest's recorded base-shard hash. The intended use is
A/B: give each checkout a SEPARATE COPY of the same base shards (e.g. ``cp -r base A`` and
``cp -r base B``) — NOT one shared, already-built ``feature_dir`` — then capture this snapshot on
the pre-#530 checkout over copy A and on the #530 branch over copy B, and ``diff`` the two JSON
files. They must be identical (the base-shard-hash cache is an output-invariant optimization, so it
must not perturb a single overlay byte). Reusing ONE populated ``feature_dir`` would make the second
run skip its already-built overlays (percentiles/token_boundaries ``run()`` early-return when the
overlay exists) and re-snapshot the first run's bytes — a vacuously-identical result.

Why the snapshot hashes DECOMPRESSED tensors, never raw ``.zst`` bytes: ``save_features`` (and
hence base-shard bytes) are non-reproducible across processes (safetensors' Rust ``HashMap``
metadata key order is randomized per call), and even overlay ``.zst`` bytes can drift with the
system zstd build. The decompressed tensor content is the real invariant, so we hash that.

CROSS-VERSION CONSTRAINT — this script is executed on BOTH the pre-#530 checkout and the #530
branch by the controller, so it imports ONLY symbols present on both: the producers' ``run``,
``compute_base_shard_content_hash`` (called with ``cache_dir=None``), ``load_overlay_shard``,
``glob_features``, ``read_manifest``. It does NOT import ``BASE_SHARD_HASH_CACHE_FILENAME`` or any
other #530-only symbol; the cache dotfile is excluded by its literal name so the snapshot never
depends on a symbol the old checkout lacks.

Usage::

    uv run python -m scripts.benchmark.verify_overlay_cache_byte_identity \\
        feature_dir=path/to/features/s128/... segment_size=128 \\
        data_dir=path/to/data out=out/bench/overlay_snapshot.json

Optional: ``context_len=...`` + ``max_shards=...`` + ``sae_width=...`` to also
run packed_rows (best-effort; skipped on any failure so the
percentiles+token_boundaries snapshot is always produced).
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.preprocess.precompute_packed_rows import run as run_packed
from scripts.preprocess.precompute_percentiles import run as run_pct
from scripts.preprocess.precompute_token_boundaries import run as run_tok
from steering.feature_io import glob_features
from steering.overlay_io import (
    compute_base_shard_content_hash,
    overlay_content_digest,
    read_manifest,
)

# Literal name of the base-shard hash cache dotfile (#530). Hardcoded — NOT imported — so this
# harness runs unchanged on a pre-#530 checkout that lacks the symbol. Excluded from the snapshot
# because it is an output-invariant cache, not an overlay artifact.
_CACHE_DOTFILE_NAME = ".base_shard_hash_cache.json"
_COMPRESSED_EXT = ".safetensors.zst"
_MANIFEST_NAME = "manifest.json"
_PACKED_ROWS_BUFFER_SIZE = 1_000
_PACKED_ROWS_PER_SHARD = 16_384
_PACKED_ROWS_ZSTD_LEVEL = 12


def _hash_overlay_tensors(path: Path) -> str:
    """Compatibility wrapper for the shared decompressed-content digest."""
    return overlay_content_digest(path)


def _is_base_shard(path: Path, base_shard_set: set[Path]) -> bool:
    return path.resolve() in base_shard_set


def _snapshot(feature_dir: Path) -> dict[str, str | None]:
    """Walk *feature_dir* and snapshot every overlay shard + manifest under it.

    Keys are paths RELATIVE to *feature_dir* (POSIX). Overlay-shard values are the decompressed
    tensor hash; manifest values are the recorded ``base_shard_content_hash`` (or null for a legacy
    v1 manifest). Base shards (``*_features.safetensors.zst`` at the root) and the cache dotfile are
    excluded.
    """
    base_shard_set = {p.resolve() for p in glob_features(feature_dir)}
    snapshot: dict[str, str | None] = {}

    for path in sorted(feature_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name == _CACHE_DOTFILE_NAME or name.startswith("."):
            continue  # never snapshot the output-invariant cache (or any dotfile)
        rel = path.relative_to(feature_dir).as_posix()
        if name.endswith(_COMPRESSED_EXT):
            if _is_base_shard(path, base_shard_set):
                continue  # immutable base shard — not an overlay artifact
            snapshot[rel] = _hash_overlay_tensors(path)
        elif name == _MANIFEST_NAME:
            manifest = read_manifest(path.parent)
            base_hash = manifest.get("base_shard_content_hash") if manifest else None
            snapshot[rel] = base_hash

    return snapshot


def _run_producers(cfg: DictConfig, feature_dir: Path) -> None:
    """Run percentiles (required) → token_boundaries + packed_rows (best-effort)."""
    segment_size = int(cfg.segment_size)
    data_dir = cfg.get("data_dir", None)

    pct_cfg = OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "segment_size": segment_size,
            "device": "cpu",
            "dry_run": False,
            "resume": False,
            "force": False,
            "zstd_level": 12,
        }
    )
    assert run_pct(pct_cfg) == 0, "precompute_percentiles failed"
    logger.info("percentiles done")

    tok_cfg = OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "data_dir": data_dir,
            "segment_size": segment_size,
            "dry_run": False,
            "resume": False,
            "force": False,
            "workers": int(cfg.get("workers", 1)),
        }
    )
    # Best-effort like packed_rows: token_boundaries needs a real tokenizer (get_tokenizer());
    # in environments without one it is skipped so the percentiles snapshot is still produced. The
    # A/B diff stays valid — whatever producers run, they run identically on both checkouts.
    try:
        ret_tok = run_tok(tok_cfg)
        if ret_tok == 0:
            logger.info("token_boundaries done")
        else:
            logger.warning(f"token_boundaries returned {ret_tok}; snapshot excludes token boundaries")
    except Exception as exc:  # noqa: BLE001 - best-effort; snapshot must still be produced
        logger.warning(f"token_boundaries skipped ({exc!r}); snapshot excludes token boundaries")

    # packed_rows is optional: it needs the pct+tok overlays just written AND >= 2 parquet shards.
    # Run it only when a context_len is supplied, and never let its failure void the snapshot.
    context_len = cfg.get("context_len", None)
    if context_len is None:
        logger.info("no context_len supplied; skipping packed_rows")
        return
    try:
        packed_cfg = OmegaConf.create(
            {
                "feature_dir": str(feature_dir),
                "data_dir": data_dir,
                "segment_size": segment_size,
                "sae_width": str(cfg.sae_width),
                "context_len": int(context_len),
                "max_shards": (
                    None if cfg.get("max_shards", None) is None else int(cfg.max_shards)
                ),
                "buffer_size": _PACKED_ROWS_BUFFER_SIZE,
                "rows_per_shard": _PACKED_ROWS_PER_SHARD,
                "zstd_level": _PACKED_ROWS_ZSTD_LEVEL,
                "resume": False,
                "resume_from_shard": None,
                "compress_workers": 0,
                "profile": False,
                "profile_summary_output": None,
            }
        )
        ret = run_packed(packed_cfg)
        if ret == 0:
            logger.info("packed_rows done")
        else:
            logger.warning(f"packed_rows returned {ret}; snapshot excludes packed rows")
    except Exception as exc:  # noqa: BLE001 - best-effort; snapshot must still be produced
        logger.warning(f"packed_rows skipped ({exc!r}); snapshot excludes packed rows")


def run_verify(cfg: DictConfig) -> dict[str, str | None]:
    """Run the producers on ``cfg.feature_dir`` then write + return the JSON snapshot."""
    feature_dir = Path(str(cfg.feature_dir))
    if not feature_dir.is_dir():
        raise SystemExit(f"feature_dir not found: {feature_dir}")

    # A cross-version reference value, streamed (cache_dir=None) over the on-disk base shards. Folded
    # into the snapshot under a fixed sentinel key so a diff also catches base-hash drift directly.
    base_paths = glob_features(feature_dir)
    if base_paths:
        ref = compute_base_shard_content_hash(base_paths, cache_dir=None)
    else:
        ref = None

    _run_producers(cfg, feature_dir)

    snapshot = _snapshot(feature_dir)
    snapshot["__base_shard_content_hash_reference__"] = ref

    out_path = Path(str(cfg.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    logger.info(f"wrote snapshot ({len(snapshot)} entries) → {out_path}")
    return snapshot


def main() -> None:
    cfg = OmegaConf.from_cli()
    missing = [k for k in ("feature_dir", "segment_size", "out") if k not in cfg]
    if missing:
        raise SystemExit(
            "verify_overlay_cache_byte_identity: required arg(s) missing: "
            f"{', '.join(missing)}=... (e.g. feature_dir=... segment_size=128 out=snap.json)"
        )
    assert isinstance(cfg, DictConfig)
    run_verify(cfg)


if __name__ == "__main__":
    main()
