"""Auto-discovery of encoder-decoder configs for the completion TUI.

Scans the filesystem for available checkpoints, matches them to feature
directories and parquet data, and returns launchable TUIConfig objects.
"""

from __future__ import annotations

import glob as glob_mod
import json
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nanochat.checkpoint_manager import find_last_step
from nanochat.common import get_base_dir, get_project_root
from steering.feature_paths import parse_feature_dir, resolve_data_dir


@dataclass
class TUIConfig:
    """A discovered, launchable configuration."""

    model_tag: str
    checkpoint_dir: str
    nested_path: str  # e.g. "s128/t2048.9" (for display)
    step: int
    feature_dir: str
    parquet_dir: str
    segment_size: int
    num_buckets: int
    has_negatives: bool
    label: str


def _enumerate_model_tags(checkpoints_dir: str) -> list[str]:
    """List model tag directories (e.g. d26, d27) under checkpoints_dir."""
    if not os.path.isdir(checkpoints_dir):
        return []
    return sorted(
        d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))
    )


def _find_all_checkpoint_leaves(root_dir: str) -> list[str]:
    """Find all directories under root_dir that contain model_*.pt files.

    Unlike checkpoint_manager._find_checkpoint_leaf which returns only the
    first match, this returns all leaf directories with checkpoints.
    """
    results: list[str] = []

    if glob_mod.glob(os.path.join(root_dir, "model_*.pt")):
        results.append(root_dir)
        return results

    if not os.path.isdir(root_dir):
        return results

    subdirs = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    for subdir in subdirs:
        results.extend(_find_all_checkpoint_leaves(os.path.join(root_dir, subdir)))
    return results


def _find_last_step_safe(checkpoint_dir: str) -> int | None:
    """Find the latest checkpoint step, or None if no checkpoints exist.

    Wraps checkpoint_manager.find_last_step, converting FileNotFoundError
    to None for discovery's silent-skip semantics.
    """
    try:
        return find_last_step(checkpoint_dir)
    except FileNotFoundError:
        return None


def _load_meta(checkpoint_dir: str, step: int) -> dict | None:
    """Load meta JSON for a checkpoint step, or None on failure."""
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Failed to load metadata: {meta_path}")
        return None


def _nested_path(tag_dir: str, leaf_dir: str) -> str:
    """Compute the relative nested path from tag_dir to leaf_dir for display."""
    rel = os.path.relpath(leaf_dir, tag_dir)
    return rel if rel != "." else ""


def _scan_feature_dirs(feature_root: str) -> list[Path]:
    """Recursively find all feature directories that contain statistics files."""
    results: list[Path] = []
    root = Path(feature_root)
    if not root.is_dir():
        return results
    for stats_path in sorted(root.rglob("feature_statistics.safetensors")):
        results.append(stats_path.parent)
    return results


def discover_configs(
    base_dir: str | None = None,
    feature_root: str | None = None,
) -> list[TUIConfig]:
    """Scan for available encoder-decoder configurations.

    Algorithm:
    1. Resolve base_dir via get_base_dir()
    2. Scan conditional_encdec_checkpoints/ for model tags
    3. For each model tag, find all checkpoint leaf directories
    4. Load meta JSON to get num_buckets, has_negatives, segment_size
    5. Scan feature root for dirs matching segment_size
    6. Match parquet dirs by resolving data phase from feature path
    7. Return list of TUIConfig, sorted by model_tag then nested_path
    """
    if base_dir is None:
        base_dir = get_base_dir()
    if feature_root is None:
        feature_root = os.path.join(get_project_root(), "out", "data", "features")

    encdec_root = os.path.join(base_dir, "conditional_encdec_checkpoints")

    model_tags = _enumerate_model_tags(encdec_root)
    if not model_tags:
        return []

    # Pre-scan feature directories, indexed by segment_size
    feature_dirs = _scan_feature_dirs(feature_root)
    features_by_seg_size: dict[int, list[Path]] = {}
    for fd in feature_dirs:
        try:
            parsed = parse_feature_dir(fd)
            seg_size = parsed.segment_size
            features_by_seg_size.setdefault(seg_size, []).append(fd)
        except ValueError:
            continue

    configs: list[TUIConfig] = []

    for tag in model_tags:
        tag_dir = os.path.join(encdec_root, tag)
        leaves = _find_all_checkpoint_leaves(tag_dir)

        for leaf in leaves:
            step = _find_last_step_safe(leaf)
            if step is None:
                continue

            meta = _load_meta(leaf, step)
            if meta is None:
                continue

            user_cfg = meta.get("user_config", {})
            num_buckets = user_cfg.get("num_buckets")
            if num_buckets is None:
                continue

            has_negatives = user_cfg.get("negative_selection") is not None
            segment_size = user_cfg.get("segment_size")
            if segment_size is None:
                continue

            nested = _nested_path(tag_dir, leaf)

            # Match feature directories by segment_size
            matching_features = features_by_seg_size.get(segment_size, [])

            if not matching_features:
                # No matching features — still create a config without feature/parquet
                # (user can manually specify later)
                continue

            for feat_dir in matching_features:
                try:
                    parsed = parse_feature_dir(feat_dir)
                except ValueError:
                    continue

                # Resolve parquet directory from dataset phase
                phase_str = parsed.dataset_phase.value
                try:
                    parquet_dir = resolve_data_dir(None, phase_str)
                except Exception:
                    continue

                if not parquet_dir.is_dir():
                    continue

                # Build display label
                width = parsed.width.value
                layer = parsed.layer
                act = parsed.activation_type.value
                pool = parsed.pooling.value
                phase = parsed.dataset_phase.value
                cluster_suffix = ""
                if meta.get("feature_clusters_descriptor") is not None:
                    canonical = meta.get("feature_clusters_canonical")
                    if isinstance(canonical, dict) and isinstance(canonical.get("clusters"), list):
                        cluster_suffix = f", {len(canonical['clusters'])} clusters"
                label = (f"{tag} / {nested}" if nested else tag) + (
                    f" (step {step}, {width} features, layer {layer},"
                    f" {act}/{pool}, phase={phase}{cluster_suffix})"
                )

                configs.append(
                    TUIConfig(
                        model_tag=tag,
                        checkpoint_dir=leaf,
                        nested_path=nested,
                        step=step,
                        feature_dir=str(feat_dir),
                        parquet_dir=str(parquet_dir),
                        segment_size=segment_size,
                        num_buckets=num_buckets,
                        has_negatives=has_negatives,
                        label=label,
                    )
                )

    configs.sort(key=lambda c: (c.model_tag, c.nested_path))
    return configs


_AUTO_SELECT_SECS = 10


def prompt_selection(configs: list[TUIConfig]) -> TUIConfig:
    """Interactive terminal selection of a TUIConfig.

    Prints a numbered list and reads user input. Auto-selects the first option
    after ``_AUTO_SELECT_SECS`` seconds of inactivity. Any keypress cancels the
    countdown and waits for Enter normally.
    """
    print("\nAvailable encoder-decoder models:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.label}")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf: list[str] = []
    timed_out = False

    try:
        tty.setraw(fd)
        deadline = time.monotonic() + _AUTO_SELECT_SECS
        timer_active = True

        sys.stdout.write(f"\nSelect [1] (auto in {_AUTO_SELECT_SECS}s): ")
        sys.stdout.flush()

        while True:
            if timer_active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                wait = min(0.2, remaining)
            else:
                wait = 60.0

            readable, _, _ = select.select([sys.stdin], [], [], wait)

            if not readable:
                if timer_active:
                    secs = max(1, int(deadline - time.monotonic()) + 1)
                    sys.stdout.write(f"\r\x1b[KSelect [1] (auto in {secs}s): {''.join(buf)}")
                    sys.stdout.flush()
                continue

            if timer_active:
                timer_active = False
                sys.stdout.write(f"\r\x1b[KSelect [1]: {''.join(buf)}")
                sys.stdout.flush()

            ch = sys.stdin.read(1)

            if ch in ("\r", "\n"):
                break
            if ch == "\x03":
                sys.stdout.write("\r\n")
                raise KeyboardInterrupt
            if ch == "\x04":
                sys.stdout.write("\r\n")
                raise EOFError
            if ch in ("\x7f", "\x08"):  # backspace
                if buf:
                    buf.pop()
                    sys.stdout.write(f"\r\x1b[KSelect [1]: {''.join(buf)}")
                    sys.stdout.flush()
            elif ch.isprintable():
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()

    except (EOFError, KeyboardInterrupt):
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()
        raise SystemExit(1) from None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\r\n")
        sys.stdout.flush()

    if timed_out:
        print(f"Auto-selecting: {configs[0].label}")
        return configs[0]

    raw = "".join(buf).strip()
    if not raw:
        return configs[0]
    try:
        idx = int(raw)
        if 1 <= idx <= len(configs):
            return configs[idx - 1]
    except ValueError:
        pass
    print(f"Invalid selection, using default: {configs[0].label}")
    return configs[0]
