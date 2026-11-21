"""Offline analysis and visualization for encoder-decoder sweep results.

Usage:
    uv run python -m scripts.eval.analyze_sweep results=path/to/sweep_results.jsonl
    uv run python -m scripts.eval.analyze_sweep results=path/to/sweep_results.jsonl output_dir=plots/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from steering.eval.progression_results import SuiteResultKind

# ---------------------------------------------------------------------------
# Primary sweep axes (used for marginal analysis)
# ---------------------------------------------------------------------------

PRIMARY_AXES: list[str] = [
    "fs_distribution",
    "fs_mean",
    "ns_mean",
    "fd_mode",
    "fp_weight",
]


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------


def load_results(path: Path) -> list[dict]:
    """Load sweep results from a JSONL file.

    Each line is one JSON object. Blank lines are skipped.
    Works with partial files (mid-sweep).
    """
    results: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                results.append(json.loads(stripped_line))
    return results


def is_valid(record: dict) -> bool:
    """Return True if the record should be included in analysis.

    Invalid if any of: error is set, min_val_bpb is None, val_bpb is None.
    """
    if record.get("error") is not None:
        return False
    if record.get("min_val_bpb") is None:
        return False
    return record.get("val_bpb") is not None


def filter_valid(results: list[dict]) -> list[dict]:
    """Filter results to only valid records."""
    return [r for r in results if is_valid(r)]


# ---------------------------------------------------------------------------
# Kind-based filtering (static_yaml / target_real_lockstep / target_real_pooled)
# ---------------------------------------------------------------------------

def result_kind(record: dict) -> str:
    """Return the kind discriminator for a results record.

    Sweep/result records must carry an explicit ``kind`` field. Bare seed
    records are rejected instead of being inferred as a separate baseline.
    """
    explicit = record.get("kind")
    if explicit is None:
        raise ValueError(
            "sweep record missing required 'kind' field; bare seed records are "
            "not accepted"
        )
    return SuiteResultKind(str(explicit)).value


def filter_by_kind(results: list[dict], kind: str | None) -> list[dict]:
    """Return records matching ``kind``.

    ``kind`` is an optional narrowing filter, not a required field:

    * ``None`` or empty string — return every record unchanged, whether or not
      it declares a ``kind``. This is the default ``kind: null`` path, used to
      analyze kind-less training-sweep rows written by ``run_encdec_sweep_fast``.
    * a known kind string — return only records whose explicit ``kind`` matches.
      Records without a ``kind`` field do not match and are dropped (kind only
      narrows records that declare one). Known kinds:

      * ``"static_yaml"`` — static YAML feature conditioning
      * ``"target_real_lockstep"`` — per-token lockstep target_real conditioning
      * ``"target_real_pooled"`` — pooled target_real conditioning
    """
    if not kind:
        return list(results)
    normalized = SuiteResultKind(str(kind)).value
    filtered: list[dict] = []
    for record in results:
        if record.get("kind") is None:
            continue
        if result_kind(record) == normalized:
            filtered.append(record)
    return filtered


# ---------------------------------------------------------------------------
# Analysis functions (pure — operate on valid records only)
# ---------------------------------------------------------------------------


def compute_summary_stats(results: list[dict]) -> dict:
    """Compute summary statistics for min_val_bpb across results."""
    values = [r["min_val_bpb"] for r in results]
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    arr = np.array(values)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def rank_configs(
    results: list[dict],
    key: str = "min_val_bpb",
    n: int = 10,
    ascending: bool = True,
) -> list[dict]:
    """Return top-N configs sorted by key."""
    sorted_results = sorted(results, key=lambda r: r[key], reverse=not ascending)
    return sorted_results[:n]


def _eta_squared(groups: dict[str, list[float]]) -> float:
    """Compute eta-squared: between-group variance / total variance.

    eta_sq = sum(n_k * (mean_k - grand_mean)^2) / sum((x_i - grand_mean)^2)
    """
    all_values: list[float] = []
    for vals in groups.values():
        all_values.extend(vals)
    if len(all_values) < 2:
        return 0.0

    grand_mean = np.mean(all_values)
    ss_total = sum((x - grand_mean) ** 2 for x in all_values)
    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for vals in groups.values():
        n_k = len(vals)
        mean_k = np.mean(vals)
        ss_between += n_k * (mean_k - grand_mean) ** 2

    return float(ss_between / ss_total)


def compute_marginal_effects(results: list[dict]) -> dict:
    """Compute marginal effect of each primary axis on min_val_bpb.

    Returns {axis_name: {"groups": {value: [bpb_values]}, "eta_squared": float}}
    """
    marginals = {}
    for axis in PRIMARY_AXES:
        groups: dict[str, list[float]] = {}
        for r in results:
            key = str(r.get(axis))
            groups.setdefault(key, []).append(r["min_val_bpb"])
        marginals[axis] = {
            "groups": groups,
            "eta_squared": _eta_squared(groups),
        }
    return marginals


def compute_interaction(
    results: list[dict],
    axis1: str,
    axis2: str,
) -> dict[str, dict[str, float]]:
    """Compute mean min_val_bpb for each (axis1_value, axis2_value) pair.

    Returns nested dict: {axis1_val: {axis2_val: mean_bpb}}.
    """
    cells: dict[str, dict[str, list[float]]] = {}
    for r in results:
        k1 = str(r.get(axis1))
        k2 = str(r.get(axis2))
        cells.setdefault(k1, {}).setdefault(k2, []).append(r["min_val_bpb"])

    return {
        k1: {k2: float(np.mean(vals)) for k2, vals in k2_dict.items()}
        for k1, k2_dict in cells.items()
    }


def compute_pareto_front(results: list[dict]) -> list[dict]:
    """Find Pareto-optimal configs minimizing both time and min_val_bpb.

    A point is Pareto-optimal if no other point is strictly better on both axes.
    """
    if not results:
        return []

    pareto = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if (
                other["total_training_time"] <= r["total_training_time"]
                and other["min_val_bpb"] <= r["min_val_bpb"]
                and (
                    other["total_training_time"] < r["total_training_time"]
                    or other["min_val_bpb"] < r["min_val_bpb"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda r: r["total_training_time"])
    return pareto


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_FD_MODE_COLORS: dict[str, str] = {
    "None": "#1f77b4",
    "prev_segment": "#ff7f0e",
    "cum_max": "#2ca02c",
}


def _fd_color(fd_mode: str | None) -> str:
    return _FD_MODE_COLORS.get(str(fd_mode), "#999999")


# ---------------------------------------------------------------------------
# Plot functions (each saves one PNG)
# ---------------------------------------------------------------------------


def plot_marginal_effects(marginals: dict, output_dir: Path) -> None:
    """Create one violin plot per primary axis showing min_val_bpb distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for axis, data in marginals.items():
        groups = data["groups"]
        eta_sq = data["eta_squared"]

        # Sort by median (best = lowest on left)
        sorted_keys = sorted(groups.keys(), key=lambda k: np.median(groups[k]))
        sorted_values = [groups[k] for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(max(6, len(sorted_keys) * 1.5), 5))
        ax.violinplot(sorted_values, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(sorted_keys) + 1))
        ax.set_xticklabels(sorted_keys, rotation=45, ha="right")
        ax.set_ylabel("min_val_bpb")
        ax.set_title(f"{axis} (eta² = {eta_sq:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / f"marginal_{axis}.png", dpi=150)
        plt.close(fig)


def plot_interaction_heatmap(
    interaction: dict[str, dict[str, float]],
    axis1: str,
    axis2: str,
    output_dir: Path,
) -> None:
    """Create a 2D heatmap of mean min_val_bpb for an axis pair."""
    output_dir.mkdir(parents=True, exist_ok=True)

    keys1 = sorted(interaction.keys())
    keys2 = sorted({k2 for k2_dict in interaction.values() for k2 in k2_dict})

    matrix = np.full((len(keys1), len(keys2)), np.nan)
    for i, k1 in enumerate(keys1):
        for j, k2 in enumerate(keys2):
            if k2 in interaction.get(k1, {}):
                matrix[i, j] = interaction[k1][k2]

    fig, ax = plt.subplots(figsize=(max(6, len(keys2) * 1.2), max(4, len(keys1) * 0.8)))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(keys2)))
    ax.set_xticklabels(keys2, rotation=45, ha="right")
    ax.set_yticks(range(len(keys1)))
    ax.set_yticklabels(keys1)
    ax.set_xlabel(axis2)
    ax.set_ylabel(axis1)
    ax.set_title(f"{axis1} x {axis2} → mean min_val_bpb")

    # Annotate cells
    for i in range(len(keys1)):
        for j in range(len(keys2)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="mean min_val_bpb")
    fig.tight_layout()
    fig.savefig(output_dir / f"heatmap_{axis1}_x_{axis2}.png", dpi=150)
    plt.close(fig)


def plot_pareto_frontier(
    results: list[dict],
    pareto: list[dict],
    output_dir: Path,
) -> None:
    """Scatter plot of time vs BPB with Pareto frontier highlighted."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # All points
    times = [r["total_training_time"] for r in results]
    bpbs = [r["min_val_bpb"] for r in results]
    colors = [_fd_color(r.get("fd_mode")) for r in results]
    ax.scatter(times, bpbs, c=colors, alpha=0.5, s=20, label="all configs")

    # Pareto points
    if pareto:
        p_times = [r["total_training_time"] for r in pareto]
        p_bpbs = [r["min_val_bpb"] for r in pareto]
        ax.scatter(p_times, p_bpbs, c="red", s=80, zorder=5, marker="*", label="Pareto optimal")
        ax.plot(p_times, p_bpbs, "r--", alpha=0.5, zorder=4)

    ax.set_xlabel("total_training_time (s)")
    ax.set_ylabel("min_val_bpb")
    ax.set_title("Cost-Efficiency Frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_frontier.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_all(results: list[dict], output_dir: Path) -> None:
    """Run all analyses and save plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = filter_valid(results)

    if not valid:
        print("No valid results to analyze.")
        return

    # Marginal effects
    marginals = compute_marginal_effects(valid)
    plot_marginal_effects(marginals, output_dir)

    # Interaction heatmaps
    for ax1, ax2 in [
        ("fs_mean", "ns_mean"),
        ("fd_mode", "fp_weight"),
        ("fs_distribution", "fd_mode"),
    ]:
        interaction = compute_interaction(valid, ax1, ax2)
        plot_interaction_heatmap(interaction, ax1, ax2, output_dir)

    # Pareto frontier
    pareto = compute_pareto_front(valid)
    plot_pareto_frontier(valid, pareto, output_dir)


# ---------------------------------------------------------------------------
# Summary table & CLI
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], expected_total: int | None = None) -> None:
    """Print summary table to stdout."""
    valid = filter_valid(results)
    invalid = [r for r in results if not is_valid(r)]

    total_loaded = len(results)
    if expected_total is None:
        # Estimate from Cartesian product of unique primary axis values
        axis_counts = []
        for axis in PRIMARY_AXES:
            unique_vals = {str(r.get(axis)) for r in results}
            axis_counts.append(len(unique_vals))
        expected_total = 1
        for c in axis_counts:
            expected_total *= c

    print("=" * 80)
    print("SWEEP ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  Loaded:    {total_loaded} of {expected_total} configs")
    print(f"  Valid:     {len(valid)}")
    print(f"  Invalid:   {len(invalid)}")
    print()

    if valid:
        stats = compute_summary_stats(valid)
        print("Overall min_val_bpb statistics:")
        print(f"  Mean:   {stats['mean']:.6f}")
        print(f"  Median: {stats['median']:.6f}")
        print(f"  Std:    {stats['std']:.6f}")
        print(f"  Min:    {stats['min']:.6f}")
        print(f"  Max:    {stats['max']:.6f}")
        print()

        print("Top 10 configs (lowest min_val_bpb):")
        for r in rank_configs(valid, "min_val_bpb", 10, ascending=True):
            print(f"  {r['min_val_bpb']:.6f}  {r['config_label']}  (t={r['total_training_time']:.1f}s)")
        print()

        print("Bottom 10 configs (highest min_val_bpb):")
        for r in rank_configs(valid, "min_val_bpb", 10, ascending=False):
            print(f"  {r['min_val_bpb']:.6f}  {r['config_label']}  (t={r['total_training_time']:.1f}s)")
        print()

        # Marginal effect ranking
        marginals = compute_marginal_effects(valid)
        ranked_axes = sorted(marginals.items(), key=lambda kv: kv[1]["eta_squared"], reverse=True)
        print("Axis effect sizes (eta-squared, descending):")
        for axis, data in ranked_axes:
            print(f"  {axis:20s}  eta² = {data['eta_squared']:.4f}")
        print()

    if invalid:
        print("Failed/invalid configs:")
        for r in invalid:
            err = r.get("error", "no error field")
            print(f"  {r.get('config_label', r.get('model_tag', '?'))}  error={err}")

    print("=" * 80)


@hydra.main(
    config_path="../../configs/eval", config_name="analyze_sweep", version_base=None
)
def main(cfg: DictConfig) -> None:
    results_path = Path(cfg.results)

    if not results_path.exists():
        print(f"Error: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = results_path.parent / "sweep_analysis"

    results = load_results(results_path)
    if not results:
        print("No results found in file.")
        sys.exit(0)

    # Optional kind filter: restrict analysis to one explicit result variant
    # (static_yaml / target_real_lockstep / target_real_pooled).
    #
    # NOTE: the kind filter operates on the explicit ``kind`` field stamped by
    # chat_eval / training_progression. Sweep JSONL records without a ``kind``
    # field are rejected instead of being inferred as a separate baseline.
    kind_filter: str | None = cfg.get("kind") or None
    if kind_filter is not None:
        try:
            kind_filter = SuiteResultKind(kind_filter).value
        except ValueError:
            known = ", ".join(sorted(SuiteResultKind))
            print(
                f"Error: unknown kind '{kind_filter}'. Known kinds: {known}",
                file=sys.stderr,
            )
            sys.exit(1)

    before = len(results)
    try:
        results = filter_by_kind(results, kind_filter)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if kind_filter is not None:
        print(f"Kind filter '{kind_filter}': {before} → {len(results)} records")
        if not results:
            print(f"No results match kind='{kind_filter}'.")
            sys.exit(0)

    print_summary(results, expected_total=cfg.expected_total)
    generate_all(results, output_dir)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
