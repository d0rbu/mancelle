"""One-shot recovery: re-populate empty target_real chat seeds in
progression results.json by parsing per-task accuracies from the run log.

Background: training_progression._run_chat_target_real_eval was reading
payload["results"] / payload["chatcore_metric"], but _chat_eval_target_real_per_config
wraps the body under payload["seeds"]["adhoc"] via
build_target_real_payload. So the per-task accuracies were computed and
LOGGED but not written into the JSON. The fix landed in
training_progression.py, but the runs that completed before the patch
have empty target_real chat seeds. This script reconstructs them from
log lines like:

    target_real['lockstep'] SpellingBee: 0.00% (0/100)

The chatcore_metric is recomputed using the SAME baseline-centered
formula chat_eval.py:_chat_eval_target_real_per_config uses (the
BASELINE_ACCURACIES dict in scripts/eval/chat_eval.py).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Mirror chat_eval.ALL_CHAT_TASKS + BASELINE_ACCURACIES (loaded inline so
# this script doesn't depend on import paths).
from scripts.eval.chat_eval import ALL_CHAT_TASKS, BASELINE_ACCURACIES

# Each row in a chat target_real run prints exactly one line of the form:
#   target_real['<config_name>'] <task_name>: <pct>% (<num>/<total>)
LINE_RE = re.compile(
    r"target_real\['(?P<cfg>[^']+)'\] (?P<task>\S+): (?P<pct>[\d.]+)% \((?P<num>\d+)/(?P<total>\d+)\)"
)


def parse_log_per_checkpoint(log_path: Path):
    """Walk the log and emit (variant_name, task, accuracy) tuples
    bucketed by phase+step.

    The log structure for chat target_real is sequential within each
    checkpoint: a "Running chat/target_real/<variant> (seed=...)" line
    starts a block, then 6 lines for the 6 chat tasks, then a "complete"
    line. Checkpoints are bracketed by "=== <phase> step <N> ===".
    """
    ckpt_re = re.compile(r"=== (?P<phase>\w+) step (?P<step>\d+) ===")
    current_ckpt: tuple[str, int] | None = None
    out: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for raw in log_path.read_text(errors="replace").splitlines():
        m_ck = ckpt_re.search(raw)
        if m_ck:
            current_ckpt = (m_ck.group("phase"), int(m_ck.group("step")))
            continue
        m = LINE_RE.search(raw)
        if not m or current_ckpt is None:
            continue
        cfg = m.group("cfg")
        task = m.group("task")
        if task not in ALL_CHAT_TASKS:
            # We only patch chat target_real; CORE target_real lines have
            # the same format (target_real['pooled'] CORE <task>: ...) but
            # task names differ — those are handled correctly already.
            continue
        total = int(m.group("total"))
        num = int(m.group("num"))
        acc = num / total if total > 0 else 0.0
        out[current_ckpt][cfg][task] = acc
    return out


def compute_chatcore_metric(per_task: dict[str, float]) -> float | None:
    if not all(t in per_task for t in ALL_CHAT_TASKS):
        return None
    centered_sum = 0.0
    for task_name, acc in per_task.items():
        baseline = BASELINE_ACCURACIES.get(task_name, 0.0)
        centered = (acc - baseline) / (1.0 - baseline)
        centered_sum += centered
    return centered_sum / len(per_task)


def patch_results_json(
    results_path: Path,
    log_path: Path,
    *,
    seed_key: str = "1234",
    dry_run: bool = True,
) -> dict:
    """Patch a progression results.json + its per-checkpoint files in place.

    Returns a summary dict for reporting.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"results.json missing: {results_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"log missing: {log_path}")
    parsed = parse_log_per_checkpoint(log_path)
    summary = {"results_path": str(results_path), "patched": [], "skipped": []}
    if not parsed:
        summary["error"] = "no chat target_real lines found in log"
        return summary

    with results_path.open() as f:
        merged = json.load(f)

    ckpt_dir = results_path.parent / "checkpoints"

    for ck in merged.get("checkpoints", []):
        phase = ck.get("phase")
        step = int(ck.get("step", -1))
        per_variant = parsed.get((phase, step))
        if not per_variant:
            summary["skipped"].append(
                f"{phase} step={step}: no log lines"
            )
            continue
        chat = ck.get("evals", {}).get("chat", {})
        tr = chat.get("target_real", {}).get("configs", {})
        for variant_name, per_task in per_variant.items():
            if variant_name not in tr:
                summary["skipped"].append(
                    f"{phase} step={step}: variant {variant_name} not in JSON"
                )
                continue
            existing = tr[variant_name].get("seeds", {}).get(seed_key, {})
            if existing:
                summary["skipped"].append(
                    f"{phase} step={step}: {variant_name} seed={seed_key} "
                    f"already populated, refusing to overwrite"
                )
                continue
            body = dict(per_task)
            cm = compute_chatcore_metric(per_task)
            if cm is not None:
                body["chatcore_metric"] = cm
            tr[variant_name].setdefault("seeds", {})[seed_key] = body
            summary["patched"].append(
                f"{phase} step={step}: {variant_name} ← "
                f"{len(per_task)} tasks, chatcore={cm}"
            )
        # Also patch the per-checkpoint JSON if present (the merger reads it
        # on next run).
        ck_path = ckpt_dir / f"{phase}_step_{step:06d}.json"
        if ck_path.exists() and not dry_run:
            with ck_path.open() as f:
                ck_data = json.load(f)
            ck_chat_tr = (
                ck_data.get("evals", {})
                .get("chat", {})
                .get("target_real", {})
                .get("configs", {})
            )
            for variant_name, per_task in per_variant.items():
                if variant_name not in ck_chat_tr:
                    continue
                if ck_chat_tr[variant_name].get("seeds", {}).get(seed_key):
                    continue
                body = dict(per_task)
                cm = compute_chatcore_metric(per_task)
                if cm is not None:
                    body["chatcore_metric"] = cm
                ck_chat_tr[variant_name].setdefault("seeds", {})[seed_key] = body
            with ck_path.open("w") as f:
                json.dump(ck_data, f, indent=2, sort_keys=True)

    if not dry_run:
        with results_path.open("w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="path to results.json")
    p.add_argument("--log", required=True, help="path to run log")
    p.add_argument("--seed", default="1234")
    p.add_argument("--apply", action="store_true",
                   help="actually write changes (default is dry-run)")
    args = p.parse_args()
    s = patch_results_json(
        Path(args.results), Path(args.log),
        seed_key=args.seed, dry_run=not args.apply,
    )
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
