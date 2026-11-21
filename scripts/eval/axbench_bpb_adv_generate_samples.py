"""CLI entry point for the AxBench-BPB-Adv per-datapoint sampled completions.

Generates, for each decoder-only steering FAMILY of a FINISHED bpb-adv run (at
its representative coefficient -- the SAME one the plot selects), ``n`` decoder-
only completions at every counter-steering strength from a handful of concept
core-prefixes, under the EXACT steered intervention the eval scored. Writes one
nested YAML per family under ``samples/<mode>/<method>__<adv>.yaml``.

Decoder-only ONLY; an encoder-decoder target is rejected loudly (a deliberate
follow-up).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.eval.axbench_bpb_adv.sample_generation import (
    run_generate_samples_from_dictconfig,
)


@hydra.main(
    config_path="../../configs/eval/axbench_bpb_adv",
    config_name="generate_samples",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_generate_samples_from_dictconfig(cfg)


if __name__ == "__main__":
    main()
