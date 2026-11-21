"""Render saved AxBench judge distributions without loading models or calling a judge."""

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from steering.eval.axbench_histograms import generate_histogram_reports


@hydra.main(config_path="../../configs/plot", config_name="axbench_histograms", version_base=None)
def main(cfg: DictConfig) -> None:
    values = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if isinstance(values, dict):
        values.setdefault("pressure", "gepa")
    if not isinstance(values, dict) or set(values) != {
        "input_dir",
        "output_dir",
        "pipeline",
        "method_factors",
        "bins",
        "pressure",
    }:
        raise ValueError("unknown histogram configuration fields")
    for field in ("input_dir", "output_dir", "pipeline"):
        if not isinstance(values[field], str) or not values[field].strip():
            raise ValueError(f"{field} must be a nonblank string")
    path = generate_histogram_reports(
        input_dir=Path(str(values["input_dir"])),
        output_dir=Path(str(values["output_dir"])),
        pipeline=values["pipeline"],
        method_factors=values["method_factors"],
        bins=values["bins"],
        pressure=values["pressure"],
    )
    logger.info("Histogram report complete: {}", path)


if __name__ == "__main__":
    main()
