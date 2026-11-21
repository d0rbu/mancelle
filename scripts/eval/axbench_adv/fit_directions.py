"""Fit checkpoint-bound project directions from real-conditioned AxBench rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from steering.eval.axbench_adv.direction_fitting import fit_project_directions

_CONFIG_KEYS = {
    "prepared_dir",
    "target_meta_path",
    "feature_statistics_path",
    "output_root",
    "max_concepts",
    "layer",
    "seed",
    "device",
}


def _path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value == "???":
        raise ValueError(f"fit config {field} must be a nonblank resolved path")
    return Path(value).expanduser()


def _resolved_config(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, Mapping) or set(raw) != _CONFIG_KEYS:
        observed = (
            sorted(repr(key) for key in raw)
            if isinstance(raw, Mapping)
            else type(raw).__name__
        )
        raise ValueError(
            f"fit config fields must be exactly {sorted(_CONFIG_KEYS)}, got {observed}"
        )
    return cast(dict[str, Any], dict(raw))


def run_from_config(cfg: DictConfig):
    """Resolve the closed Hydra surface and run the production fitter."""

    raw = _resolved_config(cfg)
    return fit_project_directions(
        prepared_dir=_path(raw["prepared_dir"], field="prepared_dir"),
        target_meta_path=_path(raw["target_meta_path"], field="target_meta_path"),
        feature_statistics_path=_path(
            raw["feature_statistics_path"], field="feature_statistics_path"
        ),
        output_root=_path(raw["output_root"], field="output_root"),
        max_concepts=raw["max_concepts"],
        layer=raw["layer"],
        seed=raw["seed"],
        device=raw["device"],
    )


@hydra.main(
    config_path="../../../configs/eval/axbench_adv",
    config_name="fit_directions",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    publication = run_from_config(cfg)
    logger.info(
        "AxBench-Adv project direction fit complete: {} artifacts, manifest {}",
        len(publication.artifacts),
        publication.manifest_path,
    )


if __name__ == "__main__":
    main()
