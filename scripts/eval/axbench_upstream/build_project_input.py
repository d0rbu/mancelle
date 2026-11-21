"""Build the model-free input carrier for AxBench project methods."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from steering._common.hydra_cli import parse_config_name_and_overrides
from steering.eval.axbench_upstream.project_input import write_project_input_from_dictconfig

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_upstream"


def main(argv: list[str] | None = None) -> None:
    """Compose the strict project-input config and publish its artifact."""

    config_name, overrides = parse_config_name_and_overrides(
        sys.argv[1:] if argv is None else argv,
        default_config_name="project_input",
    )
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    write_project_input_from_dictconfig(cfg)


if __name__ == "__main__":
    main()
