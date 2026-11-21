"""Pull artifacts from the project's HF storage bucket to local FS.

Examples:
  python -m scripts.utils.hf_pull category=base_checkpoints
  python -m scripts.utils.hf_pull category=features dry_run=true
  python -m scripts.utils.hf_pull category=features \\
      +include='["s128/l8/**"]'   # only the s128/l8 feature shards

Pull never deletes local files (no safe symmetric use case for that).
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.artifacts.config import ArtifactsConfig
from steering.artifacts.sync import sync_pull


@hydra.main(
    config_path="../../configs/artifacts",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    artifacts_cfg = ArtifactsConfig.from_config(cfg)
    sync_pull(
        cfg=artifacts_cfg,
        category_name=cfg.category,
        user_include=list(cfg.get("include") or []) or None,
        user_exclude=list(cfg.get("exclude") or []) or None,
        no_default_include=bool(cfg.no_default_include),
        no_default_exclude=bool(cfg.no_default_exclude),
        dry_run=bool(cfg.dry_run),
    )


if __name__ == "__main__":
    main()
