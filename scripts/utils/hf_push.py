"""Push local artifacts to the project's HF storage bucket.

Examples:
  python -m scripts.utils.hf_push category=base_checkpoints
  python -m scripts.utils.hf_push category=features dry_run=true
  python -m scripts.utils.hf_push category=base_checkpoints \\
      +include='["*.safetensors"]' +exclude='["scratch/**"]'
  python -m scripts.utils.hf_push category=base_checkpoints delete=true   # destructive!
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from steering.artifacts.config import ArtifactsConfig
from steering.artifacts.sync import sync_push


@hydra.main(
    config_path="../../configs/artifacts",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    artifacts_cfg = ArtifactsConfig.from_config(cfg)
    sync_push(
        cfg=artifacts_cfg,
        category_name=cfg.category,
        user_include=list(cfg.get("include") or []) or None,
        user_exclude=list(cfg.get("exclude") or []) or None,
        no_default_include=bool(cfg.no_default_include),
        no_default_exclude=bool(cfg.no_default_exclude),
        dry_run=bool(cfg.dry_run),
        delete=bool(cfg.delete),
    )


if __name__ == "__main__":
    main()
