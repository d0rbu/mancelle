"""extract_features.yaml no longer exposes ignore_bos_features.

The option was removed: BOS-feature exclusion is frozen ALWAYS-ON in the
extraction producer (see test_ignore_bos_producer_freeze.py). These tests pin the
config-surface consequences:
  * the key is absent from the composed base config;
  * a bare ``ignore_bos_features=...`` override is rejected by Hydra (struct mode)
    at compose time, before main() runs — Hydra's own error, not ours, and it
    stops the stale flag from taking effect;
  * a ``+ignore_bos_features=...`` append injects the key (the only CLI path that
    reaches the runtime ``_resolve_ignore_bos_features`` logger.error).
"""

import logging
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

from steering.extract_features import _resolve_ignore_bos_features

CFG_DIR = str(Path("configs/extraction").resolve())


def test_key_absent_from_composed_config():
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name="extract_features")
    assert "ignore_bos_features" not in cfg


def test_bare_override_rejected_by_hydra():
    """A bare ``ignore_bos_features=true`` override targets a now-absent struct
    key, so Hydra raises at compose time (before main())."""
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        with pytest.raises(ConfigCompositionException):
            compose(
                config_name="extract_features",
                overrides=["ignore_bos_features=true"],
            )


def test_append_override_injects_key():
    """``+ignore_bos_features=true`` appends the key — the only CLI path that
    reaches the runtime stale-key ``logger.error`` (the bare form is rejected
    above). The composed cfg carries the key so the ``"... in cfg"`` check fires."""
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(
            config_name="extract_features",
            overrides=["+ignore_bos_features=true"],
        )
    assert "ignore_bos_features" in cfg
    assert cfg.ignore_bos_features is True


def test_composed_stale_key_resolves_to_true_and_logs(caplog):
    """End-to-end: a stale key injected via Hydra composition (the realistic
    local-override / `+append` path that survives compose) is caught by
    ``_resolve_ignore_bos_features`` — it returns True (always-on) and logs an
    error, never silently honoring the stale value. This wires the compose-time
    path to the runtime helper, complementing the helper-level unit tests in
    test_ignore_bos_producer_freeze.py."""
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(
            config_name="extract_features",
            overrides=["+ignore_bos_features=false"],  # user tries to turn it OFF
        )
    with caplog.at_level(logging.ERROR):
        result = _resolve_ignore_bos_features(cfg)
    assert result is True, "a stale composed false must NOT disable always-on"
    assert any(
        "ignore_bos_features" in r.message and r.levelno >= logging.ERROR
        for r in caplog.records
    )
