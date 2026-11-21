# tests/axbench_base/unit/test_axbench_base_yaml_composability.py
"""Regression test for axbench_base.yaml: orphan-key guard.

Pins the precise invariant introduced by commit 15ae10fd: the
``self_conditioned`` and ``self_cond_segment_size`` keys must not be
re-introduced at the top level of ``configs/eval/axbench_base/axbench_base.yaml``.
``AxBenchBaseConfig`` does not accept either field, so a regression that
re-adds them would crash ``cfg_to_dataclass`` at CLI startup with a
``TypeError`` wrapped as ``AxBenchBaseError``.

Generic Hydra-composition + cfg_to_dataclass round-trip coverage is
already provided by ``tests/axbench_base/unit/test_config_yaml.py``
(``TestHydraComposeAPI::test_hydra_compose_api_works`` +
``TestConfigConvertsToDataclass::test_config_converts_to_dataclass``);
this file deliberately covers ONLY the orphan-key-specific regression.
"""
from __future__ import annotations

import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _compose_default_axbench_base_cfg():
    """Compose the axbench_base YAML the same way the CLI does, but
    in-process. Mirrors the helper in
    tests/axbench_bpb/unit/test_axbench_bpb_yaml_defaults.py."""
    repo_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
    cfg_dir = os.path.join(repo_root, "configs", "eval", "axbench_base")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        # Provide the mandatory fields so OmegaConf does not raise on
        # MissingMandatoryValue. We feed the typical CLI overrides.
        return compose(
            config_name="axbench_base",
            overrides=[
                "feature_dir=/tmp/feat",
                "feature_statistics_path=/tmp/stats.safetensors",
                "decoder_only.source=base",
                "decoder_only.model_tag=d13",
                "decoder_only.step=10000",
                "encdec.source=encdec",
                "encdec.model_tag=d13",
                "encdec.step=10000",
            ],
        )


def test_axbench_base_yaml_does_not_contain_self_conditioned_keys():
    """Specific guard against the keys removed by commit 15ae10fd.

    A future PR that re-adds either key (even as a comment-only or
    null-valued field) trips this test with a descriptive failure
    message naming the offending key — orphans cannot creep back in.
    """
    cfg = _compose_default_axbench_base_cfg()
    # OmegaConf DictConfig supports `key in cfg`.
    assert "self_conditioned" not in cfg, (
        "self_conditioned key is back in axbench_base.yaml — "
        "AxBenchBaseConfig does not accept it; see commit 15ae10fd."
    )
    assert "self_cond_segment_size" not in cfg, (
        "self_cond_segment_size key is back in axbench_base.yaml — "
        "AxBenchBaseConfig does not accept it; see commit 15ae10fd."
    )
