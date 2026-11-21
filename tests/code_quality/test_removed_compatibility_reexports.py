"""Keep retired relocation and test-facade imports out of production modules."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import scripts.eval.axbench_bpb_adv_eval as axbench_bpb_adv_cli
import steering.eval.feature_probe as feature_probe
import steering.eval.target_segments as target_segments

_INCIDENTAL_IMPORT_FACADES = {
    "steering.conditional.encoder_decoder.cross_attention": ("FP8_ALIGNMENT",),
    "steering.conditional.encoder_decoder.eval_adapter": ("ContextFeatureSegment",),
    "steering.conditional.encoder_decoder.feature_prediction": (
        "FeaturePredLossFn",
        "FeaturePredLossMode",
    ),
    "steering.eval.autointerp_filter.source_index": ("SCHEMA_VERSION",),
    "steering.eval.axbench_base.cluster_artifacts": ("ClusterFeatureContract",),
    "steering.eval.axbench_base.stage0c_api_generation": (
        "ResponseEnvelope",
        "STAGE0C_API_CHECKPOINT_SCHEMA_VERSION",
    ),
    "steering.eval.axbench_bpb_adv.gcg_bags": (
        "read_ours_gcg_bag",
        "read_ours_gcg_scored_prefixes",
        "write_ours_gcg_bag",
        "write_ours_gcg_scored_prefixes",
    ),
    "steering.eval.context_cache_validation": ("ContextCacheValidationError",),
    "steering.eval.feature_filter.types": ("FeatureFilterError",),
    "steering.eval.inline_steering_eval": ("ProbeModel",),
}


def test_retired_compatibility_reexports_are_absent() -> None:
    assert find_spec("steering.eval.axbench_bpb_adv.faster_gcg") is None
    assert "compute_mean_stderr" not in feature_probe.__dict__
    assert "CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS" not in target_segments.__dict__
    assert "ContextFeatureSegment" not in target_segments.__dict__
    assert "PoolingOp" not in target_segments.__dict__
    assert "cfg_to_dataclass" not in axbench_bpb_adv_cli.__dict__


def test_imported_names_are_not_incidental_public_facades() -> None:
    for module_name, incidental_names in _INCIDENTAL_IMPORT_FACADES.items():
        namespace = vars(import_module(module_name))
        leaked = set(incidental_names) & namespace.keys()
        assert not leaked, f"{module_name} exposes incidental imports: {sorted(leaked)}"
