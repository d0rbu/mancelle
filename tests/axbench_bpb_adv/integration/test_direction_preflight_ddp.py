"""Real two-rank acceptance for BPB-Adv direction preflight failure."""

from __future__ import annotations

from pathlib import Path

import scripts.eval.axbench_bpb_adv_eval as cli
import steering.eval.axbench_bpb_adv.source_validation as source_validation
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from tests.axbench_bpb_adv.source_run_builders import (
    SourceSeedTokenizer,
    both_class_rows,
    write_scalar_direction_baselines,
    write_stage_0_3_source_run,
)
from tests.runtime_attribute_helpers import set_runtime_attribute


def _mismatched_preflight_worker(
    rank: int,
    world_size: int,
    source_root: str,
    output_root: str,
) -> dict[str, object]:
    set_runtime_attribute(
        source_validation,
        "get_tokenizer",
        lambda: SourceSeedTokenizer(),
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=source_root,
        output_root=output_root,
        run_tag="mismatch",
        device="cpu",
        base_methods=["diffmean_decoder_only"],
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(optimized_token_counts=(0,), max_steps=0)
        ],
        feature_dir="/unused/features",
        feature_statistics_path="/unused/statistics.safetensors",
        diffmean_and_probe_layer=6,
    )
    try:
        cli._preflight_and_install_feature_contract_marker_ddp(
            cfg,
            None,
            rank=rank,
            world_size=world_size,
        )
    except AxBenchBpbAdvError as exc:
        error = str(exc)
    else:
        error = None
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="mismatch")
    return {
        "rank": rank,
        "error": error,
        "marker_exists": (paths.root / "feature_contract.json").exists(),
        "source_contract_exists": paths.source_contract().exists(),
    }


def test_two_rank_layer_mismatch_broadcasts_before_any_marker(
    gloo_multiprocess,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_stage_0_3_source_run(
        source_root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    write_scalar_direction_baselines(
        source_root / "baselines",
        layer=10,
        families=("diffmean",),
        targets=("decoder_only",),
        modes=("positive",),
    )

    results = gloo_multiprocess(
        2,
        _mismatched_preflight_worker,
        str(source_root),
        str(tmp_path / "out"),
    )

    errors = {result["error"] for result in results}
    assert len(errors) == 1
    error = errors.pop()
    assert isinstance(error, str)
    assert "requested/applied layer=6" in error
    assert "artifact training layer=10" in error
    assert all(not result["marker_exists"] for result in results)
    assert all(not result["source_contract_exists"] for result in results)
