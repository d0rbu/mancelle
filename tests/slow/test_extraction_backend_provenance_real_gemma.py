"""Real extraction-to-CLI replay gate for #1637; uses locally cached Gemma weights.

Run with HF_HUB_OFFLINE=1 to keep qualification independent of downloads.
TransformerLens uses fp32 and nnterp uses bf16, each replayed at its extraction
dtype. These are same-backend gates, not a cross-backend equivalence claim.
The one document's independent segments fit in one extraction batch, matching
the validator's per-document batch. Batch-size-invariant bf16 replay is a
separate limitation: singleton extraction versus batched replay can fail on main.
"""

import gc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.extract_features import (
    Backend,
    _build_accelerated_backend,
    load_sae_and_model,
    process_shard,
)
from steering.feature_io import glob_features, read_shard_metadata
from steering.segmentation import SegmentMode

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parents[2]
MODEL = "gemma-3-1b-pt"
RELEASE = "gemma-scope-2-1b-pt-res"
SAE_ID = "layer_7_width_16k_l0_medium"


@pytest.mark.parametrize("backend,dtype", [(Backend.nnterp, "bfloat16"), (Backend.transformerlens, "float32")])
def test_real_shards_choose_their_own_backend(tmp_path, backend, dtype):
    if not t.cuda.is_available():
        pytest.skip("real provenance gate requires CUDA")
    model_dtype = t.bfloat16 if dtype == "bfloat16" else t.float32
    model, sae, _tokenizer = load_sae_and_model(
        model_name=MODEL, layer=7, device="cuda", sae_release=RELEASE, sae_id=SAE_ID,
    )
    nn_model, native_model = _build_accelerated_backend(
        backend, model, "cuda", model_dtype, SimpleNamespace(model=MODEL, layer=7),
    )
    if nn_model is not None:
        model.to("cpu")
    sae.to(model_dtype)
    turns = [
        "A botanist examined the garden and recorded the colors of its flowers.\n",
        "She compared her notes with observations from the previous spring.\n",
        "The report explained how rainfall affected the different plants.",
    ]
    text = "".join(turns)
    starts = [0, len(turns[0]), len(turns[0]) + len(turns[1])]
    cells = []
    for name, mode, independent in (
        ("fixed", SegmentMode.fixed_tokens, False),
        ("independent", SegmentMode.fixed_tokens, True),
        ("turns", SegmentMode.turn_delimited_fixed_tokens, True),
    ):
        source = tmp_path / name / "parquet"
        source.mkdir(parents=True)
        parquet = source / "shard_00000.parquet"
        pq.write_table(pa.table({"text": [text], "turn_char_starts": [starts]}), parquet)
        features = tmp_path / name / "features"
        process_shard(
            str(parquet), model, sae, layer=7, output_dir=str(features),
            context_window=128, segment_size=16, segment_mode=mode,
            context_independent_features=independent, ignore_bos_features=True,
            dataset_phase="standard_sft" if mode is SegmentMode.turn_delimited_fixed_tokens else None,
            hook_name="blocks.7.hook_resid_post", gpu_batch_size=32,
            cpu_batch_size=1, pipeline_post=False, nnterp_model=nn_model,
            native_model=native_model,
        )
        shards = glob_features(features)
        assert len(shards) == 1
        assert read_shard_metadata(shards[0])["extraction_backend"] == backend.value
        cells.append((source, features))
    del model, sae, nn_model, native_model
    gc.collect()
    t.cuda.empty_cache()

    for source, features in cells:
        result = subprocess.run(
            [sys.executable, "-m", "steering.validate_features",
             f"feature_dir={features}", f"parquet_dir={source}",
             f"model={MODEL}", f"sae_release={RELEASE}", f"sae_id={SAE_ID}",
             f"model_dtype={dtype}", "n_spot_check=1"],
            cwd=REPO, capture_output=True, text=True, timeout=240, check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert f"recorded={backend.value}, configured=null (metadata-first), effective={backend.value}" in output
        assert "[PASS] shard_00000_features.safetensors.zst" in output
        assert "BACKEND MISMATCH" not in output
