"""Unit tests for CLI helper functions in ``scripts.eval.axbench_bpb_eval``."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_bpb_eval import _cfg_to_dataclass, _generate_run_tag
from steering.conditional.eval_conditioning import EvalConditioning
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.sae_resolver import ResolvedSAE


def test_generate_run_tag_format():
    """Run tag is YYYYMMDD-HHMMSS-<8 hex chars>."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    tag = _generate_run_tag(cfg)
    assert re.match(r"^\d{8}-\d{6}-[0-9a-f]{8}$", tag), f"bad format: {tag}"


def test_generate_run_tag_is_deterministic_per_cfg():
    """The suffix is a SHA-256 digest of the cfg JSON. Two calls with
    the same cfg in the same UTC second produce IDENTICAL tags.

    This is a behavioural inversion vs. the original time-ns-based
    generator (see commit fixing _generate_run_tag for DDP coherence):
    under torchrun every rank computes the same cfg → same tag, so all
    ranks write to the same output tree. Different cfgs still produce
    different suffixes (covered by
    test_generate_run_tag_changes_when_cfg_changes in
    tests/axbench_bpb/integration/test_axbench_bpb_eval_ddp_init.py).
    """
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    tags = {_generate_run_tag(cfg) for _ in range(10)}
    # The timestamp prefix may roll across a UTC second boundary mid-loop;
    # the suffix is the deterministic-over-cfg part, so assert THAT is
    # constant rather than the full tag.
    suffixes = {t.split("-")[-1] for t in tags}
    assert len(suffixes) == 1, (
        f"_generate_run_tag must be deterministic over cfg; "
        f"got {len(suffixes)} distinct suffixes: {suffixes}"
    )


def test_generate_run_tag_omits_disabled_extraction_model_field_exactly() -> None:
    """A new null field must not orphan every historical BPB run tag."""

    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    legacy_payload = dataclasses.asdict(cfg)
    assert legacy_payload.pop("feature_extraction_model_id") is None
    expected = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]

    assert _generate_run_tag(cfg).split("-")[-1] == expected


def test_generate_run_tag_includes_explicit_extraction_model_field() -> None:
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        feature_extraction_model_id="gemma-3-4b-pt",
    )
    payload = dataclasses.asdict(cfg)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]

    assert _generate_run_tag(cfg).split("-")[-1] == expected


def test_cfg_to_dataclass_round_trip():
    """DictConfig → AxBenchBpbConfig preserves nested structure."""
    cfg_dict = {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "run_tag": "test",
        "feature_dir": "/tmp/feat",
        "feature_statistics_path": "/tmp/stats.safetensors",
        "stage": "all",
        "enabled_methods": None,
        "disabled_methods": [],
        "filter_unaccepted_samples": True,
        "device_batch_size": 16,
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cuda",
        "extraction_tokenizer_id": "google/gemma-2-2b",
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_source": "8-gemmascope-res-16k",
        "extraction_max_tokens": 2048,
        "extraction_threshold": 0.0,
        "log_progress_every": 1000,
        "offload_between_stages": True,
        "context_cache": {"extraction_batch_size": 32, "force_rebuild": False},
        "aggregation": {
            "min_rows_per_side": 10,
            "invalid_warning_threshold": 0.01,
            "invalid_drop_threshold": 0.05,
            "primary_metric": "macro_mean_best_primary_margin",
        },
    }
    dc = _cfg_to_dataclass(OmegaConf.create(cfg_dict))
    assert isinstance(dc, AxBenchBpbConfig)
    assert dc.source_run_dir == "/tmp/source"
    assert dc.context_cache.extraction_batch_size == 32
    assert dc.aggregation.min_rows_per_side == 10
    assert dc.decoder_only == {"source": "base", "model_tag": "d13", "step": 10000}


def test_cfg_to_dataclass_strips_hydra_block():
    """``hydra`` internal block is dropped, not passed through to AxBenchBpbConfig."""
    cfg_dict = {
        "source_run_dir": "/tmp/source",
        "output_root": "/tmp/out",
        "feature_dir": "/tmp/feat",
        "feature_statistics_path": "/tmp/stats.safetensors",
        "decoder_only": {},
        "encdec": {},
        "hydra": {"output_subdir": None},  # should be stripped
    }
    dc = _cfg_to_dataclass(OmegaConf.create(cfg_dict))
    assert isinstance(dc, AxBenchBpbConfig)


def test_runtime_state_construction():
    """RuntimeState post-init builds SourcePaths + OutputPaths from cfg."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        run_tag="test",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    assert rs.source_paths().source_run_dir == Path("/tmp/source")
    assert rs.output_paths().output_root == Path("/tmp/out")


def test_runtime_state_resources_start_unloaded():
    """RuntimeState's lazy resources are None until first access."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    assert rs._decoder_only is None
    assert rs._encdec is None
    assert rs._base_gemma is None
    assert rs._base_sae is None


def test_runtime_state_loads_sparse_final_source_concepts_once(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    concepts = [
        {
            "concept_id": concept_id,
            "feature_index": feature_index,
            "layer": 8,
            "sae_source": "gemmascope",
            "description": f"concept {concept_id}",
            "corpus_count": 10,
            "description_source": "test",
        }
        for concept_id, feature_index in ((2, 20), (9, 90))
    ]
    (source / "concepts_stage1c.json").write_text(
        json.dumps(concepts),
        encoding="utf-8",
    )
    cfg = AxBenchBpbConfig(
        source_run_dir=str(source),
        output_root=str(tmp_path / "out"),
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
    )
    state = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    loaded = state.source_concepts()
    (source / "concepts_stage1c.json").write_text("corrupt", encoding="utf-8")

    assert [concept.concept_id for concept in loaded] == [2, 9]
    assert state.source_concepts() is loaded
    assert state.source_concept_ids() == frozenset({2, 9})


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"concept_id": 1}],
        [
            {
                "concept_id": 1,
                "feature_index": True,
                "layer": 8,
                "sae_source": "gemmascope",
                "description": "one",
                "corpus_count": 10,
                "description_source": "test",
            }
        ],
        [
            {
                "concept_id": 3,
                "feature_index": 30,
                "layer": 8,
                "sae_source": "gemmascope",
                "description": "three",
                "corpus_count": 10,
                "description_source": "test",
            },
            {
                "concept_id": 2,
                "feature_index": 20,
                "layer": 8,
                "sae_source": "gemmascope",
                "description": "two",
                "corpus_count": 10,
                "description_source": "test",
            },
        ],
    ],
)
def test_runtime_state_rejects_invalid_final_source_concepts(
    tmp_path,
    payload,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "concepts_stage1c.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    cfg = AxBenchBpbConfig(
        source_run_dir=str(source),
        output_root=str(tmp_path / "out"),
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
    )
    state = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    with pytest.raises(AxBenchBpbError):
        state.source_concepts()


def test_runtime_state_decoder_only_wires_to_load_model():
    """RuntimeState.decoder_only() returns a TargetModelBundle wrapping
    the result of nanochat.checkpoint_manager.load_model."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        decoder_only={"source": "base", "model_tag": "tiny", "step": 0},
        encdec={"source": "encdec", "model_tag": "tiny", "step": 0},
        device="cpu",
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    fake_model = type("FakeModel", (), {"config": type("C", (), {"sequence_len": 64})()})()
    fake_tok = object()
    fake_meta = {"checkpoint_dir": "/tmp/somewhere"}

    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_model",
        return_value=(fake_model, fake_tok, fake_meta),
    ) as mocked:
        bundle = rs.decoder_only()

    assert isinstance(bundle, TargetModelBundle)
    assert bundle.model is fake_model
    assert bundle.tokenizer is fake_tok
    assert bundle.target_name == "decoder_only"
    assert bundle.max_seq_len == 64
    mocked.assert_called_once_with(
        source="base", device="cpu", phase="eval",
        model_tag="tiny", step=0,
        eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"),
    )

    # Second call should use cached value (no re-call to load_model)
    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_model",
        side_effect=AssertionError("re-loaded"),
    ):
        bundle2 = rs.decoder_only()
    assert bundle2 is bundle


def test_runtime_state_encdec_wires_to_load_model():
    """RuntimeState.encdec() returns a TargetModelBundle wrapping
    the result of nanochat.checkpoint_manager.load_model."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        decoder_only={"source": "base", "model_tag": "tiny", "step": 0},
        encdec={"source": "encdec", "model_tag": "tiny-enc", "step": 100},
        device="cuda",
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    # EncoderDecoderGPT exposes dec_config, not config.
    fake_model = type("FakeModel", (), {"dec_config": type("C", (), {"sequence_len": 128})()})()
    fake_tok = object()
    fake_meta = {"checkpoint_dir": "/tmp/encdec"}

    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_model",
        return_value=(fake_model, fake_tok, fake_meta),
    ) as mocked:
        bundle = rs.encdec()

    assert bundle.target_name == "encdec"
    assert bundle.max_seq_len == 128
    assert bundle.model._loaded_meta_data is fake_meta
    mocked.assert_called_once_with(
        source="encdec", device="cuda", phase="eval",
        model_tag="tiny-enc", step=100,
        eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"),
    )


def test_loaded_target_metadata_is_exact_but_cannot_mutate_retained_state() -> None:
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        decoder_only={"source": "base", "model_tag": "tiny", "step": 0},
        device="cpu",
    )
    state = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    fake_model = type("FakeModel", (), {"config": type("C", (), {"sequence_len": 64})()})()
    retained = {
        "model_config": {"depth": 12},
        "recovered": ["ancestor"],
        "checkpoint_dir": "/tmp/checkpoint",
    }
    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_model",
        return_value=(fake_model, object(), retained),
    ):
        observed = state.loaded_target_metadata("decoder_only")

    assert dict(observed) == retained
    with pytest.raises(TypeError):
        observed["new"] = "forbidden"  # ty: ignore[invalid-assignment]
    observed["model_config"]["depth"] = 99
    observed["recovered"].append("mutation")
    assert state.loaded_target_metadata("decoder_only") == retained
    assert retained == {
        "model_config": {"depth": 12},
        "recovered": ["ancestor"],
        "checkpoint_dir": "/tmp/checkpoint",
    }


def test_loaded_target_metadata_rejects_unknown_target_and_missing_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )
    state = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    with pytest.raises(AxBenchBpbError, match="unknown target metadata request"):
        state.loaded_target_metadata("unknown")
    monkeypatch.setattr(RuntimeState, "decoder_only", lambda _self: object())
    with pytest.raises(AxBenchBpbError, match="was not retained after loading"):
        state.loaded_target_metadata("decoder_only")


def test_runtime_state_extraction_tokenizer_populated_by_lazy_load(monkeypatch):
    """extraction_tokenizer() returns the gemma tokenizer from load_sae_and_model."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        neuronpedia_model_id="gemma-2-2b",
        neuronpedia_source="8-gemmascope-res-16k",
        device="cpu",
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    fake_model = object()
    fake_sae = object()
    fake_tok = object()
    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_sae_and_model",
        return_value=(fake_model, fake_sae, fake_tok),
    ):
        tok = rs.extraction_tokenizer()
    assert tok is fake_tok


def test_runtime_state_base_gemma_and_sae_wires_to_resolver():
    """RuntimeState.base_gemma_and_sae() resolves Neuronpedia source via
    parse_neuronpedia_source + resolve_sae, then calls load_sae_and_model."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        neuronpedia_model_id="gemma-2-2b",
        neuronpedia_source="8-gemmascope-res-16k",
        device="cpu",
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    fake_model = object()
    fake_sae = object()
    fake_tok = object()

    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_sae_and_model",
        return_value=(fake_model, fake_sae, fake_tok),
    ) as mocked:
        gemma, sae = rs.base_gemma_and_sae()

    assert gemma is fake_model
    assert sae is fake_sae
    # Verify the call was made with a resolved ResolvedSAE
    _args, kwargs = mocked.call_args
    assert kwargs["model_name"] == "gemma-2-2b"
    assert kwargs["device"] == "cpu"
    assert isinstance(kwargs["resolved"], ResolvedSAE)
    assert kwargs["resolved"].release.startswith("gemma-scope-2b-pt-res")
    assert "layer_8" in kwargs["resolved"].sae_id
    assert "width_16k" in kwargs["resolved"].sae_id

    # Cached on second call
    with patch(
        "steering.eval.axbench_bpb.runtime_state.load_sae_and_model",
        side_effect=AssertionError("re-loaded"),
    ):
        gemma2, sae2 = rs.base_gemma_and_sae()
    assert gemma2 is gemma
    assert sae2 is sae
