"""Integration test: ``run_bpb_score_stage`` forwards ``inactive_sampler``
all the way to ``build_intervention``.

This locks numerical correctness for negative-mode ``ours_*`` methods: the
inactive partition must be populated from the encdec target's
``negative_selection`` metadata, not silently empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from steering.eval.axbench_bpb import bpb_scoring as bpb_scoring_module
from steering.eval.axbench_bpb.bpb_scoring import run_bpb_score_stage
from steering.eval.axbench_bpb.sample_loader import (
    make_prefix_hash,
    make_sample_id,
)
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import (
    BpbScoreRow,
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from tests.cross_tokenizer_helpers import SimpleTokenizer


@dataclass(frozen=True)
class _FakeMethod:
    name: str
    baseline: str
    mode: SteeringMode
    target_model: str


@dataclass(frozen=True)
class _FakeConcept:
    concept_id: int


@dataclass
class _FakeOutputPaths:
    root: Path

    def bpb_scores_rank_dir(self, rank: int) -> Path:
        d = self.root / "bpb_scores" / f"rank_{rank:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bpb_cache_path(self, rank: int) -> Path:
        return self.root / "bpb_scores" / f"rank_{rank:04d}.sqlite"


@dataclass
class _Cfg:
    log_progress_every: int = 10
    # BPB scoring no longer reads SAE feature statistics for scalar steering;
    # this ours-only harness keeps a dummy config value for schema compatibility.
    feature_statistics_path: str = "unused.safetensors"


def _make_scoring_input(tok, prefix, output, concept_id, mode):
    prefix_tokens = encode_prefix(prefix, tok)
    output_tokens = encode_output(output, tok)
    byte_weights = torch.ones(len(output_tokens), dtype=torch.int64)
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=concept_id, mode=mode,
            content_class="concept_rich",
            prefix_text=prefix, output_text=output,
        ),
        concept_id=concept_id, mode=mode,
        content_class="concept_rich",
        prefix_text=prefix, prefix_hash=make_prefix_hash(prefix),
        prefix_tokens=prefix_tokens, output_tokens=output_tokens,
        prefix_context_segments=(),
        n_prefix=len(prefix_tokens), n_output=len(output_tokens),
        output_byte_weights=byte_weights, truncated=False,
    )


def _sentinel_inactive_sampler(*, rng):
    """Sentinel inactive sampler — returns a deterministic mapping so the
    integration test can assert it was forwarded unchanged."""
    return {99: 0}


def test_run_bpb_score_stage_threads_inactive_sampler(tmp_path, monkeypatch):
    """``run_bpb_score_stage`` must forward an externally-supplied
    ``inactive_sampler`` to ``build_intervention`` (NOT replace it with
    None). Without this, negative-mode ``ours_*`` methods run with an
    empty inactive partition — silent numerical correctness bug.
    """
    # Spy on build_intervention to record the inactive_sampler kwarg it sees.
    captured_kwargs: list[dict] = []

    def _spy_build_intervention(**kwargs):
        captured_kwargs.append(kwargs)
        method = kwargs["method"]
        return Intervention.prompting(target_model=method.target_model, prompt_prefix_tokens=[])

    monkeypatch.setattr(
        bpb_scoring_module, "build_intervention", _spy_build_intervention,
    )

    # Stub score_sample so we don't need a real model — we only care
    # about parameter threading.
    def _fake_score_sample(*, inputs, target_bundle, intervention,
                           device, coefficient_str, method_name,
                           lora=None):
        return BpbScoreRow(
            method_name=method_name, concept_id=inputs.concept_id,
            mode=inputs.mode.value, content_class=inputs.content_class,
            target=target_bundle.target_name,
            coefficient_str=coefficient_str, sample_id=inputs.sample_id,
            bpb_intervened=0.0, n_output_tokens=inputs.n_output,
            n_output_bytes=int(inputs.output_byte_weights.sum().item()),
            truncated=False,
        )

    monkeypatch.setattr(
        bpb_scoring_module, "score_sample", _fake_score_sample,
    )

    tok = SimpleTokenizer()
    # Bundle is opaque since score_sample is stubbed.
    bundle = TargetModelBundle(
        model=object(), tokenizer=tok,
        max_seq_len=64, target_name="encdec",
    )

    methods = [_FakeMethod(
        name="neg_ours_replace_random", baseline="ours",
        mode=SteeringMode.NEGATIVE, target_model="encdec",
    )]
    concepts = [_FakeConcept(concept_id=1)]
    samples_by_concept_mode = {
        (1, SteeringMode.NEGATIVE): [
            _make_scoring_input(tok, "abc", "x", 1, SteeringMode.NEGATIVE),
        ],
    }
    coefficient_grids = {"neg_ours_replace_random": [3]}

    run_bpb_score_stage(
        cfg=_Cfg(),
        output_paths=_FakeOutputPaths(root=tmp_path),
        source_paths=None,
        methods=methods, concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
        decoder_only_bundle=bundle, encdec_bundle=bundle,
        target_model_metas={"decoder_only": None, "encdec": None},
        tokenizer_nanochat=tok,
        device=torch.device("cpu"),
        ddp_rank=0, ddp_world_size=1,
        diffmean_layer=1,
        inactive_sampler=_sentinel_inactive_sampler,
    )

    # Exactly one work unit should have been processed; the captured kwargs
    # must reference our sentinel sampler — NOT None.
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["inactive_sampler"] is _sentinel_inactive_sampler


def test_run_bpb_score_stage_defaults_inactive_sampler_to_none(
    tmp_path, monkeypatch,
):
    """When ``inactive_sampler`` is omitted, it must default to ``None`` —
    preserves the legacy behaviour for callers that don't pass one."""
    captured_kwargs: list[dict] = []

    def _spy_build_intervention(**kwargs):
        captured_kwargs.append(kwargs)
        method = kwargs["method"]
        return Intervention.prompting(target_model=method.target_model, prompt_prefix_tokens=[])

    monkeypatch.setattr(
        bpb_scoring_module, "build_intervention", _spy_build_intervention,
    )

    def _fake_score_sample(*, inputs, target_bundle, intervention,
                           device, coefficient_str, method_name,
                           lora=None):
        return BpbScoreRow(
            method_name=method_name, concept_id=inputs.concept_id,
            mode=inputs.mode.value, content_class=inputs.content_class,
            target=target_bundle.target_name,
            coefficient_str=coefficient_str, sample_id=inputs.sample_id,
            bpb_intervened=0.0, n_output_tokens=inputs.n_output,
            n_output_bytes=int(inputs.output_byte_weights.sum().item()),
            truncated=False,
        )

    monkeypatch.setattr(
        bpb_scoring_module, "score_sample", _fake_score_sample,
    )

    tok = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=object(), tokenizer=tok,
        max_seq_len=64, target_name="decoder_only",
    )

    methods = [_FakeMethod(
        name="m1", baseline="ours", mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )]
    concepts = [_FakeConcept(concept_id=1)]
    samples_by_concept_mode = {
        (1, SteeringMode.POSITIVE): [
            _make_scoring_input(tok, "abc", "x", 1, SteeringMode.POSITIVE),
        ],
    }
    coefficient_grids = {"m1": [0]}

    # No inactive_sampler kwarg passed — default must be None.
    run_bpb_score_stage(
        cfg=_Cfg(),
        output_paths=_FakeOutputPaths(root=tmp_path),
        source_paths=None,
        methods=methods, concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
        decoder_only_bundle=bundle, encdec_bundle=bundle,
        target_model_metas={"decoder_only": None, "encdec": None},
        tokenizer_nanochat=tok,
        device=torch.device("cpu"),
        ddp_rank=0, ddp_world_size=1,
        diffmean_layer=1,
    )

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["inactive_sampler"] is None
