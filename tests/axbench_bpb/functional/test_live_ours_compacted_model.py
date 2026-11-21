from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
import torch as t
from safetensors.torch import save_file

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base.encoder_input import build_mode_aware_encoder_input
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept, TargetModelMetadata
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    score_sample,
)
from steering.eval.axbench_bpb.tokenization import encode_output, encode_prefix
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer

N_SAE = 16
ALLOWED = (1, 3, 7, 11)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)


def _model(
    feature_vocab: FeatureVocab | None = VOCAB,
) -> EncoderDecoderGPT:
    with t.device("meta"):
        model = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=16,
                vocab_size=128,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="L",
            ),
            EncoderConfig(
                n_embd=32,
                depth=2,
                n_head=None,
                n_kv_head=None,
                head_dim=16,
            ),
            CrossAttentionConfig(
                frequency=1,
                n_head=None,
                n_kv_head=None,
            ),
            n_sae_features=N_SAE,
            num_buckets=4,
            feature_vocab=feature_vocab,
            pad_vocab_size_to=128,
        )
    model.to_empty(device="cpu")
    t.manual_seed(721)
    model.init_weights()
    model.eval()
    return model


def _source_paths(root: Path) -> SourcePaths:
    root.mkdir(parents=True, exist_ok=True)
    feature_ids = t.tensor([[3, 9, 11]], dtype=t.int64)
    for name in ("concept_features.safetensors", "concept_features_neg.safetensors"):
        save_file(
            {"feature_ids": feature_ids},
            str(root / name),
            metadata={"mode": "canonical"},
        )
    return SourcePaths(source_run_dir=root)


def _concept() -> Concept:
    return Concept(
        concept_id=0,
        feature_index=3,
        layer=8,
        sae_source="test",
        description="concept",
        corpus_count=1,
        description_source="test",
    )


def _meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=4,
        n_sae_features=N_SAE,
        negative_selection={"distribution": "geometric", "mean": 3.0},
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _inputs(
    tokenizer: SimpleTokenizer,
    *,
    mode: SteeringMode,
) -> ScoringInputs:
    prefix_text = "abc"
    output_text = "de"
    prefix_tokens = encode_prefix(prefix_text, tokenizer)
    output_tokens = encode_output(output_text, tokenizer)
    return ScoringInputs(
        sample_id=make_sample_id(
            concept_id=0,
            mode=mode,
            content_class="concept_rich",
            prefix_text=prefix_text,
            output_text=output_text,
        ),
        concept_id=0,
        mode=mode,
        content_class="concept_rich",
        prefix_text=prefix_text,
        prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=(
            ContextFeatureSegment(
                feature_ids=(1, 7),
                buckets=(1, 2),
                char_start=0,
                char_end=len(prefix_text) + 1,
            ),
        ),
        n_prefix=len(prefix_tokens),
        n_output=len(output_tokens),
        output_byte_weights=t.ones(len(output_tokens), dtype=t.int64),
        truncated=False,
    )


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [
        ("ours_extract_and_add_encdec", {3: 2, 11: 2}),
        ("neg_ours_replace_random_negatives_encdec", {3: 3, 11: 3}),
    ],
)
def test_real_compacted_model_scores_exactly_like_manually_filtered_oracle(
    tmp_path: Path,
    method_name: str,
    expected: dict[int, int],
) -> None:
    model = _model()
    tokenizer = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=16,
        target_name="encdec",
    )
    method = get_method_by_name(method_name)

    intervention = build_intervention(
        method=method,
        concept=_concept(),
        coefficient=2,
        source_paths=_source_paths(tmp_path),
        diffmean_layer=8,
        target_model_meta=_meta(),
        inactive_sampler=lambda *, rng: {1: 3, 7: 3},
        tokenizer_nanochat=tokenizer,
        rng=random.Random(0),
        apply_max_act_scale=False,
        model=model,
    )
    assert intervention.feature_conditions == expected

    inputs = _inputs(tokenizer, mode=method.mode)
    row = score_sample(
        inputs=inputs,
        target_bundle=bundle,
        intervention=intervention,
        device=t.device("cpu"),
        coefficient_str="2",
        method_name=method_name,
    )
    oracle = score_sample(
        inputs=inputs,
        target_bundle=bundle,
        intervention=Intervention.ours(feature_conditions=expected),
        device=t.device("cpu"),
        coefficient_str="2",
        method_name=f"{method_name}_oracle",
    )

    assert math.isfinite(row.bpb_intervened)
    assert row.bpb_intervened == oracle.bpb_intervened
    assert row.n_output_tokens == oracle.n_output_tokens
    assert row.n_output_bytes == oracle.n_output_bytes
    assert row.truncated == oracle.truncated


def test_real_compacted_model_teacher_force_backstop_still_rejects_raw_oov() -> None:
    model = _model()
    tokenizer = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=16,
        target_name="encdec",
    )
    with pytest.raises(AxBenchBpbError, match="offending ids.*9"):
        score_sample(
            inputs=_inputs(tokenizer, mode=SteeringMode.POSITIVE),
            target_bundle=bundle,
            intervention=Intervention.ours(feature_conditions={9: 2}),
            device=t.device("cpu"),
            coefficient_str="2",
            method_name="raw_oov_negative_control",
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "ours_extract_and_add_encdec",
        "ours_replace_random_encdec",
        "ours_replace_all_encdec",
        "neg_ours_remove_encdec",
        "neg_ours_add_to_negatives_encdec",
        "neg_ours_replace_random_negatives_encdec",
        "neg_ours_replace_all_negatives_encdec",
    ],
)
def test_real_uncompacted_model_matches_raw_legacy_score_oracle(
    tmp_path: Path,
    method_name: str,
) -> None:
    model = _model(feature_vocab=None)
    tokenizer = SimpleTokenizer()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=16,
        target_name="encdec",
    )
    method = get_method_by_name(method_name)
    meta = _meta()

    def sampler(*, rng: random.Random) -> dict[int, int]:
        del rng
        return {1: 3, 7: 3}

    actual_rng = random.Random(721)
    oracle_rng = random.Random(721)
    actual = build_intervention(
        method=method,
        concept=_concept(),
        coefficient=2,
        source_paths=_source_paths(tmp_path),
        diffmean_layer=8,
        target_model_meta=meta,
        inactive_sampler=sampler,
        tokenizer_nanochat=tokenizer,
        rng=actual_rng,
        apply_max_act_scale=False,
        model=model,
    )
    oracle_conditions = build_mode_aware_encoder_input(
        method=method,
        active={},
        inactive={},
        top_k_concept_fids=[3, 9, 11],
        coefficient=2,
        target_model_meta=meta,
        inactive_sampler=sampler,
        rng=oracle_rng,
    )

    assert list((actual.feature_conditions or {}).items()) == list(oracle_conditions.items())
    assert actual_rng.getstate() == oracle_rng.getstate()

    inputs = _inputs(tokenizer, mode=method.mode)
    actual_row = score_sample(
        inputs=inputs,
        target_bundle=bundle,
        intervention=actual,
        device=t.device("cpu"),
        coefficient_str="2",
        method_name=method_name,
    )
    oracle_row = score_sample(
        inputs=inputs,
        target_bundle=bundle,
        intervention=Intervention.ours(feature_conditions=oracle_conditions),
        device=t.device("cpu"),
        coefficient_str="2",
        method_name=f"{method_name}_oracle",
    )
    assert actual_row.bpb_intervened == oracle_row.bpb_intervened
    assert actual_row.n_output_tokens == oracle_row.n_output_tokens
    assert actual_row.n_output_bytes == oracle_row.n_output_bytes
    assert actual_row.truncated == oracle_row.truncated
