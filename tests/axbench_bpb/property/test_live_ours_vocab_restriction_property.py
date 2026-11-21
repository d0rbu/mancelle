from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.encoder_input import build_mode_aware_encoder_input
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths

LIVE_METHODS = (
    "ours_extract_and_add_encdec",
    "ours_replace_random_encdec",
    "ours_replace_all_encdec",
    "neg_ours_remove_encdec",
    "neg_ours_add_to_negatives_encdec",
    "neg_ours_replace_random_negatives_encdec",
    "neg_ours_replace_all_negatives_encdec",
)


@st.composite
def _cases(draw: Any):
    width = draw(st.integers(min_value=2, max_value=32))
    allowed = sorted(
        draw(
            st.sets(
                st.integers(min_value=0, max_value=width - 1),
                min_size=1,
                max_size=width - 1,
            )
        )
    )
    top_k = draw(
        st.lists(
            st.integers(min_value=0, max_value=width - 1),
            min_size=0,
            max_size=10,
        )
    )
    sampled = draw(
        st.lists(
            st.sampled_from(allowed),
            min_size=0,
            max_size=min(8, len(allowed)),
            unique=True,
        )
    )
    num_buckets = draw(st.integers(min_value=2, max_value=8))
    coefficient = draw(st.integers(min_value=0, max_value=num_buckets - 1))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return allowed, top_k, sampled, num_buckets, coefficient, seed


def _meta(width: int, *, num_buckets: int = 4) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=num_buckets,
        n_sae_features=width,
        negative_selection={"distribution": "geometric", "mean": 3.0},
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


@pytest.mark.parametrize("method_name", LIVE_METHODS)
@settings(max_examples=25, deadline=None)
@given(case=_cases())
def test_compacted_output_matches_independently_filtered_modifier_oracle(
    method_name: str,
    case: tuple[list[int], list[int], list[int], int, int, int],
) -> None:
    allowed, top_k, sampled, num_buckets, coefficient, seed = case
    method = get_method_by_name(method_name)
    width = max(allowed + top_k + [1]) + 1
    meta = _meta(width, num_buckets=num_buckets)
    filtered = [fid for fid in top_k if fid in set(allowed)]

    def sampler(*, rng: random.Random) -> dict[int, int]:
        del rng
        return dict.fromkeys(sampled, num_buckets - 1)

    actual_rng = random.Random(seed)
    expected_rng = random.Random(seed)
    with patch.object(
        intervention_module,
        "_load_top_k_concept_fids",
        return_value=top_k,
    ):
        actual = build_intervention(
            method=method,
            concept=SimpleNamespace(concept_id=0),
            coefficient=coefficient,
            source_paths=SourcePaths(source_run_dir=Path("/unused")),
            diffmean_layer=8,
            target_model_meta=meta,
            inactive_sampler=sampler,
            tokenizer_nanochat=None,
            rng=actual_rng,
            model=SimpleNamespace(
                feature_vocab=object(),
                encoder=SimpleNamespace(feature_vocab_ids=t.tensor(allowed, dtype=t.int64)),
            ),
            apply_max_act_scale=False,
        )
    expected = build_mode_aware_encoder_input(
        method=method,
        active={},
        inactive={},
        top_k_concept_fids=filtered,
        coefficient=coefficient,
        target_model_meta=meta,
        inactive_sampler=sampler,
        rng=expected_rng,
    )

    assert actual.feature_conditions == expected
    assert set(actual.feature_conditions or {}).issubset(allowed)
    assert actual_rng.getstate() == expected_rng.getstate()


@pytest.mark.parametrize("method_name", LIVE_METHODS)
@settings(max_examples=20, deadline=None)
@given(
    top_k=st.lists(st.integers(min_value=0, max_value=31), max_size=10),
    coefficient=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_uncompacted_output_is_exact_raw_legacy_oracle(
    method_name: str,
    top_k: list[int],
    coefficient: int,
    seed: int,
) -> None:
    method = get_method_by_name(method_name)
    meta = _meta(32)

    def sampler(*, rng: random.Random) -> dict[int, int]:
        del rng
        return {1: 3, 5: 3, 9: 3}

    actual_rng = random.Random(seed)
    expected_rng = random.Random(seed)
    with patch.object(
        intervention_module,
        "_load_top_k_concept_fids",
        return_value=top_k,
    ):
        actual = build_intervention(
            method=method,
            concept=SimpleNamespace(concept_id=0),
            coefficient=coefficient,
            source_paths=SourcePaths(source_run_dir=Path("/unused")),
            diffmean_layer=8,
            target_model_meta=meta,
            inactive_sampler=sampler,
            tokenizer_nanochat=None,
            rng=actual_rng,
            model=SimpleNamespace(feature_vocab=None),
            apply_max_act_scale=False,
        )
    expected = build_mode_aware_encoder_input(
        method=method,
        active={},
        inactive={},
        top_k_concept_fids=top_k,
        coefficient=coefficient,
        target_model_meta=meta,
        inactive_sampler=sampler,
        rng=expected_rng,
    )
    assert list((actual.feature_conditions or {}).items()) == list(expected.items())
    assert actual_rng.getstate() == expected_rng.getstate()
