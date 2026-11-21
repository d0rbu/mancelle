"""Property test: synthesizing a record's expected YAML schema and parsing
it back yields the same record. Detects schema drift between writer and
parser."""

from __future__ import annotations

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.harmfulness_progression import HarmfulnessMetrics
from tests.harmfulness_progression.unit.test_parse_harmfulness_yaml import parse_harmfulness_yaml

_safe_floats = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_mean_scores = st.floats(
    min_value=1.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_signal_strings = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters="\n"),
    min_size=1,
    max_size=20,
)


@settings(max_examples=30, deadline=None)
@given(
    model_tag=st.sampled_from(["d12_baseline", "d13_base", "d12_fp"]),
    source=st.sampled_from(["base", "base/safety_cond", "base/safety_cond/capabilities_cond"]),
    feature_configs=_signal_strings,
    asr=_safe_floats,
    mean_score=_mean_scores,
    safe_rate=_safe_floats,
    compliance_rate=_safe_floats,
)
def test_round_trip_record_through_yaml(
    tmp_path_factory: pytest.TempPathFactory,
    model_tag: str,
    source: str,
    feature_configs: str,
    asr: float,
    mean_score: float,
    safe_rate: float,
    compliance_rate: float,
) -> None:
    p = tmp_path_factory.mktemp("rt") / "rec.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "harmfulness_eval_schema_version": 4,
                "config": {
                    "model_tag": model_tag,
                    "num_harmful": 700,
                    "source": source,
                    "feature_configs": feature_configs,
                },
                "harmful_prompts": {
                    "asr": asr,
                    "safe_response_rate": safe_rate,
                },
                "benign_prompts": {"compliance_rate": compliance_rate},
                "overall": {
                    "mean_score": mean_score,
                    "safe_response_rate": safe_rate,
                },
            }
        )
    )

    rec = parse_harmfulness_yaml(p)
    assert rec is not None
    assert rec.model_tag == model_tag
    assert rec.source == source
    assert rec.signal == feature_configs
    assert rec.full_conditioning is False
    assert rec.metrics == HarmfulnessMetrics(
        asr=float(asr),
        mean_score=float(mean_score),
        safe_rate=float(safe_rate),
        compliance_rate=float(compliance_rate),
    )
