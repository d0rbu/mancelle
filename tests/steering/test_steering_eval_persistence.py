"""Behavioral tests for steering_eval.py persistence + config validation.

Covers three still-live, formerly-tested areas that lost coverage in the
encdec-full-conditioning-cleanup PR:

* ``save_raw_data`` / ``load_raw_data`` round-trip (steering_eval.py:568/609)
* ``save_distractor_data`` / ``load_distractor_data`` round-trip (688/745)
* ``load_config`` validation (steering_eval.py:138)

Round-trip tests write real YAML to disk and read it back, asserting the
reconstructed dataclasses equal the originals — not just that the call returns.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

import steering.eval.steering_eval as steering_eval
from steering.eval.steering_eval import (
    DistractorConfig,
    FeatureSpec,
    GenerationConfig,
    JudgeChoice,
    JudgeConfig,
    RawFeatureData,
    RawJudgment,
    RawMagnitudeData,
    SteeringEvalConfig,
    Verdict,
    load_config,
    load_distractor_data,
    load_raw_data,
    save_distractor_data,
    save_raw_data,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# load_config — parsing + validation
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_config_parses_all_fields(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 42
                description: "sycophancy"
                confidence: 4
              - id: 99
                description: "humor"
                confidence: 2
            generation:
              max_new_tokens: 128
              temperature: 0.9
              top_k: 40
              samples_per_condition: 25
            judge:
              model: "openai/gpt-4o"
            magnitude_levels: [1, 5, 10]
            prompts:
              - "Tell me about dogs."
              - "Write a poem."
            """,
        )
        cfg = load_config(path)

        assert isinstance(cfg, SteeringEvalConfig)
        assert cfg.features == [
            FeatureSpec(id=42, description="sycophancy", confidence=4),
            FeatureSpec(id=99, description="humor", confidence=2),
        ]
        assert cfg.generation == GenerationConfig(
            max_new_tokens=128, temperature=0.9, top_k=40, samples_per_condition=25
        )
        assert cfg.judge == JudgeConfig(model="openai/gpt-4o")
        assert cfg.magnitude_levels == [1, 5, 10]
        assert cfg.prompts == ["Tell me about dogs.", "Write a poem."]
        assert cfg.distractors is None

    def test_defaults_applied(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 1
                description: "test feature"
                confidence: 3
            """,
        )
        cfg = load_config(path)
        assert cfg.generation == GenerationConfig()
        assert cfg.judge == JudgeConfig()
        assert cfg.magnitude_levels is None
        assert cfg.prompts is None

    def test_missing_features_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            generation:
              max_new_tokens: 100
            """,
        )
        with pytest.raises(ValueError, match="features"):
            load_config(path)

    def test_empty_features_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "features: []\n")
        with pytest.raises(ValueError, match="features"):
            load_config(path)

    def test_missing_feature_id_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - description: "no id here"
                confidence: 3
            """,
        )
        with pytest.raises(ValueError, match="id"):
            load_config(path)

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 10
                confidence: 3
            """,
        )
        with pytest.raises(ValueError, match="description"):
            load_config(path)

    def test_missing_confidence_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 1
                description: "no confidence"
            """,
        )
        with pytest.raises(ValueError, match="confidence"):
            load_config(path)

    def test_confidence_out_of_range_raises(self, tmp_path: Path) -> None:
        # Below 1, above 5, and negative are all rejected.
        for bad_value in (0, 6, -1):
            path = _write_yaml(
                tmp_path,
                f"""\
                features:
                  - id: 1
                    description: "bad confidence"
                    confidence: {bad_value}
                """,
            )
            with pytest.raises(ValueError, match="confidence"):
                load_config(path)

    def test_distractors_section_parses(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 42
                description: "sycophancy"
                confidence: 4
            distractors:
              stats_path: "/tmp/stats.safetensors"
              distractor_counts: [0, 1, 5]
              seed: 99
            """,
        )
        cfg = load_config(path)
        assert cfg.distractors == DistractorConfig(
            stats_path="/tmp/stats.safetensors",
            distractor_counts=[0, 1, 5],
            seed=99,
        )

    def test_distractor_counts_auto_prepend_zero(self, tmp_path: Path) -> None:
        # When 0 is absent from distractor_counts it is prepended.
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 1
                description: "test"
                confidence: 3
            distractors:
              stats_path: "/tmp/stats.safetensors"
              distractor_counts: [1, 5, 20]
            """,
        )
        cfg = load_config(path)
        assert cfg.distractors is not None
        assert cfg.distractors.distractor_counts == [0, 1, 5, 20]

    def test_distractor_counts_default_includes_zero(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 1
                description: "test"
                confidence: 3
            distractors:
              stats_path: "/tmp/stats.safetensors"
            """,
        )
        cfg = load_config(path)
        assert cfg.distractors is not None
        assert cfg.distractors.distractor_counts[0] == 0

    def test_distractors_missing_stats_path_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            """\
            features:
              - id: 1
                description: "test"
                confidence: 3
            distractors:
              distractor_counts: [0, 1]
            """,
        )
        with pytest.raises(ValueError, match="stats_path"):
            load_config(path)


# ---------------------------------------------------------------------------
# save_raw_data / load_raw_data — round-trip
# ---------------------------------------------------------------------------


class TestRawDataRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        data = {
            583: RawFeatureData(
                feature_id=583,
                description="formal style",
                confidence=4,
                magnitude_data={
                    0: RawMagnitudeData(
                        level=0,
                        conditioned_samples=[("prompt", "gen1"), ("prompt", "gen2")],
                        baseline_samples=[("prompt", "base1"), ("prompt", "base2")],
                        judgments=[
                            RawJudgment(
                                pair_index=0,
                                conditioned_text="gen1",
                                baseline_text="base1",
                                ab_response=JudgeChoice.A,
                                ba_response=JudgeChoice.B,
                                verdict=Verdict.WIN,
                            ),
                            RawJudgment(
                                pair_index=1,
                                conditioned_text="gen2",
                                baseline_text="base2",
                                ab_response=JudgeChoice.TIE,
                                ba_response=JudgeChoice.TIE,
                                verdict=Verdict.TIE,
                            ),
                        ],
                    ),
                },
            ),
        }
        raw_path = tmp_path / "raw_data.yaml"
        save_raw_data(data, raw_path)
        loaded = load_raw_data(raw_path)

        assert set(loaded) == {583}
        feat = loaded[583]
        assert feat.feature_id == 583
        assert feat.description == "formal style"
        assert feat.confidence == 4

        mag0 = feat.magnitude_data[0]
        assert mag0.level == 0
        assert mag0.conditioned_samples == [("prompt", "gen1"), ("prompt", "gen2")]
        assert mag0.baseline_samples == [("prompt", "base1"), ("prompt", "base2")]
        assert len(mag0.judgments) == 2
        assert mag0.judgments[0].verdict == Verdict.WIN
        assert mag0.judgments[0].ab_response == JudgeChoice.A
        assert mag0.judgments[0].ba_response == JudgeChoice.B
        assert mag0.judgments[1].verdict == Verdict.TIE

    def test_invalid_verdict_preserves_raw_responses(self, tmp_path: Path) -> None:
        # INVALID verdicts may carry unparseable judge strings; these must
        # round-trip verbatim (not coerced into the JudgeChoice enum).
        data = {
            1: RawFeatureData(
                feature_id=1,
                description="test",
                confidence=3,
                magnitude_data={
                    0: RawMagnitudeData(
                        level=0,
                        conditioned_samples=[("prompt", "gen1")],
                        baseline_samples=[("prompt", "base1")],
                        judgments=[
                            RawJudgment(
                                pair_index=0,
                                conditioned_text="gen1",
                                baseline_text="base1",
                                ab_response="GARBAGE",
                                ba_response="NOPE",
                                verdict=Verdict.INVALID,
                            ),
                        ],
                    ),
                },
            ),
        }
        raw_path = tmp_path / "raw_data.yaml"
        save_raw_data(data, raw_path)
        loaded = load_raw_data(raw_path)

        j = loaded[1].magnitude_data[0].judgments[0]
        assert j.verdict == Verdict.INVALID
        assert j.ab_response == "GARBAGE"
        assert j.ba_response == "NOPE"

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        assert load_raw_data(tmp_path / "nonexistent.yaml") == {}


# ---------------------------------------------------------------------------
# save_distractor_data / load_distractor_data — round-trip
# ---------------------------------------------------------------------------


class TestDistractorDataRoundTrip:
    def test_round_trip_nested_keys(self, tmp_path: Path) -> None:
        data: dict[int, dict[int, RawFeatureData]] = {
            42: {
                0: RawFeatureData(
                    feature_id=42,
                    description="test",
                    confidence=4,
                    magnitude_data={
                        0: RawMagnitudeData(
                            level=0,
                            conditioned_samples=[("p", "c1")],
                            baseline_samples=[("p", "b1")],
                            judgments=[
                                RawJudgment(
                                    pair_index=0,
                                    conditioned_text="c1",
                                    baseline_text="b1",
                                    ab_response=JudgeChoice.A,
                                    ba_response=JudgeChoice.B,
                                    verdict=Verdict.WIN,
                                ),
                            ],
                        ),
                    },
                ),
                5: RawFeatureData(
                    feature_id=42,
                    description="test",
                    confidence=4,
                    magnitude_data={
                        0: RawMagnitudeData(
                            level=0,
                            conditioned_samples=[("p", "c2")],
                            baseline_samples=[("p", "b2")],
                            judgments=[
                                RawJudgment(
                                    pair_index=0,
                                    conditioned_text="c2",
                                    baseline_text="b2",
                                    ab_response=JudgeChoice.TIE,
                                    ba_response=JudgeChoice.TIE,
                                    verdict=Verdict.TIE,
                                ),
                            ],
                        ),
                    },
                ),
            },
        }
        path = tmp_path / "distractor_raw_data.yaml"
        save_distractor_data(data, path)
        loaded = load_distractor_data(path)

        assert set(loaded) == {42}
        assert set(loaded[42]) == {0, 5}
        assert loaded[42][0].magnitude_data[0].judgments[0].verdict == Verdict.WIN
        assert loaded[42][0].magnitude_data[0].conditioned_samples == [("p", "c1")]
        assert loaded[42][5].magnitude_data[0].judgments[0].verdict == Verdict.TIE
        assert loaded[42][5].confidence == 4

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        assert load_distractor_data(tmp_path / "nonexistent.yaml") == {}

    def test_distractor_run_resumes_only_canonical_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        existing = RawMagnitudeData(
            level=1,
            conditioned_samples=[("prompt", "conditioned")],
            baseline_samples=[("prompt", "baseline")],
            judgments=[
                RawJudgment(
                    pair_index=0,
                    conditioned_text="conditioned",
                    baseline_text="baseline",
                    ab_response=JudgeChoice.A,
                    ba_response=JudgeChoice.B,
                    verdict=Verdict.WIN,
                )
            ],
        )
        canonical_path = tmp_path / "distractor_raw_data.yaml"
        save_distractor_data(
            {
                42: {
                    0: RawFeatureData(
                        feature_id=42,
                        description="test",
                        confidence=4,
                        magnitude_data={1: existing},
                    )
                }
            },
            canonical_path,
        )

        resumed: list[RawMagnitudeData] = []

        def _resume_existing(**kwargs: Any) -> RawMagnitudeData:
            existing_mag_data = kwargs["existing_mag_data"]
            assert isinstance(existing_mag_data, RawMagnitudeData)
            resumed.append(existing_mag_data)
            return existing_mag_data

        def _reject_legacy_alias(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("distractor mode accessed the legacy raw_data.yaml alias")

        monkeypatch.setattr(steering_eval, "load_raw_data", _reject_legacy_alias)
        monkeypatch.setattr(steering_eval, "save_raw_data", _reject_legacy_alias)
        monkeypatch.setattr(
            steering_eval,
            "_evaluate_feature_at_magnitude",
            _resume_existing,
        )
        monkeypatch.setattr(
            steering_eval,
            "_compute_level_stats",
            lambda *_args: ([1.0], [1.0], [1.0], {}),
        )
        monkeypatch.setattr(steering_eval, "compute_spearman", lambda *_args: (1.0, 0.0))
        monkeypatch.setattr(steering_eval, "compute_isotonic_r2", lambda *_args: 1.0)
        monkeypatch.setattr(steering_eval, "plot_dose_response", lambda **_kwargs: None)
        monkeypatch.setattr(
            steering_eval,
            "plot_robustness_overlay",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            steering_eval,
            "plot_robustness_summary",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            steering_eval,
            "generate_distractor_report",
            lambda **_kwargs: None,
        )

        config = SteeringEvalConfig(
            features=[FeatureSpec(id=42, description="test", confidence=4)],
            distractors=DistractorConfig(
                stats_path="unused.safetensors",
                distractor_counts=[0],
            ),
        )
        steering_eval._run_distractor_eval(
            config,
            cast(Any, object()),
            tmp_path,
            tmp_path / "plots",
            [1],
            model=cast(Any, object()),
            tokenizer=object(),
            device=cast(Any, object()),
            context_provider=cast(Any, object()),
            target_mode=cast(Any, object()),
        )

        assert len(resumed) == 1
        assert resumed[0] == existing
        assert load_distractor_data(canonical_path)[42][0].magnitude_data[1] == existing
        assert not (tmp_path / "raw_data.yaml").exists()
