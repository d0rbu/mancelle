"""Autointerp-ranked -> offline description JSON conversion and its id space."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch as t
from omegaconf import MissingMandatoryValue

import nanochat.flash_attention as _fa_mod
import steering.conditional.encoder_decoder.self_conditioned as selfcond_module
from nanochat.flash_attention import FlashAttnImpl
from scripts.chat_tui.descriptions import (
    OfflineDescriptionService,
    descriptions_from_autointerp_ranked,
    write_offline_descriptions,
)
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.refined_types import PositiveInt, UnitInterval
from tests.encdec_test_helpers import build_tiny_model
from tests.steering._self_conditioned_state import build_test_self_cond_state

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_RANKED_YAML = (
    _REPO_ROOT
    / "out"
    / "autointerp_filter"
    / "gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_20260710_145543"
    / "autointerp_ranked.yaml"
)

# Verbatim rows from the artifact above, including its optional error_message
# key and a target whose Neuronpedia lookup returned nothing.
_REAL_ROWS = """\
feature_dir: /out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt
score_threshold: 0.6
top_k: null
accepted_feature_ids:
- 3993
targets:
- description: such as
  feature_ids:
  - 3993
  score: 1.0
- description: override
  feature_ids:
  - 13842
  score: 1.0
  error_message: partial score from 16/40 requested token trials
- description: null
  feature_ids:
  - 21231
  score: 0.4
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "autointerp_ranked.yaml"
    path.write_text(text)
    return path


def _targets(body: str) -> str:
    return f"feature_dir: /features\nscore_threshold: 0.6\ntop_k: null\ntargets:\n{body}"


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Parsing the real artifact shape
# ---------------------------------------------------------------------------


def test_real_singleton_rows_parse_and_skip_absent_descriptions(tmp_path: Path) -> None:
    descriptions = descriptions_from_autointerp_ranked(_write(tmp_path, _REAL_ROWS))

    assert descriptions == {3993: "such as", 13842: "override"}


def test_multi_feature_targets_use_the_per_feature_list_shape(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _targets(
            "- description:\n"
            "  - feature_id: 11\n"
            "    description: alpha\n"
            "  - feature_id: 22\n"
            "    description: beta\n"
            "  feature_ids:\n"
            "  - 11\n"
            "  - 22\n"
            "  score: 0.9\n"
        ),
    )

    assert descriptions_from_autointerp_ranked(path) == {11: "alpha", 22: "beta"}


def test_repeated_identical_descriptions_are_accepted(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _targets(
            "- description: shared\n  feature_ids:\n  - 5\n  score: 1.0\n"
            "- description:\n"
            "  - feature_id: 5\n"
            "    description: shared\n"
            "  - feature_id: 6\n"
            "    description: other\n"
            "  feature_ids:\n"
            "  - 5\n"
            "  - 6\n"
            "  score: 0.8\n"
        ),
    )

    assert descriptions_from_autointerp_ranked(path) == {5: "shared", 6: "other"}


@pytest.mark.skipif(
    not _REAL_RANKED_YAML.is_file(),
    reason=f"real autointerp ranked artifact is absent: {_REAL_RANKED_YAML}",
)
def test_real_on_disk_artifact_parses_into_original_sae_ids() -> None:
    descriptions = descriptions_from_autointerp_ranked(_REAL_RANKED_YAML)

    assert len(descriptions) == 50931
    assert descriptions[3993] == "such as"
    assert descriptions[10638] == "locations"
    assert descriptions[21272] == "announcing or releasing statements"
    assert 21231 in descriptions
    assert max(descriptions) < 65536


# ---------------------------------------------------------------------------
# Fail-loud parsing
# ---------------------------------------------------------------------------


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    absent = tmp_path / "absent.yaml"

    with pytest.raises(FileNotFoundError, match=f"does not exist: {absent}"):
        descriptions_from_autointerp_ranked(absent)


def test_non_mapping_document_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a YAML mapping, got list"):
        descriptions_from_autointerp_ranked(_write(tmp_path, "- one\n- two\n"))


def test_missing_targets_list_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no 'targets' list"):
        descriptions_from_autointerp_ranked(_write(tmp_path, "score_threshold: 0.6\n"))


def test_non_mapping_target_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target 0 is not a mapping, got str"):
        descriptions_from_autointerp_ranked(_write(tmp_path, _targets("- just a string\n")))


def test_empty_feature_ids_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target 0 has no non-empty 'feature_ids' list"):
        descriptions_from_autointerp_ranked(
            _write(tmp_path, _targets("- description: x\n  feature_ids: []\n"))
        )


def test_boolean_feature_id_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target 0 has non-integer feature id True"):
        descriptions_from_autointerp_ranked(
            _write(tmp_path, _targets("- description: x\n  feature_ids:\n  - true\n"))
        )


def test_string_description_with_several_feature_ids_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pairs one 'description' string with 2 feature ids"):
        descriptions_from_autointerp_ranked(
            _write(tmp_path, _targets("- description: x\n  feature_ids:\n  - 1\n  - 2\n"))
        )


def test_unsupported_description_type_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'description' of unsupported type int"):
        descriptions_from_autointerp_ranked(
            _write(tmp_path, _targets("- description: 7\n  feature_ids:\n  - 1\n"))
        )


def test_description_entry_for_an_undeclared_feature_fails_loud(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _targets(
            "- description:\n"
            "  - feature_id: 99\n"
            "    description: stray\n"
            "  feature_ids:\n"
            "  - 1\n"
            "  - 2\n"
        ),
    )

    with pytest.raises(ValueError, match=r"describes feature 99, which is absent from its"):
        descriptions_from_autointerp_ranked(path)


def test_description_entry_without_a_feature_id_fails_loud(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _targets("- description:\n  - description: nameless\n  feature_ids:\n  - 1\n  - 2\n"),
    )

    with pytest.raises(ValueError, match="'feature_id' is not a feature id: None"):
        descriptions_from_autointerp_ranked(path)


def test_non_string_description_entry_fails_loud(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _targets(
            "- description:\n  - feature_id: 1\n    description: 5\n  feature_ids:\n  - 1\n  - 2\n"
        ),
    )

    with pytest.raises(ValueError, match="non-string 'description' for feature 1, got int"):
        descriptions_from_autointerp_ranked(path)


def test_conflicting_descriptions_for_one_feature_fail_loud(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _targets(
            "- description: first\n  feature_ids:\n  - 4\n  score: 1.0\n"
            "- description: second\n  feature_ids:\n  - 4\n  score: 0.9\n"
        ),
    )

    with pytest.raises(ValueError, match="feature 4 conflicting descriptions 'first' and 'second'"):
        descriptions_from_autointerp_ranked(path)


# ---------------------------------------------------------------------------
# Generator config contract
# ---------------------------------------------------------------------------


def test_generator_config_requires_a_ranked_path_and_defaults_the_output(hydra_compose) -> None:
    config_dir = str(_REPO_ROOT / "configs" / "extraction")
    cfg = hydra_compose(config_dir, "build_chat_tui_descriptions")

    assert sorted(cfg.keys()) == ["output_path", "ranked_yaml_path"]
    assert cfg.output_path == "out/chat_tui/descriptions.json"
    with pytest.raises(MissingMandatoryValue, match="ranked_yaml_path"):
        _ = cfg.ranked_yaml_path


# ---------------------------------------------------------------------------
# Writing the runtime JSON
# ---------------------------------------------------------------------------


def test_written_json_is_numerically_ordered_and_loads_back(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "descriptions.json"

    write_offline_descriptions({10: "ten", 2: "two", 100: "hundred"}, path)

    assert list(json.loads(path.read_text())) == ["2", "10", "100"]
    service = OfflineDescriptionService(path)
    resolved: list[tuple[int, str]] = []
    service.request_description(10, lambda feature_id, text: resolved.append((feature_id, text)))
    assert resolved == [(10, "ten")]
    assert len(service) == 3


def test_writer_rejects_non_integer_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'seven' is not a feature id"):
        write_offline_descriptions({"seven": "text"}, tmp_path / "descriptions.json")  # type: ignore[dict-item]


def test_writer_rejects_non_string_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature 7 must be a string, got int"):
        write_offline_descriptions({7: 7}, tmp_path / "descriptions.json")  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Id space: the ledger records ORIGINAL SAE ids, never compact head rows
# ---------------------------------------------------------------------------

_ORIGINAL_IDS = (1, 3, 5, 7)
_N_SAE_FEATURES = 8


def _compact_provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a compacted self-conditioning provider with a pinned selection size."""

    def fixed_counts(*_args: Any, **_kwargs: Any) -> tuple[t.Tensor, t.Tensor]:
        return t.tensor([2]), t.tensor([1])

    monkeypatch.setattr(selfcond_module, "sample_action_counts", fixed_counts)
    t.manual_seed(0)
    vocab = FeatureVocab(n_sae_features=_N_SAE_FEATURES, original_ids=_ORIGINAL_IDS)
    model = build_tiny_model(
        n_sae_features=_N_SAE_FEATURES,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
        feature_vocab=vocab,
    )
    rng = t.Generator(device="cpu")
    rng.manual_seed(123)
    config = SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(4),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=1,
                max_k=3,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=t.zeros((_N_SAE_FEATURES, 99), dtype=t.float64),
            rng=rng,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=PositiveInt.parse(_N_SAE_FEATURES),
        score_policy=SelfCondScoreInterpolation(
            target_scores=t.zeros(len(_ORIGINAL_IDS)),
            weight=UnitInterval.parse(0.0),
            negative_selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=0,
                max_k=2,
            ),
        ),
    )
    return SelfConditionedFeatureProvider(model, config, t.device("cpu"))


def test_compact_checkpoint_records_original_ids_not_head_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _compact_provider(monkeypatch)
    state = build_test_self_cond_state(provider)

    provider.step(state, t.tensor([[[9.0, 8.0, -8.0, -9.0]]]))

    # Head rows 0/1 rank highest and row 3 lowest; original_ids maps those to
    # SAE ids 1/3 and 7. The compact-row reading would be {0: 0, 1: 0, 3: 1}.
    assert provider.last_appended_segment == {1: 0, 3: 0, 7: 1}


def test_descriptions_key_on_the_recorded_ledger_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keying by compact head rows would mislabel ids the two spaces share."""

    provider = _compact_provider(monkeypatch)
    state = build_test_self_cond_state(provider)
    provider.step(state, t.tensor([[[9.0, 8.0, -8.0, -9.0]]]))
    segment = provider.last_appended_segment
    assert segment is not None

    ranked = _write(
        tmp_path,
        _targets(
            "".join(
                f"- description: original {feature_id}\n"
                f"  feature_ids:\n  - {feature_id}\n  score: 1.0\n"
                for feature_id in _ORIGINAL_IDS
            )
        ),
    )
    json_path = tmp_path / "descriptions.json"
    write_offline_descriptions(descriptions_from_autointerp_ranked(ranked), json_path)
    service = OfflineDescriptionService(json_path)
    resolved: dict[int, str] = {}
    for feature_id in sorted(segment):
        service.request_description(
            feature_id,
            lambda returned_id, text: resolved.__setitem__(returned_id, text),
        )

    assert resolved == {1: "original 1", 3: "original 3", 7: "original 7"}
    head_rows = {index for index, _ in enumerate(_ORIGINAL_IDS)}
    assert head_rows != set(segment)
