"""Tests for the shared axbench_base conftest fixtures.

Every fixture in ``tests/axbench_base/conftest.py`` and every mock class
under ``tests/axbench_base/fixtures/`` is exercised here. Integration
tests (Tasks 46/47) assume this coverage exists — keep these tests
green before adding new fixtures.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch as t

from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import Concept
from tests.axbench_base.fixtures.mock_base_sae import MockBaseSAE, _SAECfg
from tests.axbench_base.fixtures.mock_openrouter import MockOpenRouterClient

# ---------------------------------------------------------------------------
# MockOpenRouterClient
# ---------------------------------------------------------------------------


def test_mock_openrouter_returns_deterministic_responses(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    """Same prompt twice → same response."""
    client = mock_openrouter_client({"hi": "hello"})
    first = client.complete("hi")
    second = client.complete("hi")
    assert first == "hello"
    assert second == "hello"


def test_mock_openrouter_call_count_tracked(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    client = mock_openrouter_client({None: "fallback"})
    assert client.call_count == 0
    client.complete("prompt-a")
    client.complete("prompt-b")
    client.complete("prompt-c")
    assert client.call_count == 3
    assert client.recorded_prompts == ["prompt-a", "prompt-b", "prompt-c"]


def test_mock_openrouter_prompt_prefix_match(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    """If no exact match, first prefix match wins (insertion order)."""
    client = mock_openrouter_client({"concept:": "CONCEPT", "fluency:": "FLUENCY"})
    assert client.complete("concept: the color blue") == "CONCEPT"
    assert client.complete("fluency: the text is well-formed") == "FLUENCY"


def test_mock_openrouter_fallback_none_key(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    """None key is used as a wildcard fallback."""
    client = mock_openrouter_client({"exact": "EXACT", None: "DEFAULT"})
    assert client.complete("exact") == "EXACT"
    assert client.complete("anything else") == "DEFAULT"


def test_mock_openrouter_raises_keyerror_when_no_match(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    client = mock_openrouter_client({"only": "this"})
    with pytest.raises(KeyError, match="No mock response"):
        client.complete("something unrelated")


def test_mock_openrouter_score_method_parses_digits(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    """``score`` returns the first 0/1/2 digit in the canned response."""
    client = mock_openrouter_client({
        "p0": "rating: 0",
        "p1": "the score is 1 out of 2",
        "p2": "definitely a 2",
    })
    assert client.score("p0") == (0, 1)
    assert client.score("p1") == (1, 1)
    assert client.score("p2") == (2, 1)


def test_mock_openrouter_score_invalid_returns_none(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    """``"invalid"`` literal and digit-free responses both yield ``None``."""
    client = mock_openrouter_client({
        "bad": "invalid",
        "empty": "no digits here",
    })
    assert client.score("bad") == (None, 1)
    assert client.score("empty") == (None, 1)


def test_mock_openrouter_context_manager_closes(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    client = mock_openrouter_client({None: "ok"})
    with client as inner:
        assert inner is client
        inner.complete("x")
    # close is a no-op but must not raise.
    client.close()
    assert client.call_count == 1


def test_mock_openrouter_complete_accepts_extra_kwargs(
    mock_openrouter_client: Callable[..., MockOpenRouterClient],
) -> None:
    """``complete`` must accept the kwargs the real clients use."""
    client = mock_openrouter_client({None: "ok"})
    # AxBenchJudgeClient-style + LLMJudgeClient-style keyword args.
    assert client.complete("hi", cache_key="abc") == "ok"
    assert client.complete("hi", temperature=0.5, max_tokens=10) == "ok"


# ---------------------------------------------------------------------------
# MockBaseSAE
# ---------------------------------------------------------------------------


def test_mock_base_sae_deterministic_for_same_input(
    mock_base_sae: Callable[..., MockBaseSAE],
) -> None:
    """Two instances with the same seed + input shape agree exactly."""
    sae1 = mock_base_sae(feature_dim=128, seed=7)
    sae2 = mock_base_sae(feature_dim=128, seed=7)
    residual = t.zeros(2, 3, 16)
    out1 = sae1.encode(residual)
    out2 = sae2.encode(residual)
    assert t.equal(out1, out2)


def test_mock_base_sae_encode_shape(
    mock_base_sae: Callable[..., MockBaseSAE],
) -> None:
    sae = mock_base_sae(feature_dim=64, seed=1)
    residual = t.zeros(2, 5, 32)
    out = sae.encode(residual)
    assert out.shape == (2, 5, 64)


def test_mock_base_sae_produces_sparse_output(
    mock_base_sae: Callable[..., MockBaseSAE],
) -> None:
    """Exactly 5 features are active per token (ignoring collisions)."""
    sae = mock_base_sae(feature_dim=1024, seed=42)
    residual = t.zeros(1, 4, 16)
    out = sae.encode(residual)
    # Active count per token is at most 5; allow fewer on rare collisions.
    per_token_active = (out != 0).sum(dim=-1)
    assert (per_token_active <= 5).all()
    assert (per_token_active >= 1).all()
    total_zeros = (out == 0).sum().item()
    total_elements = out.numel()
    # Sparse: the overwhelming majority of the output must be zero.
    assert total_zeros / total_elements > 0.9


def test_mock_base_sae_different_seeds_differ(
    mock_base_sae: Callable[..., MockBaseSAE],
) -> None:
    sae_a = mock_base_sae(feature_dim=256, seed=1)
    sae_b = mock_base_sae(feature_dim=256, seed=2)
    residual = t.zeros(1, 3, 16)
    out_a = sae_a.encode(residual)
    out_b = sae_b.encode(residual)
    assert not t.equal(out_a, out_b)


def test_mock_base_sae_rejects_non_3d_input(
    mock_base_sae: Callable[..., MockBaseSAE],
) -> None:
    sae = mock_base_sae(feature_dim=16, seed=0)
    with pytest.raises(ValueError, match="expects a .B, T, C."):
        sae.encode(t.zeros(4, 8))


def test_mock_base_sae_cfg_and_d_sae(
    mock_base_sae: Callable[..., MockBaseSAE],
) -> None:
    sae = mock_base_sae(feature_dim=4096, seed=0)
    assert sae.d_sae == 4096
    assert isinstance(sae.cfg, _SAECfg)
    assert sae.cfg.d_sae == 4096
    assert sae.cfg.hook_name == "blocks.0.hook_resid_post"
    assert sae.cfg.release == "mock_release"
    assert sae.cfg.n_ctx == 2048


# ---------------------------------------------------------------------------
# tiny_concepts_path
# ---------------------------------------------------------------------------


def test_tiny_concepts_path_loads_valid_json(tiny_concepts_path: Path) -> None:
    assert tiny_concepts_path.exists()
    data = json.loads(tiny_concepts_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    # Every entry parses cleanly into the Concept dataclass.
    for item in data:
        concept = Concept(**item)
        assert isinstance(concept.concept_id, int)
        assert isinstance(concept.description, str)


def test_tiny_concepts_has_three_entries(tiny_concepts_path: Path) -> None:
    data: list[dict[str, Any]] = json.loads(
        tiny_concepts_path.read_text(encoding="utf-8"),
    )
    assert len(data) == 3
    ids = sorted(item["concept_id"] for item in data)
    assert ids == [0, 1, 2]


def test_tiny_concepts_have_unicode(tiny_concepts_path: Path) -> None:
    """Concept 2's description round-trips through UTF-8 without mojibake."""
    data: list[dict[str, Any]] = json.loads(
        tiny_concepts_path.read_text(encoding="utf-8"),
    )
    by_id = {item["concept_id"]: item for item in data}
    assert by_id[2]["description"] == "unicode test: café résumé"
    assert by_id[0]["description"] == "the color blue"
    assert by_id[1]["description"] == "scientific measurement"


def test_tiny_concepts_path_is_copy_not_source(
    tiny_concepts_path: Path,
) -> None:
    """Mutating the fixture's file must not touch the committed source."""
    source = Path(__file__).parents[1] / "fixtures" / "tiny_concepts.json"
    assert tiny_concepts_path != source
    tiny_concepts_path.write_text("[]", encoding="utf-8")
    # Source file unchanged.
    source_data = json.loads(source.read_text(encoding="utf-8"))
    assert len(source_data) == 3


# ---------------------------------------------------------------------------
# tiny_validation_shard_path
# ---------------------------------------------------------------------------


def test_tiny_validation_shard_path_returns_parquet_and_features(
    tiny_validation_shard_path: tuple[Path, Path],
) -> None:
    parquet_path, feature_path = tiny_validation_shard_path
    assert parquet_path.exists()
    assert parquet_path.suffix == ".parquet"
    # save_features appends .safetensors.zst to the stem.
    assert feature_path.exists()
    assert feature_path.name.endswith(".safetensors.zst")


# ---------------------------------------------------------------------------
# tmp_run_root
# ---------------------------------------------------------------------------


def test_tmp_run_root_sets_env_var(tmp_run_root: Path) -> None:
    assert os.environ["NANOCHAT_BASE_DIR"] == str(tmp_run_root)
    assert tmp_run_root.exists()
    assert tmp_run_root.is_dir()


def test_tmp_run_root_is_isolated_per_test(tmp_run_root: Path) -> None:
    """A second test sees its own ``tmp_path``; the env var tracks it."""
    probe = tmp_run_root / "probe.txt"
    probe.write_text("ok", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "ok"
    assert os.environ["NANOCHAT_BASE_DIR"] == str(tmp_run_root)


# ---------------------------------------------------------------------------
# make_valid_axbench_cfg
# ---------------------------------------------------------------------------


def test_make_valid_axbench_cfg_returns_config(
    make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
) -> None:
    cfg = make_valid_axbench_cfg()
    assert isinstance(cfg, AxBenchBaseConfig)
    assert cfg.run_tag == "test"
    assert cfg.device == "cuda"
    # Sensible locked-in defaults flow through untouched.
    assert cfg.concept_feature_mode == "canonical"
    assert cfg.n_positives_per_concept == 72


def test_make_valid_axbench_cfg_override_works(
    make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
) -> None:
    cfg = make_valid_axbench_cfg(
        run_tag="custom",
        max_concepts=3,
        n_positives_per_concept=2,
    )
    assert cfg.run_tag == "custom"
    assert cfg.max_concepts == 3
    assert cfg.n_positives_per_concept == 2
    # Non-overridden fields still have defaults.
    assert cfg.concept_feature_mode == "canonical"


def test_make_valid_axbench_cfg_output_root_in_tmp_path(
    make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
    tmp_path: Path,
) -> None:
    """``output_root`` points under the test's ``tmp_path`` by default."""
    cfg = make_valid_axbench_cfg()
    assert cfg.output_root is None
