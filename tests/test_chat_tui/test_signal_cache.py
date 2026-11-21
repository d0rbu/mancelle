"""Contract-keyed persistence for fitted steering signals."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch as t
from loguru import logger
from safetensors.torch import load_file

from scripts.chat_tui.setup import load_chat_setup
from scripts.chat_tui.steering import (
    SignalSource,
    SignalStore,
    _compute_contract_digest,
    contract_digest_for_setup,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from tests.test_chat_tui.test_setup_loader import (
    FIXTURES,
    REAL_STATISTICS,
    _nested_lineage,
    requires_real_statistics,
)


def _counting_fitter(
    calls: list[int],
    scores: t.Tensor,
) -> Callable[[], t.Tensor]:
    def fit() -> t.Tensor:
        calls.append(len(calls))
        return scores.clone()

    return fit


def test_signal_store_roundtrip_fits_once_and_disk_reload_is_exact(tmp_path: Path) -> None:
    scores = t.tensor([-1.0, -0.0, 0.25, 1.0], dtype=t.float32)
    calls: list[int] = []
    store = SignalStore(tmp_path, "contract_a")
    fitter = _counting_fitter(calls, scores)

    fitted = store.get_or_fit(
        17,
        "text_a",
        fitter,
        name="honesty",
        source=SignalSource.AXBENCH_CONCEPT,
    )
    loaded = SignalStore(tmp_path, "contract_a").get_or_fit(
        17,
        "text_a",
        fitter,
        name="honesty",
        source=SignalSource.AXBENCH_CONCEPT,
    )

    assert calls == [0]
    assert t.equal(fitted.target_scores, scores)
    assert t.equal(loaded.target_scores, scores)
    assert loaded.identity == "contract_a/17_text_a"
    assert t.equal(
        load_file(str(tmp_path / "contract_a" / "17_text_a.safetensors"))["target_scores"],
        scores,
    )


def test_signal_store_key_is_sensitive_to_contract_text_and_concept(tmp_path: Path) -> None:
    scores = t.tensor([-0.5, 0.5], dtype=t.float32)
    calls: list[int] = []
    fitter = _counting_fitter(calls, scores)

    for contract, concept_id, text_digest in (
        ("contract_a", 17, "text_a"),
        ("contract_b", 17, "text_a"),
        ("contract_a", 17, "text_b"),
        ("contract_a", 18, "text_a"),
    ):
        SignalStore(tmp_path, contract).get_or_fit(
            concept_id,
            text_digest,
            fitter,
            name="concept",
            source=SignalSource.AXBENCH_CONCEPT,
        )

    assert len(calls) == 4


def test_corrupt_signal_warns_once_refits_and_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "contract_a" / "17_text_a.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not safetensors")
    scores = t.tensor([-0.75, 0.75], dtype=t.float32)
    calls: list[int] = []
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        signal = SignalStore(tmp_path, "contract_a").get_or_fit(
            17,
            "text_a",
            _counting_fitter(calls, scores),
            name="concept",
            source=SignalSource.AXBENCH_CONCEPT,
        )
    finally:
        logger.remove(sink)

    assert calls == [0]
    assert t.equal(signal.target_scores, scores)
    assert len(warnings) == 1
    assert str(path) in warnings[0]
    assert t.equal(load_file(str(path))["target_scores"], scores)


def test_signal_store_leaves_no_temporary_file(tmp_path: Path) -> None:
    SignalStore(tmp_path, "contract_a").get_or_fit(
        17,
        "text_a",
        lambda: t.tensor([-1.0, 1.0], dtype=t.float32),
        name="concept",
        source=SignalSource.AXBENCH_CONCEPT,
    )

    assert not list(tmp_path.rglob("*.tmp"))


def test_contract_digest_uses_canonical_descriptor_identity_payload() -> None:
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (7, 2, 5))
    cluster_sha = "a" * 64

    actual = _compute_contract_digest(
        feature_mask=feature_mask,
        cluster_descriptor_sha=cluster_sha,
        vocab_fingerprint="vocab_fingerprint",
        n_sae_features=65_536,
        stats_sha256="b" * 64,
    )

    assert actual == "d2efa4484fcebf89"


@pytest.mark.parametrize(
    "override",
    (
        {"feature_mask": FeatureMask(FeatureMaskMode.whitelist, (2, 5))},
        {"cluster_descriptor_sha": "c" * 64},
        {"vocab_fingerprint": "different_vocab"},
        {"stats_sha256": "d" * 64},
    ),
)
def test_contract_digest_changes_with_each_contract_component(override: dict) -> None:
    kwargs = {
        "feature_mask": FeatureMask(FeatureMaskMode.whitelist, (2, 5, 7)),
        "cluster_descriptor_sha": "a" * 64,
        "vocab_fingerprint": "vocab_fingerprint",
        "n_sae_features": 65_536,
        "stats_sha256": "b" * 64,
    }
    baseline = _compute_contract_digest(**kwargs)
    changed = _compute_contract_digest(**(kwargs | override))

    assert baseline == "d2efa4484fcebf89"
    assert changed != baseline


def test_contract_digest_full_vocab_fallback_depends_on_sae_width() -> None:
    first = _compute_contract_digest(
        feature_mask=None,
        cluster_descriptor_sha=None,
        vocab_fingerprint=None,
        n_sae_features=8,
        stats_sha256="b" * 64,
    )
    second = _compute_contract_digest(
        feature_mask=None,
        cluster_descriptor_sha=None,
        vocab_fingerprint=None,
        n_sae_features=9,
        stats_sha256="b" * 64,
    )

    assert first == "a809dd6459ea907f"
    assert second == "e6817680ba1a83a2"


@requires_real_statistics
def test_real_leaf_setup_contract_digest_golden(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    setup = load_chat_setup(leaf_path, REAL_STATISTICS)

    # golden: recompute only on deliberate identity change
    assert contract_digest_for_setup(setup) == "13f7c6bc888cbea3"


@requires_real_statistics
def test_pretrain_and_sft_leaf_share_signal_contract_identity(tmp_path: Path) -> None:
    leaf_path, _ancestor_path = _nested_lineage(tmp_path)
    leaf_setup = load_chat_setup(leaf_path, REAL_STATISTICS)
    pretrain_setup = load_chat_setup(FIXTURES / "meta_026500.json", REAL_STATISTICS)

    # Signals transfer because mask, clusters, compact vocab, and stats all match.
    assert contract_digest_for_setup(leaf_setup) == contract_digest_for_setup(pretrain_setup)
