"""Base-equivalence goldens for the local precomputed packed loader.

The embedded digests were captured from a pristine origin/main 908d1c8cb
worktree on this box on 2026-08-30 via the differential harness whose fixture
writers, seeds, arm matrix, and digest function this module reproduces. Like the
#1596 prechange goldens, these digests are intentionally sensitive to PyTorch
behavior; the capture environment used torch 2.13.0+cu130.
"""

import contextlib
import copy
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.node_shard_cache import NodeLocalShardCache, NodeShardCacheConfig
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import save_packed_row_shard, save_selection_shard
from steering.refined_types import NonNegativeInt, PositiveInt
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    _make_identity_selection,
    _make_test_shard,
    build_threshold_pipeline,
)
from tests.steering.test_feature_clusters_precomputed_loader import (
    N_SAE,
    _pipeline,
    _streaming_splice_pipeline,
    _write_streaming_splice_fixture,
)
from tests.steering.test_padf_certificate import (
    _add_splice_certificate,
    _write_generic_fixture,
)

_BASE_DIGESTS: dict[str, list[str]] = {
    "clustered_cert": [
        "36e89c300a279e25f80a96bf6cd4774cbb4048f73fc49f5c41063c75fafd0332",
        "a5ecd0f17ca7321c5c9eef9cae9461ad21921efc3d7c62babf51680403c29440",
        "810ade9b60a967379cbef64abb777fc20a7e2a0ebab2dcf8c464ccd1ea228197",
        "899f9687a8b5b864d912cff65cfd6e4283396bb3cbceffd976f5062588e8df26",
        "41683f6b1437a4368f83fceed5276e976dc18ccdf4db7ed4b51585c99c06bf3a",
        "9e45fdfd60adec30c2d5d546c94a48c6b3598a807fdbcb761838981748c05759",
    ],
    "clustered_nocert": [
        "36e89c300a279e25f80a96bf6cd4774cbb4048f73fc49f5c41063c75fafd0332",
        "a5ecd0f17ca7321c5c9eef9cae9461ad21921efc3d7c62babf51680403c29440",
        "810ade9b60a967379cbef64abb777fc20a7e2a0ebab2dcf8c464ccd1ea228197",
        "899f9687a8b5b864d912cff65cfd6e4283396bb3cbceffd976f5062588e8df26",
        "41683f6b1437a4368f83fceed5276e976dc18ccdf4db7ed4b51585c99c06bf3a",
        "9e45fdfd60adec30c2d5d546c94a48c6b3598a807fdbcb761838981748c05759",
    ],
    "ddp_rank0": [
        "659906eee9045647cf8edac4278125bc456d85ee53d75e933ce43d8c7fe15923",
        "10138d5a8e63407cc5d487b32d8adc3a9d9559246dc6b2208c8c5e4285fb6d75",
        "8b8a2f59535956f320954f6ac80bc778868540086e720b61f0f28a54fc8842e8",
        "642f412ab56767503349852cf1ca5bd72f4d388d74e76cf7fa1e1502bc1bb04a",
        "9aba83f2640b8eaa9f11e52e7c844e51e70afc91f408968021cf379fe3aef76a",
        "d62a349c2c91bd8b714570d501f80e898f1aed629677e6301aa8d1f12a520fac",
        "cbb39bb9f7fd2c0250bd3c2bb3a239675299ad82e6329c8f606e87f99003df39",
        "0937c558db1e37c4343a162d9e1f9e6a01f9a1d3f766bb674ad1a7f1310e607d",
    ],
    "ddp_rank1": [
        "7bcbf008e000b4242678209857ec7a3dd4944d58b96d476c2b20df2498237679",
        "691245fc28610d3c15a3d2a5155b1a5ac2e3a80a5670dcbbba0617db019d602e",
        "3edaf8e5fecfe8ac1a07cb944529514bcc5548584110fa4e06d0beba522f485e",
        "8f9cc72314418b642fdbbe6dac4c6e8b30422cd1e8324f3b68b1bc6b65cc5f06",
        "e05d78720d104ac7770e2422fea94369e45460951c18a619b8821efb641d200e",
        "614ae2582cfdd6f91626416b530aea91e2a409627a0954a2e650ae2bdc0c9d17",
        "10f405c5e544bf2cea042292f9daacfb793ecd185fc7bd9f871032020b504796",
        "540d64cc1351d39e9a48d0cf8f1e62d7ab9311a5e25e8520c51352d95e13d5f3",
    ],
    "exact_resume": [
        "8b16297c178ed04039e700b5b554a4796d53500a9d561578f8c635b93b17ce21",
        "6b760029ae4afa6b2e3407cbe5ad3b333de09804d35bee35a7c2dd9fbb5a18e8",
        "8dfcef00e8497d3092821db191685d9e542ec5260bbe92e6c71d7622b950b35f",
        "2f2c9aa746ad5ab0c02c5ac600ef6dbf565a6018f99b64d074fc44cb319ecad5",
    ],
    "fused": [
        "d66deafebac831dc10cd86109a1139338757914d1eac7d2e992d871573b0e7c7",
        "8cdf6f6d92ab53efce222f15552a5491280363628ae1555cb8abe8de6c5eab3f",
        "4534f96522a0773dbfc310c9ce848bdee3984228cd3f2d4b92bca480e351283d",
        "07843363ef5aea1760257c77a71a16308f6d75483cfd9cbe887f9ad0d29071f9",
        "03aac3fe661ca39150dbcffd19cc2433699575757503dae1d7b0f6662560daa1",
        "ce9444da6babeadaec15fe3fbeb6f70f5611c8d06add257421239a1a29ae0a27",
        "872cb4ed97e5b26f771ad21d865453e860bfb74d6b83253612c0fd1ee59d8f27",
        "a06de2e88941dc218bd5fea0abc405e5e98aca84b293c324f54c65e876121ba1",
    ],
    "generic_selection": [
        "493ec58a5d4f804ce42f5273126b3985433016e54f5a5c60979900c6e54135de",
        "1b550dd13cfedade708dfb867a1c1395a310967018fff6cdc4ecd62df74506ed",
        "8b16297c178ed04039e700b5b554a4796d53500a9d561578f8c635b93b17ce21",
        "6b760029ae4afa6b2e3407cbe5ad3b333de09804d35bee35a7c2dd9fbb5a18e8",
        "8dfcef00e8497d3092821db191685d9e542ec5260bbe92e6c71d7622b950b35f",
        "2f2c9aa746ad5ab0c02c5ac600ef6dbf565a6018f99b64d074fc44cb319ecad5",
        "de66dd5eb9e8eac78b8863be3ff8b01f9f0e45498f5fd0b62d5822516e24abd0",
        "6b4058e351a0806e1248c6486dcf9ec11f210b11ba52c8ecff6c621c6ddc0f1d",
    ],
    "label_dropout": [
        "8f3709f88e01bac369f9e0188e7ba052e67a16220bcf39ba540ba3ead045cd54",
        "4d0ba105bbd85fad69aca4a41192f8e79eb95177d7ea72a98ff59e68e313da0a",
        "da5d967a8429272f0aea0e6636d150b7e91ab8bd00141c4a7d45740b08f878a8",
        "927b4150cd4cb68857c0fdd7cc356907fb84b33cf3df6e3f5d07290606c7f31e",
        "0262e51239d1e657c7a218fc7fefb3bb419710c8dceeaaa7bdf070781c4ba40d",
        "17ac4254d0beaaa312c22532d8a3faca34d0ab039f94ff79e2afabbc21784054",
        "39915a8e674e9ed204aaf3c09106e00708740c97aebde190e5a8dda1fb95728d",
        "baeaacc0caaa31c7451ca73c5f79d7c645260954cda1ce4acf1de8217ccf8079",
    ],
    "lookahead2": [
        "493ec58a5d4f804ce42f5273126b3985433016e54f5a5c60979900c6e54135de",
        "1b550dd13cfedade708dfb867a1c1395a310967018fff6cdc4ecd62df74506ed",
        "8b16297c178ed04039e700b5b554a4796d53500a9d561578f8c635b93b17ce21",
        "6b760029ae4afa6b2e3407cbe5ad3b333de09804d35bee35a7c2dd9fbb5a18e8",
        "8dfcef00e8497d3092821db191685d9e542ec5260bbe92e6c71d7622b950b35f",
        "2f2c9aa746ad5ab0c02c5ac600ef6dbf565a6018f99b64d074fc44cb319ecad5",
        "de66dd5eb9e8eac78b8863be3ff8b01f9f0e45498f5fd0b62d5822516e24abd0",
        "6b4058e351a0806e1248c6486dcf9ec11f210b11ba52c8ecff6c621c6ddc0f1d",
    ],
    "splice_cert": [
        "4b8d7664c842934b3f812c45107a05880ac75b6367326cdc74b3dc84222a5e67",
        "2395126149553986e984d511494b00ae1bf76dd825689a3aa38c3214be90baba",
        "232a26b16c62d73e44bc781bc46efa3d43d3e5dcc4c16376064e6c00eabcba4d",
        "771fe936c348b0f00cf6450af1b21f17dc88a176696c861d4545211dc11fe968",
        "1566eb66e619d78d2c90aa63e25a2661f13abd2f54b78df50dad94d8ee54b363",
        "e06e7e9379149d28b228d573cd4d3eed5627cb21aa2d18ed6a8973b7a2269d03",
    ],
    "splice_nocert": [
        "4b8d7664c842934b3f812c45107a05880ac75b6367326cdc74b3dc84222a5e67",
        "2395126149553986e984d511494b00ae1bf76dd825689a3aa38c3214be90baba",
        "232a26b16c62d73e44bc781bc46efa3d43d3e5dcc4c16376064e6c00eabcba4d",
        "771fe936c348b0f00cf6450af1b21f17dc88a176696c861d4545211dc11fe968",
        "1566eb66e619d78d2c90aa63e25a2661f13abd2f54b78df50dad94d8ee54b363",
        "e06e7e9379149d28b228d573cd4d3eed5627cb21aa2d18ed6a8973b7a2269d03",
    ],
}


def _batch_digest(batch: tuple) -> str:
    hasher = hashlib.sha256()
    for value in batch[:9]:
        if value is None:
            hasher.update(b"none\0")
            continue
        hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.cpu().contiguous().numpy().tobytes())
        hasher.update(b"\0")
    hasher.update(json.dumps(batch[9], sort_keys=True).encode())
    return hasher.hexdigest()


def _build_pipeline():
    boundaries = (
        t.linspace(
            1 / NUM_BUCKETS,
            1.0 - 1 / NUM_BUCKETS,
            NUM_BUCKETS - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _build_fixtures(root: Path) -> None:
    packed = root / "plain" / "packed"
    selection = root / "plain" / "sel"
    packed.mkdir(parents=True)
    selection.mkdir(parents=True)
    for index in range(4):
        shard = _make_test_shard(n_rows=5, seed=17 + 6 * index)
        name = f"shard_{index:05d}"
        save_packed_row_shard(shard, packed / name)
        save_selection_shard(_make_identity_selection(shard), selection / name)
    (packed / "manifest.json").write_text(json.dumps({"metadata": {}}))
    write_manifest(
        selection,
        "diff-harness-selection",
        OverlayKind.PACKED_ROW,
        metadata={"packed_rows": {"packed_row_fingerprint": packed_row_fingerprint(packed)}},
    )

    clustered_root = root / "clustered"
    clustered_root.mkdir(parents=True)
    fixture = _write_generic_fixture(clustered_root)
    manifest_path = fixture.selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    (clustered_root / "manifest_with_cert.json").write_text(json.dumps(manifest, indent=2))
    stripped = copy.deepcopy(manifest)
    stripped["metadata"].pop("clustered_selection_validation", None)
    (clustered_root / "manifest_without_cert.json").write_text(
        json.dumps(stripped, indent=2)
    )
    (clustered_root / "cluster_source.json").write_text(
        (clustered_root / "cluster-input" / "clusters.json").read_text()
    )

    splice_root = root / "splice"
    splice_root.mkdir(parents=True)
    splice_fixture = _write_streaming_splice_fixture(splice_root)
    manifest_path = splice_fixture.selection_dir / "manifest.json"
    without_cert = json.loads(manifest_path.read_text())
    _add_splice_certificate(splice_fixture)
    (splice_root / "manifest_with_cert.json").write_text(manifest_path.read_text())
    (splice_root / "manifest_without_cert.json").write_text(
        json.dumps(without_cert, indent=2)
    )


def _clustered_loader_kwargs(root: Path) -> dict:
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(root / "clustered" / "cluster_source.json")}},
        N_SAE,
    )
    clusters = FeatureClusterConsolidation.build(resolved, None)
    return {
        "packed_row_dir": root / "clustered" / "packed",
        "selection_dir": root / "clustered" / "packed" / "sel" / ("a" * 64),
        "feature_pipeline": _pipeline(clusters.feature_mask),
        "feature_clusters": clusters,
        "n_sae_features": N_SAE,
        "effective_buckets": 4,
    }


def _drain(loader: Iterator[tuple], count: int) -> tuple[list[str], list[dict]]:
    digests = []
    states = []
    with contextlib.closing(loader):
        for _ in range(count):
            batch = next(loader)
            digests.append(_batch_digest(batch))
            states.append(copy.deepcopy(batch[9]))
    return digests, states


def _run_arms(root: Path, cache: NodeLocalShardCache | None) -> dict[str, list[str]]:
    def make(arm_kwargs: dict) -> Iterator[tuple]:
        kwargs = dict(arm_kwargs)
        if cache is not None:
            kwargs["node_cache"] = cache
        return precomputed_data_loader(device="cpu", **kwargs)

    plain_packed = root / "plain" / "packed"
    plain_selection = root / "plain" / "sel"
    pipeline = _build_pipeline()
    results: dict[str, list[str]] = {}
    resume_states: dict[str, list[dict]] = {}

    def arm(name: str, count: int = 8, **kwargs) -> None:
        t.manual_seed(0)
        digests, states = _drain(make(kwargs), count)
        results[name] = digests
        resume_states[name] = states

    arm(
        "fused",
        B=3,
        packed_row_dir=plain_packed,
        feature_pipeline=pipeline,
        n_sae_features=N_SAE_FEATURES,
    )
    arm(
        "generic_selection",
        B=3,
        packed_row_dir=plain_packed,
        selection_dir=plain_selection,
        feature_pipeline=pipeline,
        n_sae_features=N_SAE_FEATURES,
    )
    arm(
        "label_dropout",
        B=3,
        packed_row_dir=plain_packed,
        selection_dir=plain_selection,
        feature_pipeline=pipeline,
        n_sae_features=N_SAE_FEATURES,
        label_dropout=0.3,
    )
    for rank in (0, 1):
        arm(
            f"ddp_rank{rank}",
            B=2,
            packed_row_dir=plain_packed,
            selection_dir=plain_selection,
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
            ddp_rank=rank,
            ddp_world_size=2,
        )
    arm(
        "lookahead2",
        B=3,
        packed_row_dir=plain_packed,
        selection_dir=plain_selection,
        feature_pipeline=pipeline,
        n_sae_features=N_SAE_FEATURES,
        train_shard_lookahead=2,
    )

    t.manual_seed(0)
    digests, _states = _drain(
        make(
            {
                "B": 3,
                "packed_row_dir": plain_packed,
                "selection_dir": plain_selection,
                "feature_pipeline": pipeline,
                "n_sae_features": N_SAE_FEATURES,
                "resume_state_dict": copy.deepcopy(resume_states["generic_selection"][2]),
            }
        ),
        4,
    )
    results["exact_resume"] = digests

    clustered = _clustered_loader_kwargs(root)
    clustered_manifest = clustered["selection_dir"] / "manifest.json"
    for name, source in (
        ("clustered_cert", "manifest_with_cert.json"),
        ("clustered_nocert", "manifest_without_cert.json"),
    ):
        clustered_manifest.write_text((root / "clustered" / source).read_text())
        t.manual_seed(0)
        digests, _states = _drain(
            make({"B": 2, "ddp_rank": 1, "ddp_world_size": 2, **clustered}),
            6,
        )
        results[name] = digests

    splice_root = root / "splice"
    splice_packed = splice_root / "producer" / "family" / "c1024" / "sall"
    splice_selection = splice_packed / "sel" / ("c" * 64)
    splice_features = splice_root / "producer" / "derived"
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(splice_root / "cluster-input" / "clusters.json")}},
        N_SAE,
    )
    splice_clusters = FeatureClusterConsolidation.build(resolved, None)
    splice_manifest = splice_selection / "manifest.json"
    for name, source in (
        ("splice_cert", "manifest_with_cert.json"),
        ("splice_nocert", "manifest_without_cert.json"),
    ):
        splice_manifest.write_text((splice_root / source).read_text())
        t.manual_seed(0)
        digests, _states = _drain(
            make(
                {
                    "B": 2,
                    "ddp_rank": 0,
                    "ddp_world_size": 2,
                    "packed_row_dir": splice_packed,
                    "selection_dir": splice_selection,
                    "splice_feature_dir": splice_features,
                    "feature_pipeline": _streaming_splice_pipeline(splice_clusters),
                    "feature_clusters": splice_clusters,
                    "n_sae_features": N_SAE,
                    "effective_buckets": 4,
                }
            ),
            6,
        )
        results[name] = digests
    return results


def test_base_equivalence_cache_off_cold_and_warm(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    _build_fixtures(fixture_root)

    cache_off = _run_arms(fixture_root, None)
    assert cache_off == _BASE_DIGESTS, "cache-off loader diverged from origin/main golden"

    cache_dir = tmp_path / "shared-cache"
    config = NodeShardCacheConfig(
        dir=cache_dir,
        max_bytes=PositiveInt.parse(10**12),
        reserve_bytes=NonNegativeInt.parse(0),
    )
    cold_cache = _run_arms(fixture_root, NodeLocalShardCache(config))
    assert cold_cache == _BASE_DIGESTS, "cold cache loader diverged from origin/main golden"

    warm_cache = _run_arms(fixture_root, NodeLocalShardCache(config))
    assert warm_cache == _BASE_DIGESTS, "warm cache loader diverged from origin/main golden"
