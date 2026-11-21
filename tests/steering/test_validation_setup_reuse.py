"""Issue #1615: persistent immutable setup with a fresh validation cursor."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import os
import textwrap
import threading
from collections.abc import Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
import torch.distributed as dist
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_train as train_module
import steering.conditional.encoder_decoder.dataloader as dataloader_module
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.encoder_decoder.dataloader import (
    PrecomputedValidationSetupCache,
    PrecomputedValidationSetupDriftError,
    precomputed_data_loader,
)
from steering.conditional.encoder_decoder.loss_eval import (
    evaluate_encdec_bpb_conditioned,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import DeltaBundle
from steering.gradient_overlay import provenance_dir
from steering.node_shard_cache import NodeLocalShardCache, NodeShardCacheConfig
from steering.refined_types import NonNegativeInt, PositiveInt
from tests.nccl_multiprocess_helpers import run_nccl_multiprocess
from tests.steering.test_feature_clusters_precomputed_loader import (
    _cluster_state,
    _pipeline,
    _streaming_splice_pipeline,
    _write_artifacts,
    _write_streaming_splice_fixture,
)
from tests.steering.test_padf_certificate import (
    _add_splice_certificate,
    _GenericFixture,
    _set_generic_certificate,
    _write_generic_fixture,
)
from tests.steering.test_precomputed_shard_lookahead import (
    _assert_batches_equal,
)

_ORIGIN_MAIN_CA1EA4E50_BATCH_SHA256 = (
    "1f367570638e2e81f4f3231a7546f55011d38fbaa830aa6a6171ba5caf3dab2c",
    "f917b574cbaca6706b8df999694925bfa831be4abc47872f7bff55d5cc734725",
    "e31f9084c38e61bfc85da6db3eff558593d99667e0bdae6505362b24ec8ccb3c",
    "729ec1e1b60d92f3844db841ed4e6711a77508224f226db2ee75ffeb7436bd22",
    "d519cf5e245bbbb8145389194e531fbe0fe4a36754d3b72dd1abbef880f2e638",
    "176b7b9d4ac28a60a621576b7ee6914fb6632f580903fca8bc15b56034b2061a",
    "79cfc4bde8da52b1e4664c0fe904633ba16360085f1e490099551e3a59b30ba1",
    "381086bfe88bd59c7d7c15d0d3cab31b699a2bf7e3e499b66fa7db1046f1bdc3",
)
_ORIGIN_MAIN_CA1EA4E50_SPLICE_BATCH_SHA256 = (
    "1604e4e6d0e0a4dc14cdb46c64231aa687a54281cae7abcbf612676577690fc6",
    "e5e51ea5980777efdc5440ed8ac9d33fa455ba275bbb0b7fb37d0973d0493edf",
    "1c80f1e858a6d7b92c3b4dfe0f6a2af6b748e6976fd82901aae60ad9b293c454",
    "934fa34b20b6b8cdb32290ed70d42159fa3a8b9f6e4459510a66188be739c740",
    "eb62f2d1b6a88b382b80ede2e3a27494b46558993dfd98bf796ce0f8fcc5f314",
    "a08d86ff53317b5da94916b918264ec7c64423364dded2a23de29bed4bd09630",
    "71bf37e52529378e9289c8ed1ca2cb1531e6f77f4234b9185202ab05f3eb623d",
    "0a8c2d973a16f7074b55813bbbfd96650cc4a95e36f2c76633b28b4b84a0664c",
)


def _canonical_batch_value(value: Any) -> Any:
    if isinstance(value, t.Tensor):
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().reshape(-1).tolist(),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[str(key), _canonical_batch_value(value[key])] for key in sorted(value)],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_canonical_batch_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_canonical_batch_value(item) for item in value],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported batch digest value: {type(value).__name__}")


def _batch_digest(batch: tuple[Any, ...]) -> str:
    payload = json.dumps(
        _canonical_batch_value(batch),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class _ContentSensitiveEvalModel:
    """Deterministic evaluator whose loss depends on every model input."""

    def __call__(
        self,
        inputs: t.Tensor,
        targets: t.Tensor,
        enc_fids: t.Tensor,
        enc_mags: t.Tensor,
        enc_mask: t.Tensor,
        seg_map: t.Tensor,
        *,
        loss_reduction: str,
    ) -> t.Tensor:
        assert loss_reduction == "none"
        feature_code = (
            (enc_fids.to(t.float32) + 1.0) * (enc_mags.to(t.float32) + 3.0) * enc_mask.to(t.float32)
        ).sum(dim=-1)
        safe_seg_map = seg_map.clamp(min=0, max=feature_code.shape[1] - 1)
        routed_code = feature_code.gather(1, safe_seg_map)
        return (
            1.0
            + inputs.to(t.float32) * 1e-3
            + targets.clamp_min(0).to(t.float32) * 2e-3
            + routed_code * 1e-5
            + safe_seg_map.to(t.float32) * 1e-4
            + (seg_map >= 0).to(t.float32) * 1e-6
        )


def _loader(
    fixture: _GenericFixture,
    *,
    setup_cache: PrecomputedValidationSetupCache | None,
    batch_size: int = 3,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    node_cache: NodeLocalShardCache | None = None,
) -> Generator[tuple[Any, ...]]:
    return precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=batch_size,
        feature_pipeline=fixture.pipeline,
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=fixture.pipeline.effective_buckets,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        feature_clusters=fixture.clusters,
        node_cache=node_cache,
        validation_setup_cache=setup_cache,
    )


def _gate(
    fixture: _GenericFixture,
    *,
    setup_cache: PrecomputedValidationSetupCache | None,
    batches: int,
    batch_size: int = 3,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    node_cache: NodeLocalShardCache | None = None,
) -> list[tuple[Any, ...]]:
    with contextlib.closing(
        _loader(
            fixture,
            setup_cache=setup_cache,
            batch_size=batch_size,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            node_cache=node_cache,
        )
    ) as loader:
        return [next(loader) for _ in range(batches)]


def _assert_streams_equal(
    actual: list[tuple[Any, ...]],
    expected: list[tuple[Any, ...]],
) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        _assert_batches_equal(actual_batch, expected_batch)


def _node_cache(root: Path) -> NodeLocalShardCache:
    return NodeLocalShardCache(
        NodeShardCacheConfig(
            dir=root,
            max_bytes=PositiveInt.parse(10_000_000),
            reserve_bytes=NonNegativeInt.parse(0),
        )
    )


def _mutate_same_length_and_restore_mtime(path: Path) -> tuple[bytes, os.stat_result]:
    before_bytes = path.read_bytes()
    before_stat = path.stat()
    assert before_bytes
    mutated = bytearray(before_bytes)
    index = len(mutated) // 2
    mutated[index] ^= 1
    path.write_bytes(mutated)
    os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    after_stat = path.stat()
    assert path.read_bytes() != before_bytes
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_ctime_ns != before_stat.st_ctime_ns
    return before_bytes, before_stat


def test_artifact_signature_rejects_wrong_object_kind(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact"
    file_path.write_bytes(b"artifact")
    directory_path = tmp_path / "directory"
    directory_path.mkdir()

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="artifact is not a file",
    ):
        dataloader_module._capture_validation_artifact(directory_path)
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="artifact is not a directory",
    ):
        dataloader_module._capture_validation_artifact(file_path, directory=True)


def test_authority_file_wraps_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"{}")
    real_read_bytes = Path.read_bytes

    def failing_read_bytes(path: Path) -> bytes:
        if path == manifest_path:
            raise PermissionError("injected manifest read failure")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="cannot read retained validation authority",
    ):
        dataloader_module._capture_validation_authority_file(manifest_path)


def test_optional_authority_wraps_lstat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"{}")
    real_lstat = Path.lstat

    def failing_lstat(path: Path):
        if path == manifest_path:
            raise PermissionError("injected lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", failing_lstat)
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="retained validation artifact is unavailable",
    ):
        dataloader_module._capture_validation_authority_file(manifest_path)


def test_authority_file_rejects_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"{}")
    real_read_bytes = Path.read_bytes
    mutated = False

    def mutating_read_bytes(path: Path) -> bytes:
        nonlocal mutated
        payload = real_read_bytes(path)
        if path == manifest_path and not mutated:
            mutated = True
            path.write_bytes(payload + b"\n")
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutating_read_bytes)
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="authority changed while it was read",
    ):
        dataloader_module._capture_validation_authority_file(manifest_path)
    assert mutated is True


def test_authority_capture_supports_splice_dir_without_sidecars(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    selection_dir = tmp_path / "selection"
    splice_dir = tmp_path / "splice"
    for directory in (packed_dir, selection_dir, splice_dir):
        directory.mkdir()
    for directory in (packed_dir, selection_dir):
        (directory / "manifest.json").write_bytes(b"{}")
    packed_body = packed_dir / "shard_00000.safetensors.zst"
    selection_body = selection_dir / "shard_00000.safetensors.zst"
    packed_body.write_bytes(b"packed")
    selection_body.write_bytes(b"selection")

    authority = dataloader_module._capture_precomputed_validation_authority(
        packed_row_dir=packed_dir,
        selection_dir=selection_dir,
        shard_paths=(packed_body,),
        selection_paths=(selection_body,),
        splice_feature_dir=splice_dir,
    )

    assert {path for path, _link, _target in authority[1]} == {
        packed_body.absolute(),
        selection_body.absolute(),
    }
    assert {path for path, _link, _target in authority[2]} == {
        splice_dir.absolute(),
    }

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="shard inventory changed",
    ):
        dataloader_module._capture_precomputed_validation_authority(
            packed_row_dir=packed_dir,
            selection_dir=selection_dir,
            shard_paths=(),
            selection_paths=(selection_body,),
            splice_feature_dir=splice_dir,
        )

    lineage_before = authority
    (splice_dir / "new-derived-file").write_bytes(b"changed inventory")
    lineage_after = dataloader_module._capture_precomputed_validation_authority(
        packed_row_dir=packed_dir,
        selection_dir=selection_dir,
        shard_paths=(packed_body,),
        selection_paths=(selection_body,),
        splice_feature_dir=splice_dir,
    )
    assert lineage_after != lineage_before


def test_three_gates_are_byte_exact_to_three_legacy_fresh_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    real_validate = dataloader_module._validate_generic_selection_dir
    validation_calls = 0

    def counted_validate(**kwargs: Any) -> int | None:
        nonlocal validation_calls
        validation_calls += 1
        return real_validate(**kwargs)

    monkeypatch.setattr(
        dataloader_module,
        "_validate_generic_selection_dir",
        counted_validate,
    )
    legacy = [_gate(fixture, setup_cache=None, batches=3) for _ in range(3)]
    assert validation_calls == 3

    validation_calls = 0
    setup_cache = PrecomputedValidationSetupCache()
    retained = [_gate(fixture, setup_cache=setup_cache, batches=3) for _ in range(3)]

    for actual_gate, expected_gate in zip(retained, legacy, strict=True):
        _assert_streams_equal(actual_gate, expected_gate)
    assert retained[0][2][9]["epoch"] == 2
    assert validation_calls == 1
    assert setup_cache._setup is not None


@pytest.mark.parametrize("use_setup_cache", [False, True])
def test_eight_batches_match_frozen_origin_main_oracle(
    tmp_path: Path,
    use_setup_cache: bool,
) -> None:
    """Pin all ten fields to untouched origin/main ca1ea4e50 output."""
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    setup_cache = PrecomputedValidationSetupCache() if use_setup_cache else None
    loader = _loader(fixture, setup_cache=setup_cache)
    with contextlib.closing(loader):
        actual = tuple(_batch_digest(next(loader)) for _ in range(8))

    assert actual == _ORIGIN_MAIN_CA1EA4E50_BATCH_SHA256


@pytest.mark.parametrize("use_setup_cache", [False, True])
def test_spliced_batches_match_frozen_origin_main_oracle(
    tmp_path: Path,
    use_setup_cache: bool,
) -> None:
    """Pin the spliced validator/traversal branch to untouched main output."""
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    pipeline = _streaming_splice_pipeline(fixture.clusters)
    loader = precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=2,
        feature_pipeline=pipeline,
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=pipeline.effective_buckets,
        splice_feature_dir=fixture.splice_feature_dir,
        feature_clusters=fixture.clusters,
        validation_setup_cache=(PrecomputedValidationSetupCache() if use_setup_cache else None),
    )
    with contextlib.closing(loader):
        actual = tuple(_batch_digest(next(loader)) for _ in range(8))

    assert actual == _ORIGIN_MAIN_CA1EA4E50_SPLICE_BATCH_SHA256


def test_legacy_packed_directory_without_manifest_reuses_exactly(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    packed_manifest = fixture.packed_dir / "manifest.json"
    packed_manifest.unlink()
    assert not packed_manifest.exists()

    expected = _gate(fixture, setup_cache=None, batches=3)
    setup_cache = PrecomputedValidationSetupCache()
    for _ in range(2):
        _assert_streams_equal(
            _gate(fixture, setup_cache=setup_cache, batches=3),
            expected,
        )

    setup = setup_cache._setup
    assert setup is not None
    _path, signature, content_sha256 = next(
        authority_file
        for authority_file in setup.authority[0]
        if authority_file[0] == packed_manifest.absolute()
    )
    assert signature is None
    assert content_sha256 is None


def test_adding_optional_packed_manifest_after_validation_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    packed_manifest = fixture.packed_dir / "manifest.json"
    packed_manifest.unlink()
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)

    packed_manifest.write_text(
        json.dumps({"checksum": "new", "metadata": {}}),
        encoding="utf-8",
    )
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="setup authority changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


@pytest.mark.parametrize(
    ("object_kind", "error_pattern"),
    [
        ("directory", "artifact is not a file"),
        ("dangling-symlink", "artifact is unavailable"),
        ("fifo", "artifact is not a file"),
    ],
)
def test_nonregular_optional_packed_manifest_appearance_fails_closed(
    tmp_path: Path,
    object_kind: str,
    error_pattern: str,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    packed_manifest = fixture.packed_dir / "manifest.json"
    packed_manifest.unlink()
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)

    if object_kind == "directory":
        packed_manifest.mkdir()
    elif object_kind == "dangling-symlink":
        packed_manifest.symlink_to("missing-manifest-target.json")
    elif object_kind == "fifo":
        os.mkfifo(packed_manifest)
    else:
        raise AssertionError(f"unhandled test object kind: {object_kind}")
    assert os.path.lexists(packed_manifest)

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match=error_pattern,
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


def test_missing_selection_manifest_preserves_canonical_failure(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    (fixture.selection_dir / "manifest.json").unlink()

    failures: list[str] = []
    for setup_cache in (None, PrecomputedValidationSetupCache()):
        with pytest.raises(ValueError, match="selection_dir .* is missing manifest.json") as exc:
            _gate(fixture, setup_cache=setup_cache, batches=1)
        failures.append(str(exc.value))
    assert failures[0] == failures[1]


def test_partial_gate_and_manual_failure_cannot_advance_the_next_gate(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    expected = _gate(fixture, setup_cache=None, batches=3)
    setup_cache = PrecomputedValidationSetupCache()

    with contextlib.closing(_loader(fixture, setup_cache=setup_cache)) as loader:
        first = next(loader)
        _assert_batches_equal(first, expected[0])
        # Simulate a model/evaluation failure after partial consumption. The
        # gate-local generator is abandoned exactly as the trainer abandons it.

    actual = _gate(fixture, setup_cache=setup_cache, batches=3)
    _assert_streams_equal(actual, expected)
    assert setup_cache._setup is not None


def test_unadvanced_generator_close_does_not_publish_setup(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()

    loader = _loader(fixture, setup_cache=setup_cache)
    loader.close()

    assert setup_cache._setup is None


def test_interleaved_warm_generators_own_independent_cursors(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    expected = _gate(fixture, setup_cache=None, batches=2)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)

    first = _loader(fixture, setup_cache=setup_cache)
    second = _loader(fixture, setup_cache=setup_cache)
    with contextlib.closing(first), contextlib.closing(second):
        _assert_batches_equal(next(first), expected[0])
        _assert_batches_equal(next(second), expected[0])
        _assert_batches_equal(next(first), expected[1])
        _assert_batches_equal(next(second), expected[1])


def test_failed_initial_validation_never_publishes_partial_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    setup_cache = PrecomputedValidationSetupCache()
    real_validate = dataloader_module._validate_generic_selection_dir
    fail_once = True

    def injected_failure(**kwargs: Any) -> int | None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("injected complete-validation failure")
        return real_validate(**kwargs)

    monkeypatch.setattr(
        dataloader_module,
        "_validate_generic_selection_dir",
        injected_failure,
    )
    with pytest.raises(RuntimeError, match="injected complete-validation failure"):
        _gate(fixture, setup_cache=setup_cache, batches=1)
    assert setup_cache._setup is None

    expected = _gate(fixture, setup_cache=None, batches=2)
    actual = _gate(fixture, setup_cache=setup_cache, batches=2)
    _assert_streams_equal(actual, expected)
    assert setup_cache._setup is not None


def test_setup_cache_rejects_double_publication(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    setup = setup_cache._setup
    assert setup is not None

    with pytest.raises(RuntimeError, match="published more than once"):
        setup_cache._publish(setup)


def test_direct_body_mutation_during_initial_validation_fails_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    real_validate = dataloader_module._validate_generic_selection_dir

    def mutate_after_validation(**kwargs: Any) -> int | None:
        result = real_validate(**kwargs)
        _mutate_same_length_and_restore_mtime(fixture.selection_paths[0])
        return result

    monkeypatch.setattr(
        dataloader_module,
        "_validate_generic_selection_dir",
        mutate_after_validation,
    )
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="direct bodies, or lineage directories changed during the complete canonical validation",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)
    assert setup_cache._setup is None


def test_every_simulated_ddp_rank_rewinds_its_exact_stripe(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    first_rows_by_rank: dict[int, list[list[Any]]] = {}
    for rank in (0, 1):
        expected = _gate(
            fixture,
            setup_cache=None,
            batches=3,
            batch_size=2,
            ddp_rank=rank,
            ddp_world_size=2,
        )
        setup_cache = PrecomputedValidationSetupCache()
        for _ in range(3):
            actual = _gate(
                fixture,
                setup_cache=setup_cache,
                batches=3,
                batch_size=2,
                ddp_rank=rank,
                ddp_world_size=2,
            )
            _assert_streams_equal(actual, expected)
        first_rows_by_rank[rank] = expected[0][9]["batch_rows"]
        assert setup_cache._setup is not None

    assert first_rows_by_rank[0] != first_rows_by_rank[1]
    rank_union = {(row[1], row[2]) for rows in first_rows_by_rank.values() for row in rows}
    assert len(rank_union) == 4
    assert {row_idx for _shard, row_idx in rank_union} == {0, 1}


def test_one_setup_is_independent_of_gate_batch_size(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    setup_cache = PrecomputedValidationSetupCache()

    _assert_streams_equal(
        _gate(
            fixture,
            setup_cache=setup_cache,
            batches=2,
            batch_size=3,
        ),
        _gate(
            fixture,
            setup_cache=None,
            batches=2,
            batch_size=3,
        ),
    )
    _assert_streams_equal(
        _gate(
            fixture,
            setup_cache=setup_cache,
            batches=3,
            batch_size=2,
        ),
        _gate(
            fixture,
            setup_cache=None,
            batches=3,
            batch_size=2,
        ),
    )


def test_one_setup_is_independent_of_rank_local_cursor(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    setup_cache = PrecomputedValidationSetupCache()

    for rank in (0, 1, 0):
        _assert_streams_equal(
            _gate(
                fixture,
                setup_cache=setup_cache,
                batches=3,
                batch_size=2,
                ddp_rank=rank,
                ddp_world_size=2,
            ),
            _gate(
                fixture,
                setup_cache=None,
                batches=3,
                batch_size=2,
                ddp_rank=rank,
                ddp_world_size=2,
            ),
        )


def test_retained_paths_survive_cwd_change_and_relative_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path / "artifacts", n_shards=4)
    expected = _gate(fixture, setup_cache=None, batches=2)
    setup_cache = PrecomputedValidationSetupCache()
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()

    def relative_gate(cwd: Path) -> list[tuple[Any, ...]]:
        monkeypatch.chdir(cwd)
        loader = precomputed_data_loader(
            packed_row_dir=Path(os.path.relpath(fixture.packed_dir, cwd)),
            B=3,
            feature_pipeline=fixture.pipeline,
            device="cpu",
            selection_dir=Path(os.path.relpath(fixture.selection_dir, cwd)),
            effective_buckets=fixture.pipeline.effective_buckets,
            feature_clusters=fixture.clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            return [next(loader) for _ in range(2)]

    _assert_streams_equal(relative_gate(first_cwd), expected)
    setup = setup_cache._setup
    assert setup is not None
    assert all(path.is_absolute() for path in setup.shard_paths)
    assert all(path.is_absolute() for path in setup.selection_paths)
    _assert_streams_equal(relative_gate(second_cwd), expected)


def test_uncertified_generic_body_scan_is_retained_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    _set_generic_certificate(fixture, present=False)
    real_validate = dataloader_module._validate_generic_selection_dir
    real_load_selection = dataloader_module.load_selection_shard
    selection_loads = 0
    validation_scan_loads: list[int] = []

    def counted_load_selection(path: Path):
        nonlocal selection_loads
        selection_loads += 1
        return real_load_selection(path)

    def counted_validate(**kwargs: Any) -> int | None:
        before = selection_loads
        result = real_validate(**kwargs)
        validation_scan_loads.append(selection_loads - before)
        return result

    monkeypatch.setattr(dataloader_module, "load_selection_shard", counted_load_selection)
    monkeypatch.setattr(
        dataloader_module,
        "_validate_generic_selection_dir",
        counted_validate,
    )
    expected = _gate(fixture, setup_cache=None, batches=2)
    validation_scan_loads.clear()
    setup_cache = PrecomputedValidationSetupCache()
    first = _gate(fixture, setup_cache=setup_cache, batches=2)
    second = _gate(fixture, setup_cache=setup_cache, batches=2)

    _assert_streams_equal(first, expected)
    _assert_streams_equal(second, expected)
    assert validation_scan_loads == [len(fixture.selection_paths)]


@pytest.mark.parametrize("masked", [False, True], ids=["unclustered", "masked-clustered"])
def test_uncertified_selection_variants_reuse_exactly(
    tmp_path: Path,
    masked: bool,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,)) if masked else None
    clusters = _cluster_state(tmp_path / "clusters", mask=mask) if masked else None
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        mask=mask,
    )
    pipeline = _pipeline(mask)

    def batches(setup_cache: PrecomputedValidationSetupCache | None) -> list[tuple[Any, ...]]:
        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=selection_dir,
            effective_buckets=pipeline.effective_buckets,
            feature_clusters=clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            return [next(loader) for _ in range(4)]

    expected = batches(None)
    setup_cache = PrecomputedValidationSetupCache()
    _assert_streams_equal(batches(setup_cache), expected)
    _assert_streams_equal(batches(setup_cache), expected)


@pytest.mark.parametrize("changed_identity", ["feature_mask", "feature_clusters"])
def test_mask_and_cluster_identity_drift_fail_before_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_identity: str,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    clusters = _cluster_state(tmp_path / "clusters", mask=mask)
    packed_dir, selection_dir = _write_artifacts(
        tmp_path,
        clusters=clusters,
        mask=mask,
    )
    pipeline = _pipeline(mask)
    setup_cache = PrecomputedValidationSetupCache()
    loader = precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=2,
        feature_pipeline=pipeline,
        device="cpu",
        selection_dir=selection_dir,
        effective_buckets=pipeline.effective_buckets,
        feature_clusters=clusters,
        validation_setup_cache=setup_cache,
    )
    with contextlib.closing(loader):
        next(loader)

    if changed_identity == "feature_mask":
        changed_mask = FeatureMask(FeatureMaskMode.blacklist, (2,))
        changed_clusters = _cluster_state(tmp_path / "changed-mask", mask=changed_mask)
        changed_pipeline = _pipeline(changed_mask)
    else:
        changed_clusters = _cluster_state(
            tmp_path / "changed-clusters",
            mask=mask,
            clusters=[[3, 6], [7, 8]],
        )
        changed_pipeline = pipeline

    def unexpected_discovery(_directory: Path) -> list[Path]:
        raise AssertionError("artifact discovery ran before request-identity rejection")

    monkeypatch.setattr(dataloader_module, "glob_overlay_shards", unexpected_discovery)
    changed_loader = precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=2,
        feature_pipeline=changed_pipeline,
        device="cpu",
        selection_dir=selection_dir,
        effective_buckets=changed_pipeline.effective_buckets,
        feature_clusters=changed_clusters,
        validation_setup_cache=setup_cache,
    )
    with (
        contextlib.closing(changed_loader),
        pytest.raises(
            PrecomputedValidationSetupDriftError,
            match="different artifact or configuration identity",
        ),
    ):
        next(changed_loader)


def test_original_and_spliced_named_caches_are_independent(
    tmp_path: Path,
) -> None:
    original = _write_generic_fixture(tmp_path / "original", n_shards=4)
    spliced_named = _write_generic_fixture(tmp_path / "spliced", n_shards=4)
    original_cache = PrecomputedValidationSetupCache()
    spliced_cache = PrecomputedValidationSetupCache()

    original_expected = _gate(original, setup_cache=None, batches=2)
    spliced_expected = _gate(spliced_named, setup_cache=None, batches=2)
    for _ in range(3):
        _assert_streams_equal(
            _gate(original, setup_cache=original_cache, batches=2),
            original_expected,
        )
        _assert_streams_equal(
            _gate(spliced_named, setup_cache=spliced_cache, batches=2),
            spliced_expected,
        )

    assert original_cache._setup is not None
    assert spliced_cache._setup is not None


def test_reusing_one_cache_for_a_different_spec_fails_before_yield(
    tmp_path: Path,
) -> None:
    first = _write_generic_fixture(tmp_path / "first")
    second = _write_generic_fixture(tmp_path / "second")
    setup_cache = PrecomputedValidationSetupCache()
    _gate(first, setup_cache=setup_cache, batches=1)

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="different artifact or configuration identity",
    ):
        _gate(second, setup_cache=setup_cache, batches=1)


@pytest.mark.parametrize("changed_path", ["packed", "selection"])
def test_each_artifact_path_identity_mismatch_fails_before_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_path: str,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)

    def unexpected_discovery(_directory: Path) -> list[Path]:
        raise AssertionError("artifact discovery ran before request-identity rejection")

    monkeypatch.setattr(dataloader_module, "glob_overlay_shards", unexpected_discovery)
    loader = precomputed_data_loader(
        packed_row_dir=(
            tmp_path / "different-packed" if changed_path == "packed" else fixture.packed_dir
        ),
        B=3,
        feature_pipeline=fixture.pipeline,
        device="cpu",
        selection_dir=(
            tmp_path / "different-selection"
            if changed_path == "selection"
            else fixture.selection_dir
        ),
        effective_buckets=fixture.pipeline.effective_buckets,
        feature_clusters=fixture.clusters,
        validation_setup_cache=setup_cache,
    )
    with (
        contextlib.closing(loader),
        pytest.raises(
            PrecomputedValidationSetupDriftError,
            match="different artifact or configuration identity",
        ),
    ):
        next(loader)


def test_effective_bucket_change_fails_before_reusing_setup(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    loader = precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=3,
        feature_pipeline=fixture.pipeline,
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=fixture.pipeline.effective_buckets + 1,
        feature_clusters=fixture.clusters,
        validation_setup_cache=setup_cache,
    )
    with (
        contextlib.closing(loader),
        pytest.raises(
            PrecomputedValidationSetupDriftError,
            match="different artifact or configuration identity",
        ),
    ):
        next(loader)


def test_pipeline_effective_bucket_change_fails_before_reusing_setup(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    original_effective_buckets = fixture.pipeline.effective_buckets
    _gate(fixture, setup_cache=setup_cache, batches=1)
    fixture.pipeline.effective_buckets += 1

    loader = precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=3,
        feature_pipeline=fixture.pipeline,
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=original_effective_buckets,
        feature_clusters=fixture.clusters,
        validation_setup_cache=setup_cache,
    )
    with (
        contextlib.closing(loader),
        pytest.raises(
            PrecomputedValidationSetupDriftError,
            match="different artifact or configuration identity",
        ),
    ):
        next(loader)


def test_irrelevant_splice_feature_dir_is_ignored_for_generic_selection(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    expected = _gate(fixture, setup_cache=None, batches=2)

    for ignored_path in (tmp_path / "missing-a", tmp_path / "missing-b"):
        loader = precomputed_data_loader(
            packed_row_dir=fixture.packed_dir,
            B=3,
            feature_pipeline=fixture.pipeline,
            device="cpu",
            selection_dir=fixture.selection_dir,
            effective_buckets=fixture.pipeline.effective_buckets,
            feature_clusters=fixture.clusters,
            splice_feature_dir=ignored_path,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            actual = [next(loader) for _ in range(2)]
        _assert_streams_equal(actual, expected)


@pytest.mark.parametrize(
    "changed_identity",
    ["splice_feature_dir", "boundaries", "delta_config"],
)
def test_spliced_request_identity_drift_fails_before_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_identity: str,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    pipeline = _streaming_splice_pipeline(fixture.clusters)
    setup_cache = PrecomputedValidationSetupCache()

    def splice_loader(splice_feature_dir: Path):
        return precomputed_data_loader(
            packed_row_dir=fixture.packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=fixture.selection_dir,
            effective_buckets=pipeline.effective_buckets,
            splice_feature_dir=splice_feature_dir,
            feature_clusters=fixture.clusters,
            validation_setup_cache=setup_cache,
        )

    with contextlib.closing(splice_loader(fixture.splice_feature_dir)) as loader:
        next(loader)

    changed_splice_feature_dir = fixture.splice_feature_dir
    if changed_identity == "splice_feature_dir":
        changed_splice_feature_dir = tmp_path / "different-splice-feature-dir"
    elif changed_identity == "boundaries":
        before = pipeline.boundaries.clone()
        pipeline.boundaries.add_(1)
        assert not t.equal(pipeline.boundaries, before)
    else:
        assert pipeline.delta_bundle is not None
        pipeline.delta_bundle = DeltaBundle(
            config=FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=0.5,
            ),
            percentile_table=pipeline.delta_bundle.percentile_table,
        )

    def unexpected_discovery(_directory: Path) -> list[Path]:
        raise AssertionError("artifact discovery ran before request-identity rejection")

    monkeypatch.setattr(dataloader_module, "glob_overlay_shards", unexpected_discovery)

    with (
        contextlib.closing(splice_loader(changed_splice_feature_dir)) as loader,
        pytest.raises(
            PrecomputedValidationSetupDriftError,
            match="different artifact or configuration identity",
        ),
    ):
        next(loader)


@pytest.mark.parametrize("target", ["packed", "selection"])
def test_same_length_body_mutation_with_restored_mtime_fails_closed(
    tmp_path: Path,
    target: str,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    body = (
        sorted(fixture.packed_dir.glob("*.safetensors.zst"))[0]
        if target == "packed"
        else fixture.selection_paths[0]
    )
    _mutate_same_length_and_restore_mtime(body)

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="setup authority changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


@pytest.mark.parametrize("manifest_owner", ["packed", "selection"])
def test_manifest_content_mutation_fails_closed(
    tmp_path: Path,
    manifest_owner: str,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    directory = fixture.packed_dir if manifest_owner == "packed" else fixture.selection_dir
    manifest_path = directory / "manifest.json"
    before = manifest_path.read_bytes()
    manifest_path.write_bytes(before + b"\n")
    assert manifest_path.read_bytes() != before

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="setup authority changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


def test_manifest_content_hash_catches_drift_even_with_spoofed_stat_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    setup = setup_cache._setup
    assert setup is not None
    manifest_path = fixture.selection_dir / "manifest.json"
    expected_signature = next(
        signature
        for path, signature, _content_sha256 in setup.authority[0]
        if path == manifest_path.absolute()
    )
    assert expected_signature is not None
    real_capture = dataloader_module._capture_validation_artifact

    def spoofed_capture(path: Path, *, directory: bool = False):
        if path.absolute() == manifest_path.absolute():
            return expected_signature
        return real_capture(path, directory=directory)

    monkeypatch.setattr(
        dataloader_module,
        "_capture_validation_artifact",
        spoofed_capture,
    )
    _mutate_same_length_and_restore_mtime(manifest_path)

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="setup authority changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


def test_manifest_authority_retains_digest_not_payload(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    setup = setup_cache._setup
    assert setup is not None

    manifest_authorities = setup.authority[0]
    assert len(manifest_authorities) == 2
    assert all(
        isinstance(content_sha256, str) and len(content_sha256) == 64
        for _path, _signature, content_sha256 in manifest_authorities
    )
    assert not any(
        isinstance(value, bytes) for authority in manifest_authorities for value in authority
    )


def test_certificate_mutation_fails_closed(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    manifest_path = fixture.selection_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    certificate = payload["metadata"]["clustered_selection_validation"]
    certificate["shards"][0]["max_features_per_segment"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"][
            "clustered_selection_validation"
        ]["shards"][0]["max_features_per_segment"]
        != fixture.certificate["shards"][0]["max_features_per_segment"]
    )

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="setup authority changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


def test_inventory_addition_fails_closed(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    extra = fixture.selection_dir / "shard_99999.safetensors.zst"
    extra.write_bytes(b"not trusted")
    assert extra.is_file()

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="shard inventory changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


def test_unrelated_packed_cache_publication_does_not_invalidate_setup(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    expected = _gate(fixture, setup_cache=None, batches=2)
    _assert_streams_equal(
        _gate(fixture, setup_cache=setup_cache, batches=2),
        expected,
    )
    cache_file = fixture.packed_dir / ".concurrent-fingerprint-cache.json"
    cache_file.write_text('{"published_by": "another_rank"}', encoding="utf-8")
    _assert_streams_equal(
        _gate(fixture, setup_cache=setup_cache, batches=2),
        expected,
    )


def test_warm_authority_reads_only_manifests_and_inventories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    setup = setup_cache._setup
    assert setup is not None
    real_read_bytes = Path.read_bytes
    real_glob = dataloader_module.glob_overlay_shards
    byte_reads: list[Path] = []
    inventories: list[Path] = []

    def recording_read_bytes(path: Path) -> bytes:
        byte_reads.append(path)
        return real_read_bytes(path)

    def recording_glob(directory: Path) -> list[Path]:
        inventories.append(directory)
        return real_glob(directory)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)
    monkeypatch.setattr(dataloader_module, "glob_overlay_shards", recording_glob)

    assert setup_cache._lookup(setup.request_identity) is setup
    assert byte_reads == [
        fixture.packed_dir / "manifest.json",
        fixture.selection_dir / "manifest.json",
    ]
    assert inventories == [fixture.packed_dir, fixture.selection_dir]
    assert not any(path.name.endswith(".safetensors.zst") for path in byte_reads)


def test_missing_direct_body_fails_before_traversal(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    removed = fixture.selection_paths[0]
    removed.unlink()
    assert not removed.exists()

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="shard inventory changed",
    ):
        _gate(fixture, setup_cache=setup_cache, batches=1)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"selection_dir": None}, "selection_dir is absent"),
        ({"label_dropout": 0.1}, "label_dropout is nonzero"),
        ({"resume_state_dict": {}}, "resume_state_dict is present"),
        ({"n_sae_features": 32}, "feature-prediction targets are active"),
        ({"fp_target_layout": object()}, "feature-prediction targets are active"),
        ({"packed_materializer": object()}, "packed materialization is active"),
        ({"packed_identity_materializer": object()}, "packed materialization is active"),
        ({"remote_materializer": object()}, "selection materialization is active"),
        (
            {"remote_derived_feature_fingerprint": "a" * 64},
            "remote derived-feature identity is active",
        ),
        ({"remote_read_cancel_event": threading.Event()}, "remote cancellation is active"),
        ({"train_shard_lookahead": 1}, "shard lookahead is active"),
    ],
)
def test_cache_rejects_non_validation_boundaries(
    tmp_path: Path,
    overrides: dict[str, Any],
    reason: str,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    kwargs: dict[str, Any] = {
        "packed_row_dir": fixture.packed_dir,
        "B": 3,
        "feature_pipeline": fixture.pipeline,
        "device": "cpu",
        "selection_dir": fixture.selection_dir,
        "feature_clusters": fixture.clusters,
        "validation_setup_cache": PrecomputedValidationSetupCache(),
    }
    kwargs.update(overrides)
    loader = precomputed_data_loader(**kwargs)
    with contextlib.closing(loader), pytest.raises(ValueError, match=reason):
        next(loader)


def test_cache_contains_no_cursor_tensor_handle_or_decoded_shard(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    setup_cache = PrecomputedValidationSetupCache()
    _gate(fixture, setup_cache=setup_cache, batches=1)
    setup = setup_cache._setup
    assert setup is not None
    assert isinstance(setup.request_identity, str)
    assert all(isinstance(path, Path) for path in setup.shard_paths)
    assert not any(isinstance(value, t.Tensor) for value in vars(setup).values())
    assert not any(isinstance(value, Iterator) for value in vars(setup).values())


def test_metric_outputs_are_exact_across_legacy_and_reused_gates(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    token_bytes = t.ones(32_000, dtype=t.int64)
    expected = evaluate_encdec_bpb_conditioned(
        _ContentSensitiveEvalModel(),
        _loader(fixture, setup_cache=None),
        eval_steps=3,
        token_bytes=token_bytes,
        device=t.device("cpu"),
    )
    setup_cache = PrecomputedValidationSetupCache()
    for _ in range(3):
        actual = evaluate_encdec_bpb_conditioned(
            _ContentSensitiveEvalModel(),
            _loader(fixture, setup_cache=setup_cache),
            eval_steps=3,
            token_bytes=token_bytes,
            device=t.device("cpu"),
        )
        assert actual == expected


def test_model_failure_during_evaluation_cannot_advance_the_next_gate(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    token_bytes = t.arange(32_768, dtype=t.int64).remainder(7).add(1)
    expected = evaluate_encdec_bpb_conditioned(
        _ContentSensitiveEvalModel(),
        _loader(fixture, setup_cache=None),
        eval_steps=3,
        token_bytes=token_bytes,
        device=t.device("cpu"),
    )
    setup_cache = PrecomputedValidationSetupCache()

    class FailOnSecondBatch(_ContentSensitiveEvalModel):
        calls = 0

        def __call__(self, *args: Any, **kwargs: Any) -> t.Tensor:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected evaluation failure")
            return super().__call__(*args, **kwargs)

    failed_loader = _loader(fixture, setup_cache=setup_cache)
    with (
        contextlib.closing(failed_loader),
        pytest.raises(RuntimeError, match="injected evaluation failure"),
    ):
        evaluate_encdec_bpb_conditioned(
            FailOnSecondBatch(),
            failed_loader,
            eval_steps=3,
            token_bytes=token_bytes,
            device=t.device("cpu"),
        )
    assert setup_cache._setup is not None

    actual = evaluate_encdec_bpb_conditioned(
        _ContentSensitiveEvalModel(),
        _loader(fixture, setup_cache=setup_cache),
        eval_steps=3,
        token_bytes=token_bytes,
        device=t.device("cpu"),
    )
    assert actual == expected


def test_zero_width_selection_retains_pad_f_one_exactly(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4, zero_width=True)
    expected = _gate(fixture, setup_cache=None, batches=3)
    setup_cache = PrecomputedValidationSetupCache()
    for _ in range(2):
        actual = _gate(fixture, setup_cache=setup_cache, batches=3)
        _assert_streams_equal(actual, expected)
        assert all(batch[2].shape[-1] == 1 for batch in actual)
    setup = setup_cache._setup
    assert setup is not None
    assert setup.pad_f == 1


def test_node_cache_off_cold_and_warm_are_exact_with_setup_reuse(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path / "artifacts", n_shards=4)
    baseline = _gate(fixture, setup_cache=None, batches=4)
    node_cache = _node_cache(tmp_path / "node-cache")
    setup_cache = PrecomputedValidationSetupCache()
    cold = _gate(
        fixture,
        setup_cache=setup_cache,
        batches=4,
        node_cache=node_cache,
    )
    cold_counters = node_cache.counters()
    warm = _gate(
        fixture,
        setup_cache=setup_cache,
        batches=4,
        node_cache=node_cache,
    )
    warm_counters = node_cache.counters()

    _assert_streams_equal(cold, baseline)
    _assert_streams_equal(warm, baseline)
    assert cold_counters["fills"] >= 8
    assert warm_counters["hits"] > cold_counters["hits"]
    assert setup_cache._setup is not None


def test_spliced_fingerprint_and_provenance_validation_run_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    pipeline = _streaming_splice_pipeline(fixture.clusters)
    names = (
        "_validate_splice_counterfactual_selection_dir",
        "validate_packed_provenance",
        "splice_sidecar_fingerprint",
        "derived_feature_fingerprint",
        "assert_clustered_selection_shard_integrity",
    )
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(dataloader_module, name)

        def counted(*args: Any, _name: str = name, _original=original, **kwargs: Any):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(dataloader_module, name, counted)

    def splice_gate(
        setup_cache: PrecomputedValidationSetupCache | None,
    ) -> list[tuple[Any, ...]]:
        loader = precomputed_data_loader(
            packed_row_dir=fixture.packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=fixture.selection_dir,
            effective_buckets=pipeline.effective_buckets,
            splice_feature_dir=fixture.splice_feature_dir,
            feature_clusters=fixture.clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            return [next(loader) for _ in range(2)]

    expected = splice_gate(None)
    calls.update(dict.fromkeys(names, 0))
    setup_cache = PrecomputedValidationSetupCache()
    first = splice_gate(setup_cache)
    second = splice_gate(setup_cache)
    third = splice_gate(setup_cache)
    _assert_streams_equal(first, expected)
    _assert_streams_equal(second, expected)
    _assert_streams_equal(third, expected)
    assert calls == dict.fromkeys(names, 1)
    assert setup_cache._setup is not None


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("delete", "setup authority changed"),
        ("add", "setup authority changed"),
        ("replace", "setup authority changed"),
    ],
)
def test_spliced_packed_provenance_drift_fails_before_warm_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_pattern: str,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    pipeline = _streaming_splice_pipeline(fixture.clusters)
    setup_cache = PrecomputedValidationSetupCache()

    def splice_gate() -> list[tuple[Any, ...]]:
        loader = precomputed_data_loader(
            packed_row_dir=fixture.packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=fixture.selection_dir,
            effective_buckets=pipeline.effective_buckets,
            splice_feature_dir=fixture.splice_feature_dir,
            feature_clusters=fixture.clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            return [next(loader) for _ in range(2)]

    splice_gate()
    provenance_paths = sorted(provenance_dir(fixture.packed_dir).glob("*.safetensors.zst"))
    assert provenance_paths
    if mutation == "delete":
        provenance_paths[-1].unlink()
        assert not provenance_paths[-1].exists()
    elif mutation == "add":
        extra = provenance_dir(fixture.packed_dir) / "shard_99999.safetensors.zst"
        extra.write_bytes(b"untrusted provenance")
        assert extra.is_file()
    elif mutation == "replace":
        _mutate_same_length_and_restore_mtime(provenance_paths[-1])
    else:
        raise AssertionError(f"unhandled provenance mutation: {mutation}")

    def unexpected_traversal(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("body traversal ran before provenance drift rejection")

    for name in (
        "_load_packed_row_shard_with_node_cache",
        "_load_selection_shard_with_node_cache",
        "_shared_decode_load_base_shard",
        "load_packed_row_shard",
        "load_selection_shard",
    ):
        monkeypatch.setattr(dataloader_module, name, unexpected_traversal)

    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match=error_pattern,
    ):
        splice_gate()


@pytest.mark.parametrize("lineage_directory", ["derived-feature", "sidecar"])
def test_spliced_lineage_directory_drift_fails_before_warm_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage_directory: str,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    pipeline = _streaming_splice_pipeline(fixture.clusters)
    setup_cache = PrecomputedValidationSetupCache()

    def splice_gate() -> tuple[Any, ...]:
        loader = precomputed_data_loader(
            packed_row_dir=fixture.packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=fixture.selection_dir,
            effective_buckets=pipeline.effective_buckets,
            splice_feature_dir=fixture.splice_feature_dir,
            feature_clusters=fixture.clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            return next(loader)

    splice_gate()
    changed_directory = (
        fixture.splice_feature_dir
        if lineage_directory == "derived-feature"
        else fixture.splice_feature_dir / "_splice_provenance"
    )
    before = changed_directory.stat()
    (changed_directory / "unexpected-artifact").write_bytes(b"drift")
    published = changed_directory.stat()
    os.utime(
        changed_directory,
        ns=(published.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    after = changed_directory.stat()
    assert (after.st_mtime_ns, after.st_ctime_ns) != (
        before.st_mtime_ns,
        before.st_ctime_ns,
    )

    def unexpected_traversal(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("body traversal ran before lineage drift rejection")

    monkeypatch.setattr(
        dataloader_module,
        "_load_packed_row_shard_with_node_cache",
        unexpected_traversal,
    )
    with pytest.raises(
        PrecomputedValidationSetupDriftError,
        match="setup authority changed",
    ):
        splice_gate()


@pytest.mark.parametrize("artifact", ["derived-feature", "sidecar"])
def test_spliced_lineage_body_is_trusted_until_process_restart(
    tmp_path: Path,
    artifact: str,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    pipeline = _streaming_splice_pipeline(fixture.clusters)

    def splice_gate(
        setup_cache: PrecomputedValidationSetupCache,
    ) -> list[tuple[Any, ...]]:
        loader = precomputed_data_loader(
            packed_row_dir=fixture.packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=fixture.selection_dir,
            effective_buckets=pipeline.effective_buckets,
            splice_feature_dir=fixture.splice_feature_dir,
            feature_clusters=fixture.clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            return [next(loader) for _ in range(2)]

    setup_cache = PrecomputedValidationSetupCache()
    expected = splice_gate(setup_cache)
    path = (
        next(fixture.splice_feature_dir.glob("*_features.safetensors.zst"))
        if artifact == "derived-feature"
        else next((fixture.splice_feature_dir / "_splice_provenance").glob("*.json"))
    )
    parent_before = path.parent.stat()
    _mutate_same_length_and_restore_mtime(path)
    parent_after = path.parent.stat()
    assert parent_after.st_mtime_ns == parent_before.st_mtime_ns
    assert parent_after.st_ctime_ns == parent_before.st_ctime_ns

    # The approved contract trusts these already-certified lineage bodies for
    # this process lifetime, avoiding 2,800 per-file checks at every gate.
    reused = splice_gate(setup_cache)
    _assert_streams_equal(reused, expected)
    assert setup_cache._setup is not None

    # A fresh process/cache runs the complete validator and detects the drift.
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        splice_gate(PrecomputedValidationSetupCache())


def test_training_validation_loader_forwards_cache_only_to_local_precomputed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    setup_cache = PrecomputedValidationSetupCache()
    calls: list[dict[str, Any]] = []

    def recording_loader(**kwargs: Any) -> Iterator[tuple[Any, ...]]:
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(train_module, "precomputed_data_loader", recording_loader)
    spec = train_module.ValidationEvalSpec(
        metric_prefix="val",
        log_label="validation",
        packed_row_dir=tmp_path / "packed",
        selection_dir=selection_dir,
    )
    shared = SimpleNamespace(
        pad_F=32,
        ddp_rank=0,
        ddp_world_size=1,
        feature_clusters=None,
        remote_materializers={},
    )
    pipeline = SimpleNamespace(effective_buckets=10)

    train_module._build_validation_loader(
        spec=spec,
        cfg=OmegaConf.create({"device_batch_size": 2}),
        shared=shared,  # ty: ignore[invalid-argument-type]
        feature_pipeline=pipeline,  # ty: ignore[invalid-argument-type]
        tokenizer=object(),
        device=t.device("cpu"),
        validation_setup_cache=setup_cache,
    )

    assert len(calls) == 1
    assert calls[0]["validation_setup_cache"] is setup_cache
    assert calls[0]["remote_materializer"] is None


def test_training_validation_loader_never_forwards_cache_to_remote_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    setup_cache = PrecomputedValidationSetupCache()
    calls: list[dict[str, Any]] = []
    health_calls: list[str] = []

    class RemoteMaterializer:
        def request_startup_window(self) -> None:
            health_calls.append("request")

        def check_worker_errors(self) -> None:
            health_calls.append("check")

    remote_materializer = RemoteMaterializer()

    def recording_loader(**kwargs: Any) -> Iterator[tuple[Any, ...]]:
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(train_module, "precomputed_data_loader", recording_loader)
    spec = train_module.ValidationEvalSpec(
        metric_prefix="val",
        log_label="validation",
        packed_row_dir=tmp_path / "packed",
        selection_dir=selection_dir,
    )
    shared = SimpleNamespace(
        pad_F=32,
        ddp_rank=0,
        ddp_world_size=1,
        feature_clusters=None,
        remote_materializers={str(selection_dir.resolve()): remote_materializer},
    )

    train_module._build_validation_loader(
        spec=spec,
        cfg=OmegaConf.create({"device_batch_size": 2}),
        shared=shared,  # ty: ignore[invalid-argument-type]
        feature_pipeline=SimpleNamespace(effective_buckets=10),  # ty: ignore[invalid-argument-type]
        tokenizer=object(),
        device=t.device("cpu"),
        validation_setup_cache=setup_cache,
    )

    assert health_calls == ["request", "check"]
    assert len(calls) == 1
    assert calls[0]["validation_setup_cache"] is None
    assert calls[0]["remote_materializer"] is remote_materializer


def test_training_live_validation_loader_ignores_setup_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def recording_loader(**kwargs: Any) -> Iterator[tuple[Any, ...]]:
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(
        train_module,
        "encoder_decoder_data_loader_with_state",
        recording_loader,
    )
    spec = train_module.ValidationEvalSpec(
        metric_prefix="val",
        log_label="validation",
        packed_row_dir=None,
        selection_dir=None,
    )
    shared = SimpleNamespace(
        feature_index=object(),
        extraction_segment_size=128,
        pad_F=32,
        live_pad_n_seg=16,
    )

    train_module._build_validation_loader(
        spec=spec,
        cfg=OmegaConf.create(
            {
                "device_batch_size": 2,
                "max_seq_len": 8,
                "max_train_shards": None,
                "next_segment_conditioning_offset": False,
            }
        ),
        shared=shared,  # ty: ignore[invalid-argument-type]
        feature_pipeline=object(),  # ty: ignore[invalid-argument-type]
        tokenizer=object(),
        device=t.device("cpu"),
        validation_setup_cache=PrecomputedValidationSetupCache(),
    )

    assert len(calls) == 1
    assert calls[0]["split"] is train_module.DatasetSplit.VAL
    assert "validation_setup_cache" not in calls[0]


def test_trainer_owns_one_cache_per_packed_selection_spec(tmp_path: Path) -> None:
    original = train_module.ValidationEvalSpec(
        metric_prefix="val",
        log_label="original validation",
        packed_row_dir=tmp_path / "original-packed",
        selection_dir=tmp_path / "original-selection",
    )
    spliced = train_module.ValidationEvalSpec(
        metric_prefix="val_spliced",
        log_label="spliced validation",
        packed_row_dir=tmp_path / "spliced-packed",
        selection_dir=tmp_path / "spliced-selection",
    )
    live = train_module.ValidationEvalSpec(
        metric_prefix="live",
        log_label="live validation",
        packed_row_dir=None,
        selection_dir=None,
    )
    fused = train_module.ValidationEvalSpec(
        metric_prefix="fused",
        log_label="fused validation",
        packed_row_dir=tmp_path / "packed",
        selection_dir=None,
    )
    caches: dict[str, PrecomputedValidationSetupCache] = {}

    original_cache = train_module._validation_setup_cache_for_spec(original, caches)
    assert original_cache is not None
    assert train_module._validation_setup_cache_for_spec(original, caches) is original_cache
    spliced_cache = train_module._validation_setup_cache_for_spec(spliced, caches)
    assert spliced_cache is not None
    assert spliced_cache is not original_cache
    assert train_module._validation_setup_cache_for_spec(live, caches) is None
    assert train_module._validation_setup_cache_for_spec(fused, caches) is None
    assert caches == {"val": original_cache, "val_spliced": spliced_cache}


def test_trainer_cache_mapping_dominates_loop_and_is_passed_to_loader() -> None:
    """Pin the real trainer's cache lifetime and pass-through structurally."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(train_module.train_one_config)))
    while_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.Constant)
        and node.test.value is True
    ]
    assert len(while_loops) == 1
    loop = while_loops[0]
    cache_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "validation_setup_caches"
    ]
    assert len(cache_assignments) == 1
    assert cache_assignments[0].lineno < loop.lineno

    cache_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validation_setup_cache_for_spec"
    ]
    assert len(cache_calls) == 1
    assert any(
        isinstance(argument, ast.Name) and argument.id == "validation_setup_caches"
        for argument in cache_calls[0].args
    )
    loader_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_validation_loader"
    ]
    assert len(loader_calls) == 1
    cache_keywords = [
        keyword for keyword in loader_calls[0].keywords if keyword.arg == "validation_setup_cache"
    ]
    assert len(cache_keywords) == 1
    assert isinstance(cache_keywords[0].value, ast.Name)
    assert cache_keywords[0].value.id == "validation_setup_cache"


def _gloo_rewind_target(
    rank: int,
    world_size: int,
    packed_dir: str,
    selection_dir: str,
    clusters: Any,
    pipeline: Any,
) -> dict[str, Any]:
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    dataloader_module._loader_health_reset()
    setup_cache = PrecomputedValidationSetupCache()
    baseline_loader = precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=2,
        feature_pipeline=pipeline,
        device="cpu",
        selection_dir=selection_dir,
        effective_buckets=pipeline.effective_buckets,
        ddp_rank=rank,
        ddp_world_size=world_size,
        feature_clusters=clusters,
    )
    with contextlib.closing(baseline_loader):
        baseline_batches = [next(baseline_loader) for _ in range(2)]
    baseline_digests = [_batch_digest(batch) for batch in baseline_batches]
    baseline_rows = [batch[9]["batch_rows"] for batch in baseline_batches]
    dist.barrier()

    gate_digests: list[list[str]] = []
    gate_rows: list[list[list[Any]]] = []
    for _ in range(3):
        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=selection_dir,
            effective_buckets=pipeline.effective_buckets,
            ddp_rank=rank,
            ddp_world_size=world_size,
            feature_clusters=clusters,
            validation_setup_cache=setup_cache,
        )
        with contextlib.closing(loader):
            batches = [next(loader) for _ in range(2)]
        gate_digests.append([_batch_digest(batch) for batch in batches])
        gate_rows.append([batch[9]["batch_rows"] for batch in batches])
        dist.barrier()

    token_bytes = t.arange(32_768, dtype=t.int64).remainder(7).add(1)

    def evaluate(setup: PrecomputedValidationSetupCache | None) -> dict[str, Any]:
        active_loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=selection_dir,
            effective_buckets=pipeline.effective_buckets,
            ddp_rank=rank,
            ddp_world_size=world_size,
            feature_clusters=clusters,
            validation_setup_cache=setup,
        )
        with contextlib.closing(active_loader):
            metrics = evaluate_encdec_bpb_conditioned(
                _ContentSensitiveEvalModel(),
                active_loader,
                eval_steps=2,
                token_bytes=token_bytes,
                device=t.device("cpu"),
            )
        return vars(metrics)

    baseline_metrics = evaluate(None)
    dist.barrier()
    retained_metrics = [evaluate(setup_cache) for _ in range(2)]
    dist.barrier()
    return {
        "baseline_digests": baseline_digests,
        "baseline_rows": baseline_rows,
        "gate_digests": gate_digests,
        "gate_rows": gate_rows,
        "baseline_metrics": baseline_metrics,
        "retained_metrics": retained_metrics,
        "loader_health": dataloader_module.loader_health_snapshot(),
        "setup_published": setup_cache._setup is not None,
    }


def test_real_two_process_gloo_rewinds_each_rank(
    tmp_path: Path,
    gloo_multiprocess,
) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    results = gloo_multiprocess(
        2,
        _gloo_rewind_target,
        str(fixture.packed_dir),
        str(fixture.selection_dir),
        fixture.clusters,
        fixture.pipeline,
    )
    for result in results:
        assert result["gate_digests"] == [result["baseline_digests"]] * 3
        assert result["gate_rows"] == [result["baseline_rows"]] * 3
        assert result["retained_metrics"] == [result["baseline_metrics"]] * 2
        assert result["gate_rows"][0] == result["gate_rows"][1]
        assert result["gate_rows"][1] == result["gate_rows"][2]
        assert result["setup_published"] is True
    rank_rows = [
        {tuple(row) for batch_rows in result["baseline_rows"] for row in batch_rows}
        for result in results
    ]
    assert rank_rows[0].isdisjoint(rank_rows[1])
    assert rank_rows[0] | rank_rows[1] == {
        (1, f"shard_{shard_index:05d}.safetensors.zst", row_index)
        for shard_index in range(4)
        for row_index in range(2)
    }
    assert (
        results[0]["baseline_metrics"]["bpb_conditioned"]
        == results[1]["baseline_metrics"]["bpb_conditioned"]
    )
    assert sum(result["loader_health"]["owner_decodes"] for result in results) > 0
    assert sum(result["loader_health"]["shared_attaches"] for result in results) > 0
    assert sum(result["loader_health"]["fallbacks"] for result in results) == 0


def _nccl_cuda_loader_target(
    rank: int,
    world_size: int,
    packed_dir: str,
    selection_dir: str,
    clusters: Any,
    pipeline: Any,
) -> dict[str, Any]:
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    device = t.device("cuda", rank)
    dataloader_module._loader_health_reset()
    setup_cache = PrecomputedValidationSetupCache()

    def loader(setup: PrecomputedValidationSetupCache | None):
        return precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=2,
            feature_pipeline=pipeline,
            device=str(device),
            selection_dir=selection_dir,
            effective_buckets=pipeline.effective_buckets,
            ddp_rank=rank,
            ddp_world_size=world_size,
            feature_clusters=clusters,
            validation_setup_cache=setup,
        )

    def batch_digests(setup: PrecomputedValidationSetupCache | None) -> list[str]:
        active_loader = loader(setup)
        with contextlib.closing(active_loader):
            return [_batch_digest(next(active_loader)) for _ in range(2)]

    baseline_digests = batch_digests(None)
    dist.barrier()
    cold_digests = batch_digests(setup_cache)
    dist.barrier()
    warm_digests = batch_digests(setup_cache)
    dist.barrier()

    token_bytes = t.arange(32_768, dtype=t.int64, device=device).remainder(7).add(1)

    def metrics(setup: PrecomputedValidationSetupCache | None) -> dict[str, Any]:
        active_loader = loader(setup)
        with contextlib.closing(active_loader):
            result = evaluate_encdec_bpb_conditioned(
                _ContentSensitiveEvalModel(),
                active_loader,
                eval_steps=2,
                token_bytes=token_bytes,
                device=device,
            )
        return vars(result)

    baseline_metrics = metrics(None)
    dist.barrier()
    warm_metrics = metrics(setup_cache)
    dist.barrier()
    return {
        "baseline_digests": baseline_digests,
        "cold_digests": cold_digests,
        "warm_digests": warm_digests,
        "baseline_metrics": baseline_metrics,
        "warm_metrics": warm_metrics,
        "loader_health": dataloader_module.loader_health_snapshot(),
        "setup_published": setup_cache._setup is not None,
    }


@pytest.mark.slow
def test_real_two_gpu_nccl_cuda_loader_and_evaluator_reuse(tmp_path: Path) -> None:
    """Exercise real packed files, CUDA transfers, shared decode, and NCCL BPB."""
    fixture = _write_generic_fixture(tmp_path, n_shards=4)
    results = run_nccl_multiprocess(
        2,
        _nccl_cuda_loader_target,
        str(fixture.packed_dir),
        str(fixture.selection_dir),
        fixture.clusters,
        fixture.pipeline,
    )
    for result in results:
        assert result["cold_digests"] == result["baseline_digests"]
        assert result["warm_digests"] == result["baseline_digests"]
        assert result["warm_metrics"] == result["baseline_metrics"]
        assert result["setup_published"] is True
    assert (
        results[0]["baseline_metrics"]["bpb_conditioned"]
        == results[1]["baseline_metrics"]["bpb_conditioned"]
    )
    assert sum(result["loader_health"]["owner_decodes"] for result in results) > 0
    assert sum(result["loader_health"]["shared_attaches"] for result in results) > 0
    assert sum(result["loader_health"]["fallbacks"] for result in results) == 0
