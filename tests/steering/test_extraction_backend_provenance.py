"""Backend provenance survives extraction/resume and controls validation (#1637)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import zstandard
from loguru import logger
from omegaconf import OmegaConf
from safetensors.torch import save as safetensors_save

from steering import extract_features as ef
from steering import validate_features as vf
from steering.feature_io import (
    ExtractionBackend,
    ExtractionBackendProvenance,
    FeatureDirectoryMismatchError,
    RegularFileIdentity,
    assert_extraction_backend_compatible,
    extraction_backend_from_metadata,
    glob_features,
    load_features,
    load_features_stable,
    read_extraction_backend_provenance,
    read_shard_metadata,
    save_features,
    save_features_presorted,
)
from steering.refined_types import NonNegativeInt
from tests.steering.test_extract_features_loop import _staging, tiny_params
from tests.steering.test_feature_io import _make_shard_dict


@pytest.fixture
def logs():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    yield messages
    logger.remove(sink)


def _shard(backend=None):
    data = _make_shard_dict(n_segments=2, n_docs=1, feats_per_seg=1)
    data["feature_ids"] = t.tensor([1, 2], dtype=t.int32)
    if backend is not None:
        data["extraction_backend"] = backend
    return data


@pytest.mark.parametrize("backend", [None, *ExtractionBackend])
@pytest.mark.parametrize("writer", [save_features, save_features_presorted])
def test_real_io_roundtrip_preserves_provenance_and_tensors(tmp_path, backend, writer):
    data = _shard(backend)
    path = writer(data, tmp_path / "shard_00000_features")
    legacy = dict(data)
    legacy.pop("extraction_backend", None)
    reference = load_features(writer(legacy, tmp_path / "reference"))
    for loaded in (load_features(path), load_features_stable(
        path, expected_identity=RegularFileIdentity.capture(path),
    )):
        assert loaded.get("extraction_backend") == backend
        for key, value in reference.items():
            if isinstance(value, t.Tensor):
                assert t.equal(value, loaded[key]), key
            else:
                assert loaded[key] == value
    assert load_features(path, keys=set()).get("extraction_backend") == backend
    assert read_shard_metadata(path).get("extraction_backend") == backend
    if backend is None:
        assert "extraction_backend" not in load_features(path)


@pytest.mark.parametrize("bad", ["auto", "", "None", "NNTERP", "other", None, 1, True])
def test_invalid_provenance_cannot_be_written(tmp_path, bad):
    data = _shard()
    data["extraction_backend"] = bad
    with pytest.raises(ValueError, match="Invalid extraction_backend"):
        save_features(data, tmp_path / "shard_00000_features")
    assert not glob_features(tmp_path)


def _raw_invalid_shard(path, backend):
    payload = safetensors_save({}, metadata={"extraction_backend": backend})
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))


@pytest.mark.parametrize("bad", ["auto", "", "None", "NNTERP", "other"])
def test_invalid_header_fails_cpu_validation_and_load(tmp_path, bad):
    path = tmp_path / "shard_00000_features.safetensors.zst"
    _raw_invalid_shard(path, bad)
    with pytest.raises(ValueError, match="Invalid extraction_backend"):
        load_features(path)
    result = vf.validate_feature_directory(tmp_path)
    assert not result.valid
    assert path.name in result.violations[0]


@pytest.mark.parametrize("backends,valid,unknown", [
    ([None, None], True, 2),
    (["nnterp", "nnterp"], True, 0),
    (["native", "native"], True, 0),
    (["transformerlens", "transformerlens"], True, 0),
    ([None, "nnterp"], True, 1),
    (["nnterp", None], True, 1),
    (["nnterp", "native"], False, 0),
    ([None, "nnterp", "transformerlens"], False, 1),
])
def test_directory_known_uniformity_with_unknown_count(tmp_path, logs, backends, valid, unknown):
    for i, backend in enumerate(backends):
        save_features(_shard(backend), tmp_path / f"shard_{i:05d}_features")
    result = vf.validate_feature_directory(tmp_path)
    assert result.valid is valid, result.violations
    if not valid:
        assert "extraction_backend disagrees" in result.violations[0]
    elif unknown:
        assert any(f"{unknown}/{len(backends)} shards have unknown backend" in m for m in logs)


def test_empty_directory_is_invalid_and_corruption_is_not_unknown(tmp_path):
    assert not vf.validate_feature_directory(tmp_path).valid
    path = tmp_path / "shard_00000_features.safetensors.zst"
    path.write_bytes(b"not zstd")
    result = vf.validate_feature_directory(tmp_path)
    assert not result.valid
    assert path.name in result.violations[0]


@pytest.mark.parametrize("recorded", [None, *ExtractionBackend])
@pytest.mark.parametrize("configured", [None, "auto", "nnterp", "transformerlens", "native"])
def test_selection_matrix(recorded, configured):
    provenance = ExtractionBackendProvenance(recorded, int(recorded is None), 1)
    if configured == "native" or (configured is None and recorded == "native"):
        with pytest.raises(ValueError, match="layer-bound"):
            vf.resolve_spot_check_backend(configured, provenance)
        return
    selection = vf.resolve_spot_check_backend(configured, provenance)
    expected = configured or recorded or "auto"
    assert selection.backend.value == expected
    effective = "nnterp" if expected == "auto" else expected
    mismatch = recorded is not None and recorded != effective
    assert selection.needs_warning == (mismatch or recorded is None)
    assert ("BACKEND MISMATCH" in selection.summary) == mismatch
    assert ("UNVERIFIED" in selection.summary) == (recorded is None)
    assert f"effective={effective}" in selection.summary


def _run_loop(tmp_path, params, paths, output_depth=0, input_depth=0):
    return ef.run_extraction_loop(
        params, paths, rank=0, world_size=1, local_rank=0,
        shard_start=NonNegativeInt.parse(0), shard_end=NonNegativeInt.parse(len(paths)),
        staging_cfg=_staging(tmp_path, output_depth, input_depth),
        profile_enabled=False, profile_output=str(tmp_path / "profile"),
        flops_per_token=1.0, gpu_peak_flops=1.0,
    )


@pytest.mark.parametrize("configured", list(ef.Backend))
@pytest.mark.parametrize("output_depth", [0, 1])
def test_builder_to_real_extraction_and_save(tmp_path, monkeypatch, configured, output_depth):
    params = tiny_params(tmp_path / "out")
    nn_model, native = object(), object()
    monkeypatch.setattr(ef, "build_nnterp_model", lambda *a, **kw: nn_model)
    monkeypatch.setattr(ef, "build_native_model", lambda *a, **kw: native)
    monkeypatch.setattr(params.model, "to", lambda *_: None, raising=False)
    monkeypatch.setattr(ef.t.cuda, "empty_cache", lambda: None)
    nn_built, native_built = ef._build_accelerated_backend(
        configured, params.model, "cuda", t.float32, SimpleNamespace(model="gemma-test", layer=1),
    )
    calls = []

    def capture(backend, tokens, *a, **kw):
        calls.append(backend)
        return params.model._embedding[tokens]

    monkeypatch.setattr(ef, "_get_activations_nnterp", capture)
    monkeypatch.setattr(ef, "_get_activations_native", capture)
    params = replace(params, nnterp_model=nn_built, native_model=native_built)
    path = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": ["alpha beta", "gamma delta"]}), path)
    _run_loop(tmp_path, params, [str(path)], output_depth)
    output = glob_features(params.output_dir)[0]
    expected = "nnterp" if configured is ef.Backend.auto else configured.value
    assert read_shard_metadata(output)["extraction_backend"] == expected
    assert load_features(output)["extraction_backend"] == expected
    if expected != "transformerlens":
        assert calls and all(call is (native if expected == "native" else nn_model) for call in calls)


@pytest.mark.parametrize("existing", [None, "transformerlens", "nnterp"])
def test_real_resume_preserves_old_bytes_and_rejects_known_conflict(tmp_path, existing, logs):
    params = tiny_params(tmp_path / "out")
    old_path = save_features(_shard(existing), Path(params.output_dir) / "shard_00000_features")
    before = old_path.read_bytes()
    # The completed shard's parquet deliberately does not exist: resume must skip it.
    paths = [str(tmp_path / "shard_00000.parquet"), str(tmp_path / "shard_00001.parquet")]
    pq.write_table(pa.table({"text": ["alpha beta"]}), paths[1])
    if existing == "nnterp":
        with pytest.raises(FeatureDirectoryMismatchError, match="existing=nnterp, new=transformerlens"):
            _run_loop(tmp_path, params, paths)
        assert len(glob_features(params.output_dir)) == 1
    else:
        results = _run_loop(tmp_path, params, paths)
        assert len(results) == 1
        provenance = read_extraction_backend_provenance(glob_features(params.output_dir))
        assert provenance.backend is ExtractionBackend.transformerlens
        assert provenance.unknown_shards == int(existing is None)
        if existing is None:
            assert any("1/1 shards have unknown backend" in m for m in logs)
    assert old_path.read_bytes() == before


def test_resume_checks_shards_outside_requested_range(tmp_path):
    save_features(_shard("native"), tmp_path / "shard_00999_features")
    with pytest.raises(FeatureDirectoryMismatchError, match="existing=native"):
        assert_extraction_backend_compatible(tmp_path, ExtractionBackend.nnterp)


def test_completed_legacy_resume_is_a_noop(tmp_path):
    params = tiny_params(tmp_path / "out")
    path = save_features(_shard(), Path(params.output_dir) / "shard_00000_features")
    before = path.read_bytes()
    assert _run_loop(tmp_path, params, [str(tmp_path / "shard_00000.parquet")]) == []
    assert path.read_bytes() == before
    assert "extraction_backend" not in read_shard_metadata(path)


def test_ambiguous_execution_objects_are_rejected_before_extraction():
    with pytest.raises(ValueError, match="only one accelerated backend"):
        ef.resolved_extraction_backend(object(), object())


def _install_cli(tmp_path, monkeypatch, backends):
    out = tmp_path / "features"
    data_dir = tmp_path / "parquets"
    data_dir.mkdir()
    for i, backend in enumerate(backends):
        save_features(_shard(backend), out / f"shard_{i:05d}_features")
        pq.write_table(pa.table({"text": ["abcd"]}), data_dir / f"shard_{i:05d}.parquet")
    model = MagicMock(cfg=SimpleNamespace(n_ctx=1024))
    load = MagicMock(return_value=(model, MagicMock(), object()))
    build = MagicMock(return_value=(object(), None))
    activation = MagicMock(return_value=vf.ValidationResult(True, []))
    monkeypatch.setattr(vf, "load_sae_and_model", load)
    monkeypatch.setattr(vf, "_build_accelerated_backend", build)
    monkeypatch.setattr(vf.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vf.t.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(vf, "validate_shard_activations", activation)
    monkeypatch.setattr(vf, "validate_shard_against_source", lambda *a, **kw: vf.ValidationResult(True, []))
    monkeypatch.setattr(vf, "resolve_tokenizer", lambda _: object())
    monkeypatch.setattr(vf, "resolve_sae_max_context", lambda _: 1024)
    monkeypatch.setattr(vf.HydraConfig, "initialized", lambda: False)
    cfg = OmegaConf.create({
        "feature_dir": str(out), "parquet_dir": str(data_dir), "model": "gemma-test",
        "sae_release": "release", "sae_id": "sae", "n_spot_check": 1,
        "model_dtype": "bfloat16", "backend": None,
        "feature_count_rtol": 0.01, "jaccard_threshold": 0.99,
    })
    return cfg, load, build, activation


@pytest.mark.parametrize("recorded,configured,expected", [
    ("transformerlens", None, ef.Backend.transformerlens),
    ("nnterp", None, ef.Backend.nnterp),
    (None, None, ef.Backend.auto),
    ("transformerlens", "auto", ef.Backend.auto),
    ("native", "nnterp", ef.Backend.nnterp),
])
def test_cli_reads_provenance_before_build_and_repeats_warnings(
    tmp_path, monkeypatch, logs, recorded, configured, expected,
):
    cfg, load, build, activation = _install_cli(tmp_path, monkeypatch, [recorded])
    cfg.backend = configured
    if expected is ef.Backend.transformerlens:
        build.return_value = (None, None)
    with pytest.raises(SystemExit) as exc:
        vf.main.__wrapped__(cfg)
    assert exc.value.code == 0
    load.assert_called_once()
    assert build.call_args.args[0] is expected
    activation.assert_called_once()
    summaries = [m for m in logs if "Spot-check backend provenance:" in m]
    assert len(summaries) == 2
    assert summaries[0].split("Spot-check backend provenance:")[1] == summaries[1].split("Spot-check backend provenance:")[1]
    assert activation.call_args.kwargs["jaccard_threshold"] == 0.99


@pytest.mark.parametrize("backends", [["native"], [None, "nnterp"], ["nnterp", "transformerlens"]])
def test_cli_invalid_or_unsupported_selection_fails_before_loading(tmp_path, monkeypatch, backends):
    cfg, load, build, activation = _install_cli(tmp_path, monkeypatch, backends)
    with pytest.raises((ValueError, FeatureDirectoryMismatchError)):
        vf.main.__wrapped__(cfg)
    load.assert_not_called()
    build.assert_not_called()
    activation.assert_not_called()


def test_mixed_provenance_explicit_override_runs_and_unknown_stays_unknown(tmp_path, monkeypatch, logs):
    cfg, _, _, activation = _install_cli(tmp_path, monkeypatch, [None, "nnterp"])
    cfg.backend = "transformerlens"
    monkeypatch.setattr(vf, "_build_accelerated_backend", lambda *a: (None, None))
    before = {path: path.read_bytes() for path in glob_features(cfg.feature_dir)}
    with pytest.raises(SystemExit) as exc:
        vf.main.__wrapped__(cfg)
    assert exc.value.code == 0
    assert activation.call_count == 2
    assert any("BACKEND MISMATCH" in m and "UNVERIFIED" in m for m in logs)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("backends", [[None, "nnterp"], ["native"]])
def test_cpu_only_ignores_backend_and_sae_settings(tmp_path, monkeypatch, backends):
    cfg, load, build, activation = _install_cli(tmp_path, monkeypatch, backends)
    cfg.n_spot_check = 0
    cfg.backend = "unused-invalid-backend"
    with pytest.raises(SystemExit) as exc:
        vf.main.__wrapped__(cfg)
    assert exc.value.code == 0
    load.assert_not_called()
    build.assert_not_called()
    activation.assert_not_called()


def test_cli_malformed_provenance_fails_before_loading(tmp_path, monkeypatch):
    cfg, load, build, activation = _install_cli(tmp_path, monkeypatch, ["nnterp"])
    path = glob_features(cfg.feature_dir)[0]
    _raw_invalid_shard(path, "auto")
    with pytest.raises(ValueError, match="Invalid extraction_backend"):
        vf.main.__wrapped__(cfg)
    load.assert_not_called()
    build.assert_not_called()
    activation.assert_not_called()


def test_cli_backend_failure_never_runs_replay(tmp_path, monkeypatch):
    cfg, _, build, activation = _install_cli(tmp_path, monkeypatch, ["nnterp"])
    build.side_effect = RuntimeError("backend build failed")
    with pytest.raises(RuntimeError, match="backend build failed"):
        vf.main.__wrapped__(cfg)
    activation.assert_not_called()


def test_cli_numerical_failure_stays_failure_under_an_override(tmp_path, monkeypatch, logs):
    cfg, _, _, activation = _install_cli(tmp_path, monkeypatch, ["transformerlens"])
    cfg.backend = "nnterp"
    activation.return_value = vf.ValidationResult(False, ["Jaccard below threshold"])
    with pytest.raises(SystemExit) as exc:
        vf.main.__wrapped__(cfg)
    assert exc.value.code == 1
    assert sum("BACKEND MISMATCH" in message for message in logs) == 2


def test_metadata_parser_distinguishes_absence_from_explicit_null():
    assert extraction_backend_from_metadata({}, source="legacy") is None
    with pytest.raises(ValueError, match="Legacy unknown provenance must omit"):
        extraction_backend_from_metadata({"extraction_backend": None}, source="invalid")
