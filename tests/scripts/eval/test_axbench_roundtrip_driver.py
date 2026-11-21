"""Functional tests for the axbench_upstream roundtrip orchestrator and CLI.

These drive ``execute_roundtrip``/``run_roundtrip.main`` end to end with the
probe seam faked out: no weights are downloaded, no CUDA is touched, and no
network call is made. What is exercised for real is everything the
orchestrator itself owns -- path policy, the read/validate/score/summarize/
manifest composition, and the artifact write order.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch as t

from scripts.eval.axbench_upstream import run_roundtrip
from scripts.eval.axbench_upstream.merge_methods import METHOD_FACTORS
from steering.eval.axbench_upstream import roundtrip
from steering.eval.axbench_upstream.roundtrip import (
    METHODS,
    PROBE_SAE_LAYER,
    PROBE_TOKENIZER_ID,
    ROUNDTRIP_BATCH_SIZE,
    ROUNDTRIP_DATA_FILENAME,
    ROUNDTRIP_MANIFEST_FILENAME,
    ROUNDTRIP_SUMMARY_FILENAME,
    build_summary,
    execute_roundtrip,
    read_merged_frame,
    sae_weights_sha256,
    score_frame,
)
from steering.immutable_io import stream_hash_file_descriptor_bound
from steering.sae_resolver import ResolvedSAE
from tests.scripts.eval.test_axbench_roundtrip import (
    DECLARED_PER_CONCEPT,
    N_CTX,
    N_SAE_FEATURES,
    PROBE_FEATURE,
    _distinct_acts,
    _nl_roundtrip_frame,
    _roundtrip_frame,
    _write_frame,
)

#: The probe geometry, the probed feature column, and the per-concept declared
#: count are the unit suite's -- imported, not re-declared, so the two suites
#: cannot drift into scoring differently shaped fakes. ``WRONG_FEATURE`` is a
#: different, always-zero column used here to falsify the metric.
WRONG_FEATURE = PROBE_FEATURE + 1

#: The ``ResolvedSAE`` a real load returns for the fixed probe identity, and
#: the backend a CPU load reports (pinned in the unit suite; repeated here as
#: the fake loader's return so the manifest assertions below are about the
#: orchestrator recording what the loader gave it).
PROBE_RESOLVED = ResolvedSAE(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_20/width_16k/canonical",
    hook_name="blocks.20.hook_resid_post",
)
PROBE_BACKEND = "transformerlens"

REPO_ROOT = Path(roundtrip.__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _forbid_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this file may reach a real weight download or a real CUDA build.

    Mirrors ``_forbid_external_work`` in the evaluation-launcher suite: the
    real loaders -- and the tokenizer fetch, which also reaches HuggingFace --
    are replaced with failures, so a regression that bypasses the faked
    ``load_roundtrip_probe``/``get_extraction_tokenizer`` seams fails loudly
    here instead of quietly pulling gemma-2-2b from HuggingFace.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the roundtrip driver tests must never load real probe weights")

    monkeypatch.setattr(roundtrip, "load_sae_and_model", forbidden)
    monkeypatch.setattr(roundtrip, "build_nnterp_model", forbidden)
    monkeypatch.setattr(roundtrip, "get_extraction_tokenizer", forbidden)


class _FakeProbeSae(t.nn.Module):
    """Stand-in for the loaded SAELens SAE: the two things the run reads off it.

    ``cfg.d_sae`` is where ``n_sae_features`` now comes from (through
    ``_infer_n_sae_features``, the one helper the dense pool also uses), and it
    is deliberately a NUMPY scalar. ``_infer_n_sae_features``
    (``steering/eval/axbench_base/extraction.py:301``) already returns
    ``int(n_features)``, so this numpy seeding alone does NOT catch an
    orchestrator that dropped its own (redundant) ``int()`` wrapper around
    that call -- see ``test_manifest_records_plain_python_scalars`` for why
    that conjunct is a pin rather than a live discriminator. What the numpy
    seeding DOES catch: ``W_enc`` is deliberately NOT square, and disagrees
    with ``cfg.d_sae`` on its second dimension, so a reader that went back to
    ``W_enc.shape[1]`` -- which would return a Python ``int`` for free,
    masking a broken derivation -- is caught rather than silently agreeing;
    ``state_dict()`` is what ``sae_weights_sha256`` fingerprints for the
    manifest.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cfg = types.SimpleNamespace(d_sae=np.int64(N_SAE_FEATURES))
        self.W_enc = t.nn.Parameter(
            t.zeros((4, N_SAE_FEATURES + 1), dtype=t.float32), requires_grad=False
        )


class _FakeTokenizer:
    """Stand-in for the extraction tokenizer, counting whitespace words.

    ``add_special_tokens`` is keyword-only and REQUIRED, so an orchestrator
    that left it at the tokenizer's default (which prepends BOS) fails here
    rather than silently counting one token too many against ``n_ctx``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return [len(word) for word in text.split()]


def _count_words(text: str) -> int:
    """The token count ``_FakeTokenizer`` produces, for the oracle scorer."""
    return len(text.split())


def _activation_tensor(texts: list[str], acts: dict[str, float], *, feature_index: int) -> t.Tensor:
    """The dense max-pool matrix: ``acts[text]`` in one column, zeros elsewhere."""
    out = t.zeros((len(texts), N_SAE_FEATURES), dtype=t.float32)
    for row, text in enumerate(texts):
        out[row, feature_index] = acts[text]
    return out


class _StubPool:
    """Stand-in for ``_batched_dense_sae_max_pool``, recording every call.

    Takes the production primitive's exact signature (texts and the model/SAE
    positionally, the rest keyword-only), so an orchestrator that wired the
    arguments differently fails here.
    """

    def __init__(self, acts: dict[str, float], *, feature_index: int = PROBE_FEATURE) -> None:
        self.acts = acts
        self.feature_index = feature_index
        self.calls: list[dict[str, Any]] = []
        self.batches: list[tuple[list[str], t.Tensor]] = []

    def __call__(
        self,
        texts: list[str],
        base_model: object,
        base_sae: object,
        *,
        sae_layer: int,
        tokenizer_id: str,
        batch_size: int,
    ) -> t.Tensor:
        out = _activation_tensor(texts, self.acts, feature_index=self.feature_index)
        self.calls.append(
            {
                "texts": list(texts),
                "base_model": base_model,
                "base_sae": base_sae,
                "sae_layer": sae_layer,
                "tokenizer_id": tokenizer_id,
                "batch_size": batch_size,
            }
        )
        self.batches.append((list(texts), out))
        return out


class _InstalledProbe:
    """The fakes patched into the roundtrip module, plus what they observed."""

    def __init__(self, model: Any, sae: _FakeProbeSae, pool: _StubPool) -> None:
        self.model = model
        self.sae = sae
        self.pool = pool
        self.tokenizer = _FakeTokenizer()
        self.tokenizer_ids: list[str] = []
        self.probe_devices: list[t.device] = []


def _install_probe(
    monkeypatch: pytest.MonkeyPatch,
    acts: dict[str, float],
    *,
    feature_index: int = PROBE_FEATURE,
) -> _InstalledProbe:
    """Replace the probe load, the dense pool, and the tokenizer with fakes.

    ``cfg.n_ctx`` is deliberately a NUMPY scalar: it flows into the manifest,
    and ``atomic_json_dump`` cannot serialize one, so an orchestrator that
    forwarded it unconverted fails every test here at the manifest write
    rather than only in production.
    """
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(n_ctx=np.int64(N_CTX)))
    installed = _InstalledProbe(
        model, _FakeProbeSae(), _StubPool(acts, feature_index=feature_index)
    )

    def fake_load_probe(device: t.device) -> tuple[Any, Any, ResolvedSAE, str]:
        installed.probe_devices.append(device)
        return installed.model, installed.sae, PROBE_RESOLVED, PROBE_BACKEND

    def fake_get_tokenizer(tokenizer_id: str) -> _FakeTokenizer:
        installed.tokenizer_ids.append(tokenizer_id)
        return installed.tokenizer

    monkeypatch.setattr(roundtrip, "load_roundtrip_probe", fake_load_probe)
    monkeypatch.setattr(roundtrip, "_batched_dense_sae_max_pool", installed.pool)
    monkeypatch.setattr(roundtrip, "get_extraction_tokenizer", fake_get_tokenizer)
    return installed


def _expected_artifacts(
    input_path: Path,
    acts: dict[str, float],
    feature_by_concept: dict[int, int],
    *,
    methods: tuple[str, ...] = METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Score the same input directly, as the oracle the run must reproduce.

    Built from the module's own ``score_frame``/``build_summary`` with an
    independent stub, so it pins the orchestrator's COMPOSITION (which frame,
    which feature map, which probe geometry) rather than re-deriving the
    scoring rules that the unit suite already pins.
    """
    oracle = _StubPool(acts)

    def dense_max_pool(texts: list[str]) -> t.Tensor:
        return oracle(
            texts,
            None,
            None,
            sae_layer=PROBE_SAE_LAYER,
            tokenizer_id=PROBE_TOKENIZER_ID,
            batch_size=ROUNDTRIP_BATCH_SIZE,
        )

    data, n_unique = score_frame(
        read_merged_frame(input_path, methods=methods),
        feature_by_concept,
        dense_max_pool_fn=dense_max_pool,
        count_tokens_fn=_count_words,
        n_ctx=N_CTX,
        n_sae_features=N_SAE_FEATURES,
        methods=methods,
    )
    return (
        data,
        build_summary(data, max_concepts=len(feature_by_concept), methods=methods),
        n_unique,
    )


def _write_input(tmp_path: Path, frame: pd.DataFrame, *, name: str = "merged") -> Path:
    return _write_frame(frame, tmp_path / name / "steering_data.parquet")


def _planted_acts(frame: pd.DataFrame, *, method: str, factor: float) -> dict[str, float]:
    """Zero everywhere except this (method, factor)'s generations, which fire at 1.0."""
    acts = dict.fromkeys(_distinct_acts(frame), 0.0)
    planted = frame.loc[frame["factor"].eq(factor), f"{method}_steered_generation"]
    for text in planted:
        acts[str(text)] = 1.0
    return acts


def _rates(summary: pd.DataFrame) -> dict[tuple[str, float], float]:
    return {
        (str(method), float(factor)): float(rate)
        for method, factor, rate in zip(
            summary["method"], summary["factor"], summary["activation_rate"], strict=True
        )
    }


def test_main_writes_data_summary_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end on a 2-concept frame: exit 0, three artifacts, oracle-equal."""
    frame = _roundtrip_frame(max_concepts=2, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    acts = _distinct_acts(frame)
    probe = _install_probe(monkeypatch, acts)
    expected_data, expected_summary, expected_unique = _expected_artifacts(
        input_path, acts, {0: PROBE_FEATURE, 1: PROBE_FEATURE}
    )

    exit_code = run_roundtrip.main(
        ["--input", str(input_path), "--output-dir", str(output_dir), "--device", "cpu"]
    )

    assert exit_code == 0
    data_path = output_dir / ROUNDTRIP_DATA_FILENAME
    summary_path = output_dir / ROUNDTRIP_SUMMARY_FILENAME
    manifest_path = output_dir / ROUNDTRIP_MANIFEST_FILENAME
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        (ROUNDTRIP_DATA_FILENAME, ROUNDTRIP_MANIFEST_FILENAME, ROUNDTRIP_SUMMARY_FILENAME)
    )
    pd.testing.assert_frame_equal(pd.read_parquet(data_path), expected_data)
    pd.testing.assert_frame_equal(pd.read_parquet(summary_path), expected_summary)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["input"]["path"] == str(input_path.resolve())
    assert manifest["input"]["sha256"] == stream_hash_file_descriptor_bound(input_path).sha256
    assert manifest["input"]["n_rows"] == len(frame)
    assert manifest["input"]["max_concepts"] == 2
    assert manifest["probe"]["device"] == "cpu"
    assert manifest["probe"]["backend"] == PROBE_BACKEND
    assert manifest["probe"]["release"] == PROBE_RESOLVED.release
    assert manifest["probe"]["n_ctx"] == N_CTX
    assert manifest["probe"]["n_sae_features"] == N_SAE_FEATURES
    assert manifest["probe"]["sae_weights_sha256"] == sae_weights_sha256(probe.sae)
    assert expected_unique == 2 * DECLARED_PER_CONCEPT
    assert manifest["accounting"] == {
        "per_method": {
            "DiffMean": {"n_declared": 240, "n_blank": 0, "n_scored": 240},
            "LinearProbe": {"n_declared": 240, "n_blank": 0, "n_scored": 240},
            "PromptSteering": {"n_declared": 20, "n_blank": 0, "n_scored": 20},
            "LoRA": {"n_declared": 20, "n_blank": 0, "n_scored": 20},
            "EncDecDiffMean": {"n_declared": 240, "n_blank": 0, "n_scored": 240},
            "EncDecLinearProbe": {"n_declared": 240, "n_blank": 0, "n_scored": 240},
            "EncDecPromptSteering": {"n_declared": 20, "n_blank": 0, "n_scored": 20},
            "EncDecLoRA": {"n_declared": 20, "n_blank": 0, "n_scored": 20},
            "SelfConditioning": {"n_declared": 240, "n_blank": 0, "n_scored": 240},
            "SelfConditioningMasked": {"n_declared": 240, "n_blank": 0, "n_scored": 240},
        },
        "n_unique_texts_per_concept_sum": 2 * DECLARED_PER_CONCEPT,
    }


def test_explicit_nl_cli_writes_the_exact_seven_method_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public flag threads one method tuple through read, score, summary, and manifest."""
    frame = _nl_roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    acts = _distinct_acts(frame, methods=roundtrip.NL_METHODS)
    _install_probe(monkeypatch, acts)
    expected_data, expected_summary, expected_unique = _expected_artifacts(
        input_path,
        acts,
        {0: PROBE_FEATURE},
        methods=roundtrip.NL_METHODS,
    )

    exit_code = run_roundtrip.main([
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--include-nl-steering",
    ])

    assert exit_code == 0
    data = pd.read_parquet(output_dir / ROUNDTRIP_DATA_FILENAME)
    summary = pd.read_parquet(output_dir / ROUNDTRIP_SUMMARY_FILENAME)
    manifest = json.loads((output_dir / ROUNDTRIP_MANIFEST_FILENAME).read_text())
    pd.testing.assert_frame_equal(data, expected_data, check_exact=True)
    pd.testing.assert_frame_equal(summary, expected_summary, check_exact=True)
    assert data.shape == (len(frame), 26)
    assert len(summary) == 62
    assert manifest["scoring"]["methods"] == list(roundtrip.NL_METHODS)
    assert list(manifest["accounting"]["per_method"]) == list(roundtrip.NL_METHODS)
    assert manifest["accounting"]["n_unique_texts_per_concept_sum"] == expected_unique


def test_manifest_records_plain_python_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``n_ctx``/``n_sae_features``/``max_concepts`` must be ``int``, not numpy scalars.

    Asserted on the manifest object HANDED TO ``atomic_json_dump``, not on the
    JSON read back: a round-trip turns every integer into a Python ``int``, so
    reading the file could only ever confirm what JSON guarantees.

    Every source the three values are derived from is seeded with a numpy
    scalar, but only one of the three conjuncts is a live discriminator today:

    * ``n_ctx`` <- the fake probe's ``model.cfg.n_ctx``, cast with a bare
      ``int(model.cfg.n_ctx)`` (``roundtrip.py:960``) and nothing upstream of
      that call coerces it first. This is the one live conjunct: drop that
      ``int()`` and the numpy scalar reaches the manifest dict unchanged.
    * ``n_sae_features`` <- the fake SAE's ``cfg.d_sae``, read through
      ``_infer_n_sae_features`` (``steering/eval/axbench_base/extraction.py:301``),
      which already ``return``s ``int(n_features)``. The value is therefore a
      plain Python ``int`` by the time it reaches the orchestrator regardless
      of whether the orchestrator's own ``int(...)`` wrapper
      (``roundtrip.py:959``) is present, so this conjunct CANNOT fail today --
      it is a pin against a future re-derivation that reads a numpy-backed
      ``cfg.d_sae``/``cfg.n_features`` directly and skips the helper, not a
      live discriminator.
    * ``max_concepts`` <- ``len()`` of the feature map, seeded here with numpy
      keys and values. This one conjunct CANNOT fail today either -- ``len``
      is typed to return ``int`` whatever it counts -- so it is a pin against
      a future re-derivation that reads the count off the frame (e.g. a
      numpy-returning ``nunique``) rather than a live discriminator.

    Stated rather than implied, so nobody reads the ``n_sae_features``/
    ``max_concepts`` conjuncts as evidence they are not pins. The single live
    conjunct is non-vacuous: dropping the ``n_ctx`` ``int()`` leaves an
    unserializable value in the dict, and the final assertion (the file
    parses back equal) is what would then fail.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    probe = _install_probe(monkeypatch, _distinct_acts(frame))
    assert type(probe.model.cfg.n_ctx) is not int
    assert type(probe.sae.cfg.d_sae) is not int

    real_validate = roundtrip.validate_probe_identity

    def numpy_feature_map(scored_frame: pd.DataFrame) -> dict[Any, Any]:
        return {
            np.int64(concept_id): np.int64(feature)
            for concept_id, feature in real_validate(scored_frame).items()
        }

    monkeypatch.setattr(roundtrip, "validate_probe_identity", numpy_feature_map)
    real_json_dump = roundtrip.atomic_json_dump
    written: list[dict[str, Any]] = []

    def recording_json_dump(obj: dict[str, Any], final_path: Path) -> None:
        written.append(obj)
        real_json_dump(obj, final_path)

    monkeypatch.setattr(roundtrip, "atomic_json_dump", recording_json_dump)

    execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")

    (manifest,) = written
    for value in (
        manifest["probe"]["n_ctx"],
        manifest["probe"]["n_sae_features"],
        manifest["input"]["max_concepts"],
    ):
        assert type(value) is int
    assert manifest["probe"]["n_sae_features"] == N_SAE_FEATURES
    assert json.loads((output_dir / ROUNDTRIP_MANIFEST_FILENAME).read_text()) == manifest


def test_execute_roundtrip_wires_the_probe_into_the_dense_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One dense call per concept, carrying the loaded probe and the fixed policy."""
    frame = _roundtrip_frame(max_concepts=2, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    input_path = _write_input(tmp_path, frame)
    probe = _install_probe(monkeypatch, _distinct_acts(frame))

    execute_roundtrip(input_path=input_path, output_dir=tmp_path / "roundtrip", device_arg="cpu")

    assert probe.probe_devices == [t.device("cpu")]
    assert probe.tokenizer_ids == [PROBE_TOKENIZER_ID]
    assert len(probe.pool.calls) == 2
    for call in probe.pool.calls:
        assert call["base_model"] is probe.model
        assert call["base_sae"] is probe.sae
        assert call["sae_layer"] == PROBE_SAE_LAYER
        assert call["tokenizer_id"] == PROBE_TOKENIZER_ID
        assert call["batch_size"] == ROUNDTRIP_BATCH_SIZE
    assert probe.tokenizer.calls
    assert all(add_special is False for _text, add_special in probe.tokenizer.calls)


def test_a_missing_package_version_fails_before_the_probe_and_the_scoring_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest's versions are resolved in preflight, not after hours of scoring.

    A distribution missing from the environment makes the manifest -- and so
    the whole run -- impossible, but the manifest is built only after every SAE
    pass. Resolving the versions up front is what turns that into a
    first-second failure, so this asserts the run dies before the probe is even
    loaded, leaving no output directory behind.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    probe = _install_probe(monkeypatch, _distinct_acts(frame))

    def missing_distribution() -> dict[str, str]:
        raise importlib.metadata.PackageNotFoundError("sae-lens")

    monkeypatch.setattr(roundtrip, "resolve_manifest_versions", missing_distribution)

    with pytest.raises(importlib.metadata.PackageNotFoundError, match="sae-lens"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")

    assert probe.probe_devices == []
    assert probe.pool.calls == []
    assert not output_dir.exists()


def test_an_unavailable_cuda_device_fails_before_any_input_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--device cuda`` on a CPU-only box dies in preflight, naming the fix.

    ``resolve_probe_device`` only consults ``cuda.is_available()`` for
    ``"auto"``, so an explicit ``cuda`` reached the loader and failed inside
    torch after the frame had been read, validated and hashed. The device check
    is now the FIRST thing the stage does, which is what the "probe was never
    loaded, nothing was read, no output directory" assertions below pin --
    ``test_device_argument_defaults_to_auto_and_is_resolved_once`` is the
    non-vacuity, since the same fakes complete a whole run on ``cpu``.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    probe = _install_probe(monkeypatch, _distinct_acts(frame))
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"cuda.*is_available\(\) is False") as excinfo:
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cuda")

    assert "--device cpu" in str(excinfo.value)
    assert probe.probe_devices == []
    assert probe.pool.calls == []
    assert probe.tokenizer_ids == []
    assert not output_dir.exists()


def test_an_unsupported_device_type_fails_before_any_input_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the whitelist half of the same preflight: ``mps`` is refused."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    probe = _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(RuntimeError, match="not a supported roundtrip probe device"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="mps")

    assert probe.probe_devices == []
    assert not output_dir.exists()


def test_device_argument_defaults_to_auto_and_is_resolved_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's default reaches ``resolve_probe_device``, whose result is what runs."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    probe = _install_probe(monkeypatch, _distinct_acts(frame))
    requested: list[str] = []

    def fake_resolve(device: str) -> t.device:
        requested.append(device)
        return t.device("cpu")

    monkeypatch.setattr(roundtrip, "resolve_probe_device", fake_resolve)

    assert run_roundtrip.main(["--input", str(input_path), "--output-dir", str(output_dir)]) == 0

    assert requested == ["auto"]
    assert probe.probe_devices == [t.device("cpu")]
    manifest = json.loads((output_dir / ROUNDTRIP_MANIFEST_FILENAME).read_text())
    assert manifest["probe"]["device"] == "cpu"


def test_device_argument_is_passed_through_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the default above: an explicit ``--device`` is not rewritten."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    _install_probe(monkeypatch, _distinct_acts(frame))
    requested: list[str] = []

    def fake_resolve(device: str) -> t.device:
        requested.append(device)
        return t.device("cpu")

    monkeypatch.setattr(roundtrip, "resolve_probe_device", fake_resolve)

    run_roundtrip.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "roundtrip"),
            "--device",
            "cuda:1",
        ]
    )

    assert requested == ["cuda:1"]


def test_two_runs_of_the_same_input_produce_identical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Determinism: same input bytes, same three artifacts, byte-for-byte manifest."""
    frame = _roundtrip_frame(max_concepts=2, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    input_path = _write_input(tmp_path, frame)
    _install_probe(monkeypatch, _distinct_acts(frame))
    first = tmp_path / "first"
    second = tmp_path / "second"

    for output_dir in (first, second):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")

    pd.testing.assert_frame_equal(
        pd.read_parquet(first / ROUNDTRIP_DATA_FILENAME),
        pd.read_parquet(second / ROUNDTRIP_DATA_FILENAME),
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first / ROUNDTRIP_SUMMARY_FILENAME),
        pd.read_parquet(second / ROUNDTRIP_SUMMARY_FILENAME),
    )
    assert json.loads((first / ROUNDTRIP_MANIFEST_FILENAME).read_text()) == json.loads(
        (second / ROUNDTRIP_MANIFEST_FILENAME).read_text()
    )


def test_the_metric_recovers_a_planted_pattern_the_wrong_feature_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falsification: the summary finds a known gap, and a wrong probe does not.

    The stub fires ONLY on the DiffMean factor-1.0 generations, and only in
    SAE column :data:`PROBE_FEATURE`. A frame whose concept names that feature
    must report exactly ``activation_rate == 1.0`` there and ``0.0``
    everywhere else; the very same activations read through a different
    feature index must report ``0.0`` everywhere -- so a summary that reported
    1.0 unconditionally, or 0.0 unconditionally, fails one of the two halves.
    """
    right_frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    wrong_frame = _roundtrip_frame(feature_ids=(WRONG_FEATURE,))
    acts = _planted_acts(right_frame, method="DiffMean", factor=1.0)
    _install_probe(monkeypatch, acts)
    right_out = tmp_path / "right-out"
    wrong_out = tmp_path / "wrong-out"

    execute_roundtrip(
        input_path=_write_input(tmp_path, right_frame, name="right-in"),
        output_dir=right_out,
        device_arg="cpu",
    )
    execute_roundtrip(
        input_path=_write_input(tmp_path, wrong_frame, name="wrong-in"),
        output_dir=wrong_out,
        device_arg="cpu",
    )

    right = _rates(pd.read_parquet(right_out / ROUNDTRIP_SUMMARY_FILENAME))
    wrong = _rates(pd.read_parquet(wrong_out / ROUNDTRIP_SUMMARY_FILENAME))
    assert right[("DiffMean", 1.0)] == 1.0
    assert all(
        rate == 0.0
        for (method, factor), rate in right.items()
        if (method, factor) != ("DiffMean", 1.0)
    )
    assert set(wrong.values()) == {0.0}
    assert set(right) == set(wrong)


def test_sidecar_max_act_equals_the_probe_activation_column_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every declared cell's ``max_act`` is exactly the pool tensor's feature column."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    probe = _install_probe(monkeypatch, _distinct_acts(frame))

    execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")

    data = pd.read_parquet(output_dir / ROUNDTRIP_DATA_FILENAME)
    ((texts, tensor),) = probe.pool.batches
    row_of_text = {text: row for row, text in enumerate(texts)}
    checked = 0
    for method in METHODS:
        declared = frame["factor"].isin(METHOD_FACTORS[method]).tolist()
        generations = frame[f"{method}_steered_generation"].tolist()
        for position, (text, is_declared) in enumerate(zip(generations, declared, strict=True)):
            if not is_declared:
                continue
            expected = float(tensor[row_of_text[str(text)], PROBE_FEATURE])
            assert data[f"{method}_roundtrip_max_act"].iloc[position] == expected
            checked += 1
    assert checked == DECLARED_PER_CONCEPT


def test_rejects_a_nonempty_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    output_dir.mkdir()
    (output_dir / "roundtrip_data.parquet").write_text("stale artifact from an earlier run")
    _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(FileExistsError, match="output directory must be empty"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")


def test_rejects_an_output_path_that_is_not_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    output_dir.write_text("a file where a directory belongs")
    _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(NotADirectoryError, match="output path is not a directory"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")


def test_accepts_an_existing_empty_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the freshness check: "absent OR empty" really allows empty."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    output_dir.mkdir()
    _install_probe(monkeypatch, _distinct_acts(frame))

    execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")

    assert (output_dir / ROUNDTRIP_MANIFEST_FILENAME).is_file()


@pytest.mark.parametrize("relative", [".", "nested", ".."], ids=["equal", "inside", "contains"])
def test_rejects_an_output_dir_overlapping_the_input_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    """Writing the sidecar into (or over) the merged run's own directory is refused.

    ``.`` is the input's own directory, ``nested`` sits inside it, and ``..``
    contains it -- one case per conjunct of the disjointness check. The
    end-to-end test above is the non-vacuity: a sibling directory is accepted.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(ValueError, match="disjoint"):
        execute_roundtrip(
            input_path=input_path,
            output_dir=input_path.parent / relative,
            device_arg="cpu",
        )


def test_rejects_an_input_with_the_wrong_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_frame(frame, tmp_path / "merged" / "merged.parquet")
    output_dir = tmp_path / "roundtrip"
    _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(ValueError, match="must be named steering_data.parquet"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")
    assert not output_dir.exists()


def test_rejects_a_noncanonically_ordered_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _roundtrip_frame(max_concepts=2, feature_ids=(PROBE_FEATURE, PROBE_FEATURE))
    swapped = frame.iloc[[1, 0, *range(2, len(frame))]].reset_index(drop=True)
    input_path = _write_input(tmp_path, swapped)
    output_dir = tmp_path / "roundtrip"
    _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(ValueError, match="canonical.*concept_id, input_id, factor.*order"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")
    assert not output_dir.exists()


def test_rejects_an_already_evaluated_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judged frame carries evaluator columns the merged-frame contract forbids."""
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    evaluated = frame.copy()
    evaluated["DiffMean_LMJudgeEvaluator"] = 1.0
    input_path = _write_input(tmp_path, evaluated)
    output_dir = tmp_path / "roundtrip"
    _install_probe(monkeypatch, _distinct_acts(frame))

    with pytest.raises(ValueError, match="columns do not match the contract") as excinfo:
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")
    assert "DiffMean_LMJudgeEvaluator" in str(excinfo.value)
    assert not output_dir.exists()


def test_a_failed_manifest_write_leaves_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is the completion marker: it is written last, or not at all.

    Both Parquets are already on disk when the manifest write fails, so a
    reader that keys on the manifest's presence correctly sees an incomplete
    run -- which is the whole point of writing it last.
    """
    frame = _roundtrip_frame(feature_ids=(PROBE_FEATURE,))
    input_path = _write_input(tmp_path, frame)
    output_dir = tmp_path / "roundtrip"
    _install_probe(monkeypatch, _distinct_acts(frame))

    def exploding_json_dump(_obj: object, _final_path: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(roundtrip, "atomic_json_dump", exploding_json_dump)

    with pytest.raises(OSError, match="no space left on device"):
        execute_roundtrip(input_path=input_path, output_dir=output_dir, device_arg="cpu")

    assert (output_dir / ROUNDTRIP_DATA_FILENAME).is_file()
    assert (output_dir / ROUNDTRIP_SUMMARY_FILENAME).is_file()
    assert not (output_dir / ROUNDTRIP_MANIFEST_FILENAME).exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--input", "steering_data.parquet"],
        ["--output-dir", "out"],
    ],
    ids=["nothing", "input_only", "output_only"],
)
def test_cli_requires_both_input_and_output_dir(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_roundtrip.main(argv)
    assert excinfo.value.code == 2


def test_public_roundtrip_rejects_nonboolean_nl_selection(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="include_nl_steering must be a boolean"):
        execute_roundtrip(
            input_path=tmp_path / "steering_data.parquet",
            output_dir=tmp_path / "roundtrip",
            include_nl_steering=cast(Any, 1),
        )


def test_cli_help_runs_as_a_module() -> None:
    """``python -m scripts.eval.axbench_upstream.run_roundtrip --help`` exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval.axbench_upstream.run_roundtrip", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_cli_explicitly_propagates_nl_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run_roundtrip, "execute_roundtrip", lambda **kwargs: calls.append(kwargs))

    assert (
        run_roundtrip.main(
            [
                "--input",
                "/tmp/merged/steering_data.parquet",
                "--output-dir",
                "/tmp/roundtrip",
                "--include-nl-steering",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "input_path": Path("/tmp/merged/steering_data.parquet"),
            "output_dir": Path("/tmp/roundtrip"),
            "device_arg": "auto",
            "include_nl_steering": True,
        }
    ]
