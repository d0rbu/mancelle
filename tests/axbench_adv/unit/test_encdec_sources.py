"""Producer-to-consumer artifact tests; neural work and judge responses are synthetic."""

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch as t

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.eval.axbench_adv import encdec_sources
from steering.eval.axbench_adv.encdec_sources import validate_encdec_sources
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_upstream.cell_store import CellStoreError
from steering.eval.axbench_upstream.encdec_baselines import directions, lora
from steering.eval.axbench_upstream.encdec_baselines import run as producer
from tests.scripts.eval.test_axbench_encdec_driver import _install_fakes, _write_inputs
from tests.scripts.eval.test_axbench_evaluation_launcher import (
    _merged_frame,
    _write_evaluation_outputs,
)


@contextmanager
def _restore_files(*paths: Path) -> Iterator[None]:
    snapshots = {path: path.read_bytes() if path.exists() else None for path in paths}
    try:
        yield
    finally:
        for path, before in snapshots.items():
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(before)


def _replace_json(path: Path, keys: tuple[str, ...], value: Any) -> None:
    payload = json.loads(path.read_text())
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(payload))


def _refresh_fit_inventory(root: Path) -> None:
    """Publish a coherent inventory to test semantic fit checks after hashes pass."""
    train_dir = root / "train"
    files = producer._artifact_inventory(
        producer._expected_artifacts(train_dir, max_concepts=50), train_dir=train_dir
    )
    _replace_json(root / "run_manifest.json", ("train_artifacts", "files"), files)
    _replace_json(root / "_encdec_cell_resume_v1/fit_artifacts.json", ("files",), files)


@pytest.fixture(scope="module")
def positive_sources(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("encdec-positive-sources")
    with pytest.MonkeyPatch.context() as patch:
        # Actual stores, atomic installs and safetensors; disk-flush durability is
        # covered separately by test_immutable_io.
        patch.setattr("steering.immutable_io.os.fsync", lambda _fd: None)
        # Review agents may edit the checkout during this synthetic producer run.
        # Keep its declared checkout snapshot fixed for both publications.
        header = producer._manifest_header()
        patch.setattr(producer, "_manifest_header", lambda: header)
        kwargs = _write_inputs(root, max_concepts=50)
        kwargs["max_new_tokens"] = 128
        kwargs["target_meta_path"].write_text('{"step": 1013}')
        (root / "model_000007.pt").rename(root / "model_001013.pt")
        setup = _install_fakes(patch, kwargs)["setup"]
        setup.meta["step"] = 1013
        setup.meta["user_config"]["model_tag"] = encdec_sources.TARGET_MODEL_TAG
        setup.meta["model_config"].update(n_embd=3, sequence_len=256)
        producer.load_target_model(setup, "cpu")[0].dec_config.sequence_len = 256
        fit_diffmean = producer.fit_diffmean_direction

        def fit(*args: Any, **kwargs: Any) -> directions.FittedDirection:
            fitted = fit_diffmean(*args, **kwargs)
            if kwargs["surface"].startswith("concept 49 "):
                return directions.FittedDirection(t.zeros(3), 0.0, fitted.metadata)
            return fitted

        patch.setattr(producer, "fit_diffmean_direction", fit)
        for name in ("save_direction", "load_direction"):
            patch.setattr(producer, name, getattr(directions, name))
        for name in ("save_lora", "load_lora"):
            patch.setattr(producer, name, getattr(lora, name))
        patch.setattr(
            producer,
            "train_encdec_lora",
            lambda *_a, layers, **_k: {
                f"layer_{layer}_attn_c_proj_lora_{side}": t.zeros(shape)
                for layer in layers
                for side, shape in (("A", (lora.LORA_RANK, 3)), ("B", (3, lora.LORA_RANK)))
            },
        )
        patch.setattr(producer, "_generate_encdec_cell", lambda cell, **_k: f"answer {cell}")
        patch.setattr(
            producer,
            "_generate_encdec_direction_cells",
            lambda cells, **_k: tuple(f"answer {cell}" for cell in cells),
        )
        producer.run_encdec_baselines(**kwargs)
        positive_root = kwargs["manifest_path"].parent
        main = pd.read_parquet(kwargs["output_main_path"])
        aux = pd.read_parquet(kwargs["output_aux_path"])
        merged = _merged_frame(max_concepts=50)
        merged.loc[:, list(INPUT_COLUMNS)] = main.loc[:, list(INPUT_COLUMNS)]
        keys = ["concept_id", "input_id", "factor"]
        for frame in (main, aux):
            indexed = frame.set_index(keys).reindex(merged.set_index(keys).index)
            for column in frame.columns:
                if column not in INPUT_COLUMNS:
                    merged[column] = indexed[column].to_numpy()
        merged_path = root / "merged/steering_data.parquet"
        merged_path.parent.mkdir()
        merged.to_parquet(merged_path, index=False)
        judged_dir = root / "judged"
        _write_evaluation_outputs(merged, judged_dir)
        # Pin the fixture's real bytes; only model/tokenizer loading is replaced.
        patch.setattr(
            encdec_sources, "TARGET_META_SHA256", encdec_sources._sha(kwargs["target_meta_path"])
        )
        patch.setattr(
            encdec_sources, "TARGET_MODEL_SHA256", encdec_sources._sha(root / "model_001013.pt")
        )
        patch.setattr(encdec_sources, "load_method_setup", lambda *_a, **_k: setup)
        patch.setattr(
            encdec_sources, "_tokenizer_artifact_identity", producer._tokenizer_artifact_identity
        )
        yield {
            "positive_root": positive_root,
            "source_main_path": kwargs["main_path"],
            "source_aux_path": kwargs["aux_path"],
            "prepared_dir": kwargs["prepared_dir"],
            "target_meta_path": kwargs["target_meta_path"],
            "feature_statistics_path": kwargs["feature_statistics_path"],
            "judged_input_path": merged_path,
            "judged_dir": judged_dir,
        }


def test_completed_producer_is_accepted(positive_sources: dict[str, Path]) -> None:
    sources = validate_encdec_sources(**positive_sources)
    assert set(sources.fitted) == set(sources.lora_states) == set(range(50))
    assert sources.fitted[49]["diffmean"].max_activation == 0.0
    assert len(sources.source_main) == 6000
    pd.testing.assert_frame_equal(
        sources.source_main, pd.read_parquet(positive_sources["judged_input_path"])
    )
    assert set(sources.best_factors.values()) == {0.0, 1.0}


def test_completed_producer_replays_after_relocation(
    positive_sources: dict[str, Path], tmp_path: Path
) -> None:
    original = positive_sources["positive_root"].parent
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    relocated = {name: copied / path.relative_to(original) for name, path in positive_sources.items()}
    before = validate_encdec_sources(**positive_sources)
    after = validate_encdec_sources(**relocated)
    assert after.source_digest == before.source_digest
    assert after.best_factors == before.best_factors
    pd.testing.assert_frame_equal(after.source_main, before.source_main)


def test_selected_factors_come_from_each_encoder_decoder_methods_own_ratings(
    positive_sources: dict[str, Path],
) -> None:
    jsonl = positive_sources["judged_dir"] / "steering.jsonl"
    parquet = positive_sources["judged_dir"] / "steering_data.parquet"
    axes = ("relevance_concept_ratings", "fluency_ratings", "relevance_instruction_ratings")
    preferred = {"EncDecDiffMean": 2.0, "EncDecLinearProbe": 3.0, "DiffMean": 6.0}
    with _restore_files(jsonl, parquet):
        entries = [json.loads(line) for line in jsonl.read_text().splitlines()]
        frame = pd.read_parquet(parquet)
        for method, factor in preferred.items():
            scores = frame["factor"].eq(factor).map({True: 2.0, False: 1.0})
            frame[f"{method}_LMJudgeEvaluator"] = scores
            for axis in axes:
                frame[f"{method}_LMJudgeEvaluator_{axis}"] = scores
                completion = axis.replace("ratings", "completions")
                frame[f"{method}_LMJudgeEvaluator_{completion}"] = scores.map(
                    lambda value: f"Rating: [[{int(value)}]]"
                )
            for entry in entries:
                block = entry["results"]["LMJudgeEvaluator"][method]
                raw = scores.loc[frame["concept_id"].eq(entry["concept_id"])].tolist()
                summary = [2.0 if value == factor else 1.0 for value in block["factor"]]
                block["lm_judge_rating"] = summary
                block["raw_aggregated_ratings"] = raw
                for axis in axes:
                    block[axis] = summary
                    block[f"raw_{axis}"] = raw
                    block[axis.replace("ratings", "completions")] = [
                        f"Rating: [[{int(value)}]]" for value in raw
                    ]
        frame.to_parquet(parquet, index=False)
        jsonl.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        sources = validate_encdec_sources(**positive_sources)
        assert sources.best_factors == {
            "EncDecDiffMean": 2.0,
            "EncDecLinearProbe": 3.0,
            "EncDecPromptSteering": 1.0,
            "EncDecLoRA": 1.0,
        }


@pytest.mark.parametrize("axis", ["relevance_concept_ratings", "fluency_ratings"])
def test_summary_roundoff_cannot_change_raw_rating_tie(
    positive_sources: dict[str, Path], axis: str,
) -> None:
    path = positive_sources["judged_dir"] / "steering.jsonl"
    with _restore_files(path):
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        for entry in entries:
            block = entry["results"]["LMJudgeEvaluator"]["EncDecDiffMean"]
            # All raw grades remain 1. A tolerated summary discrepancy must
            # preserve their tie, whose winner is the first declared factor.
            block[axis][block["factor"].index(6.0)] += 5e-13
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        sources = validate_encdec_sources(**positive_sources)
        assert sources.best_factors["EncDecDiffMean"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("prompt", "PromptSteering text"),
        ("source_row", "positive auxiliary source rows"),
        ("generation", "judged positive EncDecLoRA generations differ from producer"),
        ("strength", "EncDecLoRA strengths"),
        ("fit", "positive fit inventory"),
        ("summary", "summary differs"),
    ],
)
def test_completed_producer_mutations_rejected(
    positive_sources: dict[str, Path],
    mutation: str,
    message: str,
) -> None:
    root = positive_sources["positive_root"]
    if mutation == "fit":
        path = root / "train/concept_00000_diffmean.safetensors"
    elif mutation == "summary":
        path = positive_sources["judged_dir"] / "steering.jsonl"
    else:
        path = root / "aux/inference/steering_data.parquet"
    with _restore_files(path):
        if mutation == "fit":
            path.write_bytes(path.read_bytes() + b"corrupt")
        elif mutation == "summary":
            # The fixture is JSONL; preserve raw ratings and only falsify a mean.
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            entries[0]["results"]["LMJudgeEvaluator"]["EncDecDiffMean"]["fluency_ratings"][0] = 0.5
            path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        else:
            frame = pd.read_parquet(path)
            column, value = {
                "prompt": ("steered_input", "wrong steering prompt"),
                "source_row": ("original_prompt", "wrong original prompt"),
                "generation": ("EncDecLoRA_steered_generation", "changed generation"),
                "strength": ("EncDecLoRA_strength", 2.0),
            }[mutation]
            frame.loc[0, column] = value
            frame.to_parquet(path, index=False)
        with pytest.raises(AxBenchAdvError, match=message):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize(
    ("artifact", "keys", "value", "message"),
    [
        ("run_manifest.json", ("schema_version",), 2, "producer schema"),
        ("_encdec_cell_resume_v1/identity.json", ("schema_version",), 2, "producer schema"),
        (
            "_encdec_cell_resume_v1/identity.json",
            ("semantic_identity",), None, "resume identity",
        ),
        (
            "_encdec_cell_resume_v1/identity.json",
            ("semantic_identity", "driver"), "project", "resume identity",
        ),
        ("run_manifest.json", ("repository_git_commit",), "different", "repository_git_commit"),
        ("run_manifest.json", ("configuration",), None, "configuration or inputs"),
        (
            "_encdec_cell_resume_v1/identity.json",
            ("semantic_identity", "inputs"), None, "configuration or inputs",
        ),
        ("run_manifest.json", ("configuration", "max_concepts"), 49, "completed c50"),
        ("run_manifest.json", ("configuration", "max_concepts"), 50.0, "completed c50"),
        ("run_manifest.json", ("configuration", "seed"), 43, "generation contract"),
        ("run_manifest.json", ("configuration", "max_new_tokens"), 127, "generation contract"),
        *[
            (
                "run_manifest.json", ("configuration", "require_exact_statistics_identity"),
                flag, "statistics.*identity",
            )
            for flag in (False, None, 1, "true")
        ],
        (
            "_encdec_cell_resume_v1/identity.json",
            ("semantic_identity", "configuration", "seed"), 43, "source seed",
        ),
        ("run_manifest.json", ("target", "step"), 1012, "target step"),
        ("run_manifest.json", ("target", "model_tag"), "other-model", "model tag"),
        ("run_manifest.json", ("generation", "top_k"), 49, "generation sampling"),
        ("run_manifest.json", ("layer_resolution",), None, "layer resolution"),
        ("run_manifest.json", ("layer_resolution", "target_n_layer"), 13, "target depth"),
        (
            "run_manifest.json", ("layer_resolution", "resolved_intervention_layer"),
            8, "positive layer",
        ),
        (
            "run_manifest.json", ("layer_resolution", "resolved_lora_layers"),
            [2, 5, 8, 9], "positive LoRA layers",
        ),
        ("run_manifest.json", ("methods",), None, "method manifest coverage"),
        ("run_manifest.json", ("methods",), {}, "method manifest coverage"),
        ("run_manifest.json", ("method_order",), [], "method order"),
        (
            "run_manifest.json", ("methods", "EncDecDiffMean", "factors"),
            [1.0], "factor grid",
        ),
        (
            "run_manifest.json",
            ("methods", "EncDecPromptSteering", "steering_prompt_sha256_by_concept", "0"),
            "0" * 64, "steering prompt",
        ),
        (
            "run_manifest.json",
            ("methods", "EncDecDiffMean", "max_activation_by_concept", "0"),
            "1.0", "calibration",
        ),
        (
            "run_manifest.json", ("methods", "EncDecLoRA", "training_by_concept", "0"),
            None, "LoRA metadata",
        ),
        (
            "_encdec_cell_resume_v1/fit_artifacts.json", ("semantic_fingerprint",),
            "0" * 64, "fit receipt",
        ),
    ],
)
def test_producer_metadata_inconsistency_rejected(
    positive_sources: dict[str, Path],
    artifact: str,
    keys: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    path = positive_sources["positive_root"] / artifact
    with _restore_files(path):
        _replace_json(path, keys, value)
        with pytest.raises(AxBenchAdvError, match=message):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize(
    ("key", "relative", "message"),
    [
        ("source_main_path", None, "main_sha256"),
        ("source_aux_path", None, "aux_sha256"),
        ("target_meta_path", None, "target_meta_sha256"),
        ("feature_statistics_path", None, "feature_statistics_sha256"),
        ("prepared_dir", "prepare_manifest.json", "prepared_manifest_sha256"),
        ("prepared_dir", "generate/metadata.jsonl", "prepared_metadata"),
        ("prepared_dir", "generate/train_data.parquet", "prepared_training_data"),
    ],
)
def test_changed_upstream_artifact_rejected_before_use(
    positive_sources: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    key: str, relative: str | None, message: str,
) -> None:
    monkeypatch.setattr(
        encdec_sources, "load_method_setup",
        lambda *_a, **_k: pytest.fail("changed artifact must fail before method setup"),
    )
    path = positive_sources[key]
    if relative is not None:
        path /= relative
    with _restore_files(path):
        path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(AxBenchAdvError, match=message):
            validate_encdec_sources(**positive_sources)


def test_unbound_pretrain_metadata_rejected(
    positive_sources: dict[str, Path], tmp_path: Path
) -> None:
    metadata = tmp_path / "pretrain.json"
    metadata.write_text("{}")
    with pytest.raises(AxBenchAdvError, match="pretrain metadata"):
        validate_encdec_sources(**positive_sources, pretrain_meta_path=metadata)


def test_missing_model_geometry_rejected(
    positive_sources: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = encdec_sources.load_method_setup(
        positive_sources["target_meta_path"], positive_sources["feature_statistics_path"]
    )
    monkeypatch.setitem(setup.meta, "model_config", None)
    with pytest.raises(AxBenchAdvError, match="model geometry"):
        validate_encdec_sources(**positive_sources)


def test_positive_main_input_rows_must_match_authenticated_source(
    positive_sources: dict[str, Path],
) -> None:
    path = positive_sources["positive_root"] / "main/inference/steering_data.parquet"
    with _restore_files(path):
        frame = pd.read_parquet(path)
        frame.loc[frame["concept_id"].eq(0), "original_prompt"] = "different prompt"
        frame.to_parquet(path, index=False)
        with pytest.raises(AxBenchAdvError, match="positive main source rows"):
            validate_encdec_sources(**positive_sources)


def test_unexpected_fit_file_rejected(positive_sources: dict[str, Path]) -> None:
    path = positive_sources["positive_root"] / "train/unclaimed-fit.safetensors"
    with _restore_files(path):
        path.write_bytes(b"unused")
        with pytest.raises(AxBenchAdvError, match="unexpected files"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("mutation", ["method", "layer", "width", "zero_calibration"])
def test_authenticated_direction_semantics_rejected(
    positive_sources: dict[str, Path], mutation: str
) -> None:
    root = positive_sources["positive_root"]
    path = root / "train/concept_00000_diffmean.safetensors"
    with _restore_files(
        path, root / "run_manifest.json", root / "_encdec_cell_resume_v1/fit_artifacts.json"
    ):
        fitted = directions.load_direction(path)
        metadata = dict(fitted.metadata)
        if mutation == "method":
            metadata["method"] = "probe"
        if mutation == "layer":
            metadata["intervention_layer"] = 8
        directions.save_direction(
            path,
            directions.FittedDirection(
                fitted.direction[:2] if mutation == "width" else fitted.direction,
                0.0 if mutation == "zero_calibration" else fitted.max_activation,
                metadata,
            ),
        )
        _refresh_fit_inventory(root)
        with pytest.raises(AxBenchAdvError, match="positive EncDecDiffMean fit metadata"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("mutation", ["metadata", "layers"])
def test_authenticated_lora_semantics_rejected(
    positive_sources: dict[str, Path], mutation: str
) -> None:
    root = positive_sources["positive_root"]
    path = root / "train/concept_00000_lora.safetensors"
    with _restore_files(
        path, root / "run_manifest.json", root / "_encdec_cell_resume_v1/fit_artifacts.json"
    ):
        state, metadata = lora.load_lora(path)
        if mutation == "metadata":
            metadata["seed"] = "43"
        else:
            # A valid serialized adapter pair, attached to the wrong layer.
            state = {key.replace("layer_2_", "layer_1_"): value for key, value in state.items()}
        lora.save_lora(path, state, metadata)
        _refresh_fit_inventory(root)
        with pytest.raises(AxBenchAdvError, match="positive LoRA (metadata|layer coverage)"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("method", ["EncDecDiffMean", "EncDecLinearProbe"])
def test_positive_strengths_must_replay_saved_calibration(
    positive_sources: dict[str, Path], method: str
) -> None:
    path = positive_sources["positive_root"] / "main/inference/steering_data.parquet"
    with _restore_files(path):
        frame = pd.read_parquet(path)
        frame.loc[0, f"{method}_strength"] += 1.0
        frame.to_parquet(path, index=False)
        with pytest.raises(AxBenchAdvError, match=f"positive {method} strengths"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("mutation", ["missing_batch", "tripwire", "fingerprint"])
def test_durable_producer_cell_store_must_remain_complete(
    positive_sources: dict[str, Path], mutation: str
) -> None:
    resume = positive_sources["positive_root"] / "_encdec_cell_resume_v1"
    if mutation == "missing_batch":
        path = sorted((resume / "rank_00000").glob("batch_*.parquet"))[-1]
    elif mutation == "tripwire":
        path = resume / "tripwire.json"
    else:
        path = resume / "identity.json"
    with _restore_files(path):
        if mutation == "fingerprint":
            _replace_json(path, ("semantic_fingerprint",), "0" * 64)
        else:
            path.unlink()
        with pytest.raises(CellStoreError):
            validate_encdec_sources(**positive_sources)


def test_judged_input_must_preserve_producer_rows(positive_sources: dict[str, Path]) -> None:
    path = positive_sources["judged_input_path"]
    with _restore_files(path):
        frame = pd.read_parquet(path)
        frame.loc[0, "original_prompt"] = "different judged input"
        frame.to_parquet(path, index=False)
        with pytest.raises(AxBenchAdvError, match="judged input base rows"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("method", encdec_sources.ENCDEC_BASELINE_METHODS)
@pytest.mark.parametrize("alter_publication", [False, True])
def test_judged_generations_must_match_durable_producer(
    positive_sources: dict[str, Path], method: str, alter_publication: bool
) -> None:
    paths = [
        positive_sources["judged_input_path"],
        positive_sources["judged_dir"] / "steering_data.parquet",
    ]
    if alter_publication:
        branch = "aux" if method in {"EncDecPromptSteering", "EncDecLoRA"} else "main"
        paths.append(positive_sources["positive_root"] / branch / "inference/steering_data.parquet")
    with _restore_files(*paths):
        for path in paths:
            frame = pd.read_parquet(path)
            frame.loc[frame["factor"].eq(1.0), f"{method}_steered_generation"] = "substituted"
            frame.to_parquet(path, index=False)
        message = (
            "published generation differs from durable cell"
            if alter_publication else f"judged positive {method} generations"
        )
        with pytest.raises(AxBenchAdvError, match=message):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("method", encdec_sources.ENCDEC_BASELINE_METHODS)
@pytest.mark.parametrize(
    "axis", ["relevance_concept_ratings", "fluency_ratings", "relevance_instruction_ratings"]
)
def test_raw_judgment_must_match_evaluated_rows(
    positive_sources: dict[str, Path], method: str, axis: str
) -> None:
    path = positive_sources["judged_dir"] / "steering.jsonl"
    with _restore_files(path):
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        ratings = entries[0]["results"]["LMJudgeEvaluator"][method][f"raw_{axis}"]
        first_declared = next(index for index, value in enumerate(ratings) if value is not None)
        ratings[first_declared] = 2.0
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        with pytest.raises(AxBenchAdvError, match=f"judged {method} raw_{axis}"):
            validate_encdec_sources(**positive_sources)


def test_invalid_raw_ratings_cannot_cancel_to_a_valid_summary(
    positive_sources: dict[str, Path],
) -> None:
    jsonl = positive_sources["judged_dir"] / "steering.jsonl"
    parquet = positive_sources["judged_dir"] / "steering_data.parquet"
    method = "EncDecDiffMean"
    axes = ("relevance_concept_ratings", "fluency_ratings", "relevance_instruction_ratings")
    with _restore_files(jsonl, parquet):
        entries = [json.loads(line) for line in jsonl.read_text().splitlines()]
        frame = pd.read_parquet(parquet)
        concept_rows = frame.loc[frame["concept_id"].eq(0)]
        # Different inputs at the same factor: (-1 + 3) / 2 == (1 + 1) / 2.
        positions = [index for index, factor in enumerate(concept_rows["factor"]) if factor == 1.0]
        block = entries[0]["results"]["LMJudgeEvaluator"][method]
        for position, rating in zip(positions[:2], (-1.0, 3.0), strict=True):
            row = concept_rows.index[position]
            frame.loc[row, f"{method}_LMJudgeEvaluator"] = rating
            block["raw_aggregated_ratings"][position] = rating
            for axis in axes:
                frame.loc[row, f"{method}_LMJudgeEvaluator_{axis}"] = rating
                block[f"raw_{axis}"][position] = rating
                completion = axis.replace("ratings", "completions")
                text = f"Rating: [[{rating}]]"
                frame.loc[row, f"{method}_LMJudgeEvaluator_{completion}"] = text
                block[completion][position] = text
        frame.to_parquet(parquet, index=False)
        jsonl.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        with pytest.raises(AxBenchAdvError, match="rating"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize(
    "axis", ["relevance_concept_ratings", "fluency_ratings", "relevance_instruction_ratings"]
)
def test_consistent_numeric_judgments_must_match_their_completions(
    positive_sources: dict[str, Path], axis: str
) -> None:
    jsonl = positive_sources["judged_dir"] / "steering.jsonl"
    parquet = positive_sources["judged_dir"] / "steering_data.parquet"
    method = "EncDecDiffMean"
    with _restore_files(jsonl, parquet):
        entries = [json.loads(line) for line in jsonl.read_text().splitlines()]
        frame = pd.read_parquet(parquet)
        concept_rows = frame.loc[frame["concept_id"].eq(0)]
        block = entries[0]["results"]["LMJudgeEvaluator"][method]
        factor_index = block["factor"].index(1.0)
        # Change one axis from 1 to 2 at every prompt for factor 1. All numeric
        # copies and means agree; both saved completions still say [[1]].
        block[axis][factor_index] = 2.0
        block["lm_judge_rating"][factor_index] = 1.2
        for position, factor in enumerate(concept_rows["factor"]):
            if factor == 1.0:
                row = concept_rows.index[position]
                frame.loc[row, f"{method}_LMJudgeEvaluator_{axis}"] = 2.0
                frame.loc[row, f"{method}_LMJudgeEvaluator"] = 1.2
                block[f"raw_{axis}"][position] = 2.0
                block["raw_aggregated_ratings"][position] = 1.2
        frame.to_parquet(parquet, index=False)
        jsonl.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        with pytest.raises(AxBenchAdvError, match="judge rating differs from its completion"):
            validate_encdec_sources(**positive_sources)


@pytest.mark.parametrize("axis", ["relevance_concept", "fluency", "relevance_instruction"])
def test_evaluated_completions_must_match_jsonl(
    positive_sources: dict[str, Path], axis: str
) -> None:
    path = positive_sources["judged_dir"] / "steering_data.parquet"
    with _restore_files(path):
        frame = pd.read_parquet(path)
        frame.loc[0, f"EncDecDiffMean_LMJudgeEvaluator_{axis}_completions"] = "Rating: [[2]]"
        frame.to_parquet(path, index=False)
        with pytest.raises(AxBenchAdvError, match="completion"):
            validate_encdec_sources(**positive_sources)
