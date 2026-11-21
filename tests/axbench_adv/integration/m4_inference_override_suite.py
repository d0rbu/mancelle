"""Vendored-environment suite for the attacked steering-input override."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from axbench.scripts import inference
from axbench.scripts.args import dataset_args


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: object, *_args: Any, **_kwargs: Any) -> None:
        self.messages.append(str(message))


def _override_frame(num_examples: int, factor: float = 1.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept_id": [0] * num_examples,
            "input_id": list(range(num_examples)),
            "factor": [factor] * num_examples,
            "input": [f"attacked-prompt-{index}" for index in range(num_examples)],
            "steered_input": [f"attacked-steered-{index}" for index in range(num_examples)],
        }
    )


def test_dataset_args_declares_override_default_none() -> None:
    field = dataset_args.DatasetArgs.__dataclass_fields__["overwrite_steering_input_parquet"]
    assert field.default is None


def test_load_override_selects_concept_and_rederives_sae() -> None:
    current, link, sae_id = inference.load_steering_override_df(
        _override_frame(3).iloc[::-1],
        0,
        [{"concept_id": 0, "ref": "https://example/neuronpedia/7"}],
        3,
        [1.5],
    )
    assert list(current["input_id"]) == [0, 1, 2]
    assert (link, sae_id) == ("https://example/neuronpedia/7", 7)
    assert set(current["sae_id"]) == {7}
    with pytest.raises(ValueError, match="integer SAE id"):
        inference.load_steering_override_df(
            _override_frame(1),
            0,
            [{"concept_id": 0, "ref": "not-an-integer"}],
            1,
            [1.5],
        )


@pytest.mark.parametrize(
    ("frame", "concept_id", "examples", "match"),
    [
        (_override_frame(2), 1, 2, "metadata"),
        (_override_frame(3).iloc[1:], 0, 3, "exactly one row"),
        (_override_frame(2).assign(input_id=[0, 0]), 0, 2, "exactly one row"),
        (_override_frame(2).assign(concept_id=[0, 7]), 0, 2, "exactly one row"),
        (_override_frame(2).drop(columns="input"), 0, 2, "missing required column"),
        (_override_frame(2).drop(columns="steered_input"), 0, 2, "missing required column"),
        (_override_frame(2).drop(columns="concept_id"), 0, 2, "missing required column"),
        (_override_frame(2).assign(factor=[1.5, 2.0]), 0, 2, "configured factor"),
        (_override_frame(2).assign(input_id=[0.0, 1.0]), 0, 2, "integer typed"),
    ],
)
def test_load_override_rejects_bad_coverage(
    frame: pd.DataFrame, concept_id: int, examples: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        inference.load_steering_override_df(
            frame,
            concept_id,
            [{"concept_id": 0, "ref": "x/1"}],
            examples,
            [1.5],
        )


@pytest.mark.parametrize(
    ("configured", "match"),
    [([1.0], "configured factor"), ([1.5, 2.0], "exactly one"), ([float("nan")], "finite")],
)
def test_load_override_rejects_bad_configured_factor(configured: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        inference.load_steering_override_df(
            _override_frame(2), 0, [{"concept_id": 0, "ref": "x/1"}], 2, configured
        )


def test_load_override_requires_global_concept_input_coverage() -> None:
    with pytest.raises(ValueError, match="exactly one row"):
        inference.load_steering_override_df(
            _override_frame(2),
            0,
            [
                {"concept_id": 0, "ref": "x/1"},
                {"concept_id": 1, "ref": "x/2"},
            ],
            2,
            [1.5],
        )


def test_build_current_df_preserves_default_and_switches_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    sentinel = pd.DataFrame({"concept_id": [0]})
    captured: list[tuple[Any, ...]] = []

    def fake_create(*args: Any) -> tuple[pd.DataFrame, tuple[int, str, int]]:
        captured.append(args)
        return sentinel, (0, "sae-link", 3)

    monkeypatch.setattr(inference, "create_data_steering", fake_create)
    args = SimpleNamespace(overwrite_steering_input_parquet=None)
    current, link, sae_id = inference.build_current_df_for_concept(
        args, "factory", ["meta"], 0, 10, [1.0], ["AlpacaEval"], "generate"
    )
    assert current is sentinel
    assert (link, sae_id) == ("sae-link", 3)
    assert captured[0][6] is args

    override = tmp_path / "attacked.parquet"
    _override_frame(2).to_parquet(override)
    args.overwrite_steering_input_parquet = str(override)
    current, _, sae_id = inference.build_current_df_for_concept(
        args,
        "factory",
        [{"concept_id": 0, "ref": "x/5"}],
        0,
        2,
        [1.5],
        ["AlpacaEval"],
        "generate",
    )
    assert list(current["input_id"]) == [0, 1]
    assert sae_id == 5
    assert len(captured) == 1


def test_infer_steering_uses_override_for_all_methods_despite_completed_main_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    override = tmp_path / "attacked.parquet"
    _override_frame(2).to_parquet(override)
    dump_dir = tmp_path / "dump"
    main_inference_dir = dump_dir / "inference"
    main_inference_dir.mkdir(parents=True)
    pd.DataFrame().to_parquet(main_inference_dir / "latent_data.parquet")
    inference.save_state(main_inference_dir, {"last_concept_id": 0}, "steering", rank=0)
    output_dir = tmp_path / "attacked-output"
    methods = ("DiffMean", "LinearProbe", "LoRA", "PromptSteering")
    observed: dict[str, pd.DataFrame] = {}

    class _Tokenizer:
        unk_token = "<unk>"
        pad_token = "<pad>"
        padding_side = "left"

    class _Model:
        def eval(self) -> _Model:
            return self

    class _SteeringModel:
        def __init__(self, method: str) -> None:
            self.method = method

        def load(self, **_kwargs: Any) -> None:
            return None

        def to(self, _device: str) -> _SteeringModel:
            return self

        def pre_compute_mean_activations(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def predict_steer(self, frame: pd.DataFrame, **_kwargs: Any) -> dict[str, list[str]]:
            observed[self.method] = frame.copy(deep=True)
            return {"steered_generation": [f"{self.method}-{i}" for i in range(len(frame))]}

    monkeypatch.setattr(inference, "load_config", lambda _path: None)
    monkeypatch.setattr(
        inference,
        "load_metadata_flatten",
        lambda _path, _limit: [{"concept_id": 0, "ref": "x/9"}],
    )
    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(
        inference.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _Model(),
    )
    monkeypatch.setattr(inference, "SteeringDatasetFactory", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(inference, "create_openrouter_client", lambda *_args: object())
    for method in methods:
        monkeypatch.setattr(
            inference.axbench,
            method,
            lambda *_args, _method=method, **_kwargs: _SteeringModel(_method),
        )
    monkeypatch.setattr(inference, "save", lambda *_args: None)
    monkeypatch.setattr(inference.dist, "barrier", lambda: None)
    monkeypatch.setattr(inference, "merge_rank_steering_outputs", lambda *_args: None)
    args = SimpleNamespace(
        data_dir=tmp_path,
        train_dir=tmp_path,
        dump_dir=dump_dir,
        overwrite_inference_dump_dir=output_dir,
        overwrite_steering_input_parquet=override,
        steering_num_of_examples=2,
        steering_layer=0,
        steering_layers=None,
        steering_factors=[1.5],
        steering_datasets=["AlpacaEval"],
        models=list(methods),
        max_concepts=1,
        steering_model_name="fake-model",
        steering_model_revision=None,
        model_name="fake-model",
        model_revision=None,
        master_data_dir=None,
        lm_model=None,
        use_bf16=False,
        disable_neuronpedia_max_act=False,
        steering_batch_size=2,
        steering_output_length=8,
        temperature=0.0,
        steering_intervention_type="addition",
        intervene_on_prompt=True,
    )
    model_args = SimpleNamespace(low_rank_dimension=1, intervention_positions="all")
    inference.infer_steering(
        args,
        rank=0,
        world_size=1,
        device="cpu",
        logger=_Logger(),
        training_args=SimpleNamespace(
            model_name="fake-model",
            models=dict.fromkeys(methods, model_args),
        ),
        generate_args=SimpleNamespace(),
    )
    assert set(observed) == set(methods)
    for method in methods:
        assert observed[method]["input"].tolist() == [
            "attacked-prompt-0",
            "attacked-prompt-1",
        ]
        assert observed[method]["steered_input"].tolist() == [
            "attacked-steered-0",
            "attacked-steered-1",
        ]
        assert observed[method]["factor"].tolist() == [1.5, 1.5]
    assert (output_dir / "steering_inference_state.pkl_rank_0").is_file()


def test_rank_merge_sorts_deletes_and_handles_empty_or_bad_names(tmp_path: Any) -> None:
    merge_dir = tmp_path / "inference"
    merge_dir.mkdir()
    pd.DataFrame(
        {
            "concept_id": [1, 1],
            "input_id": [1, 0],
            "factor": [1.0, 1.0],
        }
    ).to_parquet(merge_dir / "rank_1_steering_data.parquet")
    pd.DataFrame({"concept_id": [0], "input_id": [0], "factor": [1.0]}).to_parquet(
        merge_dir / "rank_0_steering_data.parquet"
    )
    logger = _Logger()
    inference.merge_rank_steering_outputs(merge_dir, logger)
    merged = pd.read_parquet(merge_dir / "steering_data.parquet")
    assert list(zip(merged["concept_id"], merged["input_id"], strict=True)) == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    assert not list(merge_dir.glob("rank_*_steering_data.parquet"))

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "rank_bad_steering_data.parquet").write_text("ignored")
    inference.merge_rank_steering_outputs(empty, logger)
    assert any("does not match" in message for message in logger.messages)
    assert any("No results" in message for message in logger.messages)
