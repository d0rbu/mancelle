"""Offline contracts for the vendored AxBench baseline pipeline."""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _ROOT / "vendor" / "axbench"
_AXBENCH = _VENDOR_ROOT / "axbench"
_MODEL_UTILS = _AXBENCH / "utils" / "model_utils.py"
_CONSTANTS = _AXBENCH / "utils" / "constants.py"
_TRAIN = _AXBENCH / "scripts" / "train.py"
_INFERENCE = _AXBENCH / "scripts" / "inference.py"
_EVALUATE = _AXBENCH / "scripts" / "evaluate.py"
_DATASET = _AXBENCH / "utils" / "dataset.py"
_MODEL = _AXBENCH / "models" / "model.py"
_PROMPT = _AXBENCH / "models" / "prompt.py"
_REFT = _AXBENCH / "models" / "reft.py"
_HYPERNET_UTILS = _AXBENCH / "models" / "hypernet" / "utils.py"
_PREPARE = _VENDOR_ROOT / "prepare.py"
_RUN_BASELINES = _ROOT / "scripts" / "eval" / "axbench_upstream" / "run_baselines.py"
_CONFIG_DIR = _AXBENCH / "sweep" / "nano_gpt_pretrain_steer"
_MAIN_CONFIG = _CONFIG_DIR / "qwen2_5_3b_main.yaml"
_AUX_CONFIG = _CONFIG_DIR / "qwen2_5_3b_aux.yaml"
_QWEN = "Qwen/Qwen2.5-3B-Instruct"
_LUNA = "openai/gpt-5.6-luna"
_FACTORS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_function(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in _tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{path} has no top-level function {name}")


def _node_source(path: Path, node: ast.AST) -> str:
    source = path.read_text(encoding="utf-8")
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _compile_functions(
    path: Path,
    names: tuple[str, ...],
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile small standalone vendor helpers without importing AxBench."""
    wanted = set(names)
    nodes = [
        node
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    future = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    result: dict[str, Any] = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), result)
    return result


def _literal_assignment(path: Path, name: str) -> Any:
    for node in _tree(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path} has no literal assignment {name}")


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_vendored_environment_declares_stanza_and_current_pyvene_lock() -> None:
    project = tomllib.loads((_VENDOR_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert any(
        dependency.split("=", 1)[0].strip().lower() == "stanza" for dependency in dependencies
    )
    assert "pyvene>=0.1.8" in dependencies

    lock = tomllib.loads((_VENDOR_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}
    assert "stanza" in locked
    assert tuple(map(int, locked["pyvene"].split("."))) >= (0, 1, 8)


def test_qwen_is_chat_model_without_explicit_system_prompt() -> None:
    assert _QWEN in _literal_assignment(_CONSTANTS, "CHAT_MODELS")
    assert _QWEN not in _literal_assignment(_CONSTANTS, "HAS_SYSTEM_PROMPT_MODELS")


def test_chat_template_token_helpers_preserve_non_bos_prefixes_and_zero_suffix() -> None:
    helpers = _compile_functions(
        _MODEL_UTILS,
        ("strip_leading_bos", "strip_chat_template_tokens", "_unwrap_chat_template_tokens"),
    )
    strip_bos = helpers["strip_leading_bos"]
    strip_chat = helpers["strip_chat_template_tokens"]

    with_bos = SimpleNamespace(bos_token_id=1)
    no_bos = SimpleNamespace(bos_token_id=None)
    assert strip_bos([1, 10, 11], with_bos) == [10, 11]
    assert strip_bos([10, 11], with_bos) == [10, 11]
    assert strip_bos([10, 11], no_bos) == [10, 11]
    assert strip_bos([], with_bos) == []
    assert strip_chat([1, 10, 11], with_bos, 0) == [10, 11]
    assert strip_chat([1, 10, 11, 12], with_bos, 2) == [10]
    assert strip_chat([10, 11, 12], with_bos, 1) == [10, 11]


def test_bosless_qwen_opening_marker_and_detected_prefix_stay_aligned() -> None:
    helpers = _compile_functions(
        _MODEL_UTILS,
        ("strip_leading_bos", "get_prefix_length", "_unwrap_chat_template_tokens"),
    )

    class QwenTokenizer:
        bos_token_id = None

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool = False,
        ) -> list[int]:
            assert tokenize
            del add_generation_prompt
            content = messages[-1]["content"]
            return [151644, 872, 198, 16 if content == "1" else 17, 151645]

    tokenizer = QwenTokenizer()
    tokens_a = tokenizer.apply_chat_template([{"role": "user", "content": "1"}], tokenize=True)
    tokens_b = tokenizer.apply_chat_template([{"role": "user", "content": "2"}], tokenize=True)
    retained_a = helpers["strip_leading_bos"](tokens_a, tokenizer)
    retained_b = helpers["strip_leading_bos"](tokens_b, tokenizer)
    actual_common_prefix = next(
        index
        for index, pair in enumerate(zip(retained_a, retained_b, strict=True))
        if pair[0] != pair[1]
    )

    assert retained_a[0] == 151644
    assert helpers["get_prefix_length"](tokenizer) == actual_common_prefix == 3


def test_prefix_and_suffix_length_unwrap_a_batchencoding_chat_template_result() -> None:
    """transformers 5.16.1 can return a dict-like BatchEncoding, not a list."""
    helpers = _compile_functions(
        _MODEL_UTILS,
        ("get_prefix_length", "get_suffix_length", "_unwrap_chat_template_tokens"),
    )

    class BatchEncoding(dict):
        """Minimal stand-in: dict-like, but deliberately NOT list-like."""

    class SmolLM3Tokenizer:
        bos_token_id = None
        chat_template = "{{ messages }}"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool = False,
        ) -> BatchEncoding:
            assert tokenize
            del add_generation_prompt
            content = messages[-1]["content"]
            return BatchEncoding(
                input_ids=[7, 8, 9, 16 if content == "1" else 17, 7],
                attention_mask=[1, 1, 1, 1, 1],
            )

        def decode(self, token_ids: list[int]) -> str:
            return " ".join(str(token_id) for token_id in token_ids)

    tokenizer = SmolLM3Tokenizer()
    assert helpers["get_prefix_length"](tokenizer) == 3
    suffix_length, suffix_text = helpers["get_suffix_length"](tokenizer)
    assert suffix_length == 1


def test_selected_chat_template_call_sites_use_bos_aware_helpers() -> None:
    continues = _node_source(_MODEL_UTILS, _top_level_function(_MODEL_UTILS, "get_model_continues"))
    assert continues.count("strip_leading_bos(") == 2

    train_prepare = _node_source(_TRAIN, _top_level_function(_TRAIN, "prepare_df"))
    assert train_prepare.count("strip_chat_template_tokens(") == 2
    assert train_prepare.count("strip_leading_bos(") == 1

    inference_prepare = _node_source(
        _INFERENCE,
        _top_level_function(_INFERENCE, "prepare_df"),
    )
    assert inference_prepare.count("strip_chat_template_tokens(") == 2

    assert _DATASET.read_text(encoding="utf-8").count("strip_leading_bos(") == 6
    assert "strip_leading_bos(" in _PROMPT.read_text(encoding="utf-8")
    assert "strip_leading_bos(" in _REFT.read_text(encoding="utf-8")
    hypernet_source = _HYPERNET_UTILS.read_text(encoding="utf-8")
    assert hypernet_source.count("strip_leading_bos(") == 2
    assert hypernet_source.count("strip_chat_template_tokens(") == 2


def test_no_chat_template_call_blindly_drops_the_first_token() -> None:
    for path in (
        _TRAIN,
        _INFERENCE,
        _DATASET,
        _MODEL_UTILS,
        _PROMPT,
        _REFT,
        _HYPERNET_UTILS,
    ):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
                continue
            lower = node.slice.lower
            if not isinstance(lower, ast.Constant) or lower.value != 1:
                continue
            assert "apply_chat_template" not in _node_source(path, node), (
                f"blind chat-template BOS strip remains in {path}:{node.lineno}"
            )


def test_completed_chat_templates_only_request_generation_for_user_only_turns() -> None:
    """Completed training turns must not append a fresh assistant marker."""

    def generation_prompt_values(path: Path, function_name: str) -> list[bool]:
        function = _top_level_function(path, function_name)
        template_calls = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply_chat_template"
            ),
            key=lambda node: node.lineno,
        )
        values: list[bool] = []
        for call in template_calls:
            keyword = next(
                (item for item in call.keywords if item.arg == "add_generation_prompt"),
                None,
            )
            value = False if keyword is None else ast.literal_eval(keyword.value)
            assert type(value) is bool
            values.append(value)
        return values

    # The first two calls render completed user+assistant examples for the
    # direction methods. The third renders a user-only LoRA prompt.
    assert generation_prompt_values(_TRAIN, "prepare_df") == [False, False, True]
    # HyperNet likewise has two completed-turn branches followed by two
    # user-only instruction-tuning branches.
    assert generation_prompt_values(_HYPERNET_UTILS, "prepare_df_combined") == [
        False,
        False,
        True,
        True,
    ]
    vendored = (_VENDOR_ROOT / "VENDORED.md").read_text(encoding="utf-8")
    assert "avoid appending a new assistant-generation prompt" in vendored


def test_stanza_pipeline_is_lazy_cached_and_uses_prepared_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _tree(_EVALUATE)
    eager_pipeline_calls = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "stanza"
                and node.func.attr == "Pipeline"
            ):
                eager_pipeline_calls.append(node)
    assert eager_pipeline_calls == []

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    pipeline = object()

    def construct(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return pipeline

    helpers = _compile_functions(
        _EVALUATE,
        ("_get_stanza_pipeline",),
        {
            "cache": functools.cache,
            "lru_cache": functools.lru_cache,
            "os": os,
            "stanza": SimpleNamespace(Pipeline=construct),
            "_stanza_pipeline": None,
        },
    )
    get_pipeline = helpers["_get_stanza_pipeline"]
    monkeypatch.delenv("STANZA_RESOURCES_DIR", raising=False)
    assert get_pipeline() is pipeline
    assert get_pipeline() is pipeline
    assert calls == [(("en",), {"processors": "tokenize,pos", "device": "cpu"})]

    get_pipeline.__globals__["_stanza_pipeline"] = None
    monkeypatch.setenv("STANZA_RESOURCES_DIR", "/prepared/stanza")
    assert get_pipeline() is pipeline
    assert calls[-1] == (
        ("en",),
        {
            "processors": "tokenize,pos",
            "device": "cpu",
            "model_dir": "/prepared/stanza",
            "download_method": None,
        },
    )
    assert calls == [
        (("en",), {"processors": "tokenize,pos", "device": "cpu"}),
        (
            ("en",),
            {
                "processors": "tokenize,pos",
                "device": "cpu",
                "model_dir": "/prepared/stanza",
                "download_method": None,
            },
        ),
    ]

    evaluator = _node_source(
        _EVALUATE,
        _top_level_function(_EVALUATE, "eval_steering_single_task"),
    )
    assert "_get_stanza_pipeline()" in evaluator


def test_prepare_metadata_download_has_a_bounded_timeout(tmp_path: Path) -> None:
    payload = b"pinned metadata\n"
    calls: list[tuple[str, int]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return payload

    def urlopen(url: str, *, timeout: int) -> Response:
        calls.append((url, timeout))
        return Response()

    helpers = _compile_functions(
        _PREPARE,
        ("_sha256", "_download_metadata"),
        {
            "hashlib": hashlib,
            "urllib": SimpleNamespace(request=SimpleNamespace(urlopen=urlopen)),
            "AXBENCH_METADATA_URL": "https://example.invalid/metadata.jsonl",
            "AXBENCH_METADATA_SHA256": hashlib.sha256(payload).hexdigest(),
            "METADATA_DOWNLOAD_TIMEOUT_SECONDS": 60,
        },
    )
    destination = tmp_path / "metadata.jsonl"
    helpers["_download_metadata"](destination)

    assert destination.read_bytes() == payload
    assert calls == [("https://example.invalid/metadata.jsonl", 60)]


def _metadata_row(concept_id: int) -> dict[str, Any]:
    concept = f"concept-{concept_id}"
    return {
        "concept": concept,
        "ref": f"repo/feature/{concept_id}",
        "concept_genres_map": {concept: ["genre"]},
        "concept_id": concept_id,
    }


def test_metadata_loader_enforces_max_concepts_in_source_order(tmp_path: Path) -> None:
    rows = [_metadata_row(concept_id) for concept_id in (7, 3, 9)]
    (tmp_path / "metadata.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    loader = _compile_functions(
        _INFERENCE,
        ("load_metadata_flatten",),
        {"json": json, "Path": Path, "METADATA_FILE": "metadata.jsonl"},
    )["load_metadata_flatten"]

    assert [row["concept_id"] for row in loader(tmp_path)] == [7, 3, 9]
    assert [row["concept_id"] for row in loader(tmp_path, 2)] == [7, 3]
    assert [row["concept_id"] for row in loader(tmp_path, "2")] == [7, 3]
    with pytest.raises(ValueError, match=r"20.*exceeds available metadata rows \(3\)"):
        loader(tmp_path, 20)
    for invalid in (0, -1, "nope"):
        with pytest.raises(ValueError, match="positive integer"):
            loader(tmp_path, invalid)


@pytest.mark.parametrize(
    "function_name",
    ("infer_steering", "infer_latent", "infer_latent_imbalance", "infer_latent_on_train_data"),
)
def test_every_inference_path_passes_max_concepts_to_metadata_loader(
    function_name: str,
) -> None:
    source = _node_source(_INFERENCE, _top_level_function(_INFERENCE, function_name))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_metadata_flatten"
    ]
    assert len(calls) == 1
    call = calls[0]
    values = [*call.args, *(keyword.value for keyword in call.keywords)]
    assert any(
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "args"
        and value.attr == "max_concepts"
        for value in values
    )


def test_provider_client_gates_cover_methods_datasets_and_latent_fallbacks(
    tmp_path: Path,
) -> None:
    required_datasets = _literal_assignment(_INFERENCE, "STEERING_DATASETS_REQUIRING_LM")
    assert required_datasets == {
        "AlpacaEvalSuppress",
        "AttackMultiShot",
        "AttackOverwrite",
        "AttackOverwritePrefix",
    }

    steering = _node_source(_INFERENCE, _top_level_function(_INFERENCE, "infer_steering"))
    assert "STEERING_DATASETS_REQUIRING_LM" in steering
    assert "create_openrouter_client(" in steering

    latent = _node_source(_INFERENCE, _top_level_function(_INFERENCE, "infer_latent"))
    latent_on_train = _node_source(
        _INFERENCE,
        _top_level_function(_INFERENCE, "infer_latent_on_train_data"),
    )
    for source in (latent, latent_on_train):
        assert "has_pregenerated_latent_data(" in source
        assert "create_openrouter_client(" in source
    latent_imbalance = _node_source(
        _INFERENCE,
        _top_level_function(_INFERENCE, "infer_latent_imbalance"),
    )
    for source in (latent, latent_imbalance, latent_on_train):
        assert source.index("has_latent_models(args.models)") < source.index(
            "AutoTokenizer.from_pretrained("
        )
    assert latent.index("has_latent_models(args.models)") < latent.index("create_openrouter_client(")
    assert latent_on_train.index("has_latent_models(args.models)") < latent_on_train.index(
        "create_openrouter_client("
    )

    assert "has_pregenerated_latent_data(" in latent_imbalance
    assert "latent_imbalance requires" in latent_imbalance
    assert "client = None" in latent_imbalance

    has_prepared = _compile_functions(
        _INFERENCE,
        ("has_pregenerated_latent_data",),
        {"Path": Path},
    )["has_pregenerated_latent_data"]
    assert has_prepared(None) is False
    assert has_prepared(tmp_path / "absent") is False
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with pytest.raises(FileNotFoundError, match="latent_eval_data.parquet"):
        has_prepared(prepared)
    (prepared / "latent_eval_data.parquet").touch()
    assert has_prepared(prepared) is True

    has_latent_models = _compile_functions(
        _INFERENCE,
        ("has_latent_models",),
        {"LATENT_EXCLUDE_MODELS": _literal_assignment(_INFERENCE, "LATENT_EXCLUDE_MODELS")},
    )["has_latent_models"]
    assert has_latent_models(["PromptSteering", "LoRA"]) is False
    assert has_latent_models(["PromptSteering", "DiffMean"]) is True


def test_empty_latent_ranks_reach_barrier_and_resume_state_round_trips() -> None:
    latent = _top_level_function(_INFERENCE, "infer_latent")
    no_concepts = next(
        node
        for node in ast.walk(latent)
        if isinstance(node, ast.If)
        and "len(my_concept_ids) == 0" in _node_source(_INFERENCE, node.test)
    )
    assert not any(isinstance(node, ast.Return) for node in ast.walk(no_concepts))

    source = _node_source(_INFERENCE, latent)
    assert source.index("len(my_concept_ids) == 0") < source.index("dist.barrier()")
    assert "save_state(dump_dir, current_state, 'latent', rank)" in source


def test_latent_token_labels_use_the_exact_encoded_activation_span() -> None:
    tokens_for_span = _compile_functions(
        _MODEL,
        ("_tokens_for_activation_span",),
    )["_tokens_for_activation_span"]

    class EncodedIds:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def __getitem__(self, item: slice) -> SimpleNamespace:
            assert isinstance(item, slice)
            return SimpleNamespace(tolist=lambda: self.values[item])

    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda values: [f"token-{value}" for value in values]
    )
    assert tokens_for_span(tokenizer, EncodedIds([151644, 872, 198, 42, 151645, 0]), 3, 2) == [
        "token-42",
        "token-151645",
    ]

    model_source = _MODEL.read_text(encoding="utf-8")
    assert model_source.count("_tokens_for_activation_span(") == 3
    assert 'tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:]' not in model_source


def test_required_max_activations_fail_loudly_and_preserve_batch_order() -> None:
    require = _compile_functions(_MODEL, ("_require_max_activations",))["_require_max_activations"]
    assert require({4: 8.0, 1: 2.5}, [1, 4, 1]) == [2.5, 8.0, 2.5]

    with pytest.raises(ValueError, match=r"0"):
        require({}, [0])
    with pytest.raises(ValueError) as error:
        require({1: 2.5}, [3, 1, 2])
    assert "2" in str(error.value) and "3" in str(error.value)

    model_tree = _tree(_MODEL)
    predict = next(
        node
        for node in ast.walk(model_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "predict_steer"
        and "all_strenghts" in _node_source(_MODEL, node)
    )
    predict_source = _node_source(_MODEL, predict)
    assert "_require_max_activations(" in predict_source
    assert ".get(id, 1.0)" not in predict_source
    assert "mag*max_acts" in predict_source.replace(" ", "")

    precompute = next(
        node
        for node in ast.walk(model_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "pre_compute_mean_activations"
        and "concept_latent" in _node_source(_MODEL, node)
    )
    assert ".max()" in _node_source(_MODEL, precompute)

    steering = _node_source(_INFERENCE, _top_level_function(_INFERENCE, "infer_steering"))
    assert steering.index("latent_data_path.is_file()") < steering.index(
        "AutoTokenizer.from_pretrained("
    )


def test_main_qwen_config_pins_direction_methods_and_full_factor_grid() -> None:
    config = _load_yaml(_MAIN_CONFIG)
    train, generate, inference = (config[name] for name in ("train", "generate", "inference"))

    assert train["model_name"] == inference["model_name"] == _QWEN
    assert train["seed"] == generate["seed"] == inference["seed"] == 42
    assert generate["lm_model"] == inference["lm_model"] == _LUNA
    assert train["layer"] == inference["steering_layer"] == 28
    assert inference["steering_layers"] == [28]
    assert set(train["models"]) == {"DiffMean", "LinearProbe"}
    assert inference["models"] == ["DiffMean", "LinearProbe"]
    assert inference["steering_factors"] == _FACTORS
    assert inference["latent_num_of_examples"] == 36
    assert inference["latent_batch_size"] == 16
    assert train["models"]["DiffMean"] == {
        "batch_size": 6,
        "n_epochs": 1,
        "binarize_dataset": True,
        "low_rank_dimension": 1,
    }
    assert train["models"]["LinearProbe"] == {
        "batch_size": 12,
        "gradient_accumulation_steps": 4,
        "n_epochs": 24,
        "lr": 0.005,
        "weight_decay": 0.001,
        "coeff_l1_loss": 0.0,
        "binarize_dataset": True,
        "low_rank_dimension": 1,
    }


def test_aux_qwen_config_trains_only_lora_and_uses_single_factor() -> None:
    config = _load_yaml(_AUX_CONFIG)
    train, generate, inference = (config[name] for name in ("train", "generate", "inference"))

    assert train["model_name"] == inference["model_name"] == _QWEN
    assert train["seed"] == generate["seed"] == inference["seed"] == 42
    assert generate["lm_model"] == inference["lm_model"] == _LUNA
    assert train["layer"] == inference["steering_layer"] == 28
    assert inference["steering_layers"] == [28]
    assert list(train["models"]) == ["LoRA"]
    assert "PromptSteering" not in train["models"]
    assert set(inference["models"]) == {"PromptSteering", "LoRA"}
    assert inference["steering_factors"] == [1.0]
    assert train["models"]["LoRA"] == {
        "batch_size": 6,
        "gradient_accumulation_steps": 6,
        "n_epochs": 24,
        "lr": 0.0009,
        "weight_decay": 0.0,
        "low_rank_dimension": 4,
        "lora_layers": [7, 14, 21, 28],
        "lora_components": ["o_proj"],
        "lora_alpha": 32,
        "binarize_dataset": False,
        "train_on_negative": False,
        "exclude_bos": True,
    }


def test_launcher_classification_sets_match_vendored_inference_literals() -> None:
    from scripts.eval.axbench_upstream import run_baselines

    assignments = (
        ("STEERING_SKIPPED_MODELS", "STEERING_EXCLUDE_MODELS"),
        ("LATENT_EXCLUDED_MODELS", "LATENT_EXCLUDE_MODELS"),
        ("LM_BACKED_STEERING_DATASETS", "STEERING_DATASETS_REQUIRING_LM"),
    )
    for launcher_name, vendored_name in assignments:
        assert getattr(run_baselines, launcher_name) == _literal_assignment(
            _INFERENCE, vendored_name
        )


def test_launcher_config_normalizes_yaml_to_json_once(tmp_path: Path) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    config = _load_yaml(_MAIN_CONFIG)
    config["inference"]["steering_factor_by_layer"] = {28: 1.0}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = run_baselines._load_config(config_path)
    assert loaded["inference"]["steering_factor_by_layer"] == {"28": 1.0}

    config["train"]["models"] = {123: {}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid model name"):
        run_baselines._load_config(config_path)

    config = _load_yaml(_MAIN_CONFIG)
    config["generate"]["non_finite"] = float("nan")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float values"):
        run_baselines._load_config(config_path)

    config["generate"].pop("non_finite")
    config["generate"]["run_date"] = date(2026, 8, 10)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(TypeError, match="not JSON serializable"):
        run_baselines._load_config(config_path)


def test_launcher_manifest_writes_are_same_directory_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text('{"generation": 1}\n', encoding="utf-8")
    real_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def checked_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent == tmp_path
        replace_calls.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(run_baselines.os, "replace", checked_replace)
    run_baselines._write_json_atomic(manifest_path, {"generation": 2})
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"generation": 2}
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == manifest_path
    assert not list(tmp_path.glob(".run_manifest.json.*.tmp"))

    ensure_source = _node_source(
        _RUN_BASELINES,
        _top_level_function(_RUN_BASELINES, "ensure_run_manifest"),
    )
    assert ensure_source.count("_write_json_atomic(manifest_path,") == 2
    assert "manifest_path.write_text(" not in ensure_source


@pytest.mark.parametrize(
    ("stage", "script_name", "mode"),
    (
        ("train", "train.py", None),
        ("latent", "inference.py", "latent"),
        ("steering", "inference.py", "steering"),
    ),
)
def test_launcher_builds_direct_frozen_vendor_command(
    tmp_path: Path,
    stage: str,
    script_name: str,
    mode: str | None,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    config_path = tmp_path / "config.yaml"
    dump_dir = tmp_path / "run"
    prepared_dir = tmp_path / "prepared"
    command = run_baselines.build_command(
        stage,
        config_path=config_path,
        dump_dir=dump_dir,
        prepared_dir=prepared_dir,
        max_concepts=2,
        nproc_per_node=3,
    )

    assert command[:6] == [
        "uv",
        "run",
        "--project",
        str(_VENDOR_ROOT),
        "--frozen",
        "torchrun",
    ]
    assert "--standalone" in command
    assert "--nproc_per_node=3" in command
    assert str((_AXBENCH / "scripts" / script_name).resolve()) in command
    assert command[command.index("--config") + 1] == str(config_path.resolve())
    assert command[command.index("--dump_dir") + 1] == str(dump_dir.resolve())
    assert command[command.index("--max_concepts") + 1] == "2"
    assert any(value.startswith(str(prepared_dir.resolve())) for value in command)
    assert "-m" not in command
    if mode is None:
        assert "--mode" not in command
    else:
        assert command[command.index("--mode") + 1] == mode


def test_launcher_manifest_records_ordinary_scientific_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    prepared_dir = tmp_path / "prepared"
    for relative_path in run_baselines.REQUIRED_PREPARED_PATHS:
        path = prepared_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (prepared_dir / "generate" / "metadata.jsonl").write_text(
        "".join(f"{json.dumps(_metadata_row(concept_id))}\n" for concept_id in range(4)),
        encoding="utf-8",
    )
    prepared_manifest = {"benchmark": {"revision": "fixture-revision"}}
    (prepared_dir / "prepare_manifest.json").write_text(
        json.dumps(prepared_manifest),
        encoding="utf-8",
    )
    dump_dir = tmp_path / "run"
    monkeypatch.setattr(run_baselines, "_git_commit", lambda: "commit-a")
    monkeypatch.setattr(
        run_baselines,
        "_git_worktree_state",
        lambda: {
            "dirty": True,
            "tracked_changes": [" M vendor/axbench/axbench/models/model.py"],
            "untracked_paths": ["notes.txt"],
        },
    )
    manifest_path = run_baselines.ensure_run_manifest(
        dump_dir,
        stage="train",
        config_path=_MAIN_CONFIG,
        prepared_dir=prepared_dir,
        max_concepts=2,
        nproc_per_node=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_lock_digest = hashlib.sha256((_VENDOR_ROOT / "uv.lock").read_bytes()).hexdigest()

    assert manifest["repository_git_commit"] == "commit-a"
    assert manifest["repository_git_worktree"] == {
        "dirty": True,
        "tracked_changes": [" M vendor/axbench/axbench/models/model.py"],
        "untracked_paths": ["notes.txt"],
    }
    assert manifest["vendored_axbench_upstream_commit"].startswith("41c8332")
    assert manifest["target_model"] == _QWEN
    assert manifest["dataset_revision"] == "fixture-revision"
    assert manifest["seed"] == 42
    assert manifest["max_concepts"] == 2
    assert manifest["nproc_per_node"] == 1
    assert manifest["config"] == _load_yaml(_MAIN_CONFIG)
    assert manifest["prepared_manifest"] == prepared_manifest
    assert manifest["package_lock_sha256"] == expected_lock_digest
    assert manifest["stage_provenance"] == {
        "train": {
            "repository_git_commit": "commit-a",
            "repository_git_worktree": manifest["repository_git_worktree"],
        }
    }

    monkeypatch.setattr(run_baselines, "_git_commit", lambda: "commit-b")
    monkeypatch.setattr(
        run_baselines,
        "_git_worktree_state",
        lambda: {"dirty": False, "tracked_changes": [], "untracked_paths": []},
    )
    assert (
        run_baselines.ensure_run_manifest(
            dump_dir,
            stage="latent",
            config_path=_MAIN_CONFIG,
            prepared_dir=prepared_dir,
            max_concepts=2,
            nproc_per_node=1,
        )
        == manifest_path
    )
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["repository_git_commit"] == "commit-a"
    assert resumed["repository_git_worktree"] == manifest["repository_git_worktree"]
    assert resumed["stage_provenance"] == {
        "train": manifest["stage_provenance"]["train"],
        "latent": {
            "repository_git_commit": "commit-b",
            "repository_git_worktree": {
                "dirty": False,
                "tracked_changes": [],
                "untracked_paths": [],
            },
        },
    }

    before_replay = manifest_path.read_bytes()
    assert (
        run_baselines.ensure_run_manifest(
            dump_dir,
            stage="latent",
            config_path=_MAIN_CONFIG,
            prepared_dir=prepared_dir,
            max_concepts=2,
            nproc_per_node=1,
        )
        == manifest_path
    )
    assert manifest_path.read_bytes() == before_replay

    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy.pop("stage_provenance")
    legacy.pop("repository_git_worktree")
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(run_baselines, "_git_commit", lambda: "commit-c")
    monkeypatch.setattr(
        run_baselines,
        "_git_worktree_state",
        lambda: {"dirty": True, "tracked_changes": [" M inference.py"], "untracked_paths": []},
    )
    run_baselines.ensure_run_manifest(
        dump_dir,
        stage="steering",
        config_path=_MAIN_CONFIG,
        prepared_dir=prepared_dir,
        max_concepts=2,
        nproc_per_node=1,
    )
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["stage_provenance"] == {
        "legacy_creation": {
            "repository_git_commit": "commit-a",
            "repository_git_worktree": None,
        },
        "steering": {
            "repository_git_commit": "commit-c",
            "repository_git_worktree": {
                "dirty": True,
                "tracked_changes": [" M inference.py"],
                "untracked_paths": [],
            },
        },
    }

    with pytest.raises(RuntimeError, match="use a new --dump-dir"):
        run_baselines.ensure_run_manifest(
            dump_dir,
            stage="steering",
            config_path=_MAIN_CONFIG,
            prepared_dir=prepared_dir,
            max_concepts=3,
            nproc_per_node=1,
        )


def test_launcher_rejects_more_concepts_than_prepared_metadata(tmp_path: Path) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    prepared_dir = tmp_path / "prepared"
    metadata_path = prepared_dir / "generate" / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "".join(f"{json.dumps(_metadata_row(concept_id))}\n" for concept_id in range(2)),
        encoding="utf-8",
    )

    run_baselines._validate_requested_concepts(prepared_dir, 2)
    with pytest.raises(ValueError, match=r"--max-concepts=3 exceeds the 2 concepts"):
        run_baselines._validate_requested_concepts(prepared_dir, 3)

    for relative_path in run_baselines.REQUIRED_PREPARED_PATHS:
        path = prepared_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    (prepared_dir / "prepare_manifest.json").write_text("{}\n", encoding="utf-8")
    dump_dir = tmp_path / "run"
    with pytest.raises(ValueError, match=r"--max-concepts=3 exceeds the 2 concepts"):
        run_baselines.ensure_run_manifest(
            dump_dir,
            stage="train",
            config_path=_MAIN_CONFIG,
            prepared_dir=prepared_dir,
            max_concepts=3,
            nproc_per_node=1,
        )
    assert not dump_dir.exists()


def test_launcher_custom_prepare_remediation_names_the_requested_directory(
    tmp_path: Path,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    prepared_dir = tmp_path / "custom-prepared"
    with pytest.raises(FileNotFoundError) as error:
        run_baselines.validate_prepared_dir(prepared_dir)

    message = str(error.value)
    assert "vendor/axbench/prepare.py" in message
    assert "--prepared-dir" in message
    assert str(prepared_dir.resolve()) in message


@pytest.mark.parametrize("models", (None, "PromptSteering"))
def test_launcher_rejects_null_or_scalar_inference_models(
    tmp_path: Path,
    models: object,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    config = _load_yaml(_MAIN_CONFIG)
    config["inference"]["models"] = models
    config_path = tmp_path / "bad-models.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="inference.models.*list of non-empty strings"):
        run_baselines._load_config(config_path)


def test_launcher_provider_preflight_covers_synergy_and_lm_backed_datasets(
    tmp_path: Path,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    config = _load_yaml(_MAIN_CONFIG)
    config["train"]["models"] = {"LsReFT": {"use_synergy": True}}
    config["inference"]["models"] = ["LsReFT"]
    config["inference"]["steering_datasets"] = ["AttackMultiShot"]
    config_path = tmp_path / "provider.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = run_baselines._load_config(config_path)

    assert run_baselines._steering_provider_reasons(loaded, config_path) == [
        "LsReFT with use_synergy=true",
        "LM-backed steering datasets ['AttackMultiShot']",
    ]


def test_launcher_requires_latent_output_only_for_methods_that_consume_max_act(
    tmp_path: Path,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    main = run_baselines._load_config(_MAIN_CONFIG)
    aux = run_baselines._load_config(_AUX_CONFIG)
    dump_dir = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match="run the latent stage"):
        run_baselines._validate_steering_latent_output(main, _MAIN_CONFIG, dump_dir)
    run_baselines._validate_steering_latent_output(aux, _AUX_CONFIG, dump_dir)

    latent_path = dump_dir / "inference" / "latent_data.parquet"
    latent_path.parent.mkdir(parents=True)
    latent_path.touch()
    run_baselines._validate_steering_latent_output(main, _MAIN_CONFIG, dump_dir)


@pytest.mark.parametrize(
    "provider_case",
    ("prompt", "synergy", "lm-dataset"),
)
def test_launcher_provider_failures_happen_before_manifest_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_case: str,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    config = _load_yaml(_MAIN_CONFIG)
    if provider_case == "prompt":
        config["inference"]["models"] = ["PromptSteering"]
    elif provider_case == "synergy":
        config["train"]["models"] = {"LsReFT": {"use_synergy": True}}
        config["inference"]["models"] = ["LsReFT"]
    else:
        config["inference"]["models"] = ["LoRA"]
        config["inference"]["steering_datasets"] = ["AttackOverwrite"]
    config_path = tmp_path / f"{provider_case}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    manifest_called = False
    subprocess_called = False

    def fake_manifest(*args: Any, **kwargs: Any) -> Path:
        nonlocal manifest_called
        del args, kwargs
        manifest_called = True
        return tmp_path / "run" / "run_manifest.json"

    def fake_run(*args: Any, **kwargs: Any) -> None:
        nonlocal subprocess_called
        del args, kwargs
        subprocess_called = True

    monkeypatch.setattr(run_baselines, "ensure_run_manifest", fake_manifest)
    monkeypatch.setattr(run_baselines.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_baselines.main(
            [
                "steering",
                "--config",
                str(config_path),
                "--dump-dir",
                str(tmp_path / "run"),
                "--max-concepts",
                "2",
            ]
        )
    assert manifest_called is False
    assert subprocess_called is False


def test_launcher_latent_preflight_blocks_main_but_not_aux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    dump_dir = tmp_path / "run"
    manifest_calls: list[dict[str, Any]] = []
    subprocess_calls: list[list[str]] = []

    def fake_manifest(*args: Any, **kwargs: Any) -> Path:
        del args
        manifest_calls.append(kwargs)
        return dump_dir / "run_manifest.json"

    def fake_run(command: list[str], **kwargs: Any) -> None:
        del kwargs
        subprocess_calls.append(command)

    monkeypatch.setattr(run_baselines, "ensure_run_manifest", fake_manifest)
    monkeypatch.setattr(run_baselines.subprocess, "run", fake_run)
    with pytest.raises(FileNotFoundError, match="run the latent stage"):
        run_baselines.main(
            [
                "steering",
                "--config",
                str(_MAIN_CONFIG),
                "--dump-dir",
                str(dump_dir),
                "--max-concepts",
                "2",
            ]
        )
    assert manifest_calls == []
    assert subprocess_calls == []

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    assert (
        run_baselines.main(
            [
                "steering",
                "--config",
                str(_AUX_CONFIG),
                "--dump-dir",
                str(dump_dir),
                "--max-concepts",
                "2",
            ]
        )
        == 0
    )
    assert manifest_calls[0]["stage"] == "steering"
    assert len(subprocess_calls) == 1


def test_launcher_main_uses_vendor_cwd_and_copied_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.eval.axbench_upstream import run_baselines

    dump_dir = tmp_path / "run"
    prepared_dir = tmp_path / "prepared"
    manifest_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    subprocess_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_manifest(*args: Any, **kwargs: Any) -> Path:
        manifest_calls.append((args, kwargs))
        return dump_dir / "run_manifest.json"

    def fake_run(command: list[str], **kwargs: Any) -> None:
        subprocess_calls.append((command, kwargs))

    monkeypatch.setattr(run_baselines, "ensure_run_manifest", fake_manifest)
    monkeypatch.setattr(run_baselines.subprocess, "run", fake_run)
    monkeypatch.setenv("UNCHANGED_PARENT_VALUE", "present")

    assert (
        run_baselines.main(
            [
                "train",
                "--config",
                str(_MAIN_CONFIG),
                "--dump-dir",
                str(dump_dir),
                "--prepared-dir",
                str(prepared_dir),
                "--max-concepts",
                "2",
            ]
        )
        == 0
    )
    assert len(manifest_calls) == 1
    assert len(subprocess_calls) == 1
    command, kwargs = subprocess_calls[0]
    assert command[0:2] == ["uv", "run"]
    assert kwargs["cwd"] == _VENDOR_ROOT
    assert kwargs["check"] is True
    assert kwargs["env"]["STANZA_RESOURCES_DIR"] == str(prepared_dir.resolve() / "stanza")
    assert kwargs["env"]["UNCHANGED_PARENT_VALUE"] == "present"


def test_prepare_is_idempotent_and_force_rebuilds_only_prepared_dir(
    tmp_path: Path,
) -> None:
    code = r"""
import json
import sys
from pathlib import Path

import prepare

prepared = Path(sys.argv[1])
calls = []

real_rmtree = prepare.shutil.rmtree
deleted = []
prepare.shutil.rmtree = lambda path: deleted.append(Path(path))
for protected in (
    Path("/"),
    Path("/var"),
    Path.home(),
    prepare.REPO_ROOT,
    prepare.REPO_ROOT / "vendor" / "axbench",
    Path.cwd(),
):
    try:
        prepare.prepare(protected, force=True)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted broad preparation target: {protected}")
assert deleted == []

unmarked = prepared.parent / "unmarked"
unmarked.mkdir(parents=True)
important = unmarked / "important.txt"
important.write_text("keep")
try:
    prepare.prepare(unmarked, force=True)
except ValueError:
    pass
else:
    raise AssertionError("accepted an unmarked custom directory with unrelated contents")
assert important.read_text() == "keep"

unrelated_data_project = prepared.parent / "unrelated-data-project"
(unrelated_data_project / "data").mkdir(parents=True)
(unrelated_data_project / "data" / "important.txt").write_text("keep")
try:
    prepare.prepare(unrelated_data_project, force=True)
except ValueError:
    pass
else:
    raise AssertionError("accepted an unrelated custom directory containing only data/")
assert (unrelated_data_project / "data" / "important.txt").read_text() == "keep"

real_parent = prepared.parent / "real-parent"
real_parent.mkdir()
symlink = prepared.parent / "parent-link"
symlink.symlink_to(real_parent, target_is_directory=True)
assert deleted == []
prepare.shutil.rmtree = real_rmtree

def download_metadata(destination):
    calls.append("metadata")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("{}\n")

def download_file(repo_id, revision, filename, destination):
    del repo_id, revision, filename
    calls.append(destination.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"fixture")

def seed(destination):
    calls.append(destination.name)
    destination.mkdir(parents=True, exist_ok=True)

def seed_summary(data_dir):
    del data_dir
    return {
        "seed_sentences": {"train": 1},
        "seed_instructions": {"train": 1},
    }

def stanza_download(*args, **kwargs):
    del args
    calls.append("stanza")
    Path(kwargs["model_dir"]).mkdir(parents=True, exist_ok=True)

def validate(directory):
    return json.loads((directory / "prepare_manifest.json").read_text())

prepare._download_metadata = download_metadata
prepare._download_file = download_file
prepare._seed_sentences = seed
prepare._seed_instructions = seed
prepare._seed_summary = seed_summary
prepare.stanza.download = stanza_download
prepare._validate = validate

manifest = prepare.prepare(prepared)
first_calls = list(calls)
sentinel = prepared / "only-force-removes-this"
sentinel.write_text("sentinel")
assert prepare.prepare(prepared) == manifest
assert calls == first_calls
assert sentinel.exists()

assert prepare.prepare(prepared, force=True) == manifest
assert calls == first_calls + first_calls
assert not sentinel.exists()
assert prepared.parent.exists()

symlink_prepared = symlink / "prepared"
symlink_manifest = prepare.prepare(symlink_prepared)
assert symlink_manifest == real_parent / "prepared" / "prepare_manifest.json"
assert prepare.prepare(symlink_prepared) == symlink_manifest
assert calls == first_calls + first_calls + first_calls

deleted = []
prepare.shutil.rmtree = lambda path: deleted.append(Path(path))
try:
    prepare.prepare(symlink_prepared, force=True)
except ValueError:
    pass
else:
    raise AssertionError("force accepted a nonempty preparation target beneath a symlink")
assert deleted == []
assert symlink_manifest.is_file()
prepare.shutil.rmtree = real_rmtree

cli_calls = []
def record_prepare(path, *, force=False):
    cli_calls.append((path, force))
    return path / "prepare_manifest.json"

prepare.prepare = record_prepare
custom = prepared.parent / "custom-prepared"
assert prepare.main(["--prepared-dir", str(custom), "--force"]) == 0
assert cli_calls == [(custom, True)]
"""
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(_VENDOR_ROOT),
            "--frozen",
            "python",
            "-c",
            code,
            str(tmp_path / "prepared"),
        ],
        cwd=_VENDOR_ROOT,
        check=True,
    )
