"""Conditioning preflight exactly matches the target's extraction token count."""

import json
from types import SimpleNamespace

import pytest
import tiktoken
from omegaconf import OmegaConf
from transformers import Gemma3TextConfig

from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from steering.eval.axbench_negative import admission, build
from steering.eval.axbench_negative.source import iter_assistant_prefixes
from steering.eval.chat_generation import render_messages_for_chat_completion
from steering.eval.context_enumeration import context_text_for_extraction
from steering.sae_resolver import ResolvedSAE
from steering.segmentation import SegmentMode, truncate_to_training_window
from steering.sft.turn_spans import rendered_text_and_turn_char_starts


class ExtractionTokenizer:
    def __call__(self, text, padding=False):
        assert padding is False
        return {"input_ids": [0, *[ord(char) for char in text]]}


def _feature_prediction(loss_mode="max_pool"):
    return {"weight": 0.1, "layer": -1, "loss_mode": loss_mode,
            "loss_fn": "bce", "block": "mlp", "head_layers": 0}


def _ancestor(directory, step, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"meta_{step:06d}.json").write_text(json.dumps(metadata))
    # The production ancestor walker selects the latest model filename, then
    # reads its metadata. Admission must never deserialize these dummy weights.
    (directory / f"model_{step:06d}.pt").write_bytes(b"metadata-only admission")


@pytest.fixture
def native_tokenizer():
    return RustBPETokenizer(
        tiktoken.Encoding(
            name="negative-admission-byte-test",
            pat_str=r"(?s).",
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
        ),
        "<bos>",
    )


def _setup(mode=SegmentMode.fixed_tokens, window=1000):
    return SimpleNamespace(
        segment_mode=mode,
        prompt_context_window=window,
        prompt_segment_size=32,
        sae_model="gemma-3-1b-pt",
        sae_layer=10,
        n_sae_features=16384,
        resolved_sae=ResolvedSAE("fixture", "fixture", "blocks.10.hook_resid_post"),
    )


@pytest.mark.parametrize(
    "mode", [SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens]
)
def test_conditioning_counts_match_actual_production_helpers(mode, native_tokenizer):
    messages = [
        {"role": "system", "content": " Unicode é "},
        {"role": "user", "content": "question <assistant_start>"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "follow-up"},
    ]
    prompt = render_messages_for_chat_completion(native_tokenizer, messages)
    setup, tokenizer = _setup(mode, 20), ExtractionTokenizer()
    preflight = admission.ConditioningAdmission(setup, tokenizer, 4096, {"context_window": 20})
    text = (
        context_text_for_extraction(native_tokenizer, prompt.token_ids)
        if mode is SegmentMode.fixed_tokens
        else rendered_text_and_turn_char_starts(prompt.token_ids, native_tokenizer)[0]
    )
    kept, expected_count = truncate_to_training_window(text, tokenizer, 20, 4096)
    assert len(kept) == 20 and expected_count > 20
    assert preflight.prompt_token_count(prompt.token_ids, native_tokenizer) == expected_count
    assert expected_count == len(text) + 1  # exact extraction BOS, markers retained


def test_weight_free_loader_binds_policies_and_detects_input_change(tmp_path, monkeypatch):
    meta, stats, pretrain = [
        tmp_path / name for name in ("meta.json", "stats.safetensors", "pretrain.json")
    ]
    for path in (meta, stats, pretrain):
        path.write_bytes(b"pinned fixture")
    meta.write_text(json.dumps({"user_config": {"feature_prediction": _feature_prediction()}}))
    setup, tokenizer = _setup(), ExtractionTokenizer()
    monkeypatch.setattr(admission, "load_method_setup", lambda *_args, **_kwargs: setup)
    monkeypatch.setattr(admission, "resolve_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(
        admission,
        "get_pretrained_model_config",
        lambda _model, *, device: SimpleNamespace(n_ctx=4096),
    )
    monkeypatch.setattr(admission, "tokenizer_sha256", lambda _: "tokenizer-content-hash")
    loaded = admission.load_conditioning_admission(meta, stats, pretrain)
    assert loaded.identity == admission.conditioning_identity(
        setup,
        tokenizer,
        4096,
        statistics_path=stats,
        pretrain_meta_path=pretrain,
    )
    assert loaded.identity["context_window"] == 1000
    assert loaded.identity["segment_mode"] == "fixed_tokens"
    assert loaded.identity["statistics_sha256"] == admission.file_sha256(stats)
    with pytest.raises(ValueError, match="model context"):
        admission.conditioning_identity(
            setup, tokenizer, 999, statistics_path=stats, pretrain_meta_path=None
        )

    def mutate(_):
        stats.write_bytes(b"changed during tokenizer/config resolution")
        return tokenizer

    monkeypatch.setattr(admission, "resolve_tokenizer", mutate)
    with pytest.raises(ValueError, match="changed during admission"):
        admission.load_conditioning_admission(meta, stats, pretrain)


def test_gemma3_admission_uses_actual_loader_context_not_hf_architecture_maximum(
    tmp_path, monkeypatch
):
    meta, stats = tmp_path / "meta.json", tmp_path / "stats.safetensors"
    meta.write_text(json.dumps({"feature_prediction_config": _feature_prediction()}))
    stats.write_bytes(b"statistics fixture")
    setup = _setup(window=2048)
    setup.sae_model = "gemma-3-4b-pt"
    monkeypatch.setattr(admission, "load_method_setup", lambda *_args, **_kwargs: setup)
    monkeypatch.setattr(admission, "resolve_tokenizer", lambda _: ExtractionTokenizer())
    monkeypatch.setattr(admission, "tokenizer_sha256", lambda _: "tokenizer-content-hash")

    # Real config-only factories: neither constructs a model or downloads weights.
    assert Gemma3TextConfig().max_position_embeddings == 131072
    loaded = admission.load_conditioning_admission(meta, stats)
    runtime_config = admission.get_pretrained_model_config(setup.sae_model, device="cpu")
    assert loaded.model_context_window == runtime_config.n_ctx == 8192
    assert loaded.identity["model_context_window"] == runtime_config.n_ctx
    assert loaded.identity["context_window"] == 2048


@pytest.mark.parametrize(
    "feature_prediction,error",
    [
        (
            {"weight": 0.1, "layer": -1, "loss_mode": "max_pool", "loss_fn": "bce"},
            "requires explicit block and head_layers",
        ),
        (
            {
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "last_token",
                "loss_fn": "bce",
                "block": "mlp",
                "head_layers": 0,
            },
            "SelfConditioning requires feature_prediction.loss_mode=max_pool",
        ),
    ],
)
@pytest.mark.parametrize("location", ["local", "ancestor_user", "ancestor_top"])
def test_build_rejects_unsupported_head_before_setup_scorer_or_api(
    tmp_path, monkeypatch, native_tokenizer, feature_prediction, error, location,
):
    target_dir = tmp_path / "sft"
    target_dir.mkdir()
    if location != "local":
        _ancestor(tmp_path, 1, (
            {"user_config": {"feature_prediction": feature_prediction}}
            if location == "ancestor_user" else {"feature_prediction_config": feature_prediction}
        ))
    meta = target_dir / "meta_000749.json"
    meta.write_text(
        json.dumps(
            {
                "step": 749,
                "encoder_config": {},
                "model_config": {"sequence_len": 1024},
                "user_config": {"feature_prediction": feature_prediction} if location == "local" else {},
            }
        )
    )
    model, stats, tokenizer_path = (
        target_dir / name for name in ("model_000749.pt", "stats.safetensors", "tokenizer.pkl")
    )
    for path in (model, stats, tokenizer_path):
        path.write_bytes(b"never loaded")
    contract = {
        "family": "encdec",
        "meta_path": str(meta),
        "tokenizer_path": str(tokenizer_path),
        "max_new_tokens": 4,
        "expected_meta_sha256": admission.file_sha256(meta),
        "expected_model_sha256": admission.file_sha256(model),
        "expected_tokenizer_sha256": admission.file_sha256(tokenizer_path),
        "feature_statistics_path": str(stats),
        "pretrain_meta_path": None,
    }

    def forbidden(*args, **kwargs):
        pytest.fail("invalid head metadata reached setup, scorer or API")

    monkeypatch.setattr(build, "load_source_candidates", lambda *_: [])
    monkeypatch.setattr(build, "load_feature_targets", lambda *_: ([], "fixture"))
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: native_tokenizer)
    monkeypatch.setattr(admission, "load_method_setup", forbidden)
    monkeypatch.setattr(admission, "resolve_tokenizer", forbidden)
    monkeypatch.setattr(build, "build_chat_scorer", forbidden)
    monkeypatch.setattr(build, "create_chat_api", forbidden)
    cfg = OmegaConf.create(
        {
            "source_path": "unused",
            "source_sha256": "unused",
            "source_manifest_path": None,
            "metadata_path": "unused",
            "metadata_sha256": "unused",
            "concept_ids": [0],
            "extractor": {},
            "target_admission": [contract],
        }
    )
    with pytest.raises(ValueError, match=error):
        build.run_negative_build(cfg)


@pytest.mark.parametrize("layout", ["user", "top"])
def test_compatible_head_recovery_uses_nearest_latest_ancestor(tmp_path, monkeypatch, layout):
    _ancestor(tmp_path, 3, {"feature_prediction_config": _feature_prediction("last_token")})
    parent = tmp_path / "parent"
    _ancestor(parent, 1, {"feature_prediction_config": _feature_prediction("last_token")})
    metadata = ({"user_config": {"feature_prediction": _feature_prediction()}}
                if layout == "user" else {"feature_prediction_config": _feature_prediction()})
    _ancestor(parent, 2, metadata)
    middle = parent / "intermediate_sft"
    _ancestor(middle, 4, {"user_config": {}})
    target = middle / "final_sft"
    _ancestor(target, 5, {"user_config": {}})
    stats = target / "stats.safetensors"
    stats.write_bytes(b"statistics fixture")
    monkeypatch.setattr(admission, "load_method_setup", lambda *_args, **_kwargs: _setup())
    monkeypatch.setattr(admission, "resolve_tokenizer", lambda _: ExtractionTokenizer())
    monkeypatch.setattr(admission, "get_pretrained_model_config", lambda *_a, **_kw: SimpleNamespace(n_ctx=4096))
    monkeypatch.setattr(admission, "tokenizer_sha256", lambda _: "tokenizer-content-hash")

    admitted = admission.load_conditioning_admission(target / "meta_000005.json", stats)
    assert admitted.identity["context_window"] == 1000


@pytest.mark.parametrize("null_mirror", [False, True])
def test_unresolved_or_conflicting_inherited_head_fails_before_setup(tmp_path, monkeypatch, null_mirror):
    target = tmp_path / "sft"
    metadata: dict[str, object] = {"user_config": {}}
    if null_mirror:
        _ancestor(tmp_path, 1, {"feature_prediction_config": _feature_prediction()})
        metadata["feature_prediction_config"] = None
    _ancestor(target, 2, metadata)
    stats = target / "stats.safetensors"
    stats.write_bytes(b"statistics fixture")
    monkeypatch.setattr(admission, "load_method_setup", lambda *_a, **_kw: pytest.fail("unusable head reached setup"))
    error = ("conflicting feature-prediction metadata mirrors" if null_mirror
             else "SelfConditioning requires checkpoint feature_prediction metadata")
    with pytest.raises(ValueError, match=error):
        admission.load_conditioning_admission(target / "meta_000002.json", stats)


def test_common_target_admission_rejects_conditioning_overflow_before_sampling(
    tmp_path, monkeypatch, native_tokenizer
):
    meta = tmp_path / "meta_000001.json"
    meta.write_text('{"step":1,"encoder_config":{},"model_config":{"sequence_len":4096}}')
    model = tmp_path / "model_000001.pt"
    model.write_bytes(b"not loaded for admission")
    native_path = tmp_path / "tokenizer.pkl"
    native_path.write_bytes(b"native tokenizer content")
    config = {
        "family": "encdec",
        "meta_path": str(meta),
        "tokenizer_path": str(native_path),
        "max_new_tokens": 4,
        "expected_meta_sha256": admission.file_sha256(meta),
        "expected_model_sha256": admission.file_sha256(model),
        "expected_tokenizer_sha256": admission.file_sha256(native_path),
        "feature_statistics_path": "stats",
        "pretrain_meta_path": None,
    }
    rows = list(
        iter_assistant_prefixes(
            {
                "source_id": "a",
                "group_id": "a",
                "split": "test",
                "source_row": 0,
                "messages": [{"role": "user", "content": "question"}],
            }
        )
    )
    monkeypatch.setattr(build, "load_render_tokenizer", lambda *_: native_tokenizer)
    conditioner = admission.ConditioningAdmission(
        _setup(window=20), ExtractionTokenizer(), 4096, {"context_window": 20}
    )
    monkeypatch.setattr(build, "load_conditioning_admission", lambda *_: conditioner)
    accepted, contracts, rejected = build.admit_target_contexts(rows, [config])
    assert accepted == []
    assert contracts[0]["conditioning"] == {"context_window": 20}
    assert rejected[0]["reasons"] == [
        {"family": "encdec", "prompt_tokens": 48, "reason": "target_conditioning_context_overflow"}
    ]
    # Decoder prompt fits. Increasing only conditioning headroom admits the same source.
    conditioner.identity["context_window"] = 48
    accepted, _contracts, rejected = build.admit_target_contexts(rows, [config])
    assert accepted == rows and rejected == []
