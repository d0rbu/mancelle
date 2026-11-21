"""Suppression semantics, immutable grids and complete failure accounting."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
from omegaconf import OmegaConf

from nanochat import checkpoint_manager
from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig, FeaturePredLossMode
from steering.conditional.eval_conditioning import EvalConditioning
from steering.eval import dense_feature_targets
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_negative import admission as negative_admission
from steering.eval.axbench_negative import api as negative_api
from steering.eval.axbench_negative import consumer as negative_consumer
from steering.eval.axbench_negative import evaluate as evaluation_launcher
from steering.eval.axbench_negative import targets as negative_targets
from steering.eval.axbench_negative.api import (
    ChatApiClient,
    ConversationQualityJudge,
    QualityAssessment,
    SamplingConfig,
    parse_api_response,
    parse_presence_response,
)
from steering.eval.axbench_negative.consumer import (
    EvaluationConfig,
    build_grid,
    evaluate_negative_source,
)
from steering.eval.axbench_negative.evaluate import ConceptPresenceJudge
from steering.eval.axbench_negative.source import iter_assistant_prefixes
from steering.eval.axbench_negative.targets import (
    EncdecNegativeTarget,
    NativeNegativeTarget,
    _admission,
    _fit_cache,
    _training_contract,
    file_sha256,
    negative_direction,
    suppression_policy,
)
from steering.eval.axbench_upstream import methods
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    fit_diffmean_direction,
)
from steering.eval.axbench_upstream.native_decoder import NativeDecoderAdapter
from steering.eval.chat_generation import ChatGenerationResult, render_messages_for_chat_completion
from steering.eval.inference_engine import InferenceEngine
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.immutable_io import ImmutableArtifactError
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import build_tiny_model
from tests.nanochat.test_checkpoint_encdec_backward_compat import (
    _encdec_model_config,
    _encdec_user_config,
    _make_encdec_with_feature_prediction,
)
from tests.scripts.eval.test_axbench_project_methods import _setup
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.test_self_conditioned_interpolation import _run_and_capture


class TextTokenizer:
    def __init__(self):
        self.seen = []

    def render_conversation(self, conversation, max_tokens=2048):
        self.seen.append((conversation, max_tokens))
        ids = [1]
        for message in conversation["messages"]:
            ids.extend(ord(char) + 10 for char in message["content"])
        return ids, [0] * len(ids)

    def encode_special(self, value):
        return {"<assistant_start>": 2, "<assistant_end>": 3}[value]

    def get_bos_token_id(self):
        return 1

    def decode(self, values):
        return " ".join(str(value) for value in values)


def test_complete_history_render_is_uncapped_and_preserves_bytes():
    tokenizer = TextTokenizer()
    messages = [{"role": "system", "content": " précise "}, {"role": "user", "content": "a"},
                {"role": "assistant", "content": " b "}, {"role": "user", "content": "z" * 2100}]
    prompt = render_messages_for_chat_completion(tokenizer, messages)
    assert len(prompt.token_ids) > 2048 and prompt.token_ids[-1] == 2 and prompt.assistant_end_id == 3
    assert tokenizer.seen == [({"messages": messages}, None)]
    tokenizer.seen[0][0]["messages"][0]["content"] = "mutated tokenizer copy"
    assert messages[0]["content"] == " précise "


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "x"}],
    [{"role": "system", "content": "x"}], [{"role": "user", "content": "x", "name": "unknown"}],
    [{"role": "user", "content": "   "}]])
def test_render_rejects_unsupported_histories(messages):
    with pytest.raises(ValueError):
        render_messages_for_chat_completion(TextTokenizer(), messages)


def test_native_message_adapter_keeps_single_user_parity_and_exhaustion():
    adapter: Any = NativeDecoderAdapter.__new__(NativeDecoderAdapter)
    adapter.tokenizer = TextTokenizer()
    calls = []
    def stream(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return iter([8, 9])
    adapter.engine = SimpleNamespace(generate_stream=stream)
    messages = [{"role": "user", "content": " exact "}]
    assert adapter.prompt_messages(messages) == adapter.prompt_tokens(" exact ")
    result = adapter.generate_messages(messages, seed=51, max_new_tokens=2)
    assert result == ChatGenerationResult("8 9", (8, 9), True)
    assert calls[0][1]["stop_token_ids"] == {1, 3} and calls[0][1]["seed"] == 51


def _hook_model():
    model = nn.Module()
    model.register_parameter("anchor", nn.Parameter(t.zeros(1)))
    model.transformer = nn.ModuleDict({"h": nn.ModuleList([nn.Identity()])})
    return model


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_negative_residual_is_calibrated_and_removed_on_error(dtype):
    model = _hook_model()
    fit = FittedDirection(t.tensor([0.123456, -0.76]), 2.5, {})
    values = t.ones((1, 3, 2), dtype=dtype)
    block = model.transformer["h"][0]
    with pytest.raises(RuntimeError, match="deliberate"):
        with negative_direction(model, layer=0, fit=fit, magnitude=3):
            expected = values + (-7.5 * fit.direction.to(dtype))[None, None, :]
            assert t.equal(block(values), expected)
            raise RuntimeError("deliberate")
    assert t.equal(block(values), values) and not block._forward_hooks
    with negative_direction(model, layer=0, fit=fit, magnitude=0):
        assert not block._forward_hooks


def test_native_control_installs_no_hook_and_negative_uses_production_hook():
    target: Any = NativeNegativeTarget.__new__(NativeNegativeTarget)
    model = _hook_model()
    block = model.transformer["h"][0]
    def generate(messages, **kwargs):
        observed = block(t.zeros((1, 1, 2))).flatten()
        return ChatGenerationResult(str(observed.tolist()), (8,), False)
    target.adapter = SimpleNamespace(model=model, generate_messages=generate)
    target.methods = ("DiffMean",)
    target.contract = SimpleNamespace(intervention_layer=0)
    target.fits = {"f7_DiffMean": FittedDirection(t.tensor([1., -1.]), 2., {})}
    group = SimpleNamespace(feature_id=7, messages=({"role": "user", "content": "x"},))
    args = {"seed": 1, "selection_seed": 2, "max_new_tokens": 3}
    assert target.generate(group, method="Control", magnitude=0, **args).text == "[0.0, 0.0]"
    assert target.generate(group, method="DiffMean", magnitude=3, **args).text == "[-6.0, 6.0]"
    assert not block._forward_hooks


def test_self_conditioning_suppression_reverses_real_selection(monkeypatch):
    setup = _setup(n_sae_features=6)
    direction = t.tensor([.9, .7, .5, -.5, -.7, -.9])
    policy = suppression_policy(setup, direction, 6)
    segment, scores = _run_and_capture(monkeypatch, diffmean=policy.target_scores, weight=float(policy.weight),
                                      logits=t.zeros((1, 3, 6)))
    assert segment == {5: 0, 4: 0, 0: 1, 1: 1}
    assert t.equal(scores, -direction[None, :])
    pure = suppression_policy(setup, None, 0)
    segment, scores = _run_and_capture(monkeypatch, diffmean=pure.target_scores, weight=float(pure.weight),
        logits=t.tensor([[[-20., -21., 20., 21., 0., 1.]]]))
    assert segment == {3: 0, 2: 0, 1: 1, 0: 1}
    assert t.equal(scores, t.tensor([[-20., -21., 20., 21., 0., 1.]]))


def test_encdec_adapter_preserves_exact_history_policy_and_paired_seeds(monkeypatch):
    target: Any = EncdecNegativeTarget.__new__(EncdecNegativeTarget)
    target.methods = ("SelfConditioning",)
    target.setup = _setup()
    target.tokenizer, target.model = TextTokenizer(), object()
    group = SimpleNamespace(feature_id=7, messages=({"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}))
    tokens = tuple(render_messages_for_chat_completion(target.tokenizer, group.messages).token_ids)
    target.prepared = {tokens: object()}
    target.fits = {"f7_SelfConditioning": FittedDirection(t.tensor([.8, .1, -.2, -.9]), 1., {})}
    seen = []
    def generate(setup, model, tokenizer, prepared, policy, **kwargs):
        seen.append((prepared, policy, kwargs))
        return ChatGenerationResult("answer", (9,), False)
    monkeypatch.setattr(methods, "generate_self_conditioning_result", generate)
    for method, magnitude in (("Control", 0), ("SelfConditioning", 3)):
        target.generate(group, method=method, magnitude=magnitude, seed=1, selection_seed=2, max_new_tokens=4)
    assert all(item[0] is target.prepared[tokens] for item in seen)
    assert t.equal(seen[0][1].target_scores, t.zeros(4)) and float(seen[0][1].weight) == 0
    assert t.equal(seen[1][1].target_scores, t.tensor([-.8, -.1, .2, .9])) and float(seen[1][1].weight) == .5
    assert seen[0][2] == seen[1][2] == {"seed": 1, "selection_seed": 2, "max_new_tokens": 4}


def test_fit_cache_replays_existing_fit_numerics_and_rejects_corruption(tmp_path, monkeypatch):
    positive, negative = [t.tensor([[2., 0.], [3., 1.]])], [t.tensor([[0., 2.]])]
    expected = fit_diffmean_direction(positive, negative, intervention_layer=0, surface="test")
    fits, sha = _fit_cache(tmp_path, {"training": "pinned"}, lambda: {"f7_DiffMean": expected})
    assert t.equal(fits["f7_DiffMean"].direction, expected.direction)
    assert fits["f7_DiffMean"].max_activation == expected.max_activation
    def forbidden():
        raise AssertionError("fit repeated")
    assert _fit_cache(tmp_path, {"training": "pinned"}, forbidden)[1] == sha
    with pytest.raises(ImmutableArtifactError):
        _fit_cache(tmp_path, {"training": "changed"}, forbidden)
    with monkeypatch.context() as patch:
        patch.setattr(negative_targets, "FIT_POLICY", "changed-fit-policy")
        with pytest.raises(ValueError, match="identity"):
            _fit_cache(tmp_path, {"training": "pinned"}, forbidden)
    path = tmp_path / "vectors.safetensors"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        _fit_cache(tmp_path, {"training": "pinned"}, forbidden)


def test_training_contract_binds_features_and_rejects_prefix_leakage(tmp_path):
    root = tmp_path / "train" / "generate"
    root.mkdir(parents=True)
    prefix = "https://www.neuronpedia.org/model/source"
    (root / "metadata.jsonl").write_text(json.dumps({"concept_id": 0, "concept": "target concept", "ref": prefix + "/7"}) + "\n")
    pq.write_table(pa.Table.from_pylist([{"input": "TRAIN only", "output": "answer"}]), root / "train_data.parquet")
    config = {"prepared_dir": str(root.parent), "metadata_sha256": file_sha256(root / "metadata.jsonl"),
              "data_sha256": file_sha256(root / "train_data.parquet"), "ref_prefix": prefix}
    bank = _bank(tmp_path)
    identity, concepts = _training_contract(config, bank.groups)
    assert concepts == {7: 0} and identity["concepts"][0]["metadata"]["concept"] == "target concept"
    with pytest.raises(ValueError, match="uniquely"):
        _training_contract({**config, "ref_prefix": "https://www.neuronpedia.org/wrong/source"}, bank.groups)
    bank.groups[0].messages = ({"role": "user", "content": "TRAIN only"},)
    with pytest.raises(ValueError, match="overlaps"):
        _training_contract(config, bank.groups)


def test_both_target_families_generate_through_real_cpu_kv_engines():
    tokenizer = ByteChatTokenizer()
    messages = ({"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"})
    group = SimpleNamespace(feature_id=7, messages=messages)
    t.manual_seed(173)
    native_model: Any = GPT(GPTConfig(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2,
                                n_embd=64, window_pattern="L"))
    native_model.init_weights()
    native_model.eval()
    adapter: Any = NativeDecoderAdapter.__new__(NativeDecoderAdapter)
    adapter.model, adapter.tokenizer = native_model, tokenizer
    adapter.engine = InferenceEngine(native_model, tokenizer, t.device("cpu"))
    native: Any = NativeNegativeTarget.__new__(NativeNegativeTarget)
    native.adapter, native.methods = adapter, ("DiffMean",)
    native.contract = SimpleNamespace(intervention_layer=0)
    native.fits = {"f7_DiffMean": FittedDirection(t.linspace(-1., 1., 64), 1., {})}
    kwargs = {"seed": 7, "selection_seed": 8, "max_new_tokens": 2}
    control = native.generate(group, method="Control", magnitude=0, **kwargs)
    assert control == native.generate(group, method="Control", magnitude=0, **kwargs)
    assert len(native.generate(group, method="DiffMean", magnitude=1, **kwargs).token_ids) <= 2
    assert not native_model.transformer.h[0]._forward_hooks
    model = build_tiny_model(n_sae_features=6, num_buckets=2, n_layer=2, seq_len=64, vocab_size=512,
        xattn_frequency=1, feature_pred_config=FeaturePredictionConfig(loss_mode=FeaturePredLossMode.MAX_POOL))
    model.eval()
    encdec: Any = EncdecNegativeTarget.__new__(EncdecNegativeTarget)
    encdec.model, encdec.tokenizer, encdec.setup, encdec.methods = model, tokenizer, _setup(n_sae_features=6), ("SelfConditioning",)
    prompt = render_messages_for_chat_completion(tokenizer, messages)
    prepared = methods.PreparedPromptContext(tuple(prompt.token_ids), prompt.assistant_end_id,
        (ContextFeatureSegment((0, 1), (0, 1), 0, len(tokenizer.decode(prompt.token_ids))),))
    encdec.prepared = {tuple(prompt.token_ids): prepared}
    encdec.fits = {"f7_SelfConditioning": FittedDirection(t.tensor([.9, .7, .5, -.5, -.7, -.9]), 1., {})}
    control = encdec.generate(group, method="Control", magnitude=0, **kwargs)
    assert control == encdec.generate(group, method="Control", magnitude=0, **kwargs)
    assert len(encdec.generate(group, method="SelfConditioning", magnitude=6, **kwargs).token_ids) <= 2


@pytest.mark.parametrize("sae_model", ["gemma-3-4b-pt", "gemma-2-2b"])
def test_encdec_target_never_allocates_unused_gemma3_tl_weights(tmp_path, monkeypatch, sae_model):
    setup = _setup()
    setup.sae_model = sae_model
    setup.meta = {"step": 1, "model_config": {"sequence_len": 1024}}
    setup.resolved_sae = SimpleNamespace(release="fixture", sae_id="fixture", hook_name="blocks.22.hook_resid_post")
    root = tmp_path / "assets"
    (root / "tokenizer").mkdir(parents=True)
    for name in ("meta_000001.json", "model_000001.pt", "stats.safetensors", "tokenizer/tokenizer.pkl"):
        (root / name).write_bytes(b"authenticated fixture")
    config = {"methods": ["SelfConditioning"], "training": {}, "meta_path": str(root / "meta_000001.json"),
        "feature_statistics_path": str(root / "stats.safetensors"), "pretrain_meta_path": None,
        "tokenizer_path": str(root / "tokenizer/tokenizer.pkl"), "device": "cpu", "extraction_batch_size": 1}
    bank = _bank(tmp_path)
    checkpoint = {key: file_sha256(root / file) for key, file in (
        ("meta_sha256", "meta_000001.json"), ("model_sha256", "model_000001.pt"), ("tokenizer_sha256", "tokenizer/tokenizer.pkl"))}
    bank.identity["targets"] = [{"family": "encdec", **checkpoint, "sequence_len": 1024, "max_new_tokens": 128}]
    tokenizer, cfg, events = object(), SimpleNamespace(n_ctx=8192, model_name=sae_model), []
    sae = nn.Linear(2, 2)
    nnterp = SimpleNamespace(model=nn.Linear(2, 2))
    monkeypatch.setattr(negative_targets, "_training_contract", lambda *args: ({}, {7: 0}))
    monkeypatch.setattr(methods, "load_method_setup", lambda *args, **kwargs: setup)
    monkeypatch.setattr(negative_targets, "get_base_dir", lambda: str(root))
    monkeypatch.setattr(negative_targets, "get_tokenizer", TextTokenizer)
    def forbidden_tl(**kwargs):
        raise AssertionError("unused TransformerLens weights must not be allocated")
    def legacy_loader(actual_setup, device):
        assert sae_model == "gemma-2-2b" and actual_setup is setup and device == "cpu"
        events.append("legacy")
        return SimpleNamespace(cfg=cfg, tokenizer=tokenizer), sae
    def nnterp_factory(**kwargs):
        assert kwargs == {"tl_model": None, "device": "cpu", "model_dtype": t.bfloat16, "model_name": sae_model}
        events.append("nnterp")
        return nnterp
    monkeypatch.setattr(methods, "load_extraction_stack", legacy_loader)
    monkeypatch.setattr(dense_feature_targets, "load_sae_and_model", forbidden_tl)
    monkeypatch.setattr(dense_feature_targets, "get_pretrained_model_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(dense_feature_targets, "resolve_tokenizer", lambda name: tokenizer)
    monkeypatch.setattr(dense_feature_targets.SAE, "from_pretrained", lambda **kwargs: sae)
    monkeypatch.setattr(dense_feature_targets, "build_nnterp_model", nnterp_factory)
    class ReachedConditioning(Exception):
        pass
    def conditioning(actual_setup, extraction_tokenizer, model_context, **kwargs):
        assert actual_setup is setup and extraction_tokenizer is tokenizer and model_context == 8192
        if sae_model.startswith("gemma-3"):
            assert next(sae.parameters()).dtype == t.bfloat16
        raise ReachedConditioning
    monkeypatch.setattr(negative_targets, "conditioning_identity", conditioning)
    with pytest.raises(ReachedConditioning):
        EncdecNegativeTarget(config, bank, bank.groups, max_new_tokens=128)
    assert events == (["nnterp"] if sae_model.startswith("gemma-3") else ["legacy"])


@pytest.fixture
def encdec_fit_fixture(tmp_path, monkeypatch):
    setup = _setup()
    setup.sae_model = "gemma-2-2b"
    setup.meta = {"step": 1, "model_config": {"sequence_len": 1024}}
    setup.resolved_sae = SimpleNamespace(release="fixture", sae_id="fixture", hook_name="blocks.22.hook_resid_post")
    setup.prompt_context_window, setup.prompt_segment_size = 512, 32
    root = tmp_path / "assets"
    (root / "tokenizer").mkdir(parents=True)
    for name in ("meta_000001.json", "model_000001.pt", "stats.safetensors", "tokenizer/tokenizer.pkl"):
        (root / name).write_bytes(b"authenticated fixture")
    config = {"methods": ["SelfConditioning"], "training": {"prepared_dir": str(tmp_path)}, "meta_path": str(root / "meta_000001.json"),
        "feature_statistics_path": str(root / "stats.safetensors"), "pretrain_meta_path": None,
        "tokenizer_path": str(root / "tokenizer/tokenizer.pkl"), "device": "cpu", "extraction_batch_size": 1,
        "fit_dir": str(tmp_path / "fits")}
    bank = _bank(tmp_path)
    checkpoint = {key: file_sha256(root / file) for key, file in (
        ("meta_sha256", "meta_000001.json"), ("model_sha256", "model_000001.pt"), ("tokenizer_sha256", "tokenizer/tokenizer.pkl"))}
    model: Any = nn.LayerNorm(2, eps=1e-5)
    sae: Any = nn.LayerNorm(2, eps=1e-5)
    model.cfg = SimpleNamespace(n_ctx=8192, normalization_eps=model.eps)
    model.tokenizer = object()
    sae.cfg = SimpleNamespace(normalization_eps=sae.eps)
    monkeypatch.setattr(negative_admission, "tokenizer_sha256", lambda _: "unchanged-tokenizer")
    monkeypatch.setattr(negative_targets, "tokenizer_sha256", lambda _: "unchanged-tokenizer")
    conditioning = negative_admission.conditioning_identity(setup, model.tokenizer, 8192,
        statistics_path=root / "stats.safetensors", pretrain_meta_path=None)
    bank.identity["targets"] = [{"family": "encdec", **checkpoint, "sequence_len": 1024,
                                "max_new_tokens": 128, "conditioning": conditioning}]
    monkeypatch.setattr(negative_targets, "_training_contract", lambda *args: ({}, {7: 0}))
    monkeypatch.setattr(methods, "load_method_setup", lambda *args, **kwargs: setup)
    monkeypatch.setattr(negative_targets, "get_base_dir", lambda: str(root))
    monkeypatch.setattr(negative_targets, "get_tokenizer", TextTokenizer)
    monkeypatch.setattr(methods, "load_extraction_stack", lambda *args: (model, sae))
    monkeypatch.setattr(negative_targets, "select_training_texts", lambda *args, **kwargs: {0: object()})
    fitted = []
    def fit(*args, **kwargs):
        fitted.append(True)
        return {0: t.zeros(setup.n_sae_features)}
    monkeypatch.setattr(methods, "fit_concept_dense_diffmeans", fit)
    class ReachedPromptPreparation(Exception):
        pass
    def prepare(*args, **kwargs):
        raise ReachedPromptPreparation
    monkeypatch.setattr(methods, "prepare_prompt_context", prepare)
    return SimpleNamespace(setup=setup, root=root, config=config, bank=bank, model=model, sae=sae,
                           fitted=fitted, reached_prompt=ReachedPromptPreparation)


@pytest.mark.parametrize("changed_asset", ["model", "sae"])
def test_encdec_fit_rejects_extraction_configuration_drift(encdec_fit_fixture, changed_asset):
    """Equal tensors do not imply equal extraction numerics or compatible fits."""
    fixture = encdec_fit_fixture
    for _ in range(2):
        with pytest.raises(fixture.reached_prompt):
            EncdecNegativeTarget(fixture.config, fixture.bank, fixture.bank.groups, max_new_tokens=128)
    assert len(fixture.fitted) == 1  # The unchanged second construction really replayed the fit.
    changed = fixture.model if changed_asset == "model" else fixture.sae
    tensor_hash = negative_targets.module_weights_sha256(changed)
    probe = t.tensor([[1., 1.01]])
    original = changed(probe).detach().clone()
    changed.eps = changed.cfg.normalization_eps = 0.1
    assert not t.equal(original, changed(probe))
    assert negative_targets.module_weights_sha256(changed) == tensor_hash
    with pytest.raises(ImmutableArtifactError):
        EncdecNegativeTarget(fixture.config, fixture.bank, fixture.bank.groups, max_new_tokens=128)
    assert len(fixture.fitted) == 1


def test_encdec_generation_identity_binds_actual_recovered_head(encdec_fit_fixture, monkeypatch):
    """Ancestor recovery can alter the head while every child checkpoint byte stays fixed."""
    fixture = encdec_fit_fixture
    head = FeaturePredictionConfig(loss_mode=FeaturePredLossMode.MAX_POOL, layer=-1)
    model = _make_encdec_with_feature_prediction(head)
    metadata = {"step": 1, "model_config": _encdec_model_config(), "user_config": _encdec_user_config()}
    checkpoint_manager.save_checkpoint(str(fixture.root), 1, model.state_dict(), None, metadata)
    fixture.setup.meta = metadata
    fixture.setup.n_sae_features = model._n_sae_features
    ancestor = fixture.root.parent / "meta_000001.json"
    (ancestor.parent / "model_000001.pt").write_bytes(b"ancestor discovery only")
    ancestor.write_text(json.dumps({"user_config": {"feature_prediction": asdict(head)}}))
    contract = fixture.bank.identity["targets"][0]
    contract.update({"meta_sha256": file_sha256(fixture.root / "meta_000001.json"),
                     "model_sha256": file_sha256(fixture.root / "model_000001.pt"),
                     "sequence_len": model.dec_config.sequence_len,
                     "conditioning": negative_admission.conditioning_identity(
                         fixture.setup, fixture.model.tokenizer, 8192,
                         statistics_path=fixture.root / "stats.safetensors", pretrain_meta_path=None)})
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", MockTokenizer)
    monkeypatch.setattr(methods, "prepare_prompt_context", lambda *args, **kwargs: object())
    def load_target(setup, device):
        loaded, tokenizer, _ = checkpoint_manager.build_encdec_model(
            str(fixture.root), 1, device, "eval",
            eval_conditioning=EvalConditioning.deferred("negative target identity test"),
        )
        return loaded, tokenizer
    monkeypatch.setattr(methods, "load_target_model", load_target)
    first = EncdecNegativeTarget(fixture.config, fixture.bank, fixture.bank.groups, max_new_tokens=8)
    same = EncdecNegativeTarget(fixture.config, fixture.bank, fixture.bank.groups, max_new_tokens=8)
    assert first.identity == same.identity
    head.layer = 0
    ancestor.write_text(json.dumps({"user_config": {"feature_prediction": asdict(head)}}))
    changed = EncdecNegativeTarget(fixture.config, fixture.bank, fixture.bank.groups, max_new_tokens=8)
    assert first.model.feature_pred_config.layer == -1 and changed.model.feature_pred_config.layer == 0
    assert first.model._feature_pred_layer_idx == 1 and changed.model._feature_pred_layer_idx == 0
    assert negative_targets.module_weights_sha256(first.model) == negative_targets.module_weights_sha256(changed.model)
    assert first.identity["checkpoint"] == changed.identity["checkpoint"]
    assert len(fixture.fitted) == 1  # Head placement has no effect on the TRAIN direction.
    assert first.identity != changed.identity


def _bank(tmp_path):
    candidate = tuple(iter_assistant_prefixes({"source_id": "s", "group_id": "g", "split": "test", "source_row": 0,
        "messages": [{"role": "user", "content": "original one"}, {"role": "assistant", "content": "history"},
                     {"role": "user", "content": "original two"}]}))[-1]
    group = SimpleNamespace(**candidate.to_dict(), feature_id=7, member_ids=(7,), description="target concept")
    group.messages = tuple(group.messages)
    return SimpleNamespace(root=tmp_path / "bank", manifest_sha256="a" * 64,
                           identity={"scorer": {"name": "fixed"}}, groups=(group,))


class RecordingTarget:
    family = "native"
    methods = ("DiffMean",)
    identity = {"family": "native", "checkpoint": "pinned", "fit": "pinned"}

    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs or {"Control": "rich", "DiffMean": "avoiding"}

    def generate(self, group, **kwargs):
        self.calls.append((group.messages, kwargs))
        value = self.outputs[kwargs["method"]]
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, ChatGenerationResult) else ChatGenerationResult(value, (8, 9), False)


class Scorer:
    identity = {"name": "fixed"}

    def assess_completion(self, candidate, response, target):
        status = "no_tokens" if not response else "active" if response == "rich" else "inactive"
        return SimpleNamespace(to_dict=lambda: {"status": status, "max_activation": 1. if status == "active" else 0.})


class Judge:
    identity = {"rubric": "full-conversation"}

    def assess(self, messages, response, *args, **kwargs):
        assert messages[-1]["content"] == "original two" and len(messages) == 3
        rubric: dict[str, Any]
        if args:
            assert args == ("target concept",)
            rubric = {"concept_present": response == "rich", "reason": "fixture"}
        else:
            rubric = {"useful": True, "refusal": False, "coherent": True, "substantive": True,
                      "task_fulfilled": True, "reason": "useful"}
        observed = parse_api_response({"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(rubric)}}]}, "offline-judge")
        if args:
            return parse_presence_response(observed)
        return QualityAssessment(**rubric, api_response=observed.to_dict(),
                                 requested_sampling=SamplingConfig().to_dict())


def test_paired_grid_resume_and_unknowns_have_explicit_denominators(tmp_path):
    bank, target, config = _bank(tmp_path), RecordingTarget(), EvaluationConfig(tmp_path / "eval", magnitudes=(1., 2.))
    cells = build_grid(bank.groups, target, config, bank.manifest_sha256)
    assert len({cell["cell_id"] for cell in cells}) == 3
    assert len({cell["seed"] for cell in cells}) == len({cell["selection_seed"] for cell in cells}) == 1
    result = evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), replace(config, max_cells=1))
    assert not result["complete"] and result["summary"]["pending_cells"] == 2
    result = evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), config)
    assert result["complete"] and len(target.calls) == 3
    assert result["summary"]["arms"][1]["paired_active_change"] == {"n": 1, "mean": -1.0}
    assert result["summary"]["arms"][1]["paired_useful_change"] == {"n": 1, "mean": 0.0}
    assert result["summary"]["arms"][1]["counts"]["useful_strict_zero"] == 1
    assert evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), config) == result
    assert len(target.calls) == 3
    with pytest.raises(ImmutableArtifactError):
        evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), replace(config, seed=43))


def test_interruption_after_generation_commit_resumes_without_resampling(tmp_path):
    class InterruptedJudge(Judge):
        def assess(self, *args, **kwargs):
            raise KeyboardInterrupt("interrupted before assessment commit")

    bank, target = _bank(tmp_path), RecordingTarget()
    config = EvaluationConfig(tmp_path / "eval", magnitudes=(1.,))
    with pytest.raises(KeyboardInterrupt, match="before assessment commit"):
        evaluate_negative_source(bank, target, Scorer(), InterruptedJudge(), Judge(), config)
    generations = list((config.output_dir / "cells").glob("*.generation.json"))
    assert len(generations) == 1 and len(target.calls) == 1
    assert not list((config.output_dir / "cells").glob("*.assessment.json"))
    committed_bytes = generations[0].read_bytes()
    # A fresh control draw would be distinguishable; resume must use the
    # already committed observation and must not call the target again.
    target.outputs["Control"] = "different control must never be generated"
    result = evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), config)
    assert result["complete"] and result["summary"]["recorded_cells"] == 2
    assert [call[1]["method"] for call in target.calls] == ["Control", "DiffMean"]
    assert generations[0].read_bytes() == committed_bytes
    assert result["summary"]["arms"][1]["paired_active_change"] == {"n": 1, "mean": -1.0}


@pytest.mark.parametrize("change", ["quality_rubric", "presence_rubric", "evaluation_protocol"])
@pytest.mark.parametrize("completed", [False, True])
def test_changed_evaluation_semantics_reject_resume_before_calls(tmp_path, monkeypatch, change, completed):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    calls = []

    def complete(self, messages, *, cache_key):
        calls.append((messages, cache_key))
        request = json.loads(messages[-1]["content"])
        value = ({"concept_present": request["response"] == "rich", "reason": "fixture"}
                 if "concept" in request else
                 {"useful": True, "refusal": False, "coherent": True, "substantive": True,
                  "task_fulfilled": True, "reason": "fixture"})
        return parse_api_response({"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(value)}}]}, "offline-judge")

    monkeypatch.setattr(ChatApiClient, "complete", complete)
    client = OpenRouterLLMClient(LLMClientConfig(model="offline-judge", requests_per_minute=0), max_retries=0)
    api = ChatApiClient(client, SamplingConfig())
    bank, target = _bank(tmp_path), RecordingTarget()
    config = EvaluationConfig(tmp_path / "eval", magnitudes=(1.,), max_cells=None if completed else 1)
    result = evaluate_negative_source(bank, target, Scorer(), ConversationQualityJudge(api), ConceptPresenceJudge(api), config)
    assert result["complete"] == completed
    previous_calls, generations = len(calls), len(target.calls)
    if change == "quality_rubric":
        monkeypatch.setattr(negative_api, "QUALITY_RUBRIC", negative_api.QUALITY_RUBRIC + " Require factual correctness.")
    elif change == "presence_rubric":
        monkeypatch.setattr(evaluation_launcher, "PRESENCE_RUBRIC", evaluation_launcher.PRESENCE_RUBRIC + " Require an explicit mention.")
    else:
        monkeypatch.setattr(negative_consumer, "FORMAT", "changed-evaluation-protocol")
    with pytest.raises(ImmutableArtifactError, match="refusing to overwrite"):
        evaluate_negative_source(bank, target, Scorer(), ConversationQualityJudge(api),
                                 ConceptPresenceJudge(api), replace(config, max_cells=None))
    assert len(calls) == previous_calls and len(target.calls) == generations


@pytest.mark.parametrize("output,status", [(RuntimeError("failed generation"), "failed"),
    (ChatGenerationResult("", (), False), "empty"), (ChatGenerationResult("avoiding", (8, 9), True), "truncated")])
def test_failed_empty_truncated_never_count_as_suppression(tmp_path, output, status):
    target = RecordingTarget({"Control": "rich", "DiffMean": output})
    result = evaluate_negative_source(_bank(tmp_path), target, Scorer(), Judge(), Judge(),
        EvaluationConfig(tmp_path / "eval", magnitudes=(1.,)))
    arm = result["summary"]["arms"][1]
    assert arm["counts"][status] == 1 and arm["counts"]["useful_strict_zero"] == 0
    assert arm["paired_active_change"] == {"n": 0, "mean": None}
    assert result["complete"] and result["summary"]["declared_cells"] == 2


def test_judge_failure_is_unknown_without_discarding_sae_measurement(tmp_path):
    class FailedJudge(Judge):
        def assess(self, *args, **kwargs):
            raise RuntimeError("unavailable")
    result = evaluate_negative_source(_bank(tmp_path), RecordingTarget(), Scorer(), FailedJudge(), Judge(),
        EvaluationConfig(tmp_path / "eval", magnitudes=(1.,)))
    arm = result["summary"]["arms"][1]
    assert arm["counts"]["sae_measured"] == 1 and arm["counts"]["quality_judged"] == 0
    assert arm["counts"]["useful_strict_zero"] == 0 and arm["paired_useful_change"]["n"] == 0


def test_shared_control_generated_once_but_assessed_for_every_feature(tmp_path):
    bank = _bank(tmp_path)
    second = SimpleNamespace(**{**vars(bank.groups[0]), "feature_id": 8, "member_ids": (8,)})
    bank.groups = (*bank.groups, second)
    target = RecordingTarget()
    seen = []
    class CountingScorer(Scorer):
        def assess_completion(self, candidate, response, target):
            seen.append((response, target.feature_id))
            return super().assess_completion(candidate, response, target)
    config = EvaluationConfig(tmp_path / "eval", magnitudes=(1.,))
    first = evaluate_negative_source(bank, target, CountingScorer(), Judge(), Judge(), replace(config, max_cells=2))
    assert not first["complete"]
    result = evaluate_negative_source(bank, target, CountingScorer(), Judge(), Judge(), config)
    assert result["complete"] and result["summary"]["declared_cells"] == 4
    assert sum(call[1]["method"] == "Control" for call in target.calls) == 1
    assert len(target.calls) == 3 and ("rich", 7) in seen and ("rich", 8) in seen


def test_negative_activation_is_not_strict_zero_success(tmp_path):
    class NegativeScorer(Scorer):
        def assess_completion(self, candidate, response, target):
            return SimpleNamespace(to_dict=lambda: {"status": "inactive", "max_activation": -.2})
    result = evaluate_negative_source(_bank(tmp_path), RecordingTarget(), NegativeScorer(), Judge(), Judge(),
        EvaluationConfig(tmp_path / "eval", magnitudes=(1.,)))
    assert result["summary"]["arms"][1]["counts"]["useful_strict_zero"] == 0


def test_changed_generation_after_final_publication_fails(tmp_path):
    bank, target, config = _bank(tmp_path), RecordingTarget(), EvaluationConfig(tmp_path / "eval", magnitudes=(1.,))
    evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), config)
    path = next((config.output_dir / "cells").glob("*.generation.json"))
    value = json.loads(path.read_text())
    value["payload"]["text"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash"):
        evaluate_negative_source(bank, target, Scorer(), Judge(), Judge(), config)


def test_frozen_target_admission_and_scorer_drift_fail(tmp_path):
    bank = _bank(tmp_path)
    checkpoint = {key: key for key in ("model_sha256", "meta_sha256", "tokenizer_sha256")}
    bank.identity["targets"] = [{"family": "native", **checkpoint, "sequence_len": 1024, "max_new_tokens": 128}]
    _admission(bank, "native", checkpoint, max_new_tokens=128, sequence_len=1024)
    with pytest.raises(ValueError, match="budget"):
        _admission(bank, "native", checkpoint, max_new_tokens=129, sequence_len=1024)
    with pytest.raises(ValueError, match="checkpoint"):
        _admission(bank, "native", {**checkpoint, "model_sha256": "different"}, max_new_tokens=128, sequence_len=1024)
    bank.identity["scorer"] = {"name": "changed"}
    with pytest.raises(ValueError, match="SAE assessment"):
        evaluate_negative_source(bank, RecordingTarget(), Scorer(), Judge(), Judge(), EvaluationConfig(tmp_path / "eval"))


@pytest.mark.parametrize("change", [{"magnitudes": (0.,)}, {"magnitudes": (-1.,)}, {"magnitudes": (float("nan"),)},
                                  {"replicates": True}, {"split": "train"}, {"magnitudes": (1., 1.)}])
def test_bad_evaluation_grids_fail(change):
    with pytest.raises(ValueError):
        replace(EvaluationConfig(Path("unused")), **change).validate("native")


@pytest.mark.parametrize("body,status,present", [
    ('{"concept_present":false,"reason":"only in history"}', "assessed", False),
    ('{"concept_present":"false","reason":"invalid type"}', "unavailable", None),
    ('{"concept_present":true,"concept_present":false,"reason":"duplicate"}', "unavailable", None),
])
def test_presence_judge_preserves_full_history_and_unknowns(body, status, present):
    messages = [{"role": "system", "content": "Original rules λ"},
                {"role": "user", "content": "earlier"}, {"role": "assistant", "content": "target only here"},
                {"role": "user", "content": "Answer the follow-up  "}]
    response = parse_api_response({"choices": [{"message": {"content": body}, "finish_reason": "stop"}]}, "judge")
    calls = []
    api: Any = SimpleNamespace(identity={"model": "judge"}, complete=lambda sent, **kwargs: calls.append(sent) or response)
    result = ConceptPresenceJudge(api).assess(messages, "unrelated answer", "target", cache_key="key")
    data = json.loads(calls[0][1]["content"])
    assert data == {"conversation": messages, "response": "unrelated answer", "concept": "target"}
    assert result["status"] == status and result["concept_present"] is present
    assert result["api_response"] == response.to_dict()


@pytest.mark.parametrize("refusal_kind", ["flag", "content_filter"])
def test_presence_provider_refusal_cannot_be_a_valid_assessment(refusal_kind):
    message = {"content": '{"concept_present":false,"reason":"valid JSON inside refusal"}'}
    if refusal_kind == "flag":
        message["refusal"] = "Cannot judge this request"
    response = parse_api_response({"choices": [{"message": message,
        "finish_reason": "content_filter" if refusal_kind == "content_filter" else "stop"}]}, "judge")
    assert response.refusal and response.status == "complete"
    api: Any = SimpleNamespace(identity={"model": "judge"}, complete=lambda *args, **kwargs: response)
    result = ConceptPresenceJudge(api).assess([{"role": "user", "content": "task"}], "answer", "target", cache_key="key")
    assert result["status"] == "unavailable" and result["concept_present"] is None
    assert result["api_response"] == response.to_dict()


def test_launcher_prepares_target_before_loading_detector_and_pins_bank(tmp_path, monkeypatch):
    bank, order = _bank(tmp_path), []
    config = OmegaConf.create({"source_dir": str(bank.root), "source_manifest_sha256": bank.manifest_sha256,
        "output_dir": str(tmp_path / "eval"), "split": "test", "magnitudes": [1.], "replicates": 1,
        "seed": 42, "max_new_tokens": 128, "max_cells": None,
        "response_cache": str(tmp_path / "eval/api.sqlite"),
        "target": {"family": "native", "fit_dir": str(tmp_path / "fits")},
        "render_tokenizer_path": "tokenizer.pkl", "render_tokenizer_sha256": "sha", "extractor": {}, "judge": {}})
    monkeypatch.setattr(evaluation_launcher, "load_negative_source", lambda path: bank)
    monkeypatch.setattr(evaluation_launcher, "build_target", lambda *args, **kwargs: order.append("target") or RecordingTarget())
    monkeypatch.setattr(evaluation_launcher, "load_render_tokenizer", lambda *args: order.append("tokenizer") or object())
    monkeypatch.setattr(evaluation_launcher, "build_chat_scorer", lambda *args: order.append("scorer") or Scorer())
    monkeypatch.setattr(evaluation_launcher, "SynthResponseCache", lambda path: nullcontext(None))
    monkeypatch.setattr(evaluation_launcher, "create_chat_api", lambda *args: object())
    expected = {"complete": True, "summary": {"recorded_cells": 0, "declared_cells": 0}}
    monkeypatch.setattr(evaluation_launcher, "evaluate_negative_source", lambda *args: expected)
    assert evaluation_launcher.run_negative_evaluation(config) == expected
    assert order == ["target", "tokenizer", "scorer"]
    order.clear()
    config.source_manifest_sha256 = "changed"
    with pytest.raises(ValueError, match="declared bank"):
        evaluation_launcher.run_negative_evaluation(config)
    assert not order


@pytest.mark.parametrize("placement", ["same", "descendant", "ancestor"])
def test_launcher_rejects_fit_directory_overlap_before_target_preparation(tmp_path, monkeypatch, placement):
    bank = _bank(tmp_path)
    fit_dir = {"same": bank.root, "descendant": bank.root / "fits", "ancestor": bank.root.parent}[placement]
    config = OmegaConf.create({
        "source_dir": str(bank.root), "source_manifest_sha256": bank.manifest_sha256,
        "output_dir": str(tmp_path / "eval"), "split": "test", "magnitudes": [1.],
        "replicates": 1, "seed": 42, "max_new_tokens": 128, "max_cells": None,
        "response_cache": str(tmp_path / "eval/api.sqlite"),
        "target": {"family": "native", "fit_dir": str(fit_dir)},
        "render_tokenizer_path": "tokenizer.pkl", "render_tokenizer_sha256": "sha",
        "extractor": {}, "judge": {},
    })
    monkeypatch.setattr(evaluation_launcher, "load_negative_source", lambda _path: bank)
    monkeypatch.setattr(evaluation_launcher, "build_target", lambda *_args, **_kwargs:
                        pytest.fail("overlapping fit directory reached target preparation"))
    with pytest.raises(ValueError, match="disjoint"):
        evaluation_launcher.run_negative_evaluation(config)


@pytest.mark.parametrize("judge_field", ["quality", "presence"])
@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("corruption", ["stripped", "contradictory"])
def test_evaluation_resume_rejects_missing_or_contradictory_judge_evidence(tmp_path, judge_field, completed, corruption):
    calls = []

    class EvidenceApi:
        identity = {"provider": "fixture", "model": "offline-judge"}

        def requested_sampling(self, _key):
            return SamplingConfig(temperature=0, max_tokens=100).to_dict()

        def complete(self, messages, *, cache_key):
            calls.append(cache_key)
            request = json.loads(messages[-1]["content"])
            value = ({"concept_present": False, "reason": "fixture"} if "concept" in request else
                     {"useful": True, "refusal": False, "coherent": True, "substantive": True,
                      "task_fulfilled": True, "reason": "fixture"})
            return parse_api_response({"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(value)}}]}, "offline-judge")

    api: Any = EvidenceApi()
    bank, target = _bank(tmp_path), RecordingTarget({"Control": "avoiding", "DiffMean": "avoiding"})
    quality, presence = ConversationQualityJudge(api), ConceptPresenceJudge(api)
    config = EvaluationConfig(tmp_path / "eval", magnitudes=(1.,), max_cells=None if completed else 1)
    assert evaluate_negative_source(bank, target, Scorer(), quality, presence, config)["complete"] == completed
    path = next((config.output_dir / "cells").glob("*.assessment.json"))
    record = json.loads(path.read_text())
    judgment = record["payload"][judge_field]
    assert judgment["api_response"] is not None
    if corruption == "stripped":
        del judgment["api_response"]
    elif judge_field == "quality":
        # This remains an internally valid rubric, but contradicts the raw verdict.
        judgment.update(useful=False, task_fulfilled=False)
    else:
        judgment["concept_present"] = True
    record["payload_sha256"] = negative_consumer.digest(record["payload"])
    path.write_bytes(negative_consumer.canonical_bytes(record))
    if completed:
        manifest_path = config.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            if item["path"] == path.relative_to(config.output_dir).as_posix():
                item["sha256"] = file_sha256(path)
        manifest_path.write_bytes(negative_consumer.canonical_bytes(manifest))
    before = len(calls), len(target.calls)
    with pytest.raises(ValueError, match="quality|presence|provider|evidence"):
        evaluate_negative_source(bank, target, Scorer(), quality, presence, replace(config, max_cells=None))
    assert (len(calls), len(target.calls)) == before
