"""Real tiny-model forwards through the counter campaign's generation dispatcher.

The SAE and tokenizer are small deterministic fixtures; decoder, feature head,
KV cache, feature selection, LoRA, interventions and artifact writes are real.
Token choices are scripted to exercise a fixed decode trajectory and stop token.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
import torch.nn.functional as F
from omegaconf import DictConfig

import nanochat.flash_attention as flash_attention
import steering.eval.inference_engine as inference
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig, FeaturePredLossMode
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.eval.axbench_adv import encdec_baseline_counter as campaign
from steering.eval.axbench_adv.encdec_sources import EncDecSources
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.native_mailbox import read_json
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from steering.eval.axbench_upstream import methods as upstream_methods
from steering.eval.axbench_upstream.encdec_baselines import run as producer
from steering.eval.axbench_upstream.encdec_baselines.directions import FittedDirection
from steering.eval.axbench_upstream.encdec_baselines.lora import LORA_RANK, RsLoRAAdapter
from steering.eval.axbench_upstream.methods import PreparedPromptContext
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from tests.axbench_adv.unit.test_encdec_baseline_counter import _source
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import build_tiny_model, method_setup_double


@pytest.fixture
def tiny_campaign(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    tokenizer = MockTokenizer()
    with t.random.fork_rng(devices=[]):
        t.manual_seed(712)
        model = build_tiny_model(
            n_sae_features=PositiveInt.parse(8),
            num_buckets=PositiveInt.parse(4),
            feature_pred_config=FeaturePredictionConfig(loss_mode=FeaturePredLossMode.MAX_POOL),
            seq_len=256,
            vocab_size=tokenizer.vocab_size,
            n_layer=2,
            xattn_frequency=1,
        ).eval()
        # Initialization zeros output projections. Give all three routes a real
        # signal so a dropped intervention or conditioning bag cannot pass.
        with t.no_grad():
            for name, parameter in model.named_parameters():
                if "c_proj.weight" in name or "lm_head.weight" in name:
                    parameter.normal_(std=0.03)
    width = model.dec_config.n_embd
    fitted = {
        "diffmean": FittedDirection(t.linspace(-0.5, 0.5, width), 3.0, {}),
        "probe": FittedDirection(t.linspace(0.4, -0.6, width), 5.0, {}),
    }
    state = {
        "layer_0_attn_c_proj_lora_A": t.linspace(-0.05, 0.07, LORA_RANK * width).reshape(LORA_RANK, width),
        "layer_0_attn_c_proj_lora_B": t.linspace(0.08, -0.03, width * LORA_RANK).reshape(width, LORA_RANK),
    }
    setup = method_setup_double(
        n_sae_features=8,
        feature_vocab=None,
        feature_mask=None,
        consolidation=None,
        percentile_table=t.zeros((8, 99), dtype=t.float64),
        output_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            delta_weights=None,
            min_k=1,
            max_k=3,
            mode=SamplingMode.top_k_percentile,
        ),
    )
    source = cast(EncDecSources, SimpleNamespace(**{
        **vars(_source()),
        "setup": setup,
        "fitted": {0: fitted},
        "lora_states": {0: state},
    }))
    preparations = []

    def prepare(_setup: Any, _base: Any, _sae: Any, _tokenizer: Any, *,
                token_ids: tuple[int, ...], assistant_end_id: int) -> PreparedPromptContext:
        # Distinguish original and steered SAE bags, not just their token text.
        steered = "steer" in tokenizer.decode(list(token_ids))
        features = (4, 5) if steered else (1, 2)
        end = len(tokenizer.decode(list(token_ids)))
        turn = setup.segment_mode is SegmentMode.turn_delimited_fixed_tokens
        segments = (ContextFeatureSegment(features, (0, 1), 0, end - int(turn)),)
        if turn:
            segments += (ContextFeatureSegment((6, 7), (0, 1), end - 1, end),)
        prepared = PreparedPromptContext(
            tuple(token_ids), assistant_end_id,
            segments,
            segment_mode=setup.segment_mode,
            token_assignments=(0,) * (len(token_ids) - 1) + (1,) if turn else None,
        )
        preparations.append(prepared)
        return prepared

    monkeypatch.setattr(campaign, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(campaign, "load_extraction_stack", lambda *_: (None, None))
    monkeypatch.setattr(campaign, "load_target_model", lambda *_: (model, tokenizer))
    monkeypatch.setattr(producer, "prepare_prompt_context", prepare)
    counter = SimpleNamespace(artifacts={
        (family, 0): SimpleNamespace(loaded=SimpleNamespace(direction=direction))
        for family, direction in (
            ("diffmean", t.linspace(-0.01, 0.01, width)),
            ("linear_probe", t.linspace(0.02, -0.01, width)),
        )
    })
    return SimpleNamespace(
        source=source, model=model, tokenizer=tokenizer, counter=counter,
        cfg=cast(DictConfig, SimpleNamespace(device="cpu", counter_layer=0)),
        identity={"concept_ids": [0], "magnitudes": [256.0], "runtime": {"tiny": "cpu"},
                  "phase": "pilot"},
        preparations=preparations,
    )


@pytest.mark.parametrize(("mode", "corruption"), [
    (SegmentMode.fixed_tokens, None),
    (SegmentMode.turn_delimited_fixed_tokens, None),
    pytest.param(SegmentMode.fixed_tokens, "zero", id="zero-replay-mismatch"),
    pytest.param(SegmentMode.fixed_tokens, "archive", id="archived-positive-mismatch"),
])
def test_real_generation_composes_all_methods_and_resumes_without_state_leaks(
    tiny_campaign: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: SegmentMode,
    corruption: str | None,
) -> None:
    run = tiny_campaign
    run.source.setup.segment_mode = mode
    run.source.setup.turn_self_cond_segment_size_rustbpe = 2
    run.source.identity["runtime"] = run.identity["runtime"]
    for method in ENCDEC_BASELINE_METHODS:
        run.source.source_main[f"{method}_steered_generation"] = (
            "different archived positive" if corruption == "archive" else "hijk"
        )
    model = run.model
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    projection = model._decoder_blocks[0].attn.c_proj
    real_generate = campaign._generate_encdec_cell
    generations = []
    trajectory = [106, 107, 108, 109, run.tokenizer.encode_special("<assistant_end>")]

    def generate(cell: Any, **kwargs: Any) -> str:
        logits, samples, forwards, raw_blocks, cross_inputs, encoded_ids = [], [], [], [], [], []
        actual_forward = model.forward
        actual_config = upstream_methods.build_self_conditioning_config
        before_callback = model._post_cross_attention_eval_callback
        shifts = []

        def config(*args: Any, **options: Any) -> Any:
            result = actual_config(*args, **options)
            assert isinstance(result.selection, SamplingBundle)
            assert result.selection.rng.initial_seed() == cell.selection_seed
            assert result.score_policy is None
            return result

        def sample(values: t.Tensor, rng: t.Generator, temperature: float, top_k: int) -> t.Tensor:
            assert rng.initial_seed() == cell.token_seed
            assert (temperature, top_k) == (1.0, 50)
            assert t.isfinite(values).all()
            logits.append(values.detach().clone())
            token = trajectory[len(samples)]
            samples.append(token)
            return t.tensor([[token]])

        def forward(ids: t.Tensor, *args: Any, **options: Any) -> Any:
            assert isinstance(model._decoder_blocks[0].attn.c_proj, RsLoRAAdapter) == (
                cell.method == "EncDecLoRA"
            )
            forwards.append((ids.clone(), options["seg_map"].clone(),
                             options.get("return_fp_logits", False)))
            return actual_forward(ids, *args, **options)

        def callback(layer: int, residual: t.Tensor) -> t.Tensor:
            result = before_callback(layer, residual)
            if layer == 0:
                shifts.append((residual.clone(), result.clone()))
            return result

        handles = [
            model._decoder_blocks[0].register_forward_hook(
                lambda _module, _inputs, result: raw_blocks.append(result.detach().clone())),
            model._cross_attention_layer("0").register_forward_pre_hook(
                lambda _module, inputs: cross_inputs.append(inputs[0].detach().clone())),
            model.encoder.register_forward_pre_hook(
                lambda _module, inputs: encoded_ids.append(inputs[0].clone())),
        ]
        if before_callback is not None:
            model._post_cross_attention_eval_callback = callback
        try:
            with monkeypatch.context() as patch:
                patch.setattr(inference, "sample_next_token", sample)
                patch.setattr(model, "forward", forward)
                patch.setattr(upstream_methods, "build_self_conditioning_config", config)
                result = real_generate(cell, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            model._post_cross_attention_eval_callback = before_callback
        assert len(samples) == 5 and result == run.tokenizer.decode(trajectory[:-1])
        assert len(raw_blocks) == len(cross_inputs) == len(forwards)
        fit_name = {"EncDecDiffMean": "diffmean", "EncDecLinearProbe": "probe"}.get(cell.method)
        for raw, cross in zip(raw_blocks, cross_inputs, strict=True):
            expected = raw
            if fit_name is not None:
                fit = run.source.fitted[0][fit_name]
                expected = raw + (2.0 * fit.max_activation) * fit.direction.to(raw)
            t.testing.assert_close(cross, F.rms_norm(expected, (expected.size(-1),)), rtol=0, atol=0)
        prompt_features = {4, 5} if cell.method == "EncDecPromptSteering" else {1, 2}
        assert set(encoded_ids[0].flatten().tolist()) == prompt_features
        assert forwards[0][2]  # Real prompt FP read seeds generated features.
        slots = [int(item[1].flatten()[-1]) for item in forwards]
        if mode is SegmentMode.turn_delimited_fixed_tokens:
            assert slots == [0, 1, 1, 2, 2, 3]
        else:
            assert slots == [0, 1, 1, 1, 1, 1]
        assert model._decoder_blocks[0].attn.c_proj is projection
        generations.append((cell.method, logits, shifts))
        return "different zero replay" if corruption == "zero" and len(generations) == 2 else result

    monkeypatch.setattr(campaign, "_generate_encdec_cell", generate)
    if corruption is not None:
        reason = "positive-only zero" if corruption == "zero" else "same-runtime archived positive"
        with pytest.raises(AxBenchAdvError, match=f"{reason} replay differs"):
            campaign.generate(run.cfg, run.source, run.counter, tmp_path, run.identity)
        assert len(generations) == (2 if corruption == "zero" else 1)
        control_path = tmp_path / "generations/EncDecPromptSteering/c00/i00/shared_zero/m0.0.json"
        assert set(tmp_path.rglob("*.json")) == {control_path}
        assert read_json(control_path)["generation"] == "hijk"
    else:
        result = campaign.generate(run.cfg, run.source, run.counter, tmp_path, run.identity)
        assert result["positive_cells"] == 4 and result["counter_cells"] == 8
        assert len(run.preparations) == 20  # All ten source + ten steered contexts.
        assert len(generations) == 16
        for method in ENCDEC_BASELINE_METHODS:
            control, replay, diffmean, probe = [entry for entry in generations if entry[0] == method]
            assert not control[2] and not replay[2]
            for left, right in zip(control[1], replay[1], strict=True):
                t.testing.assert_close(left, right, rtol=0, atol=0)
            for attack, family in ((diffmean, "diffmean"), (probe, "linear_probe")):
                assert not t.equal(attack[1][0], control[1][0])
                direction = run.counter.artifacts[(family, 0)].loaded.direction
                for before, after in attack[2]:
                    t.testing.assert_close(after, before - 256.0 * direction.to(before), rtol=0, atol=0)
        records = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
        assert set(records) == {
            tmp_path / "generations" / method / "c00/i00" / point
            for method in ENCDEC_BASELINE_METHODS
            for point in ("shared_zero/m0.0.json", "diffmean_counter/m256.0.json",
                          "linear_probe_counter/m256.0.json")
        }
        # Completed cells are read, with only the four required fresh zero replays.
        campaign.generate(run.cfg, run.source, run.counter, tmp_path, run.identity)
        assert len(generations) == 20
        assert records == {path: path.read_bytes() for path in records}
        assert all(read_json(path)["generation"] == "hijk" for path in records)
    assert model._post_cross_attention_eval_callback is None
    assert not model._decoder_blocks[0]._forward_hooks
    assert model._decoder_blocks[0].attn.c_proj is projection
    for key, value in model.state_dict().items():
        assert t.equal(value, original_state[key]), key


@pytest.mark.parametrize("method", ENCDEC_BASELINE_METHODS)
def test_generation_failure_removes_both_counter_and_positive_state(
    tiny_campaign: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    run = tiny_campaign
    original = {key: value.clone() for key, value in run.model.state_dict().items()}
    projection = run.model._decoder_blocks[0].attn.c_proj
    real_generate = campaign._generate_encdec_cell
    active = None

    def generate(cell: Any, **kwargs: Any) -> str:
        nonlocal active
        active = cell.method
        return real_generate(cell, **kwargs)

    def sample(*_args: Any) -> t.Tensor:
        if active == method and run.model._post_cross_attention_eval_callback is not None:
            raise RuntimeError("decode interrupted with both interventions active")
        return t.tensor([[run.tokenizer.encode_special("<assistant_end>")]])

    monkeypatch.setattr(campaign, "_generate_encdec_cell", generate)
    monkeypatch.setattr(inference, "sample_next_token", sample)
    with pytest.raises(RuntimeError, match="decode interrupted"):
        campaign.generate(run.cfg, run.source, run.counter, tmp_path, run.identity)
    assert run.model._post_cross_attention_eval_callback is None
    assert run.model._decoder_blocks[0].attn.c_proj is projection
    assert not run.model._decoder_blocks[0]._forward_hooks
    for key, value in run.model.state_dict().items():
        assert t.equal(value, original[key]), key
    assert not campaign._record_path(tmp_path, method, 0, 0, "diffmean_counter", 256.0).exists()


def test_real_token_sampling_replays_after_global_rng_changes(
    tiny_campaign: Any, tmp_path: Path,
) -> None:
    """Use the real sampler and full 128-token budget, including natural stops."""
    run = tiny_campaign
    first, second = tmp_path / "first", tmp_path / "second"
    campaign.generate(run.cfg, run.source, run.counter, first, run.identity)
    # LoRA attachment consumes global randomness; feature and token sampling
    # must still restart from their cell seeds on a fresh run or shard resume.
    with t.random.fork_rng(devices=[]):
        t.manual_seed(9999)
        t.rand(1000)
        campaign.generate(run.cfg, run.source, run.counter, second, run.identity)
    for method in ENCDEC_BASELINE_METHODS:
        control = read_json(campaign._record_path(first, method, 0, 0, "shared_zero", 0.0))["generation"]
        attacks = [read_json(campaign._record_path(first, method, 0, 0, pressure, 256.0))["generation"]
                   for pressure in ("diffmean_counter", "linear_probe_counter")]
        assert all(attack != control for attack in attacks)
    first_records = {path.relative_to(first): path.read_bytes() for path in first.rglob("*.json")}
    assert len(first_records) == 12
    assert first_records == {path.relative_to(second): path.read_bytes() for path in second.rglob("*.json")}


@pytest.mark.parametrize("failure_index", [1, 11])
def test_extraction_error_releases_cuda_cache_before_any_target_load(
    tiny_campaign: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int,
) -> None:
    run = tiny_campaign
    run.cfg.device = "cuda:1"
    prepare = producer.prepare_prompt_context
    calls, released = [], []

    def failing_prepare(*args: Any, **kwargs: Any) -> PreparedPromptContext:
        calls.append(kwargs["token_ids"])
        if len(calls) == failure_index:
            raise RuntimeError("SAE extraction interrupted")
        return prepare(*args, **kwargs)

    monkeypatch.setattr(producer, "prepare_prompt_context", failing_prepare)
    monkeypatch.setattr(t.cuda, "empty_cache", lambda: released.append(True))
    monkeypatch.setattr(campaign, "load_target_model", lambda *_: pytest.fail("loaded after extraction error"))
    with pytest.raises(RuntimeError, match="SAE extraction interrupted"):
        campaign.generate(run.cfg, run.source, run.counter, tmp_path, run.identity)
    assert released == [True]
    assert len(calls) == failure_index
    assert not (tmp_path / "generations").exists()


def test_cuda_runtime_requires_real_availability_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(campaign, "_runtime_identity", lambda _: pytest.fail("probed unavailable CUDA"))
    with pytest.raises(AxBenchAdvError, match="CUDA requested but unavailable"):
        campaign._runtime("cuda:1")
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(campaign, "_runtime_identity", lambda _: {"gpu_name": "unavailable-test-double"})
    with pytest.raises(AxBenchAdvError, match="CUDA runtime identity is unavailable"):
        campaign._runtime("cuda:1")
    calls = []
    identity = {"gpu_name": "test hardware", "device_type": "cuda"}

    def runtime(device: str) -> dict[str, Any]:
        calls.append(device)
        return identity

    monkeypatch.setattr(campaign, "_runtime_identity", runtime)
    actual = campaign._runtime("cuda:1")
    assert actual == identity and actual is not identity
    assert calls == ["cuda:1"]
