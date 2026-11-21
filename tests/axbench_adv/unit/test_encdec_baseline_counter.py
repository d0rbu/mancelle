"""CPU contracts for the explicit encoder-decoder baseline counter lane."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from nanochat.common import COMPUTE_DTYPE
from steering.conditional.encoder_decoder.eval_residual import post_cross_attention_residual_counter
from steering.eval.axbench_adv import encdec_baseline_counter as campaign
from steering.eval.axbench_adv import final_judge as final_judge_module
from steering.eval.axbench_adv.encdec_sources import EncDecSources
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.native_mailbox import digest_json, read_json, write_json
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from steering.eval.axbench_upstream.encdec_baselines import run as producer
from steering.eval.axbench_upstream.encdec_baselines.generation import steering_delta
from tests.axbench_adv.unit.test_direction_fitting import _publication
from tests.axbench_adv.unit.test_final_judge import _outputs


def _source() -> EncDecSources:
    rows = [
        {
            "dataset_name": "AlpacaEval",
            "concept_id": 0,
            "input_id": input_id,
            "factor": factor,
            "original_prompt": f"prompt-{input_id}",
            "input_concept": "concept",
            "steered_input": "",
            "input": "",
            "suppress_original": "",
            "suppress_rewrite": "",
            "steered_prompt": "",
            "defense": "",
            "sae_link": "",
            "sae_id": 0,
        }
        for input_id in range(10)
        for factor in (1.0, 2.0)
    ]
    frame = pd.DataFrame(rows)
    return cast(
        EncDecSources,
        SimpleNamespace(
            source_main=frame,
            best_factors={
                method: 1.0 if method in {"EncDecPromptSteering", "EncDecLoRA"} else 2.0
                for method in ENCDEC_BASELINE_METHODS
            },
            steering_prompts={0: "steer"},
            setup=object(),
            manifest={
                "layer_resolution": {"resolved_lora_layers": [0], "resolved_intervention_layer": 0}
            },
            identity={"runtime": {"python": "test"}},
            fitted={0: {}},
            lora_states={0: {}},
        ),
    )


class _Block(nn.Module):
    def forward(self, value: t.Tensor) -> t.Tensor:
        return value


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict({"h": nn.ModuleList([_Block()])})
        self.weight = nn.Parameter(t.zeros(1))
        self.dec_config = SimpleNamespace(n_embd=4, n_layer=1, sequence_len=256)
        self._post_cross_attention_eval_callback = None
        self.eval()

    def get_device(self) -> t.device:
        return self.weight.device

    def forward(self, value: t.Tensor) -> t.Tensor:
        transformer = self.transformer
        blocks = cast(nn.ModuleList, transformer["h"])
        residual = blocks[0](value)
        callback = self._post_cross_attention_eval_callback
        return residual if callback is None else callback(0, residual)


def test_positive_and_raw_counter_compose_at_distinct_sites() -> None:
    model = _Model()
    positive = t.tensor([1.0, 0.0, 0.0, 0.0])
    counter = t.tensor([0.0, 1.0, 0.0, 0.0])
    value = t.ones((1, 1, 4), dtype=COMPUTE_DTYPE)
    with steering_delta(model, direction=positive, coefficient=2.0 * 3.0, layer=0):
        with post_cross_attention_residual_counter(
            model, layer=0, direction=counter, magnitude=256.0
        ):
            result = model(value)
    assert result[0, 0].tolist() == [7.0, -255.0, 1.0, 1.0]
    assert model._post_cross_attention_eval_callback is None
    assert model(value).equal(value)
    with post_cross_attention_residual_counter(model, layer=0, direction=counter, magnitude=0.0):
        assert model(value).equal(value)


@pytest.fixture
def artifact_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Use synthetic generations and providers with real artifact validation and merge."""
    # Exercise real atomic installation; physical flush durability has its own tests.
    monkeypatch.setattr("steering.immutable_io.os.fsync", lambda _fd: None)
    # Artifact fixtures should not depend on unrelated checkout edits. The
    # provenance regression separately exercises real commits and dirty files.
    monkeypatch.setattr(campaign, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "_repository_source_sha256", lambda: "b" * 64)
    source = _source()
    source = cast(
        EncDecSources,
        SimpleNamespace(
            **{
                **vars(source),
                "source_main": pd.concat(
                    [source.source_main.assign(concept_id=cid) for cid in range(50)],
                    ignore_index=True,
                ),
                "identity": {"inputs": {"target_model": {"sha256": "a" * 64}}},
                "source_digest": {"manifest": "b" * 64},
            }
        ),
    )
    factors = dict(source.best_factors)
    counter = SimpleNamespace(
        manifest_sha256="c" * 64,
        artifacts={
            (family, cid): SimpleNamespace(sha256=f"{family}-{cid}")
            for family in ("diffmean", "linear_probe")
            for cid in range(50)
        },
    )
    cache = tmp_path / "cache.sqlite"
    cache.touch()
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "counter_layer": 9,
            "counter_fit_seed": 42,
            "cache_only": True,
            "judge_cache": str(cache),
            "api_base_url": "https://unused.invalid",
        }
    )
    calls = []

    def raw(generation: str) -> RawEvaluation:
        return RawEvaluation(
            generation,
            0.75,
            1.5,
            '{"concept":"concept Rating: [[0.75]]","fluency":"fluency Rating: [[1.5]]"}',
            0,
        )

    class Judge:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def rebind(self, _payload: Any) -> None:
            pass

        def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
            calls.append(("inner", input_id))
            return raw(generation)

        def close(self) -> None:
            pass

    class FinalProcess:
        def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert operation == "evaluate"
            calls.append(("final", payload["output_dir"]))
            table = pq.read_table(Path(payload["input_dir"]) / "steering_data.parquet")
            emissions = {
                (method, row["concept_id"], row["input_id"]): raw(
                    row[f"{method}_steered_generation"]
                )
                for row in table.to_pylist()
                for method, factor in factors.items()
                if row["factor"] == factor
            }
            concepts = tuple(sorted(set(table["concept_id"].to_pylist())))
            output = Path(payload["output_dir"])
            _outputs(output, table, emissions, factors=factors, concept_ids=concepts)
            count = len(emissions) * 3
            write_json(
                output / "judge_call_accounting.json",
                {"total_calls": count, "total_cache_hits": count},
            )
            (output / "plot.pdf").write_bytes(b"a valid extra vendor output")
            return {}

    def final(**kwargs: Any) -> Any:
        return final_judge_module.run_final_judge(**kwargs, _process=cast(Any, FinalProcess()))

    def point(**kwargs: Any) -> Any:
        return final_judge_module.judge_generated_point(**kwargs, judge=final)

    monkeypatch.setattr(campaign, "AxBenchAdvJudgeClient", Judge)
    monkeypatch.setattr(campaign, "judge_generated_point", point)

    def finish(concepts: list[int], name: str) -> tuple[Path, dict[str, Any]]:
        output = tmp_path / name
        identity = campaign._identity(
            cfg=cfg,
            sources=source,
            counter=counter,
            concepts=concepts,
            magnitudes=[256.0],
            phase="generate",
            runtime={"device_type": "cpu"},
        )
        campaign._install_identity(output, identity)
        _, cells = campaign._cells(source, concepts, pilot=False)
        for cell in cells:
            for pressure, magnitude in campaign._point_coordinates(identity):
                campaign._record(
                    output=output,
                    identity_sha=digest_json(identity),
                    cell=cell,
                    pressure=pressure,
                    magnitude=magnitude,
                    generate=lambda cell=cell, pressure=pressure, magnitude=magnitude: (
                        f"{cell.method} generation c{cell.concept_id} i{cell.input_id} {pressure} {magnitude}"
                    ),
                )
        campaign.judge(cfg, source, output, identity)
        campaign.finalize(cfg, source, counter, output, identity)
        return output, identity

    return SimpleNamespace(source=source, counter=counter, cfg=cfg, finish=finish, calls=calls)


def test_artifact_lifecycle_resume_and_checked_c50_merge(
    artifact_campaign: Any, tmp_path: Path
) -> None:
    run = artifact_campaign
    left, identity = run.finish(list(range(25)), "left")
    right, _ = run.finish(list(range(25, 50)), "right")
    before = len(run.calls)
    campaign.judge(run.cfg, run.source, left, identity)
    campaign.finalize(run.cfg, run.source, run.counter, left, identity)
    assert len(run.calls) == before
    for roots in ([left], [left, left]):
        run.cfg.shard_roots = [str(root) for root in roots]
        with pytest.raises(AxBenchAdvError, match="cover concepts 0..49 exactly once"):
            campaign.merge(run.cfg, run.source, run.counter)
    run.cfg.shard_roots = [str(right), str(left)]
    run.cfg.output_dir = str(tmp_path / "merged")
    manifest = campaign.merge(run.cfg, run.source, run.counter)
    assert manifest["concept_ids"] == list(range(50))
    scores = pq.read_table(tmp_path / "merged/aggregate.parquet").to_pandas()
    assert len(scores) == 12
    assert set(scores["method"]) == set(ENCDEC_BASELINE_METHODS)
    assert set(scores["pressure"]) == {"shared_zero", "diffmean_counter", "linear_probe_counter"}
    assert set(scores["n_inputs"]) == {500}
    assert set(scores["concept_score"]) == {0.75}
    assert set(scores["fluency_score"]) == {1.5}
    assert set(scores["instruction_score"]) == {1.25}
    assert set(scores["harmonic_quality_2"]) == {1.0}
    assert len(pq.read_table(tmp_path / "merged/per_concept.parquet")) == 600


def test_interrupted_atomic_writes_are_not_committed_cells(
    artifact_campaign: Any, tmp_path: Path,
) -> None:
    run = artifact_campaign
    root = tmp_path / "shard"
    root.mkdir()
    (root / ".identity.json.tmp.123.abcdef01").write_text("interrupted identity")
    _, identity = run.finish([0], "shard")
    for tree in ("generations", "judgments"):
        record = next((root / tree).rglob("*.json"))
        record.with_name(f".{record.name}.tmp.123.abcdef01").write_text("interrupted record")
    before = len(run.calls)
    campaign.judge(run.cfg, run.source, root, identity)
    campaign.finalize(run.cfg, run.source, run.counter, root, identity)
    assert len(run.calls) == before
    next((root / "generations").rglob("*.json")).unlink()
    with pytest.raises(AxBenchAdvError, match="missing generation record"):
        campaign.judge(run.cfg, run.source, root, identity)


@pytest.mark.parametrize(
    "mutation", ["generation", "judgment", "extra", "missing", "identity", "score", "manifest"]
)
def test_finished_shard_rejects_mutation(artifact_campaign: Any, mutation: str) -> None:
    run = artifact_campaign
    root, identity = run.finish([0], "shard")
    if mutation in {"generation", "judgment"}:
        tree = "generations" if mutation == "generation" else "judgments"
        path = next((root / tree).rglob("*.json"))
        value = read_json(path)
        if mutation == "generation":
            value["generation"] = "corrupted generation"
        else:
            value["key"]["input_id"] = 999
        write_json(path, value)
    elif mutation == "extra":
        (root / "generations/extra.txt").write_text("unexpected")
    elif mutation == "missing":
        next((root / "generations").rglob("*.json")).unlink()
    elif mutation == "identity":
        identity["generation"]["seed"] = 43
        write_json(root / "identity.json", identity)
    elif mutation == "score":
        table = pq.read_table(root / "aggregate.parquet").to_pandas()
        table.loc[0, "concept_score"] = 0.5
        table.to_parquet(root / "aggregate.parquet", index=False)
    else:
        value = read_json(root / "manifest.json")
        value["methods"] = []
        write_json(root / "manifest.json", value)
    with pytest.raises(AxBenchAdvError):
        campaign.finalize(run.cfg, run.source, run.counter, root, identity)


def test_method_cell_selection_and_subset_coverage() -> None:
    source = _source()
    frame, pilot = campaign._cells(source, (0,), pilot=True)
    assert len(frame) == 20
    assert {(cell.method, cell.input_id) for cell in pilot} == {
        (method, 0) for method in ENCDEC_BASELINE_METHODS
    }
    assert {
        cell.factor_hex for cell in pilot if cell.method in {"EncDecDiffMean", "EncDecLinearProbe"}
    } == {(2.0).hex()}
    _, full = campaign._cells(source, (0,), pilot=False)
    assert len(full) == 40
    with pytest.raises(AxBenchAdvError, match="exact concept/input product"):
        campaign._cells(
            cast(
                EncDecSources,
                SimpleNamespace(**{**vars(source), "source_main": source.source_main.iloc[:-1]}),
            ),
            (0,),
            pilot=False,
        )
    with pytest.raises(AxBenchAdvError):
        campaign._concepts([0, 0])
    with pytest.raises(AxBenchAdvError):
        campaign._concepts([50])


def test_campaign_rejects_counter_fit_mismatch(tmp_path: Path) -> None:
    source = cast(
        EncDecSources,
        SimpleNamespace(
            **{
                **vars(_source()),
                "identity": {"inputs": {"target_model": {"sha256": "a" * 64}}},
                "setup": SimpleNamespace(meta={"model_config": {"n_embd": 4}}),
                "source_main": pd.DataFrame(
                    {"concept_id": range(50), "input_concept": ["concept"] * 50}
                ),
            }
        ),
    )
    root, _ = _publication(tmp_path)
    cfg = cast(
        DictConfig,
        SimpleNamespace(
            counter_layer=9,
            counter_fit_seed=42,
            counter_root=root,
            target_meta_path=tmp_path / "meta.json",
        ),
    )
    (tmp_path / "meta.json").write_text("{}")

    with pytest.raises(AxBenchAdvError, match="fit identity does not match"):
        campaign._counter_sources(cfg, source)
    cfg.counter_layer = 8
    with pytest.raises(AxBenchAdvError, match="counter layer must be 9"):
        campaign._counter_sources(cfg, source)


def test_generation_record_resume_rejects_tamper(tmp_path: Path) -> None:
    source = _source()
    _, cells = campaign._cells(source, (0,), pilot=True)
    cell = cells[0]

    def record(generate: Any) -> str:
        return campaign._record(
            output=tmp_path,
            identity_sha="a" * 64,
            cell=cell,
            pressure="shared_zero",
            magnitude=0.0,
            generate=generate,
        )

    assert record(lambda: "first") == "first"
    assert record(lambda: pytest.fail("resume regenerated")) == "first"
    path = campaign._record_path(
        tmp_path,
        method=cell.method,
        concept_id=0,
        input_id=0,
        pressure="shared_zero",
        magnitude=0.0,
    )
    saved = read_json(path)
    saved["generation"] = "tampered"
    write_json(path, saved)
    with pytest.raises(AxBenchAdvError, match="saved generation record differs"):
        record(lambda: "first")


def test_shards_preserve_producer_prompt_sampling_and_cell_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    source = cast(
        EncDecSources,
        SimpleNamespace(**{
            **vars(source),
            "source_main": pd.concat(
                [source.source_main.assign(concept_id=cid, original_prompt=f"prompt-{cid}") for cid in (0, 1)],
                ignore_index=True,
            ),
            "steering_prompts": {0: "steer", 1: "other steer"},
            "lora_states": {0: {}, 1: {}},
        }),
    )
    rng = t.Generator()
    model = _Model()
    monkeypatch.setattr(campaign, "get_tokenizer", lambda: object())
    monkeypatch.setattr(
        campaign, "render_prompt_for_chat_completion",
        lambda _, prompt: SimpleNamespace(token_ids=tuple(prompt.encode()), assistant_end_id=0),
    )
    # Keep the producer's actual prompt plans and preparation traversal. Stand in
    # for SAE work with stateful feature sampling, not a constant mock context.
    monkeypatch.setattr(
        producer, "prepare_prompt_context",
        lambda *_a, **_k: t.randint(0, 2**31, (4,), generator=rng).tolist(),
    )
    monkeypatch.setattr(campaign, "load_extraction_stack", lambda *_a: (None, None))
    monkeypatch.setattr(campaign, "load_target_model", lambda *_a: (model, None))
    monkeypatch.setattr(campaign, "_validate_all", lambda *_a, **_k: None)
    monkeypatch.setattr(campaign, "validate_lora_state", lambda *_a, **_k: None)
    monkeypatch.setattr(campaign, "_reset_compiler_state_before_generation", lambda: None)

    def generate(cell: Any, **kwargs: Any) -> str:
        cell.validated_source_values(kwargs["rows"])
        prefix = "steered_" if cell.method == "EncDecPromptSteering" else ""
        plan = kwargs["steered_plan" if prefix else "prompt_plan"]
        tokens = plan.rendered_tokens_by_pair[(cell.concept_id, cell.input_id)]
        features = kwargs[f"{prefix}prepared_by_tokens"][tokens]
        return f"{features}:{cell.selection_seed}:{cell.token_seed}"

    monkeypatch.setattr(campaign, "_generate_encdec_cell", generate)
    counter = SimpleNamespace(artifacts={
        (family, cid): SimpleNamespace(loaded=SimpleNamespace(direction=t.ones(4)))
        for family in ("diffmean", "linear_probe") for cid in (0, 1)
    })
    cfg = cast(DictConfig, SimpleNamespace(device="cpu", counter_layer=0))
    for concepts, name in (([0, 1], "full"), ([0], "left"), ([1], "right")):
        rng.manual_seed(42)
        identity = {"phase": "generate", "concept_ids": concepts, "magnitudes": [256.0], "runtime": {"test": True}}
        campaign.generate(cfg, source, counter, tmp_path / name, identity)
    for name in ("left", "right"):
        for path in (tmp_path / name / "generations").rglob("*.json"):
            full = tmp_path / "full" / path.relative_to(tmp_path / name)
            assert read_json(path)["generation"] == read_json(full)["generation"]
