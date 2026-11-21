"""CPU-only campaign preflight, turn routing, response resume, and judge boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.eval import campaign2 as runner
from scripts.utils import campaign2_eval_view as view
from scripts.utils.campaign2 import install_json_once, load_strict_json_object
from steering.eval import campaign2 as contracts
from steering.eval.chat_eval_result_identity import semantic_digest
from steering.eval.harmfulness_generation import HarmfulnessResponseIdentityError
from steering.eval.harmfulness_self_conditioning import harmfulness_prompt_seeds
from steering.immutable_io import ImmutableArtifactError
from steering.segmentation import SegmentMode
from tests.scripts.campaign2_fixtures import campaign_config as config


def test_missing_safety_target_and_judge_permission_fail_loud(tmp_path: Path) -> None:
    cfg = config(tmp_path, variant="encdec_safety")
    with pytest.raises(ValueError, match="targets must be a nonblank path"):
        contracts.preflight(cfg)
    with pytest.raises(ValueError, match="allow_paid_judge=true"):
        contracts.preflight(config(tmp_path, mode="judge"))


def test_prompt_records_require_exact_coverage_and_identity(tmp_path: Path) -> None:
    rows = [{"prompt": "Question", "pool": "benign", "category": "test"}]
    path = tmp_path / "records/000000.json"
    install_json_once(path, {**rows[0], "response": "Answer"})
    assert runner.collect_records(tmp_path, rows)[0]["response"] == "Answer"
    install_json_once(path, {**rows[0], "response": "Answer"})
    with pytest.raises(ImmutableArtifactError, match="artifact.*differs"):
        install_json_once(path, {**rows[0], "response": "Changed"})
    with pytest.raises(ValueError, match="prompt identity mismatch") as identity_error:
        runner.collect_records(tmp_path, [{**rows[0], "prompt": "different"}])
    assert isinstance(identity_error.value.__cause__, HarmfulnessResponseIdentityError)
    install_json_once(tmp_path / "records/000001.json", {**rows[0], "response": "Extra"})
    with pytest.raises(ValueError, match="coverage differs"):
        runner.collect_records(tmp_path, rows)


@pytest.mark.parametrize("body", ['{"key": 1, "key": 2}', '{"key": NaN}'])
def test_campaign_json_loader_rejects_ambiguous_values(tmp_path: Path, body: str) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(body)
    with pytest.raises(ValueError):
        load_strict_json_object(artifact)


def test_encdec_reuses_turn_ownership_and_zero_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    setup = SimpleNamespace(n_sae_features=4, feature_vocab=None, output_negative_selection=None)
    model = SimpleNamespace(parameters=lambda: iter([t.zeros(1)]), dec_config=SimpleNamespace(sequence_len=2048))
    tokenizer = SimpleNamespace(get_bos_token_id=lambda: 0, decode=lambda ids: "answer")
    context = SimpleNamespace(token_ids=(0, 3, 4), segments=("segment",), assistant_end_id=9,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens, token_assignments=(-1, 0, 0))
    monkeypatch.setattr(runner, "load_method_setup", lambda *args: setup)
    monkeypatch.setattr(runner, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(runner, "load_extraction_stack", lambda *args: ("model", "sae"))
    monkeypatch.setattr(runner, "render_prompt_for_chat_completion", lambda *args: context)
    monkeypatch.setattr(runner, "prepare_prompt_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(runner, "load_target_model", lambda *args: (model, tokenizer))
    monkeypatch.setattr(runner, "SelfCondScoreInterpolation", lambda **kwargs: calls.append(kwargs) or kwargs)
    monkeypatch.setattr(runner, "build_self_conditioning_config", lambda *args, **kwargs: "config")
    monkeypatch.setattr(runner, "SingleEntryContextProvider", lambda *args, **kwargs: calls.append(kwargs) or "provider")
    monkeypatch.setattr(runner, "build_engine_self_conditioned", lambda *args, **kwargs:
        SimpleNamespace(generate_stream=lambda *args, **kwargs: iter([7])))
    row = {"prompt": "Q", "pool": "benign", "category": "test"}
    cfg = config(tmp_path, device="cpu", step=7, statistics="/statistics")
    runner.generate_encdec(cfg, [(0, row)], tmp_path)
    assert float(calls[0]["weight"]) == 0 and t.equal(calls[0]["target_scores"], t.zeros(4))
    assert calls[1] == {"segment_mode": SegmentMode.turn_delimited_fixed_tokens, "token_assignments": (-1, 0, 0)}
    assert runner.collect_records(tmp_path, [row])[0]["response"] == "answer"


def test_generation_resume_skips_completed_rows_and_never_judges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"prompt": "Q0", "pool": "harmful", "category": "test"},
            {"prompt": "Q1", "pool": "benign", "category": "test"}]
    cfg = config(tmp_path, variant="pythia", mode="generate")
    output = Path(cfg.output_dir)
    install_json_once(output / "records/000000.json", {**rows[0], "response": "prior"})
    monkeypatch.setattr(runner, "preflight", lambda cfg: {"identity": "fixed"})
    monkeypatch.setattr(runner, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=rows))
    seen = []

    def generate(cfg: Any, pending: list, output: Path) -> None:
        seen.extend(index for index, _ in pending)
        for index, row in pending:
            install_json_once(output / "records" / f"{index:06d}.json", {**row, "response": "new"})

    monkeypatch.setattr(runner, "generate_pythia", generate)
    monkeypatch.setattr(runner, "judge", lambda *args: pytest.fail("generate called paid judge"))
    runner.main.__wrapped__(cfg)
    runner.main.__wrapped__(cfg)
    assert seen == [1]


def test_eval_view_measures_cadence_without_mutating_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Store(dict):
        segment_mode = SegmentMode.turn_delimited_fixed_tokens

    metadata = tmp_path / "meta_000007.json"
    original = {"step": 7, "dataset": "safety", "sft_loss_mask": "completion_only",
        "sft_segment_mode": "turn_delimited_fixed_tokens", "model_config": {"sequence_len": 2048},
        "user_config": {"segment_size": 32, "min_loss_tokens": 8, "feature_source_bos_char_len": 5}}
    metadata.write_text(json.dumps(original))
    (tmp_path / "model_000007.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(view, "inventory", lambda *args: 1)
    monkeypatch.setattr(view, "load_sft_features", lambda *args: Store())
    monkeypatch.setattr(view, "build_task_mixtures", lambda *args, **kwargs: ("task", None))
    monkeypatch.setattr(view, "_assert_sft_mixture_fingerprint_matches_task", lambda **kwargs: None)
    monkeypatch.setattr(view, "_run_sft_content_alignment_probe", lambda **kwargs: None)
    monkeypatch.setattr(view, "assert_sft_source_bos_char_len_matches", lambda *args, **kwargs: None)
    monkeypatch.setattr(view, "get_tokenizer", lambda: "tokenizer")
    monkeypatch.setattr(view, "measure_turn_self_cond_segment_size_rustbpe", lambda **kwargs:
        SimpleNamespace(cadence=33, documents=10, windows=30, mean=33, minimum=29, maximum=38))
    output = tmp_path / "eval-view"
    view.main.__wrapped__(OmegaConf.create({"metadata": str(metadata), "feature_dir": "/features",
        "parquet_dir": "/parquet", "steered_jsonl": None, "output_dir": str(output)}))
    assert json.loads(metadata.read_text()) == original
    assert json.loads((output / metadata.name).read_text())["turn_self_cond_segment_size_rustbpe"] == 33
    assert (output / "model_000007.pt").resolve() == tmp_path / "model_000007.pt"


def safety_identity(cfg: Any, rows: list[dict]) -> dict:
    return {"config": OmegaConf.to_container(cfg, resolve=True), "safety_binding": {
        "prompts_sha256": semantic_digest(rows),
        "prompt_seeds": [list(harmfulness_prompt_seeds(base_seed=cfg.seed,
            stable_pool_index=index, prompt=row["prompt"], pool=row["pool"], category=row["category"]))
            for index, row in enumerate(rows)]}}


@pytest.mark.parametrize("strength", [0, 3, 6])
def test_safety_routes_to_encdec_resumes_original_indexes_and_binds_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strength: int,
) -> None:
    rows = [{"prompt": f"Q{index}", "pool": "harmful", "category": "test"} for index in range(3)]
    cfg = config(tmp_path, variant="encdec_safety", mode="generate", safety={"strength": strength})
    identity = safety_identity(cfg, rows)
    output = contracts.cell_output_dir(cfg)
    assert output == Path(cfg.output_dir) / f"strength-{strength}"
    prior = {**rows[1], "response": "prior"}
    install_json_once(output / "records/000001.json",
        {**prior, "generation": contracts.safety_record_receipt(identity, 1, prior)})
    monkeypatch.setattr(runner, "preflight", lambda cfg: identity)
    monkeypatch.setattr(runner, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=rows))
    seen = []

    def generate(cfg: Any, pending: list, output: Path) -> None:
        seen.extend(index for index, _ in pending)
        for index, row in pending:
            record = {**row, "response": "new"}
            install_json_once(output / "records" / f"{index:06d}.json",
                {**record, "generation": contracts.safety_record_receipt(identity, index, record)})

    monkeypatch.setattr(runner, "generate_encdec", generate)
    monkeypatch.setattr(runner, "generate_axbench", lambda *args: pytest.fail("safety dispatched to AxBench"))
    monkeypatch.setattr(runner, "judge", lambda *args: pytest.fail("generation called paid judge"))
    runner.main.__wrapped__(cfg)
    runner.main.__wrapped__(cfg)
    assert seen == [0, 2]
    records = runner.collect_records(output, rows, identity)
    assert records[1] == prior
    assert json.loads((output / "responses.json").read_text()) == {
        "cell_sha256": semantic_digest(identity), "records_sha256": semantic_digest(records)}


@pytest.mark.parametrize("fault", ["text", "prompt", "seed", "cell", "index", "missing", "extra_record", "pool"])
def test_safety_refuses_stale_partial_records_before_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    rows = [{"prompt": f"Q{index}", "pool": "harmful", "category": "test"} for index in range(2)]
    cfg = config(tmp_path, variant="encdec_safety", mode="generate")
    identity = safety_identity(cfg, rows)
    record = {**rows[0], "response": "prior"}
    receipt = contracts.safety_record_receipt(identity, 0, record)
    record["generation"] = receipt
    if fault == "text":
        record["response"] = "tampered"
    elif fault == "prompt":
        record["prompt"] = "other"
    elif fault == "seed":
        receipt["prompt_seeds"] = [1, 2]
    elif fault == "cell":
        receipt["cell_sha256"] = "0" * 64
    elif fault == "index":
        receipt["stable_pool_index"] = 1
    elif fault == "missing":
        record.pop("generation")
    elif fault == "pool":
        identity["safety_binding"]["prompts_sha256"] = "0" * 64
    output = contracts.cell_output_dir(cfg)
    install_json_once(output / "records" / ("000002.json" if fault == "extra_record" else "000000.json"), record)
    monkeypatch.setattr(runner, "preflight", lambda cfg: identity)
    monkeypatch.setattr(runner, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=rows))
    monkeypatch.setattr(runner, "generate_encdec", lambda *args: pytest.fail("forwarded before validating partial records"))
    with pytest.raises(ValueError, match="safety"):
        runner.main.__wrapped__(cfg)


def test_safety_generate_uses_shared_runtime_and_original_pending_index(tmp_path, monkeypatch):
    @dataclass(frozen=True)
    class Runtime:
        model: Any
        device: t.device

    rows = [{"prompt": f"Q{index}", "pool": "harmful", "category": "test"} for index in range(3)]
    cfg = config(tmp_path, variant="encdec_safety", device="cpu", step=7, statistics="/statistics",
        safety={"targets": "/targets", "strength": 3})
    identity = safety_identity(cfg, rows)
    install_json_once(tmp_path / "identity.json", identity)
    calls, checks = [], []
    runtime = Runtime(t.nn.Linear(2, 2), t.device("cpu"))
    cell = SimpleNamespace(runtime=runtime, rows=rows, assert_stable=lambda: checks.append(True))
    monkeypatch.setattr(runner, "load_method_setup", lambda *args: "setup")
    monkeypatch.setattr(runner, "prepare_safety_cell", lambda cfg, setup: cell)
    monkeypatch.setattr(runner, "preflight", lambda cfg, **kwargs: identity)
    monkeypatch.setattr(runner, "load_extraction_stack", lambda *args: pytest.fail("safety ran extraction"))
    monkeypatch.setattr(runner, "load_target_model", lambda *args: pytest.fail("safety used legacy loader"))
    monkeypatch.setattr(runner, "generate_harmfulness_response_self_conditioned",
        lambda **kwargs: calls.append(kwargs) or "shared response")
    runner.generate_encdec(cfg, [(2, rows[2])], tmp_path)
    assert calls[0]["stable_pool_index"] == 2 and calls[0]["base_seed"] == cfg.seed
    assert calls[0]["setup"].device.type == "cpu"
    assert len(checks) == 3
    record = json.loads((tmp_path / "records/000002.json").read_text())
    assert contracts.validate_safety_record(record, rows[2], 2, identity)["response"] == "shared response"


@pytest.mark.parametrize("fault", [None, "target", "prompts", "record", "aggregate", "orphan"])
def test_safety_preflight_validates_existing_cells_without_writing(tmp_path, monkeypatch, fault):
    rows = [{"prompt": f"Q{index}", "pool": "harmful", "category": "test"} for index in range(2)]
    cfg = config(tmp_path, variant="encdec_safety", mode="preflight")
    identity = safety_identity(cfg, rows)
    output = contracts.cell_output_dir(cfg)
    prior = {**rows[0], "response": "prior"}
    record = {**prior, "generation": contracts.safety_record_receipt(identity, 0, prior)}
    if fault == "record":
        record["response"] = "changed"
    if fault != "orphan":
        install_json_once(output / "identity.json", identity)
    install_json_once(output / "prompts.json", rows if fault != "prompts" else [])
    install_json_once(output / "records/000000.json", record)
    if fault == "aggregate":
        install_json_once(output / "responses.json", {"records_sha256": "bad"})
    if fault == "target":
        identity["safety_binding"]["targets_sha256"] = "changed"
    before = {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(runner, "preflight", lambda cfg: identity)
    monkeypatch.setattr(runner, "resolve_harmfulness_prompt_pool", lambda **kwargs: SimpleNamespace(rows=rows))
    monkeypatch.setattr(runner, "generate_encdec", lambda *args: pytest.fail("preflight forwarded"))
    monkeypatch.setattr(runner, "judge", lambda *args: pytest.fail("preflight judged"))
    if fault is None:
        runner.main.__wrapped__(cfg)
    else:
        with pytest.raises(ValueError, match="safety|coverage"):
            runner.main.__wrapped__(cfg)
    assert before == {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}
