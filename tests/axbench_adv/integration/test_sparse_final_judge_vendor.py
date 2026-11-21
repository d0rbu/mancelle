"""Run the real vendored evaluator over one sparse all-method point frame."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_adv.pilot_budget import BUDGET_ENV, budget_summary, create_pilot_budget
from steering.eval.axbench_adv.encdec_sources import METHOD_FACTORS_ENCDEC
from steering.eval.axbench_adv.final_judge import (
    build_point_frame,
    run_final_judge,
    validate_final_judge_outputs,
    write_point_frame,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS, METHODS, NATIVE_METHOD
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient
from steering.eval.axbench_histograms import generate_histogram_reports
from steering.eval.axbench_upstream.native_contract import (
    FORMAT,
    POLICY,
    native_input_frame,
    sha256,
)
from steering.eval.axbench_upstream.native_judge import run_native_evaluation
from tests.scripts.eval.test_axbench_native_decoder import source_frame

FACTORS = {
    "DiffMean": 1.0,
    "LinearProbe": 3.0,
    "LoRA": 1.0,
    "PromptSteering": 1.0,
    "SelfConditioning": 6.0,
    "SelfConditioningMasked": 5.0,
}
COMPLETION = "Local integration judge rationale. Rating: [[1]]"


class _Handler(BaseHTTPRequestHandler):
    calls = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        assert self.path == "/v1/chat/completions"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "openai/gpt-5.6-luna"
        assert body["temperature"] == 0.0
        assert len(body["messages"]) == 1
        with self.lock:
            type(self).calls += 1
        response = json.dumps(
            {
                "id": "local-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": COMPLETION},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args
        return


class _Server:
    def __enter__(self) -> str:
        _Handler.calls = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host = self.server.server_address[0]
        port = self.server.server_address[1]
        return f"http://{host}:{port}/v1"

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _source(concept_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "AlpacaEval",
                "concept_id": concept_id,
                "input_concept": "target concept",
                "input_id": input_id,
                "factor": 1.0,
                "original_prompt": f"original prompt {input_id}",
            }
            for input_id in range(10)
        ]
    )


def _emissions(
    methods: tuple[str, ...] = METHODS, concept_id: int = 0,
) -> dict[tuple[str, int, int], RawEvaluation]:
    rationale = json.dumps(
        {"concept": COMPLETION, "fluency": COMPLETION},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        (method, concept_id, input_id): RawEvaluation(
            generation=f"{method} generated response {input_id}",
            concept_rating=1.0,
            fluency_rating=1.0,
            judge_rationale=rationale,
            paid_judge_cache_misses=0,
        )
        for method in methods
        for input_id in range(10)
    }


def test_real_native_upstream_judge_receipt_recovery_and_cache_replay(tmp_path, monkeypatch):
    frame = native_input_frame(source_frame(), max_concepts=1)
    frame[f"{NATIVE_METHOD}_steered_generation"] = [f"Rent a tool {i}." for i in range(10)]
    inputs = tmp_path / "input"
    inputs.mkdir()
    input_path = inputs / "steering_data.parquet"
    frame.to_parquet(input_path, index=False)
    manifest_path = inputs / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "format": FORMAT, "method": NATIVE_METHOD,
        "factors": [1.0], "complete": True, "policy": {"name": POLICY},
        "source": {"max_concepts": 1},
        "output": {"path": str(input_path), "sha256": sha256(input_path), "rows": 10},
    }))
    output = tmp_path / "judged"
    cfg = OmegaConf.create({
        "input_path": str(input_path), "manifest_path": str(manifest_path),
        "output_dir": str(output), "judge_cache": str(tmp_path / "judge.sqlite"),
        "max_concepts": 1, "num_workers": 1, "cache_only": False,
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-test-key")
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    with _Server() as api_base_url:
        cfg.api_base_url = api_base_url
        run_native_evaluation(cfg)
    assert _Handler.calls == 30
    receipt_path = output / "native_judge_receipt.json"
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    assert receipt["scores"] == dict.fromkeys(("concept", "instruction", "fluency", "aggregate"), 1.0)
    assert receipt["call_accounting"]["reported_noncache_calls"] == 30
    assert receipt["call_accounting"]["accounting_complete"] is True
    monkeypatch.delenv("OPENROUTER_API_KEY")
    run_native_evaluation(cfg)
    receipt_path.unlink()
    run_native_evaluation(cfg)
    assert receipt_path.read_bytes() == original_receipt
    cfg.output_dir = str(tmp_path / "replay")
    cfg.cache_only = True
    run_native_evaluation(cfg)
    replay = json.loads((Path(cfg.output_dir) / receipt_path.name).read_text())
    assert replay["call_accounting"]["reported_noncache_calls"] == 0
    assert replay["call_accounting"]["total_cache_hits"] == 30
    for name in ("steering.jsonl", "steering_data.parquet"):
        assert (Path(cfg.output_dir) / name).read_bytes() == (output / name).read_bytes()
    assert _Handler.calls == 30


@pytest.mark.parametrize("scope,concept_id", [
    ("baseline", 0), ("all", 0), ("nl_only", 0), ("baseline", 4), ("nl_only", 4),
    ("native", 0), ("native", 4), ("encdec", 4),
])
def test_real_vendored_evaluator_accepts_sparse_point_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    concept_id: int,
) -> None:
    methods = (
        ENCDEC_BASELINE_METHODS if scope == "encdec" else
        (NATIVE_METHOD,) if scope == "native" else ("NLSteering",) if scope == "nl_only"
        else (*METHODS, "NLSteering") if scope == "all" else METHODS
    )
    all_factors = {
        **FACTORS, "NLSteering": 2.0, NATIVE_METHOD: 1.0,
        "EncDecPromptSteering": 1.0, "EncDecLoRA": 1.0,
        "EncDecDiffMean": 2.0, "EncDecLinearProbe": 3.0,
    }
    factors = {method: all_factors[method] for method in methods}
    method_factors = METHOD_FACTORS_ENCDEC if scope == "encdec" else None
    root_method = methods[0] if len(methods) == 1 or scope == "encdec" else "SelfConditioning"
    concept_ids = (concept_id,)
    max_concepts = concept_id + 1
    emissions = _emissions(methods, concept_id)
    table = build_point_frame(
        source_frame=_source(concept_id),
        best_factors=factors,
        emissions=emissions,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
        method_factors=method_factors,
    )
    input_path = tmp_path / "input" / "steering_data.parquet"
    write_point_frame(
        table, input_path, best_factors=factors,
        max_concepts=max_concepts, concept_ids=concept_ids,
        method_factors=method_factors,
    )
    cache = tmp_path / "judge.sqlite"
    cache.touch()
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-test-key")
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    budget_path = tmp_path / "pilot.sqlite3"
    inner_methods = methods if scope == "encdec" else (root_method,)
    with _Server() as api_base_url:
        monkeypatch.setenv("OPENROUTER_BASE_URL", api_base_url)
        if scope == "nl_only":
            create_pilot_budget(
                budget_path, limit_usd="20", request_bound_usd="1",
                model="openai/gpt-5.6-luna", endpoint=f"{api_base_url}/chat/completions",
                max_output_tokens=128000, pricing_evidence="offline HTTP fixture: USD 0.001 per attempt",
            )
            monkeypatch.setenv(BUDGET_ENV, str(budget_path))
        init_payload = {
            "api_base_url": api_base_url,
            "cache_only": False,
            "concept": "target concept",
            "concept_id": concept_id,
            "factor": factors[root_method],
            "judge_cache": str(cache),
            "judge_model": "openai/gpt-5.6-luna",
            "method": root_method,
            "runtime_kind": "judge",
        }
        with AxBenchAdvJudgeClient(
            init_payload=init_payload,
            stderr_path=tmp_path / "judge-only.log",
        ) as judge:
            for method in inner_methods:
                if scope == "encdec":
                    judge.rebind(init_payload | {"method": method, "factor": factors[method]})
                for input_id in range(10):
                    key = (method, concept_id, input_id)
                    judged = judge.evaluate(
                        generation=emissions[key].generation,
                        input_id=input_id,
                    )
                    assert judged.generation == emissions[key].generation
                    assert judged.concept_rating == emissions[key].concept_rating
                    assert judged.fluency_rating == emissions[key].fluency_rating
                    assert judged.judge_rationale == emissions[key].judge_rationale
                    assert judged.paid_judge_cache_misses == 2
        assert _Handler.calls == len(inner_methods) * 10 * 2
        # Make root-module propagation load-bearing, not inherited from pytest.
        monkeypatch.delenv("PYTHONPATH", raising=False)
        result = run_final_judge(
            input_table=table,
            input_path=input_path,
            output_dir=tmp_path / "judged",
            cache_path=cache,
            best_factors=factors,
            emissions=emissions,
            max_concepts=max_concepts,
            concept_ids=concept_ids,
            method_factors=method_factors,
            api_base_url=api_base_url,
        )
    replay = run_final_judge(
        input_table=table,
        input_path=input_path,
        output_dir=tmp_path / "judged-replay",
        cache_path=cache,
        best_factors=factors,
        emissions=emissions,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
        method_factors=method_factors,
        cache_only=True,
        api_base_url=api_base_url,
    )
    assert result.steering_jsonl.is_file()
    assert result.actual_evaluator_calls == len(methods) * 10 * 3
    assert result.paid_judge_cache_misses == len(methods) * 10 * 3 - len(inner_methods) * 10 * 2
    assert replay.actual_evaluator_calls == len(methods) * 10 * 3
    assert replay.paid_judge_cache_misses == 0
    assert _Handler.calls == len(methods) * 10 * 3
    if scope == "nl_only":
        summary = budget_summary(budget_path)
        assert summary["attempts"] == _Handler.calls == 30
        assert summary["committed_nanos"] == summary["known_actual_nanos"] == 30_000_000
        assert summary["halted"] == 0
    validate_final_judge_outputs(
        input_table=table,
        output_dir=result.steering_jsonl.parent,
        best_factors=factors,
        emissions=emissions,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
        method_factors=method_factors,
    )
    if scope in {"native", "encdec"}:
        # Native campaigns publish their own receipts, not the legacy schema-2 histogram manifest.
        return
    # The same reader must accept the real sparse union-factor serialization.
    report_root = tmp_path / "adv-report-source"
    files = {}
    for level in (0, 1, 2, 4, 8):
        judged = report_root / "points" / f"p{level}" / "judged"
        judged.mkdir(parents=True)
        for name in ("steering.jsonl", "steering_data.parquet"):
            payload = (result.steering_jsonl.parent / name).read_bytes()
            target = judged / name
            target.write_bytes(payload)
            files[str(target.relative_to(report_root))] = hashlib.sha256(payload).hexdigest()
    (report_root / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "methods": list(methods), "best_factors": factors,
        "max_concepts": max_concepts, "concept_ids": list(concept_ids), "files": files,
    }))
    report = generate_histogram_reports(
        input_dir=report_root,
        output_dir=tmp_path / "histograms",
        pipeline="adv",
        method_factors={},
    )
    metadata = json.loads(report.read_text())
    for level in (0, 1, 2, 4, 8):
        scores = pd.read_csv(report.parent / f"p{level}" / "concept_scores.csv")
        assert set(scores["concept_id"]) == {concept_id}
        assert set(scores["method"]) == set(methods)
    assert len(metadata["reports"]["p0"]) == len(methods) * 3
    assert all(
        row["mean"] == 1.0 and row["sample_std"] is None for row in metadata["reports"]["p0"]
    )
