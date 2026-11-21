"""Serve sequential final AxBench evaluations in one vendored Python process."""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, cast

from axbench.scripts import evaluate

from scripts.eval.axbench_adv.vendored.protocol import (
    PROTOCOL_VERSION,
    decode_frame,
    write_frame,
)

VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor" / "axbench"


def _request(value: object) -> tuple[int, str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"id", "op", "payload", "version"}:
        raise ValueError("final worker request fields are malformed")
    frame = cast(dict[str, Any], value)
    request_id = frame["id"]
    operation = frame["op"]
    payload = frame["payload"]
    if type(request_id) is not int or request_id < 1:
        raise ValueError("final worker request id must be positive")
    if type(frame["version"]) is not int or frame["version"] != PROTOCOL_VERSION:
        raise ValueError("final worker protocol version mismatch")
    if operation not in {"evaluate", "close"} or not isinstance(payload, dict):
        raise ValueError("final worker operation or payload is malformed")
    return request_id, operation, cast(dict[str, Any], payload)


def _evaluate(payload: Mapping[str, Any]) -> None:
    expected = {
        "config_path", "input_dir", "output_dir", "cache_path", "cache_only", "api_base_url"
    }
    if set(payload) != expected:
        raise ValueError("final worker evaluation fields are malformed")
    if type(payload["cache_only"]) is not bool or any(
        not isinstance(payload[field], str) or not payload[field].strip()
        for field in expected - {"cache_only"}
    ):
        raise ValueError("final worker evaluation values are malformed")
    previous = sys.argv
    try:
        sys.argv = [
            "evaluate.py", "--config", payload["config_path"],
            "--mode", "steering",
            "--overwrite_inference_dump_dir", payload["input_dir"],
            "--overwrite_evaluate_dump_dir", payload["output_dir"],
            "--judge_cache", payload["cache_path"],
            "--cache_only", str(payload["cache_only"]).lower(),
            "--api_base_url", payload["api_base_url"],
        ]
        evaluate.main()
    finally:
        sys.argv = previous


def serve(reader: BinaryIO, writer: BinaryIO) -> int:
    os.chdir(VENDOR_ROOT)
    while True:
        request_id: int | None = None
        operation = "invalid"
        try:
            request_id, operation, payload = _request(decode_frame(reader))
            if operation == "evaluate":
                _evaluate(payload)
            elif payload:
                raise ValueError("final worker close payload must be empty")
            write_frame(
                writer,
                {
                    "id": request_id,
                    "op": operation,
                    "result": {},
                    "status": "ok",
                    "version": PROTOCOL_VERSION,
                },
            )
            if operation == "close":
                return 0
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            if request_id is not None:
                try:
                    write_frame(
                        writer,
                        {
                            "error": {"message": str(exc), "type": type(exc).__name__},
                            "id": request_id,
                            "op": operation,
                            "status": "error",
                            "version": PROTOCOL_VERSION,
                        },
                    )
                except BaseException:
                    traceback.print_exc(file=sys.stderr)
            return 1
