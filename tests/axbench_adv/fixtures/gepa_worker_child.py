"""Persistent fake semantic worker for the real-GEPA/client functional test."""

from __future__ import annotations

import sys

from scripts.eval.axbench_adv.vendored.protocol import (
    PROTOCOL_VERSION,
    decode_frame,
    write_frame,
)


def _ok(request: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return {
        "id": request["id"],
        "op": request["op"],
        "result": result,
        "status": "ok",
        "version": PROTOCOL_VERSION,
    }


def main() -> int:
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    init = decode_frame(reader)
    payload = init["payload"]
    write_frame(
        writer,
        _ok(
            init,
            {
                "concept_id": payload["concept_id"],
                "factor": float(payload["factor"]),
                "method": payload["method"],
            },
        ),
    )
    seen: set[tuple[int, str]] = set()
    while True:
        request = decode_frame(reader)
        if request["op"] == "close":
            write_frame(writer, _ok(request, {}))
            return 0
        row = request["payload"]
        key = (int(row["input_id"]), str(row["input"]))
        paid = 0 if key in seen else 2
        seen.add(key)
        improved = "better" in str(row["input"])
        generation = (
            "A useful substantive answer that avoids the target."
            if improved
            else "A useful substantive answer about the target concept."
        )
        write_frame(
            writer,
            _ok(
                request,
                {
                    "concept_rating": 1.0 if improved else 2.0,
                    "fluency_rating": 2.0,
                    "generation": generation,
                    "input_id": row["input_id"],
                    "judge_rationale": "offline deterministic judge",
                    "paid_judge_cache_misses": paid,
                },
            ),
        )


if __name__ == "__main__":
    raise SystemExit(main())
