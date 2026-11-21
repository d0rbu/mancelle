"""Small subprocess used to falsify root-side protocol/lifecycle oracles."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from scripts.eval.axbench_adv.vendored.protocol import PROTOCOL_VERSION, decode_frame, encode_frame


def _string_keyed_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _response(request: dict[str, object], **updates: object) -> dict[str, object]:
    payload = _string_keyed_mapping(request["payload"])
    if payload is None:
        raise TypeError("fixture request payload must be a string-keyed mapping")
    result: dict[str, object] = {
        "id": request["id"],
        "op": request["op"],
        "result": payload,
        "status": "ok",
        "version": PROTOCOL_VERSION,
    }
    result.update(updates)
    return result


def main() -> int:
    mode = sys.argv[1]
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    request = decode_frame(reader)
    print(f"child-mode={mode}", file=sys.stderr, flush=True)
    if mode == "eof":
        return 7
    if mode == "timeout":
        time.sleep(60)
        return 8
    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
        return 8
    if mode == "grandchild":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(sys.argv[2]).write_text(str(child.pid), encoding="utf-8")
        time.sleep(60)
        return 9
    if mode == "corrupt":
        writer.write(b"oops")
        writer.flush()
        time.sleep(60)
        return 10
    if mode == "error":
        response = _response(request)
        response.pop("result")
        response.update(
            {
                "error": {"message": "deliberate child failure", "type": "ValueError"},
                "status": "error",
            }
        )
    elif mode == "wrong-id":
        response = _response(request, id=int(request["id"]) + 1)
    elif mode == "wrong-op":
        response = _response(request, op="other")
    elif mode == "wrong-version":
        response = _response(request, version=PROTOCOL_VERSION + 1)
    elif mode == "bad-fields":
        response = _response(request, extra=True)
    elif mode == "bad-status":
        response = _response(request, status="mystery")
    else:
        response = _response(request)
    encoded = encode_frame(response)
    if mode == "partial":
        for byte in encoded:
            os.write(writer.fileno(), bytes([byte]))
    else:
        writer.write(encoded)
        writer.flush()
    if mode not in {
        "success",
        "partial",
        "close-hang",
        "close-nonempty",
        "close-nonzero",
    }:
        return 0
    close = decode_frame(reader)
    close_result = {"unexpected": True} if mode == "close-nonempty" else {}
    writer.write(encode_frame(_response(close, result=close_result)))
    writer.flush()
    if mode == "close-hang":
        time.sleep(60)
    return 7 if mode == "close-nonzero" else 0


if __name__ == "__main__":
    raise SystemExit(main())
