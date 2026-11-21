"""Reserve framed stdout before importing any third-party worker dependencies."""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Callable
from typing import BinaryIO, cast

WORKER_MODULE = "scripts.eval.axbench_adv.vendored.worker"


def serve_module(module: str = WORKER_MODULE) -> int:
    """Import a worker with diagnostics on stderr and a private protocol pipe.

    Duplicating the pipe before redirecting descriptor 1 also catches native
    extensions and libraries that write through ``sys.__stdout__`` or ``os.write``.
    The worker's imports remain at module scope, and its existing ``serve``
    implementation owns every byte written to the reserved descriptor.
    """
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as protocol_writer:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        namespace = runpy.run_module(module, run_name="__axbench_adv_worker__", alter_sys=True)
        serve = namespace.get("serve")
        if not callable(serve):
            raise TypeError("AxBench-Adv worker module must expose callable serve")
        return cast(Callable[[BinaryIO, BinaryIO], int], serve)(sys.stdin.buffer, protocol_writer)


def main() -> int:
    return serve_module()


if __name__ == "__main__":
    raise SystemExit(main())
