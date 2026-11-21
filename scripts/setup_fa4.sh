#!/bin/bash
# Install (through the locked project dependency) and verify the production FA4 path.
#
# Usage:
#   bash scripts/setup_fa4.sh          # verify FA4 forward/backward on Hopper+
#   bash scripts/setup_fa4.sh --fa3    # also verify the FA3 KV-cache fallback

set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--fa3" ) ]]; then
    echo "Usage: bash scripts/setup_fa4.sh [--fa3]" >&2
    exit 2
fi

if command -v uv &>/dev/null; then
    PY=(uv run python3)
    echo "[INFO] Using uv-managed environment"
else
    PY=(python3)
fi

"${PY[@]}" - <<'PY'
import importlib.metadata

import torch as t
import torch.nn.functional as F

import nanochat.flash_attention as fa


if not t.cuda.is_available():
    raise SystemExit("[ERROR] FA4 verification requires a CUDA GPU")

major, minor = t.cuda.get_device_capability()
if major < 9:
    raise SystemExit(
        f"[ERROR] FA4 requires Hopper or newer (sm90+); found sm{major}{minor}"
    )

try:
    fa4_version = importlib.metadata.version("flash-attn-4")
except importlib.metadata.PackageNotFoundError as exc:
    raise SystemExit(
        "[ERROR] flash-attn-4 is missing; run `uv sync` from the project root"
    ) from exc

print(f"[INFO] PyTorch: {t.__version__}")
print(f"[INFO] GPU: {t.cuda.get_device_name()} (sm{major}{minor})")
print(f"[INFO] flash-attn-4: {fa4_version}")

if not fa.HAS_FA4_BACKEND:
    raise SystemExit(
        "[ERROR] nanochat did not detect the FA4 backend; flash_attn.cute must be importable"
    )

# Hopper FA4 block-sparse masks require 128-wide blocks. Going through the
# repository wrapper exercises the same block mask and compiled call as training.
B, H, S, D = 1, 4, 128, 64
t.manual_seed(0)
q = t.randn(B, H, S, D, device="cuda", dtype=t.bfloat16, requires_grad=True)
k = t.randn(B, H, S, D, device="cuda", dtype=t.bfloat16, requires_grad=True)
v = t.randn(B, H, S, D, device="cuda", dtype=t.bfloat16, requires_grad=True)

out = fa.flash_attn.flash_attn_func(q, k, v, causal=True)
reference = F.scaled_dot_product_attention(
    q.detach(), k.detach(), v.detach(), is_causal=True
)
t.testing.assert_close(out, reference, rtol=3e-2, atol=3e-2)
out.float().square().mean().backward()
t.cuda.synchronize()

if not all(x.grad is not None and bool(t.isfinite(x.grad).all()) for x in (q, k, v)):
    raise SystemExit("[ERROR] FA4 backward produced missing or non-finite gradients")
if not fa.has_fa4_backend() or fa._active_impl() != fa.FlashAttnImpl.FA4:
    raise SystemExit("[ERROR] FA4 was disabled during its runtime compatibility check")

print("[INFO] FA4 production forward/backward and SDPA parity: PASS")
PY

if [[ "${1:-}" == "--fa3" ]]; then
    "${PY[@]}" - <<'PY'
import nanochat.flash_attention as fa


if not fa.HAS_FA3:
    raise SystemExit(
        "[ERROR] FA3 could not be loaded through the kernels package; "
        "set LOGURU_LEVEL=DEBUG for the loader error"
    )
print("[INFO] FA3 KV-cache fallback import: PASS")
PY
fi
