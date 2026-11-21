"""
Common utilities for nanochat.
"""

import os
import sys
import urllib.parse
import urllib.request
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from filelock import FileLock
from loguru import logger

_ALLOWED_DOWNLOAD_SCHEMES = frozenset({"http", "https"})


class DeviceType(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


# Configure loguru: rank-aware logging
logger.remove()  # Remove default handler
_ddp_rank = int(os.environ.get("RANK", 0))
# Rank-0 console level is env-overridable so long-running stages can surface
# their DEBUG progress instrumentation without a code change. Defaults to INFO,
# so unset behaviour is byte-identical to before.
_console_level = os.environ.get("NANOCHAT_LOG_LEVEL", "INFO").upper()
if _ddp_rank == 0:
    logger.add(
        sys.stderr,
        level=_console_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
else:
    # Non-zero ranks only log warnings and above
    logger.add(
        sys.stderr,
        # Non-zero ranks stay quiet by default, but honour an explicit override
        # so a DDP debugging session is not silently rank-0-only.
        level=os.environ.get("NANOCHAT_LOG_LEVEL_NONZERO_RANK", "WARNING").upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>rank {extra[rank]}</cyan> | {message}",
    )
    logger.configure(extra={"rank": _ddp_rank})


def get_project_root() -> str:
    """Return the repository root (parent of the nanochat package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_project_dotenv(
    project_root: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> bool:
    """Load environment variables from the project root's ``.env`` file.

    Pins the dotenv load to ``<project_root>/.env`` so that
    :func:`dotenv.load_dotenv` does NOT walk ancestor directories looking
    for a ``.env`` file. That default behavior is a security hole: an
    attacker who plants ``.env`` in any parent of the user's cwd (e.g.
    ``/tmp/.env``) could inject arbitrary environment variables — including
    API keys — into the Python process.

    Args:
        project_root: Directory whose ``.env`` should be loaded. Defaults
            to :func:`get_project_root` (the repository root). An explicit
            value is primarily useful in tests.
        override: Whether values from the ``.env`` file should overwrite
            existing environment variables. Matches :func:`dotenv.load_dotenv`.

    Returns:
        ``True`` if at least one env var was set, ``False`` otherwise.
        Missing ``.env`` files are silently treated as no-ops (matching
        the baseline behavior of :func:`dotenv.load_dotenv`), but ancestor
        directories are never searched.
    """
    root = Path(project_root) if project_root is not None else Path(get_project_root())
    dotenv_path = root / ".env"
    return load_dotenv(dotenv_path=dotenv_path, override=override)


def get_base_dir() -> str:
    nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR", "")
    if not nanochat_dir:
        # Default: nanochat_cache in the project root (next to nanochat/ package)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nanochat_dir = os.path.join(repo_root, "nanochat_cache")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.

    Only http(s) URLs are accepted. Other schemes (e.g. file://, ftp://,
    data:, javascript:, or empty/bare paths) are rejected up front to
    prevent SSRF / arbitrary local-file reads if the URL is ever sourced
    from user-controlled config.
    """
    # Scheme validation: reject anything that isn't http/https, case-insensitive.
    # This blocks `file://`, `FILE://`, `FiLe://`, `ftp://`, `data:`, `javascript:`,
    # `ssh://`, `gopher://`, single-slash `file:/...`, and bare paths with no scheme.
    scheme = urllib.parse.urlparse(url).scheme.lower() if isinstance(url, str) else ""
    if scheme not in _ALLOWED_DOWNLOAD_SCHEMES:
        raise ValueError(
            f"download_file_with_lock: unsupported URL scheme {scheme!r} "
            f"for {url!r}; only http/https are allowed"
        )

    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        logger.info(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # Write to local file
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s="", **kwargs):
    """Rank-0 print. Delegates to loguru for simple calls, uses print() for kwargs (end=, flush=)."""
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        if kwargs:
            print(s, **kwargs)
        else:
            logger.info(str(s))


def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)


def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    return False, 0, 0, 1


def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = DeviceType.CUDA
    elif torch.backends.mps.is_available():
        device_type = DeviceType.MPS
    else:
        device_type = DeviceType.CPU
    print0(f"Autodetected device type: {device_type}")
    return device_type


def _resolve_nccl_timeout_seconds() -> int | None:
    """Parse the optional ``NCCL_TIMEOUT_SEC`` override.

    Contract:

    * **Unset** → return ``None``. The caller passes NO ``timeout=`` to
      ``init_process_group``, so PyTorch's backend default is preserved
      (10 min for NCCL / 30 min for GLOO). This is deliberate: we do NOT
      impose a global default — only an explicit opt-in changes it.
    * **Set** → return the validated positive ``int`` number of seconds.

    Validation (set case), all failing BEFORE ``init_process_group`` with
    an actionable ``ValueError`` instead of an opaque crash at DDP
    startup (PyTorch's internal ``_check_valid_timeout`` only
    *type*-checks — a zero/negative ``timedelta`` is silently accepted
    and would make every collective abort immediately):

    * non-integer (e.g. ``""``, ``"abc"``, ``"12.5"``) → ``ValueError``.
    * integer ``< 1`` → ``ValueError``.
    * integer so large that ``datetime.timedelta(seconds=value)`` would
      raise ``OverflowError`` → caught and converted to a clear
      ``ValueError`` (the value is out of ``timedelta``'s range); an
      uncaught ``OverflowError`` must never escape.
    """
    raw = os.environ.get("NCCL_TIMEOUT_SEC")
    if raw is None:
        return None
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"NCCL_TIMEOUT_SEC must be a positive integer number of "
            f"seconds, got {raw!r}. Unset it to preserve PyTorch's "
            f"backend default process-group timeout."
        ) from None
    if seconds < 1:
        raise ValueError(
            f"NCCL_TIMEOUT_SEC must be >= 1, got {seconds}. A zero or "
            f"negative process-group timeout makes every NCCL collective "
            f"abort immediately. Unset it to preserve PyTorch's default."
        )
    try:
        timedelta(seconds=seconds)
    except OverflowError:
        raise ValueError(
            f"NCCL_TIMEOUT_SEC={seconds} is too large: it overflows "
            f"datetime.timedelta and is out of the representable range "
            f"for a process-group timeout. Use a smaller value (or unset "
            f"it to preserve PyTorch's default)."
        ) from None
    return seconds


def compute_init(device_type="cuda"):  # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in DeviceType, (
        f"Invalid device type: {device_type}. Must be one of {list(DeviceType)}"
    )
    if device_type == DeviceType.CUDA:
        assert torch.cuda.is_available(), (
            "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
        )
    if device_type == DeviceType.MPS:
        assert torch.backends.mps.is_available(), (
            "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
        )

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == DeviceType.CUDA:
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == DeviceType.CUDA:
        torch.set_float32_matmul_precision(
            "high"
        )  # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == DeviceType.CUDA:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        # Process-group collective timeout. This is intentionally NOT a
        # global default change: when ``NCCL_TIMEOUT_SEC`` is UNSET we
        # pass no ``timeout=``, so PyTorch's backend default (~10 min
        # NCCL / 30 min GLOO) is preserved for every DDP job. Only an
        # explicit ``NCCL_TIMEOUT_SEC`` (e.g. the axbench-base Stage 0c
        # launcher opting itself in to 1800s for its ~25-min rank-0-only
        # Stage 0a, or ``NCCL_TIMEOUT_SEC=43200`` for a 12h pipeline)
        # overrides it — for that job only.
        timeout_sec = _resolve_nccl_timeout_seconds()
        if timeout_sec is None:
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(
                backend="nccl", device_id=device,
                timeout=timedelta(seconds=timeout_sec),
            )
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()


class _DummySummary:
    """Absorbs attribute writes (e.g. wandb_run.summary["key"] = val)."""

    def __setitem__(self, key, value):
        pass

    def __setattr__(self, key, value):
        pass


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""

    def __init__(self):
        self.summary = _DummySummary()

    def log(self, *args, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def configure_wandb_step_metric(wandb_run) -> None:
    """Make the user-supplied "step" field the x-axis for all logged metrics.

    Why: without this, `wandb.log({"step": s, ...})` records `step` as just another
    scalar and charts default to wandb's internal `_step` counter (one tick per log
    call), which makes long resumed runs look truncated.
    """
    wandb_run.define_metric("step")
    wandb_run.define_metric("*", step_metric="step")


# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast weights to input dtype, reducing (but not eliminating) reliance on
# torch.amp.autocast. Script-level autocast still handles intermediate ops (RMSNorm, sigmoid,
# softmax, etc.) for training scripts. See docs/architecture.md "Mixed Precision Contract".
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"


COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


# Hardcoded BF16 dense tensor core peak FLOPS for various GPUs.
# All values are BF16 dense (without sparsity) from official spec sheets.
# Note: FP8 tensor cores achieve ~2× these values. MFU is reported relative to
# BF16 peak for consistency across precisions — with FP8 training, MFU can
# exceed 100%, which indicates the model is benefiting from FP8 hardware support.
#
# Sources:
#   NVIDIA datacenter: https://www.nvidia.com/en-us/data-center/
#   NVIDIA consumer:   https://www.nvidia.com/en-us/geforce/
#   AMD CDNA:          https://www.amd.com/en/products/accelerators/instinct.html
#   torchtitan ref:    https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell — https://www.nvidia.com/en-us/data-center/gb200-nvl72/
        (["gb200"], 2.5e15),           # GB200 NVL72 per-GPU
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),           # B200 SXM
        (["b100"], 1.8e15),            # B100
        (["rtx pro 6000", "blackwell"], 503.8e12),  # RTX PRO 6000 Blackwell
        # NVIDIA Hopper — https://www.nvidia.com/en-us/data-center/h100/
        (["h200", "nvl"], 836e12),     # H200 NVL (lower TDP per GPU)
        (["h200", "pcie"], 836e12),    # H200 PCIe
        (["h200"], 989e12),            # H200 SXM (same die as H100 SXM)
        (["h100", "nvl"], 835e12),     # H100 NVL
        (["h100", "pcie"], 756e12),    # H100 PCIe
        (["h100"], 989e12),            # H100 SXM5
        (["h800", "nvl"], 989e12),     # H800 NVL (China variant)
        (["h800"], 756e12),            # H800 PCIe (China variant)
        # NVIDIA Ampere — https://www.nvidia.com/en-us/data-center/a100/
        (["a100"], 312e12),            # A100 SXM4 80GB
        (["a800"], 312e12),            # A800 (China variant, same compute)
        (["a40"], 149.7e12),           # A40
        (["a30"], 165e12),             # A30
        # NVIDIA Ada Lovelace — https://www.nvidia.com/en-us/data-center/l40s/
        (["l40s"], 362e12),            # L40S
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),              # L4
        # AMD CDNA — https://www.amd.com/en/products/accelerators/instinct.html
        (["mi355"], 2.5e15),           # MI355X (CDNA 4)
        (["mi325"], 1.3074e15),        # MI325X (CDNA 3)
        (["mi300x"], 1.3074e15),       # MI300X (CDNA 3)
        (["mi300a"], 980.6e12),        # MI300A APU
        (["mi250x"], 383e12),          # MI250X (CDNA 2)
        (["mi250"], 362.1e12),         # MI250 (CDNA 2)
        # Consumer RTX — https://www.nvidia.com/en-us/geforce/
        (["5090"], 209.5e12),          # RTX 5090 (Blackwell)
        (["4090"], 165.2e12),          # RTX 4090 (Ada Lovelace)
        (["3090"], 71e12),             # RTX 3090 (Ampere)
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")
