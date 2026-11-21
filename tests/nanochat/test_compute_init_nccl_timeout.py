"""NCCL_TIMEOUT_SEC parsing + SCOPED propagation into compute_init.

Scoped contract: ``NCCL_TIMEOUT_SEC`` unset → ``compute_init`` passes
NO ``timeout=`` to ``dist.init_process_group``, so PyTorch's backend
default is preserved for every DDP job (the "global default preserved"
invariant). Only an explicit ``NCCL_TIMEOUT_SEC`` overrides it, and only
for the job that sets it. The axbench-base Stage 0c launcher opts itself
in via ``os.environ.setdefault("NCCL_TIMEOUT_SEC", "1800")``.

The gloo multiprocess harness CANNOT reach nanochat/common.py's
``dist.init_process_group(backend="nccl", ...)`` (it is gated on
device_type==CUDA; the harness is gloo + device=cpu). So we drive the
CUDA+DDP branch with init_process_group mocked. PyTorch's
_check_valid_timeout only type-checks, so a 0/negative timedelta would
be silently accepted and abort every collective instantly — hence the
strict validation under test.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

import nanochat.common as common
from nanochat.common import _resolve_nccl_timeout_seconds


def test_resolve_nccl_timeout_none_when_unset(monkeypatch):
    """Unset → None (caller preserves PyTorch's backend default)."""
    monkeypatch.delenv("NCCL_TIMEOUT_SEC", raising=False)
    assert _resolve_nccl_timeout_seconds() is None


def test_resolve_nccl_timeout_valid_override(monkeypatch):
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", "43200")
    assert _resolve_nccl_timeout_seconds() == 43200


@pytest.mark.parametrize("bad", ["", "abc", "12.5", "1e3", " ", "0x10"])
def test_resolve_nccl_timeout_non_int_raises_clear(monkeypatch, bad):
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", bad)
    with pytest.raises(ValueError, match=r"NCCL_TIMEOUT_SEC must be a positive integer"):
        _resolve_nccl_timeout_seconds()


@pytest.mark.parametrize("bad", ["0", "-1", "-5"])
def test_resolve_nccl_timeout_non_positive_raises_clear(monkeypatch, bad):
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", bad)
    with pytest.raises(ValueError, match=r"NCCL_TIMEOUT_SEC must be >= 1"):
        _resolve_nccl_timeout_seconds()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 5 ", 5),
        ("+5", 5),
        ("1_000", 1000),
        ("007", 7),
        ("\t9\t", 9),
        # int() accepts unicode decimal digits; these are intentionally
        # treated as valid positive-second intents (Arabic-Indic 5,
        # fullwidth 23) rather than silently rejected.
        ("٥", 5),
        ("２３", 23),
    ],
)
def test_resolve_nccl_timeout_parsing_quirks_pinned(monkeypatch, raw, expected):
    """G1: pin the exact ``int()``-accepted surface so it cannot silently
    drift. Leading/trailing whitespace, an explicit ``+`` sign, ``_``
    digit grouping, leading zeros, tabs, and unicode decimal digits are
    all accepted as the operator's positive-second intent."""
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", raw)
    assert _resolve_nccl_timeout_seconds() == expected


def test_resolve_nccl_timeout_overflow_raises_value_error(monkeypatch):
    """Finding-6: a value that overflows ``datetime.timedelta`` must be
    converted to a clear ``ValueError`` — an uncaught ``OverflowError``
    must never escape the parser."""
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", "100000000000000000000")
    with pytest.raises(ValueError, match=r"NCCL_TIMEOUT_SEC") as excinfo:
        _resolve_nccl_timeout_seconds()
    # Specifically NOT an OverflowError leaking through.
    assert not isinstance(excinfo.value, OverflowError)


def test_resolve_nccl_timeout_large_but_valid_ok(monkeypatch):
    """A large-but-representable value (1 year) parses and constructs a
    valid timedelta — the overflow guard must not over-reject."""
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", "31536000")
    assert _resolve_nccl_timeout_seconds() == 31536000
    # Sanity: the very thing the parser validated internally.
    assert datetime.timedelta(seconds=31536000)


def _force_cuda_ddp(monkeypatch, spy):
    """Drive compute_init down the CUDA+DDP branch with a spy PG init."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.manual_seed", lambda *a, **k: None)
    monkeypatch.setattr("torch.cuda.set_device", lambda *a, **k: None)
    monkeypatch.setattr(
        "torch.set_float32_matmul_precision", lambda *a, **k: None,
    )
    monkeypatch.setattr(common.dist, "init_process_group", spy)
    monkeypatch.setattr(common.dist, "barrier", lambda *a, **k: None)


def test_compute_init_unset_preserves_pytorch_default(monkeypatch):
    """Critical invariant: with NCCL_TIMEOUT_SEC unset, compute_init
    passes NO ``timeout=`` kwarg, so PyTorch's backend default is
    preserved for every DDP job (no global default change)."""
    monkeypatch.delenv("NCCL_TIMEOUT_SEC", raising=False)
    seen = {}
    _force_cuda_ddp(monkeypatch, lambda *a, **k: seen.update(k))
    common.compute_init(device_type="cuda")
    assert "timeout" not in seen
    assert seen["backend"] == "nccl"


def test_compute_init_passes_override_timeout_to_init_pg(monkeypatch):
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", "43200")
    seen = {}
    _force_cuda_ddp(monkeypatch, lambda *a, **k: seen.update(k))
    common.compute_init(device_type="cuda")
    assert seen["timeout"] == datetime.timedelta(seconds=43200)
    assert seen["backend"] == "nccl"


@pytest.mark.parametrize("bad", ["abc", "12.5", "1e3", " ", "0x10", "", "0", "-5"])
def test_compute_init_rejects_bad_timeout_before_init_pg(monkeypatch, bad):
    """Malformed NCCL_TIMEOUT_SEC must raise BEFORE init_process_group."""
    monkeypatch.setenv("NCCL_TIMEOUT_SEC", bad)
    called = {"n": 0}

    def spy(*args, **kwargs):
        called["n"] += 1

    _force_cuda_ddp(monkeypatch, spy)
    with pytest.raises(ValueError, match=r"NCCL_TIMEOUT_SEC"):
        common.compute_init(device_type="cuda")
    assert called["n"] == 0


def test_compute_init_single_process_does_not_init_pg(monkeypatch):
    """No DDP env → init_process_group never called (NCCL_TIMEOUT_SEC inert)."""
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(
        common.dist, "init_process_group",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    common.compute_init(device_type="cpu")
    assert called["n"] == 0


def test_compute_init_source_wires_scoped_timeout():
    """Pin the SCOPED wiring at source level so a refactor that drops the
    conditional, the env read, or the timeout= kwarg is caught even
    without the mocked tests. The helper no longer hardcodes a "1800"
    default — it reads the bare env var and returns None when unset."""
    src = inspect.getsource(common.compute_init)
    assert "_resolve_nccl_timeout_seconds()" in src
    assert "if timeout_sec is None:" in src
    assert "dist.init_process_group(backend=\"nccl\", device_id=device)" in src
    assert "timeout=timedelta(seconds=timeout_sec)" in src
    helper_src = inspect.getsource(_resolve_nccl_timeout_seconds)
    assert 'os.environ.get("NCCL_TIMEOUT_SEC")' in helper_src
    # The old hardcoded default must be gone.
    assert 'os.environ.get("NCCL_TIMEOUT_SEC", "1800")' not in helper_src


def _axbench_base_run_source() -> str:
    """Extract the entrypoint's ``_run`` function source via the module's
    AST, WITHOUT importing it.

    Importing ``scripts.eval.axbench_base_eval`` pulls in the entire
    heavy axbench dependency graph (hydra + torch.distributed + dozens of
    steering modules), and exercising ``_run`` would require a full Hydra
    config and a real distributed/CUDA environment — far too heavy and
    flaky for a focused unit test. Parsing the file's AST and pulling
    just the ``_run`` source segment is deterministic and side-effect
    free, yet still pins the load-bearing wiring.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(common.__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "eval" / "axbench_base_eval.py"
    source = mod_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run":
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError("_run not found in scripts/eval/axbench_base_eval.py")


def test_axbench_base_eval_entrypoint_opts_into_stage0c_default():
    """The axbench-base Stage 0c launcher must opt itself in to a 1800s
    process-group timeout immediately before ``compute_init``.

    Static-source check (see ``_axbench_base_run_source`` for why we do
    not invoke ``_run``). It robustly pins (a) that the launcher uses
    ``setdefault`` semantics — so an operator-set ``NCCL_TIMEOUT_SEC`` is
    preserved — and (b) that it runs BEFORE the ``compute_init(`` call,
    which is the load-bearing ordering.
    """
    src = _axbench_base_run_source()
    assert 'os.environ.setdefault("NCCL_TIMEOUT_SEC", "1800")' in src
    setdefault_at = src.index('os.environ.setdefault("NCCL_TIMEOUT_SEC", "1800")')
    compute_init_at = src.index("compute_init(")
    assert setdefault_at < compute_init_at, (
        "NCCL_TIMEOUT_SEC opt-in must run before compute_init()"
    )
