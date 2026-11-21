"""Smoke tests for benchmark_native_compile.py (CPU, no real model/GPU).

Exercises the script's measurement + ORACLE-gating + reporting plumbing using
tiny Gemma twins and stubs so the full wiring is verified without a GPU or
real model download.

Test groups:
  (a) table/JSON produced and parseable — the script runs end-to-end and
      produces a valid result dict with the required schema.
  (b) gate=FAIL guard — injecting a stubbed gate verdict=FAIL ⇒ the row shows
      oracle_verdict="FAIL", compiled=DISABLED, and no speedup is reported as
      a win (assert the formatter refuses it); the honest-reporting guard is
      structural, not cosmetic.
  (c) gate=PASS + compiled slower — compiled path activated but slower ⇒
      speedup < 1.0 flag ("SLOWER") is present and visible in the formatted row.

Mutation-probe (b): removing the `if not oracle_pass: return row` guard in
`build_row` causes test (b) to fail because the script would continue to measure
the disabled compiled path and potentially report a result.

All tests are deterministic, fast, and GPU-free.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch as t
from transformers import Gemma2Config, Gemma2ForCausalLM

import steering.extract_features as ef
from scripts.benchmark.benchmark_native_compile import (
    build_row,
    format_row,
    run_smoke_cell,
)
from steering.native_collector import NativeCollector

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "benchmark" / "benchmark_native_compile.py"

# --------------------------------------------------------------------------- #
# Shared tiny-model constants (mirror conftest.py from native suite so the    #
# smoke harness is byte-identical to what the unit tests use).                #
# --------------------------------------------------------------------------- #
_TINY_VOCAB = 64
_TINY_D_MODEL = 32
_TINY_LAYERS = 4
_TINY_HEADS = 4
_TINY_KV_HEADS = 2
_TINY_HEAD_DIM = 8
_TINY_FFN = 64
_TINY_CTX = 64
_TINY_SEED = 20260516

# Probe input: ragged right-padded (row 1 has 2 trailing pads) — exercises
# the production maskless / right-pad contract at the unit level.
_PROBE_IDS = t.tensor([[3, 9, 1, 7, 2], [5, 5, 8, 0, 0]], dtype=t.long)

# Layer to truncate — a real interior layer (0 <= 2 < _TINY_LAYERS).
_LAYER = 2


def test_documented_direct_script_invocation_bootstraps_repo_from_arbitrary_cwd(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "benchmark_native_compile" in completed.stdout


def _build_tiny_gemma2() -> Gemma2ForCausalLM:
    """Fresh seeded tiny Gemma2 (byte-identical to conftest's _build_tiny_gemma2)."""
    with t.random.fork_rng():
        t.manual_seed(_TINY_SEED)
        cfg = Gemma2Config(
            vocab_size=_TINY_VOCAB,
            hidden_size=_TINY_D_MODEL,
            intermediate_size=_TINY_FFN,
            num_hidden_layers=_TINY_LAYERS,
            num_attention_heads=_TINY_HEADS,
            num_key_value_heads=_TINY_KV_HEADS,
            head_dim=_TINY_HEAD_DIM,
            max_position_embeddings=_TINY_CTX,
            attn_implementation="eager",  # ty: ignore[unknown-argument]
        )
        return Gemma2ForCausalLM(cfg).to(t.float32).eval()  # ty: ignore[invalid-argument-type]


class _TinySAE:
    """Minimal SAE-protocol stub: linear+ReLU, deterministic seeded weights.

    Same shape contract as test_native_compile_gate_unit.py's `_SAE` (has
    `parameters()`, `to()`, and `encode(x)` mapping `[*, d] -> [*, f]`).
    Deterministic seeded weights ensure reproducible feature activations.
    """

    def __init__(self, d: int = _TINY_D_MODEL, f: int = 48) -> None:
        with t.random.fork_rng():
            t.manual_seed(1234)
            self._w = t.randn(d, f)
            self._b = t.randn(f)
        self.d = d
        self.f = f

    def parameters(self):
        yield from []

    def to(self, *args: Any, **kwargs: Any) -> _TinySAE:
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x.float().matmul(self._w) + self._b)


def _fresh_nc() -> NativeCollector:
    """Fresh truncated NativeCollector on a fresh tiny Gemma2 twin."""
    return NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER
    )


@pytest.fixture(autouse=True)
def _stub_gate_fp32_reference_build(monkeypatch):
    """The ideal-math gate (``extract_features._native_compile_gate``) builds
    its OWN fp32-eager reference via ``build_native_model(native_model.
    model_name, ...)``. In this CPU smoke harness that would load the REAL
    gemma-2-2b (hidden 2304) and break the tiny 32-wide ``_TinySAE`` with a
    matmul shape error. Patch it to return a fresh tiny Gemma2 twin (same as
    the gate-unit suite's ``_patch_build_ref``) so the gate's fp32-ref capture
    stays at ``_TINY_D_MODEL`` and the smoke harness exercises the real
    benchmark plumbing without any real-model download."""
    monkeypatch.setattr(ef, "build_native_model", lambda *a, **k: _fresh_nc())


# --------------------------------------------------------------------------- #
# Group (a): schema — table/JSON produced and parseable.                      #
# --------------------------------------------------------------------------- #


class TestSchemaAndJson:
    """(a) The script produces a result dict with the required schema columns."""

    def test_build_row_returns_required_columns_gate_pass(self):
        """When the gate passes (eps=0 ≡ compiled==eager), all schema columns
        are present: model, layer, dtype, eager_*, oracle_pass, oracle_verdict,
        compiled_*, speedup, speedup_flag.
        """
        nc = _fresh_nc()
        sae = _TinySAE()
        # Stub: compiled ≡ eager (eps=0 → gate PASSES).
        nc.enable_compiled()
        nc._compiled_base = nc._base  # identical forward → gate PASSES
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        # Required schema columns must all be present.
        for col in (
            "model", "layer", "dtype",
            "eager_median_ms", "eager_tok_per_s", "eager_peak_mb",
            "oracle_pass", "oracle_verdict",
        ):
            assert col in row, f"missing required column {col!r}"
        assert row["oracle_verdict"] in ("PASS", "FAIL")

    def test_build_row_json_serialisable(self):
        """The row dict is JSON-serialisable (all values are JSON-native or None
        which maps to null), as required by the `json.dump(..., default=str)` call
        in main().
        """
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        nc._compiled_base = nc._base
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        serialised = json.dumps(row, default=str)
        parsed = json.loads(serialised)
        assert isinstance(parsed, dict)
        assert parsed["model"] == "gemma-2-2b"

    def test_format_row_gate_pass_contains_oracle_pass(self):
        """format_row on a gate-PASS row must contain 'ORACLE=PASS'."""
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        nc._compiled_base = nc._base
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        formatted = format_row(row)
        assert "ORACLE=PASS" in formatted, (
            f"gate-PASS row must contain 'ORACLE=PASS'; got: {formatted!r}"
        )


# --------------------------------------------------------------------------- #
# Group (b): gate=FAIL guard — the formatter refuses to present a            #
# gate-FAIL row as a win.                                                     #
# --------------------------------------------------------------------------- #


class _PerturbedCompiled:
    """Wraps the eager base and adds a large perturbation to last_hidden_state.

    A large eps causes the SAE features to diverge massively, so the gate
    oracle FAILS deterministically regardless of model / tolerance settings.
    """

    def __init__(self, base: Any, eps: float = 50.0) -> None:
        self._base = base
        self._eps = eps

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._base(*args, **kwargs)
        out.last_hidden_state = out.last_hidden_state + self._eps
        return out


class TestOracleFail:
    """(b) Injecting a gate verdict=FAIL ensures DISABLED row, no win reported.

    The HONEST-REPORTING invariant: if the oracle FAILS, the row MUST show
    oracle_verdict='FAIL' and must NOT contain a valid speedup number
    (speedup=None, speedup_flag='ORACLE=FAIL').  The formatted string must
    contain 'ORACLE=FAIL' and 'DISABLED'.

    Mutation-probe: removing the `if not oracle_pass: return row` guard in
    `build_row` causes the compiled path to be measured anyway, and the
    assertions below fail (compiled fields would be non-None, speedup would
    be a float, the format_row string would change).
    """

    def test_gate_fail_row_has_oracle_fail_verdict(self):
        """Gate-FAIL row: oracle_verdict='FAIL', oracle_pass=False."""
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        # Large eps → gate FAILS.
        nc._compiled_base = _PerturbedCompiled(nc._base, eps=50.0)
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        assert row["oracle_pass"] is False, (
            "large perturbation must cause oracle_pass=False"
        )
        assert row["oracle_verdict"] == "FAIL", (
            f"oracle_verdict must be 'FAIL' when gate fails; got {row['oracle_verdict']!r}"
        )

    def test_gate_fail_row_compiled_fields_are_none(self):
        """HONEST-REPORTING: gate-FAIL row must NOT contain compiled performance
        numbers — compiled_median_ms, compiled_tok_per_s must be None.

        Mutation-probe: removing the `if not oracle_pass: return row` guard in
        `build_row` causes compiled fields to be populated, breaking this test.
        """
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        nc._compiled_base = _PerturbedCompiled(nc._base, eps=50.0)
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        assert row["compiled_median_ms"] is None, (
            "gate-FAIL row must NOT have compiled_median_ms (must be None)"
        )
        assert row["compiled_tok_per_s"] is None, (
            "gate-FAIL row must NOT have compiled_tok_per_s (must be None)"
        )
        assert row["speedup"] is None, (
            "gate-FAIL row must NOT have a speedup value (must be None)"
        )

    def test_gate_fail_speedup_flag_is_oracle_fail(self):
        """Gate-FAIL row: speedup_flag='ORACLE=FAIL'."""
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        nc._compiled_base = _PerturbedCompiled(nc._base, eps=50.0)
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        assert row["speedup_flag"] == "ORACLE=FAIL", (
            f"speedup_flag must be 'ORACLE=FAIL'; got {row['speedup_flag']!r}"
        )

    def test_format_row_gate_fail_shows_disabled_and_oracle_fail(self):
        """Formatter: gate-FAIL row must contain both 'DISABLED' and 'ORACLE=FAIL'.

        This is the structural honest-reporting check: the formatted output
        makes the gate-FAIL verdict unmissable.
        """
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        nc._compiled_base = _PerturbedCompiled(nc._base, eps=50.0)
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        formatted = format_row(row)
        assert "DISABLED" in formatted, (
            f"formatter must show DISABLED for gate-FAIL; got: {formatted!r}"
        )
        assert "ORACLE=FAIL" in formatted, (
            f"formatter must show ORACLE=FAIL for gate-FAIL; got: {formatted!r}"
        )

    def test_format_row_gate_fail_no_speedup_as_win(self):
        """Gate-FAIL formatted row must NOT contain a speedup number presented
        as a win: no 'speedup=<float>x' pattern where the float is a valid
        speed ratio.  speedup='N/A' is acceptable; a numeric speedup is not.

        This is the direct structural enforcement of spec §3.2 honest-reporting:
        it is structurally impossible to present a fast-but-oracle-FAIL cell
        as a win when this test is in the suite.
        """
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        nc._compiled_base = _PerturbedCompiled(nc._base, eps=50.0)
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=2,
        )
        formatted = format_row(row)
        # The formatter MUST NOT contain a numeric speedup like "speedup=1.234x".
        # It must say "speedup=N/A".
        assert "speedup=N/A" in formatted, (
            f"gate-FAIL row must show speedup=N/A, not a numeric speedup; "
            f"got: {formatted!r}"
        )


# --------------------------------------------------------------------------- #
# Group (c): gate=PASS + compiled slower → speedup<1.0 flag visible.         #
# --------------------------------------------------------------------------- #


class _SlowCompiled:
    """Wraps the eager base but adds artificial latency (via a sleep-like loop).

    Injects extra scalar ops to make the 'compiled' path measurably slower
    than eager in a CPU smoke context.  The perturbation is tiny (eps=0) so
    the SAE oracle still PASSES — only the latency is artificially inflated.
    """

    def __init__(self, base: Any, extra_ops: int = 500_000) -> None:
        self._base = base
        self._extra_ops = extra_ops

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._base(*args, **kwargs)
        # Burn CPU time so compiled appears slower in wall-clock timing.
        x = t.zeros(1)
        for _ in range(self._extra_ops):
            x = x + 1e-10
        return out


class TestCompiledSlower:
    """(c) gate=PASS + artificially slower compiled ⇒ speedup<1.0 flag present."""

    def test_slower_compiled_sets_speedup_flag(self):
        """When the gate PASSES (eps=0 oracle-OK) but compiled is artificially
        slower, speedup < 1.0 and speedup_flag='SLOWER'.

        This ensures the SLOWER flag is structurally set — a non-win on a
        gate-PASS cell is never silently presented as OK.
        """
        nc = _fresh_nc()
        sae = _TinySAE()
        nc.enable_compiled()
        # eps=0 so the gate oracle PASSES; but add heavy CPU ops to inflate
        # the 'compiled' timing above eager.
        nc._compiled_base = _SlowCompiled(nc._base, extra_ops=500_000)
        row = build_row(
            "gemma-2-2b", _LAYER, t.float32, _PROBE_IDS, nc, sae,
            n_warmup=1, n_iters=3,
        )
        assert row["oracle_pass"] is True, (
            f"eps=0 with only timing overhead must PASS the oracle; "
            f"got oracle_pass={row['oracle_pass']}"
        )
        assert row["speedup"] is not None, "PASS row must have a speedup value"
        if row["speedup"] < 1.0:
            assert row["speedup_flag"] == "SLOWER", (
                f"speedup={row['speedup']:.4f} < 1.0 but speedup_flag="
                f"{row['speedup_flag']!r}; must be 'SLOWER'"
            )

    def test_format_row_slower_shows_slower_flag(self):
        """Formatter: when speedup < 1.0 the formatted row contains 'SLOWER'."""
        # Construct a gate-PASS row with a manually-set speedup < 1.0.
        row = {
            "model": "gemma-2-2b",
            "layer": _LAYER,
            "dtype": "fp32",
            "eager_median_ms": 10.0,
            "eager_tok_per_s": 1000,
            "eager_peak_mb": 100.0,
            "oracle_pass": True,
            "oracle_verdict": "PASS",
            "compiled_median_ms": 20.0,   # slower than eager
            "compiled_tok_per_s": 500,
            "compiled_peak_mb": 110.0,
            "compiled_warmup_s": 5.0,
            "speedup": 0.5,               # < 1.0
            "speedup_flag": "SLOWER",
        }
        formatted = format_row(row)
        assert "SLOWER" in formatted, (
            f"speedup=0.5x must display [SLOWER] flag; got: {formatted!r}"
        )


# --------------------------------------------------------------------------- #
# Integration smoke: run_smoke_cell end-to-end.                               #
# --------------------------------------------------------------------------- #


class TestRunSmokeCellEndToEnd:
    """Integration smoke: run_smoke_cell constructs, gates, and returns a row."""

    def test_smoke_cell_returns_valid_row(self):
        """run_smoke_cell on a tiny gemma2 twin returns a valid row dict."""
        sae = _TinySAE()
        row = run_smoke_cell(
            "gemma-2-2b", _LAYER, t.float32,
            _build_tiny_gemma2, sae, _PROBE_IDS,
            n_warmup=1, n_iters=2,
        )
        assert isinstance(row, dict)
        assert row["model"] == "gemma-2-2b"
        assert row["layer"] == _LAYER
        assert row["dtype"] == "fp32"
        assert "oracle_pass" in row
        assert "oracle_verdict" in row
        assert "eager_median_ms" in row

    def test_smoke_cell_json_round_trip(self):
        """The smoke cell result survives a JSON round-trip."""
        sae = _TinySAE()
        row = run_smoke_cell(
            "gemma-2-2b", _LAYER, t.float32,
            _build_tiny_gemma2, sae, _PROBE_IDS,
            n_warmup=1, n_iters=2,
        )
        s = json.dumps(row, default=str)
        parsed = json.loads(s)
        assert parsed["model"] == row["model"]
        assert parsed["layer"] == row["layer"]


# --------------------------------------------------------------------------- #
# NOTE on the honest-reporting structural guard's coverage:                    #
#                                                                             #
# A prior `test_build_row_source_contains_oracle_fail_guard` grepped          #
# `build_row`'s SOURCE for the substring "not oracle_pass". That check was     #
# WEAK and misleading: a guard-disabling mutation such as                      #
# `if False and not oracle_pass: return row` keeps the substring while         #
# defeating the early return, so the source-grep would still pass. It was      #
# DELETED rather than rewritten — robustly asserting "build_row returns        #
# before any compiled measurement when the oracle fails" via source           #
# inspection is brittle, and the behavioural `TestOracleFail` tests above      #
# already enforce exactly that property NON-VACUOUSLY: they call `build_row`   #
# with a perturbed compiled path that makes the gate FAIL and assert          #
# `compiled_median_ms is None`, `compiled_tok_per_s is None`,                  #
# `speedup is None`, `speedup_flag == "ORACLE=FAIL"`, and that `format_row`    #
# shows DISABLED / ORACLE=FAIL with no numeric speedup. Any mutation that      #
# defeats the `if not oracle_pass: return row` early-return (including the     #
# `if False and ...` mutation the grep could not catch) causes the compiled   #
# path to be measured and those fields to be populated, breaking the          #
# behavioural tests — they are the real, non-vacuous structural guard.        #
# --------------------------------------------------------------------------- #
