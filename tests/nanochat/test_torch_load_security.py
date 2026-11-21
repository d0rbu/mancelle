"""Security tests for ``torch.load`` sites in the repository.

These tests enforce the Wave 3B P0 remediation: every ``torch.load`` call MUST
pin ``weights_only=True`` (pickle RCE hardening) unless explicitly documented.

The critical test is ``test_checkpoint_load_refuses_malicious_pickle``: it
constructs an actual malicious pickle payload via ``__reduce__`` that would
write a sentinel file on load, invokes the public ``load_checkpoint``
entrypoint, and asserts the sentinel is NEVER created. No mocking - the real
load path is exercised.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
import torch

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from tests.conftest import _make_gpt, _make_gpt_config

_OS_SYSTEM = cast(Callable[[str], int], os.__dict__["system"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Sites that load optimizer state dicts. Optimizer state still fits within
# weights_only=True safe-globals, so they must also be pinned. The allowlist
# here is for sites that - by documented design - need weights_only=False.
# Format: {(relative_file_path, line_number): "reason"}
#
# After the fix, this allowlist MUST be empty. Any entry requires a comment
# in the source starting with "torch.load weights_only=False is required".
ALLOWED_WEIGHTS_ONLY_FALSE: dict[tuple[str, int], str] = {}


def _build_malicious_pickle_payload(sentinel_path: Path) -> bytes:
    """Return raw pickle bytes whose unpickling would create *sentinel_path*.

    We use ``__reduce__`` returning ``(os.system, ("touch <sentinel>",))``.
    If the loader uses ``weights_only=False`` (unrestricted pickle), unpickling
    this payload executes the ``os.system`` call. If ``weights_only=True`` is
    pinned, pickle refuses the ``GLOBAL os.system`` opcode and raises.

    We wrap the bomb inside a torch-save-compatible archive: a dict containing
    the bomb object as a value. This mirrors how a malicious attacker would
    embed the payload inside a legitimate-looking checkpoint file.
    """

    # Use a string quoted safely for the shell. tmp_path's absolute form is
    # guaranteed by pytest to be a clean absolute path.
    sentinel_str = str(sentinel_path)
    if " " in sentinel_str or "'" in sentinel_str or '"' in sentinel_str:
        raise AssertionError(
            "tmp_path unexpectedly contains quoting hazards: " + sentinel_str
        )

    class _PickleBomb:
        """Object whose __reduce__ triggers os.system on unpickle."""

        def __reduce__(self) -> tuple[Callable[[str], int], tuple[str]]:
            return (_OS_SYSTEM, (f"touch {sentinel_str}",))

    bomb = _PickleBomb()
    # Wrap in a dict so the file *looks* like a legitimate state dict.
    wrapper = {"w": bomb}
    return pickle.dumps(wrapper)


def _write_malicious_torch_archive(dst: Path, sentinel: Path) -> None:
    """Write a file at *dst* that torch.load will attempt to unpickle.

    torch.save ultimately writes a zip archive containing a ``data.pkl`` entry.
    To produce a file that torch.load will parse, we craft it by:

    1. Building a legitimate torch.save archive for a dict.
    2. Replacing the ``data.pkl`` content with our malicious pickle bytes.

    A simpler path: directly invoke ``pickle.dump`` into a file that resembles
    the *legacy* torch format. Modern ``torch.load`` auto-detects the legacy
    pickle format when the file is not a zip archive. We use that.
    """
    with dst.open("wb") as f:
        f.write(_build_malicious_pickle_payload(sentinel))


def _collect_torch_load_calls(
    source_paths: list[Path],
) -> list[tuple[Path, int, ast.Call]]:
    """Parse *source_paths* with AST and return all ``torch.load(...)`` calls.

    Returns list of (path, line_number, Call node). Skips calls embedded in
    docstrings / string literals (AST only catches real code).
    """
    results: list[tuple[Path, int, ast.Call]] = []
    for path in source_paths:
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            pytest.fail(f"Cannot parse {path}: {e}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match `torch.load(...)` and `t.load(...)` (where t is the alias).
            if isinstance(func, ast.Attribute) and func.attr == "load":
                if isinstance(func.value, ast.Name) and func.value.id in {
                    "torch",
                    "t",
                }:
                    results.append((path, node.lineno, node))
    return results


def _find_weights_only_kwarg(call: ast.Call) -> ast.expr | None:
    """Return the AST node for ``weights_only=<expr>`` or None if absent."""
    for kw in call.keywords:
        if kw.arg == "weights_only":
            return kw.value
    return None


# ---------------------------------------------------------------------------
# Test 1 (CRITICAL): malicious pickle must NOT run code via load_checkpoint
# ---------------------------------------------------------------------------


class TestMaliciousPickleRefused:
    """The load_checkpoint entrypoint MUST refuse a malicious pickle payload."""

    def test_checkpoint_load_refuses_malicious_pickle(self, tmp_path):
        """Construct a .pt file with os.system('touch sentinel') payload.

        Call load_checkpoint via the public entrypoint. Assert that the
        sentinel file is NEVER created (regardless of whether load succeeds
        or raises).
        """
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        sentinel = tmp_path / "pwned_model"
        assert not sentinel.exists(), "sentinel should not pre-exist"

        # Write the malicious model file at the path load_checkpoint expects:
        # model_{step:06d}.pt.
        malicious_model_path = ckpt_dir / "model_000000.pt"
        _write_malicious_torch_archive(malicious_model_path, sentinel)

        # Also write a minimal valid meta file so load_checkpoint can reach
        # the torch.load line without short-circuiting on missing meta.
        meta_path = ckpt_dir / "meta_000000.json"
        meta_path.write_text(json.dumps({"step": 0}), encoding="utf-8")

        # Invoke the real load path. A hardened loader will raise before
        # executing the payload; an un-hardened loader executes os.system.
        with pytest.raises(pickle.UnpicklingError):
            load_checkpoint(str(ckpt_dir), 0, torch.device("cpu"))

        # CRITICAL assertion: the malicious touch command MUST NOT have run.
        assert not sentinel.exists(), (
            f"SECURITY REGRESSION: malicious pickle executed os.system and "
            f"created {sentinel}. torch.load is not pinned to weights_only=True."
        )

    def test_checkpoint_load_refuses_malicious_optimizer_pickle(self, tmp_path):
        """Same attack but embedded in the optimizer file."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        sentinel = tmp_path / "pwned_opt"
        assert not sentinel.exists()

        # Legit model file, malicious optimizer file.
        model_path = ckpt_dir / "model_000000.pt"
        torch.save({"w": torch.zeros(3)}, model_path)

        opt_path = ckpt_dir / "optim_000000_rank0.pt"
        _write_malicious_torch_archive(opt_path, sentinel)

        meta_path = ckpt_dir / "meta_000000.json"
        meta_path.write_text(json.dumps({"step": 0}), encoding="utf-8")

        with pytest.raises(pickle.UnpicklingError):
            load_checkpoint(
                str(ckpt_dir),
                0,
                torch.device("cpu"),
                load_optimizer=True,
                rank=0,
            )

        assert not sentinel.exists(), (
            f"SECURITY REGRESSION: malicious optimizer pickle executed "
            f"os.system and created {sentinel}."
        )

    def test_tokenizer_get_token_bytes_refuses_malicious_pickle(
        self, tmp_path, monkeypatch
    ):
        """nanochat.tokenizer.get_token_bytes must also refuse a pickle bomb."""
        # get_token_bytes looks at ``{base_dir}/tokenizer/token_bytes.pt``.
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        sentinel = tmp_path / "pwned_tokenizer"
        assert not sentinel.exists()

        tb_path = tokenizer_dir / "token_bytes.pt"
        _write_malicious_torch_archive(tb_path, sentinel)

        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

        from nanochat.tokenizer import get_token_bytes

        with pytest.raises(pickle.UnpicklingError):
            get_token_bytes(device="cpu")

        assert not sentinel.exists(), (
            "SECURITY REGRESSION: tokenizer.get_token_bytes executed "
            "malicious pickle payload."
        )


# ---------------------------------------------------------------------------
# Test 2: AST-check every torch.load site pins weights_only
# ---------------------------------------------------------------------------


class TestAllTorchLoadSitesHardened:
    """Every torch.load call in the non-test codebase must pin weights_only."""

    # Files under test. tests/ itself is excluded because tests construct
    # controlled fixtures; however, we include top-level production code:
    # nanochat/, steering/, scripts/, dev/, tasks/.
    PRODUCTION_DIRS: tuple[str, ...] = (
        "nanochat",
        "steering",
        "scripts",
        "dev",
        "tasks",
    )

    def _production_py_files(self) -> list[Path]:
        files: list[Path] = []
        for directory in self.PRODUCTION_DIRS:
            root = REPO_ROOT / directory
            if not root.exists():
                continue
            files.extend(root.rglob("*.py"))
        return files

    def test_torch_load_sites_use_weights_only(self):
        """Every torch.load call in production code passes weights_only=True.

        Exceptions MUST be registered in ``ALLOWED_WEIGHTS_ONLY_FALSE`` with
        a reason, AND the source line MUST include a comment beginning with
        ``torch.load weights_only=False is required``.
        """
        offenders: list[str] = []
        for path, lineno, call in _collect_torch_load_calls(
            self._production_py_files()
        ):
            rel = path.relative_to(REPO_ROOT).as_posix()
            key = (rel, lineno)

            wo_node = _find_weights_only_kwarg(call)

            if wo_node is None:
                # Torch default in 2.6+ is True, but we require explicit
                # pinning for auditability - any future version downgrade
                # must not silently regress.
                if key in ALLOWED_WEIGHTS_ONLY_FALSE:
                    continue
                offenders.append(
                    f"{rel}:{lineno}: torch.load() missing explicit "
                    f"weights_only kwarg"
                )
                continue

            # Literal True is the only acceptable value unless allowlisted.
            is_true = (
                isinstance(wo_node, ast.Constant) and wo_node.value is True
            )
            is_false = (
                isinstance(wo_node, ast.Constant) and wo_node.value is False
            )

            if is_true:
                continue

            if is_false:
                if key not in ALLOWED_WEIGHTS_ONLY_FALSE:
                    offenders.append(
                        f"{rel}:{lineno}: torch.load(weights_only=False) "
                        f"but site is not in ALLOWED_WEIGHTS_ONLY_FALSE"
                    )
                    continue
                # Verify the documented comment is present in source.
                source_lines = path.read_text(encoding="utf-8").splitlines()
                # Look within 3 lines above and on the call line for the marker.
                window = source_lines[max(0, lineno - 4) : lineno]
                joined = "\n".join(window)
                if "torch.load weights_only=False is required" not in joined:
                    offenders.append(
                        f"{rel}:{lineno}: allowlisted weights_only=False "
                        f"lacks required '# torch.load weights_only=False is "
                        f"required ...' comment"
                    )
                continue

            # Non-literal (e.g. variable expression) is not auditable.
            offenders.append(
                f"{rel}:{lineno}: torch.load(weights_only=<non-literal>) "
                f"is not statically auditable"
            )

        assert not offenders, (
            "torch.load hardening regression:\n  "
            + "\n  ".join(offenders)
        )


# ---------------------------------------------------------------------------
# Test 3: regression - valid checkpoint still loads
# ---------------------------------------------------------------------------


class TestCleanCheckpointStillLoads:
    """Pinning weights_only=True must not break legitimate checkpoints."""

    def test_load_model_accepts_clean_checkpoint(self, tmp_path):
        """A plain tensor state_dict saved via save_checkpoint reloads OK."""
        model = _make_gpt()
        state = model.state_dict()
        step = 0
        meta = {
            "step": step,
            "model_config": asdict(_make_gpt_config()),
            "user_config": {
                "depth": 2,
                "aspect_ratio": 32,
                "head_dim": 32,
                "max_seq_len": 128,
            },
            "val_bpb": 3.14,
        }
        save_checkpoint(str(tmp_path), step, state, None, meta, rank=0)

        loaded_state, opt_state, loaded_meta = load_checkpoint(
            str(tmp_path), step, torch.device("cpu")
        )
        assert opt_state is None
        assert loaded_meta["step"] == step
        assert set(loaded_state.keys()) == set(state.keys())
        for key, val in state.items():
            assert torch.equal(val, loaded_state[key]), f"Mismatch on {key}"

    def test_load_model_accepts_clean_checkpoint_with_optimizer(self, tmp_path):
        """Clean model + optimizer state loads via load_checkpoint."""
        model = _make_gpt()
        opt_sd = {
            "state": {
                0: {
                    "step": torch.tensor(5.0),
                    "exp_avg": torch.randn(4),
                    "exp_avg_sq": torch.rand(4),
                }
            },
            "param_groups": [
                {
                    "lr": 1e-3,
                    "betas": (0.9, 0.999),
                    "weight_decay": 0.0,
                    "eps": 1e-8,
                    "params": [0],
                }
            ],
        }
        step = 1
        meta = {
            "step": step,
            "model_config": asdict(_make_gpt_config()),
            "user_config": {
                "depth": 2,
                "aspect_ratio": 32,
                "head_dim": 32,
                "max_seq_len": 128,
            },
            "val_bpb": 3.14,
        }
        save_checkpoint(str(tmp_path), step, model.state_dict(), opt_sd, meta, rank=0)

        _, loaded_opt, _ = load_checkpoint(
            str(tmp_path), step, torch.device("cpu"), load_optimizer=True
        )
        assert loaded_opt is not None
        assert set(loaded_opt["state"].keys()) == set(opt_sd["state"].keys())
        for pid, entry in opt_sd["state"].items():
            loaded_entry = loaded_opt["state"][pid]
            for field, val in entry.items():
                loaded_val = loaded_entry[field]
                if isinstance(val, torch.Tensor):
                    assert torch.equal(val, loaded_val), (
                        f"Optim mismatch: param {pid}, field {field}"
                    )
                else:
                    assert val == loaded_val


class TestLoadOptimizerStateWorks:
    """Dedicated regression for optimizer state dict load (the path most
    likely to break if a PyTorch safe-globals gap is hit)."""

    def test_load_optimizer_state_works(self, tmp_path):
        model = _make_gpt()
        opt_sd = {
            "state": {
                i: {
                    "step": torch.tensor(float(i)),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
                for i, p in enumerate(model.parameters())
            },
            "param_groups": [
                {
                    "lr": 1e-3,
                    "betas": (0.9, 0.999),
                    "weight_decay": 0.01,
                    "eps": 1e-8,
                    "params": list(range(len(list(model.parameters())))),
                }
            ],
        }
        step = 7
        meta = {
            "step": step,
            "model_config": asdict(_make_gpt_config()),
            "user_config": {
                "depth": 2,
                "aspect_ratio": 32,
                "head_dim": 32,
                "max_seq_len": 128,
            },
            "val_bpb": 2.71,
        }
        save_checkpoint(
            str(tmp_path), step, model.state_dict(), opt_sd, meta, rank=0
        )

        _, loaded_opt, _ = load_checkpoint(
            str(tmp_path),
            step,
            torch.device("cpu"),
            load_optimizer=True,
            rank=0,
        )
        assert loaded_opt is not None
        assert "state" in loaded_opt
        assert "param_groups" in loaded_opt
        assert loaded_opt["param_groups"][0]["lr"] == 1e-3
        assert loaded_opt["param_groups"][0]["betas"] == (0.9, 0.999)


# ---------------------------------------------------------------------------
# Test 4: enumerate all call sites (informational, printed on failure)
# ---------------------------------------------------------------------------


class TestEnumerateTorchLoadSites:
    """Informational: print every torch.load site with its kwarg status.

    This test always passes; its purpose is to make the audit visible in
    pytest -v output so a reviewer can confirm completeness.
    """

    def test_enumerate_torch_load_sites(self, capsys):
        src_roots = [REPO_ROOT / d for d in (
            "nanochat",
            "steering",
            "scripts",
            "dev",
            "tasks",
            "tests",
        )]
        py_files = [p for root in src_roots if root.exists() for p in root.rglob("*.py")]
        sites = _collect_torch_load_calls(py_files)
        lines = []
        for path, lineno, call in sites:
            rel = path.relative_to(REPO_ROOT).as_posix()
            wo = _find_weights_only_kwarg(call)
            if wo is None:
                status = "MISSING"
            elif isinstance(wo, ast.Constant):
                status = f"weights_only={wo.value}"
            else:
                status = "weights_only=<expr>"
            lines.append(f"  {rel}:{lineno}: {status}")
        print("\n".join(["torch.load call inventory:", *lines]))
        assert len(sites) > 0, "no torch.load sites found - AST walker broken"
