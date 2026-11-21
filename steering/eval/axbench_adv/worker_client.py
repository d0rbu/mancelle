"""Root-side owner of one persistent vendored AxBench worker process."""

from __future__ import annotations

import hashlib
import math
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import torch as t
from beartype import beartype
from safetensors import SafetensorError, safe_open

from scripts.eval.axbench_adv.soft_prompt_tensors import (
    validate_soft_embeddings,
    write_soft_embeddings,
)
from scripts.eval.axbench_adv.vendored.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_frame,
    write_frame,
)
from steering._common.strict_json import string_keyed_mapping
from steering.eval.axbench_adv.attacked_prompts import ATTACK_COLUMN
from steering.eval.axbench_adv.counter import CounterDirection
from steering.eval.axbench_adv.errors import AXBENCH_ADV_BEARTYPE, AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.prompt_surgery import (
    CONTROL_TOKENS_BY_MARKER,
    compose_suffix,
    split_terminal_marker,
)
from steering.eval.axbench_adv.seeding import row_seed
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS, ROOT_METHODS
from steering.refined_types import (
    NonNegativeIntLike,
    UniqueNonEmptyNonNegativeIntSequenceLike,
)

if TYPE_CHECKING:
    _RawEvaluationsLike = tuple[RawEvaluation, ...]
else:
    # Avoid beartype's randomized sampling of parameterized return containers.
    _RawEvaluationsLike = tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "vendor" / "axbench"
WORKER_MODULE = "scripts.eval.axbench_adv.vendored.bootstrap"
DEFAULT_TIMEOUT_SECONDS = 120.0
_TERMINATE_GRACE_SECONDS = 2.0


@beartype(conf=AXBENCH_ADV_BEARTYPE)
@dataclass(frozen=True)
class WorkerExample:
    """The two source prompts needed to evaluate one fixed input row."""

    input_id: NonNegativeIntLike
    input: str
    steered_input: str


@dataclass(frozen=True)
class GcgPrepared:
    """Validated immutable search state returned by one worker prepare call."""

    embedding_weight: t.Tensor
    allowed_token_ids: tuple[int, ...]
    seed_token_ids: tuple[int, ...]
    member_count: int
    identity: Mapping[str, Any]
    layouts_sha256: str
    direction_calibration: Mapping[str, Any] | None
    forbidden_control_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class GcgFilteredCandidate:
    token_ids: tuple[int, ...]
    text: str
    valid: bool
    error: str | None


def _lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_tensor_sidecar(
    raw: object, *, expected_path: Path, expected_key: str, expected_rank: int
) -> t.Tensor:
    sidecar = string_keyed_mapping(raw)
    if sidecar is None or set(sidecar) != {"shape", "sha256"}:
        raise AxBenchAdvError("GCG tensor sidecar descriptor is malformed")
    shape = sidecar["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != expected_rank
        or any(type(size) is not int or size <= 0 for size in shape)
        or not _lower_sha256(sidecar["sha256"])
    ):
        raise AxBenchAdvError("GCG tensor sidecar identity is invalid")
    expected_shape = tuple(cast(list[int], shape))
    try:
        stat_result = expected_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AxBenchAdvError(f"GCG tensor sidecar is unavailable: {expected_path}") from exc
    if not stat.S_ISREG(stat_result.st_mode):
        raise AxBenchAdvError("GCG tensor sidecar must be a regular file")
    if stat_result.st_uid != os.getuid() or stat_result.st_mode & 0o077:
        raise AxBenchAdvError("GCG tensor sidecar must be private to the current user")
    try:
        actual_sha256 = _sha256_path(expected_path)
    except OSError as exc:
        raise AxBenchAdvError(f"GCG tensor sidecar is unavailable: {expected_path}") from exc
    if actual_sha256 != sidecar["sha256"]:
        raise AxBenchAdvError("GCG tensor sidecar SHA-256 differs")
    try:
        with safe_open(expected_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {expected_key}:
                raise AxBenchAdvError("GCG tensor sidecar key is invalid")
            tensor = handle.get_tensor(expected_key)
    except (OSError, SafetensorError) as exc:
        raise AxBenchAdvError("GCG tensor sidecar is not valid safetensors") from exc
    if tensor.dtype is not t.float32 or tuple(tensor.shape) != expected_shape:
        raise AxBenchAdvError("GCG tensor sidecar tensor identity is invalid")
    return tensor


def _int_list(value: object, *, label: str, nonempty: bool = True) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise AxBenchAdvError(f"{label} must be a list of nonnegative token ids")
    return tuple(cast(list[int], value))


def _gradient_result(
    result: Mapping[str, Any],
    expected_path: Path,
    member_indices: Sequence[int],
    token_count: int,
    embedding_dim: int,
) -> list[t.Tensor]:
    if set(result) != {"gradient_sidecar", "member_indices"}:
        raise AxBenchAdvError("worker gradient result is malformed")
    echoed = _int_list(result["member_indices"], label="gradient member_indices")
    if echoed != tuple(member_indices):
        raise AxBenchAdvError("worker gradient changed member order")
    gradient = _load_tensor_sidecar(
        result["gradient_sidecar"],
        expected_path=expected_path,
        expected_key="gradients",
        expected_rank=3,
    )
    if tuple(gradient.shape) != (len(member_indices), token_count, embedding_dim):
        raise AxBenchAdvError("worker gradient tensor shape is invalid")
    if not bool(t.isfinite(gradient).all()):
        raise AxBenchAdvError("worker gradient contains non-finite values")
    return list(gradient.unbind(0))


def _raw_evaluation(result: object, *, input_id: int) -> RawEvaluation:
    expected = {
        "concept_rating",
        "fluency_rating",
        "generation",
        "input_id",
        "judge_rationale",
        "paid_judge_cache_misses",
    }
    result_mapping = string_keyed_mapping(result)
    if result_mapping is None or set(result_mapping) != expected:
        raise AxBenchAdvError("worker evaluate result fields are malformed")
    result = result_mapping
    result_input_id = result["input_id"]
    if type(result_input_id) is not int or result_input_id < 0:
        raise AxBenchAdvError("worker evaluate result input_id must be a non-negative integer")
    if result_input_id != input_id:
        raise AxBenchAdvError("worker evaluate result input_id mismatch")
    generation = result["generation"]
    rationale = result["judge_rationale"]
    if not isinstance(generation, str) or not isinstance(rationale, str):
        raise AxBenchAdvError("worker generation and rationale must be strings")
    ratings: list[float] = []
    for field in ("concept_rating", "fluency_rating"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxBenchAdvError(f"worker {field} must be numeric")
        rating = float(value)
        if not math.isfinite(rating) or not 0.0 <= rating <= 2.0:
            raise AxBenchAdvError(f"worker {field} must be finite in [0, 2]")
        ratings.append(rating)
    paid = result["paid_judge_cache_misses"]
    if isinstance(paid, bool) or not isinstance(paid, int) or not 0 <= paid <= 2:
        raise AxBenchAdvError("worker paid_judge_cache_misses must be in [0, 2]")
    return RawEvaluation(generation, ratings[0], ratings[1], rationale, paid)


def build_worker_command() -> list[str]:
    """Return the fixed Python-3.12 vendored-environment worker command."""

    return [
        "uv",
        "run",
        "--project",
        str(VENDOR_ROOT),
        "--frozen",
        "python",
        "-m",
        WORKER_MODULE,
    ]


class _FramedProcess:
    """Synchronous, single-request-at-a-time framed subprocess."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float | None,
        stderr_path: Path | None,
        cleanup_callback: Callable[[], None] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise AxBenchAdvError("worker timeout_seconds must be finite and positive")
        self._timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self._next_id = 1
        self._closed = False
        self._cleanup_callback = cleanup_callback
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if stderr_path is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="axbench-adv-worker-")
            self.stderr_path = Path(self._temporary.name) / "stderr.log"
        else:
            self.stderr_path = Path(stderr_path).resolve()
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = self.stderr_path.open("xb")
        try:
            self._process = subprocess.Popen(
                list(command),
                cwd=Path(cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                bufsize=0,
                start_new_session=True,
                env=None if env is None else dict(env),
            )
        except BaseException as exc:
            try:
                stderr_handle.close()
            except BaseException as cleanup_exc:
                exc.add_note(f"stderr cleanup failed: {cleanup_exc}")
            try:
                self._cleanup()
            except BaseException as cleanup_exc:
                exc.add_note(f"temporary cleanup failed: {cleanup_exc}")
            raise
        stderr_handle.close()
        self._stdin = cast(BinaryIO, self._process.stdin)
        self._stdout = cast(BinaryIO, self._process.stdout)
        self._deadline: float | None = None

    def _cleanup(self) -> None:
        errors: list[BaseException] = []
        temporary = self._temporary
        self._temporary = None
        if temporary is not None:
            try:
                temporary.cleanup()
            except BaseException as exc:
                errors.append(exc)
        callback = self._cleanup_callback
        self._cleanup_callback = None
        if callback is not None:
            try:
                callback()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def _stderr_tail(self) -> str:
        try:
            return self.stderr_path.read_text(encoding="utf-8", errors="replace")[-4_000:]
        except OSError:
            return "<stderr unavailable>"

    def _context(self) -> str:
        return f"worker_exit={self._process.poll()}, stderr_tail={self._stderr_tail()!r}"

    def _read_exact(self, _reader: BinaryIO, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        descriptor = self._stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while remaining:
                wait_seconds = None if self._deadline is None else self._deadline - time.monotonic()
                if wait_seconds is not None and wait_seconds <= 0:
                    raise TimeoutError("worker response timed out")
                if not selector.select(wait_seconds):
                    raise TimeoutError("worker response timed out")
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    raise EOFError(f"worker stdout closed with {remaining} bytes left")
                chunks.append(chunk)
                remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _validate_response(
        response: object,
        *,
        request_id: int,
        operation: str,
    ) -> dict[str, Any]:
        response_mapping = string_keyed_mapping(response)
        if response_mapping is None:
            raise ProtocolError("worker response must be a JSON object")
        response = response_mapping
        status = response.get("status")
        expected_keys = (
            {"error", "id", "op", "status", "version"}
            if status == "error"
            else {"id", "op", "result", "status", "version"}
        )
        if set(response) != expected_keys:
            raise ProtocolError("worker response fields do not match the protocol")
        if type(response["version"]) is not int or response["version"] != PROTOCOL_VERSION:
            raise ProtocolError("worker response protocol version mismatch")
        if type(response["id"]) is not int or response["id"] != request_id:
            raise ProtocolError("worker response id mismatch")
        if response["op"] != operation:
            raise ProtocolError("worker response operation mismatch")
        if status == "error":
            error = string_keyed_mapping(response["error"])
            if (
                error is None
                or set(error) != {"message", "type"}
                or not all(isinstance(error[key], str) for key in error)
            ):
                raise ProtocolError("worker error response is malformed")
            raise AxBenchAdvError(
                f"worker {operation!r} failed with {error['type']}: {error['message']}"
            )
        result = string_keyed_mapping(response["result"])
        if status != "ok" or result is None:
            raise ProtocolError("worker success response is malformed")
        return result

    def request(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise AxBenchAdvError("worker process is closed")
        request_id = self._next_id
        self._next_id += 1
        try:
            write_frame(
                self._stdin,
                {
                    "id": request_id,
                    "op": operation,
                    "payload": dict(payload),
                    "version": PROTOCOL_VERSION,
                },
            )
            self._deadline = (
                None if self._timeout_seconds is None else time.monotonic() + self._timeout_seconds
            )
            response = decode_frame(self._stdout, read_exact_fn=self._read_exact)
            return self._validate_response(
                response,
                request_id=request_id,
                operation=operation,
            )
        except BaseException as exc:
            self._abort()
            if isinstance(exc, AxBenchAdvError):
                raise
            raise AxBenchAdvError(
                f"worker {operation!r} protocol failure: {type(exc).__name__}: {exc}; "
                f"{self._last_context}"
            ) from exc

    def _abort(self) -> None:
        if self._closed:
            return
        cleanup_errors: list[BaseException] = []
        try:
            if self._process.poll() is None:
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=_TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self._process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except BaseException as exc:
            cleanup_errors.append(exc)
        finally:
            try:
                self._last_context = self._context()
            except BaseException as exc:
                cleanup_errors.append(exc)
                self._last_context = "worker context unavailable"
            for stream in (self._stdin, self._stdout):
                try:
                    stream.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            self._closed = True
            try:
                self._cleanup()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            details = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            self._last_context = f"{self._last_context}; cleanup_errors={details}"

    def close(self) -> None:
        if self._closed:
            return
        result = self.request("close", {})
        if result:
            self._abort()
            raise AxBenchAdvError("worker close response result must be empty")
        try:
            return_code = self._process.wait(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self._abort()
            raise AxBenchAdvError(f"worker did not exit after close; {self._last_context}") from exc
        if return_code != 0:
            context = self._context()
            self._abort()
            raise AxBenchAdvError(f"worker exited nonzero after close; {context}")
        cleanup_errors: list[BaseException] = []
        for stream in (self._stdin, self._stdout):
            try:
                stream.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        self._closed = True
        try:
            self._cleanup()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            raise cleanup_errors[0]

    def __enter__(self) -> _FramedProcess:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        if exc_type is None:
            self.close()
        else:
            self._abort()


class AxBenchAdvWorkerClient:
    """Bind source rows to one exact method/concept/factor worker."""

    _raw_evaluation = staticmethod(_raw_evaluation)

    def __init__(
        self,
        *,
        init_payload: Mapping[str, Any],
        examples: Sequence[WorkerExample],
        master_seed: int,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        stderr_path: Path | None = None,
    ) -> None:
        self._closed = False
        self._soft_prepared: GcgPrepared | None = None
        method = init_payload.get("method")
        concept_id = init_payload.get("concept_id")
        factor = init_payload.get("factor")
        if not isinstance(method, str) or method not in ATTACK_COLUMN:
            raise AxBenchAdvError(f"unknown worker method {method!r}")
        if isinstance(concept_id, bool) or not isinstance(concept_id, int):
            raise AxBenchAdvError("worker concept_id must be an integer")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise AxBenchAdvError("worker factor must be numeric")
        numeric_factor = float(factor)
        if not math.isfinite(numeric_factor):
            raise AxBenchAdvError("worker factor must be finite")
        self._method = method
        self._concept_id = concept_id
        self._factor = numeric_factor
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or not 0 <= master_seed <= (1 << 63) - 1
        ):
            raise AxBenchAdvError("worker master_seed must be in [0, 2^63 - 1]")
        self._master_seed = master_seed
        materialized = list(examples)
        if not materialized:
            raise AxBenchAdvError("worker examples must not be empty")
        self._examples: dict[int, WorkerExample] = {}
        for example in materialized:
            if not isinstance(example, WorkerExample):
                raise AxBenchAdvError("worker examples must be WorkerExample values")
            if example.input_id in self._examples:
                raise AxBenchAdvError(f"duplicate worker input_id {example.input_id}")
            self._examples[example.input_id] = example
        self._gcg_sidecar_temp = tempfile.TemporaryDirectory(prefix="axbench-gcg-root-")
        sidecar_dir = Path(self._gcg_sidecar_temp.name).resolve()
        try:
            os.chmod(sidecar_dir, 0o700)
            self._process = _FramedProcess(
                build_worker_command(),
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
                stderr_path=stderr_path,
                cleanup_callback=self._gcg_sidecar_temp.cleanup,
            )
            worker_payload = {**init_payload, "gcg_sidecar_dir": str(sidecar_dir)}
            identity = self._process.request("init", worker_payload)
            expected_identity = {
                "concept_id": self._concept_id,
                "factor": self._factor,
                "method": self._method,
            }
            if (
                set(identity) != set(expected_identity)
                or type(identity.get("concept_id")) is not int
                or type(identity.get("factor")) is not float
                or not isinstance(identity.get("method"), str)
                or identity != expected_identity
            ):
                raise AxBenchAdvError(
                    f"worker init identity mismatch: expected {expected_identity}, got {identity}"
                )
        except BaseException:
            process = getattr(self, "_process", None)
            if process is not None:
                process._abort()
            self._gcg_sidecar_temp.cleanup()
            self._closed = True
            raise

    @beartype(conf=AXBENCH_ADV_BEARTYPE)
    def evaluate(
        self,
        suffix: str,
        input_ids: UniqueNonEmptyNonNegativeIntSequenceLike,
    ) -> _RawEvaluationsLike:
        """Evaluate distinct rows sequentially so every row gets its own seed reset."""

        if self._closed:
            raise AxBenchAdvError("worker client is closed")
        return self._evaluate(suffix, input_ids)

    def evaluate_counter(
        self, counter: CounterDirection, magnitude: float, input_ids: Sequence[int]
    ) -> _RawEvaluationsLike:
        return self._evaluate("", input_ids, counter={**counter.provenance, "magnitude": magnitude})

    def prepare_gcg(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str = "\n\n",
    ) -> GcgPrepared:
        """Prepare one fixed-k search cell and validate its tensor sidecar."""

        return self._prepare_search(
            objective=objective,
            alpha=alpha,
            rows=rows,
            suffix_token_count=suffix_token_count,
            device_batch_size=device_batch_size,
            separator=separator,
            operation="gcg_prepare",
        )

    def prepare_soft_prompt(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str = "\n\n",
    ) -> GcgPrepared:
        prepared = self._prepare_search(
            objective=objective,
            alpha=alpha,
            rows=rows,
            suffix_token_count=suffix_token_count,
            device_batch_size=device_batch_size,
            separator=separator,
            operation="soft_prepare",
        )
        layout = prepared.identity.get("embedding_layout")
        if (
            not isinstance(layout, list)
            or len(layout) != 1
            or not isinstance(layout[0], dict)
            or set(layout[0]) != {"name", "width", "dtype"}
            or layout[0]["name"] != "input_embeddings"
            or type(layout[0]["width"]) is not int
            or layout[0]["width"] != prepared.embedding_weight.shape[1]
            or not isinstance(layout[0]["dtype"], str)
            or layout[0]["dtype"]
            not in {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
            or prepared.identity.get("generation_conditioning") != "ordinary_prompt"
        ):
            raise AxBenchAdvError("worker soft prepare embedding layout is malformed")
        self._soft_prepared = prepared
        return prepared

    def _prepare_search(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str,
        operation: str,
    ) -> GcgPrepared:
        self._soft_prepared = None

        expected_path = Path(self._gcg_sidecar_temp.name).resolve() / "input_embeddings.safetensors"
        expected_path.unlink(missing_ok=True)
        result = self._process.request(
            operation,
            {
                "objective": objective,
                "alpha": alpha,
                "rows": [dict(row) for row in rows],
                "suffix_token_count": suffix_token_count,
                "device_batch_size": device_batch_size,
                "separator": separator,
            },
        )
        expected = {
            "embedding_sidecar",
            "allowed_token_ids",
            "seed_token_ids",
            "member_count",
            "identity",
            "layouts_sha256",
            "direction_calibration",
        }
        if set(result) != expected:
            raise AxBenchAdvError("worker GCG prepare result fields are malformed")
        allowed = _int_list(result["allowed_token_ids"], label="allowed_token_ids")
        seed = _int_list(
            result["seed_token_ids"],
            label="seed_token_ids",
            nonempty=suffix_token_count > 0,
        )
        if len(seed) != suffix_token_count:
            raise AxBenchAdvError("worker GCG seed token count differs from requested k")
        member_count = result["member_count"]
        identity = string_keyed_mapping(result["identity"])
        calibration_raw = result["direction_calibration"]
        calibration = None if calibration_raw is None else string_keyed_mapping(calibration_raw)
        if (
            type(member_count) is not int
            or member_count != len(rows)
            or identity is None
            or not _lower_sha256(result["layouts_sha256"])
            or (calibration_raw is not None and calibration is None)
        ):
            raise AxBenchAdvError("worker GCG prepare identity is malformed")
        if objective == "activation_l2" and calibration is None:
            raise AxBenchAdvError("activation-L2 GCG requires direction calibration")
        if objective == "response_margin" and calibration is not None:
            raise AxBenchAdvError("response-margin GCG must not return direction calibration")
        embedding = _load_tensor_sidecar(
            result["embedding_sidecar"],
            expected_path=expected_path,
            expected_key="embedding_weight",
            expected_rank=2,
        )
        vocab_size, embedding_dim = embedding.shape
        if (
            identity.get("vocab_size") != vocab_size
            or ("embedding_dim" in identity and identity.get("embedding_dim") != embedding_dim)
            or len(set(allowed)) != len(allowed)
            or any(token_id >= vocab_size for token_id in allowed)
            or any(token_id >= vocab_size for token_id in seed)
        ):
            raise AxBenchAdvError("worker GCG tokens are invalid for the embedding snapshot")
        self._gcg_embedding_dim = int(embedding_dim)
        self._gcg_member_count = member_count
        prompt = getattr(next(iter(self._examples.values())), ATTACK_COLUMN[self._method])
        _, marker = split_terminal_marker(prompt)
        return GcgPrepared(
            embedding_weight=embedding,
            allowed_token_ids=allowed,
            seed_token_ids=seed,
            member_count=member_count,
            identity=dict(identity),
            layouts_sha256=result["layouts_sha256"],
            direction_calibration=None if calibration is None else dict(calibration),
            forbidden_control_tokens=CONTROL_TOKENS_BY_MARKER[marker],
        )

    def gcg_filter(
        self,
        candidate_token_ids: Sequence[Sequence[int]],
    ) -> tuple[GcgFilteredCandidate, ...]:
        result = self._process.request(
            "gcg_filter",
            {"candidate_token_ids": [list(tokens) for tokens in candidate_token_ids]},
        )
        if set(result) != {"candidates"} or not isinstance(result["candidates"], list):
            raise AxBenchAdvError("worker GCG filter result is malformed")
        filtered: list[GcgFilteredCandidate] = []
        for index, raw in enumerate(result["candidates"]):
            row = string_keyed_mapping(raw)
            if row is None or set(row) != {"token_ids", "text", "valid", "error"}:
                raise AxBenchAdvError(f"worker GCG filter candidate {index} is malformed")
            ids = _int_list(row["token_ids"], label=f"filter candidate {index}", nonempty=False)
            text_value, valid, error = row["text"], row["valid"], row["error"]
            if (
                not isinstance(text_value, str)
                or type(valid) is not bool
                or (valid and error is not None)
                or (not valid and (not isinstance(error, str) or not error))
                or ids != tuple(candidate_token_ids[index])
            ):
                raise AxBenchAdvError(f"worker GCG filter candidate {index} fields are invalid")
            filtered.append(
                GcgFilteredCandidate(
                    ids,
                    text_value,
                    valid,
                    error if isinstance(error, str) else None,
                )
            )
        if len(filtered) != len(candidate_token_ids):
            raise AxBenchAdvError("worker GCG filter changed candidate cardinality")
        return tuple(filtered)

    def gcg_score(
        self,
        candidate_token_ids: Sequence[Sequence[int]],
        member_indices: Sequence[int],
        *,
        candidate_eval_device_batch_size: int,
    ) -> list[float]:
        result = self._process.request(
            "gcg_score",
            {
                "candidate_token_ids": [list(tokens) for tokens in candidate_token_ids],
                "member_indices": list(member_indices),
                "candidate_eval_device_batch_size": candidate_eval_device_batch_size,
            },
        )
        expected = {"candidate_count", "member_count", "losses"}
        losses = result.get("losses")
        if (
            set(result) != expected
            or result["candidate_count"] != len(candidate_token_ids)
            or result["member_count"] != len(member_indices)
            or not isinstance(losses, list)
            or len(losses) != len(candidate_token_ids) * len(member_indices)
        ):
            raise AxBenchAdvError("worker GCG score result shape is malformed")
        normalized: list[float] = []
        for value in losses:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AxBenchAdvError("worker GCG score loss must be numeric")
            scalar = float(value)
            if not math.isfinite(scalar):
                raise AxBenchAdvError("worker GCG score loss must be finite")
            normalized.append(scalar)
        return normalized

    def gcg_gradient(
        self,
        token_ids: Sequence[int],
        member_indices: Sequence[int],
    ) -> list[t.Tensor]:
        embedding_dim = getattr(self, "_gcg_embedding_dim", None)
        member_count = getattr(self, "_gcg_member_count", None)
        if type(embedding_dim) is not int or type(member_count) is not int:
            raise AxBenchAdvError("worker GCG gradient requires a successful prepare")
        if (
            not member_indices
            or len(set(member_indices)) != len(member_indices)
            or any(
                type(index) is not int or not 0 <= index < member_count for index in member_indices
            )
        ):
            raise AxBenchAdvError("worker GCG gradient member indices are invalid")
        # Preserve the historical RPC boundaries because each request owns one
        # model-state/RNG guard and may end on a partial physical forward batch.
        bytes_per_member = len(token_ids) * embedding_dim * 4
        max_chunk_members = max(1, (40 * 1024 * 1024) // max(1, bytes_per_member))
        gradients: list[t.Tensor] = []
        for start in range(0, len(member_indices), max_chunk_members):
            chunk = member_indices[start : start + max_chunk_members]
            gradients.extend(self._gcg_gradient_chunk(token_ids, chunk))
        return gradients

    def _gcg_gradient_chunk(
        self,
        token_ids: Sequence[int],
        member_indices: Sequence[int],
    ) -> list[t.Tensor]:
        expected_path = Path(self._gcg_sidecar_temp.name).resolve() / "gradients.safetensors"
        expected_path.unlink(missing_ok=True)
        result = self._process.request(
            "gcg_gradient",
            {"token_ids": list(token_ids), "member_indices": list(member_indices)},
        )
        return _gradient_result(
            result, expected_path, member_indices, len(token_ids), self._gcg_embedding_dim
        )

    def _soft_state(self) -> GcgPrepared:
        prepared = getattr(self, "_soft_prepared", None)
        if getattr(self, "_closed", False) or prepared is None:
            raise AxBenchAdvError("soft operation requires a successful soft prepare")
        return prepared

    def _soft_indices(self, member_indices: Sequence[int]) -> None:
        prepared = self._soft_state()
        if (
            not member_indices
            or len(set(member_indices)) != len(member_indices)
            or any(
                type(index) is not int or not 0 <= index < prepared.member_count
                for index in member_indices
            )
        ):
            raise AxBenchAdvError("worker soft member indices are invalid")

    def _soft_request(
        self, operation: str, embeddings: t.Tensor, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        prepared = self._soft_state()
        try:
            validate_soft_embeddings(
                embeddings, (len(prepared.seed_token_ids), prepared.embedding_weight.shape[1])
            )
        except ValueError as exc:
            raise AxBenchAdvError(str(exc)) from exc
        path = Path(self._gcg_sidecar_temp.name).resolve() / "soft_embeddings.safetensors"
        try:
            write_soft_embeddings(path, embeddings)
            return self._process.request(operation, payload)
        except BaseException:
            self._soft_prepared = None
            raise
        finally:
            path.unlink(missing_ok=True)

    def soft_score(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[float]:
        self._soft_indices(member_indices)
        result = self._soft_request(
            "soft_score", embeddings, {"member_indices": list(member_indices)}
        )
        if set(result) != {"losses", "member_indices"} or _int_list(
            result["member_indices"], label="soft score member_indices"
        ) != tuple(member_indices):
            raise AxBenchAdvError("worker soft score changed member order")
        losses = result["losses"]
        if not isinstance(losses, list) or len(losses) != len(member_indices):
            raise AxBenchAdvError("worker soft score changed member cardinality")
        if any(
            isinstance(loss, bool) or not isinstance(loss, (float, int)) or not math.isfinite(loss)
            for loss in losses
        ):
            raise AxBenchAdvError("worker soft score losses must be finite numbers")
        return [float(loss) for loss in losses]

    def soft_gradient(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[t.Tensor]:
        self._soft_indices(member_indices)
        prepared = self._soft_state()
        k, width = len(prepared.seed_token_ids), prepared.embedding_weight.shape[1]
        max_chunk_members = max(1, (40 * 1024 * 1024) // (k * width * 4))
        expected_path = Path(self._gcg_sidecar_temp.name).resolve() / "gradients.safetensors"
        gradients: list[t.Tensor] = []
        for start in range(0, len(member_indices), max_chunk_members):
            chunk = member_indices[start : start + max_chunk_members]
            expected_path.unlink(missing_ok=True)
            result = self._soft_request(
                "soft_gradient", embeddings, {"member_indices": list(chunk)}
            )
            gradients.extend(_gradient_result(result, expected_path, chunk, k, width))
        return gradients

    def evaluate_soft(self, embeddings: t.Tensor, input_ids: Sequence[int]) -> _RawEvaluationsLike:
        prepared = self._soft_state()
        if (
            not input_ids
            or len(set(input_ids)) != len(input_ids)
            or any(type(index) is not int or index not in self._examples for index in input_ids)
        ):
            raise AxBenchAdvError("worker soft evaluation input ids are invalid")
        seed_text = " ".join(["!"] * len(prepared.seed_token_ids))
        return self._evaluate(seed_text, input_ids, soft_embeddings=embeddings)

    def _evaluate(
        self,
        suffix: str,
        input_ids: Sequence[int],
        counter: Mapping[str, Any] | None = None,
        *,
        soft_embeddings: t.Tensor | None = None,
    ) -> _RawEvaluationsLike:
        if self._closed:
            raise AxBenchAdvError("worker client is closed")
        target_column = ATTACK_COLUMN[self._method]
        evaluations: list[RawEvaluation] = []
        for input_id in input_ids:
            if input_id not in self._examples:
                raise AxBenchAdvError(f"unknown worker input_id {input_id!r}")
            example = self._examples[input_id]
            unattacked = getattr(example, target_column)
            attacked = compose_suffix(unattacked, suffix)
            payload = {
                "input": attacked if target_column == "input" else example.input,
                "input_id": input_id,
                "row_seed": row_seed(
                    master_seed=self._master_seed,
                    method=self._method,
                    concept_id=self._concept_id,
                    input_id=input_id,
                    factor=self._factor,
                    unattacked_prompt=unattacked,
                ),
                "steered_input": (
                    attacked if target_column == "steered_input" else example.steered_input
                ),
            }
            if counter is not None:
                payload["counter"] = dict(counter)
            evaluations.append(
                _raw_evaluation(
                    self._process.request("evaluate", payload)
                    if soft_embeddings is None
                    else self._soft_request("evaluate_soft", soft_embeddings, payload),
                    input_id=input_id,
                )
            )
        return tuple(evaluations)

    def close(self) -> None:
        try:
            if not self._closed:
                self._process.close()
        finally:
            self._closed = True
            self._soft_prepared = None

    def __enter__(self) -> AxBenchAdvWorkerClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        if exc_type is None:
            self.close()
        else:
            try:
                self._process._abort()
            finally:
                self._closed = True
                self._soft_prepared = None


def _judge_identity(init_payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "api_base_url",
        "cache_only",
        "concept",
        "concept_id",
        "factor",
        "judge_cache",
        "judge_model",
        "method",
        "runtime_kind",
    }
    if set(init_payload) not in (expected, expected | {"pair_concurrency"}):
        raise AxBenchAdvError("judge init payload fields differ from the contract")
    if init_payload.get("runtime_kind") != "judge":
        raise AxBenchAdvError("judge init payload runtime_kind must be 'judge'")
    if init_payload.get("method") not in (
        *ROOT_METHODS,
        "NativeDecoderPromptSteering",
        *ENCDEC_BASELINE_METHODS,
    ):
        raise AxBenchAdvError(
            "judge client method is not a supported project, native, or EncDec baseline"
        )
    method = init_payload["method"]
    pair_concurrency = init_payload.get("pair_concurrency", 1)
    if type(pair_concurrency) is not int or pair_concurrency not in (1, 2):
        raise AxBenchAdvError("judge pair_concurrency must be 1 or 2")
    if method not in ROOT_METHODS and pair_concurrency != 1:
        raise AxBenchAdvError("judge pair_concurrency=2 is project-only")
    concept_id = init_payload.get("concept_id")
    factor = init_payload.get("factor")
    if type(concept_id) is not int or concept_id < 0:
        raise AxBenchAdvError("judge concept_id must be a non-negative integer")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise AxBenchAdvError("judge factor must be numeric")
    numeric_factor = float(factor)
    if not math.isfinite(numeric_factor):
        raise AxBenchAdvError("judge factor must be finite")
    return {"concept_id": concept_id, "factor": numeric_factor, "method": method}


def _assert_judge_identity(
    identity: Mapping[str, Any], expected: Mapping[str, Any], op: str
) -> None:
    if (
        set(identity) != set(expected)
        or type(identity.get("concept_id")) is not int
        or type(identity.get("factor")) is not float
        or not isinstance(identity.get("method"), str)
        or identity != expected
    ):
        raise AxBenchAdvError(f"judge {op} identity mismatch: expected {expected}, got {identity}")


class AxBenchAdvJudgeClient:
    """Bind root-method concepts to one vendored exact judge process."""

    def __init__(
        self,
        *,
        init_payload: Mapping[str, Any],
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        stderr_path: Path | None = None,
    ) -> None:
        self._closed = False
        expected_identity = _judge_identity(init_payload)
        self._concept_id = expected_identity["concept_id"]
        self._factor = expected_identity["factor"]
        self._process = _FramedProcess(
            build_worker_command(),
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
            stderr_path=stderr_path,
        )
        try:
            identity = self._process.request("init", init_payload)
            _assert_judge_identity(identity, expected_identity, "init")
        except BaseException:
            self._process._abort()
            self._closed = True
            raise

    def rebind(self, init_payload: Mapping[str, Any]) -> None:
        if self._closed:
            raise AxBenchAdvError("judge client is closed")
        expected_identity = _judge_identity(init_payload)
        if expected_identity["method"] not in (*ROOT_METHODS, *ENCDEC_BASELINE_METHODS):
            raise AxBenchAdvError("judge rebind accepts only project or EncDec methods")
        try:
            identity = self._process.request("rebind", init_payload)
            _assert_judge_identity(identity, expected_identity, "rebind")
        except BaseException as exc:
            try:
                self.abort()
            except BaseException as cleanup_exc:
                exc.add_note(f"judge rebind abort failed: {cleanup_exc}")
            raise
        self._concept_id = expected_identity["concept_id"]
        self._factor = expected_identity["factor"]

    def abort(self) -> None:
        if not self._closed:
            try:
                self._process._abort()
            finally:
                self._closed = True

    @beartype(conf=AXBENCH_ADV_BEARTYPE)
    def evaluate(self, *, generation: str, input_id: NonNegativeIntLike) -> RawEvaluation:
        if self._closed:
            raise AxBenchAdvError("judge client is closed")
        try:
            generation.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise AxBenchAdvError("judge generation must be valid UTF-8") from exc
        return _raw_evaluation(
            self._process.request(
                "evaluate",
                {"generation": generation, "input_id": input_id},
            ),
            input_id=input_id,
        )

    def close(self) -> None:
        if not self._closed:
            self._process.close()
            self._closed = True

    def __enter__(self) -> AxBenchAdvJudgeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        if exc_type is None:
            self.close()
        else:
            self._process._abort()
            self._closed = True


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "AxBenchAdvJudgeClient",
    "AxBenchAdvWorkerClient",
    "WorkerExample",
    "build_worker_command",
]
