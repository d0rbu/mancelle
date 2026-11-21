#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m scripts.chat.chat_web

- 4 GPUs
python -m scripts.chat.chat_web num_gpus=4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import asyncio
import json
import logging
import os
import random
import sys
from collections.abc import AsyncGenerator
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from hydra import compose, initialize_config_dir
from loguru import logger as app_logger
from pydantic import BaseModel

from nanochat.checkpoint_manager import ModelPhase, ModelSource, load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.tokenizer_protocols import ChatTokenizer, require_chat_tokenizer

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0  # 0 disables top-k filtering, using full vocabulary
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

_config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "serving")
with initialize_config_dir(config_dir=_config_dir, version_base=None):
    _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
    cfg = compose(config_name="chat_web", overrides=_overrides)

# `logger` itself is fine to bind at module top — getLogger() has no side
# effects beyond returning a Logger instance. The actual `basicConfig()` call
# (which mutates global Python logging state) is deferred into main().
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
ptdtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16


# --- Security / abuse-prevention helpers ---------------------------------
# In-flight concurrency cap. Capped independently of the GPU worker pool so a
# flood of clients cannot drain the event loop while workers are busy.
_max_in_flight = int(getattr(cfg, "max_in_flight", 4))
if _max_in_flight < 1:
    _max_in_flight = 1
in_flight_semaphore: asyncio.Semaphore = asyncio.Semaphore(_max_in_flight)


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests when cfg.api_key is set but the header is missing/wrong.

    If cfg.api_key is None/empty, the server is open — this dependency is a
    no-op. If cfg.api_key is a non-empty string, requests without a matching
    X-API-Key header receive 401.
    """
    expected = getattr(cfg, "api_key", None)
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""

    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: ChatTokenizer
    autocast_ctx: AbstractContextManager[None]


class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: int | None = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1  # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: list[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(
        self, source: str, model_tag: str | None = None, step: int | None = None
    ):
        """Load model on each GPU."""
        app_logger.info(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", (
                "Only CUDA supports multiple workers/GPUs. cpu|mps does not."
            )

        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                app_logger.info(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type)  # e.g. cpu|mps
                app_logger.info(f"Loading model on {device_type}...")

            model, tokenizer, meta = load_model(
                ModelSource(source),
                device,
                phase=ModelPhase.EVAL,
                model_tag=model_tag,
                step=step,
                eval_conditioning=EvalConditioning.deferred("chat_web"),
            )
            reject_feature_clusters_metadata(meta, "chat_web checkpoint")
            tokenizer = require_chat_tokenizer(
                tokenizer,
                surface="chat_web checkpoint",
            )
            engine = Engine(model, tokenizer)
            autocast_ctx = (
                torch.amp.autocast(device_type=device_type, dtype=ptdtype)
                if device_type == "cuda"
                else nullcontext()
            )

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        app_logger.info(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_k: int | None = None


def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request",
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(
                status_code=400, detail=f"Message {i} has empty content"
            )

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message",
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed",
        )

    # Validate role values
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user' or 'assistant'",
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}",
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}",
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}",
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    app_logger.info("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=cfg.num_gpus)
    await app.state.worker_pool.initialize(
        cfg.source, model_tag=cfg.model_tag, step=cfg.step
    )
    app_logger.info(f"Server ready at http://localhost:{cfg.port}")
    yield


app = FastAPI(lifespan=lifespan)

# CORS: restrict to an explicit allow-list from config. Defaults to [] (no
# cross-origin browsers allowed). Operators opt in via cfg.allow_origins.
# allow_credentials is False because the API uses header-based auth, not cookies.
_allow_origins = list(getattr(cfg, "allow_origins", []) or [])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", dependencies=[Depends(require_api_key)])
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';",
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg", dependencies=[Depends(require_api_key)])
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")


async def generate_stream(
    worker: Worker, tokens, temperature=None, max_new_tokens=None, top_k=None
) -> AsyncGenerator[str]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else cfg.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_tokens
    top_k = top_k if top_k is not None else cfg.top_k

    assistant_end = worker.tokenizer.encode_special("<assistant_end>")
    bos = worker.tokenizer.get_bos_token_id()

    # The client-visible stream is the complete Engine output, including
    # forced tool markers/results. The request is stateless, so no separate
    # next-turn transcript needs to be retained here.
    display_tokens: list[int] = []
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, _token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1),
        ):
            token = token_column[0]
            # Stopping criteria
            if token in (assistant_end, bos):
                break

            display_tokens.append(token)
            current_text = worker.tokenizer.decode(display_tokens)
            if not current_text.endswith("�"):
                new_text = current_text[len(last_clean_text):]
                if new_text:
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU.

    Concurrency is capped by in_flight_semaphore (cfg.max_in_flight); additional
    clients block until a slot is free.
    """

    # Basic validation to prevent abuse
    validate_chat_request(request)

    # Cap in-flight requests to prevent DoS. Acquired BEFORE acquiring a worker
    # so that queued clients do not tie up the worker pool.
    await in_flight_semaphore.acquire()

    # Log incoming conversation to console. Content is only emitted when
    # cfg.log_content is truthy — otherwise just sizes/counts to avoid
    # persisting PII/secrets that users may paste.
    logger.info("=" * 20)
    if getattr(cfg, "log_content", False):
        for message in request.messages:
            logger.info(f"[{message.role.upper()}]: {message.content}")
    else:
        logger.info(
            f"[REQUEST] messages={len(request.messages)} "
            f"total_chars={sum(len(m.content) for m in request.messages)}"
        )
    logger.info("-" * 20)

    # Acquire a worker from the pool (will wait if all are busy)
    worker_pool = app.state.worker_pool
    try:
        worker = await worker_pool.acquire_worker()
    except BaseException:
        # Always release the semaphore if worker acquisition fails.
        in_flight_semaphore.release()
        raise

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<user_start>")
        user_end = worker.tokenizer.encode_special("<user_end>")
        assistant_start = worker.tokenizer.encode_special("<assistant_start>")
        assistant_end = worker.tokenizer.encode_special("<assistant_end>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        response_tokens = []

        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                ):
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console (gated on cfg.log_content).
                full_response = "".join(response_tokens)
                if getattr(cfg, "log_content", False):
                    logger.info(
                        f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}"
                    )
                else:
                    logger.info(
                        f"[RESPONSE] gpu={worker.gpu_id} "
                        f"chars={len(full_response)} tokens={len(response_tokens)}"
                    )
                logger.info("=" * 20)
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)
                # Release the in-flight semaphore so another client can run.
                in_flight_semaphore.release()

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")
    except Exception:
        # Make sure to release worker AND semaphore even on error
        await worker_pool.release_worker(worker)
        in_flight_semaphore.release()
        raise


@app.get("/health", dependencies=[Depends(require_api_key)])
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize()
        if worker_pool
        else 0,
    }


@app.get("/stats", dependencies=[Depends(require_api_key)])
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers)
        - worker_pool.available_workers.qsize(),
        "workers": [
            {"gpu_id": w.gpu_id, "device": str(w.device)} for w in worker_pool.workers
        ],
    }


def main():
    """Run the chat web server."""
    # Configure logging for conversation traffic. Done here (not at module
    # top) so importing this module from tests doesn't mutate Python's
    # global logging configuration.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    compute_init(device_type)
    app_logger.info("Starting NanoChat Web Server")
    app_logger.info(
        f"Temperature: {cfg.temperature}, Top-k: {cfg.top_k}, Max tokens: {cfg.max_tokens}"
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
