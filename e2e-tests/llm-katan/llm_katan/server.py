"""
FastAPI server implementation for LLM Katan

Provides OpenAI-compatible endpoints for lightweight LLM serving.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .config import ServerConfig

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "unknown"
from .model import ModelBackend, create_backend

logger = logging.getLogger(__name__)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Optional[Dict] = None


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_tokens_generated: int
    average_response_time: float
    model: str
    backend: str


# Global backend instance and metrics
backend: Optional[ModelBackend] = None
metrics = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "response_times": [],
    "start_time": time.time(),
}

# Sleep mode state
sleep_state = {
    "is_sleeping": False,
    "sleep_level": 0,
    "sleep_time": None,
    "wake_up_delay_seconds": 0.5,  # Simulated wake-up delay
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global backend
    config = app.state.config

    logger.info(f"🚀 Starting LLM Katan server with model: {config.model_name}")
    logger.info(f"🔧 Backend: {config.backend}")
    logger.info(f"📛 Served model name: {config.served_model_name}")

    # Create and load model backend
    backend = create_backend(config)
    await backend.load_model()

    logger.info("✅ LLM Katan server started successfully")
    yield

    logger.info("🛑 Shutting down LLM Katan server")
    backend = None


def create_app(config: ServerConfig) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="LLM Katan - Lightweight LLM Server",
        description="A lightweight LLM serving package for testing and development",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        # When sleeping, health check should indicate server is not ready
        if sleep_state["is_sleeping"]:
            raise HTTPException(
                status_code=503,
                detail="Server is in sleep mode"
            )
        return HealthResponse(
            status="ok",
            model=config.served_model_name,
            backend=config.backend,
        )

    @app.post("/sleep")
    async def sleep(level: int = 1):
        """
        Put the server into sleep mode (vLLM-compatible endpoint).

        Sleep levels:
        - Level 1: Offloads model weights to CPU RAM and discards KV cache
        - Level 2: Discards both model weights and KV cache

        This is a mock implementation for testing the semantic-router's
        sleep mode management functionality.
        """
        if sleep_state["is_sleeping"]:
            return {
                "status": "already_sleeping",
                "level": sleep_state["sleep_level"],
                "message": "Server is already in sleep mode"
            }

        sleep_state["is_sleeping"] = True
        sleep_state["sleep_level"] = level
        sleep_state["sleep_time"] = time.time()

        logger.info(f"😴 Server entering sleep mode (level {level})")

        return {
            "status": "sleeping",
            "level": level,
            "message": f"Server is now in sleep mode (level {level})"
        }

    @app.post("/wake_up")
    async def wake_up(tags: Optional[str] = None):
        """
        Wake up the server from sleep mode (vLLM-compatible endpoint).

        Supports optional `tags` query parameter for partial wake-up:
        - `?tags=weights` - Only restore model weights
        - `?tags=kv_cache` - Only restore KV cache
        - No tags - Full wake-up (restore everything)

        This simulates the wake-up process with a configurable delay.
        """
        if not sleep_state["is_sleeping"]:
            return {
                "status": "already_awake",
                "message": "Server is already awake"
            }

        # Simulate wake-up delay
        wake_delay = sleep_state["wake_up_delay_seconds"]
        logger.info(f"⏰ Server waking up (simulated delay: {wake_delay}s, tags: {tags})")
        await asyncio.sleep(wake_delay)

        sleep_duration = time.time() - sleep_state["sleep_time"] if sleep_state["sleep_time"] else 0
        previous_level = sleep_state["sleep_level"]

        # For partial wake-up with tags, we simulate staying in sleep mode
        # until all components are restored (vLLM behavior)
        if tags is None:
            # Full wake-up
            sleep_state["is_sleeping"] = False
            sleep_state["sleep_level"] = 0
            sleep_state["sleep_time"] = None
            logger.info(f"☀️ Server fully woke up after {sleep_duration:.2f}s of sleep")
        else:
            # Partial wake-up - server reports sleeping until all components restored
            logger.info(f"🌤️ Server partially woke up (tags: {tags}), still sleeping")

        return {
            "status": "awake" if tags is None else "partial_wake",
            "previous_level": previous_level,
            "sleep_duration_seconds": sleep_duration,
            "tags": tags,
            "message": "Server is now awake" if tags is None else f"Restored: {tags}"
        }

    @app.post("/collective_rpc")
    async def collective_rpc(request: Request):
        """
        Perform a collective remote procedure call (vLLM-compatible endpoint).

        Used for operations like `reload_weights` during RLHF weight updates.
        """
        try:
            body = await request.json()
            method = body.get("method", "")
        except Exception:
            method = ""

        logger.info(f"🔄 Collective RPC called with method: {method}")

        # Mock implementation - just acknowledge the RPC
        return {
            "status": "ok",
            "method": method,
            "message": f"Collective RPC '{method}' executed successfully"
        }

    @app.get("/is_sleeping")
    async def is_sleeping():
        """Check if the model is sleeping (vLLM-compatible endpoint)."""
        return sleep_state["is_sleeping"]

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models"""
        if sleep_state["is_sleeping"]:
            raise HTTPException(
                status_code=503,
                detail="Server is in sleep mode. Call /wake_up to resume."
            )
        if backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model_info = backend.get_model_info()
        return ModelsResponse(data=[ModelInfo(**model_info)])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        """Chat completions endpoint (OpenAI compatible)"""
        if sleep_state["is_sleeping"]:
            raise HTTPException(
                status_code=503,
                detail="Server is in sleep mode. Call /wake_up to resume."
            )
        if backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()
        client_ip = http_request.client.host

        # Log the incoming request with model and prompt info
        user_prompt = request.messages[-1].content if request.messages else "No prompt"
        logger.info(
            f"💬 Chat request from {client_ip} | Model: {config.served_model_name} | "
            f"Prompt: '{user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}'"
        )

        try:
            # Convert messages to dict format
            messages = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]

            # Update metrics
            metrics["total_requests"] += 1

            if request.stream:
                # Streaming response
                async def generate_stream():
                    async for chunk in backend.generate(
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True,
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type",
                    },
                )
            else:
                # Non-streaming response
                response_generator = backend.generate(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=False,
                )
                response = await response_generator.__anext__()

                # Log response and update metrics
                response_time = time.time() - start_time
                metrics["response_times"].append(response_time)
                if "choices" in response and response["choices"]:
                    generated_text = (
                        response["choices"][0].get("message", {}).get("content", "")
                    )
                    token_count = len(generated_text.split())  # Rough token estimate
                    metrics["total_tokens_generated"] += token_count

                    logger.info(
                        f"✅ Response sent | Model: {config.served_model_name} | "
                        f"Tokens: ~{token_count} | Time: {response_time:.2f}s | "
                        f"Response: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'"
                    )

                return response

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(
                f"❌ Error in chat completions | Model: {config.served_model_name} | "
                f"Time: {response_time:.2f}s | Error: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def get_metrics():
        """Prometheus-style metrics endpoint"""
        avg_response_time = (
            sum(metrics["response_times"]) / len(metrics["response_times"])
            if metrics["response_times"]
            else 0.0
        )

        uptime = time.time() - metrics["start_time"]

        # Return Prometheus-style metrics
        prometheus_metrics = f"""# HELP llm_katan_requests_total Total number of requests processed
# TYPE llm_katan_requests_total counter
llm_katan_requests_total{{model="{config.served_model_name}",backend="{config.backend}"}} {metrics["total_requests"]}

# HELP llm_katan_tokens_generated_total Total number of tokens generated
# TYPE llm_katan_tokens_generated_total counter
llm_katan_tokens_generated_total{{model="{config.served_model_name}",backend="{config.backend}"}} {metrics["total_tokens_generated"]}

# HELP llm_katan_response_time_seconds Average response time in seconds
# TYPE llm_katan_response_time_seconds gauge
llm_katan_response_time_seconds{{model="{config.served_model_name}",backend="{config.backend}"}} {avg_response_time:.4f}

# HELP llm_katan_uptime_seconds Server uptime in seconds
# TYPE llm_katan_uptime_seconds gauge
llm_katan_uptime_seconds{{model="{config.served_model_name}",backend="{config.backend}"}} {uptime:.2f}
"""

        return PlainTextResponse(content=prometheus_metrics, media_type="text/plain")

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "LLM Katan - Lightweight LLM Server",
            "version": __version__,
            "model": config.served_model_name,
            "backend": config.backend,
            "docs": "/docs",
            "metrics": "/metrics",
        }

    return app


async def run_server(config: ServerConfig):
    """Run the server with uvicorn"""
    import uvicorn

    app = create_app(config)

    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True,
    )

    server = uvicorn.Server(uvicorn_config)
    await server.serve()
