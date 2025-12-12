import asyncio
import math
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

# Sleep mode state (vLLM-compatible)
sleep_state = {
    "is_sleeping": False,
    "sleep_level": 0,
    "sleep_time": None,
    "wake_up_delay_seconds": 0.1,  # Simulated wake-up delay
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2


@app.get("/health")
async def health():
    """Health check endpoint - returns 503 when sleeping."""
    if sleep_state["is_sleeping"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "sleeping", "detail": "Server is in sleep mode"}
        )
    return {"status": "ok"}


@app.post("/sleep")
async def sleep(level: int = 1):
    """
    Put the server into sleep mode (vLLM-compatible endpoint).

    Sleep levels:
    - Level 1: Offloads model weights to CPU RAM and discards KV cache
    - Level 2: Discards both model weights and KV cache
    """
    if sleep_state["is_sleeping"]:
        return {
            "status": "already_sleeping",
            "level": sleep_state["sleep_level"],
        }

    sleep_state["is_sleeping"] = True
    sleep_state["sleep_level"] = level
    sleep_state["sleep_time"] = time.time()

    return {
        "status": "sleeping",
        "level": level,
    }


@app.post("/wake_up")
async def wake_up(tags: Optional[str] = None):
    """
    Wake up the server from sleep mode (vLLM-compatible endpoint).

    Supports optional `tags` query parameter for partial wake-up:
    - `?tags=weights` - Only restore model weights
    - `?tags=kv_cache` - Only restore KV cache
    - No tags - Full wake-up (restore everything)
    """
    if not sleep_state["is_sleeping"]:
        return {
            "status": "already_awake",
        }

    # Simulate wake-up delay
    await asyncio.sleep(sleep_state["wake_up_delay_seconds"])

    # For partial wake-up with tags, stay in sleep mode until all components restored
    if tags is None:
        sleep_state["is_sleeping"] = False
        sleep_state["sleep_level"] = 0
        sleep_state["sleep_time"] = None

    return {
        "status": "awake" if tags is None else "partial_wake",
        "tags": tags,
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

    return {
        "status": "ok",
        "method": method,
    }


@app.get("/is_sleeping")
async def is_sleeping():
    """Check if the model is sleeping (vLLM-compatible endpoint)."""
    return sleep_state["is_sleeping"]


@app.get("/v1/models")
async def models():
    """List available models - returns 503 when sleeping."""
    if sleep_state["is_sleeping"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is in sleep mode. Call /wake_up to resume."}
        )
    return {"data": [{"id": "openai/gpt-oss-20b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """Chat completions endpoint - returns 503 when sleeping."""
    if sleep_state["is_sleeping"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is in sleep mode. Call /wake_up to resume."}
        )
    # Very simple echo-like behavior
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    content = f"[mock-{req.model}] You said: {last_user}"

    # Rough token estimation: ~1 token per 4 characters (ceil)
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    prompt_text = "\n".join(
        m.content for m in req.messages if isinstance(m.content, str)
    )
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(content)
    total_tokens = prompt_tokens + completion_tokens

    created_ts = int(time.time())

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # Optional details fields some clients read when using caching/reasoning
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }

    return {
        "id": "cmpl-mock-123",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": "mock-vllm",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": usage,
        # Some SDKs look for token_usage; keep it as an alias for convenience.
        "token_usage": usage,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
