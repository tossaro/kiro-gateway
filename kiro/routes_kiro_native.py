# -*- coding: utf-8 -*-
"""
Kiro-CLI Native Protocol Proxy Route.

Transparent proxy for kiro-cli traffic that:
1. Receives raw AWS Event Stream requests from kiro-cli
2. Forwards to runtime.{region}.kiro.dev unchanged
3. Streams response back to kiro-cli unchanged
4. Taps the stream to extract contextUsagePercentage + count output tokens
5. Logs usage to the same usage.db via dashboard.log_usage()

This enables token tracking for kiro-cli using the same dashboard
and database as the OpenAI/Anthropic routes.

Usage:
    Configure kiro-cli to point at this gateway:
    kiro-cli settings api.codewhisperer.service '{"endpoint": "http://localhost:8000", "region": "us-east-1"}'
"""

import time
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from kiro.config import STREAMING_READ_TIMEOUT, REGION
from kiro.parsers import AwsEventStreamParser
from kiro.streaming_core import calculate_tokens_from_context_usage
from kiro.tokenizer import count_tokens
from kiro.dashboard import log_usage

router = APIRouter(tags=["Kiro Native Proxy"])

# Target host for kiro-cli API calls
KIRO_RUNTIME_HOST = f"https://runtime.{REGION}.kiro.dev"

# Route mapping: x-amz-target prefix → upstream host
# Streaming service goes to runtime, non-streaming goes to management/q
UPSTREAM_ROUTES = {
    "AmazonCodeWhispererStreamingService": f"https://runtime.{REGION}.kiro.dev",
    "AmazonCodeWhispererService.SendTelemetryEvent": f"https://q.{REGION}.amazonaws.com",
    "AmazonCodeWhispererService": f"https://management.{REGION}.kiro.dev",
}


def _resolve_upstream(amz_target: str) -> str:
    """Resolve the correct upstream host based on x-amz-target header."""
    # Check exact match first (e.g., SendTelemetryEvent)
    if amz_target in UPSTREAM_ROUTES:
        return UPSTREAM_ROUTES[amz_target]
    # Check prefix match (e.g., StreamingService.*)
    for prefix, host in UPSTREAM_ROUTES.items():
        if amz_target.startswith(prefix):
            return host
    # Default to runtime
    return f"https://runtime.{REGION}.kiro.dev"

# Cache last seen session_id for telemetry calls that don't have conversationState
_last_session_id_cache: dict = {}
# Cache last seen model per session (LLM requests don't include model name)
_model_cache_per_session: dict = {}

# Headers that should NOT be forwarded to upstream
# (hop-by-hop headers + proxy-related)
HOP_BY_HOP_HEADERS = frozenset([
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",  # httpx recalculates this
])


def _forward_headers(request: Request) -> dict:
    """
    Extract headers from incoming request to forward upstream.
    Strips hop-by-hop headers but preserves auth and AWS-specific headers.
    """
    forwarded = {}
    for key, value in request.headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            forwarded[key] = value
    return forwarded


@router.post("/generateAssistantResponse")
@router.post("/")
async def proxy_generate_assistant_response(request: Request):
    """
    Transparent proxy for kiro-cli's GenerateAssistantResponse calls.
    
    Forwards the request to Kiro runtime API unchanged, streams the
    response back, and taps the stream to log token usage.
    """
    start_time = time.time()

    # Read the raw request body
    body = await request.body()
    
    # Extract session_id (conversationId) from request body
    session_id = "kiro-cli"
    request_model = "auto"
    try:
        if body:
            import json as _json
            req_data = _json.loads(body)
            conv_id = req_data.get("conversationState", {}).get("conversationId")
            if conv_id:
                session_id = conv_id
                # Cache for telemetry calls that don't have conversationState
                _last_session_id_cache["value"] = conv_id
            elif _last_session_id_cache.get("value"):
                session_id = _last_session_id_cache["value"]
            # Extract model from request (telemetry has it, LLM calls don't)
            req_model = (
                req_data.get("modelId")
                or req_data.get("model")
                or req_data.get("conversationState", {}).get("currentMessage", {}).get("modelId")
            )
            if req_model and req_model != "auto":
                request_model = req_model
                _model_cache_per_session[session_id] = req_model
            elif session_id in _model_cache_per_session:
                request_model = _model_cache_per_session[session_id]
    except Exception:
        if _last_session_id_cache.get("value"):
            session_id = _last_session_id_cache["value"]
    
    # Forward headers (preserving kiro-cli's own auth)
    headers = _forward_headers(request)
    
    # Determine target from x-amz-target header
    amz_target = headers.get("x-amz-target", "")
    
    # Route to correct upstream host based on operation
    upstream_url = _resolve_upstream(amz_target)
    
    logger.info(
        f"[Kiro Native] Proxying {amz_target} "
        f"({len(body)} bytes) → {upstream_url}"
    )

    # Create HTTP client for upstream request
    timeout = httpx.Timeout(
        connect=30.0,
        read=STREAMING_READ_TIMEOUT,
        write=30.0,
        pool=30.0,
    )

    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        # Make the upstream request with streaming
        upstream_response = await client.send(
            client.build_request(
                "POST",
                upstream_url,
                headers=headers,
                content=body,
            ),
            stream=True,
        )
    except httpx.TimeoutException as e:
        await client.aclose()
        logger.error(f"[Kiro Native] Upstream timeout: {e}")
        return Response(
            content=b'{"error": "upstream timeout"}',
            status_code=504,
            media_type="application/json",
        )
    except Exception as e:
        await client.aclose()
        logger.error(f"[Kiro Native] Upstream error: {e}")
        return Response(
            content=b'{"error": "upstream connection failed"}',
            status_code=502,
            media_type="application/json",
        )

    # Build response headers to send back to kiro-cli
    response_headers = {}
    for key, value in upstream_response.headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            response_headers[key] = value

    # Only log LLM calls, skip telemetry
    if "GenerateAssistantResponse" not in amz_target:
        # Still proxy the request, but don't log usage
        async def passthrough():
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
            finally:
                await upstream_response.aclose()
                await client.aclose()

        return StreamingResponse(
            passthrough(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get(
                "content-type", "application/x-amz-json-1.0"
            ),
        )
    for key, value in upstream_response.headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            response_headers[key] = value

    async def stream_and_log():
        """
        Stream response chunks back to kiro-cli while tapping for usage data.
        """
        parser = AwsEventStreamParser()
        full_content = []
        raw_text_buffer = []  # All decoded text for output token counting
        context_usage_pct = None
        model_id = request_model
        total_bytes = 0

        try:
            async for chunk in upstream_response.aiter_bytes():
                total_bytes += len(chunk)
                # Pass through to client immediately
                yield chunk

                # Accumulate raw text for token counting
                raw_text_buffer.append(chunk.decode("utf-8", errors="ignore"))

                # Tap: parse events for usage tracking
                try:
                    events = parser.feed(chunk)
                    for event in events:
                        if event["type"] == "content":
                            full_content.append(event["data"])
                        elif event["type"] == "context_usage":
                            context_usage_pct = event["data"]
                except Exception as parse_err:
                    # Never let parsing errors break the stream
                    logger.debug(f"[Kiro Native] Parse error (non-fatal): {parse_err}")

        except Exception as e:
            logger.error(f"[Kiro Native] Stream error: {e}")
            raise
        finally:
            await upstream_response.aclose()
            await client.aclose()

            # Log usage to dashboard
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[Kiro Native] Stream complete: {total_bytes} bytes, "
                f"{len(full_content)} content chunks, ctx={context_usage_pct}%"
            )
            _log_native_usage(
                full_content=full_content,
                raw_text="".join(raw_text_buffer),
                context_usage_pct=context_usage_pct,
                model_id=model_id,
                duration_ms=duration_ms,
                status_code=upstream_response.status_code,
                session_id=session_id,
                request=request,
            )

    return StreamingResponse(
        stream_and_log(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get(
            "content-type", "application/vnd.amazon.eventstream"
        ),
    )


def _log_native_usage(
    full_content: list,
    raw_text: str,
    context_usage_pct: Optional[float],
    model_id: str,
    duration_ms: float,
    status_code: int,
    session_id: str,
    request: Request,
):
    """
    Calculate and log token usage from the tapped stream data.
    Uses the same logic as routes_anthropic.py:
    - Output tokens: tiktoken count of full content (or raw response text for tool-only responses)
    - Input tokens: derived from contextUsagePercentage
    """
    try:
        # Count output tokens
        # If we have parsed content (text responses), use that
        # Otherwise use raw response text (tool use responses)
        output_text = "".join(full_content)
        if not output_text and raw_text:
            # Tool-only response: extract JSON payloads from raw text for counting
            import re
            # Find all JSON objects in the stream (these are the tool call payloads)
            json_parts = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text)
            output_text = " ".join(json_parts)
        
        output_tokens = count_tokens(output_text) if output_text else 0

        # Calculate input tokens from context usage percentage
        input_tokens = 0
        if context_usage_pct is not None and context_usage_pct > 0:
            # Use per-model context window for accurate calculation
            from kiro.config import MODEL_CONTEXT_WINDOWS, DEFAULT_MAX_INPUT_TOKENS
            max_tokens = MODEL_CONTEXT_WINDOWS.get(model_id, DEFAULT_MAX_INPUT_TOKENS)
            total_ctx_tokens = int((context_usage_pct / 100) * max_tokens)
            input_tokens = max(0, total_ctx_tokens - output_tokens)

        total_tokens = input_tokens + output_tokens

        # Log to the same usage.db as other routes
        log_usage(
            model=model_id,
            endpoint="/kiro-native",
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            status_code=status_code,
            session_id=session_id,
        )

        logger.info(
            f"[Kiro Native] Logged: in={input_tokens:,} out={output_tokens:,} "
            f"total={total_tokens:,} ctx={context_usage_pct}% "
            f"duration={duration_ms:.0f}ms"
        )

    except Exception as e:
        logger.warning(f"[Kiro Native] Failed to log usage: {e}")
