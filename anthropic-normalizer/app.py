import json
import os
import re
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://llamacpp:8080").rstrip("/")
FAIL_CLOSED = os.environ.get("NORMALIZER_FAIL_CLOSED", "true").lower() in {"1", "true", "yes", "on"}
STRIP_THINK = os.environ.get("NORMALIZER_STRIP_THINK", "true").lower() in {"1", "true", "yes", "on"}

app = FastAPI()

THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)

# Full Qwen tool-call block.
QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*"
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<body>.*?)"
    r"</function>\s*"
    r"</tool_call>",
    re.DOTALL,
)

# Bare function block, used as a fallback when the model omits <tool_call>.
BARE_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<body>.*?)"
    r"</function>",
    re.DOTALL,
)

QWEN_PARAM_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(?P<value>.*?)"
    r"</parameter>",
    re.DOTALL,
)

ANTHROPIC_INVOKE_RE = re.compile(
    r"<invoke\s+name=[\"'](?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"']>\s*"
    r"(?P<body>.*?)"
    r"</invoke>",
    re.DOTALL,
)

ANTHROPIC_PARAM_RE = re.compile(
    r"<parameter\s+name=[\"'](?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"']>\s*"
    r"(?P<value>.*?)"
    r"</parameter>",
    re.DOTALL,
)

TOOL_XML_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "<parameter=",
    "<tool_calls>",
    "</tool_calls>",
    "<invoke ",
)


def _tool_id() -> str:
    return f"toolu_norm_{uuid.uuid4().hex[:24]}"


def _allowed_tool_names(body: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue

        # Anthropic shape.
        if isinstance(tool.get("name"), str):
            names.add(tool["name"])
            continue

        # OpenAI-ish fallback.
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])

    return names


def _coerce_value(raw: str) -> Any:
    value = raw.strip("\n")
    stripped = value.strip()

    # Preserve ordinary strings exactly-ish. Parse structured JSON-looking args.
    if stripped.startswith(("{", "[", '"')) or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    # Avoid turning file modes, numbers-in-strings, etc. into surprising types.
    return value


def _parse_params(body: str, param_re: re.Pattern[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}

    for p in param_re.finditer(body):
        key = p.group("name")
        if key in params:
            raise ValueError(f"duplicate parameter: {key}")
        params[key] = _coerce_value(p.group("value"))

    return params


def _strip_think(text: str) -> str:
    if not STRIP_THINK:
        return text
    return THINK_RE.sub("", text).strip()


def _normalise_text_to_blocks(text: str, allowed: set[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    working = _strip_think(text)

    def add_tool(name: str, params: dict[str, Any]) -> None:
        if allowed and name not in allowed:
            raise ValueError(f"model emitted unknown tool: {name}")
        blocks.append(
            {
                "type": "tool_use",
                "id": _tool_id(),
                "name": name,
                "input": params,
            }
        )

    def replace_qwen(match: re.Match[str]) -> str:
        add_tool(match.group("name"), _parse_params(match.group("body"), QWEN_PARAM_RE))
        return ""

    working = QWEN_TOOL_CALL_RE.sub(replace_qwen, working)

    # Handle Anthropic XML-ish invoke form.
    def replace_invoke(match: re.Match[str]) -> str:
        add_tool(match.group("name"), _parse_params(match.group("body"), ANTHROPIC_PARAM_RE))
        return ""

    working = ANTHROPIC_INVOKE_RE.sub(replace_invoke, working)
    working = working.replace("<tool_calls>", "").replace("</tool_calls>", "")

    # Fallback for leaked bare <function=...> blocks.
    working = BARE_FUNCTION_RE.sub(replace_qwen, working)

    if any(marker in working for marker in TOOL_XML_MARKERS):
        if FAIL_CLOSED:
            raise ValueError(f"unparsed tool XML remains in assistant text: {working[:500]!r}")

    working = working.strip()
    if working:
        # Put ordinary prose before parsed tool calls. Ideally there is none for tool calls.
        blocks.insert(0, {"type": "text", "text": working})

    return blocks


def _normalise_message_response(data: dict[str, Any], request_body: dict[str, Any]) -> dict[str, Any]:
    allowed = _allowed_tool_names(request_body)
    old_content = data.get("content") or []

    if not isinstance(old_content, list):
        return data

    new_content: list[dict[str, Any]] = []
    converted_tool = False

    for block in old_content:
        if not isinstance(block, dict):
            continue

        btype = block.get("type")

        if btype == "text":
            text = str(block.get("text") or "")
            produced = _normalise_text_to_blocks(text, allowed)
            if any(b.get("type") == "tool_use" for b in produced):
                converted_tool = True
            new_content.extend(produced)
            continue

        # Pass already-structured tool_use blocks through untouched.
        if btype == "tool_use":
            new_content.append(block)
            continue

        new_content.append(block)

    data["content"] = new_content

    if converted_tool or any(b.get("type") == "tool_use" for b in new_content if isinstance(b, dict)):
        data["stop_reason"] = "tool_use"

    return data


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _stream_anthropic_message(data: dict[str, Any]):
    message_id = data.get("id") or f"msg_norm_{uuid.uuid4().hex[:24]}"
    model = data.get("model") or "normalised-local"
    usage = data.get("usage") or {"input_tokens": 0, "output_tokens": 0}

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )

    for idx, block in enumerate(data.get("content") or []):
        btype = block.get("type")

        if btype == "text":
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text = block.get("text") or ""
            if text:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
            continue

        if btype == "tool_use":
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id") or _tool_id(),
                        "name": block.get("name"),
                        "input": {},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input") or {}, separators=(",", ":")),
                    },
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
            continue

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": data.get("stop_reason") or "end_turn",
                "stop_sequence": data.get("stop_sequence"),
            },
            "usage": usage,
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


@app.get("/health")
async def health():
    return {"ok": True, "upstream": UPSTREAM_BASE_URL}


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    wants_stream = bool(body.get("stream"))

    # Deliberately buffer upstream. Correct tool blocks beat excitingly streamed XML soup.
    upstream_body = dict(body)
    upstream_body["stream"] = False

    headers = {
        "content-type": "application/json",
    }

    # Preserve auth-ish headers for compatibility, though llama.cpp usually ignores them.
    for h in ("authorization", "x-api-key", "anthropic-version", "anthropic-beta"):
        if h in request.headers:
            headers[h] = request.headers[h]

    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.post(
            f"{UPSTREAM_BASE_URL}/v1/messages",
            headers=headers,
            json=upstream_body,
        )

    if upstream.status_code >= 400:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    try:
        data = upstream.json()
        data = _normalise_message_response(data, body)
    except Exception as e:
        if FAIL_CLOSED:
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "normalizer_error",
                        "message": str(e),
                    },
                },
            )
        data = upstream.json()

    if wants_stream:
        return StreamingResponse(
            _stream_anthropic_message(data),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        )

    return JSONResponse(content=data)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def passthrough(path: str, request: Request):
    # Basic passthrough for /v1/models, metrics, etc.
    method = request.method
    raw_body = await request.body()

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }

    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.request(
            method,
            f"{UPSTREAM_BASE_URL}/{path}",
            headers=headers,
            content=raw_body,
            params=dict(request.query_params),
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )
