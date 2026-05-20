"""Training-data recorder for successful benchmarked model calls.

This module runs a lightweight local HTTP proxy that forwards supported model
requests to the real upstream provider while capturing the raw request payload.
After a task execution completes, the benchmark merges those captured request
messages with assistant outputs from the archived transcript and writes compact
JSONL training rows.
"""

from __future__ import annotations

import copy
import http.client
import json
import logging
import ssl
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import parse


logger = logging.getLogger(__name__)

SUPPORTED_RECORDING_APIS = {"openai-completions", "anthropic-messages"}


@dataclass(frozen=True)
class ExecutionContext:
    benchmark_run_id: str
    task_id: str
    run_index: int
    api: str
    upstream_base_url: str


def supports_recording_api(api: Optional[str]) -> bool:
    return bool(api) and api in SUPPORTED_RECORDING_APIS


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _stringify_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {}, separators=(",", ":"), ensure_ascii=False)


def _normalize_tool_schema(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    if isinstance(tools, list):
        return _json_clone(tools)
    return [_json_clone(tools)]


def _normalize_openai_message(message: Dict[str, Any]) -> Dict[str, Any]:
    role = str(message.get("role", "user"))
    normalized: Dict[str, Any] = {"role": role}

    if "content" in message:
        content = message.get("content")
        normalized["content"] = "" if content is None else _json_clone(content)
    elif role == "assistant":
        normalized["content"] = ""

    if "name" in message:
        normalized["name"] = message["name"]
    if "tool_call_id" in message:
        normalized["tool_call_id"] = message["tool_call_id"]

    if role == "assistant" and message.get("tool_calls"):
        tool_calls = []
        for call in message.get("tool_calls", []):
            function = call.get("function", {}) if isinstance(call, dict) else {}
            tool_calls.append(
                {
                    "id": call.get("id", ""),
                    "type": call.get("type", "function"),
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": _stringify_tool_arguments(function.get("arguments")),
                    },
                }
            )
        normalized["tool_calls"] = tool_calls

    return normalized


def normalize_openai_request_body(body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    messages = [_normalize_openai_message(message) for message in body.get("messages", [])]
    return messages, _normalize_tool_schema(body.get("tools"))


def _anthropic_content_to_user_messages(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": _json_clone(content)}]

    emitted: List[Dict[str, Any]] = []
    text_blocks: List[Any] = []

    def flush_text_blocks() -> None:
        if not text_blocks:
            return
        if all(isinstance(block, str) for block in text_blocks):
            emitted.append({"role": "user", "content": "".join(text_blocks)})
        else:
            emitted.append({"role": "user", "content": _json_clone(text_blocks)})
        text_blocks.clear()

    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            flush_text_blocks()
            emitted.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _json_clone(block.get("content", "")),
                }
            )
            continue

        if isinstance(block, dict) and block.get("type") == "text":
            text_blocks.append(block.get("text", ""))
        else:
            text_blocks.append(_json_clone(block))

    flush_text_blocks()
    return emitted or [{"role": "user", "content": ""}]


def _anthropic_content_to_assistant_message(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        return {"role": "assistant", "content": _json_clone(content)}

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    passthrough_blocks: List[Any] = []

    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _stringify_tool_arguments(block.get("input")),
                    },
                }
            )
        else:
            passthrough_blocks.append(_json_clone(block))

    normalized: Dict[str, Any] = {"role": "assistant"}
    if passthrough_blocks:
        combined_blocks: List[Any] = []
        if text_parts:
            combined_blocks.append({"type": "text", "text": "".join(text_parts)})
        combined_blocks.extend(passthrough_blocks)
        normalized["content"] = combined_blocks
    else:
        normalized["content"] = "".join(text_parts)
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def normalize_anthropic_request_body(body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    messages: List[Dict[str, Any]] = []

    system = body.get("system")
    if system not in (None, "", []):
        messages.append({"role": "system", "content": _json_clone(system)})

    for message in body.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant":
            messages.append(_anthropic_content_to_assistant_message(content))
        elif role == "user":
            messages.extend(_anthropic_content_to_user_messages(content))
        else:
            messages.append({"role": role, "content": _json_clone(content)})

    return messages, _normalize_tool_schema(body.get("tools"))


def normalize_request_capture(api: str, body_bytes: bytes) -> Optional[Dict[str, Any]]:
    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Skipping training capture due to invalid JSON request body: %s", exc)
        return None

    if api == "openai-completions":
        messages, tools = normalize_openai_request_body(body)
    elif api == "anthropic-messages":
        messages, tools = normalize_anthropic_request_body(body)
    else:
        logger.warning("Skipping unsupported recording API: %s", api)
        return None

    sample: Dict[str, Any] = {"messages": messages}
    if tools:
        sample["tools"] = tools
    return sample


def normalize_transcript_assistant_message(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if event.get("type") != "message":
        return None
    message = event.get("message", {})
    if message.get("role") != "assistant":
        return None

    content = message.get("content", [])
    if not isinstance(content, list):
        return {"role": "assistant", "content": "" if content is None else _json_clone(content)}

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    passthrough_blocks: List[Any] = []

    for item in content:
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "text":
            text_parts.append(item.get("text", ""))
        elif item_type == "toolCall":
            tool_calls.append(
                {
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": _stringify_tool_arguments(item.get("arguments")),
                    },
                }
            )
        else:
            passthrough_blocks.append(_json_clone(item))

    normalized: Dict[str, Any] = {"role": "assistant"}
    if passthrough_blocks:
        combined_blocks: List[Any] = []
        if text_parts:
            combined_blocks.append({"type": "text", "text": "".join(text_parts)})
        combined_blocks.extend(passthrough_blocks)
        normalized["content"] = combined_blocks
    else:
        normalized["content"] = "".join(text_parts)
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def extract_transcript_assistant_messages(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for event in transcript:
        normalized = normalize_transcript_assistant_message(event)
        if normalized is not None:
            messages.append(normalized)
    return messages


class TrainingRecorder:
    """Proxy benchmarked model calls and emit compact training JSONL rows."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._contexts: Dict[str, ExecutionContext] = {}
        self._captures: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}

    @property
    def enabled(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        if self._server is not None:
            return

        recorder = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "PinchBenchTrainRecorder/1.0"
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                logger.debug("TrainingRecorder: " + format, *args)

            def do_POST(self) -> None:  # noqa: N802
                recorder._handle_post(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            "Training recorder listening on http://127.0.0.1:%s",
            self._server.server_address[1],
        )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def register_execution(
        self,
        *,
        benchmark_run_id: str,
        task_id: str,
        run_index: int,
        api: str,
        upstream_base_url: str,
    ) -> str:
        if self._server is None:
            raise RuntimeError("Training recorder has not been started")
        context_id = uuid.uuid4().hex
        context = ExecutionContext(
            benchmark_run_id=benchmark_run_id,
            task_id=task_id,
            run_index=run_index,
            api=api,
            upstream_base_url=upstream_base_url,
        )
        with self._lock:
            self._contexts[context_id] = context
            self._captures.setdefault((benchmark_run_id, task_id, run_index), [])
        return f"http://127.0.0.1:{self._server.server_address[1]}/ctx/{context_id}"

    def finalize_execution(
        self,
        *,
        benchmark_run_id: str,
        task_id: str,
        run_index: int,
        transcript: List[Dict[str, Any]],
    ) -> int:
        key = (benchmark_run_id, task_id, run_index)
        with self._lock:
            captures = list(self._captures.pop(key, []))
            context_ids = [
                cid
                for cid, ctx in self._contexts.items()
                if (ctx.benchmark_run_id, ctx.task_id, ctx.run_index) == key
            ]
            for cid in context_ids:
                self._contexts.pop(cid, None)

        assistant_messages = extract_transcript_assistant_messages(transcript)
        logger.info(
            "Training recorder finalize for %s run %d: %d capture(s), %d assistant message(s)",
            task_id,
            run_index,
            len(captures),
            len(assistant_messages),
        )
        if not captures or not assistant_messages:
            if captures and not assistant_messages:
                logger.warning(
                    "Training recorder captured %d request(s) for %s run %d but found no assistant messages in transcript",
                    len(captures),
                    task_id,
                    run_index,
                )
            return 0

        if len(captures) != len(assistant_messages):
            logger.warning(
                "Training recorder count mismatch for %s run %d: %d request(s) vs %d assistant message(s); writing the aligned prefix only",
                task_id,
                run_index,
                len(captures),
                len(assistant_messages),
            )

        rows_written = 0
        for request_capture, assistant_message in zip(captures, assistant_messages):
            sample: Dict[str, Any] = {
                "task_id": task_id,
                "run_index": run_index,
                "messages": [*copy.deepcopy(request_capture["messages"]), assistant_message],
            }
            tools = request_capture.get("tools")
            if tools:
                sample["tools"] = copy.deepcopy(tools)
            self._append_jsonl(sample)
            rows_written += 1
        return rows_written

    def _append_jsonl(self, row: Dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = parse.urlsplit(handler.path)
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) < 3 or path_parts[0] != "ctx":
            handler.send_error(404, "Unknown training recorder path")
            return

        context_id = path_parts[1]
        upstream_suffix = "/" + "/".join(path_parts[2:])
        if parsed.query:
            upstream_suffix = f"{upstream_suffix}?{parsed.query}"

        with self._lock:
            context = self._contexts.get(context_id)
        if context is None:
            handler.send_error(404, "Unknown training recorder context")
            return

        content_length = int(handler.headers.get("Content-Length", "0") or "0")
        body = handler.rfile.read(content_length) if content_length > 0 else b""
        upstream_url = parse.urljoin(context.upstream_base_url.rstrip("/") + "/", upstream_suffix.lstrip("/"))

        try:
            status, reason, response_headers, response_body = self._forward_request(
                method="POST",
                upstream_url=upstream_url,
                headers=handler.headers,
                body=body,
            )
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Training recorder upstream request failed: %s", exc)
            handler.send_error(502, f"Upstream request failed: {exc}")
            return

        handler.send_response(status, reason)
        for header_name, header_value in response_headers:
            lower_name = header_name.lower()
            if lower_name in {"transfer-encoding", "connection", "content-length"}:
                continue
            handler.send_header(header_name, header_value)
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        if response_body:
            handler.wfile.write(response_body)
            handler.wfile.flush()

        if 200 <= status < 300:
            normalized_capture = normalize_request_capture(context.api, body)
            if normalized_capture is not None:
                with self._lock:
                    capture_list = self._captures.setdefault(
                        (context.benchmark_run_id, context.task_id, context.run_index), []
                    )
                    capture_list.append(normalized_capture)
                    capture_count = len(capture_list)
                logger.info(
                    "Training recorder captured request #%d for %s run %d via %s %s",
                    capture_count,
                    context.task_id,
                    context.run_index,
                    handler.command,
                    parsed.path,
                )
            else:
                logger.warning(
                    "Training recorder skipped a %s capture for %s run %d on %s",
                    context.api,
                    context.task_id,
                    context.run_index,
                    parsed.path,
                )

    @staticmethod
    def _forward_request(
        *,
        method: str,
        upstream_url: str,
        headers: Any,
        body: bytes,
    ) -> Tuple[int, str, List[Tuple[str, str]], bytes]:
        parsed = parse.urlsplit(upstream_url)
        connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        port = parsed.port
        if parsed.scheme == "https":
            connection = connection_cls(
                parsed.hostname,
                port,
                timeout=600,
                context=ssl._create_unverified_context(),
            )
        else:
            connection = connection_cls(parsed.hostname, port, timeout=600)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        forward_headers: Dict[str, str] = {}
        for header_name, header_value in headers.items():
            lower_name = header_name.lower()
            if lower_name in {"host", "connection", "content-length"}:
                continue
            forward_headers[header_name] = header_value
        if body:
            forward_headers["Content-Length"] = str(len(body))

        try:
            connection.request(method, path, body=body, headers=forward_headers)
            response = connection.getresponse()
            response_body = response.read()
            response_headers = list(response.getheaders())
            return response.status, response.reason, response_headers, response_body
        finally:
            connection.close()
