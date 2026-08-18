"""Pi coding-agent execution helpers for PinchBench."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib_agent import (
    _coerce_subprocess_output,
    prepare_task_workspace,
)
from lib_fws import fws_available, is_fws_task, start_fws, stop_fws
from lib_tasks import Task


logger = logging.getLogger(__name__)

USE_SHELL = platform.system() == "Windows"

PI_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def pi_available() -> bool:
    return shutil.which("pi") is not None


def map_pi_thinking(level: Optional[str]) -> Optional[str]:
    """Map a CLI thinking level onto Pi's supported set."""
    if not level:
        return None
    if level == "adaptive":
        return "max"
    return level


def build_pi_command(
    *,
    prompt: str,
    model_id: str,
    session_id: Optional[str] = None,
    session_dir: Optional[Path] = None,
    thinking_level: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    """Build a non-interactive ``pi --mode json`` invocation."""
    cmd = [
        "pi",
        "-p",
        "--mode",
        "json",
        "--approve",
        "--name",
        "pinchbench",
    ]
    if session_dir is not None:
        cmd.extend(["--session-dir", str(session_dir)])
    if session_id:
        cmd.extend(["--session-id", session_id])
    if model_id:
        cmd.extend(["--model", model_id])
    mapped_thinking = map_pi_thinking(thinking_level)
    if mapped_thinking:
        cmd.extend(["--thinking", mapped_thinking])
    if api_key:
        cmd.extend(["--api-key", api_key])
    cmd.append(prompt)
    return cmd


def parse_pi_jsonl(stdout: str) -> List[Dict[str, Any]]:
    """Parse Pi JSON-mode stdout into a list of event dicts.

    Non-JSON lines (startup banners, progress) are skipped.
    """
    events: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def extract_pi_session_id(events: List[Dict[str, Any]]) -> Optional[str]:
    for event in events:
        if event.get("type") == "session" and event.get("id"):
            return str(event["id"])
    return None


def _user_content_as_list(content: Any) -> List[Any]:
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [content]


def _assistant_content_blocks(content: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        return blocks
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif item_type == "thinking":
            thinking = item.get("thinking", "")
            if thinking:
                blocks.append({"type": "text", "text": thinking})
        elif item_type == "toolCall":
            blocks.append(
                {
                    "type": "toolCall",
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or {},
                }
            )
    return blocks


def _tool_result_content(result: Any) -> List[Any]:
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            return content
        if content is not None:
            return [content]
        return [result]
    if result is None:
        return []
    return [result]


def pi_events_to_transcript(
    events: List[Dict[str, Any]],
    user_prompt: str,
) -> List[Dict[str, Any]]:
    """Convert Pi JSON events into the OpenClaw-shaped transcript graders expect."""
    transcript: List[Dict[str, Any]] = [
        {
            "type": "message",
            "message": {"role": "user", "content": [user_prompt]},
        }
    ]
    emitted_user_prompts = {user_prompt}

    for event in events:
        event_type = event.get("type")
        if event_type == "message_end":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                transcript.append(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": _assistant_content_blocks(message.get("content")),
                            "usage": message.get("usage") or {},
                        },
                    }
                )
            elif role == "user":
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                    payload: List[Any] = [content]
                else:
                    payload = _user_content_as_list(content)
                    text = ""
                    for item in payload:
                        if isinstance(item, str):
                            text = item
                            break
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = str(item.get("text", ""))
                            break
                if text in emitted_user_prompts:
                    continue
                emitted_user_prompts.add(text)
                transcript.append(
                    {
                        "type": "message",
                        "message": {"role": "user", "content": payload},
                    }
                )
            elif role == "toolResult":
                transcript.append(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "content": _tool_result_content(message.get("content")),
                            "toolName": message.get("toolName"),
                            "toolCallId": message.get("toolCallId"),
                            "isError": message.get("isError", False),
                        },
                    }
                )
        elif event_type == "tool_execution_end":
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "content": _tool_result_content(event.get("result")),
                        "toolName": event.get("toolName"),
                        "toolCallId": event.get("toolCallId"),
                        "isError": event.get("isError", False),
                    },
                }
            )
    return transcript


def extract_pi_usage(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum OpenClaw-compatible usage fields from converted assistant messages."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("usage") or {}
        if not usage:
            continue
        totals["request_count"] += 1
        totals["input_tokens"] += int(usage.get("input") or 0)
        totals["output_tokens"] += int(usage.get("output") or 0)
        totals["cache_read_tokens"] += int(usage.get("cacheRead") or 0)
        totals["cache_write_tokens"] += int(usage.get("cacheWrite") or 0)
        totals["total_tokens"] += int(usage.get("totalTokens") or 0)
        cost = usage.get("cost") or {}
        if isinstance(cost, dict):
            totals["cost_usd"] += float(cost.get("total") or 0.0)
    return totals


def _pi_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PI_SKIP_VERSION_CHECK", "1")
    env.setdefault("PI_TELEMETRY", "0")
    return env


def _run_pi_prompt(
    *,
    prompt: str,
    model_id: str,
    workspace: Path,
    session_dir: Path,
    session_id: Optional[str],
    thinking_level: Optional[str],
    api_key: Optional[str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    cmd = build_pi_command(
        prompt=prompt,
        model_id=model_id,
        session_id=session_id,
        session_dir=session_dir,
        thinking_level=thinking_level,
        api_key=api_key,
    )
    logger.info("Running: %s", " ".join(cmd[:-1] + ["<prompt>"]))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=timeout_seconds,
        check=False,
        shell=USE_SHELL,
        env=_pi_env(),
    )


def execute_pi_task(
    *,
    task: Task,
    agent_id: str,
    model_id: str,
    run_id: str,
    timeout_multiplier: float,
    skill_dir: Path,
    workspace: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    thinking_level: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a task with the Pi coding-agent CLI."""
    logger.info("π Agent [%s] starting task: %s", agent_id, task.task_id)
    logger.info("   Task: %s", task.name)
    logger.info("   Category: %s", task.category)
    if thinking_level:
        logger.info("   Thinking: %s (mapped: %s)", thinking_level, map_pi_thinking(thinking_level))
    if verbose:
        logger.info(
            "   Prompt: %s",
            task.prompt[:500] + "..." if len(task.prompt) > 500 else task.prompt,
        )

    if not pi_available():
        return {
            "agent_id": agent_id,
            "task_id": task.task_id,
            "status": "error",
            "transcript": [],
            "usage": {},
            "workspace": str(workspace),
            "exit_code": -1,
            "timed_out": False,
            "execution_time": 0.0,
            "stdout": "",
            "stderr": "pi command not found; install @earendil-works/pi-coding-agent",
            "runtime": "pi",
        }

    fws_env = None
    if is_fws_task(task.frontmatter):
        if not fws_available():
            logger.warning(
                "⚠️ Task %s requires fws but it's not installed (npm install -g @juppytt/fws)",
                task.task_id,
            )
        else:
            fws_env = start_fws()

    start_time = time.time()
    workspace = prepare_task_workspace(
        skill_dir,
        run_id,
        task,
        agent_id,
        workspace=workspace,
        bootstrap_dirs=[
            Path.home() / ".pi" / "agent",
            Path.home() / ".openclaw" / "workspace",
        ],
    )
    session_dir = workspace / ".pinchbench-pi-sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = task.timeout_seconds * timeout_multiplier
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False
    transcript: List[Dict[str, Any]] = []
    current_session_id: Optional[str] = None

    sessions = task.frontmatter.get("sessions", [])
    prompts: List[tuple[str, bool]] = []
    if sessions:
        logger.info("📋 Multi-session task with %d sessions", len(sessions))
        for session_entry in sessions:
            if isinstance(session_entry, str):
                prompts.append((session_entry, False))
            elif isinstance(session_entry, dict):
                session_prompt = session_entry.get("prompt") or session_entry.get("message", "")
                prompts.append((session_prompt, bool(session_entry.get("new_session", False))))
            else:
                logger.warning("⚠️ Skipping invalid session entry: %s", session_entry)
    else:
        prompts.append((task.prompt, False))

    try:
        for i, (session_prompt, is_new_session) in enumerate(prompts, 1):
            if is_new_session or current_session_id is None:
                current_session_id = str(uuid.uuid4())
                logger.info(
                    "   Session %d/%d (new_session=%s, session_id=%s)",
                    i,
                    len(prompts),
                    is_new_session or i == 1,
                    current_session_id,
                )
            else:
                logger.info(
                    "   Session %d/%d (continuing session_id=%s)",
                    i,
                    len(prompts),
                    current_session_id,
                )

            elapsed = time.time() - start_time
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                timed_out = True
                break
            try:
                result = _run_pi_prompt(
                    prompt=session_prompt,
                    model_id=model_id,
                    workspace=workspace,
                    session_dir=session_dir,
                    session_id=current_session_id,
                    thinking_level=thinking_level,
                    api_key=api_key,
                    timeout_seconds=remaining,
                )
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout += _coerce_subprocess_output(exc.stdout)
                stderr += _coerce_subprocess_output(exc.stderr)
                events = parse_pi_jsonl(_coerce_subprocess_output(exc.stdout))
                transcript.extend(pi_events_to_transcript(events, session_prompt))
                break
            except FileNotFoundError as exc:
                stderr = f"pi command not found: {exc}"
                break

            stdout += result.stdout or ""
            stderr += result.stderr or ""
            exit_code = result.returncode
            events = parse_pi_jsonl(result.stdout or "")
            resolved_id = extract_pi_session_id(events)
            if resolved_id:
                current_session_id = resolved_id
            transcript.extend(pi_events_to_transcript(events, session_prompt))
            if result.returncode not in (0, -1):
                break
    finally:
        if fws_env is not None:
            stop_fws(fws_env)

    usage = extract_pi_usage(transcript)
    execution_time = time.time() - start_time

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_dest = output_dir / f"{run_id}.jsonl"
        try:
            with archive_dest.open("w", encoding="utf-8") as handle:
                for entry in transcript:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info("Archived Pi transcript to %s", archive_dest)
        except OSError as exc:
            logger.warning("Failed to archive Pi transcript: %s", exc)

    status = "success"
    if timed_out:
        status = "timeout"
    if not transcript:
        status = "error"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"
    if stderr and "pi command not found" in str(stderr):
        status = "error"

    if verbose:
        logger.info("   [VERBOSE] Exit code: %s", exit_code)
        logger.info("   [VERBOSE] Execution time: %.2fs", execution_time)
        logger.info("   [VERBOSE] Workspace: %s", workspace)
        if stdout:
            logger.info("   [VERBOSE] Stdout (first 1000 chars):\n%s", stdout[:1000])
        if stderr:
            logger.info("   [VERBOSE] Stderr:\n%s", stderr[:1000])
        logger.info("   [VERBOSE] Transcript entries: %d", len(transcript))

    return {
        "agent_id": agent_id,
        "task_id": task.task_id,
        "status": status,
        "transcript": transcript,
        "usage": usage,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": stdout,
        "stderr": stderr,
        "runtime": "pi",
    }
