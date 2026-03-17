"""Thin wrapper for LLM API calls."""

from __future__ import annotations

import os
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_MODEL = "gpt-5.4"


class LLMCaller:
    """Wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_completion_tokens: int = 2000,
    ) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Stream a chat-completions request and emit incremental events.

        Returns the fully assembled assistant message payload:
            {"content": str, "tool_calls": list[dict], "finish_reason": str|None}
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_completion_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None

        stream = self.client.chat.completions.create(**request)
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            text_delta = _coerce_text(getattr(delta, "content", None))
            if text_delta:
                content_parts.append(text_delta)
                if on_stream_event:
                    on_stream_event(
                        {"type": "assistant_text_delta", "delta": text_delta}
                    )

            reasoning_delta = _extract_reasoning_delta(delta)
            if reasoning_delta and on_stream_event:
                on_stream_event(
                    {
                        "type": "assistant_reasoning_delta",
                        "delta": reasoning_delta,
                    }
                )

            for tool_delta in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(tool_delta, "index", 0) or 0)
                tool_entry = tool_calls_by_index.setdefault(
                    index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )

                tool_id = getattr(tool_delta, "id", None)
                if tool_id:
                    tool_entry["id"] = tool_id

                tool_type = getattr(tool_delta, "type", None)
                if tool_type:
                    tool_entry["type"] = tool_type

                fn = getattr(tool_delta, "function", None)
                name_delta = ""
                args_delta = ""
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        tool_entry["function"]["name"] = fn_name
                        name_delta = fn_name

                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        tool_entry["function"]["arguments"] += fn_args
                        args_delta = fn_args

                if on_stream_event and (name_delta or args_delta):
                    on_stream_event(
                        {
                            "type": "tool_call_delta",
                            "index": index,
                            "tool_call_id": tool_entry["id"],
                            "name_delta": name_delta,
                            "arguments_delta": args_delta,
                        }
                    )

        ordered_tool_calls = [
            tool_calls_by_index[index] for index in sorted(tool_calls_by_index.keys())
        ]
        message = {
            "content": "".join(content_parts),
            "tool_calls": ordered_tool_calls,
            "finish_reason": finish_reason,
        }
        if on_stream_event:
            on_stream_event(
                {
                    "type": "assistant_message_done",
                    "finish_reason": finish_reason,
                    "has_tool_calls": bool(ordered_tool_calls),
                }
            )
        return message


def _coerce_text(value: Any) -> str:
    """Best-effort extraction of text from stream deltas."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    return ""


def _extract_reasoning_delta(delta: Any) -> str:
    """Best-effort extraction of reasoning/thinking deltas when provided by API."""
    for attr in ("reasoning", "reasoning_content", "thought", "thinking"):
        value = getattr(delta, attr, None)
        text = _coerce_text(value)
        if text:
            return text
    return ""
