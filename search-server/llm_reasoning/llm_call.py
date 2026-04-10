"""Thin wrapper for LLM API calls using the Responses API."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_MODEL = "gpt-5.4"


class LLMCaller:
    """Wrapper around the OpenAI Responses API."""

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
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Stream a Responses API request and emit incremental events.

        Returns the fully assembled response payload:
            {"content": str, "tool_calls": list[dict], "output_items": list}
        """
        request: dict[str, Any] = {
            "model": self.model,
            "input": input,
            "max_output_tokens": self.max_completion_tokens,
            "stream": True,
        }
        if tools:
            request["tools"] = tools

        content_parts: list[str] = []
        # Collect completed output items (function_call and message items)
        output_items: list[Any] = []
        # Track function call arguments being streamed
        current_fn_args: dict[str, str] = {}  # call_id -> accumulated args

        stream = self.client.responses.create(**request)
        for event in stream:
            event_type = event.type

            # Text content delta
            if event_type == "response.output_text.delta":
                delta = event.delta or ""
                if delta:
                    content_parts.append(delta)
                    if on_stream_event:
                        on_stream_event(
                            {"type": "assistant_text_delta", "delta": delta}
                        )

            # Reasoning / thinking delta
            elif event_type in (
                "response.reasoning.delta",
                "response.reasoning_summary_text.delta",
            ):
                delta = event.delta or ""
                if delta and on_stream_event:
                    on_stream_event(
                        {"type": "assistant_reasoning_delta", "delta": delta}
                    )

            # Function call arguments streaming
            elif event_type == "response.function_call_arguments.delta":
                call_id = getattr(event, "item_id", "") or ""
                args_delta = event.delta or ""
                if call_id:
                    current_fn_args.setdefault(call_id, "")
                    current_fn_args[call_id] += args_delta
                if args_delta and on_stream_event:
                    on_stream_event(
                        {
                            "type": "tool_call_delta",
                            "index": 0,
                            "tool_call_id": call_id,
                            "name_delta": "",
                            "arguments_delta": args_delta,
                        }
                    )

            # An output item is fully completed
            elif event_type == "response.output_item.added":
                item = event.item
                if item and getattr(item, "type", None) == "function_call":
                    name = getattr(item, "name", "") or ""
                    if name and on_stream_event:
                        on_stream_event(
                            {
                                "type": "tool_call_delta",
                                "index": 0,
                                "tool_call_id": getattr(item, "call_id", "") or "",
                                "name_delta": name,
                                "arguments_delta": "",
                            }
                        )

            elif event_type == "response.output_item.done":
                item = event.item
                if item:
                    output_items.append(item)

            # Response completed
            elif event_type == "response.completed":
                if on_stream_event:
                    on_stream_event(
                        {
                            "type": "assistant_message_done",
                            "finish_reason": "stop",
                            "has_tool_calls": any(
                                getattr(i, "type", None) == "function_call"
                                for i in output_items
                            ),
                        }
                    )

        # Build tool_calls list from output items for compatibility
        tool_calls: list[dict[str, Any]] = []
        for item in output_items:
            if getattr(item, "type", None) == "function_call":
                call_id = getattr(item, "call_id", "") or ""
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function_call",
                        "name": getattr(item, "name", "") or "",
                        "arguments": getattr(item, "arguments", "") or "",
                    }
                )

        return {
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
            "output_items": output_items,
        }
