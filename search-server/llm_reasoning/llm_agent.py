"""Tool-calling LLM agent for dataset-grounded reasoning."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from typing import Any, Callable, Sequence

from .llm_call import DEFAULT_MODEL, LLMCaller
from .tools import TOOL_FUNCTIONS, TOOLS

from prompts import TOOL_CALLING_SYSTEM_PROMPT


class LLMAgent:
    """Runs a tool-calling loop with the configured LLM."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = TOOL_CALLING_SYSTEM_PROMPT,
        max_tool_rounds: int = 8,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds
        self.caller = LLMCaller(model=model, api_key=api_key)

    def answer_query(self, query: str, dataset_name: str) -> dict[str, Any]:
        return self.answer_query_stream(query=query, dataset_name=dataset_name)

    def answer_query_stream(
        self,
        query: str,
        dataset_name: str,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Answer a natural-language query for a given dataset using tool calls.

        Args:
            query: User query in natural language
            dataset_name: Dataset to run tools against (never exposed to model)
            on_stream_event: Callback receiving event dicts while model streams

        Returns:
            Dict containing final response text plus tool trace data.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not dataset_name or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string")

        wrapped_on_stream_event = None
        if on_stream_event:
            def _wrapped_handler(event: dict[str, Any]) -> None:
                event_type = event.get("type")
                if event_type == "assistant_reasoning_delta":
                    on_stream_event({"type": "thinking", "content": event.get("delta", "")})
                elif event_type == "tool_call_delta":
                    name_delta = event.get("name_delta")
                    if name_delta:
                        on_stream_event({"type": "thinking", "content": f"\n\n> Using tool: {name_delta}\n> Arguments: "})
                    args_delta = event.get("arguments_delta")
                    if args_delta:
                        on_stream_event({"type": "thinking", "content": args_delta})
                # elif event_type == "assistant_text_delta":
                #     delta = event.get("delta", "")
                #     if delta:
                #         on_stream_event({"type": "thinking", "content": delta})

            wrapped_on_stream_event = _wrapped_handler

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    "Use tools as needed, then respond with a JSON object containing "
                    "component_ids (list of ints) and reason (string)."
                ),
            },
        ]
        tool_trace: list[dict[str, Any]] = []

        for _ in range(self.max_tool_rounds):
            message_payload = self.caller.stream_chat(
                messages=messages,
                tools=TOOLS,
                on_stream_event=wrapped_on_stream_event,
            )

            assistant_content = (message_payload.get("content") or "").strip()
            tool_calls = message_payload.get("tool_calls", [])

            if not tool_calls:
                component_ids, reason = _parse_final_response(assistant_content)
                return {
                    "dataset_name": dataset_name,
                    "query": query,
                    "model": self.model,
                    "response": assistant_content,
                    "component_ids": component_ids,
                    "reason": reason,
                    "tool_trace": tool_trace,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            )

            for idx, call in enumerate(tool_calls):
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                tool_name = fn.get("name")
                tool_call_id = call.get("id") or f"tool_call_{idx}"
                if tool_name not in TOOL_FUNCTIONS:
                    tool_output: dict[str, Any] = {
                        "error": f"Unknown tool: {tool_name}",
                    }
                else:
                    tool_fn = TOOL_FUNCTIONS[tool_name]
                    try:
                        tool_args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError as exc:
                        tool_args = None
                        tool_output = {"error": f"Invalid tool arguments: {exc}"}

                    if tool_args is not None:
                        if not isinstance(tool_args, dict):
                            tool_output = {
                                "error": "Tool arguments must decode to a JSON object"
                            }
                        else:
                            if "dataset_name" in inspect.signature(tool_fn).parameters:
                                tool_args["dataset_name"] = dataset_name
                            try:
                                tool_output = tool_fn(**tool_args)
                            except Exception as exc:
                                tool_output = {"error": str(exc)}

                if on_stream_event:
                    on_stream_event(
                        {
                            "type": "tool_call_result",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "arguments": fn.get("arguments", "{}"),
                            "output": tool_output,
                        }
                    )

                tool_trace.append(
                    {
                        "tool_name": tool_name,
                        "arguments": fn.get("arguments", "{}"),
                        "output": tool_output,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_output),
                    }
                )

        return {
            "dataset_name": dataset_name,
            "query": query,
            "model": self.model,
            "response": "Tool-calling loop stopped before a final answer was produced.",
            "component_ids": [],
            "reason": "Tool-calling loop reached the maximum number of rounds without a final answer.",
            "tool_trace": tool_trace,
        }


def _parse_final_response(content: str) -> tuple[list[int], str]:
    """
    Extract component_ids and reason from the model's final JSON response.

    Tries json.loads on the full content first; falls back to finding the
    outermost {...} block if the model included extra prose.
    """
    text = content.strip()
    candidates: list[str] = [text]

    # Also try the last {...} block in case there is leading prose
    match = re.search(r"(\{[^{}]*\})", text, re.DOTALL)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            raw_ids = parsed.get("component_ids") or []
            component_ids: list[int] = []
            for x in raw_ids:
                try:
                    component_ids.append(int(x))
                except (TypeError, ValueError):
                    pass
            reason: str = str(parsed.get("reason") or text)
            return component_ids, reason
        except (json.JSONDecodeError, AttributeError):
            continue

    return [], text


def answer_query(
    query: str,
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Convenience function for single-shot use."""
    agent = LLMAgent(model=model)
    return agent.answer_query_stream(
        query=query,
        dataset_name=dataset_name,
        on_stream_event=on_stream_event,
    )


def _cli_stream_handler(
    show_reasoning: bool, show_tools: bool
) -> Callable[[dict[str, Any]], None]:
    def _handler(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "assistant_text_delta":
            print(event.get("delta") or "", end="", flush=True)
        elif event_type == "assistant_reasoning_delta" and show_reasoning:
            print(event.get("delta") or "", end="", flush=True)
        elif event_type == "tool_call_result" and show_tools:
            try:
                args_str = json.dumps(
                    json.loads(event.get("arguments") or "{}"), ensure_ascii=False
                )
            except (json.JSONDecodeError, TypeError):
                args_str = event.get("arguments") or "{}"
            print(
                f"\n[Tool called: {event.get('tool_name')}\nArguments: {args_str}]",
                flush=True,
            )

    return _handler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test the LLM agent from the command line."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Print reasoning tokens to stderr"
    )
    parser.add_argument(
        "--show-tools", action="store_true", help="Print tool results to stderr"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print full result JSON after response"
    )
    args = parser.parse_args(argv)

    handler = _cli_stream_handler(
        show_reasoning=args.show_reasoning, show_tools=args.show_tools
    )
    result = answer_query(
        query=args.query,
        dataset_name=args.dataset,
        model=args.model,
        on_stream_event=handler,
    )
    print()  # newline after streamed text
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
