"""Tool-calling LLM agent for dataset-grounded reasoning."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from typing import Any, Callable, Sequence

from .llm_call import DEFAULT_MODEL, LLMCaller
from .tools import TOOL_FUNCTIONS, TOOLS, THINKING_TEXTS

from prompts import TOOL_CALLING_SYSTEM_PROMPT


def call_tool(tool_name: str, arguments: str | dict[str, Any], dataset_name: str | None = None) -> dict[str, Any]:
    """Execute a registered tool by name with the given JSON-encoded arguments."""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}

    tool_fn = TOOL_FUNCTIONS[tool_name]

    if isinstance(arguments, str):
        try:
            tool_args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid tool arguments: {exc}"}
    else:
        tool_args = arguments

    if not isinstance(tool_args, dict):
        return {"error": "Tool arguments must decode to a JSON object"}

    if dataset_name is not None and "dataset_name" in inspect.signature(tool_fn).parameters:
        tool_args = tool_args.copy()
        tool_args["dataset_name"] = dataset_name

    try:
        return tool_fn(**tool_args)
    except Exception as exc:
        return {"error": str(exc)}


def _build_tool_output(tool_output: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Build the ``output`` value for a ``function_call_output`` item.

    The Responses API ``function_call_output`` supports multimodal content
    natively.  When the tool result contains images, we return a list with
    ``input_image`` entries so the model can analyse them.

    For regular (non-image) results the output is a plain JSON string.
    """
    images: list[str] = tool_output.get("images") or []
    if not images:
        return json.dumps(tool_output)

    # Build a multimodal output list for the Responses API
    non_image_fields = {k: v for k, v in tool_output.items() if k != "images"}
    parts: list[dict[str, Any]] = []

    if non_image_fields:
        parts.append({"type": "input_text", "text": json.dumps(non_image_fields)})

    for data_url in images:
        parts.append({
            "type": "input_image",
            "image_url": data_url,
        })

    return parts


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
            def _wrapped_handler(event: dict[str, Any], redacted_thinking: bool = False) -> None:
                event_type = event.get("type")
                if event_type == "assistant_reasoning_delta":
                    on_stream_event({"type": "thinking", "content": event.get("delta", "")})
                elif event_type == "tool_call_delta":
                    name_delta = event.get("name_delta")

                    if not redacted_thinking:
                        if name_delta:
                            on_stream_event({"type": "thinking", "content": f"\n\n> Using tool: {name_delta}\n> Arguments: "})
                        args_delta = event.get("arguments_delta")
                        if args_delta:
                            on_stream_event({"type": "thinking", "content": args_delta})
                    else:
                        if name_delta and name_delta in THINKING_TEXTS:
                            on_stream_event({"type": "thinking", "content": f"\n\n> {THINKING_TEXTS[name_delta]}"})

            wrapped_on_stream_event = _wrapped_handler

        input_items: list[dict[str, Any]] = [
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
                input=input_items,
                tools=TOOLS,
                on_stream_event=wrapped_on_stream_event,
            )

            assistant_content = (message_payload.get("content") or "").strip()
            tool_calls = message_payload.get("tool_calls", [])
            output_items = message_payload.get("output_items", [])

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

            # Append the model's output items back to input for multi-turn
            for item in output_items:
                input_items.append(item)

            for call in tool_calls:
                tool_name = call.get("name", "")
                call_id = call.get("id", "")
                tool_output = call_tool(
                    tool_name=tool_name,
                    arguments=call.get("arguments", "{}"),
                    dataset_name=dataset_name,
                )

                if on_stream_event:
                    on_stream_event(
                        {
                            "type": "tool_call_result",
                            "tool_name": tool_name,
                            "tool_call_id": call_id,
                            "arguments": call.get("arguments", "{}"),
                            "output": tool_output,
                        }
                    )

                tool_trace.append(
                    {
                        "tool_name": tool_name,
                        "arguments": call.get("arguments", "{}"),
                        "output": tool_output,
                    }
                )

                # Build the function_call_output content
                output_content = _build_tool_output(tool_output)

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_content,
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
