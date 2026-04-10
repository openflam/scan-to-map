"""OpenAI-based semantic search provider."""

import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from .base import SemanticSearchProvider

from prompts import (
    OPENAI_ANSWER_POOL_SUFFIX,
    OPENAI_FULL_CONTEXT_COMPONENT_MATCHING_PROMPT,
    OPENAI_FULL_CONTEXT_SYSTEM_PROMPT,
    OPENAI_FULL_CONTEXT_SYSTEM_PROMPT_WITH_ANSWER_POOL,
)

from spatial_db import database
from utils.scanqa_category import bucket

# Load variables from .env into os.environ
load_dotenv()

_DEFAULT_POOLS_PATH = Path(__file__).resolve().parents[1] / "data" / "scanqa_answer_pools" / "pools.json"


def _load_answer_pools_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pools = data.get("pools")
    if not isinstance(pools, dict):
        return None
    return data


def _format_numbered_answer_list(strings: List[str]) -> str:
    lines = []
    for i, s in enumerate(strings, start=1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


class OpenAIProvider(SemanticSearchProvider):
    """Semantic search provider using OpenAI's API."""

    def __init__(
        self,
        dataset_name: str,
        model: str = "gpt-5-mini",
        max_completion_tokens: int = 1000,
        api_key: str = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            dataset_name: Dataset to search
            model: The OpenAI model to use
            max_completion_tokens: Maximum tokens in response
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
        """
        super().__init__(dataset_name)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        pool_path = os.environ.get("SCANQA_ANSWER_POOLS_JSON")
        self._answer_pools_data = _load_answer_pools_json(
            Path(pool_path).expanduser()
            if pool_path
            else _DEFAULT_POOLS_PATH
        )
        self._answer_pool_max_lines = int(
            os.environ.get("SCANQA_ANSWER_POOL_MAX_LINES", "600")
        )

    def _load_components(self) -> Tuple[str, set]:
        """
        Fetch all components fresh from the database.

        Returns:
            Tuple of (formatted components text, set of valid component IDs)
        """
        rows = database.fetch_all_components(self.dataset_name)

        components_text = ""
        valid_ids = set()
        for row in rows:
            comp_id = row["component_id"]
            caption = row["caption"] or ""
            components_text += f"Component {comp_id}: {caption}\n\n"
            valid_ids.add(comp_id)
        return components_text, valid_ids

    def match_components(self, query: str) -> Dict[str, Any]:
        """
        Match a search query to component descriptions using OpenAI.

        Fetches the latest data from the database on every call.

        Args:
            query: The search query string

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        components_text, valid_ids = self._load_components()

        # Build the prompt using freshly loaded components text
        prompt = OPENAI_FULL_CONTEXT_COMPONENT_MATCHING_PROMPT.format(
            query=query, components_text=components_text
        )
        system_prompt = OPENAI_FULL_CONTEXT_SYSTEM_PROMPT

        if self._answer_pools_data:
            pools = self._answer_pools_data.get("pools") or {}
            cat = bucket(query)
            pool_list = pools.get(cat)
            if isinstance(pool_list, list) and len(pool_list) > 0:
                trimmed = pool_list[: self._answer_pool_max_lines]
                answer_options_text = _format_numbered_answer_list(trimmed)
                prompt += OPENAI_ANSWER_POOL_SUFFIX.format(
                    category=cat,
                    answer_options_text=answer_options_text,
                )
                system_prompt = OPENAI_FULL_CONTEXT_SYSTEM_PROMPT_WITH_ANSWER_POOL

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self.max_completion_tokens,
                response_format={"type": "json_object"},
            )

            # Extract and parse the response
            raw_content = response.choices[0].message.content
            if not raw_content or not raw_content.strip():
                raise ValueError("Empty response from OpenAI")
            result = raw_content.strip()
            parsed_result = json.loads(result)

            component_ids_str = parsed_result.get("component_ids", "")
            llm_reason = parsed_result.get("reason", "Selected based on query match.")

            # Parse comma-separated component IDs and validate against DB
            component_ids = self._parse_component_ids(component_ids_str, valid_ids)

            out: Dict[str, Any] = {"component_ids": component_ids, "reason": llm_reason}
            best = parsed_result.get("best_answer_from_pool")
            top10 = parsed_result.get("top_10_answers_from_pool")
            if best is not None or top10 is not None:
                sel: Dict[str, Any] = {"category": bucket(query)}
                if best is not None:
                    sel["best_answer_from_pool"] = best
                if isinstance(top10, list):
                    sel["top_10_answers_from_pool"] = top10[:10]
                elif top10 is not None:
                    sel["top_10_answers_from_pool"] = top10
                out["answer_selection"] = sel

            return out

        except Exception as e:
            print(f"Error processing query with OpenAI: {e}")
            fallback_id = next(iter(valid_ids)) if valid_ids else None
            if fallback_id is None:
                return {"component_ids": [], "reason": f"Error: {e}"}
            return {
                "component_ids": [fallback_id],
                "reason": f"Fallback: Error processing query with OpenAI: {e}",
            }

    def _parse_component_ids(self, component_ids_str: str, valid_ids: set) -> List[int]:
        """Parse comma-separated component IDs and validate them against the DB set."""
        try:
            component_ids = [
                int(id.strip()) for id in component_ids_str.split(",") if id.strip()
            ]
            valid = [id for id in component_ids if id in valid_ids]
            if not valid:
                raise ValueError("No valid component IDs found")
            return valid
        except (ValueError, AttributeError) as e:
            print(
                f"Warning: Could not parse component IDs from '{component_ids_str}': {e}"
            )
            fallback_id = next(iter(valid_ids)) if valid_ids else None
            return [fallback_id] if fallback_id is not None else []
