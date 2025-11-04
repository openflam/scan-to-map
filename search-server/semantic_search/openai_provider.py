"""OpenAI-based semantic search provider."""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

from .base import SemanticSearchProvider
from .prompts import build_component_matching_prompt, SYSTEM_PROMPT

# Load variables from .env into os.environ
load_dotenv()


class OpenAIProvider(SemanticSearchProvider):
    """Semantic search provider using OpenAI's API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 100,
        api_key: str = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model: The OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def match_components(
        self, query: str, component_captions: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Match a search query to component descriptions using OpenAI.

        Args:
            query: The search query string
            component_captions: Dictionary keyed by component ID with captions

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        # Build components text for the prompt
        components_text = self._build_components_text(component_captions)

        # Build the prompt
        prompt = build_component_matching_prompt(query, components_text)

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            # Extract and parse the response
            result = response.choices[0].message.content.strip()
            parsed_result = json.loads(result)

            component_ids_str = parsed_result.get("component_ids", "")
            reason = parsed_result.get("reason", "Selected based on query match.")

            # Parse comma-separated component IDs
            component_ids = self._parse_component_ids(
                component_ids_str, component_captions
            )

            return {"component_ids": component_ids, "reason": reason}

        except Exception as e:
            print(f"Error processing query with OpenAI: {e}")
            # Fallback: return the first component
            return self._get_fallback_result(
                component_captions,
                "Error occurred during search. Showing first available component.",
            )

    def _build_components_text(self, component_captions: Dict[int, Any]) -> str:
        """Build formatted text of all component descriptions."""
        components_text = ""
        for component_id, component_data in component_captions.items():
            caption = component_data.get("caption", "")
            components_text += f"Component {component_id}: {caption}\n\n"
        return components_text

    def _parse_component_ids(
        self, component_ids_str: str, component_captions: Dict[int, Any]
    ) -> List[int]:
        """Parse comma-separated component IDs and validate them."""
        try:
            component_ids = [
                int(id.strip()) for id in component_ids_str.split(",") if id.strip()
            ]
            # Validate that all IDs exist
            valid_ids = [id for id in component_ids if id in component_captions]
            if not valid_ids:
                raise ValueError("No valid component IDs found")
            return valid_ids
        except (ValueError, AttributeError) as e:
            print(
                f"Warning: Could not parse component IDs from '{component_ids_str}': {e}"
            )
            # Fallback to first component
            return [list(component_captions.keys())[0]]

    def _get_fallback_result(
        self, component_captions: Dict[int, Any], reason: str
    ) -> Dict[str, Any]:
        """Get fallback result with first component."""
        first_component_id = list(component_captions.keys())[0]
        return {"component_ids": [first_component_id], "reason": reason}
