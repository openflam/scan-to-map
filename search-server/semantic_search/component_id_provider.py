"""Direct component ID search provider."""

from typing import Any, Dict

from .base import SemanticSearchProvider


class ComponentIDProvider(SemanticSearchProvider):
    """Return the component ID supplied as the search query."""

    def match_components(self, query: str) -> Dict[str, Any]:
        """Return the queried component ID as the sole match."""
        component_id = int(query.strip())
        return {
            "component_ids": [component_id],
            "reason": f"Selected component ID {component_id}.",
        }
