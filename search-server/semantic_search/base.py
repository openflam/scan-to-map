"""Base class for semantic search providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class SemanticSearchProvider(ABC):
    """Abstract base class for semantic search providers."""

    @abstractmethod
    def match_components(
        self, query: str, component_captions: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Match a search query to component descriptions.

        Args:
            query: The search query string
            component_captions: Dictionary keyed by component ID with captions

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        pass
