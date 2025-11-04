"""Base class for semantic search providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class SemanticSearchProvider(ABC):
    """Abstract base class for semantic search providers."""

    @abstractmethod
    def match_components(
        self, query: str, component_captions: Optional[Dict[int, Any]] = None
    ) -> Dict[str, Any]:
        """
        Match a search query to component descriptions.

        Args:
            query: The search query string
            component_captions: Optional dictionary keyed by component ID with captions.
                               If not provided, uses the captions from initialization.

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        pass
