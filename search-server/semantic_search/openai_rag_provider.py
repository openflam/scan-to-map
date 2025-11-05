"""OpenAI RAG (Retrieval Augmented Generation) provider."""

import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from .base import SemanticSearchProvider
from .bm25_provider import BM25Provider

# Load variables from .env into os.environ
load_dotenv()


class OpenAIRAGProvider(SemanticSearchProvider):
    """
    Semantic search provider using OpenAI with Retrieval Augmented Generation.

    This provider:
    1. Uses OpenAI to rewrite the query into better search terms
    2. Uses BM25 to retrieve top 20 candidates
    3. Uses OpenAI to re-rank and select the best matches
    """

    def __init__(
        self,
        component_captions: Dict[int, Any],
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str = None,
        bm25_top_k: int = 20,
    ):
        """
        Initialize the OpenAI RAG provider.

        Args:
            component_captions: Dictionary keyed by component ID with captions
            model: The OpenAI model to use
            temperature: Sampling temperature
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            bm25_top_k: Number of candidates to retrieve from BM25 (default: 20)
        """
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.component_captions = component_captions

        # Initialize BM25 provider for retrieval
        self.bm25_provider = BM25Provider(
            component_captions=component_captions,
            top_k=bm25_top_k,
            gap_threshold=float("inf"),  # Disable elbow detection to get all top_k
            ratio_threshold=0.0,
        )

    def match_components(self, query: str) -> Dict[str, Any]:
        """
        Match a search query to component descriptions using OpenAI RAG.

        Args:
            query: The search query string

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice with captions
        """
        try:
            # Step 1: Rewrite query using OpenAI
            rewritten_query = self._rewrite_query(query)
            print(f"Original query: {query}")
            print(f"Rewritten query: {rewritten_query}")

            # Step 2: Retrieve top candidates using BM25
            bm25_results = self.bm25_provider.match_components(rewritten_query)
            candidate_ids = bm25_results["component_ids"]

            if not candidate_ids:
                return {
                    "component_ids": [],
                    "reason": f"[RAG] BM25 Search Terms: {rewritten_query}\n\nNo matching components found during retrieval.",
                }

            print(f"Retrieved {len(candidate_ids)} candidates from BM25")

            # Step 3: Re-rank candidates using OpenAI
            reranked_results = self._rerank_with_llm(query, candidate_ids)
            reranked_results["reason"] = (
                f"[RAG] BM25 Search Terms: {rewritten_query}\n\n"
                + reranked_results["reason"]
            )
            return reranked_results

        except Exception as e:
            print(f"Error in OpenAI RAG provider: {e}")
            return self._get_fallback_result(
                reason=f"[RAG] Error in OpenAI RAG provider: {e}"
            )

    def _rewrite_query(self, query: str) -> str:
        """
        Use OpenAI to rewrite the user query into better search terms for BM25.

        Args:
            query: Original user query

        Returns:
            Rewritten query optimized for BM25 retrieval
        """
        prompt = f"""Rewrite the following user query into a set of search terms suitable for BM25 retrieval.
Extract key concepts, expand abbreviations, and include relevant synonyms.
Focus on nouns and descriptive terms that would appear in object descriptions.

User Query: "{query}"

Respond with ONLY the rewritten search terms (no explanation needed)."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that rewrites search queries to improve retrieval. Respond with only the rewritten query.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=100,
            )

            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query

        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query  # Fallback to original query

    def _rerank_with_llm(
        self, original_query: str, candidate_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Use OpenAI to re-rank the candidate components.

        Args:
            original_query: The original user query
            candidate_ids: List of candidate component IDs from BM25

        Returns:
            Dictionary with re-ranked component_ids and reason
        """
        # Build candidate descriptions
        candidates_text = ""
        for i, comp_id in enumerate(candidate_ids):
            caption = self.component_captions[comp_id].get("caption", "")
            candidates_text += f"{i+1}. Component {comp_id}: {caption}\n\n"

        prompt = f"""Given the following search query and a list of candidate components retrieved by BM25, 
select and rank the components that best match the query.

Only return those components that contain the objects almost exclusively without unrelated objects.
Return the components in order of relevance (most relevant first).

Search Query: "{original_query}"

Candidate Components:
{candidates_text}

Respond with ONLY a JSON object containing:
1. "component_ids": a comma-separated string of integer IDs of the matching components in order of relevance (e.g., "2,5,7")
2. "reason": a brief explanation of why these components were selected and their ranking

Example response format:
{{"component_ids": "2,5,7", "reason": "Component 2 contains printers which directly match the query, followed by components 5 and 7 which also show printing equipment."}}

If no components match well, return the best matching component ID anyway."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that ranks search results. Always respond with a valid JSON object containing component_ids (comma-separated string) and reason (string).",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            result = response.choices[0].message.content.strip()
            parsed_result = json.loads(result)

            component_ids_str = parsed_result.get("component_ids", "")
            reason = parsed_result.get(
                "reason", "Selected and ranked based on query match."
            )

            # Parse component IDs
            if component_ids_str:
                reranked_ids = [
                    int(id.strip()) for id in component_ids_str.split(",") if id.strip()
                ]
                # Validate IDs
                valid_ids = [
                    id
                    for id in reranked_ids
                    if id in self.component_captions and id in candidate_ids
                ]
            else:
                valid_ids = []

            # Build detailed reason with captions
            if valid_ids:
                caption_details = []
                for comp_id in valid_ids:
                    caption = self.component_captions[comp_id].get("caption", "")
                    caption_details.append(f"Component {comp_id}: {caption}")

                detailed_reason = f"{reason}\n\n" + "\n\n".join(caption_details)
            else:
                detailed_reason = (
                    "No components matched the search criteria after re-ranking."
                )

            return {
                "component_ids": valid_ids,
                "reason": detailed_reason,
            }

        except Exception as e:
            print(f"Error re-ranking with LLM: {e}")
            # Fallback: return first few candidates
            fallback_ids = candidate_ids[:3]
            caption_details = []
            for comp_id in fallback_ids:
                caption = self.component_captions[comp_id].get("caption", "")
                caption_details.append(f"Component {comp_id}: {caption}")

            return {
                "component_ids": fallback_ids,
                "reason": f"Using BM25 results (re-ranking failed: {e})\n\n"
                + "\n\n".join(caption_details),
            }

    def _get_fallback_result(self, reason: str) -> Dict[str, Any]:
        """Get fallback result with first component."""
        first_component_id = list(self.component_captions.keys())[0]
        return {"component_ids": [first_component_id], "reason": reason}
