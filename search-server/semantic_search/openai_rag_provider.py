"""OpenAI RAG (Retrieval Augmented Generation) provider."""

import os
import json
import sqlite3
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

from .base import SemanticSearchProvider
from .bm25_provider import BM25Provider

from prompts import OPENAI_RAG_QUERY_REWRITE_PROMPT, OPENAI_RAG_RERANK_PROMPT

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
        db_path: str,
        model: str = "gpt-5-mini",
        api_key: str = None,
        bm25_top_k: int = 20,
    ):
        """
        Initialize the OpenAI RAG provider.

        Args:
            db_path: Path to the SQLite components database
            model: The OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            bm25_top_k: Number of candidates to retrieve from BM25 (default: 20)
        """
        self.db_path = db_path
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        # Initialize BM25 provider for retrieval (also uses db_path)
        self.bm25_provider = BM25Provider(
            db_path=db_path,
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
            print(f"[RAG] BM25 Search Terms: {rewritten_query}\n\n")
            return reranked_results

        except Exception as e:
            print(f"Error in OpenAI RAG provider: {e}")
            return {
                "component_ids": [],
                "reason": f"[RAG] Error in OpenAI RAG provider: {e}",
            }

    def _rewrite_query(self, query: str) -> str:
        """
        Use OpenAI to rewrite the user query into better search terms for BM25.

        Args:
            query: Original user query

        Returns:
            Rewritten query optimized for BM25 retrieval
        """
        prompt = OPENAI_RAG_QUERY_REWRITE_PROMPT.format(query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a spatial search assistant that is helping answer user queries about a room.",
                    },
                    {"role": "user", "content": prompt},
                ],
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
        # Fetch captions for candidates fresh from DB
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        placeholders = ",".join("?" * len(candidate_ids))
        cur.execute(
            f"SELECT component_id, caption FROM components WHERE component_id IN ({placeholders})",
            candidate_ids,
        )
        caption_map = {
            row["component_id"]: row["caption"] or "" for row in cur.fetchall()
        }
        con.close()

        # Build candidate descriptions
        candidates_text = ""
        for i, comp_id in enumerate(candidate_ids):
            caption = caption_map.get(comp_id, "")
            candidates_text += f"{i+1}. Component {comp_id}: {caption}\n\n"

        prompt = OPENAI_RAG_RERANK_PROMPT.format(
            original_query=original_query, candidates_text=candidates_text
        )

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
                # Validate IDs — only keep IDs that are in candidate_ids
                valid_ids = [
                    id
                    for id in reranked_ids
                    if id in caption_map and id in candidate_ids
                ]
            else:
                valid_ids = []

            # Use the LLM's reasoning
            if valid_ids:
                return {
                    "component_ids": valid_ids,
                    "reason": reason,
                }
            else:
                return {
                    "component_ids": valid_ids,
                    "reason": "No components matched the search criteria after re-ranking.",
                }

        except Exception as e:
            print(f"Error re-ranking with LLM: {e}")
            # Fallback: return first few candidates
            fallback_ids = candidate_ids[:3]
            return {
                "component_ids": fallback_ids,
                "reason": f"Using top BM25 results (re-ranking failed: {e})",
            }
