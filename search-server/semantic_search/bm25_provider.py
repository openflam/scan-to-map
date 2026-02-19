"""BM25-based semantic search provider."""

import sqlite3
from typing import Any, Dict, List, Tuple
import bm25s
import Stemmer

from .base import SemanticSearchProvider


class BM25Provider(SemanticSearchProvider):
    """Semantic search provider using BM25 algorithm."""

    def __init__(
        self,
        db_path: str,
        top_k: int = 10,
        stemmer_language: str = "english",
        stopwords: str = "en",
        gap_threshold: float = 1.0,
        ratio_threshold: float = 0.7,
    ):
        """
        Initialize the BM25 provider.

        Args:
            db_path: Path to the SQLite components database
            top_k: Number of top matching components to return
            stemmer_language: Language for the stemmer (default: "english")
            stopwords: Stopwords to use (default: "en")
            gap_threshold: Minimum gap in scores to stop including results (elbow detection)
            ratio_threshold: Minimum ratio of current/previous score to continue (elbow detection)
        """
        self.db_path = db_path
        self.top_k = top_k
        self.stemmer = Stemmer.Stemmer(stemmer_language)
        self.stopwords = stopwords
        self.gap_threshold = gap_threshold
        self.ratio_threshold = ratio_threshold

    def _load_corpus(self) -> Tuple[List[int], List[str]]:
        """Fetch all component IDs and captions fresh from the database."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "SELECT component_id, caption FROM components ORDER BY component_id"
        )
        rows = cur.fetchall()
        con.close()
        component_ids = [row["component_id"] for row in rows]
        corpus = [row["caption"] or "" for row in rows]
        return component_ids, corpus

    def match_components(self, query: str) -> Dict[str, Any]:
        """
        Match a search query to component descriptions using BM25.

        Fetches the latest data from the database on every call.

        Args:
            query: The search query string

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        # Load fresh corpus from DB
        component_ids, corpus = self._load_corpus()

        # Build a fresh BM25 index from the current corpus
        corpus_tokens = bm25s.tokenize(
            corpus, stopwords=self.stopwords, stemmer=self.stemmer
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # Tokenize the query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        # Retrieve top-k results
        results, scores = retriever.retrieve(query_tokens, k=self.top_k)

        # Apply elbow logic to determine which results to keep
        n = results.shape[1]
        keep = [0]  # Always keep the top result

        for i in range(1, n):
            prev_score = scores[0, i - 1]
            cur_score = scores[0, i]

            # Stop if score is not positive
            if cur_score <= 0:
                break

            # Apply elbow detection: stop if there's a significant gap or ratio drop
            if (prev_score - cur_score) >= self.gap_threshold or (
                cur_score / max(prev_score, 1e-9)
            ) <= self.ratio_threshold:
                break

            keep.append(i)

        # Extract component IDs for the kept results
        matched_component_ids = []

        for i in keep:
            doc_idx = results[0, i]
            comp_id = component_ids[doc_idx]
            matched_component_ids.append(comp_id)

        # Create concise reason
        if len(matched_component_ids) > 0:
            reason = f"Found {len(matched_component_ids)} matching component(s) using BM25 ranked search."
        else:
            reason = "No matching components found."

        return {
            "component_ids": matched_component_ids,
            "reason": reason,
        }
