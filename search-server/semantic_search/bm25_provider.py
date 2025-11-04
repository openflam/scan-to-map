"""BM25-based semantic search provider."""

from typing import Dict, Any, List, Optional
import bm25s
import Stemmer

from .base import SemanticSearchProvider


class BM25Provider(SemanticSearchProvider):
    """Semantic search provider using BM25 algorithm."""

    def __init__(
        self,
        component_captions: Dict[int, Any],
        top_k: int = 10,
        stemmer_language: str = "english",
        stopwords: str = "en",
        gap_threshold: float = 1.0,
        ratio_threshold: float = 0.7,
    ):
        """
        Initialize the BM25 provider and build the search index.

        Args:
            component_captions: Dictionary keyed by component ID with captions
            top_k: Number of top matching components to return
            stemmer_language: Language for the stemmer (default: "english")
            stopwords: Stopwords to use (default: "en")
            gap_threshold: Minimum gap in scores to stop including results (elbow detection)
            ratio_threshold: Minimum ratio of current/previous score to continue (elbow detection)
        """
        self.top_k = top_k
        self.stemmer = Stemmer.Stemmer(stemmer_language)
        self.stopwords = stopwords
        self.gap_threshold = gap_threshold
        self.ratio_threshold = ratio_threshold

        # Build the index from component captions
        self.component_ids = list(component_captions.keys())
        self.corpus = [
            component_captions[comp_id].get("caption", "")
            for comp_id in self.component_ids
        ]

        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(
            self.corpus, stopwords=self.stopwords, stemmer=self.stemmer
        )

        # Create and index the BM25 retriever
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def match_components(
        self, query: str, component_captions: Optional[Dict[int, Any]] = None
    ) -> Dict[str, Any]:
        """
        Match a search query to component descriptions using BM25.

        Args:
            query: The search query string
            component_captions: Not used (component captions are provided at initialization)

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """

        # Tokenize the query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        # Retrieve top-k results
        results, scores = self.retriever.retrieve(query_tokens, k=self.top_k)

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

        # Extract component IDs for the kept results and build reason with captions
        matched_component_ids = []
        caption_details = ["[BM25 Top Matches]"]

        for i in keep:
            doc_idx = results[0, i]
            comp_id = self.component_ids[doc_idx]
            caption = self.corpus[doc_idx]
            score = scores[0, i]

            matched_component_ids.append(comp_id)
            caption_details.append(
                f"Component {comp_id} (score: {score:.2f}): {caption}"
            )

        # Create reason with matched component captions
        if len(matched_component_ids) > 0:
            reason = "\n\n".join(caption_details)
        else:
            reason = "No matching components found."

        return {
            "component_ids": matched_component_ids,
            "reason": reason,
        }
