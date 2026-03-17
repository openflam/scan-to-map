"""Tools exposed to the LLM reasoning layer."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

import bm25s
import psycopg2
import Stemmer


DEFAULT_PREFILTER_LIMIT = 1000


def _table_name(dataset_name: str) -> str:
    """Return a safe PostgreSQL identifier for a dataset table name."""
    return re.sub(r"[^a-z0-9_]", "_", dataset_name.lower())


def _pg_conn() -> psycopg2.extensions.connection:
    """Open a Postgres/PostGIS connection from environment variables."""
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        dbname=os.environ.get("POSTGRES_DB", "scantomap"),
        user=os.environ.get("POSTGRES_USER", "scantomap"),
        password=os.environ.get("POSTGRES_PASSWORD", "scantomap"),
    )


def _get_dataset_name(dataset_name: str | None) -> str:
    """Resolve dataset from argument first, then from environment."""
    resolved = dataset_name or os.environ.get("DATASET_NAME")
    if not resolved:
        raise ValueError("dataset_name is required (arg or DATASET_NAME env var)")
    return resolved


def _fetch_rows(
    dataset_name: str,
    prefilter_query: str,
    prefilter_limit: int,
) -> list[tuple[int, str, str | None, str | None]]:
    """
    Fetch rows from the dataset table.

    Uses PostgreSQL full-text filtering first (index-backed if present), then
    falls back to scanning all rows if prefilter returns no matches.
    """
    table = _table_name(dataset_name)
    con = _pg_conn()
    try:
        with con.cursor() as cur:
            cur.execute(
                f"""
				SELECT component_id, caption, bbox_json, best_crop
				FROM "{table}"
				WHERE to_tsvector('english', COALESCE(caption, ''))
					@@ plainto_tsquery('english', %s)
				ORDER BY component_id
				LIMIT %s
				""",
                (prefilter_query, prefilter_limit),
            )
            rows = cur.fetchall()

            if rows:
                return rows

            cur.execute(
                f"""
				SELECT component_id, caption, bbox_json, best_crop
				FROM "{table}"
				ORDER BY component_id
				"""
            )
            return cur.fetchall()
    finally:
        con.close()


def search_terms(
    search_terms: list[str],
    dataset_name: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Search component captions with BM25 and return the best matching components.

    Args:
            search_terms: List of term strings from the LLM (e.g. ["red chair", "desk"])
            dataset_name: Dataset to search; falls back to DATASET_NAME env var
            top_k: Max number of results to return

    Returns:
            Dict with ranked matches and metadata.
    """
    if not isinstance(search_terms, list):
        raise ValueError("search_terms must be a list of strings")

    cleaned_terms = [
        term.strip() for term in search_terms if isinstance(term, str) and term.strip()
    ]
    if not cleaned_terms:
        return {
            "results": [],
            "reason": "No valid search terms were provided.",
        }

    resolved_dataset = _get_dataset_name(dataset_name)
    table = _table_name(resolved_dataset)

    rows = _fetch_rows(
        dataset_name=resolved_dataset,
        prefilter_query=" ".join(cleaned_terms),
        prefilter_limit=DEFAULT_PREFILTER_LIMIT,
    )

    if not rows:
        return {
            "dataset_name": resolved_dataset,
            "table": table,
            "search_terms": cleaned_terms,
            "results": [],
            "reason": "No components exist in the dataset table.",
        }

    component_ids = [int(row[0]) for row in rows]
    captions = [row[1] or "" for row in rows]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(captions, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    score_sum: dict[int, float] = defaultdict(float)
    matched_terms: dict[int, list[str]] = defaultdict(list)

    k_each = min(max(1, top_k), len(captions))
    for term in cleaned_terms:
        query_tokens = bm25s.tokenize(term, stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=k_each)

        for i in range(results.shape[1]):
            doc_idx = int(results[0, i])
            score = float(scores[0, i])
            if score <= 0:
                continue
            score_sum[doc_idx] += score
            matched_terms[doc_idx].append(term)

    ranked_doc_indices = sorted(
        score_sum.keys(), key=lambda idx: score_sum[idx], reverse=True
    )[: max(1, top_k)]

    results_payload: list[dict[str, Any]] = []
    for doc_idx in ranked_doc_indices:
        _, caption, bbox_json, best_crop = rows[doc_idx]
        try:
            bbox = json.loads(bbox_json) if bbox_json else {}
        except json.JSONDecodeError:
            bbox = {}

        results_payload.append(
            {
                "component_id": component_ids[doc_idx],
                "caption": caption or "",
                "best_crop": best_crop,
                "bbox": bbox,
                "score": score_sum[doc_idx],
                "matched_terms": matched_terms[doc_idx],
            }
        )

    reason = (
        f"BM25 matched {len(results_payload)} component(s) from {len(rows)} candidate caption(s) "
        f"for {len(cleaned_terms)} search term(s)."
    )

    return {
        "dataset_name": resolved_dataset,
        "table": table,
        "search_terms": cleaned_terms,
        "results": results_payload,
        "reason": reason,
    }


SEARCH_TERMS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_terms",
        "description": (
            "Search component captions using BM25 over the current dataset table and "
            "return the most relevant components."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_terms": {
                    "type": "array",
                    "description": "List of text terms to search for.",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of ranked components to return.",
                    "default": 10,
                    "minimum": 1,
                },
            },
            "required": ["search_terms"],
            "additionalProperties": False,
        },
    },
}


TOOLS = [SEARCH_TERMS_TOOL]

TOOL_FUNCTIONS = {
    "search_terms": search_terms,
}
