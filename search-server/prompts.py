# Prompts for OpenAI RAG provider
OPENAI_RAG_QUERY_REWRITE_PROMPT = \
"""Rewrite the following user query into a set of search terms suitable for BM25 retrieval.
Extract key concepts, expand abbreviations, and include relevant synonyms.
Focus on nouns and descriptive terms that would appear in object descriptions.

User Query: {query}

Respond with ONLY the rewritten search terms (no explanation needed)."""


OPENAI_RAG_RERANK_PROMPT = \
"""Given the following search query and a list of candidate components retrieved by BM25, 
select and rank the components that best match the query.

Only return those components that contain the objects almost exclusively without unrelated objects.
Return the components in order of relevance (most relevant first).

Search Query: {original_query}

Candidate Components:
{candidates_text}

Respond with ONLY a JSON object containing:
1. "component_ids": a comma-separated string of integer IDs of the matching components in order of relevance (e.g., "2,5,7")
2. "reason": a brief explanation of why these components were selected and their ranking

Example response format:
{{"component_ids": "2,5,7", "reason": "Component 2 contains printers which directly match the query, followed by components 5 and 7 which also show printing equipment."}}

If no components match well, return the best matching component ID anyway."""

# Prompts for OpenAI provider
OPENAI_FULL_CONTEXT_COMPONENT_MATCHING_PROMPT = \
"""Given the following search query and a list of object descriptions, determine which component IDs best match the query. 

Only return those components that contain the objects almost exclusively without unrelated objects.

Search Query: {query}

Available Components:
{components_text}

Respond with ONLY a JSON object containing:
1. "component_ids": a comma-separated string of integer IDs of ALL components that match (e.g., "2,5,7" or "3" for a single match)
2. "reason": a brief one-sentence explanation of why these components match the query

Example response format:
{{"component_ids": "2,5,7", "reason": "These components contain printers which match the search query."}}

If multiple components match, include all of them. If no component matches well, choose the component that is closest to the query."""

OPENAI_FULL_CONTEXT_SYSTEM_PROMPT = \
"""You are a helpful assistant that matches search queries to object descriptions. 
Always respond with a valid JSON object containing component_ids (comma-separated string of integers) and reason (string)."""