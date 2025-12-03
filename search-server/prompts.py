# Prompts for OpenAI RAG provider
OPENAI_RAG_QUERY_REWRITE_PROMPT = """Assume you are a spatial search assistant that is helping answer use queries about a room. 
The system has access to a database of LLM-generated descriptions of all the objects in the room. 
Reframe the user's questions to a list of search terms, separated by a comma. These search terms will be used to 
retrieve the relevant objects in the room using BM25. BM25 will run on LLM-generated descriptions. Be creative and liberal in the search terms. 
Example: 
Question: "Where can I get a beverage?" 
Search terms: coffee machine, tea bags, cafe, beverage etc. 

Answer ONLY with the comma-separated list of search terms and NOTHING else.

User Query: {query}
"""


OPENAI_RAG_RERANK_PROMPT = """Given the following search query and a list of candidate components retrieved by BM25, 
select and rank the components that best match the query.

Only return those components that contain the objects almost exclusively without unrelated objects.
Return the components in order of relevance (most relevant first).

Search Query: {original_query}

Candidate Components:
{candidates_text}

Respond with ONLY a JSON object containing:
1. "component_ids": a comma-separated string of integer IDs of the matching components in order of relevance (e.g., "2,5,7")
2. "reason": the answer to the user query based on the selected components in a few brief sentences. 

Example response format:
{{"component_ids": "2,5,7", "reason": "These component contain printers which match the search query. You can use them to print your documents."}}

If no components match well, return the best matching component ID anyway."""

# Prompts for OpenAI provider
OPENAI_FULL_CONTEXT_COMPONENT_MATCHING_PROMPT = """Given the following search query and a list of object descriptions, determine which component IDs best match the query. 

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

OPENAI_FULL_CONTEXT_SYSTEM_PROMPT = """You are a helpful assistant that matches search queries to object descriptions. 
Always respond with a valid JSON object containing component_ids (comma-separated string of integers) and reason (string)."""
