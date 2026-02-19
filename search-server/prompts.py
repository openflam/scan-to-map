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
2. "reason": If the user asked a question or is seeking information about the components, answer the question here. If the query is a search query, provide a brief one-sentence explanation of why these components match the query. 

Example response format:
{{"component_ids": "2,5,7", "reason": "These component contain printers which match the search query. You can use them to print your documents."}}

If no components match well, return the best matching component ID anyway."""

# Prompts for OpenAI provider
OPENAI_FULL_CONTEXT_COMPONENT_MATCHING_PROMPT = """Given the following user query and a list of component descriptions from a scene, determine which component IDs are most relevant.

The user query may be:
1. A search query (e.g., "red chairs near the window")
2. A factual question about a specific object (e.g., "What is the history of the oldest statue in this building?")
3. A comparative or superlative question (e.g., "Which is the largest painting?")
4. A descriptive question (e.g., "What material is the central table made of?")

Your task:
- Identify the component(s) that best match or answer the query.
- Select ONLY components that are strongly relevant.
- Prefer components that almost exclusively contain the object(s) needed to answer the query.
- If the question implies a superlative condition (oldest, largest, tallest, etc.), determine which component satisfies it based only on the provided descriptions.
- If multiple components equally satisfy the query, include all of them.
- If no component matches perfectly, choose the closest and most relevant one.

Search Query:
{query}

Available Components:
{components_text}

Respond with ONLY a valid JSON object containing:
1. "component_ids": a comma-separated string of integer IDs (e.g., "2,5,7" or "3")
2. "reason":
   - If the query is a search request, provide a brief one-sentence explanation of why the components match.
   - If the query is a question, directly answer the question using the information from the selected component(s).

Example (search query):
{{"component_ids": "2,5", "reason": "These components contain red chairs near the window, matching the search query."}}

Example (question):
{{"component_ids": "4", "reason": "The oldest statue is the Venus De Milo, sculpted around 130–100 BCE in ancient Greece."}}
"""


OPENAI_FULL_CONTEXT_SYSTEM_PROMPT = """You are a helpful assistant that maps user queries to scene components.

The user may submit either:
- A search query describing objects
- A factual question about objects
- A comparative question (e.g., oldest, largest, closest)
- A descriptive question about properties or history

You must:
- Select the most relevant component(s) based ONLY on the provided descriptions.
- Infer superlatives (e.g., oldest, tallest) when necessary.
- Answer factual questions in the "reason" field using information from the selected component(s).
- Always respond with a valid JSON object.

The JSON format must be:
{
  "component_ids": "comma-separated integer IDs",
  "reason": "answer or explanation"
}

Do not include any text outside the JSON object.
Do not include markdown formatting.
Do not explain your reasoning separately.
Return JSON only.
"""
