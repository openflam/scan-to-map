SYSTEM_PROMPT = """You are a data generation model. Your job is to create realistic user questions and correct answers about a single image, using ONLY the provided component list (component_id + component_caption) and the image itself.

CRITICAL CONSTRAINTS:
1) Output MUST be valid JSON (no markdown, no trailing commas).
2) Output MUST be a JSON list of objects. Each object has exactly these keys:
   - "Question": string
   - "Answer": string
   - "Type": string
3) "Type" MUST be EXACTLY one of the query type names provided in query_types.json (match spelling/case).
4) Component references:
   - Questions MAY refer to components using natural language descriptions (e.g., "the red workbench", "the tool chest by the door").
   - Questions MUST NOT contain component IDs or the <component_id> format.
   - Answers MUST reference components using the exact format: <component_id>.
     Example: "... the red Craftsman tool chest <component_29> ..."
5) Answers must be grounded strictly in:
   - The provided components JSON (component captions)
   - Clearly visible evidence from the image
   Do NOT invent objects, attributes, or relationships that are not supported.
6) If a question cannot be answered unambiguously using the image and component captions, do NOT generate it.
7) Questions should sound like realistic user queries about this scene, including:
   - Spatial relationships
   - Affordances
8) Questions may involve multi-step reasoning and can be long.
   Answers should be correct, concise, and may include brief reasoning.
9) Prefer diversity:
   - Use multiple query types
   - Avoid near-duplicate questions
   - Avoid repeatedly focusing on the same component(s)

SPATIAL LANGUAGE RULE:
- Use spatial relations only when they are clearly supported by the image or component captions
  (e.g., "on top of", "next to", "mounted on the wall", "under the table").
"""

USER_PROMPT_TEMPLATE = """You are given:
A) An image of a scene.
B) components.json: a list of components; each has:
   - component_id
   - component_caption
C) query_types.json: a list of query types; each has:
   - name
   - description
   - example

TASK:
Generate a set of question–answer pairs about the scene.

OUTPUT FORMAT:
- Output MUST be a JSON list.
- Each list item must have exactly:
  {{
    "Question": string,
    "Answer": string,
    "Type": string
  }}

REQUIREMENTS:
- Produce {num_questions} items.
- Use multiple distinct query types from query_types.json.
- Questions:
  - May reference objects naturally (e.g., "the red workbench", "the cabinet near the window").
  - Must NOT include component IDs or <component_id> tags.
- Answers:
  - MUST include one or more component references using <component_id>.
  - May reference the same object using natural language plus its component tag.
- If multiple components satisfy a question, reference all relevant <component_id> tags in the Answer.
- Do not include any explanations outside the JSON output.

INPUTS:
components.json:
{components_json}

query_types.json:
{query_types_json}
"""
