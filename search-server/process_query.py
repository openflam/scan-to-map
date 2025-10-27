import os
from openai import OpenAI


def process_query(query: str, component_captions: dict) -> dict:
    """
    Process a search query and return the bounding box of the most relevant component.

    Args:
        query: The search query string
        component_captions: Dictionary keyed by component ID with captions and bbox stored

    Returns:
        A dictionary with "bbox" (the bounding box) and "reason" (explanation for the choice)
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Build a prompt with all the component captions
    components_text = ""
    for component_id, component_data in component_captions.items():
        caption = component_data.get("caption", "")
        components_text += f"Component {component_id}: {caption}\n\n"

    prompt = f"""Given the following search query and a list of object descriptions, determine which component ID best matches the query.

Search Query: "{query}"

Available Components:
{components_text}

Respond with ONLY a JSON object containing:
1. "component_id": the integer ID of the component that best matches
2. "reason": a brief one-sentence explanation of why this component matches the query

Example response format:
{{"component_id": 5, "reason": "This component contains a printer which matches the search query."}}

If no component matches well, choose the component that is closest to the query."""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that matches search queries to object descriptions. Always respond with a valid JSON object containing component_id (integer) and reason (string).",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )

        # Extract the response and parse JSON
        result = response.choices[0].message.content.strip()
        import json

        parsed_result = json.loads(result)

        component_id = parsed_result.get("component_id")
        reason = parsed_result.get("reason", "Selected based on query match.")

        # Validate that the component ID exists
        if component_id not in component_captions:
            print(
                f"Warning: OpenAI returned invalid component ID {component_id}. Using first component."
            )
            component_id = list(component_captions.keys())[0]
            reason = "Fallback: Using first available component due to invalid ID."

        # Return the bounding box and reason
        return {"bbox": component_captions[component_id]["bbox"], "reason": reason}

    except Exception as e:
        print(f"Error processing query with OpenAI: {e}")
        # Fallback: return the bounding box of the first component
        first_component_id = list(component_captions.keys())[0]
        return {
            "bbox": component_captions[first_component_id]["bbox"],
            "reason": "Error occurred during search. Showing first available component.",
        }
