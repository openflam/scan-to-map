import os
from openai import OpenAI


def process_query(query: str, component_captions: dict) -> dict:
    """
    Process a search query and return the bounding boxes of the most relevant components.

    Args:
        query: The search query string
        component_captions: Dictionary keyed by component ID with captions and bbox stored

    Returns:
        A dictionary with "bbox" (list of bounding boxes) and "reason" (explanation for the choice)
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Build a prompt with all the component captions
    components_text = ""
    for component_id, component_data in component_captions.items():
        caption = component_data.get("caption", "")
        components_text += f"Component {component_id}: {caption}\n\n"

    prompt = f"""Given the following search query and a list of object descriptions, determine which component IDs best match the query. 

Only return those components that contain the objects almost exclusively without unrelated objects.

Search Query: "{query}"

Available Components:
{components_text}

Respond with ONLY a JSON object containing:
1. "component_ids": a comma-separated string of integer IDs of ALL components that match (e.g., "2,5,7" or "3" for a single match)
2. "reason": a brief one-sentence explanation of why these components match the query

Example response format:
{{"component_ids": "2,5,7", "reason": "These components contain printers which match the search query."}}

If multiple components match, include all of them. If no component matches well, choose the component that is closest to the query."""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that matches search queries to object descriptions. Always respond with a valid JSON object containing component_ids (comma-separated string of integers) and reason (string).",
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

        component_ids_str = parsed_result.get("component_ids", "")
        reason = parsed_result.get("reason", "Selected based on query match.")

        # Parse comma-separated component IDs
        try:
            component_ids = [
                int(id.strip()) for id in component_ids_str.split(",") if id.strip()
            ]
        except (ValueError, AttributeError):
            print(
                f"Warning: Could not parse component IDs from '{component_ids_str}'. Using first component."
            )
            component_ids = [list(component_captions.keys())[0]]
            reason = "Fallback: Using first available component due to parsing error."

        # Validate that all component IDs exist and collect their bounding boxes
        valid_bboxes = []
        invalid_ids = []

        for component_id in component_ids:
            if component_id in component_captions:
                valid_bboxes.append(component_captions[component_id]["bbox"])
            else:
                invalid_ids.append(component_id)

        # If no valid bboxes found, use first component as fallback
        if not valid_bboxes:
            print(f"Warning: No valid component IDs found. Using first component.")
            first_component_id = list(component_captions.keys())[0]
            valid_bboxes = [component_captions[first_component_id]["bbox"]]
            reason = "Fallback: Using first available component due to invalid IDs."
        elif invalid_ids:
            print(f"Warning: Some invalid component IDs were ignored: {invalid_ids}")

        # Return the list of bounding boxes and reason
        return {"bbox": valid_bboxes, "reason": reason}

    except Exception as e:
        print(f"Error processing query with OpenAI: {e}")
        # Fallback: return the bounding box of the first component
        first_component_id = list(component_captions.keys())[0]
        return {
            "bbox": [component_captions[first_component_id]["bbox"]],
            "reason": "Error occurred during search. Showing first available component.",
        }
