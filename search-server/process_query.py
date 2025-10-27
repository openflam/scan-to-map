import os
from openai import OpenAI


def process_query(query: str, component_captions: dict) -> dict:
    """
    Process a search query and return the bounding box of the most relevant component.

    Args:
        query: The search query string
        component_captions: Dictionary keyed by component ID with captions and bbox stored

    Returns:
        The bounding box dict of the component that best matches the query
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

Respond with ONLY the component ID number (just the integer) that best matches the search query. If no component matches well, respond with the component ID that is closest to the query."""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that matches search queries to object descriptions. Always respond with only a single integer representing the component ID.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        # Extract the component ID from the response
        result = response.choices[0].message.content.strip()
        component_id = int(result)

        # Validate that the component ID exists
        if component_id not in component_captions:
            print(
                f"Warning: OpenAI returned invalid component ID {component_id}. Using first component."
            )
            component_id = list(component_captions.keys())[0]

        # Return the bounding box from the component data
        return component_captions[component_id]["bbox"]

    except Exception as e:
        print(f"Error processing query with OpenAI: {e}")
        # Fallback: return the bounding box of the first component
        first_component_id = list(component_captions.keys())[0]
        return component_captions[first_component_id]["bbox"]
