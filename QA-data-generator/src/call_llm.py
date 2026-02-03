"""
LLM caller for QA generation using liteLLM.
Generates question-answer pairs for images using vision-language models.
"""

import base64
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from litellm import completion
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import prompts
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.io_paths import get_all_paths, validate_paths


def load_json_file(file_path: Path) -> Any:
    """Load and parse a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def encode_image_to_base64(image_path: Path) -> str:
    """
    Convert image to base64 encoded data URL for OpenAI API.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded data URL string
    """
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    # Determine MIME type based on file extension
    ext = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(ext, "image/jpeg")

    return f"data:{mime_type};base64,{encoded}"


def call_llm(
    dataset_name: str,
    image_name: str,
    model: str = "gpt-4o",
    num_questions: int = 10,
) -> List[Dict[str, str]]:
    """
    Call LLM to generate question-answer pairs for a given image.

    Args:
        dataset_name: Name of the dataset (e.g., 'ProjectLabStudio_NoNeg')
        image_name: Name of the image without extension (e.g., 'frame_00001')
        model: LLM model to use (default: 'gpt-4o')
        num_questions: Number of Q&A pairs to generate (default: 10)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Maximum tokens in response (default: 4000)

    Returns:
        List of dictionaries with 'Question', 'Answer', and 'Type' keys

    Raises:
        FileNotFoundError: If required files don't exist
        json.JSONDecodeError: If LLM response is not valid JSON
    """
    # Get all required paths
    paths = get_all_paths(dataset_name, image_name)

    # Validate that all files exist
    validate_paths(paths)

    # Load components and query types JSON
    components_json = load_json_file(paths["components"])
    query_types_json = load_json_file(paths["query_types"])

    # Prepare the user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        components_json=json.dumps(components_json, indent=2),
        query_types_json=json.dumps(query_types_json, indent=2),
        num_questions=num_questions,
    )

    # Encode image to base64 for OpenAI API
    image_data_url = encode_image_to_base64(paths["image"])

    # Prepare messages for liteLLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    print(f"Calling LLM model: {model}")
    print(f"Generating {num_questions} Q&A pairs for {dataset_name}/{image_name}")

    # Call liteLLM
    response = completion(model=model, messages=messages)

    # Extract the response content
    response_content = response.choices[0].message.content

    # Parse JSON response
    try:
        qa_pairs = json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Response content:\n{response_content}")
        raise

    # Validate response structure
    if not isinstance(qa_pairs, list):
        raise ValueError(f"Expected list of Q&A pairs, got {type(qa_pairs)}")

    for i, item in enumerate(qa_pairs):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dict: {item}")
        required_keys = {"Question", "Answer", "Type"}
        if not required_keys.issubset(item.keys()):
            raise ValueError(
                f"Item {i} missing required keys. Has {item.keys()}, needs {required_keys}"
            )

    print(f"Successfully generated {len(qa_pairs)} Q&A pairs")

    # Save QA pairs to output file
    output_path = paths["qa_output"]
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Saved QA pairs to: {output_path}")

    return qa_pairs


def main():
    """Example usage of call_llm function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Q&A pairs using LLM")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument(
        "image_name", type=str, help="Name of the image (without extension)"
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument(
        "--num-questions", type=int, default=10, help="Number of Q&A pairs to generate"
    )

    args = parser.parse_args()

    try:
        call_llm(
            dataset_name=args.dataset_name,
            image_name=args.image_name,
            model=args.model,
            num_questions=args.num_questions,
        )

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
