import os
import json
import re
import argparse
import glob
import requests
import string
from collections import defaultdict
from pathlib import Path

def get_question_answer_from_json(json_path):
    """
    Reads JSON file.
    Returns dataset_name, question, expected plain text answer, list of components, and benchmark_type.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset_name = data.get("dataset_name", "")
    question = data.get("question", "")
    expected_answer_raw = data.get("expected_answer", "")
    benchmark_type = data.get("benchmark_type", "Unknown")
    
    # Extract components: assume format is <component_ID>text</component_ID>
    # We will extract just the IDs to serve as our list of components.
    # E.g. <component_146>sink</component_146> -> "146"
    components = re.findall(r'<component_(\d+)>', expected_answer_raw)
    
    # Strip all XML-like tags to get plain text
    expected_answer_text = re.sub(r'<[^>]+>', '', expected_answer_raw)
    
    return dataset_name, question, expected_answer_text, components, benchmark_type


def get_answer_from_server(dataset_name, question, tools=None):
    """
    Passes the question to the local search server's stream endpoint.
    Returns plain text predicted answer and a list of predicted components.
    
    Args:
        dataset_name: Name of the dataset (e.g. 'scannetpp_5ee7c22ba0')
        question: The natural language question string
        tools: Optional list of tool names to allow. Possible values:
               - "search_terms"
               - "get_distance"
               - "search_around_component"
               - "get_images"
               If None, all tools are enabled.
    """
    url = "http://localhost:5000/search_stream"
    payload = {
        "dataset_name": dataset_name,
        "query": [{"type": "text", "value": question}],
        "method": "gpt-5.4-tools",
    }
    if tools is not None:
        payload["tools"] = tools
    
    predicted_answer_raw = ""
    predicted_components_array = []
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        event_data = json.loads(data_str)
                        if event_data.get("type") == "result":
                            result_data = event_data.get("data", {})
                            predicted_answer_raw = result_data.get("reason", "")
                            components_returned = result_data.get("components", [])
                            # Extract component IDs returned by server
                            predicted_components_array = [str(c.get("component_id")) for c in components_returned]
                            break
                        elif event_data.get("type") == "error":
                            print(f"Server returned error: {event_data.get('error')}")
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error querying server: {e}")
    
    # The server might have returned reasoning text with tags if the LLM output tags.
    # Alternatively we rely on the component IDs returned explicitly in the payload.
    # Let's extract any possible IDs from the text as well, and merge them.
    text_components = re.findall(r'<component_(\d+)>', predicted_answer_raw)
    predicted_components_combined = list(set(predicted_components_array + text_components))
    
    # Strip all XML-like tags to get plain predicted text
    predicted_answer_text = re.sub(r'<[^>]+>', '', predicted_answer_raw)
    
    return predicted_answer_text, predicted_components_combined


def main():
    parser = argparse.ArgumentParser(description="Evaluate search server benchmarks.")
    parser.add_argument(
        "--files", 
        nargs="+", 
        help="Specific JSON files to evaluate. If not provided, it will evaluate all in benchmark/data/."
    )
    parser.add_argument(
        "--out_dir", 
        default=None, 
        help="Directory to save the JSON results. Defaults to benchmark/results in repo root."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine files to process
    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        data_dir = base_dir / "benchmark" / "data"
        json_files = list(data_dir.glob("*.json"))
        
    if not json_files:
        print("No JSON files found to evaluate.")
        return

    configs = {
        "no_tools": [],
        "search_only": ["search_terms"],
        "search_distance": ["search_terms", "get_distance"],
        "search_distance_around": ["search_terms", "get_distance", "search_around_component"],
        "search_dist_around_image": ["search_terms", "get_distance", "search_around_component", "get_images"],
        "search_dist_around_image_exec": ["search_terms", "get_distance", "search_around_component", "get_images", "execute_python"],
        "only_exec": ["execute_python"],
    }

    evaluations_successful = False

    for config_name, tools in configs.items():
        print(f"\n{'='*50}")
        print(f"Running Configuration: {config_name}")
        print(f"Tools: {tools if tools is not None else 'All Tools'}")
        print(f"{'='*50}\n")
        
        config_out_dir = out_dir / config_name
        config_out_dir.mkdir(parents=True, exist_ok=True)
        
        config_successful = False
        for file_path in json_files:
            indiv_out_path = config_out_dir / f"{file_path.stem}_result.json"
            if indiv_out_path.exists():
                print(f"Skipping {file_path.name} [{config_name}] as it is already evaluated.")
                config_successful = True
                evaluations_successful = True
                continue

            print(f"--- Evaluating {file_path.name} [{config_name}] ---")
            dataset_name, question, exp_text, exp_comp, benchmark_type = get_question_answer_from_json(file_path)
            
            if not dataset_name:
                print(f"Skipping {file_path.name} as it lacks a dataset_name.")
                continue
                
            print(f"Question: {question}")
            pred_text, pred_comp = get_answer_from_server(dataset_name, question, tools=tools)
            
            print(f"Expected Text: {exp_text}")
            print(f"Predicted Text: {pred_text}")
            print(f"Expected Comp: {exp_comp}")
            print(f"Predicted Comp: {pred_comp}")
            
            result_dict = {
                "file": file_path.name,
                "config": config_name,
                "tools": tools,
                "benchmark_type": benchmark_type,
                "question": question,
                "expected_text": exp_text,
                "predicted_text": pred_text,
                "expected_components": exp_comp,
                "predicted_components": pred_comp
            }
            
            # Save individual result
            indiv_out_path = config_out_dir / f"{file_path.stem}_result.json"
            with open(indiv_out_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=4)
                
            config_successful = True
            evaluations_successful = True
            
        if not config_successful:
            print(f"No evaluations were successful for {config_name}.")

    if not evaluations_successful:
        print("No evaluations were successful across any configuration.")
        return

if __name__ == "__main__":
    main()
