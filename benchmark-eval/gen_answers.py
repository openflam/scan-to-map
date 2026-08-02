import json
import re
import argparse
import requests
from pathlib import Path
import concurrent.futures


CONFIGS = {
    "no_tools": [],
    "search_only": ["search_terms"],
    "search_distance": ["search_terms", "get_distance"],
    "search_distance_around": [
        "search_terms",
        "get_distance",
        "search_around_component",
    ],
    "search_dist_around_image": [
        "search_terms",
        "get_distance",
        "search_around_component",
        "get_images",
    ],
    "search_dist_around_image_exec": [
        "search_terms",
        "get_distance",
        "search_around_component",
        "get_images",
        "execute_python",
    ],
    "only_exec": ["execute_python"],
}

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


def get_answer_from_server(
    dataset_name,
    question,
    tools=None,
    model="gpt-5.4",
    request_timeout=600.0,
    fail_on_error=False,
):
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
        model: Optional model string. Defaults to "gpt-5.4"
    """
    url = "http://localhost:5000/search_stream"
    payload = {
        "dataset_name": dataset_name,
        "query": [{"type": "text", "value": question}],
        "model_name": model,
    }
    if tools is not None:
        payload["tools"] = tools
    
    predicted_answer_raw = ""
    predicted_components_array = []
    received_result = False
    server_error = None
    
    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=request_timeout,
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        event_data = json.loads(data_str)
                        if event_data.get("type") == "result":
                            received_result = True
                            result_data = event_data.get("data", {})
                            predicted_answer_raw = result_data.get("reason", "")
                            components_returned = result_data.get("components", [])
                            # Extract component IDs returned by server
                            predicted_components_array = [str(c.get("component_id")) for c in components_returned]
                            break
                        elif event_data.get("type") == "error":
                            server_error = event_data.get("error", "Unknown server error")
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        if fail_on_error:
            raise RuntimeError(
                f"Query failed for dataset {dataset_name}: {exc}"
            ) from exc
        print(f"Error querying server: {exc}")

    if not received_result:
        error_message = server_error or "Stream ended without a result event."
        if fail_on_error:
            raise RuntimeError(
                f"Query failed for dataset {dataset_name}: {error_message}"
            )
        if server_error:
            print(f"Server returned error: {server_error}")
    
    # The server might have returned reasoning text with tags if the LLM output tags.
    # Alternatively we rely on the component IDs returned explicitly in the payload.
    # Let's extract any possible IDs from the text as well, and merge them.
    text_components = re.findall(r'<component_(\d+)>', predicted_answer_raw)
    predicted_components_combined = sorted(
        set(predicted_components_array + text_components), key=int
    )
    
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
        "--data_dir",
        default=None,
        help=(
            "Directory containing benchmark JSON files. Defaults to benchmark/data. "
            "Cannot be combined with --files."
        ),
    )
    parser.add_argument(
        "--out_dir", 
        default=None, 
        help="Directory to save the JSON results. Defaults to benchmark/results in repo root."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=list(CONFIGS),
        help="Tool configurations to evaluate. Defaults to all configurations.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Model passed to the search server. Defaults to gpt-5.4.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds. Defaults to 600.",
    )
    parser.add_argument(
        "--fail_on_existing",
        action="store_true",
        help=(
            "Fail before querying if any intended result file already exists. "
            "Useful for isolated experiments that must not mix runs."
        ),
    )
    parser.add_argument(
        "--fail_on_query_error",
        action="store_true",
        help="Abort instead of saving an empty answer when a server query fails.",
    )
    args = parser.parse_args()

    if args.files and args.data_dir:
        parser.error("--files and --data_dir cannot be used together")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request_timeout must be positive")

    base_dir = Path(__file__).parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine files to process
    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        data_dir = (
            Path(args.data_dir)
            if args.data_dir
            else base_dir / "benchmark" / "data"
        )
        json_files = sorted(data_dir.glob("*.json"))
        
    if not json_files:
        print("No JSON files found to evaluate.")
        return

    work_items = []
    for config_name in args.configs:
        tools = CONFIGS[config_name]
        config_out_dir = out_dir / config_name
        for file_path in json_files:
            work_items.append((config_name, tools, file_path, config_out_dir))

    if args.fail_on_existing:
        existing_paths = [
            config_out_dir / f"{file_path.stem}_result.json"
            for _, _, file_path, config_out_dir in work_items
            if (config_out_dir / f"{file_path.stem}_result.json").exists()
        ]
        if existing_paths:
            preview = "\n".join(str(path) for path in existing_paths[:10])
            raise FileExistsError(
                "Refusing to mix this run with existing results. Existing files:\n"
                f"{preview}"
            )

    for config_name in args.configs:
        (out_dir / config_name).mkdir(parents=True, exist_ok=True)

    def process_item(item):
        config_name, tools, file_path, config_out_dir = item
        indiv_out_path = config_out_dir / f"{file_path.stem}_result.json"
        if indiv_out_path.exists():
            print(f"Skipping {file_path.name} [{config_name}] as it is already evaluated.")
            return True

        print(f"--- Evaluating {file_path.name} [{config_name}] ---")
        dataset_name, question, exp_text, exp_comp, benchmark_type = get_question_answer_from_json(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            benchmark_record = json.load(f)
        
        if not dataset_name:
            print(f"Skipping {file_path.name} as it lacks a dataset_name.")
            return False
            
        pred_text, pred_comp = get_answer_from_server(
            dataset_name,
            question,
            tools=tools,
            model=args.model,
            request_timeout=args.request_timeout,
            fail_on_error=args.fail_on_query_error,
        )
        
        result_dict = {
            "file": file_path.name,
            "dataset_name": dataset_name,
            "config": config_name,
            "tools": tools,
            "model": args.model,
            "benchmark_type": benchmark_type,
            "question": question,
            "expected_text": exp_text,
            "predicted_text": pred_text,
            "expected_components": exp_comp,
            "predicted_components": pred_comp
        }

        if "sensitivity" in benchmark_record:
            result_dict["sensitivity"] = benchmark_record["sensitivity"]
        
        # Save individual result
        with open(indiv_out_path, 'x', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4)
            
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_item, work_items))

    evaluations_successful = any(results)

    if not evaluations_successful:
        print("No evaluations were successful across any configuration.")
        return

if __name__ == "__main__":
    main()
