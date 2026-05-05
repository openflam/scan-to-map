import os
import json
import re
import argparse
from pathlib import Path

# Import the existing functions from gen_answers
from gen_answers import get_question_answer_from_json, get_answer_from_server

MODELS = [
    "gpt-4.1-nano",
]

import concurrent.futures

def main():
    parser = argparse.ArgumentParser(description="Evaluate search server benchmarks across multiple models.")
    parser.add_argument(
        "--files", 
        nargs="+", 
        help="Specific JSON files to evaluate. If not provided, it will evaluate all in benchmark/data/."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "benchmark" / "results_multiple_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        data_dir = base_dir / "benchmark" / "data"
        json_files = list(data_dir.glob("*.json"))
        
    if not json_files:
        print("No JSON files found to evaluate.")
        return

    work_items = []
    for model_name in MODELS:
        # Sanitize model name for directory (e.g. anthropic/claude-3-opus -> anthropic_claude-3-opus)
        safe_model_name = model_name.replace("/", "_")
        model_out_dir = out_dir / safe_model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        for file_path in json_files:
            work_items.append((model_name, file_path, model_out_dir))

    def process_item(item):
        model_name, file_path, model_out_dir = item
        indiv_out_path = model_out_dir / f"{file_path.stem}_result.json"
        if indiv_out_path.exists():
            print(f"Skipping {file_path.name} [{model_name}] as it is already evaluated.")
            return True

        print(f"--- Evaluating {file_path.name} [{model_name}] ---")
        dataset_name, question, exp_text, exp_comp, benchmark_type = get_question_answer_from_json(file_path)
        
        if not dataset_name:
            print(f"Skipping {file_path.name} as it lacks a dataset_name.")
            return False
            
        # Use all tools by default (tools=None)
        pred_text, pred_comp = get_answer_from_server(dataset_name, question, tools=None, model=model_name)
        
        result_dict = {
            "file": file_path.name,
            "model": model_name,
            "tools": None,
            "benchmark_type": benchmark_type,
            "question": question,
            "expected_text": exp_text,
            "predicted_text": pred_text,
            "expected_components": exp_comp,
            "predicted_components": pred_comp
        }
        
        with open(indiv_out_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4)
            
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_item, work_items))

    evaluations_successful = any(results)

    if not evaluations_successful:
        print("No evaluations were successful across any model.")

if __name__ == "__main__":
    main()
