import os
import json
import re
import argparse
import glob
import requests
import string
from collections import defaultdict
from pathlib import Path
import numpy as np

# Try to import evaluation libraries
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import evaluate
    from pycocoevalcap.cider.cider import Cider
    
    # Ensure necessary NLTK data is downloaded
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("Warning: Please install requirements using `pip install -r requirements.txt`")


def get_question_answer_from_json(json_path):
    """
    Reads JSON file.
    Returns dataset_name, question, expected plain text answer, and list of components.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset_name = data.get("dataset_name", "")
    question = data.get("question", "")
    expected_answer_raw = data.get("expected_answer", "")
    
    # Extract components: assume format is <component_ID>text</component_ID>
    # We will extract just the IDs to serve as our list of components.
    # E.g. <component_146>sink</component_146> -> "146"
    components = re.findall(r'<component_(\d+)>', expected_answer_raw)
    
    # Strip all XML-like tags to get plain text
    expected_answer_text = re.sub(r'<[^>]+>', '', expected_answer_raw)
    
    return dataset_name, question, expected_answer_text, components


def get_answer_from_server(dataset_name, question):
    """
    Passes the question to the local search server's stream endpoint.
    Returns plain text predicted answer and a list of predicted components.
    """
    url = "http://localhost:5000/search_stream"
    payload = {
        "dataset_name": dataset_name,
        "query": [{"type": "text", "value": question}],
        "method": "gpt-5.4-tools"
    }
    
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


def normalize_text(text):
    """Simple text normalization: lowercase and remove punctuation."""
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))


def evaluate_answer(expected_text, expected_compo, predicted_text, predicted_compo, comet_metric=None):
    """
    Takes plain text and list of components for expected and predicted.
    Returns dictionary with evaluation metrics.
    """
    metrics = {}
    
    # Component Metrics (Sets)
    exp_set = set(expected_compo)
    pred_set = set(predicted_compo)
    
    true_positives = len(exp_set.intersection(pred_set))
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(exp_set) if len(exp_set) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1 Score'] = f1_score
    
    # Text Metrics Prep
    exp_norm = normalize_text(expected_text)
    pred_norm = normalize_text(predicted_text)
    
    metrics['EM'] = 1.0 if exp_norm == pred_norm and len(exp_norm) > 0 else 0.0
    
    # NLTK Tokenization for BLEU and METEOR
    try:
        from nltk.tokenize import word_tokenize
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        exp_tokens = word_tokenize(expected_text)
        pred_tokens = word_tokenize(predicted_text)
    except Exception:
        # Fallback simple split if punkt is missing
        exp_tokens = expected_text.split()
        pred_tokens = predicted_text.split()
    
    try:
        cc = SmoothingFunction().method1
        metrics['BLEU-1'] = sentence_bleu([exp_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=cc)
        metrics['BLEU-2'] = sentence_bleu([exp_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=cc)
        metrics['BLEU-3'] = sentence_bleu([exp_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=cc)
        metrics['BLEU-4'] = sentence_bleu([exp_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc)
        metrics['METEOR'] = meteor_score([exp_tokens], pred_tokens)
    except Exception as e:
        print(f"Error computing NLTK metrics: {e}")
        metrics['BLEU-1'] = metrics['BLEU-2'] = metrics['BLEU-3'] = metrics['BLEU-4'] = metrics['METEOR'] = 0.0
        
    # COMET
    if comet_metric is not None and expected_text:
        try:
            comet_result = comet_metric.compute(predictions=[predicted_text], references=[[expected_text]])
            metrics['COMET'] = comet_result['scores'][0]
        except Exception as e:
            print(f"Error computing COMET: {e}")
            metrics['COMET'] = 0.0
    else:
        metrics['COMET'] = 0.0

    # CIDEr
    # Pycocoevalcap usually processes a corpus directly. For a single string approximation:
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        # format: dict mapping id to list of strings
        res = {1: [predicted_text]}
        gts = {1: [expected_text]}
        score, _ = cider_scorer.compute_score(gts, res)
        metrics['CIDER'] = score
    except Exception as e:
        print(f"Error computing CIDEr: {e}")
        metrics['CIDER'] = 0.0
        
    return metrics


def aggregate_results(results_list):
    """
    Takes a list of individual result dictionaries and returns an aggregated summary.
    """
    if not results_list:
        return {}
        
    aggregate_metrics = {}
    metric_keys = results_list[0]["metrics"].keys()
    
    for key in metric_keys:
        values = [r["metrics"].get(key, 0.0) for r in results_list]
        aggregate_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "25th_percentile": float(np.percentile(values, 25)),
            "75th_percentile": float(np.percentile(values, 75))
        }

    return {
        "num_questions_evaluated": len(results_list),
        "aggregate_metrics": aggregate_metrics
    }


def aggregate_from_files(out_dir):
    """
    Reads all individual result JSON files from out_dir, computes aggregate metrics,
    and returns nothing.
    """
    out_path = Path(out_dir)
    results = []
    
    # Read all files ending with _result.json
    for result_file in out_path.glob("*_result.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                results.append(data)
            except json.JSONDecodeError:
                print(f"Failed to read {result_file.name}")
                
    if not results:
        print(f"No result files found in {out_path.absolute()}")
        return
        
    aggregate_data = aggregate_results(results)
    
    agg_out_path = out_path / "aggregate_results.json"
    with open(agg_out_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate_data, f, indent=4)
        
    all_out_path = out_path / "all_results.json"
    with open(all_out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Calculated aggregates from {len(results)} files.")
    print(f"Saved aggregate results to {out_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate search server benchmarks.")
    parser.add_argument(
        "--files", 
        nargs="+", 
        help="Specific JSON files to evaluate. If not provided, it will evaluate all in benchmark_data/."
    )
    parser.add_argument(
        "--out_dir", 
        default=None, 
        help="Directory to save the JSON results. Defaults to benchmark_results in repo root."
    )
    parser.add_argument(
        "--aggregate-only", 
        action="store_true",
        help="If set, skip evaluation and just aggregate existing result files in out_dir."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_from_files(out_dir)
        return

    # Determine files to process
    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        data_dir = base_dir / "benchmark_data"
        json_files = list(data_dir.glob("*.json"))
        
    if not json_files:
        print("No JSON files found to evaluate.")
        return

    # Load COMET metric once to save time
    comet_metric = None
    try:
        import evaluate
        print("Loading COMET model (this could take a while on first run)...")
        # Load a standard comet model or a fast implementation
        comet_metric = evaluate.load("comet")
    except Exception as e:
        print(f"Could not load COMET evaluator. COMET scores will be 0. Error: {e}")

    results = []

    for file_path in json_files:
        print(f"--- Evaluating {file_path.name} ---")
        dataset_name, question, exp_text, exp_comp = get_question_answer_from_json(file_path)
        
        if not dataset_name:
            print(f"Skipping {file_path.name} as it lacks a dataset_name.")
            continue
            
        print(f"Question: {question}")
        pred_text, pred_comp = get_answer_from_server(dataset_name, question)
        
        print(f"Expected Text: {exp_text}")
        print(f"Predicted Text: {pred_text}")
        print(f"Expected Comp: {exp_comp}")
        print(f"Predicted Comp: {pred_comp}")
        
        metrics = evaluate_answer(exp_text, exp_comp, pred_text, pred_comp, comet_metric)
        print("Metrics:", metrics)
        
        result_dict = {
            "file": file_path.name,
            "question": question,
            "expected_text": exp_text,
            "predicted_text": pred_text,
            "expected_components": exp_comp,
            "predicted_components": pred_comp,
            "metrics": metrics
        }
        results.append(result_dict)

        # Save individual result
        indiv_out_path = out_dir / f"{file_path.stem}_result.json"
        with open(indiv_out_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4)

    if not results:
        print("No evaluations were successful.")
        return

    aggregate_from_files(out_dir)

if __name__ == "__main__":
    main()
