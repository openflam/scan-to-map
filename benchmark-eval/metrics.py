import os
import json
import string
import argparse
from pathlib import Path
import numpy as np

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

def normalize_text(text):
    """Simple text normalization: lowercase and remove punctuation."""
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

def evaluate_answer(expected_text, expected_compo, predicted_text, predicted_compo, question, comet_metric=None):
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
            comet_result = comet_metric.compute(sources=[question], predictions=[predicted_text], references=[[expected_text]])
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

def aggregate_from_files(results_dir_path, metrics_dir_path, comet_metric=None):
    """
    Reads all individual result JSON files from results_dir_path and its subdirectories, 
    computes missing metrics, saves them to metrics_dir_path, and calculates aggregate metrics.
    """
    results_dir = Path(results_dir_path)
    metrics_dir = Path(metrics_dir_path)
    
    dirs_to_process = [results_dir] + [d for d in results_dir.iterdir() if d.is_dir()]
    
    for d in dirs_to_process:
        results = []
        
        if d == results_dir:
            out_d = metrics_dir
        else:
            out_d = metrics_dir / d.name
            
        out_d.mkdir(parents=True, exist_ok=True)
        
        for result_file in d.glob("*_result.json"):
            with open(result_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Failed to read {result_file.name}")
                    continue
            
            # Calculate metrics
            exp_text = data.get("expected_text", "")
            exp_comp = data.get("expected_components", [])
            pred_text = data.get("predicted_text", "")
            pred_comp = data.get("predicted_components", [])
            question = data.get("question", "")
            
            metrics = evaluate_answer(exp_text, exp_comp, pred_text, pred_comp, question, comet_metric)
            data["metrics"] = metrics
            
            # Save metrics to independent file in metrics directory
            out_result_file = out_d / result_file.name
            with open(out_result_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            results.append(data)
                    
        if not results:
            continue
            
        aggregate_data = aggregate_results(results)
        
        agg_out_path = out_d / "aggregate_results.json"
        with open(agg_out_path, 'w', encoding='utf-8') as f:
            json.dump(aggregate_data, f, indent=4)
            
        all_out_path = out_d / "all_results.json"
        with open(all_out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
            
        print(f"Calculated and saved metrics from {len(results)} files to {out_d.absolute()}.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate search server metrics.")
    parser.add_argument(
        "--results_dir", 
        default=None, 
        help="Directory with the JSON results. Defaults to benchmark/results in repo root."
    )
    parser.add_argument(
        "--metrics_dir", 
        default=None, 
        help="Directory to save the JSON metrics. Defaults to benchmark/metrics in repo root."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "benchmark" / "results"
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else base_dir / "benchmark" / "metrics"

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist.")
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

    aggregate_from_files(results_dir, metrics_dir, comet_metric)

if __name__ == "__main__":
    main()
