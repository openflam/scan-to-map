import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi
from collections import defaultdict
import csv

def get_metrics(filepath):
    """
    Takes a file and gets all the metrics from this file.
    Returns: a tuple containing the metrics dictionary, benchmark_type, and ablation_category.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    benchmark_type = data.get("benchmark_type", "Unknown")
    ablation_category = data.get("config", "Unknown")
    
    return metrics, benchmark_type, ablation_category

def get_spider_plot(metric_name, data_array):
    """
    Generates a spider plot for a given metric.
    
    Args:
        metric_name (str): The name of the metric (e.g. "METEOR")
        data_array (list): Array of dicts with keys "benchmark_type", "metric_value", "ablation_category"
    """
    if not data_array:
        return
        
    # Extract unique benchmark types and ablation categories
    benchmark_types = sorted(list(set(item["benchmark_type"] for item in data_array)))
    ablation_categories = sorted(list(set(item["ablation_category"] for item in data_array)))
    
    if not benchmark_types:
        return
        
    N = len(benchmark_types)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], benchmark_types, color='black', size=11)
    ax.set_rlabel_position(0)
    
    # Determine the y limits based on max value
    max_val = max(item["metric_value"] for item in data_array) if data_array else 1.0
    upper_bound = min(1.0, max(0.2, (int(max_val * 10) + 1) / 10.0))
    ticks = np.linspace(0, upper_bound, 5)[1:]
    plt.yticks(ticks, [f"{t:.2f}" for t in ticks], color="grey", size=8)
    plt.ylim(0, upper_bound)
    
    # Organize data by ablation category and plot
    for category in ablation_categories:
        # Create a mapping for quick lookup
        val_map = {item["benchmark_type"]: item["metric_value"] for item in data_array if item["ablation_category"] == category}
        values = [val_map.get(bt, 0.0) for bt in benchmark_types]
        values_closed = values + values[:1]
        
        ax.plot(angles, values_closed, linewidth=2, linestyle='solid', label=category)
        ax.fill(angles, values_closed, alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'{metric_name} by Benchmark Type', size=15, y=1.1)
    
    out_dir = Path(__file__).parent.parent / "benchmark" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric_name.replace(' ', '_')}_spider.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {metric_name} plot to {out_path}")

def save_csv_table(metric_name, data_array):
    """
    Saves a CSV table for a given metric where rows are ablation categories, 
    columns are benchmark_type, and an additional column aggregates across all types.
    """
    if not data_array:
        return
        
    benchmark_types = sorted(list(set(item["benchmark_type"] for item in data_array)))
    ablation_categories = sorted(list(set(item["ablation_category"] for item in data_array)))
    
    out_dir = Path(__file__).parent.parent / "benchmark" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric_name.replace(' ', '_')}_table.csv"
    
    with open(out_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header
        header = ["Ablation Category"] + benchmark_types + ["Aggregate Mean"]
        writer.writerow(header)
        
        # Write rows
        for category in ablation_categories:
            vals = []
            for bt in benchmark_types:
                # Find matching value
                val = next((item["metric_value"] for item in data_array 
                            if item["ablation_category"] == category and item["benchmark_type"] == bt), 0.0)
                vals.append(val)
                
            agg_mean = sum(vals) / len(vals) if vals else 0.0
            
            # Format row
            row = [category] + [f"{v:.4f}" for v in vals] + [f"{agg_mean:.4f}"]
            writer.writerow(row)
            
    print(f"Saved {metric_name} tabl to {out_path}")

def main():
    metrics_dir = Path(__file__).parent.parent / "benchmark" / "metrics"
    
    if not metrics_dir.exists():
        print(f"Metrics directory not found at {metrics_dir.absolute()}")
        return
        
    # Dictionary to collect all metric values before averaging
    # Structure: raw_data[metric][category][bench_type] = [list of values]
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Iterate through all individual metric JSON files
    for metric_file in metrics_dir.rglob("*_result.json"):
        try:
            metrics, benchmark_type, ablation_category = get_metrics(metric_file)
            for metric_name, value in metrics.items():
                raw_data[metric_name][ablation_category][benchmark_type].append(value)
        except Exception as e:
            print(f"Error reading {metric_file}: {e}")
            
    # Process the raw data to create an array of dicts for each metric
    for metric_name, category_dict in raw_data.items():
        data_array = []
        for category, bench_dict in category_dict.items():
            for bench_type, values in bench_dict.items():
                # Compute mean for this category and benchmark type
                mean_val = sum(values) / len(values) if values else 0.0
                data_array.append({
                    "benchmark_type": bench_type,
                    "metric_value": mean_val,
                    "ablation_category": category
                })
                
        # Generate plot and table for this metric
        if data_array:
            get_spider_plot(metric_name, data_array)
            save_csv_table(metric_name, data_array)

if __name__ == "__main__":
    main()
