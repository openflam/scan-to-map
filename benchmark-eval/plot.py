import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from math import pi
from collections import defaultdict
import csv

plt.rcParams.update({
    'font.size': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
})

CATEGORY_INFO = {
    "no_tools": {"label": "No Tools", "color": "#E69F00"},
    "search_only": {"label": "Search Only", "color": "#56B4E9"},
    "search_distance": {"label": "+Distance", "color": "#009E73"},
    "search_distance_around": {"label": "+Vicinity Search", "color": "#0072B2"},
    "search_dist_around_image": {"label": "+Image", "color": "#D55E00"},
    "search_dist_around_image_exec": {"label": "+Exec", "color": "#CC79A7"}
}

INCLUDED_BENCHMARK_TYPES = [
    "Spatial Relations", 
    "Physics, safety, etc.", 
    "Functionality", 
    "Entity Search", 
    # "Affordance"
]

def get_metrics(filepath):
    """
    Takes a file and gets all the metrics from this file.
    Returns: a tuple containing the metrics dictionary, benchmark_type, and ablation_category.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    benchmark_type = data.get("benchmark_type", "Unknown")
    benchmark_name = data.get("file", "Unknown")
    
    if isinstance(filepath, str):
        filepath = Path(filepath)
    ablation_category = filepath.parent.name
    
    return metrics, benchmark_type, benchmark_name, ablation_category

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
    
    ordered_cats = [c for c in CATEGORY_INFO if c in ablation_categories]
    
    if not benchmark_types:
        return
        
    N = len(benchmark_types)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], benchmark_types, color='black', size=14)
    ax.set_rlabel_position(0)
    
    # Determine the y limits based on max value
    max_val = max(item["metric_value"] for item in data_array) if data_array else 1.0
    upper_bound = max(0.2, (int(max_val * 10) + 1) / 10.0)
    
    ticks = np.linspace(0, upper_bound, 5)[1:]
    plt.yticks(ticks, [])
    plt.ylim(0, upper_bound)
    
    # Organize data by ablation category and plot
    for category in ordered_cats:
        # Create a mapping for quick lookup
        val_map = {item["benchmark_type"]: item["metric_value"] for item in data_array if item["ablation_category"] == category}
        values = [val_map.get(bt, 0.0) for bt in benchmark_types]
        values_closed = values + values[:1]
        
        info = CATEGORY_INFO.get(category, {"label": category, "color": "black"})
        label_name = info["label"]
        color = info["color"]
        ax.plot(angles, values_closed, linewidth=2, linestyle='solid', label=label_name, color=color)
        ax.fill(angles, values_closed, alpha=0.1, color=color)
        
    # plt.title(f'{metric_name} by Benchmark Type', size=15, y=1.1)
    
    out_dir = Path(__file__).parent.parent / "benchmark" / "plots" / "spider_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric_name.replace(' ', '_')}_spider.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {metric_name} plot to {out_path}")

def save_summay_csv_table(metric_name, data_array):
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

def generate_legend(labels, colors):
    """
    Generates standalone vertical and horizontal legends for the bar charts.
    """
    import matplotlib.patches as mpatches
    out_dir = Path(__file__).parent.parent / "benchmark" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    patches = [mpatches.Patch(facecolor=colors[i], edgecolor='black', alpha=0.85, label=labels[i]) for i in range(len(labels))]

    # Vertical legend
    fig_v = plt.figure(figsize=(4, len(labels) * 0.5))
    fig_v.legend(handles=patches, loc='center', frameon=False, ncol=1)
    fig_v.gca().set_axis_off()
    out_path_v = out_dir / "bar_chart_legend_vertical.pdf"
    fig_v.savefig(out_path_v, bbox_inches='tight')
    plt.close(fig_v)
    print(f"Saved vertical legend to {out_path_v}")

    # Horizontal legend
    fig_h = plt.figure(figsize=(len(labels) * 2.5, 1))
    fig_h.legend(handles=patches, loc='center', frameon=False, ncol=len(labels))
    fig_h.gca().set_axis_off()
    out_path_h = out_dir / "bar_chart_legend_horizontal.pdf"
    fig_h.savefig(out_path_h, bbox_inches='tight')
    plt.close(fig_h)
    print(f"Saved horizontal legend to {out_path_h}")

def get_bar_chart(metric_name, category_dict):
    """
    Generates a bar chart for a given metric showing mean and std per ablation category.
    """
    if not category_dict:
        return
        
    cat_values = defaultdict(list)
    for category, bench_dict in category_dict.items():
        for values in bench_dict.values():
            cat_values[category].extend(values)
            
    existing_cats = set(cat_values.keys())
    ordered_cats = [c for c in CATEGORY_INFO if c in existing_cats]
    
    means = []
    cis = []
    labels = []
    colors = []
    
    for c in ordered_cats:
        vals = cat_values[c]
        n = len(vals)
        means.append(np.mean(vals) if n > 0 else 0)
        
        # Calculate 95% confidence interval
        if n > 1:
            ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(n)
        else:
            ci = 0
        cis.append(ci)
        info = CATEGORY_INFO.get(c, {"label": c, "color": "black"})
        labels.append(info["label"])
        colors.append(info["color"])
        
    fig, ax = plt.subplots(figsize=(6, 5))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=cis, align='center', width=0.85, alpha=0.85, ecolor='black', capsize=10, color=colors, edgecolor='black')
    ax.set_ylabel(metric_name)
    ax.set_xticks([])
    # ax.set_title(f'{metric_name} by Ablation Category')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    generate_legend(labels, colors)
    
    out_dir = Path(__file__).parent.parent / "benchmark" / "plots" / "bar_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric_name.replace(' ', '_')}_bar.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {metric_name} bar chart to {out_path}")

def plot_question_counts(benchmark_counts):
    """
    Generates a pie chart showing the proportion of questions available for each benchmark type.
    """
    if not benchmark_counts:
        return
        
    labels = sorted(list(benchmark_counts.keys()))
    counts = [len(benchmark_counts[l]) for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, 
           colors=plt.cm.Pastel1.colors, wedgeprops={'edgecolor': 'black'})
           
    plt.tight_layout()
    
    out_dir = Path(__file__).parent.parent / "benchmark" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "question_counts_pie.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved question counts plot to {out_path}")

def main():
    metrics_dir = Path(__file__).parent.parent / "benchmark" / "metrics"
    
    if not metrics_dir.exists():
        print(f"Metrics directory not found at {metrics_dir.absolute()}")
        return
        
    # Dictionary to collect all metric values before averaging
    # Structure: raw_data[metric][category][bench_type] = [list of values]
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    all_data_rows = []
    benchmark_counts = defaultdict(set)
    
    # Iterate through all individual metric JSON files
    for metric_file in metrics_dir.rglob("*_result.json"):
        try:
            metrics, benchmark_type, benchmark_name, ablation_category = get_metrics(metric_file)
            
            if benchmark_type not in INCLUDED_BENCHMARK_TYPES:
                continue
                
            benchmark_counts[benchmark_type].add(benchmark_name)
            for metric_name, value in metrics.items():
                raw_data[metric_name][ablation_category][benchmark_type].append(value)
                all_data_rows.append([benchmark_name, benchmark_type, ablation_category, metric_name, value])
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
            save_summay_csv_table(metric_name, data_array)
            get_bar_chart(metric_name, category_dict)
            
    if all_data_rows:
        tables_dir = Path(__file__).parent.parent / "benchmark" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        all_data_path = tables_dir / "AllData.csv"
        with open(all_data_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["benchmark_name", "benchmark_type", "ablation_category", "metric_name", "value"])
            for row in all_data_rows:
                writer.writerow(row)
        print(f"Saved concatenated data to {all_data_path}")
        
    if benchmark_counts:
        plot_question_counts(benchmark_counts)

if __name__ == "__main__":
    main()
