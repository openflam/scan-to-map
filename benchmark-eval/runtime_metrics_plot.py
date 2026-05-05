import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# NeurIPS ready rcParams
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

BAR_TEXT_FONT_SIZE = 12
FIG_SIZE = (9,3)
DATASETS = ["MITBoiler_NoNeg_copy"]

def read_runtime_metrics():
    """
    Reads runtime metrics for all datasets and returns the average duration for each step.
    Maintains the order of steps as they appear in the JSON file.
    """
    base_dir = Path(__file__).parent.parent / "outputs"
    
    step_durations = {}
    step_order = []
    
    for dataset in DATASETS:
        json_path = base_dir / dataset / "runtime_stats.json"
        if not json_path.exists():
            print(f"Warning: {json_path} does not exist. Skipping.")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        steps = data.get("steps", {})
        
        # Maintain order based on the first dataset encountered
        if not step_order:
            step_order = list(steps.keys())
            
        for step_name, step_data in steps.items():
            if step_name not in step_durations:
                step_durations[step_name] = []
            
            duration = step_data.get("duration_seconds", 0.0)
            step_durations[step_name].append(duration)
            
    # Calculate averages
    avg_durations = {}
    for step in step_order:
        if step in step_durations and step_durations[step]:
            avg_durations[step] = sum(step_durations[step]) / len(step_durations[step])
        else:
            avg_durations[step] = 0.0
            
    return step_order, avg_durations

def plot_runtime_stats(step_order, avg_durations):
    """
    Plots the average runtimes of different sequential stages as a waterfall chart.
    """
    # Clean up step names for plotting
    clean_names = []
    for name in step_order:
        clean_name = name.replace('_', ' ').title()
        if "Sam3" in clean_name:
            clean_name = clean_name.replace("Sam3", "SAM3")
        if "2D" in clean_name or "3D" in clean_name:
            clean_name = clean_name.replace("2D", "2D").replace("3D", "3D")
        clean_names.append(clean_name)
        
    durations = [avg_durations[name] for name in step_order]
    
    names = []
    vals = []
    for name, dur in zip(clean_names, durations):
        if dur > 0.1:  # Filter out trivial steps to keep it clean
            names.append(name)
            vals.append(dur)
            
    names.append("Total")
    vals.append(sum(vals))
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    x_pos = np.arange(len(names))
    
    # Calculate starting points for each bar
    starts = [0] * len(names)
    current = 0
    for i in range(len(vals) - 1):
        starts[i] = current
        current += vals[i]
    starts[-1] = 0 
    
    # NeurIPS appropriate color scheme (colorblind friendly)
    stage_color = '#56B4E9' 
    total_color = '#E69F00'
    colors = [stage_color] * (len(names) - 1) + [total_color]
    
    bars = ax.bar(x_pos, vals, bottom=starts, align='center', alpha=0.85, color=colors, edgecolor='black', width=0.6)
    
    # Add dashed connecting lines between the bars to show the waterfall effect
    for i in range(len(vals) - 1):
        y_val = starts[i] + vals[i]
        ax.plot([x_pos[i] + 0.3, x_pos[i+1] - 0.3], [y_val, y_val], color='gray', linestyle='--', linewidth=1.5)
                
    # Add text labels on top of the bars
    max_y = sum(vals[:-1])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        text_y = starts[i] + height + (max_y * 0.02)
        ax.text(bar.get_x() + bar.get_width()/2, text_y, 
                f'{height:.1f}s', ha='center', va='bottom', size=BAR_TEXT_FONT_SIZE)
                
    # Wrap x-axis labels
    wrapped_names = [textwrap.fill(n, width=12, break_long_words=False) for n in names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(wrapped_names, rotation=45, ha='right', rotation_mode="anchor")
    
    ax.set_ylabel('Time (seconds)')
    
    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(False) # Remove vertical grid lines for a cleaner look
    
    # Extend y-axis slightly to make room for text labels
    ax.set_ylim(0, max_y * 1.1)
    
    plt.tight_layout()
    
    out_dir = Path(__file__).parent.parent / "benchmark" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runtime_metrics_waterfall.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved runtime waterfall plot to {out_path}")

def main():
    step_order, avg_durations = read_runtime_metrics()
    if step_order:
        plot_runtime_stats(step_order, avg_durations)
    else:
        print("No runtime metrics found for the specified datasets.")

if __name__ == "__main__":
    main()
