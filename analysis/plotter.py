import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_pareto_effects(results_df, output_dir):
    """
    Plots a Pareto Chart of Standardized Effects (F-Statistics).
    Shows which factors are most important for each metric.
    """
    print("Generating Pareto Charts...")
    
    for block in results_df["Block"].unique():
        block_df = results_df[results_df["Block"] == block]
        
        for metric in block_df["Metric"].unique():
            subset = block_df[block_df["Metric"] == metric].copy()
            # Sort by F-Stat descending
            subset = subset.sort_values("F_Stat", ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=subset, x="F_Stat", y="Source", palette="viridis")
            
            # Add a red line for a rough significance threshold
            plt.axvline(x=4.0, color='red', linestyle='--', label='Approx Sig Threshold')
            
            plt.title(f"Pareto Chart of Effects (F-Stat): {metric} [{block}]")
            plt.xlabel("F-Statistic (Magnitude of Effect)")
            plt.ylabel("Factor/Interaction")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/Pareto_{block}_{metric}.png")
            plt.close()

def plot_significance_heatmap(results_df, output_dir):
    """
    Plots a heatmap of P-values.
    Rows = Factors, Columns = Metrics.
    Color = P-value (Dark/Red = Significant, Light/Blue = Not Significant).
    """
    print("Generating Significance Heatmaps...")
    
    for block in results_df["Block"].unique():
        block_df = results_df[results_df["Block"] == block]
        
        pivot_df = block_df.pivot(index="Source", columns="Metric", values="p_value")
        pivot_df = pivot_df.fillna(1.0) 
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="Reds_r", vmin=0, vmax=0.1, linewidths=.5)
        
        plt.title(f"Significance Heatmap (P-values) [{block}]")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Heatmap_Pvalues_{block}.png")
        plt.close()

def plot_interactions(data_file: str, output_dir: str):
    print("Generating Interaction Plots...")
    if not os.path.exists(data_file): return
    df = pd.read_csv(data_file)
    sns.set_theme(style="whitegrid")
    metrics = ["p95_ms", "Throughput"] 

    for block in df["Block"].unique():
        block_df = df[df["Block"] == block]
        for metric in metrics:
            if metric not in block_df.columns: continue
            
            # Model x Quant
            plt.figure(figsize=(8, 5))
            try:
                sns.pointplot(data=block_df, x="Quant", y=metric, hue="Model", errorbar="se", capsize=0.1)
                plt.title(f"{block}: {metric} (Model x Quant)")
                plt.savefig(f"{output_dir}/Interact_{block}_{metric}_ModQuant.png")
                plt.close()
            except: plt.close()

            # Model x Conc
            plt.figure(figsize=(8, 5))
            try:
                sns.pointplot(data=block_df, x="Conc", y=metric, hue="Model", errorbar="se", capsize=0.1)
                plt.title(f"{block}: {metric} (Model x Conc)")
                plt.savefig(f"{output_dir}/Interact_{block}_{metric}_ModConc.png")
                plt.close()
            except: plt.close()

def main():
    if not os.path.exists("plots"): os.makedirs("plots")
    input_file1 = "mock_data.csv"
    input_file2 = "analysis_results.csv"
    plot_interactions(input_file1, "plots")
    
    if os.path.exists(input_file2):
        results = pd.read_csv(input_file2)
        plot_pareto_effects(results, "plots")
        plot_significance_heatmap(results, "plots")
    else:
        print("File not found")

if __name__ == "__main__":
    main()