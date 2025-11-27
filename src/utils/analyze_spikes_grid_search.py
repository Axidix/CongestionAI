"""
Comprehensive analysis of spike features experiment results.
Analyzes: spike features × loss functions × model configs
NOTE: Since loss functions differ, comparisons are based on METRICS only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


def load_experiment_results(
    results_dir: str = "plots_training_dl/spike_features_experiment",
    results_file: str = "spike_experiment_results.csv"
) -> pd.DataFrame:
    """Load experiment results from CSV."""
    path = Path(results_dir) / results_file
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} experiments from {path}")
    return df


def load_training_curves(
    results_dir: str = "plots_training_dl/spike_features_experiment",
) -> Dict[str, pd.DataFrame]:
    """Load all training loss files."""
    curves = {}
    results_path = Path(results_dir)
    
    for loss_file in results_path.glob("losses_*.txt"):
        exp_name = loss_file.stem.replace("losses_", "")
        try:
            df = pd.read_csv(loss_file)
            curves[exp_name] = df
        except Exception as e:
            print(f"Warning: Could not load {loss_file}: {e}")
    
    print(f"Loaded {len(curves)} training curves")
    return curves


def identify_metric_columns(df: pd.DataFrame) -> Dict[str, list]:
    """Identify available metric columns by category."""
    metrics = {
        "mse": [],      # MSE, RMSE - comparable across all configs
        "mae": [],      # MAE - comparable across all configs
        "spike": [],    # Spike detection: recall, precision, F1
        "delta": [],    # Delta/trend correlation
        "horizon": [],  # Per-horizon metrics
        "other": [],
    }
    
    # Columns that should never be treated as metrics
    exclude_cols = {"spike_features", "loss_type", "exp_name", "history", "error"}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        col_lower = col.lower()
        
        # Skip non-numeric columns
        if df[col].dtype == 'object':
            continue
            
        if any(x in col_lower for x in ["mse", "rmse"]):
            metrics["mse"].append(col)
        elif "mae" in col_lower:
            metrics["mae"].append(col)
        elif any(x in col_lower for x in ["spike", "recall", "precision", "f1"]):
            metrics["spike"].append(col)
        elif "delta" in col_lower or "corr" in col_lower:
            metrics["delta"].append(col)
        elif any(x in col_lower for x in ["_1h", "_2h", "_4h", "_8h", "horizon"]):
            metrics["horizon"].append(col)
    
    return metrics


def safe_format(value, fmt=".4f"):
    """Safely format a value, handling strings and NaNs."""
    try:
        return f"{float(value):{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def load_and_clean_results(csv_path):
    """Load results CSV and ensure proper column types."""
    df = pd.read_csv(csv_path)
    
    # Define which columns should be numeric
    numeric_cols = [
        'n_blocks', 'hidden_dim', 'num_features',
        'final_train_loss', 'final_val_loss', 'best_val_loss',
        'n_true_spikes', 'n_pred_spikes', 'spike_frequency',
        'spike_recall', 'spike_precision', 'spike_f1',
        'spike_delta_corr', 'spike_direction_accuracy',
        'overall_delta_corr', 'Corr'
    ]
    
    # Define which columns should be strings
    string_cols = ['exp_name', 'history', 'spike_features', 'loss_type', 'error']
    
    # Convert numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure string columns are strings
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Remove rows where key metrics are NaN (failed experiments)
    df_valid = df[df['best_val_loss'].notna()].copy()
    
    print(f"Loaded {len(df)} rows, {len(df_valid)} valid experiments")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Dtypes:\n{df.dtypes}")
    
    return df, df_valid


def analyze_spike_experiment(
    results_dir: str = "plots_training_dl/spike_features_experiment",
    output_dir: Optional[str] = None,
):
    """
    Full analysis pipeline.
    
    IMPORTANT: Loss values are NOT compared since different loss functions
    produce incomparable values. All comparisons use evaluation METRICS.
    """
    
    if output_dir is None:
        output_dir = Path(results_dir) / "analysis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_experiment_results(results_dir)
    curves = load_training_curves(results_dir)
    
    # Filter out failed experiments
    df_valid = df[df["error"].isna()].copy() if "error" in df.columns else df.copy()
    print(f"Valid experiments: {len(df_valid)}/{len(df)}")
    
    # Identify available metrics
    metric_cols = identify_metric_columns(df_valid)
    print(f"\nAvailable metrics:")
    for cat, cols in metric_cols.items():
        if cols:
            print(f"  {cat}: {cols}")
    
    # Determine primary comparison metric (prefer RMSE/MSE as universal)
    if metric_cols["mse"]:
        primary_metric = metric_cols["mse"][0]
    elif metric_cols["mae"]:
        primary_metric = metric_cols["mae"][0]
    else:
        # Fallback - but warn user
        print("\n⚠️  WARNING: No universal metrics (MSE/MAE) found!")
        print("   Analysis will be limited to spike-specific metrics.")
        primary_metric = None
    
    print(f"\nPrimary comparison metric: {primary_metric}")
    
    # ================================================================
    # 1. SUMMARY STATISTICS (METRICS ONLY)
    # ================================================================
    print("\n" + "="*80)
    print("1. SUMMARY STATISTICS (Evaluation Metrics Only)")
    print("="*80)
    
    # Collect all numeric metric columns for aggregation
    agg_cols = []
    for cat in ["mse", "mae", "spike", "delta"]:
        agg_cols.extend(metric_cols[cat])
    agg_cols = [c for c in agg_cols if c in df_valid.columns and df_valid[c].notna().any()]
    
    if not agg_cols:
        print("⚠️  No valid metric columns found for aggregation!")
        return df_valid, curves
    
    # By spike features
    print("\n--- By Spike Features ---")
    spike_agg = {col: ["mean", "std", "min"] for col in agg_cols[:4]}  # Limit columns
    spike_stats = df_valid.groupby("spike_features").agg(spike_agg).round(5)
    spike_stats.columns = ["_".join(col).strip() for col in spike_stats.columns]
    print(spike_stats)
    
    # By loss type
    print("\n--- By Loss Type ---")
    loss_stats = df_valid.groupby("loss_type").agg(spike_agg).round(5)
    loss_stats.columns = ["_".join(col).strip() for col in loss_stats.columns]
    print(loss_stats)
    
    # By spike × loss combination
    print("\n--- By Spike Features × Loss Type ---")
    combo_stats = df_valid.groupby(["spike_features", "loss_type"]).agg(
        {col: "mean" for col in agg_cols[:4]}
    ).round(5)
    print(combo_stats)
    
    # ================================================================
    # 2. SPIKE DETECTION METRICS ANALYSIS
    # ================================================================
    print("\n" + "="*80)
    print("2. SPIKE DETECTION METRICS")
    print("="*80)
    
    spike_metrics = metric_cols["spike"] + metric_cols["delta"]
    
    if spike_metrics:
        print(f"\nSpike/Delta metrics: {spike_metrics}")
        
        for metric in spike_metrics:
            if metric in df_valid.columns and df_valid[metric].notna().any():
                # Higher is better for recall/precision/F1/correlation
                best_idx = df_valid[metric].idxmax()
                best = df_valid.loc[best_idx]
                print(f"\n🎯 Best {metric}: {safe_format(best[metric])}")
                print(f"   Config: {best['spike_features']} + {best['loss_type']}")
                if primary_metric and primary_metric in best:
                    print(f"   {primary_metric}: {best[primary_metric]:.5f}")
    else:
        print("No spike detection metrics found.")
    
    # ================================================================
    # 3. VISUALIZATIONS (METRICS-BASED)
    # ================================================================
    print("\n" + "="*80)
    print("3. GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 3.1 Heatmaps for key metrics
    plot_metrics = []
    if primary_metric:
        plot_metrics.append((primary_metric, "RdYlGn_r", "lower"))  # Lower is better
    for m in ["spike_f1", "spike_recall", "overall_delta_corr"]:
        if m in df_valid.columns and df_valid[m].notna().any():
            plot_metrics.append((m, "RdYlGn", "higher"))  # Higher is better
    
    if plot_metrics:
        n_plots = min(len(plot_metrics), 4)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for ax, (metric, cmap, direction) in zip(axes, plot_metrics[:n_plots]):
            pivot = df_valid.pivot_table(
                values=metric, 
                index="spike_features", 
                columns="loss_type", 
                aggfunc="mean"
            )
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap=cmap, ax=ax)
            ax.set_title(f"{metric}\n({'↓ better' if direction == 'lower' else '↑ better'})")
        
        plt.tight_layout()
        plt.savefig(output_dir / "heatmaps_metrics.png", dpi=150)
        plt.close()
        print(f"Saved: heatmaps_metrics.png")
    
    # 3.2 Bar charts for primary metric
    if primary_metric:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # By spike features
        spike_order = df_valid.groupby("spike_features")[primary_metric].mean().sort_values().index
        sns.barplot(data=df_valid, x="spike_features", y=primary_metric, 
                    order=spike_order, ax=axes[0], palette="viridis", errorbar="sd")
        axes[0].set_title(f"{primary_metric} by Spike Features (lower is better)")
        axes[0].tick_params(axis='x', rotation=45)
        
        # By loss type
        loss_order = df_valid.groupby("loss_type")[primary_metric].mean().sort_values().index
        sns.barplot(data=df_valid, x="loss_type", y=primary_metric, 
                    order=loss_order, ax=axes[1], palette="magma", errorbar="sd")
        axes[1].set_title(f"{primary_metric} by Loss Type (lower is better)")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "barplot_primary_metric.png", dpi=150)
        plt.close()
        print(f"Saved: barplot_primary_metric.png")
    
    # 3.3 Spike metrics comparison
    if metric_cols["spike"]:
        fig, axes = plt.subplots(1, len(metric_cols["spike"]), 
                                  figsize=(5*len(metric_cols["spike"]), 5))
        if len(metric_cols["spike"]) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metric_cols["spike"]):
            if df_valid[metric].notna().any():
                # Ensure metric column is numeric before groupby
                df_valid[metric] = pd.to_numeric(df_valid[metric], errors='coerce')
                spike_order = df_valid.groupby("spike_features")[metric].mean().sort_values(ascending=False).index
                sns.barplot(data=df_valid, x="spike_features", y=metric,
                           order=spike_order, ax=ax, palette="coolwarm", errorbar="sd")
                ax.set_title(f"{metric} (higher is better)")
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "barplot_spike_metrics.png", dpi=150)
        plt.close()
        print(f"Saved: barplot_spike_metrics.png")
    
    # 3.4 Tradeoff plot: Primary Metric vs Spike F1
    if primary_metric and "spike_f1" in df_valid.columns and df_valid["spike_f1"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors_map = {"no_spike": "gray", "deltas_only": "blue", 
                      "rolling_only": "green", "full_spike": "red"}
        markers_map = {"mse": "o", "spike_weighted": "s", "delta_loss": "^"}
        
        for _, row in df_valid.iterrows():
            if pd.notna(row.get("spike_f1")) and pd.notna(row.get(primary_metric)):
                ax.scatter(
                    row[primary_metric], 
                    row["spike_f1"],
                    c=colors_map.get(row["spike_features"], "gray"),
                    marker=markers_map.get(row["loss_type"], "o"),
                    s=150, alpha=0.7
                )
        
        ax.set_xlabel(f"{primary_metric} (lower is better)")
        ax.set_ylabel("Spike F1 Score (higher is better)")
        ax.set_title("Tradeoff: Prediction Accuracy vs Spike Detection")
        ax.grid(True, alpha=0.3)
        
        # Annotate Pareto-optimal points
        pareto_points = []
        sorted_by_metric = df_valid.dropna(subset=[primary_metric, "spike_f1"]).sort_values(primary_metric)
        best_f1 = -1
        for _, row in sorted_by_metric.iterrows():
            if row["spike_f1"] > best_f1:
                pareto_points.append(row)
                best_f1 = row["spike_f1"]
        
        if pareto_points:
            pareto_x = [p[primary_metric] for p in pareto_points]
            pareto_y = [p["spike_f1"] for p in pareto_points]
            ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.5, label="Pareto frontier")
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, label='no_spike'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='deltas_only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='rolling_only'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='full_spike'),
            Line2D([0], [0], marker='o', color='gray', markersize=10, label='mse'),
            Line2D([0], [0], marker='s', color='gray', markersize=10, label='spike_weighted'),
            Line2D([0], [0], marker='^', color='gray', markersize=10, label='delta_loss'),
        ]
        ax.legend(handles=legend_elements, loc="lower left")
        
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_accuracy_spike.png", dpi=150)
        plt.close()
        print(f"Saved: tradeoff_accuracy_spike.png")
    
    # 3.5 Per-horizon analysis (if available)
    if metric_cols["horizon"]:
        # Extract horizon metrics and reshape
        horizon_data = []
        for _, row in df_valid.iterrows():
            for col in metric_cols["horizon"]:
                if pd.notna(row.get(col)):
                    # Parse horizon from column name (e.g., "rmse_1h" -> 1)
                    try:
                        h = int(''.join(filter(str.isdigit, col.split("_")[-1])))
                        horizon_data.append({
                            "spike_features": row["spike_features"],
                            "loss_type": row["loss_type"],
                            "horizon": h,
                            "metric_name": col.rsplit("_", 1)[0],
                            "value": row[col]
                        })
                    except:
                        pass
        
        if horizon_data:
            horizon_df = pd.DataFrame(horizon_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=horizon_df, x="horizon", y="value", 
                        hue="spike_features", style="loss_type", 
                        markers=True, ax=ax)
            ax.set_xlabel("Forecast Horizon (h)")
            ax.set_ylabel("Metric Value")
            ax.set_title("Performance by Forecast Horizon")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "performance_by_horizon.png", dpi=150)
            plt.close()
            print(f"Saved: performance_by_horizon.png")
    
    # ================================================================
    # 4. RECOMMENDATIONS
    # ================================================================
    print("\n" + "="*80)
    print("4. RECOMMENDATIONS")
    print("="*80)
    
    # Best by primary metric
    if primary_metric:
        best_acc_idx = df_valid[primary_metric].idxmin()
        best_acc = df_valid.loc[best_acc_idx]
        
        print(f"\n🏆 BEST PREDICTION ACCURACY ({primary_metric}):")
        print(f"   Spike Features: {best_acc['spike_features']}")
        print(f"   Loss Type: {best_acc['loss_type']}")
        print(f"   {primary_metric}: {best_acc[primary_metric]:.5f}")
        if "spike_f1" in best_acc and pd.notna(best_acc.get("spike_f1")):
            print(f"   Spike F1: {best_acc['spike_f1']:.4f}")
    
    # Best spike detection
    if "spike_f1" in df_valid.columns and df_valid["spike_f1"].notna().any():
        best_f1_idx = df_valid["spike_f1"].idxmax()
        best_f1 = df_valid.loc[best_f1_idx]
        
        print(f"\n🎯 BEST SPIKE DETECTION (F1):")
        print(f"   Spike Features: {best_f1['spike_features']}")
        print(f"   Loss Type: {best_f1['loss_type']}")
        print(f"   Spike F1: {best_f1['spike_f1']:.4f}")
        if primary_metric:
            print(f"   {primary_metric}: {best_f1[primary_metric]:.5f}")
    
    # Best balanced (Pareto optimal)
    if primary_metric and "spike_f1" in df_valid.columns:
        # Normalize both metrics to [0,1] and find best sum
        df_temp = df_valid.dropna(subset=[primary_metric, "spike_f1"]).copy()
        if len(df_temp) > 0:
            df_temp["norm_acc"] = 1 - (df_temp[primary_metric] - df_temp[primary_metric].min()) / \
                                  (df_temp[primary_metric].max() - df_temp[primary_metric].min() + 1e-8)
            df_temp["norm_f1"] = (df_temp["spike_f1"] - df_temp["spike_f1"].min()) / \
                                 (df_temp["spike_f1"].max() - df_temp["spike_f1"].min() + 1e-8)
            df_temp["balanced_score"] = df_temp["norm_acc"] + df_temp["norm_f1"]
            
            best_bal_idx = df_temp["balanced_score"].idxmax()
            best_bal = df_temp.loc[best_bal_idx]
            
            print(f"\n⚖️  BEST BALANCED (Accuracy + Spike Detection):")
            print(f"   Spike Features: {best_bal['spike_features']}")
            print(f"   Loss Type: {best_bal['loss_type']}")
            print(f"   {primary_metric}: {best_bal[primary_metric]:.5f}")
            print(f"   Spike F1: {best_bal['spike_f1']:.4f}")
    
    # Feature contribution analysis
    if primary_metric:
        print("\n📊 SPIKE FEATURE CONTRIBUTION:")
        baseline_val = df_valid[df_valid["spike_features"] == "no_spike"][primary_metric].mean()
        if pd.notna(baseline_val):
            for sf in ["deltas_only", "rolling_only", "full_spike"]:
                sf_val = df_valid[df_valid["spike_features"] == sf][primary_metric].mean()
                if pd.notna(sf_val):
                    improvement = (baseline_val - sf_val) / baseline_val * 100
                    symbol = "✓" if improvement > 0 else "✗"
                    print(f"   {symbol} {sf}: {improvement:+.2f}% vs no_spike")
    
    # ================================================================
    # 5. SAVE SUMMARY REPORT
    # ================================================================
    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write("SPIKE FEATURES EXPERIMENT ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write("NOTE: Loss values NOT compared (different loss functions).\n")
        f.write("All comparisons based on evaluation METRICS.\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Primary comparison metric: {primary_metric}\n\n")
        
        if primary_metric:
            f.write(f"TOP 5 CONFIGURATIONS (by {primary_metric}):\n")
            f.write("-"*80 + "\n")
            top5 = df_valid.nsmallest(5, primary_metric)
            display_cols = ["spike_features", "loss_type", primary_metric]
            if "spike_f1" in df_valid.columns:
                display_cols.append("spike_f1")
            f.write(top5[display_cols].to_string())
            f.write("\n\n")
        
        f.write("AGGREGATED STATS BY SPIKE FEATURES:\n")
        f.write("-"*80 + "\n")
        f.write(spike_stats.to_string())
        f.write("\n\n")
        
        f.write("AGGREGATED STATS BY LOSS TYPE:\n")
        f.write("-"*80 + "\n")
        f.write(loss_stats.to_string())
        f.write("\n\n")
        
        f.write("COMBINATION STATS:\n")
        f.write("-"*80 + "\n")
        f.write(combo_stats.to_string())
    
    print(f"\n✅ Saved summary report to: {report_path}")
    print(f"✅ All plots saved to: {output_dir}")
    
    return df_valid, curves


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze spike experiment results")
    parser.add_argument("--results-dir", default="plots_training_dl/spike_features_experiment",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for analysis (default: results_dir/analysis)")
    
    args = parser.parse_args()
    
    df, curves = analyze_spike_experiment(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

