import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Config
EXP_FOLDER = "plots_training_dl/Experiment_TCN_complexity-24horizon"
RESULTS_PATH = f"{EXP_FOLDER}/grid_results.csv"
ANALYSIS_FOLDER = f"{EXP_FOLDER}/analysis"

os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

# Load results
df = pd.read_csv(RESULTS_PATH)

# Open report file
report_path = f"{ANALYSIS_FOLDER}/analysis_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    
    def log(text=""):
        print(text)
        f.write(text + "\n")
    
    log("="*60)
    log("TCN COMPLEXITY EXPERIMENT ANALYSIS")
    log("="*60)
    log(f"\nLoaded {len(df)} experiments\n")

    # ─────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────
    log("\n" + "="*60)
    log("SUMMARY - SORTED BY BEST VAL LOSS")
    log("="*60)
    summary_cols = ["exp_name", "num_channels", "kernel_size", "pooling", "use_se", 
                    "best_val_loss", "best_epoch", "final_val_loss"]
    summary_cols = [c for c in summary_cols if c in df.columns]
    log(df[summary_cols].sort_values("best_val_loss").to_string(index=False))

    # ─────────────────────────────────────────────
    # SPIKE METRICS (if available)
    # ─────────────────────────────────────────────
    spike_cols = [c for c in df.columns if "spike" in c.lower() or "recall" in c.lower() or "precision" in c.lower() or "f1" in c.lower()]
    if spike_cols:
        log("\n" + "="*60)
        log("SPIKE DETECTION METRICS")
        log("="*60)
        log(df[["exp_name"] + spike_cols].sort_values(spike_cols[0] if spike_cols else "exp_name", ascending=False).to_string(index=False))

    # ─────────────────────────────────────────────
    # DETAILED ANALYSIS
    # ─────────────────────────────────────────────
    log("\n" + "="*60)
    log("INSIGHTS")
    log("="*60)

    baseline_loss = df[df["exp_name"] == "BASELINE"]["best_val_loss"].values[0] if "BASELINE" in df["exp_name"].values else None

    if baseline_loss:
        log(f"\nBaseline val loss: {baseline_loss:.4f}")
        log("\nImprovement over baseline:")
        for _, row in df.sort_values("best_val_loss").iterrows():
            delta = ((baseline_loss - row["best_val_loss"]) / baseline_loss) * 100
            symbol = "✓" if delta > 0 else "✗"
            log(f"  {symbol} {row['exp_name']:20s}: {row['best_val_loss']:.4f} ({delta:+.2f}%)")

    # Best by category
    log("\n" + "-"*40)
    log("BEST BY CATEGORY:")
    log("-"*40)

    categories = {
        "Depth/Width": df[df["exp_name"].str.contains("DEPTH")],
        "Kernel Size": df[df["exp_name"].str.contains("K5|K7", regex=True)],
        "Attention Pool": df[df["exp_name"].str.contains("ATTENTION")],
        "SE Blocks": df[df["exp_name"].str.contains("SE_")],
    }

    for cat_name, cat_df in categories.items():
        if len(cat_df) > 0:
            best = cat_df.loc[cat_df["best_val_loss"].idxmin()]
            log(f"  {cat_name:15s}: {best['exp_name']} (loss={best['best_val_loss']:.4f})")

    log("\n" + "="*60)
    log("Analysis complete.")
    log("="*60)

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bar plot: Best val loss by experiment
ax1 = axes[0, 0]
df_sorted = df.sort_values("best_val_loss")
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
ax1.barh(df_sorted["exp_name"], df_sorted["best_val_loss"], color=colors)
ax1.set_xlabel("Best Validation Loss")
ax1.set_title("Best Validation Loss by Experiment")
ax1.axvline(df_sorted["best_val_loss"].min(), color='green', linestyle='--', alpha=0.7, label='Best')
ax1.legend()

# 2. Grouped comparison: Depth/Width effect
ax2 = axes[0, 1]
depth_exps = df[df["exp_name"].str.contains("BASELINE|DEPTH", regex=True)].copy()
if len(depth_exps) > 0:
    depth_exps = depth_exps.sort_values("best_val_loss")
    ax2.barh(depth_exps["exp_name"], depth_exps["best_val_loss"], color='steelblue')
    ax2.set_xlabel("Best Validation Loss")
    ax2.set_title("Depth/Width Comparison")

# 3. Kernel size effect
ax3 = axes[1, 0]
kernel_exps = df[df["exp_name"].str.contains("BASELINE|K5|K7", regex=True) & ~df["exp_name"].str.contains("DEEP")].copy()
if len(kernel_exps) > 0:
    kernel_exps = kernel_exps.sort_values("kernel_size")
    ax3.bar(kernel_exps["exp_name"], kernel_exps["best_val_loss"], color='coral')
    ax3.set_ylabel("Best Validation Loss")
    ax3.set_title("Kernel Size Effect (Base Architecture)")
    ax3.tick_params(axis='x', rotation=45)

# 4. Pooling & SE comparison
ax4 = axes[1, 1]
special_exps = df[df["exp_name"].str.contains("BASELINE|ATTENTION_POOL|SE_BASE", regex=True)].copy()
if len(special_exps) > 0:
    x = np.arange(len(special_exps))
    ax4.bar(x, special_exps["best_val_loss"], color=['gray', 'purple', 'orange'][:len(special_exps)])
    ax4.set_xticks(x)
    ax4.set_xticklabels(special_exps["exp_name"], rotation=45, ha='right')
    ax4.set_ylabel("Best Validation Loss")
    ax4.set_title("Pooling & SE Block Effect")

plt.tight_layout()
plt.savefig(f"{ANALYSIS_FOLDER}/summary_plots.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# HEATMAP: Kernel vs Depth interaction
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 6))

df["depth"] = df["num_channels"].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
pivot_data = df.pivot_table(values="best_val_loss", index="kernel_size", columns="depth", aggfunc="min")

if not pivot_data.empty:
    sns.heatmap(pivot_data, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("Val Loss: Kernel Size vs Network Depth")
    ax.set_xlabel("Network Depth (# layers)")
    ax.set_ylabel("Kernel Size")
    plt.savefig(f"{ANALYSIS_FOLDER}/heatmap_kernel_depth.png", dpi=150, bbox_inches='tight')
    plt.show()

print(f"\nResults saved to: {ANALYSIS_FOLDER}/")