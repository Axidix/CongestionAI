import json
import re
import csv

# Read the notebook file
with open("find_issues.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract all outputs from all cells
all_outputs = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and cell.get("outputs"):
        all_outputs.append(f"\n{'='*60}\nCELL {i}\n{'='*60}")
        for output in cell["outputs"]:
            if output.get("text"):  # stream output (print statements)
                all_outputs.append("".join(output["text"]))
            elif output.get("data", {}).get("text/plain"):  # return values
                all_outputs.append("".join(output["data"]["text/plain"]))

# Save raw outputs to file
with open("all_cell_outputs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_outputs))

print("Saved raw outputs to all_cell_outputs.txt")

# ============================================================
# EXTRACT METRICS FROM OUTPUTS
# ============================================================

full_text = "\n".join(all_outputs)

# Patterns to extract
epoch_pattern = r"Epoch (\d+)/(\d+).+?Train Loss: ([\d.]+).+?Val Loss: ([\d.]+)"
tqdm_pattern = r"Epoch (\d+)/(\d+) - training: 100%\|[.*?]+\| \d+/\d+ \[(\d+:\d+)<"
best_val_pattern = r"Best val loss: ([\d.]+)"
early_stop_pattern = r"Early stopping at epoch (\d+)"
test_loss_pattern = r"Test Loss \((.+?)\): ([\d.]+)"

# Split by experiment runs
experiment_blocks = re.split(r"={60}\nRunning:", full_text)

results = []
max_epochs = 0

for block in experiment_blocks[1:]:  # Skip first empty block
    exp_match = re.match(r"\s*(.+?)\n", block)
    if not exp_match:
        continue
    
    exp_name = exp_match.group(1).strip()
    
    # Extract all epoch losses
    epochs = re.findall(epoch_pattern, block)
    
    # Extract training times from tqdm bars
    tqdm_times = re.findall(tqdm_pattern, block)
    time_per_epoch = {}
    for t in tqdm_times:
        epoch_num = int(t[0])
        time_str = t[2]  # format: "00:31" or "1:23"
        parts = time_str.split(":")
        if len(parts) == 2:
            seconds = int(parts[0]) * 60 + int(parts[1])
        else:
            seconds = int(parts[0])
        time_per_epoch[epoch_num] = seconds
    
    # Extract best val loss
    best_val_match = re.search(best_val_pattern, block)
    best_val = float(best_val_match.group(1)) if best_val_match else None
    
    # Extract early stopping epoch
    early_stop_match = re.search(early_stop_pattern, block)
    early_stop_epoch = int(early_stop_match.group(1)) if early_stop_match else None
    
    # Extract test loss
    test_loss_match = re.search(test_loss_pattern, block)
    test_loss = float(test_loss_match.group(2)) if test_loss_match else None
    
    # Parse experiment name components
    parts = exp_name.replace("grid/", "").split("_")
    n_blocks = None
    hidden_dim = None
    
    for part in parts:
        if part.startswith("b") and part[1:].isdigit():
            n_blocks = int(part[1:])
        elif part.startswith("h") and part[1:].isdigit():
            hidden_dim = int(part[1:])
    
    # Extract history and weather config from name
    history = None
    if "24h_hourly" in exp_name:
        history = "24h_hourly"
    elif "24h_1week_daily" in exp_name:
        history = "24h_1week_daily"
    elif "24h_1week_6h" in exp_name:
        history = "24h_1week_6h"
    
    weather_lags = None
    if "_minimal" in exp_name:
        weather_lags = "minimal"
    elif "_sparse" in exp_name:
        weather_lags = "sparse_24h"
    elif "_dense" in exp_name:
        weather_lags = "dense_24h"
    
    # Calculate best val from epochs if not explicitly logged
    if best_val is None and epochs:
        best_val = min(float(e[3]) for e in epochs)
    
    # Calculate total training time
    total_time = sum(time_per_epoch.values()) if time_per_epoch else None
    
    # Build result row
    result = {
        "exp_name": exp_name,
        "n_blocks": n_blocks,
        "hidden_dim": hidden_dim,
        "history": history,
        "weather_lags": weather_lags,
        "num_epochs": len(epochs),
        "early_stop_epoch": early_stop_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "total_time_sec": total_time,
    }
    
    # Add per-epoch losses and times
    for i, epoch_data in enumerate(epochs):
        epoch_num = i + 1
        result[f"train_loss_ep{epoch_num}"] = float(epoch_data[2])
        result[f"val_loss_ep{epoch_num}"] = float(epoch_data[3])
        result[f"time_sec_ep{epoch_num}"] = time_per_epoch.get(epoch_num, None)
    
    max_epochs = max(max_epochs, len(epochs))
    results.append(result)

# Build complete fieldnames with all possible epochs
fieldnames = [
    "exp_name", "n_blocks", "hidden_dim", "history", "weather_lags",
    "num_epochs", "early_stop_epoch", "best_val_loss", "test_loss", "total_time_sec"
]
for i in range(1, max_epochs + 1):
    fieldnames.extend([f"train_loss_ep{i}", f"val_loss_ep{i}", f"time_sec_ep{i}"])

# Save metrics to CSV
if results:
    with open("experiment_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved {len(results)} experiment metrics to experiment_metrics.csv")
    
    # Print summary
    print("\n" + "="*100)
    print("EXPERIMENT METRICS SUMMARY")
    print("="*100)
    print(f"{'Experiment':<55} {'Best Val':<12} {'Test':<12} {'Time(s)':<10}")
    print("-"*100)
    for r in sorted(results, key=lambda x: x.get("best_val_loss") or 999):
        test_str = f"{r['test_loss']:.4f}" if r.get('test_loss') else "N/A"
        time_str = f"{r['total_time_sec']}" if r.get('total_time_sec') else "N/A"
        best_val_str = f"{r['best_val_loss']:.4f}" if r.get('best_val_loss') else "N/A"
        print(f"{r['exp_name']:<55} {best_val_str:<12} {test_str:<12} {time_str:<10}")
else:
    print("No experiment results found in outputs.")