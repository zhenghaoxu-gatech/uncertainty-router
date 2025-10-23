import json
import os
from pathlib import Path
import itertools
import csv

os.system("aws s3 cp ${S3_BUCKET}/eval_results/rm ./eval_results --recursive")

def get_expected_model_names():
    # Parameters from the script
    pretrain_name = "it"  # or "sft"
    data_job_names = ["hs_bin", "hs_tri"]
    loss_types = ["sigmoid", "scaled_bt"]
    learning_rates = ["2e-6", "5e-6", "8e-6", "1e-5"]
    uncertainty = "mcd"
    
    # First set of combinations
    epochs1 = ["1", "2", "3"]
    dropouts1 = ["0.0"]
    n_dropouts1 = ["1"]
    
    # Second set of combinations
    epochs2 = ["1", "2", "3", "4"]
    dropouts2 = ["0.05"]
    n_dropouts2 = ["3", "5", "10"]
    
    # Third set of combinations
    epochs3 = ["3", "4", "5"]
    dropouts3 = ["0.1"]
    n_dropouts3 = ["3", "5", "10"]
    
    model_names = set()
    
    # Generate all combinations
    combinations = [
        (epochs1, dropouts1, n_dropouts1),
        (epochs2, dropouts2, n_dropouts2),
        (epochs3, dropouts3, n_dropouts3)
    ]
    
    for epochs, dropouts, n_dropouts in combinations:
        for loss_type, lr, epoch, dropout, n_dropout, data_job_name in itertools.product(
            loss_types, learning_rates, epochs, dropouts, n_dropouts, data_job_names):
            model_name = f"Llama-3-{pretrain_name}-8B-pm-{data_job_name}-{loss_type}-lr{lr}-ep{epoch}-{uncertainty}{dropout}-n{n_dropout}"
            model_names.add(model_name)
    
    return model_names

# Define headers
headers = ["model_name", 
           "rb_overall_acc", "rb_ece",
           "rb_chat_acc", "rb_chat_hard_acc", "rb_safety_acc", "rb_reasoning_acc",
           "rm_total_avg_acc", "rm_ece",
           "rm_chat", "rm_math", "rm_code", "rm_safety", "rm_hard_acc", "rm_normal_acc", "rm_easy_acc"]

# Function to print row with aligned columns
def print_row(row, headers, widths):
    row_str = ""
    for header in headers:
        value = row.get(header, "")
        row_str += f"{str(value):<{widths[header]}} "
    print(row_str)

# Prepare data
results = []
expected_models = get_expected_model_names()

# Get all json files
eval_dir = Path("/workspace/eval_results")
for model_dir in eval_dir.iterdir():
    if model_dir.is_dir() and model_dir.name in expected_models:
        rb_path = model_dir / "rewardbench.json"
        rm_path = model_dir / "rmbench.json"
        
        row = {"model_name": model_dir.name}
        
        # Process rewardbench.json
        if rb_path.exists():
            with open(rb_path, 'r') as f:
                rb_data = json.load(f)
                row.update({
                    "rb_overall_acc": f"{rb_data['overall']['accuracy']:.4f}",
                    "rb_ece": f"{rb_data['ece']:.4f}",
                    "rb_chat_acc": f"{rb_data['categories']['chat']['accuracy']:.4f}",
                    "rb_chat_hard_acc": f"{rb_data['categories']['chat_hard']['accuracy']:.4f}",
                    "rb_safety_acc": f"{rb_data['categories']['safety']['accuracy']:.4f}",
                    "rb_reasoning_acc": f"{rb_data['categories']['reasoning']['accuracy']:.4f}"
                })
        
        # Process rmbench.json
        if rm_path.exists():
            with open(rm_path, 'r') as f:
                rm_data = json.load(f)
                row.update({
                    "rm_total_avg_acc": f"{rm_data['total_avg_acc']:.4f}",
                    "rm_ece": f"{rm_data['ece']:.4f}",
                    "rm_chat": f"{rm_data['chat']:.4f}",
                    "rm_math": f"{rm_data['math']:.4f}",
                    "rm_code": f"{rm_data['code']:.4f}",
                    "rm_safety": f"{rm_data['safety']:.4f}",
                    "rm_hard_acc": f"{rm_data['hard_acc']:.4f}",
                    "rm_normal_acc": f"{rm_data['normal_acc']:.4f}",
                    "rm_easy_acc": f"{rm_data['easy_acc']:.4f}"
                })
        
        # Only add the row if at least one of the files exists
        if len(row) > 1:
            results.append(row)

# Sort results by model name for better readability
results.sort(key=lambda x: x["model_name"])

# Calculate column widths
widths = {header: len(header) for header in headers}
for row in results:
    for header in headers:
        if header in row:
            widths[header] = max(widths[header], len(str(row[header])))

# Print headers
print("\nResults:")
print_row(dict(zip(headers, headers)), headers, widths)
print("-" * (sum(widths.values()) + len(headers)))  # Fixed line

# Print rows
for row in results:
    print_row(row, headers, widths)

# Write to CSV
output_file = "/workspace/eval_results/results.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults have been saved to {output_file}")
