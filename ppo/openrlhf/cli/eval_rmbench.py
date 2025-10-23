import torch
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
from openrlhf.utils import get_tokenizer
from openrlhf.models import get_llm_for_sequence_regression
from tqdm import tqdm
import os
from typing import List, Dict, Any
from openrlhf.utils.rmbench_utils import compute_accuracy, save_intermediate_results, load_intermediate_results
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class RMBenchDataset(Dataset):
    def __init__(self, dataset, tokenizer, model_type="reward"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        domain = item['domain']

        if self.model_type == "reward":
            return None
            # TODO
            chosen_text = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ], tokenize=False)
            
            rejected_text = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ], tokenize=False)

            return {
                'chosen_text': chosen_text,
                'rejected_text': rejected_text,
                'subset': subset
            }
        else:  # preference model
            texts_1 = []
            texts_2 = []
            for i in range(3):      # chosen idx
                for j in range(3):  # rejected idx
                    texts_1.append(
                        self.tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt},
                        {"role": "assistant_1", "content": chosen[i]},
                        {"role": "assistant_2", "content": rejected[j]},
                    ], tokenize=False)
                    )
                    texts_2.append(
                        self.tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt},
                        {"role": "assistant_1", "content": rejected[j]},
                        {"role": "assistant_2", "content": chosen[i]},
                    ], tokenize=False)
                    )

            return {
                'texts_1': texts_1,
                'texts_2': texts_2,
                'domain': domain
            }

class RewardModelEvaluator:
    def __init__(self, 
                model_path: str, 
                batch_size: int = 8, 
                value_head_prefix: str = "value_head", 
                model_type: str = "reward", 
                use_sn: bool = False,
                sn_range: float = 10.,
                use_gp: bool = False,
                gp_amplitude: float = 0.1,
                use_mcd: bool = False,
                mcd_p: float = 0.2,
                local_rank: int = -1,
        ):
        self.local_rank = local_rank
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")
        print(self.local_rank, '>>>>>>', self.device)
        
        self.model = get_llm_for_sequence_regression(
            model_path, model_type, bf16=True, use_flash_attention_2=True,
            value_head_prefix=value_head_prefix, use_sn=use_sn, sn_range=sn_range, use_gp=use_gp,
            gp_amplitude=gp_amplitude, use_mcd=use_mcd, mcd_p=mcd_p
        ).to(self.device)
        
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank])
        
        self.tokenizer = get_tokenizer(model_path, self.model, "left", None, use_fast=True)
        self.batch_size = batch_size
        self.model_type = model_type
        if use_mcd:
            self.model.train()
        else:
            self.model.eval()

    def collate_fn(self, batch):
        if self.model_type == "reward":
            return None
            chosen_texts = [item['chosen_text'] for item in batch]
            rejected_texts = [item['rejected_text'] for item in batch]
            subsets = [item['subset'] for item in batch]

            chosen_inputs = self.tokenizer(
                chosen_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=8192
            )
            rejected_inputs = self.tokenizer(
                rejected_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=8192
            )

            return {
                'chosen_inputs': chosen_inputs,
                'rejected_inputs': rejected_inputs,
                'subsets': subsets
            }
        else:
            # print(type(batch[0]['texts_1']))
            # print(len(batch[0]['texts_1']))
            texts_1 = sum([item['texts_1'] for item in batch], [])  # (9B,)
            texts_2 = sum([item['texts_2'] for item in batch], [])  # (9B,)
            domains = [item['domain'] for item in batch]            # (B,)

            inputs_1 = self.tokenizer(
                texts_1, return_tensors="pt", padding=True,
                truncation=True, max_length=8192
            )
            inputs_2 = self.tokenizer(
                texts_2, return_tensors="pt", padding=True,
                truncation=True, max_length=8192
            )

            return {
                'inputs_1': inputs_1,
                'inputs_2': inputs_2,
                'domains': domains
            }

    def evaluate_batch(self, batch):
        with torch.no_grad():
            if self.model_type == "reward":
                return None
            else:
                inputs_1 = {k: v.to(self.device) for k, v in batch['inputs_1'].items()}
                inputs_2 = {k: v.to(self.device) for k, v in batch['inputs_2'].items()}
                
                if isinstance(self.model, DDP):
                    model = self.model.module
                else:
                    model = self.model
                    
                scores_1, var_1 = model.predict(
                    inputs_1['input_ids'],
                    inputs_1['attention_mask']
                )
                scores_2, var_2 = model.predict(
                    inputs_2['input_ids'],
                    inputs_2['attention_mask']
                )
                # Average the uncertainties from both directions
                uncertainties = torch.maximum(var_1, var_2)
                return ((scores_1 - scores_2) / 2).cpu(), batch['domains'], uncertainties.cpu()

# Add this function to create and save the plots
def create_calibration_plot(reward_diff, n_bins=10):
    """
    Creates a calibration plot comparing predicted probability (based on score difference)
    to observed accuracy, ensuring probabilities are always >= 0.5
    """
    score_diffs = np.array(reward_diff)
    
    # Convert score differences to probabilities using sigmoid and ensure >= 0.5
    raw_probs = 1 / (1 + np.exp(-score_diffs))
    probs = np.where(score_diffs >= 0, raw_probs, 1 - raw_probs)
    
    # Actual correctness (1 if chosen > rejected, 0 otherwise)
    correct = (score_diffs >= 0).astype(int)
    
    # Calculate calibration curve with bins from 0.5 to 1.0
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(correct[mask]))
            bin_confidences.append(np.mean(probs[mask]))
            bin_counts.append(np.sum(mask))
    
    return np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)

def plot_score_distributions(reward_diff, uncertainties=None, save_path="score_distributions.png"):
    plt.figure(figsize=(15, 5))
    
    # Convert inputs to numpy arrays if they aren't already
    reward_diff = np.array(reward_diff)
    if uncertainties is not None:
        uncertainties = np.array(uncertainties)
    
    # Score differences with uncertainty
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot score differences histogram
    sns.histplot(data=reward_diff, bins=50, color='blue', alpha=0.6, ax=ax1)
    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.set_title('Score Differences\n(Chosen - Rejected)')
    ax1.set_xlabel('Score Difference')
    ax1.set_ylabel('Count', color='blue')
    
    if uncertainties is not None:
        # Create bins and compute mean uncertainty for each bin
        bins = np.histogram_bin_edges(reward_diff, bins=50)
        bin_indices = np.digitize(reward_diff, bins) - 1
        mean_uncertainties = []
        bin_centers = []
        
        for i in range(len(bins)-1):
            mask = (bin_indices == i)
            if mask.any():  # Changed from np.sum(mask) > 0
                mean_uncertainties.append(float(np.mean(uncertainties[mask])))
                bin_centers.append(float((bins[i] + bins[i+1]) / 2))
        
        # Convert to numpy arrays
        mean_uncertainties = np.array(mean_uncertainties)
        bin_centers = np.array(bin_centers)
        
        # Plot mean uncertainty line
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, mean_uncertainties, color='red', linewidth=2, label='Mean Uncertainty')
        ax2.set_ylabel('Uncertainty', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Calibration plot
    plt.subplot(1, 2, 2)
    conf, acc, counts = create_calibration_plot(reward_diff)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    sizes = 50 * counts / np.max(counts)
    plt.scatter(conf, acc, s=sizes, alpha=0.6, label='Model calibration')
    plt.title('Calibration Plot')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    ece = np.average(np.abs(conf - acc), weights=counts)
    print(f"\nCalibration Metrics:")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    return ece

def evaluate_on_rmbench(
        model_path: str, 
        save_path: str, 
        batch_size: int = 8, 
        value_head_prefix: str = "value_head", 
        model_type: str = "reward", 
        use_sn: bool = False,
        sn_range: float = 10.,
        use_gp: bool = False,
        gp_amplitude: float = 0.1,
        use_mcd: bool = False,
        mcd_p: float = 0.2,
        n_dropout: int = 3,
        local_rank: int = -1,
    ):
    evaluator = RewardModelEvaluator(
        model_path, 
        batch_size, 
        value_head_prefix, 
        model_type=model_type, 
        use_sn=use_sn,
        sn_range=sn_range,
        use_gp=use_gp,
        gp_amplitude=gp_amplitude,
        use_mcd=use_mcd,
        mcd_p=mcd_p,
        local_rank=local_rank
    )

    if local_rank != -1:
        print(f"Process {local_rank} using device: {evaluator.device}")
    
    # Load dataset
    dataset = load_dataset("THU-KEG/RM-Bench", split="train")
    reward_dataset = RMBenchDataset(dataset, evaluator.tokenizer, model_type)
    
    # Add DistributedSampler
    sampler = DistributedSampler(reward_dataset) if local_rank != -1 else None
    
    dataloader = DataLoader(
        reward_dataset, 
        batch_size=batch_size,
        collate_fn=evaluator.collate_fn,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None)
    )

    total_diffs = []
    total_uncertainties = []
    results = []

    for batch in tqdm(dataloader, disable=(local_rank != 0)):
        if not use_mcd:
            reward_diffs, domains, uncertainties = evaluator.evaluate_batch(batch)
        else:
            all_scores = []
            all_uncertainties = []
            for _ in range(n_dropout):
                batch_scores, domains, batch_uncertainties = evaluator.evaluate_batch(batch)
                all_scores.append(batch_scores)
                all_uncertainties.append(batch_uncertainties)
            reward_diffs = torch.stack(all_scores).mean(dim=0)
            uncertainties = torch.stack(all_uncertainties).mean(dim=0)

        total_diffs.extend(reward_diffs.cpu().tolist())
        total_uncertainties.extend(uncertainties.cpu().tolist())
        for i in range(len(domains)):
            results.append({
                "domain": domains[i],
                "reward_diff": reward_diffs[9*i:9*(i+1)].float().numpy().reshape(3, 3),
                "uncertainty": uncertainties[9*i:9*(i+1)].float().numpy().reshape(3, 3)
            })

    # Gather results from all processes
    if local_rank != -1:
        world_size = dist.get_world_size()
        
        all_results = [None] * world_size
        all_total_diffs = [None] * world_size
        all_total_uncertainties = [None] * world_size
        
        dist.all_gather_object(all_results, results)
        dist.all_gather_object(all_total_diffs, total_diffs)
        dist.all_gather_object(all_total_uncertainties, total_uncertainties)
        
        if local_rank == 0:
            results = sum(all_results, [])
            total_diffs = sum(all_total_diffs, [])
            total_uncertainties = sum(all_total_uncertainties, [])
    
    if local_rank == 0 or local_rank == -1:
        os.makedirs(save_path, exist_ok=True)
        
        # Verify the lengths match
        n_comparisons = len(results) * 9  # Each result has 9 comparisons
        assert len(total_diffs) == n_comparisons, f"Mismatch in data sizes: {len(total_diffs)} total_diffs vs {n_comparisons} expected from results"
        
        # Save intermediate results
        save_intermediate_results(save_path, results, total_diffs)
            
        final_results = compute_accuracy(results, model_type=model_type)
        ece = plot_score_distributions(total_diffs, uncertainties=total_uncertainties, save_path=os.path.join(save_path, 'score_distributions_rmbench.png'))
        final_results['ece'] = ece
        
        # Add size information to final_results
        final_results['total_samples'] = len(total_diffs)
        final_results['total_prompts'] = len(results)
        
        with open(os.path.join(save_path, 'rmbench.json'), "w") as f:
            json.dump(final_results, f)
        return final_results
    return None

def plot_from_saved_results(results_dir: str):
    """
    Create plots from saved intermediate results without running evaluation again.
    """
    try:
        # Load intermediate results
        total_diffs, results = load_intermediate_results(results_dir)
        # final_diffs = np.stack([x['reward_diff'] for x in results if x['domain'].startswith('code')]).flatten()
        final_diffs = np.stack([x['reward_diff'] for x in results]).flatten()
        final_results = compute_accuracy(results, model_type="preference")
        ece = plot_score_distributions(final_diffs, save_path=os.path.join(results_dir, 'score_distributions_rmbench_replot.png'))
        final_results['ece'] = ece
        print("Final Results:", final_results)
        
        return final_results
    except Exception as e:
        print(f"Error loading or plotting results: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Evaluate reward models on RMBench')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the reward model')
    parser.add_argument('--save_path', type=str, default='./eval_results/rmbench',
                      help='Path to the reward model')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    parser.add_argument('--value_head_prefix', type=str, default="value_head",
                      help='Prefix for the value head in the model')
    parser.add_argument('--model_type', type=str, default="reward",
                      help='Model type, reward or preference')
    parser.add_argument("--use_sn", action="store_true", default=False, help="Enable Spectral Normalization")
    parser.add_argument("--sn_range", type=float, default=10., help="Spectral Normalization Range")
    parser.add_argument("--use_gp", action="store_true", default=False, help="Enable Gaussian Process")
    parser.add_argument("--gp_amplitude", type=float, default=0.1, help="Gaussian Process Amplitude")
    parser.add_argument("--use_mcd", action="store_true", default=False, help="Enable MC Dropout")
    parser.add_argument("--mcd_p", type=float, default=0.2, help="MC Dropout rate")
    parser.add_argument('--n_dropout', type=int, default=3,
                      help='MC dropout count')
    parser.add_argument('--plot_only', action='store_true',
                      help='Only create plots from saved results without running evaluation')
    
    
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="Local rank for distributed training")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    args = parser.parse_args()
    
    if args.plot_only:
        results = plot_from_saved_results(args.save_path)
        print("Plots created from saved results")
        return
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    
    results = evaluate_on_rmbench(
        model_path=args.model_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        value_head_prefix=args.value_head_prefix,
        model_type=args.model_type,
        use_sn=args.use_sn,
        sn_range=args.sn_range,
        use_gp=args.use_gp,
        gp_amplitude=args.gp_amplitude,
        use_mcd=args.use_mcd,
        mcd_p=args.mcd_p,
        n_dropout=args.n_dropout,
        local_rank=local_rank,
    )
    
    if local_rank <= 0:  # Print only on main process or single GPU
        print(results)

    if local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
