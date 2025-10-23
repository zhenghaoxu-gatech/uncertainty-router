import torch
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
from openrlhf.utils import get_tokenizer
from openrlhf.models import get_llm_for_sequence_regression
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class SkyworkDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = self.tokenizer.apply_chat_template(item['context_messages'], tokenize=False)
        return {
            'context_messages': messages,
            'label': item['label']
        }

def create_calibration_plot(reward_diff, n_bins=10):
    score_diffs = np.array(reward_diff)
    raw_probs = 1 / (1 + np.exp(-score_diffs))
    probs = np.where(score_diffs >= 0, raw_probs, 1 - raw_probs)
    correct = (score_diffs > 0).astype(int)
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_accuracies, bin_confidences, bin_counts = [], [], []
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
    print(reward_diff.shape)
    print('uncertainty: ', uncertainties.mean(), uncertainties.std())
    # np.savez('train.npz', reward_diff=reward_diff, uncertainties=uncertainties)
    np.savez('validation.npz', reward_diff=reward_diff, uncertainties=uncertainties)
    # Score differences with uncertainty
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot score differences histogram
    sns.histplot(data=reward_diff, bins=50, color='blue', alpha=0.6, ax=ax1)
    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.set_title('Score Differences\n(Assistant_1 - Assistant_2)')
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
            if mask.any():
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

class RewardModelEvaluator:
    def __init__(self, 
                model_path: str, 
                batch_size: int = 8, 
                value_head_prefix: str = "value_head", 
                model_type: str = "preference", 
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
        
        self.model = get_llm_for_sequence_regression(
            model_path, model_type, bf16=True, use_flash_attention_2=True,
            value_head_prefix=value_head_prefix, use_sn=use_sn, sn_range=sn_range, use_gp=use_gp,
            gp_amplitude=gp_amplitude, use_mcd=use_mcd, mcd_p=mcd_p, normalize_reward=False
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
        messages_list = [item['context_messages'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        
        # Tokenize all messages
        inputs = self.tokenizer(
            messages_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        
        return {
            'inputs': inputs,
            'labels': labels
        }

    def evaluate_batch(self, batch):
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
            if isinstance(self.model, DDP):
                model = self.model.module
            else:
                model = self.model
                
            scores, uncertainties = model.predict(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            return scores.cpu(), batch['labels'], uncertainties.cpu()


def evaluate_on_skywork(
        model_path: str, 
        data_path: str, 
        save_path: str, 
        batch_size: int = 8, 
        value_head_prefix: str = "value_head", 
        model_type: str = "preference", 
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
    
    # dataset = load_dataset(data_path, split="train[:2000]")
    dataset = load_dataset(data_path, split="validation")
    eval_dataset = SkyworkDataset(dataset, evaluator.tokenizer)
    
    sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=True) if local_rank != -1 else None
    
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size,
        collate_fn=evaluator.collate_fn,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
    )

    total_diffs = []
    total_uncertainties = []
    total_labels = []
    total_results = []
    
    # First collect all scores and uncertainties
    for batch in tqdm(dataloader, disable=(local_rank != 0)):
        if not use_mcd:
            scores, labels, uncertainties = evaluator.evaluate_batch(batch)
            # if local_rank == 0:
            #     print(scores, labels, uncertainties, batch['labels'])
        else:
            pass
        total_diffs.extend(scores.flatten().tolist())
        total_uncertainties.extend(uncertainties.flatten().tolist())
        total_labels.extend(labels.flatten().tolist())
    
    np_uncertainties = np.array(total_uncertainties)
    cnt = 0
    print(f'local rank: {local_rank},\t cnt: {cnt},\t tot: {len(total_uncertainties)}')
    
    # Process scores in pairs

    if local_rank != -1:
        world_size = dist.get_world_size()
        all_labels = [None] * world_size
        all_diffs = [None] * world_size
        all_uncertainties = [None] * world_size
        
        dist.all_gather_object(all_labels, total_labels)
        dist.all_gather_object(all_diffs, total_diffs)
        dist.all_gather_object(all_uncertainties, total_uncertainties)
        
        if local_rank == 0:
            total_labels = np.array(all_labels).T.ravel().tolist()
            total_diffs = np.array(all_diffs).T.ravel().tolist()
            total_uncertainties = np.array(all_uncertainties).T.ravel().tolist()
            # total_diffs = sum(all_diffs, [])
            # total_uncertainties = sum(all_uncertainties, [])
            final_diffs = []
            final_uncertainties = []

            for i in range(0, len(total_uncertainties), 2):
                if i + 1 >= len(total_uncertainties):  # Skip if we don't have a pair
                    continue
                    
                score1 = total_diffs[i]
                score2 = total_diffs[i + 1]
                uncertainty1 = total_uncertainties[i]
                uncertainty2 = total_uncertainties[i + 1]
                label1 = total_labels[i]
                label2 = total_labels[i + 1]
                
                # Verify this is a proper pair
                if abs(label1 + label2 - 1.0) > 1e-5:
                    continue
                    
                # Combined score and uncertainty for chosen > rejected
                if abs(label1 - 1.0) < 1e-5:
                    combined_score = (score1 - score2) / 2
                    combined_uncertainty = (uncertainty1 + uncertainty2) / 2
                else:
                    combined_score = (score2 - score1) / 2
                    combined_uncertainty = (uncertainty2 + uncertainty1) / 2
                    
                correct = (combined_score > 0)
                # total_correct += correct
                # total_samples += 1
                final_diffs.append(combined_score)
                final_uncertainties.append(combined_uncertainty)
                total_results.append({
                    "correct": correct,
                    "reward_diff": combined_score,
                    "uncertainty": combined_uncertainty,
                })

    if local_rank == 0 or local_rank == -1:
        os.makedirs(save_path, exist_ok=True)
        
        accuracy = (np.array(final_diffs)>=0).sum() / len(final_diffs)
        final_results = {
            'accuracy': accuracy,
            'correct': float((np.array(final_diffs)>=0).sum()),
            'total': float(len(final_diffs))
        }

        ece = plot_score_distributions(
            final_diffs, 
            uncertainties=final_uncertainties,
            # save_path=os.path.join(save_path, 'score_distributions_train.png')
            save_path=os.path.join(save_path, 'score_distributions_validation.png')
        )
        final_results['ece'] = ece
        
        # with open(os.path.join(save_path, 'train.json'), "w") as f:
        with open(os.path.join(save_path, 'validation.json'), "w") as f:
            json.dump(final_results, f)
            
        print("\nSkywork Evaluation Results:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {final_results['correct']}/{final_results['total']}")
        return final_results
    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate reward models on Skywork validation set')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the reward model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to HF dataset')
    parser.add_argument('--save_path', type=str, default='./eval_results/validation',
                      help='Path to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    parser.add_argument('--value_head_prefix', type=str, default="value_head",
                      help='Prefix for the value head in the model')
    parser.add_argument('--model_type', type=str, default="preference",
                      help='Model type, reward or preference')
    parser.add_argument("--use_sn", action="store_true", default=False, help="Enable Spectral Normalization")
    parser.add_argument("--use_gp", action="store_true", default=False, help="Enable Gaussian Process")
    parser.add_argument("--sn_range", type=float, default=10., help="Spectral Normalization Range")
    parser.add_argument("--gp_amplitude", type=float, default=0.1, help="Gaussian Process Amplitude")
    parser.add_argument("--use_mcd", action="store_true", default=False, help="Enable MC Dropout")
    parser.add_argument("--mcd_p", type=float, default=0.2, help="MC Dropout rate")
    parser.add_argument('--n_dropout', type=int, default=3,
                      help='MC dropout count')
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    args = parser.parse_args()
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    
    results = evaluate_on_skywork(
        model_path=args.model_path,
        data_path=args.data_path,
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
    
    if local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
