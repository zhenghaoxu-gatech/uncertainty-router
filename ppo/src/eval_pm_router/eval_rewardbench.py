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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import boto3
import re
from botocore.exceptions import ClientError

def plot_accuracy_vs_uncertainty(reward_diff, uncertainties, save_path="accuracy_vs_uncertainty.png"):
    """
    Creates a plot showing the relationship between prediction accuracy and uncertainty.
    """
    plt.figure(figsize=(10, 6))
    
    # Convert inputs to numpy arrays
    reward_diff = np.array(reward_diff)
    uncertainties = np.array(uncertainties)
    
    # Calculate correctness (1 if reward_diff >= 0, 0 otherwise)
    correct = (reward_diff >= 0).astype(int)
    
    # Create uncertainty bins
    n_bins = 10
    uncertainty_bins = np.percentile(uncertainties, np.linspace(0, 100, n_bins+1))
    
    bin_accuracies = []
    bin_mean_uncertainties = []
    bin_counts = []
    bin_std_accuracies = []
    
    # Calculate accuracy for each uncertainty bin
    for i in range(len(uncertainty_bins)-1):
        mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(correct[mask]))
            bin_mean_uncertainties.append(np.mean(uncertainties[mask]))
            bin_counts.append(np.sum(mask))
            bin_std_accuracies.append(np.std(correct[mask]) / np.sqrt(np.sum(mask)))
    
    bin_accuracies = np.array(bin_accuracies)
    bin_mean_uncertainties = np.array(bin_mean_uncertainties)
    bin_counts = np.array(bin_counts)
    bin_std_accuracies = np.array(bin_std_accuracies)
    
    # Create the plot
    plt.errorbar(bin_mean_uncertainties, bin_accuracies, 
                yerr=bin_std_accuracies, 
                fmt='o-', 
                capsize=5,
                markersize=8,
                label='Accuracy per uncertainty bin')
    
    # Size of points proportional to number of samples
    sizes = 100 * bin_counts / np.max(bin_counts)
    plt.scatter(bin_mean_uncertainties, bin_accuracies, 
               s=sizes, 
               alpha=0.5)
    
    plt.xlabel('Uncertainty')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Uncertainty')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(uncertainties, correct)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation

class RewardBenchDataset(Dataset):
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
        subset = item['subset']

        if self.model_type == "reward":
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
            text_1 = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant_1", "content": chosen},
                {"role": "assistant_2", "content": rejected},
            ], tokenize=False)
            
            text_2 = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant_1", "content": rejected},
                {"role": "assistant_2", "content": chosen},
            ], tokenize=False)

            base_prompt = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
            ], tokenize=False)
            resp_1 = f"<|start_header_id|>assistant_1<|end_header_id|>\n\n{chosen}<|eot_id|>"
            resp_2 = f"<|start_header_id|>assistant_1<|end_header_id|>\n\n{rejected}<|eot_id|>"

            return {
                'text_1': text_1,
                'text_2': text_2,
                'subset': subset,
                'prompt': base_prompt,
                'resp_1': resp_1,
                'resp_2': resp_2,
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

        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")
        self.client_model_id = "us.deepseek.r1-v1:0"

    def collate_fn(self, batch):
        if self.model_type == "reward":
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
            texts_1 = [item['text_1'] for item in batch]
            texts_2 = [item['text_2'] for item in batch]
            subsets = [item['subset'] for item in batch]
            prompts = [item['prompt'] for item in batch]
            resps_1 = [item['resp_1'] for item in batch]
            resps_2 = [item['resp_2'] for item in batch]

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
                'subsets': subsets,
                'prompts': prompts,
                'resps_1': resps_1,
                'resps_2': resps_2,
            }

    def evaluate_batch(self, batch):
        with torch.no_grad():
            if self.model_type == "reward":
                # Original reward model logic...
                pass
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
                threshold = 1.4
                if var_1[0] > threshold: 
                    # print('>>>> before', scores_1)
                    scores_1[0] = self.get_deepseek_r1_response((batch['prompts'][0], batch['resps_1'][0], batch['resps_2'][0]))
                    # print('>>>> after', scores_1)
                if var_2[0] > threshold: 
                    scores_2[0] = self.get_deepseek_r1_response((batch['prompts'][0], batch['resps_2'][0], batch['resps_1'][0]))
                # Average the uncertainties from both directions
                uncertainties = torch.maximum(var_1, var_2)
                return ((scores_1 - scores_2) / 2).cpu(), batch['subsets'], uncertainties.cpu()
            
    def get_deepseek_r1_response(self, messages):
        prompt, response_1, response_2 = messages
        query = f"You are an impartial judge. Here is the context of a conversation between user and assistant: {prompt}.\nPlease compare the responses given by assistant_1 and assistant_2 and decide which one is better.\nAssistant_1: {response_1}\nAssistant_2: {response_2}\nYour answer should be put in <label></label>, 1 indicates assistant_1 is better, 2 indicates assistant_2 is better, 3 indicates they are both good or both bad and you really could not decide which one is better."
        
        # Embed the prompt in DeepSeek-R1's instruction format.
        formatted_prompt = f"""
        <｜begin▁of▁sentence｜><｜User｜>{query}<｜Assistant｜><think>\n
        """

        body = json.dumps({
            "prompt": formatted_prompt,
            "max_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.9,
        })

        try:
            # Invoke the model with the request.
            response = self.client.invoke_model(modelId="us.deepseek.r1-v1:0", body=body)

            # Read the response body.
            model_response = json.loads(response["body"].read())
            
            # Extract choices.
            response = model_response["choices"][0]['text']
            
            label = extract_label(response)
            if label == 1:
                return 2.0
            elif label == 2:
                return -2.0
            return 0.0
        
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.client_model_id}'. Reason: {e}")
            with open("/workspace/log.txt", "w") as file:
                file.write(f"ERROR: Can't invoke '{self.client_model_id}'. Reason: {e}")
            exit(1)

def extract_label(text):
    match = re.search(r'<label>(.*?)</label>', text)
    if match:
        content = match.group(1)
        if content in ['1', '2']:
            return int(content)
    return 0    
        
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
    
    # First subplot: Score distributions with uncertainty
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
    
    # Second subplot: Calibration plot
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


def save_intermediate_results(save_path, total_diffs, total_uncertainties, category_correct, category_total, subset_correct, subset_total):
    """
    Save intermediate results using numpy and pickle
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save total_diffs and uncertainties as numpy arrays
    if torch.is_tensor(total_diffs):
        total_diffs = total_diffs.cpu().numpy()
    elif isinstance(total_diffs, list):
        total_diffs = np.array(total_diffs)
    np.save(os.path.join(save_path, 'total_diffs_rewardbench.npy'), total_diffs)
    np.save(os.path.join(save_path, 'total_uncertainties_rewardbench.npy'), np.array(total_uncertainties))
    
    # Save other statistics using pickle
    statistics = {
        'category_correct': dict(category_correct),
        'category_total': dict(category_total),
        'subset_correct': dict(subset_correct),
        'subset_total': dict(subset_total)
    }
    import pickle
    with open(os.path.join(save_path, 'statistics_rewardbench.pkl'), 'wb') as f:
        pickle.dump(statistics, f)


def load_intermediate_results(results_dir):
    """
    Load intermediate results saved with numpy and pickle
    """
    total_diffs = np.load(os.path.join(results_dir, 'total_diffs_rewardbench.npy'))
    total_uncertainties = np.load(os.path.join(results_dir, 'total_uncertainties_rewardbench.npy'))
    
    import pickle
    with open(os.path.join(results_dir, 'statistics_rewardbench.pkl'), 'rb') as f:
        statistics = pickle.load(f)
    
    return total_diffs, total_uncertainties, statistics


def plot_from_saved_results(results_dir: str):
    """
    Create plots and compute metrics from saved intermediate results
    """
    try:
        # Load intermediate results
        total_diffs, total_uncertainties, statistics = load_intermediate_results(results_dir)
        
        # Verify data sizes
        print(f"Number of comparisons: {len(total_diffs)}")
        print(f"Number of samples per category:")
        for category, total in statistics['category_total'].items():
            print(f"  {category}: {total}")
        
        # Compute results
        results = {
            'overall': {
                'accuracy': sum(statistics['category_correct'].values()) / sum(statistics['category_total'].values()),
                'correct': sum(statistics['category_correct'].values()),
                'total': sum(statistics['category_total'].values())
            },
            'categories': {
                cat: {
                    'accuracy': statistics['category_correct'][cat] / statistics['category_total'][cat],
                    'correct': statistics['category_correct'][cat],
                    'total': statistics['category_total'][cat]
                } for cat in statistics['category_correct'].keys()
            },
            'subsets': {
                subset: {
                    'accuracy': statistics['subset_correct'][subset] / statistics['subset_total'][subset],
                    'correct': statistics['subset_correct'][subset],
                    'total': statistics['subset_total'][subset]
                } for subset in statistics['subset_correct'].keys()
            }
        }
        
        # Create plots
        ece = plot_score_distributions(
            total_diffs,
            uncertainties=total_uncertainties,
            save_path=os.path.join(results_dir, 'score_distributions_rewardbench_replot.png')
        )
        correlation = plot_accuracy_vs_uncertainty(
            total_diffs,
            total_uncertainties,
            save_path=os.path.join(results_dir, 'accuracy_vs_uncertainty_rewardbench_replot.png')
        )
        results['ece'] = ece
        results['uncertainty_correlation'] = correlation
        
        print("\nRecomputed Results:", results)
        return results
    
    except Exception as e:
        print(f"Error loading or plotting results: {str(e)}")
        raise


def evaluate_on_rewardbench(
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
    
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    reward_dataset = RewardBenchDataset(dataset, evaluator.tokenizer, model_type)
    
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

    categories = {
        'chat': ['alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
        'chat_hard': ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst', 
                     'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
        'safety': ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 
                  'xstest-should-respond', 'donotanswer'],
        'reasoning': ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust']
    }
    subset_to_category = {subset: category for category, subsets in categories.items() for subset in subsets}

    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    subset_correct = defaultdict(int)
    subset_total = defaultdict(int)
    total_diffs = []
    total_uncertainties = []
    
    if sampler is not None:
        sampler.set_epoch(0)

    for batch in tqdm(dataloader, disable=(local_rank != 0)):
        if not use_mcd:
            scores, subsets, uncertainties = evaluator.evaluate_batch(batch)
            total_uncertainties.extend(uncertainties.tolist())
        else:
            all_scores = []
            all_uncertainties = []
            for _ in range(n_dropout):
                batch_scores, subsets, batch_uncertainties = evaluator.evaluate_batch(batch)
                all_scores.append(batch_scores)
                all_uncertainties.append(batch_uncertainties)
            scores = torch.stack(all_scores).mean(dim=0)
            uncertainties = torch.stack(all_uncertainties).mean(dim=0)
            total_uncertainties.extend(uncertainties.tolist())

        correct = (scores >= 0).bool()
        total_diffs.extend(scores.cpu().tolist())

        for is_correct, subset in zip(correct, subsets):
            category = subset_to_category[subset]
            if is_correct:
                category_correct[category] += 1
                subset_correct[subset] += 1
            category_total[category] += 1
            subset_total[subset] += 1

    if local_rank != -1:
        world_size = dist.get_world_size()
        all_category_correct = [None] * world_size
        all_category_total = [None] * world_size
        all_subset_correct = [None] * world_size
        all_subset_total = [None] * world_size
        all_total_diffs = [None] * world_size
        all_total_uncertainties = [None] * world_size
        
        dist.all_gather_object(all_category_correct, dict(category_correct))
        dist.all_gather_object(all_category_total, dict(category_total))
        dist.all_gather_object(all_subset_correct, dict(subset_correct))
        dist.all_gather_object(all_subset_total, dict(subset_total))
        dist.all_gather_object(all_total_diffs, total_diffs)
        dist.all_gather_object(all_total_uncertainties, total_uncertainties)
        
        if local_rank == 0:
            category_correct = defaultdict(int)
            category_total = defaultdict(int)
            subset_correct = defaultdict(int)
            subset_total = defaultdict(int)
            total_diffs = []
            total_uncertainties = []
            
            for d in all_category_correct:
                for k, v in d.items():
                    category_correct[k] += v
            for d in all_category_total:
                for k, v in d.items():
                    category_total[k] += v
            for d in all_subset_correct:
                for k, v in d.items():
                    subset_correct[k] += v
            for d in all_subset_total:
                for k, v in d.items():
                    subset_total[k] += v
            for d in all_total_diffs:
                total_diffs.extend(d)
            for d in all_total_uncertainties:
                total_uncertainties.extend(d)

    if local_rank == 0 or local_rank == -1:
        os.makedirs(save_path, exist_ok=True)
        
        # Save intermediate results
        # Save intermediate results
        save_intermediate_results(
            save_path, 
            total_diffs,
            total_uncertainties,
            category_correct, 
            category_total, 
            subset_correct, 
            subset_total
        )
        
        # Compute and save final results
        results = {
            'overall': {
                'accuracy': sum(category_correct.values()) / sum(category_total.values()),
                'correct': sum(category_correct.values()),
                'total': sum(category_total.values())
            },
            'categories': {
                cat: {
                    'accuracy': category_correct[cat] / category_total[cat],
                    'correct': category_correct[cat],
                    'total': category_total[cat]
                } for cat in categories.keys()
            },
            'subsets': {
                subset: {
                    'accuracy': subset_correct[subset] / subset_total[subset],
                    'correct': subset_correct[subset],
                    'total': subset_total[subset]
                } for subset in subset_correct.keys()
            }
        }
        
        os.makedirs(save_path, exist_ok=True)
        ece = plot_score_distributions(
            total_diffs,
            uncertainties=total_uncertainties,
            save_path=os.path.join(save_path, 'score_distributions_rewardbench.png')
        )
        correlation = plot_accuracy_vs_uncertainty(
            total_diffs,
            total_uncertainties,
            save_path=os.path.join(save_path, 'accuracy_vs_uncertainty_rewardbench.png')
        )
        results['ece'] = ece
        results['uncertainty_correlation'] = correlation
        print(results)
        with open(os.path.join(save_path, 'rewardbench.json'), "w") as f:
            json.dump(results, f)
        return results
    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate reward models on RewardBench')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the reward model')
    parser.add_argument('--save_path', type=str, default='./eval_results/rewardbench',
                      help='Path to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for evaluation')
    parser.add_argument('--value_head_prefix', type=str, default="value_head",
                      help='Prefix for the value head in the model')
    parser.add_argument('--model_type', type=str, default="reward",
                      help='Model type, reward or preference')
    parser.add_argument("--use_sn", action="store_true", default=False, help="Enable Spectral Normalization")
    parser.add_argument("--use_gp", action="store_true", default=False, help="Enable Gaussian Process")
    parser.add_argument("--sn_range", type=float, default=10., help="Spectral Normalization Range")
    parser.add_argument("--gp_amplitude", type=float, default=0.1, help="Gaussian Process Amplitude")
    parser.add_argument("--use_mcd", action="store_true", default=False, help="Enable MC Dropout")
    parser.add_argument("--mcd_p", type=float, default=0.2, help="MC Dropout rate")
    parser.add_argument('--n_dropout', type=int, default=3,
                      help='MC dropout count')
    parser.add_argument('--plot_only', action='store_true',
                      help='Only create plots from saved results without running evaluation')
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    args = parser.parse_args()
    
    if args.plot_only:
        results = plot_from_saved_results(args.save_path)
        return
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    
    results = evaluate_on_rewardbench(
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
        print("\nRewardBench Evaluation Results:")
        print(f"\nModel: {args.model_path}")
        print(f"Value head prefix: {args.value_head_prefix}")
        print("\nOverall:")
        print(f"Accuracy: {results['overall']['accuracy']:.2%}")
        print(f"Correct: {results['overall']['correct']}/{results['overall']['total']}")
        
        print("\nBy Category:")
        for category, metrics in results['categories'].items():
            print(f"\n{category.upper()}:")
            print(f"Accuracy: {metrics['accuracy']:.2%}")
            print(f"Correct: {metrics['correct']}/{metrics['total']}")

    if local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
