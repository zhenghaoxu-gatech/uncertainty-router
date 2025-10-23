import argparse
import re
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from datasets import load_dataset

logger = init_logger(__name__)

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text

rm_list = [
    "/workspace/models/Llama-3-sft-8B-rm-pku-sigmoid-1", 
    "/workspace/models/Llama-3-sft-8B-rm-pku-sigmoid-2", 
    "/workspace/models/Llama-3-sft-8B-rm-pku-sigmoid-3", 
]

class RewardModelProxy:
    def __init__(self, args):
        self.reward_models = []
        for i in range(len(rm_list)):
            self.reward_models.append(
                get_llm_for_sequence_regression(
                    rm_list[i],
                    "reward",
                    normalize_reward=args.normalize_reward,
                    use_flash_attention_2=args.flash_attn,
                    bf16=args.bf16,
                    load_in_4bit=args.load_in_4bit,
                    value_head_prefix=args.value_head_prefix,
                    device_map="cuda:0",
                )
            )
            self.reward_models[-1].eval()

        self.tokenizer = get_tokenizer(
            rm_list[0], rm_list[0], "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )

        rewards_diff = []
        uncertainty_scores = []
        
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)],
                    device=self.reward_models[0].device
                )
                
                rewards = []
                for j in range(len(rm_list)):
                    r = self.reward_models[j](inputs["input_ids"], inputs["attention_mask"])
                    rewards.append(r)
                
                rewards = torch.stack(rewards)  # shape: [num_models, batch_size]
                
                # Calculate mean and std of reward differences
                reward_diff = rewards[:, 0] - rewards[:, 1]  # difference between chosen and rejected
                mean_diff = reward_diff.mean().item()
                std_diff = reward_diff.std().item()
                
                rewards_diff.append(mean_diff)
                uncertainty_scores.append(std_diff)

        return rewards_diff, uncertainty_scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

def process_dataset(reward_model, dataset, num_samples=None):
    all_reward_diffs = []
    all_uncertainties = []
    correct_predictions = 0
    total_samples = 0
    
    # Determine the number of samples to process
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    # Process samples
    for i in tqdm(range(num_samples)):
        prompts = dataset[i]['prompt']
        chosen = dataset[i]['chosen']
        rejected = dataset[i]['rejected']

        chosen_texts = reward_model.tokenizer.apply_chat_template([
            {"role": "user", "content": prompts},
            {"role": "assistant", "content": chosen}
        ], tokenize=False)
        
        rejected_texts = reward_model.tokenizer.apply_chat_template([
            {"role": "user", "content": prompts},
            {"role": "assistant", "content": rejected}
        ], tokenize=False)
        
        reward_diff, uncertainty = reward_model.get_reward([chosen_texts, rejected_texts])
        
        # Accumulate results
        all_reward_diffs.extend(reward_diff)
        all_uncertainties.extend(uncertainty)
        
        # Count correct predictions (chosen reward > rejected reward)
        correct_predictions += int(reward_diff[0] > 0)
        total_samples += 1

    return all_reward_diffs, all_uncertainties, correct_predictions, total_samples

def create_visualization(reward_diffs, uncertainties, accuracy):
    plt.figure(figsize=(10, 6))
    plt.scatter(reward_diffs, uncertainties, alpha=0.5)
    plt.xlabel('Reward Difference (Chosen - Rejected)')
    plt.ylabel('Uncertainty (std)')
    plt.title('Reward Difference vs Uncertainty')
    plt.grid(True)
    
    # Add accuracy information to the plot
    plt.text(0.05, 0.95, f'Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('reward_uncertainty_plot.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--max_len", type=int, default="8192")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (None for all)")

    # Server settings
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=True, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()

    # Initialize reward model
    reward_model = RewardModelProxy(args)
    
    # Load dataset
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    
    # Process dataset
    reward_diffs, uncertainties, correct_predictions, total_samples = process_dataset(
        reward_model, dataset, args.num_samples
    )
    
    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Total accuracy: {accuracy:.4f}")
    
    # Create and save visualization
    create_visualization(reward_diffs, uncertainties, accuracy)
    
    # Save numerical results
    results = {
        'reward_differences': reward_diffs,
        'uncertainties': uncertainties,
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions
    }
    
    np.save('reward_results.npy', results)
