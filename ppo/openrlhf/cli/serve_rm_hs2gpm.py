import argparse
import re

import torch
import numpy as np
import hashlib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

from datasets import load_dataset, Dataset, DatasetDict


logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.n_samples_per_prompt = args.n_samples_per_prompt

        self.prompt_dict = {}
        dataset = load_dataset(args.data_path, split=args.data_split)
        
        # Construct context->label mapping
        for item in dataset:
            context = item['context']
            label = item['label']
            # Use context as key since it's unique for each prompt
            if context not in self.prompt_dict:
                self.prompt_dict[context] = label

    def get_reward(self, queries):
        if isinstance(queries, list) and len(queries) == 2 and isinstance(queries[0], list) and isinstance(queries[1], list) and len(queries[0]) == len(queries[1]):
            contexts, queries = queries
        else:
            raise ValueError("Contexts missing for rule-based reward")
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
        logger.info(f"queries[0]: {queries[0]}")
        scores = []
        for i in range(0, len(queries)):
            r = -1.0  # default score for no box
            query = queries[i]
            ground_truth = self.prompt_dict[contexts[i]]
            
            # Find all boxes in the text
            # all_boxes = re.findall(r'\\boxed\{([^}]*)\}', query)
            all_boxes = re.findall(r'<label>(.*?)</label>', query)
            
            if len(all_boxes) >= 2:  # At least 2 boxes (system prompt + response)
                answer = all_boxes[-1].strip()  # Take the last box as the model's answer
                if answer:  # box is not empty
                    # Compare with ground truth
                    if answer == ground_truth:
                        r = 1.0  # correct answer
                    else: 
                        try:
                            num_answer = int(answer)
                            num_ground_truth = int(ground_truth)
                            if np.sign(num_answer) == np.sign(num_ground_truth): 
                                r = 0.5
                            else: 
                                r = -0.5
                        except ValueError as e:
                            r = -1.0    # wrong format
            
            scores.append(r)
        logger.info(f"rewards[0]: {scores[0]}")
        logger.info(f"rewards: {scores}")
        return scores

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to generative preference model dataset")
    parser.add_argument("--data_split", type=str, default="train", help="Data split")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--max_len", type=int, default=2048)

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_samples_per_prompt", type=int, default=1)

    args = parser.parse_args()
    assert args.batch_size % args.n_samples_per_prompt == 0, "n_samples_per_prompt should be divisible by batch_size"

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
