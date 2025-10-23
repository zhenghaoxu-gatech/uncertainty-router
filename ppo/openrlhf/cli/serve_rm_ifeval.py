import argparse
import re
import Levenshtein

import torch
import numpy as np
import hashlib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from instruction_following_eval import instructions_registry

from instruction_following_eval.evaluation_main import *

import json
from enum import Enum


logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text

def extract_prompt_and_response(text):
    # Pattern to match the prompt
    prompt_pattern = r'<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>'
    # prompt_pattern = r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>'
    
    # Pattern to match the response
    # response_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>'
    response_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'

    # Extract prompt
    prompt_match = re.search(prompt_pattern, text, re.DOTALL)
    prompt = prompt_match.group(1) if prompt_match else None

    # Extract response
    response_match = re.search(response_pattern, text, re.DOTALL)
    response = response_match.group(1) if response_match else None

    return prompt, response

def find_closest_key(dictionary, query):
    if not dictionary:
        return None
    
    closest_key = min(dictionary.keys(), key=lambda k: Levenshtein.distance(k, query))
    return closest_key

def string_to_seed(string):
    # Hash the string using SHA-256
    hash_object = hashlib.sha256(string.encode())
    hash_value = hash_object.hexdigest()

    # Convert the hash value to an integer
    seed = int(hash_value, 16) % (2**32)  # Ensure the seed is within a 32-bit range

    return seed

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

scores_dict = {}

# Process batches and update scores
def process_batch(prompt, batch_scores, batch_scores_rm):
    if prompt not in scores_dict:
        scores_dict[prompt] = {'scores': [], 'scores_rm': []}
    
    scores_dict[prompt]['scores'].extend(batch_scores)
    scores_dict[prompt]['scores_rm'].extend(batch_scores_rm)

class NormalizationType(Enum):
    MEAN_ZERO_SCALE = 1
    SIGMOID = 2
    MIN_MAX = 3
    Z_SCORE = 4

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
        self.prompt_list = []
        with open(args.data_path, "r") as f:
            for l in f:
                example = json.loads(l)
                self.prompt_dict[example["prompt"]] = InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"])
        self.noise_type = args.noise_type
        self.noise0 = args.noise0
        self.noise1 = args.noise1

    def normalize_and_scale_grouped(self, scores_rm, scores, norm_type=NormalizationType.MEAN_ZERO_SCALE):
        arr = np.array(scores_rm)
        scores_arr = np.array(scores)
        n_groups = len(arr) // self.n_samples_per_prompt
        arr_reshaped = arr.reshape(n_groups, self.n_samples_per_prompt)
        scores_reshaped = scores_arr.reshape(n_groups, self.n_samples_per_prompt)
        
        def normalize_group(group, scores_group):
            # Create mask for scores > 0.5
            mask = scores_group > 0.5
            
            # If no scores > 0.5, return original group
            if not np.any(mask):
                return group
            
            # Apply mask to the group
            masked_group = group[mask]
            
            if norm_type == NormalizationType.MEAN_ZERO_SCALE:
                mean = np.mean(masked_group)
                normalized = masked_group - mean
                min_value = np.min(normalized)
                scale_factor = 1 / (-min_value) if min_value < -1 else 1
                result = normalized * scale_factor
            
            elif norm_type == NormalizationType.SIGMOID:    # normalize sigmoid to [-1,1]
                mean = np.mean(masked_group)
                centered = masked_group - mean
                result = 2 / (1 + np.exp(-centered)) - 1
            
            elif norm_type == NormalizationType.MIN_MAX:
                min_val, max_val = np.min(masked_group), np.max(masked_group)
                if min_val == max_val:
                    result = np.zeros_like(masked_group)
                else:
                    result = (masked_group - min_val) / (max_val - min_val)
            
            elif norm_type == NormalizationType.Z_SCORE:
                mean, std = np.mean(masked_group), np.std(masked_group)
                if std == 0:
                    result = np.zeros_like(masked_group)
                else:
                    result = (masked_group - mean) / std
            
            else:
                raise ValueError("Invalid normalization type")
            
            # Create output array and fill in normalized values
            output = np.array(group)
            output[mask] = result
            return output

        # Apply normalization to each group
        result = np.apply_along_axis(normalize_group, 1, arr_reshaped, scores_reshaped)
        
        # Flatten the result back to a 1D array and convert to list
        return result.flatten().tolist()

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
        logger.info(f"queries[0]: {queries[0]}")
        scores = []
        for i in range(0, len(queries)):
            r = 0.
            prompt, response = extract_prompt_and_response(queries[i])
            prompt = find_closest_key(self.prompt_dict, prompt)
            # print('===>', [queries[i], prompt])
            
            inp = self.prompt_dict[prompt]
            instruction_list = inp.instruction_id_list
            check = True
            
            for index, instruction_id in enumerate(instruction_list):
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
                instruction.build_description(**kwargs)
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=inp.prompt)
                if response.strip() and instruction.check_following(response):
                    # r += 1.
                    pass
                else:
                    # r += 0.
                    check = False
                # print('>>>', prompt, response, kwargs, r)
            np.random.seed(string_to_seed(response))
            if self.noise_type == 'uniform':
                eps = np.random.uniform(-self.noise0, self.noise0)
            elif self.noise_type == 'normal':
                eps = np.random.randn()
            else:
                eps = 0.0
            if check:
                r += 1.
                eps *= self.noise1
            else:
                eps *= self.noise0
            scores.append(r+eps)

        if self.noise_type in ['rm', 'rmn', 'rmnn', 'rms']:
            scores_rm = []
            # batch
            with torch.no_grad():
                for i in range(0, len(queries), batch_size):
                    inputs = self.tokenize_fn(
                        queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                    )
                    r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                    r = r.tolist()
                    scores_rm.extend(r)

            
            if self.noise_type == 'rm':
                scores_rm = self.normalize_and_scale_grouped(scores_rm, scores, NormalizationType.MEAN_ZERO_SCALE)
                for i in range(len(scores)):
                    if scores[i] > 0.5:
                        scores[i] += scores_rm[i]
            if self.noise_type == 'rmn':
                for i in range(len(scores)):
                    scores[i] += scores_rm[i]
            if self.noise_type == 'rmnn':
                scores_rm = self.normalize_and_scale_grouped(scores_rm, np.ones(len(scores)).tolist(), NormalizationType.MEAN_ZERO_SCALE)
                for i in range(len(scores)):
                    scores[i] += scores_rm[i]
            if self.noise_type == 'rms':    #sigmoid
                scores_rm = self.normalize_and_scale_grouped(scores_rm, scores, NormalizationType.SIGMOID)
                for i in range(len(scores)):
                    if scores[i] > 0.5:
                        scores[i] += scores_rm[i]


        # print("==== scores ====", scores)
        # print("---- scores_rm ----", scores_rm)
        # process_batch(prompt, scores, scores_rm)
        # np.savez_compressed('scores_data.npz', **scores_dict)
        # import sys
        # sys.exit(4)
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
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--data_path", type=str, default=None, help="IFEval data path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--noise_type", type=str, default="uniform", help="Noise type")
    parser.add_argument("--noise0", type=float, default=0.0, help="Noise scale for r=0")
    parser.add_argument("--noise1", type=float, default=0.0, help="Noise scale for r=1")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--max_len", type=int, default="2048")

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
