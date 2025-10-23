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
from vllm import LLM, SamplingParams


logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text

def extract_json_from_llm_response(response):
    # Try to find JSON-like structure in the response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            # Parse the JSON string
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print('>>>>> Error json_str: ', [json_str])
            return None
    else:
        print("No JSON-like structure found in the response")
        return None

def extract_prompt_and_response(text):
    # Pattern to match the prompt
    prompt_pattern = r'<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>'
    
    # Pattern to match the response
    response_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'

    # Extract prompt
    prompt_match = re.search(prompt_pattern, text, re.DOTALL)
    prompt = prompt_match.group(1) if prompt_match else None

    # Extract response
    response_match = re.search(response_pattern, text, re.DOTALL)
    response = response_match.group(1) if response_match else None

    return prompt, response
# def check_all_instructions_followed(data):
#     if not data or 'instructions' not in data:
#         print("No valid instruction data found")
#         return False

#     # Extract all 'followed' values
#     return data['overall']
#     followed_values = [instruction['followed'] for instruction in data['instructions']]
    
#     # Convert to numpy array
#     followed_array = np.array(followed_values)
    
#     # Check if all are True
#     all_followed = np.all(followed_array)
    
#     return all_followed

class RewardModelProxy:
    def __init__(self, args):
        self.extract_inst_templage = """
Please analyze the following prompt, identify all explicit and verifiable instructions in the prompt, including punctuation, word count, language, format, etc. 

Prompt:
{prompt}

Provide your result as a JSON object with the following structure:
{{
  "instructionCount": <number of instructions identified>,
  "instructions": [
    {{
      "instruction": <short description of the instruction>
    }},
    ...
  ]
}}

Please only include the instructions that are explicit and verifiable. If no such instruction exists, please set \"instructionCount\" to 0.
"""
        self.judge_template = """
Please verify whether the provided response to the prompt follows the verifiable instruction. 

Response:
{response}

Prompt: 
{prompt}

Verifiable instruction: 
{instruction}

Provide your answer as a JSON object with the following structure:
{{
  "instruction": <instruction>
  "response": <response>
  "followed": <boolean>
}}
"""
        self.llm = LLM(
            model=args.pretrain,
            trust_remote_code=True,
            tensor_parallel_size=args.num_gpus,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.reward_model = get_llm_for_sequence_regression(
        #     args.reward_pretrain,
        #     "reward",
        #     normalize_reward=args.normalize_reward,
        #     use_flash_attention_2=args.flash_attn,
        #     bf16=args.bf16,
        #     load_in_4bit=args.load_in_4bit,
        #     value_head_prefix=args.value_head_prefix,
        #     device_map="auto",
        # )
        # self.reward_model.eval()

        # self.tokenizer = get_tokenizer(
        #     args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        # )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.n_samples_per_prompt = args.n_samples_per_prompt

    def batch_inference(self, processed_prompts): 
        judge_messages = [[{"role": "user", "content": prompt}] for prompt in processed_prompts]
        # message = [{"role": "user", "content": judge_prompt}]
        conversations = self.tokenizer.apply_chat_template(judge_messages, tokenize=False)
        outputs = self.llm.generate(
            conversations,
            SamplingParams(
                temperature=0.5,
                top_p=0.9,
                max_tokens=self.max_length,
                stop_token_ids=self.terminators,  # KEYPOINT HERE
            ),
            use_tqdm=False
        )
        responses = [out.outputs[0].text for out in outputs]
        return responses
    
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
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            prompts = []
            responses = []
            for j in range(len(batch_queries)):
                prompt, response = extract_prompt_and_response(batch_queries[j])
                prompts.append(prompt)
                responses.append(response)
            prompts = prompts[::self.n_samples_per_prompt]
            processed_prompts = [self.extract_inst_templage.format(prompt=f"{prompt}") for prompt in prompts]
            prompt_instructions = self.batch_inference(processed_prompts)
            instructions_json = map(extract_json_from_llm_response, prompt_instructions)
            print('>>>>> instructions_json: ', instructions_json[0])

            instructions = [prompt["instructions"] for prompt in instructions_json]

            processed_to_check = []
            for j in range(len(batch_queries)):
                instruction_list = instructions[j//self.n_samples_per_prompt]
                for inst in instruction_list:
                    processed_to_check.append(self.judge_template.format(
                        prompt=f"{prompts[j//self.n_samples_per_prompt]}", 
                        response=f"{responses[j]}",
                        instruction=f"{inst}"
                        )
                    )
            results = self.batch_inference(processed_to_check)
            results_json = map(extract_json_from_llm_response, results)
            print('>>>>> results_json: ', results_json[0])
            results_bool = [res["followed"] for res in results_json]
            for j in range(len(batch_queries)):
                flag = True
                instruction_list = instructions[j//self.n_samples_per_prompt]
                for k in range(len(instruction_list)):
                    flag = flag or results_bool[j*len(instruction_list)+k]
                r = 1. if flag else 0.
                scores.append(r)

        return scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--data_path", type=str, default=None, help="IFEval data path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--noise_type", type=str, default="uniform", help="Noise type")
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
    parser.add_argument("--num_gpus", type=int, default=1)

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
