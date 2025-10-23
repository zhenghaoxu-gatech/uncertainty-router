import argparse
import re
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch.multiprocessing as mp

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
import datetime

import numpy as np
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import json
from datasets import load_dataset
from openrlhf.utils.ifeval.ground_truth_utils import (
    verify_ifeval_sample,
)
import random
from multiprocessing import Pool
from tqdm import tqdm
import logging

def extract_label(text):
    match = re.search(r'<label>(.*?)</label>', text)
    if match:
        content = match.group(1)
        if content in ['1', '2']:
            return int(content)
    return 0

logger = init_logger(__name__)
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Returns False for messages you want to filter out
        return "Setting ds_accelerator to cuda" not in record.getMessage()
logger.addFilter(MessageFilter())

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(rank)  # Important: set device for each process

def cleanup():
    dist.destroy_process_group()

def generate_symmetric_coordinates(shape, target_size):
    assert shape[-1] == shape[-2], "Last two dimensions must be equal"
    assert target_size % 2 == 0, "Target size must be even to maintain symmetry"
    batch_size, n, _ = shape
    
    # Generate all possible symmetric pairs
    all_coords = []
    for b in range(batch_size):
        for i in range(n):
            for j in range(i + 1, n):  # This ensures i < j
                all_coords.append([b, i, j])
                all_coords.append([b, j, i])
    
    all_coords = np.array(all_coords)
    
    # Randomly select pairs to match target size
    num_pairs = target_size // 2
    pair_indices = np.random.choice(len(all_coords) // 2, num_pairs, replace=False)
    
    # Convert pair indices to coordinate indices
    coords = []
    for idx in pair_indices:
        coords.extend(all_coords[2*idx:2*idx+2])
    
    return np.array(coords)

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text

def get_deepseek_r1_response(message):
    if len(message) == 4:
        prompt, response_1, response_2, region_seed = message
    else:
        prompt, response_1, response_2 = message
        region_seed = random.randint(0, 1)
    # query = f"You are an impartial judge. Here is the context of a conversation between user and assistant: {prompt}.\nPlease compare the responses given by assistant_1 and assistant_2 and decide which one is better.\nAssistant_1: {response_1}\nAssistant_2: {response_2}\nYour answer should be put in <label></label>, 1 indicates assistant_1 is better, 2 indicates assistant_2 is better, 3 indicates they are both good or both bad and you really could not decide which one is better. Please do not let the display order affect your judgement."
    
    query = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is superior. "
        # "Begin your evaluation by thinking through the problem step by step if the answer is not obvious to you. "
        "Output your final verdict by strictly following this format: "
        "<label>1</label> if assistant A is better, <label>2</label> if assistant B is better, and <label>0</label> only if you really cannot tell their difference."
        "[The Start of Context]\n"
        "{prompt}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response_1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response_2}\n"
        "[The End of Assistant B's Response]"
    ).format(prompt=prompt, response_1=response_1, response_2=response_2)
    # print(query)

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
        regions = ["us-east-1", "us-west-2"]
        config = Config(
            retries={
                'max_attempts': 100, 
                'mode': 'standard'
            }
        )
        client = boto3.client("bedrock-runtime", region_name=regions[region_seed % 2], config=config)
        client_model_id = "us.deepseek.r1-v1:0"
        response = client.invoke_model(modelId="us.deepseek.r1-v1:0", body=body)
        
        model_response = json.loads(response["body"].read())
        response = model_response["choices"][0]['text']
        
        label = extract_label(response)
        client.close()
        if label == 1:
            return 2.0
        elif label == 2:
            return -2.0
        elif label == 0:
            return 0.0
        return None
    
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{client_model_id}'. Reason: {e}")
        with open("/workspace/log.txt", "a") as file:
            file.write(f"ERROR: Can't invoke '{client_model_id}'. Reason: {e}\n")
        return None

class PreferenceModelProxy:
    def __init__(self, args, rank):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.preference_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "preference", 
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            use_sn=args.use_sn, 
            sn_range=args.sn_range, 
            use_gp=args.use_gp,
            gp_amplitude=args.gp_amplitude, 
            use_mcd=args.use_mcd, 
            mcd_p=args.mcd_p,
        ).to(self.device)

        self.preference_model = DDP(self.preference_model, device_ids=[rank])

        if args.use_mcd:
            self.preference_model.train()
        else:
            self.preference_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.preference_model.module, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.n_samples_per_prompt = args.n_samples_per_prompt
        self.scale_factor = args.scale_factor
        self.world_size = args.world_size
        self.threshold = args.threshold
        self.verifier = args.verifier
        self.verifier_path = args.verifier_path
        self.verifier_data = load_dataset("allenai/RLVR-IFeval", split='train')
        self.model_type = args.model_type
        self.router = args.router
        self.tot_trigger = 0
        self.tot = 0
    
    def get_reward(self, queries):
        try:
            if not queries:
                return None, None if self.rank == 0 else (None, None)
            contexts = None
            if isinstance(queries, list) and len(queries) == 2 and isinstance(queries[0], list) and isinstance(queries[1], list) and len(queries[0]) == len(queries[1]):
                contexts, queries = queries
            elif self.verifier is not None:
                raise ValueError("Contexts missing for rule-based reward")
            
            if self.rank == 0:
                logger.info(f"queries[0]: {queries[0]}")

            # rule-based reward filter
            if self.verifier is not None:
                ground_truth_idxs = [int(c) for c in contexts]
                ground_truth = self.verifier_data[ground_truth_idxs]['ground_truth']
                responses = self.extract_responses(queries=queries)
                verified = []
                for i in range(len(queries)):
                    res = verify_ifeval_sample(responses[i], ground_truth[i])
                    if isinstance(res, tuple):
                        res = res[0]
                    verified.append(res)
                # verified = [verify_ifeval_sample(responses[i], ground_truth[i]) for i in range(len(queries))]
                verified_idx = np.where(verified)[0]    # idx of passed responses
                verified_queries = [queries[idx] for idx in verified_idx.tolist()]
                verified_cnt = np.array(verified).astype(int).reshape(-1, self.n_samples_per_prompt).sum(axis=1).tolist()    # verified responses for each prompt
            else:
                verified_queries = queries
                verified_cnt = [self.n_samples_per_prompt] * (len(queries) // self.n_samples_per_prompt)

            if self.model_type == "verifier": 
                rewards = np.array(verified).astype(float).reshape(-1, self.n_samples_per_prompt)
                sum = rewards.sum(axis=1, keepdims=True)
                advantages = ((rewards * self.n_samples_per_prompt - sum) / (self.n_samples_per_prompt-1)).flatten().tolist()
                uncertainties = [1.0] * len(advantages)
                if self.rank == 0:
                    return advantages, uncertainties
                else:
                    return None, None

            processed_queries, processed_tuples, processed_prompts, processed_responses = self.process_queries(verified_queries, verified_cnt)
            
            n_queries = len(processed_queries)
            queries_per_gpu = (n_queries + self.world_size - 1) // self.world_size
            start_idx = self.rank * queries_per_gpu
            end_idx = min(start_idx + queries_per_gpu, n_queries)
            
            logger.info(f"Rank {self.rank}: Processing queries from index {start_idx} to {end_idx}")
            
            local_queries = processed_queries[start_idx:end_idx]
            local_tuples = processed_tuples[start_idx:end_idx]
            local_scores = []
            local_uncertainties = []
            
            if local_queries:
                # neural preference model
                with torch.no_grad():
                    for i in range(0, len(local_queries), self.batch_size or len(local_queries)):
                        batch_queries = local_queries[i:min(len(local_queries), i + (self.batch_size or len(local_queries)))]
                        inputs = self.tokenize_fn(batch_queries, device=self.device)
                        r, u = self.preference_model.module.predict(inputs["input_ids"], inputs["attention_mask"])
                        r = r.tolist()
                        u = u.tolist()
                        local_scores.extend(r)
                        local_uncertainties.extend(u)
            
            # Gather results
            gathered_scores = [None for _ in range(self.world_size)]
            gathered_uncertainties = [None for _ in range(self.world_size)]
            
            dist.all_gather_object(gathered_scores, local_scores)
            dist.all_gather_object(gathered_uncertainties, local_uncertainties)

            if self.rank == 0:
                # Combine results only on rank 0
                scores = []
                raw_uncertainties = []
                for gpu_scores, gpu_uncertainties in zip(gathered_scores, gathered_uncertainties):
                    scores.extend(gpu_scores)
                    raw_uncertainties.extend(gpu_uncertainties)

                n_groups = len(queries) // self.n_samples_per_prompt

                if self.verifier is not None: 
                    idx_min = 0
                    advantages = []
                    uncertainties = []
                    for b in range(n_groups):
                        if verified_cnt[b] == 0:    # all responses failed
                            group_advantages = [0.] * self.n_samples_per_prompt
                            group_uncertainties = [1.] * self.n_samples_per_prompt
                            advantages.extend(group_advantages)
                            uncertainties.extend(group_uncertainties)

                        else: 
                            verified_pairs = verified_cnt[b] * (verified_cnt[b] - 1)
                            group_verified_scores = np.array(scores[idx_min:idx_min+verified_pairs]).reshape(verified_cnt[b], verified_cnt[b] - 1)
                            group_verified_pairwise_scores = np.zeros((verified_cnt[b], verified_cnt[b]))
                            group_verified_uncertainties = np.array(raw_uncertainties[idx_min:idx_min+verified_pairs]).reshape(verified_cnt[b], verified_cnt[b] - 1)
                            group_verified_pairwise_uncertainties = np.zeros((verified_cnt[b], verified_cnt[b]))
                            # logger.info(f'group_verified_scores >>>>> {group_verified_scores}')

                            for i in range(verified_cnt[b]):
                                group_verified_pairwise_scores[i, :i] = group_verified_scores[i, :i]
                                group_verified_pairwise_scores[i, i+1:] = group_verified_scores[i, i:]
                                group_verified_pairwise_uncertainties[i, :i] = group_verified_uncertainties[i, :i]
                                group_verified_pairwise_uncertainties[i, i+1:] = group_verified_uncertainties[i, i:]

                            group_verified = np.array(verified)[b*self.n_samples_per_prompt : (b+1)*self.n_samples_per_prompt]
                            group_verified_idx = np.where(group_verified>0)[0].tolist()    # idx of passed responses
                            group_unverified_idx = np.where(group_verified<=0)[0].tolist()    # idx of not passed responses
                            # logger.info(f'>>>>> group_unverified_idx >>>>> {group_unverified_idx}')

                            group_pairwise_scores = np.zeros((self.n_samples_per_prompt, self.n_samples_per_prompt))
                            group_pairwise_uncertainties = np.zeros((self.n_samples_per_prompt, self.n_samples_per_prompt))
                            group_pairwise_scores[group_unverified_idx, :] = -2.0
                            # print('scores -2 >>>>>>>>>> ', group_pairwise_scores)
                            group_pairwise_scores[:, group_unverified_idx] = 2.0
                            # print('scores + 2>>>>>>>>>> ', group_pairwise_scores)
                            group_pairwise_uncertainties[group_unverified_idx, :] = 1.0
                            group_pairwise_uncertainties[:, group_unverified_idx] = 1.0
                            group_pairwise_scores[np.ix_(group_verified_idx, group_verified_idx)] = group_verified_pairwise_scores
                            group_pairwise_uncertainties[np.ix_(group_verified_idx, group_verified_idx)] = group_verified_pairwise_uncertainties

                            # print('scores >>>>>>>>>> ', group_pairwise_scores)
                            # print('uncertainties >>>>>>>>>> ', group_pairwise_uncertainties)

                            group_pairwise_scores = (group_pairwise_scores - np.transpose(group_pairwise_scores)) / 2
                            group_pairwise_uncertainties = (group_pairwise_uncertainties + np.transpose(group_pairwise_uncertainties)) / 2

                            group_advantages = (group_pairwise_scores.sum(axis=1) / (self.n_samples_per_prompt - 1) * self.scale_factor).flatten().tolist()
                            group_uncertainties = (group_pairwise_uncertainties.sum(axis=1) / (self.n_samples_per_prompt - 1)).flatten().tolist()
                            
                            advantages.extend(group_advantages)
                            uncertainties.extend(group_uncertainties)
                            idx_min += verified_pairs
                        # logger.info(f'>>>>>>>>>>>> processed group: {b}\t advantages: {advantages}')
                    return advantages, uncertainties
                        
                else: 
                    if len(scores) > 0:
                        self.tot += len(scores) // 2
                        scores = np.array(scores).reshape(-1, self.n_samples_per_prompt, self.n_samples_per_prompt-1)
                        pairwise_scores = np.zeros((n_groups, self.n_samples_per_prompt, self.n_samples_per_prompt))
                        uncertainties = np.array(raw_uncertainties).reshape(-1, self.n_samples_per_prompt, self.n_samples_per_prompt-1)
                        pairwise_uncertainties = np.zeros((n_groups, self.n_samples_per_prompt, self.n_samples_per_prompt))
                        for b in range(n_groups):
                            for i in range(self.n_samples_per_prompt):
                                pairwise_scores[b, i, :i] = scores[b, i, :i]
                                pairwise_scores[b, i, i+1:] = scores[b, i, i:]
                                pairwise_uncertainties[b, i, :i] = uncertainties[b, i, :i]
                                pairwise_uncertainties[b, i, i+1:] = uncertainties[b, i, i:]
                            pairwise_scores[b] = (pairwise_scores[b] - np.transpose(pairwise_scores[b])) / 2
                            pairwise_uncertainties[b] = (pairwise_uncertainties[b] + np.transpose(pairwise_uncertainties[b])) / 2
                        
                        # send uncertain pairs to R1
                        # default indices for uncertainty-based routing
                        indices = np.argwhere(pairwise_uncertainties > self.threshold)
                        if len(indices) > 0:
                            if self.router == "random": 
                                indices = generate_symmetric_coordinates(pairwise_uncertainties.shape, len(indices))

                            messages = []
                            for idx in indices:
                                b, i, j = idx[0], idx[1], idx[2]
                                if i < j:   # only need to process one side by symmetry
                                    prompt = processed_prompts[b]
                                    resp_1 = processed_responses[b][i]
                                    resp_2 = processed_responses[b][j]
                                    messages.append((prompt, resp_1, resp_2))

                            num_processes = min(64, len(messages)) 
                            with Pool(processes=num_processes) as pool:
                                r1_results = list(tqdm(pool.imap(get_deepseek_r1_response, messages), total=len(messages), desc=f"Calling judge..."))
                            self.tot_trigger += len(messages)
                            
                            ptr = 0
                            for idx in indices:
                                b, i, j = idx[0], idx[1], idx[2]
                                if i < j:
                                    pairwise_scores[b, i, j] = r1_results[ptr]
                                    pairwise_scores[b, j, i] = -pairwise_scores[b, i, j]
                                    ptr += 1

                        logger.info(f"Rank {self.rank}: Total comparisons {self.tot}, judge triggered {self.tot_trigger}")
                        pairwise_scores = pairwise_scores.sum(axis=2) / (self.n_samples_per_prompt - 1)
                        pairwise_uncertainties = pairwise_uncertainties.sum(axis=2) / (self.n_samples_per_prompt - 1)
                        advantages = (pairwise_scores * self.scale_factor).flatten().tolist()
                        uncertainties = pairwise_uncertainties.flatten().tolist()
                        
                        return advantages, uncertainties
                    return [], []
            return None, None

        except Exception as e:
            logger.error(f"Error in get_reward on rank {self.rank}: {e}")
            return None, None

    def extract_responses(self, queries):
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        responses = []
        for seq in queries: 
            last_assistant_pos = seq.rfind("<|start_header_id|>assistant<|end_header_id|>")
            response_start = last_assistant_pos + len("<|start_header_id|>assistant<|end_header_id|>\n\n")
            response_end = seq.find("<|eot_id|>", response_start)
            response = seq[response_start:response_end].strip()
            responses.append(response)
        return responses
    
    def process_queries(self, queries, cnt):
        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )

        # return different processed data formats for convenience
        processed_queries = []
        processed_tuples = []
        processed_prompts = []  
        processed_responses = []
        idx_min = 0
        for i in range(len(cnt)): 
            if cnt[i] == 0:
                processed_prompts.append("")
                processed_responses.append([])
                continue
            group = queries[idx_min:idx_min+cnt[i]]
            # Extract base prompt and responses from the group
            base_prompt = None
            responses = []
            responses_with_template = []
            
            for seq in group:
                # Find the last assistant header position
                last_assistant_pos = seq.rfind("<|start_header_id|>assistant<|end_header_id|>")
                
                if base_prompt is None:
                    # Extract the base prompt (everything before the last assistant message)
                    base_prompt = seq[:last_assistant_pos]
                
                # Extract response (everything between the last assistant header and eot_id)
                response_start = last_assistant_pos + len("<|start_header_id|>assistant<|end_header_id|>\n\n")
                response_end = seq.find("<|eot_id|>", response_start)
                response = seq[response_start:response_end].strip()
                responses.append(response)
                responses_with_template.append(f"<|start_header_id|>assistant_1<|end_header_id|>\n\n{response}<|eot_id|>")

            processed_prompts.append(base_prompt)
            processed_responses.append(responses_with_template)

            for j in range(len(responses)):
                for k in range(len(responses)):
                    if j != k:
                        # Create full sequences for both arrangements
                        seq = base_prompt + f"<|start_header_id|>assistant_1<|end_header_id|>\n\n{responses[j]}<|eot_id|>" + \
                            f"<|start_header_id|>assistant_2<|end_header_id|>\n\n{responses[k]}<|eot_id|>"
                        processed_queries.append(seq)
                        processed_tuples.append((base_prompt, f"<|start_header_id|>assistant_1<|end_header_id|>\n\n{responses[j]}<|eot_id|>", f"<|start_header_id|>assistant_2<|end_header_id|>\n\n{responses[k]}<|eot_id|>"))
            idx_min += cnt[i]
        return processed_queries, processed_tuples, processed_prompts, processed_responses
                            

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

def run_server(rank, world_size, args):
    setup(rank, world_size)
    args.world_size = world_size
    
    preference_model = PreferenceModelProxy(args, rank)
    
    if rank == 0:
        app = FastAPI()

        @app.post("/get_reward")
        async def get_reward(request: Request):
            try:
                data = await request.json()
                queries = data.get("query")
                if not queries:
                    return JSONResponse({"error": "No queries provided"})
                
                # First broadcast a signal that new data is coming
                signal = ["NEW_DATA"]
                try:
                    dist.broadcast_object_list(signal, src=0)
                    # Then broadcast the actual data
                    dist.broadcast_object_list([queries], src=0)
                except Exception as e:
                    logger.error(f"Broadcast error on rank 0: {e}")
                    return JSONResponse({"error": "Failed to broadcast queries"})
                
                rewards, uncertainties = preference_model.get_reward(queries)
                
                if rewards is None:
                    return JSONResponse({"error": "Failed to process queries"})
                
                result = {"rewards": rewards, "uncertainties": uncertainties}
                logger.info(f"Processed queries: {len(queries)}, Rewards: {rewards}")
                
                # Signal completion
                signal = ["DONE"]
                dist.broadcast_object_list(signal, src=0)
                
                return JSONResponse(result)
                
            except Exception as e:
                logger.error(f"Error on rank 0: {str(e)}")
                # Signal error
                signal = ["ERROR"]
                try:
                    dist.broadcast_object_list(signal, src=0)
                except:
                    pass
                return JSONResponse({"error": str(e)})

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        while True:
            try:
                # Wait for signal
                signal = [None]
                dist.broadcast_object_list(signal, src=0)
                
                if signal[0] == "NEW_DATA":
                    # Receive actual data
                    queries_list = [None]
                    dist.broadcast_object_list(queries_list, src=0)
                    queries = queries_list[0]
                    
                    logger.info(f"Rank {rank} received queries, length: {len(queries)}")
                    
                    # Process queries
                    preference_model.get_reward(queries)
                    
                    # Wait for completion signal
                    signal = [None]
                    dist.broadcast_object_list(signal, src=0)
                    
                elif signal[0] == "ERROR" or signal[0] == "DONE":
                    continue
                else:
                    logger.warning(f"Rank {rank} received unknown signal: {signal[0]}")
                    
            except Exception as e:
                logger.error(f"Error on rank {rank}: {e}")
                torch.cuda.empty_cache()  # Try to clean up GPU memory
                continue

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Preference Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Preference Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")
    parser.add_argument('--model_type', type=str, default="preference", help='Model type, reward or preference')
    parser.add_argument("--use_sn", action="store_true", default=False, help="Enable Spectral Normalization")
    parser.add_argument("--sn_range", type=float, default=10., help="Spectral Normalization Range")
    parser.add_argument("--use_gp", action="store_true", default=False, help="Enable Gaussian Process")
    parser.add_argument("--gp_amplitude", type=float, default=0.1, help="Gaussian Process Amplitude")
    parser.add_argument("--use_mcd", action="store_true", default=False, help="Enable MC Dropout")
    parser.add_argument("--mcd_p", type=float, default=0.2, help="MC Dropout rate")
    parser.add_argument('--n_dropout', type=int, default=3, help='MC dropout count')
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Global preference scaling factor")

    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(),
                      help="Number of GPUs to use")
    
    # Routing
    parser.add_argument("--threshold", type=float, default=1.4, help="Uncertainty threshold for routing to Generative PM")
    parser.add_argument("--verifier", type=str, default=None, help="Verifier")
    parser.add_argument("--verifier_path", type=str, default=None, help="Verifier path")
    parser.add_argument("--router", type=str, default="uncertainty", help="Routing strategy: 'uncertainty' or 'random'")
    
    args = parser.parse_args()
    
    # Launch processes
    mp.spawn(
        run_server,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
