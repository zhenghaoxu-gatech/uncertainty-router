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

logger = init_logger(__name__)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600))
    torch.cuda.set_device(rank)  # Important: set device for each process

def cleanup():
    dist.destroy_process_group()

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text


class PreferenceModelProxy:
    def __init__(self, args, rank):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.preference_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            args.model_type, 
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

    def get_reward(self, queries):
        try:
            if not queries:
                return None, None if self.rank == 0 else (None, None)

            processed_queries = self.process_queries(queries)
            
            n_queries = len(processed_queries)
            queries_per_gpu = (n_queries + self.world_size - 1) // self.world_size
            start_idx = self.rank * queries_per_gpu
            end_idx = min(start_idx + queries_per_gpu, n_queries)
            
            logger.info(f"Rank {self.rank}: Processing queries from index {start_idx} to {end_idx}")
            
            local_queries = processed_queries[start_idx:end_idx]
            local_scores = []
            local_uncertainties = []
            
            if local_queries:
                with torch.no_grad():
                    for i in range(0, len(local_queries), self.batch_size or len(local_queries)):
                        batch_queries = local_queries[i:min(len(local_queries), i + (self.batch_size or len(local_queries)))]
                        inputs = self.tokenize_fn(batch_queries, device=self.device)
                        r, u = self.preference_model.module.predict(inputs["input_ids"], inputs["attention_mask"])
                        local_scores.extend(r.tolist())
                        local_uncertainties.extend(u.tolist())
            
            # Gather results
            gathered_scores = [None for _ in range(self.world_size)]
            gathered_uncertainties = [None for _ in range(self.world_size)]
            
            dist.all_gather_object(gathered_scores, local_scores)
            dist.all_gather_object(gathered_uncertainties, local_uncertainties)

            if self.rank == 0:
                # Combine results only on rank 0
                scores = []
                uncertainties = []
                for gpu_scores, gpu_uncertainties in zip(gathered_scores, gathered_uncertainties):
                    scores.extend(gpu_scores)
                    uncertainties.extend(gpu_uncertainties)

                if len(scores) > 0:
                    scores = np.array(scores).reshape(-1, self.n_samples_per_prompt, self.n_samples_per_prompt-1)
                    n_groups = len(queries) // self.n_samples_per_prompt
                    pairwise_scores = np.zeros((n_groups, self.n_samples_per_prompt, self.n_samples_per_prompt))

                    uncertainties = np.array(uncertainties).reshape(-1, self.n_samples_per_prompt, self.n_samples_per_prompt-1)
                    pairwise_uncertainties = np.zeros((n_groups, self.n_samples_per_prompt, self.n_samples_per_prompt))
                    
                    for b in range(n_groups):
                        for i in range(self.n_samples_per_prompt):
                            pairwise_scores[b, i, :i] = scores[b, i, :i]
                            pairwise_scores[b, i, i+1:] = scores[b, i, i:]
                            pairwise_uncertainties[b, i, :i] = uncertainties[b, i, :i]
                            pairwise_uncertainties[b, i, i+1:] = uncertainties[b, i, i:]
                        pairwise_scores[b] = (pairwise_scores[b] - np.transpose(pairwise_scores[b])) / 2
                        pairwise_uncertainties[b] = (pairwise_uncertainties[b] + np.transpose(pairwise_uncertainties[b])) / 2
                    
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

    
    def process_queries(self, queries):
        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        processed_queries = []
        for i in range(0, len(queries), self.n_samples_per_prompt):
                group = queries[i:i + self.n_samples_per_prompt]
                
                # Extract base prompt and responses from the group
                base_prompt = None
                responses = []
                
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

                for j in range(len(responses)):
                    for k in range(len(responses)):
                        if j != k:
                            # Create full sequences for both arrangements
                            seq = base_prompt + f"<|start_header_id|>assistant_1<|end_header_id|>\n\n{responses[j]}<|eot_id|>" + \
                                f"<|start_header_id|>assistant_2<|end_header_id|>\n\n{responses[k]}<|eot_id|>"
                            processed_queries.append(seq)
        return processed_queries
                            

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
    
    args = parser.parse_args()
    
    # Launch processes
    mp.spawn(
        run_server,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
