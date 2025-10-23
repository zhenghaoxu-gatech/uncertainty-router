from vllm import LLM, SamplingParams
import os

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name_short = "mix"
model_path_s3 = f"${S3_BUCKET}/pretrain/{model_name}"

# model_name=f"Llama-3-sft-8B-grpo8-ep1-rm-mix-nonorm-hs-bs-256-a-5e-7-c-1e-5-temp-1.0"
# model_name_short = "mix-sft-grpo8"
# model_path_s3 = f"${S3_BUCKET}/checkpoint/{model_name}"

model_path = f"/pretrain/{model_name}"
# os.system(f"aws s3 cp {model_path_s3} {model_path} --region us-east-1 --recursive --exclude *cache/* ")
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

model = LLM(model_path)

from transformers import AutoTokenizer


import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os


ds = load_dataset("nvidia/HelpSteer2")
# train = ds['train'] # len(train) = 20324 (95%)
val = ds['validation']     # len(val) = 1038 (5%)
print('HelpSteer2 data count: ', len(val))

device = "cuda:0"
device_rm = "cuda:1"

# list_score = []

'''
Generate by OpenRLHF/Llama-3-8b-rm-mixture
'''
# model_name = "OpenRLHF/Llama-3-8b-rm-mixture"
# model_name_short = "mix"
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
model_name_short = "sky"
value_head = "score"
model_path_s3 = f"${S3_BUCKET}/pretrain/{model_name}"
model_path = f"/pretrain/{model_name}"
# os.system(f"aws s3 cp {model_path_s3} {model_path} --region us-east-1 --recursive --exclude *cache/*")
import sys
sys.path.append("/home/zxugt/src/workspace/ppo")
from openrlhf.models import get_llm_for_sequence_regression
rm = get_llm_for_sequence_regression(
    model_path,
    "reward",
    use_flash_attention_2=True,
    bf16=False,
    device_map=device_rm,
    # value_head_prefix="value_head",
    value_head_prefix=value_head,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_path)

n_responses = 256

tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(n=n_responses, max_tokens=1024, stop_token_ids=[tokenizer.eos_token_id], temperature=1, skip_special_tokens=True)
list_score = np.zeros((len(val)//2, n_responses + 2))
messages = []

data_gen = []
for i, x in enumerate(val):
    if i > 0:
        break
    if i % 2 == 1: 
        prompt = x['prompt']
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": x['response']}]
        # print(conv)
        conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device_rm)

        # Get the reward scores
        with torch.no_grad():
            attn_mask1 = torch.ones_like(conv_tokenized, device=device_rm)
            score1 = rm(conv_tokenized, attn_mask1).item()
            list_score[i//2, -1] = score1
        continue
    if i // 2 >= 8:  # only run for first two prompts
        break
    prompt = x['prompt']
    prompt="How many i in incomprehensibilities? Please only output one digit as answer and nothing else, including any form of explanations."

    message = [{"role": "user", "content": prompt}]
    formatted_prompt =  tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    print(formatted_prompt)
    output = model.generate(formatted_prompt, 
                            use_tqdm=True, 
                            sampling_params=sampling_params,
                            )
    
    cnt_correct = 0
    cnt_misclassify = 0
    for j in tqdm(range(n_responses)):

        # LLM generate
        response = output[0].outputs[j].text


        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        # print(conv)
        # conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device_rm)
        conv_tokenized = rm_tokenizer(prompt+response, return_tensors="pt")["input_ids"].to(device_rm)

        # Get the reward scores
        with torch.no_grad():
            attn_mask1 = torch.ones_like(conv_tokenized, device=device_rm)
            score1 = rm(conv_tokenized, attn_mask1).item()
            list_score[i//2, j] = score1
            # if score1 > 10.25:
            #     cnt_misclassify += 1
            #     print(f"===score: {score1} ===\n{response}")
            # elif abs(score1 - 10.25) < 1e-4:
            #     cnt_correct += 1
            # else:
            data_gen.append({"prompt": formatted_prompt, "response": response, "reward": score1})

    response = "5"
    
    # conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": x['response']}]
    # conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    # # print(conv)
    # conv_tokenized = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device_rm)
    conv_tokenized = rm_tokenizer(prompt+response, return_tensors="pt")["input_ids"].to(device_rm)

    # Get the reward scores
    with torch.no_grad():
        attn_mask1 = torch.ones_like(conv_tokenized, device=device_rm)
        score1 = rm(conv_tokenized, attn_mask1).item()
        list_score[i//2, -2] = score1
print(list_score)
print(cnt_correct, cnt_misclassify)
print(len(data_gen))

import json
def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

save_to_jsonl(data_gen, "outputs.jsonl")

np.save('results.npy', list_score)

# '''
# Generate by Skywork-Reward-Llama-3.1-8B-v0.2
# '''
# model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
# model_name_short = "sky"
# rm = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=device,
#     attn_implementation="flash_attention_2",
        # value_head=
#     num_labels=1,
# )
# rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
# for i, x in tqdm(enumerate(val), total=len(val)):
#     prompt = x['prompt']
#     response1 = x['response']

#     conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]

#     conv1_tokenized = rm_tokenizer.apply_chat_template(conv1, tokenize=True, return_tensors="pt").to(device)

#     # Get the reward scores
#     with torch.no_grad():
#         score1 = rm(conv1_tokenized).logits[0][0].item()
#         list_score.append(score1)

