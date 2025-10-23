import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils import get_tokenizer

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=10):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            if "uncertainties" in response:
                return response.get(score_key), response.get("uncertainties")
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    if isinstance(scores, tuple):
        scores, uncertainties = scores
        uncertainties = torch.tensor(uncertainties)
        scores = torch.tensor(scores)
        return scores, uncertainties
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards"):
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    url = "http://0.0.0.0:5000/get_reward"

    rm_name = '/pretrain/Llama-3-it31-8B-pm-hs_bin-scaled_bt-lr4e-6-ep3-sn1gp10_4096_fullcov'
    tokenizer = get_tokenizer(rm_name, rm_name)
    queries = [["Describe the purpose of internet protocol version 6 (IPv6).\n\n Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."]]
    message = [
        {
            "role": "user",
            "content": "Describe the purpose of internet protocol version 6 (IPv6).\n\n Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        },
        {
            "role": "assistant",
            "content": "this is a test message."
        }
    ]
    message1 = [
        {
            "role": "user",
            "content": "Describe the purpose of internet protocol version 6 (IPv6).\n\n Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        },
        {
            "role": "assistant",
            "content": "this is a test message."
        }
    ]
    message2 = [
        {
            "role": "user",
            "content": "Describe the purpose of internet protocol version 6 (IPv6).\n\n Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        },
        {
            "role": "assistant",
            "content": "IPv6 is an internet protocal that is designed to expand the number of addresses of IPv4."
        }
    ]
    message3 = [
        {
            "role": "user",
            "content": "Describe the purpose of internet protocol version 6 (IPv6).\n\n Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        },
        {
            "role": "assistant",
            "content": "ipv6 is an internet protocal that is designed to expand the number of addresses of ipv4."
        }
    ]

    processed = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    processed1 = tokenizer.apply_chat_template(message1, tokenize=False, add_generation_prompt=False)
    processed2 = tokenizer.apply_chat_template(message2, tokenize=False, add_generation_prompt=False)
    processed3 = tokenizer.apply_chat_template(message3, tokenize=False, add_generation_prompt=False)

    queries = [processed] + [processed1] + [processed2] + [processed3]
    contexts = ["0"] * 4
    queries = queries * 2
    contexts = contexts * 2
    score = remote_rm_fn(url, queries=(contexts, queries))
    # print(score)