import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    action_logits: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.values = to(self.values, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.values = pin_memory(self.values)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
        disable_critic=False,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.disable_critic = disable_critic
        self.perf_stats = None

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str], Tuple[str, str], Tuple[List[str], List[str]]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # sample multiple response

        # if data in the form of (contexts, prompts)
        if args.context_key is not None:
            all_contexts, all_prompts = all_prompts
            all_contexts = sum([[context] * args.n_samples_per_prompt for context in all_contexts], [])
            all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
            experiences = []
            for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
                prompts = all_prompts[i : i + args.micro_rollout_batch_size]
                contexts = all_contexts[i : i + args.micro_rollout_batch_size]
                experiences.append(self.make_experience((contexts, prompts), **generate_kwargs))
        # if data in the form of prompts
        else: 
            all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
            experiences = []
            for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
                prompts = all_prompts[i : i + args.micro_rollout_batch_size]
                experiences.append(self.make_experience(prompts, **generate_kwargs))

        experiences = self.process_experiences(experiences)

        # calculate return and advantages
        style = "response" if "resp" in self.strategy.args.policy_gradient_style else "token"
        add_kl = 0 if "kl" in self.strategy.args.policy_gradient_style else 1
        if self.strategy.args.policy_gradient_style == "pmd":   # pmd uses raw reward for critic learning
            add_kl = 0
        for experience in experiences:
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                experience.info["reward"],
                self.kl_ctl.value*add_kl,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                style=style,
            )
            experience.advantages, experience.returns = self.get_advantages_and_returns(
                experience.values,
                reward,
                experience.action_mask,
                generate_kwargs["gamma"],
                generate_kwargs["lambd"],
                style=style,
            )
            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
        return experiences

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        if not self.disable_critic:
            self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.disable_critic: 
            value = torch.zeros_like(action_log_probs, device="cuda")
        else: 
            value = self.critic(sequences, num_actions, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if not self.disable_critic:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> List[Experience]:
        # TODO: add more methods to process experiences
        args = self.strategy.args
        assert args.policy_gradient_style in ["ppo", "grpo", "rloo", "grpo_resp", "rloo_resp", "vpg_resp", "rloo_resp_kl", "rloo_kl", "pmd"], f"process_experiences() for {args.policy_gradient_style} has not been implemented"
        if args.reward_model_type == "preference": 
            for experience in experiences:
                experience.info["raw_reward"] = experience.info["reward"]
                experience.info["raw_uncertainty"] = experience.info["uncertainty"]
            return experiences
        if "grpo" in args.policy_gradient_style:
            for experience in experiences:
                experience.info["raw_reward"] = experience.info["reward"]
                r = experience.info["reward"].reshape(-1, args.n_samples_per_prompt)
                mean = r.mean(dim=1, keepdim=True)
                std = r.std(dim=1, keepdim=True) + 1e-8
                reward = ((r - mean) / std).reshape(-1)
                experience.info["reward"] = reward
        if "rloo" in args.policy_gradient_style:
            for experience in experiences:
                experience.info["raw_reward"] = experience.info["reward"]
                if "kl" in args.policy_gradient_style:
                    r = (experience.info["reward"] - self.kl_ctl.value*experience.info["kl"]).reshape(-1, args.n_samples_per_prompt)
                else:
                    r = experience.info["reward"].reshape(-1, args.n_samples_per_prompt)
                # leave-one-out reward
                k = args.n_samples_per_prompt
                sum = r.sum(dim=1, keepdim=True)
                reward = ((r * k - sum) / (k-1)).reshape(-1)
                experience.info["reward"] = reward
        if "ppo" in args.policy_gradient_style:
            for experience in experiences:
                experience.info["raw_reward"] = experience.info["reward"]
        if "pmd" in args.policy_gradient_style:
            for experience in experiences:
                experience.info["raw_reward"] = experience.info["reward"]
        if "vpg" in args.policy_gradient_style:
            for experience in experiences:
                experience.info["raw_reward"] = experience.info["reward"]
        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
        style: str = "token",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if style == "response":
            return rewards, rewards
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str], Tuple[str, str], Tuple[List[str], List[str]]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, **generate_kwargs)
        if not self.disable_critic:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str], Tuple[str, str], Tuple[List[str], List[str]]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        args = self.strategy.args
        if args.context_key is not None:
            contexts, prompts = prompts
        # generate sequence
        start = time.time()
        if not self.packing_samples:
            sequences, attention_mask, action_mask = (
                self._generate_local(prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(prompts, **generate_kwargs)
            )
            num_actions = action_mask.size(1)
            packed_seq_lens = None
            response_length = action_mask.float().sum(dim=-1)
            total_length = attention_mask.float().sum(dim=-1)
        else:
            assert self.vllm_engines is not None, "vllm_engines must be provided for packed samples"
            sequences, attention_mask, packed_seq_lens, num_actions = self._generate_vllm(prompts, **generate_kwargs)
            action_mask = None
            response_length = torch.tensor(num_actions, device=device, dtype=torch.float)
            total_length = torch.tensor(packed_seq_lens, device=device, dtype=torch.float)
        generate_time = time.time() - start

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.strategy.args.policy_gradient_style == "pmd":
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens, return_logits=True
            )
        else:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

        # values
        if not self.disable_critic:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.disable_critic:
            ray.get([value_ref])
            ray.get([self.critic.empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            for rm in self.remote_rm_url:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                if args.context_key is not None:
                    r = remote_rm_fn_ray.remote(rm, queries=(contexts, queries))
                else:
                    r = remote_rm_fn_ray.remote(rm, queries=queries)

                r_refs.append(r)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        if self.disable_critic:
            ref_values = ray.get([base_action_log_probs_ref] + r_refs)
            # values and rewards have the same shape
            base_action_log_probs, value, rewards = ref_values[0], torch.zeros_like(ref_values[0]), ref_values[1:]
        else:
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
            if self.strategy.args.policy_gradient_style == "pmd":
                base_action_log_probs, base_action_logits = base_action_log_probs
        if isinstance(rewards[0], tuple):
            _rewards, _uncertainties = zip(*rewards)
            rewards = list(_rewards)
            uncertainties = list(_uncertainties)
        else:
            uncertainties = None
        wait_time = time.time() - start

        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        if self.strategy.args.policy_gradient_style == "pmd":
            base_action_logits = base_action_logits.to(device)
        rewards = [r.to(device) for r in rewards]
        if uncertainties is not None:
            uncertainties = [u.to(device) for u in uncertainties]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        uncertainty = uncertainties[0] if uncertainties is not None else torch.zeros_like(r, device=device)

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()


        # response (sequence) level reward/kl
        if "resp" in self.strategy.args.policy_gradient_style: 
            if not self.packing_samples:
                # TODO
                raise KeyError("Response level reward/kl is not implemented for non-packing samples yet. ")
            else:
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)
                value = unpacking_samples(value, num_actions)

                kl = []
                for i in range(len(action_log_probs)):
                    kl.append(
                        compute_approx_kl(
                            action_log_probs[i],
                            base_action_log_probs[i],
                            action_mask=None,
                            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
                            style="response",
                        )
                    )
                action_log_probs = [logp.sum(dim=-1, keepdim=True) for logp in action_log_probs]
                base_action_log_probs = [logp.sum(dim=-1, keepdim=True) for logp in base_action_log_probs]
                kl_mean = torch.tensor([each_kl for each_kl in kl], device=device)
                value_sums = torch.tensor([x.sum() for x in value], device=device)
                action_log_probs_sums = torch.tensor([x.sum() for x in action_log_probs], device=device)
                base_action_log_probs_sums = torch.tensor([x.sum() for x in base_action_log_probs], device=device)
                ratios = (action_log_probs_sums - base_action_log_probs_sums).exp()
                base_action_logits = None
        else:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )

            if not self.packing_samples:
                kl = masked_mean(kl, action_mask, dim=-1)
                value_sums = value.sum(dim=-1)
                action_log_probs_sums = action_log_probs.sum(dim=-1)
                base_action_log_probs_sums = base_action_log_probs.sum(dim=-1)
                ratios = (action_log_probs_sums - base_action_log_probs_sums).exp()
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)
                value = unpacking_samples(value, num_actions)
                if self.strategy.args.policy_gradient_style == "pmd":
                    base_action_logits = unpacking_samples(base_action_logits, num_actions)
                else:
                    base_action_logits = None

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)
                value_sums = torch.tensor([x.sum() for x in value], device=device)
                action_log_probs_sums = torch.tensor([x.sum() for x in action_log_probs], device=device)
                base_action_log_probs_sums = torch.tensor([x.sum() for x in base_action_log_probs], device=device)
                ratios = (action_log_probs_sums - base_action_log_probs_sums).exp()

        # (B,)
        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": response_length,
            "total_length": total_length,
            "num_actions": num_actions,
            "values": value_sums,
            "action_log_probs": action_log_probs_sums,
            "base_action_log_probs": base_action_log_probs_sums,
            "ratio": ratios,
            "uncertainty": uncertainty,
        }

        if self.strategy.args.perf:
            self.perf_stats["generate_time"] += generate_time
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
            base_action_logits,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        prompt_token_ids = self.tokenize_fn(prompts, self.prompt_max_len, padding=False)["input_ids"]
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        if not self.packing_samples:
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for output in outputs:
                max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[-1]
            sequences = []
            for output in outputs:
                # left padding input
                input_len = len(output.prompt_token_ids)
                input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                # right padding output
                output_len = len(output.outputs[0].token_ids)
                output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                if output_ids[output_len - 1] != eos_token_id:
                    output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = self.actor.process_sequences(
                sequences, max_input_len, eos_token_id, pad_token_id
            )
            return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")
        else:
            # NOTE: concat all outputs to following format:
            #
            # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
            # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[-1]
            sequences = []
            packed_seq_lens = []
            attention_mask = []
            num_actions = []
            for i, output in enumerate(outputs):
                input_len = len(output.prompt_token_ids)
                output_len = len(output.outputs[0].token_ids)
                packed_seq_lens.append(input_len + output_len)
                sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                attention_mask.extend([i + 1] * (input_len + output_len))

                # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                # num_actions.append(max(1, sum(current_action_mask)))
                num_actions.append(max(1, output_len))

            sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
            attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
            return sequences, attention_mask, packed_seq_lens, num_actions

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.disable_critic or self._ref is None:
            return
        ray.get(self._ref)
        self._ref = None
