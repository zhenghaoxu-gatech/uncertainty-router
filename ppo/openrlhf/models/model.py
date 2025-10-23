from typing import Optional, Union

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import convert_ring_attn_params
from .utils import reset_position_ids
from .spec_norm import spectral_norm

logger = init_logger(__name__)


# from torch.nn.utils.parametrizations import spectral_norm
import types
from functools import wraps
import math
import torch.nn.functional as F

    
def mcd_wrapper(layer, p=0.2):
    layer.dropout = nn.Dropout(p=p)
    forward = layer.forward

    @wraps(forward)
    def new_forward(self, *args, **kwargs):
        x = forward(*args, **kwargs)
        return (self.dropout(x[0]),)
    
    layer.forward = types.MethodType(new_forward, layer)
    return layer

# def sn_wrapper(layer, sn_range=10.):
#     down_proj = layer.mlp.down_proj
#     forward = down_proj.forward

#     @wraps(forward)
#     def new_forward(self, *args, **kwargs):
#         if not hasattr(self, '_u'): 
#             self.n_power_iterations = 1
#             self.eps = 1e-6
#             self.sn_range = sn_range
#             weight_mat = self.weight
#             h, w = weight_mat.size()
#             u = F.normalize(weight_mat.new_empty(h).normal_(0, 1), dim=0, eps=self.eps)
#             v = F.normalize(weight_mat.new_empty(w).normal_(0, 1), dim=0, eps=self.eps)
#             self.register_buffer('_u', u)
#             self.register_buffer('_v', v)
#             self._power_method(15)

#         if self.training:
#             sigma = self._power_method(self.n_power_iterations)
#         else: 
#             sigma = self._power_method(0)

#         x = forward(*args, **kwargs)
#         if sigma.squeeze().item() < self.sn_range: 
#             return x
#         return x / sigma
#         # return x
    
#     def _power_method(self, n_power_iterations: int):
#         with torch.no_grad():
#             for _ in range(n_power_iterations):
#                 # In-place operations for DataParallel compatibility
#                 self._u = F.normalize(
#                     torch.mv(self.weight, self._v),
#                     dim=0,
#                     eps=self.eps,
#                     out=self._u
#                 )
#                 self._v = F.normalize(
#                     torch.mv(self.weight.T, self._u),
#                     dim=0,
#                     eps=self.eps,
#                     out=self._v
#                 )
#             #     print(torch.vdot(self._u, torch.mv(self.weight, self._v)))
#             # print(torch.linalg.svdvals(self.weight.float()))
#             sigma = torch.vdot(self._u, torch.mv(self.weight, self._v))
#             return sigma
    
#     layer.mlp.down_proj.forward = types.MethodType(new_forward, down_proj)
#     layer.mlp.down_proj._power_method = types.MethodType(_power_method, down_proj)
#     return layer

def PoolLinear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class GPLayer(nn.Module):
    def __init__(
            self, 
            hidden_size: int, 
            ridge: float = 0.001, 
            amplitude: float = 0.1, 
            ema: float = 0.99, 
            packing_samples: bool = False,
            use_sn: bool = False, 
            sn_range: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if use_sn: 
            self.last_pooled_layer = spectral_norm(PoolLinear(self.hidden_size, self.hidden_size, bias=True), norm_bound=sn_range)
        else:
            self.last_pooled_layer = PoolLinear(self.hidden_size, self.hidden_size, bias=True)
        self.rf_size = 4096
        self.rf_layer = nn.Linear(self.hidden_size, self.rf_size, bias=True)
        self.out_layer = nn.Linear(self.rf_size, 1, bias=False)
        self.rf_layer.weight.requires_grad = False
        self.rf_layer.bias.requires_grad = False
        self.ridge = ridge
        self.amplitude = amplitude
        self.ema = ema
        self.register_buffer('inv_cov', torch.zeros(self.rf_size, self.rf_size))
        self.cov = None
        self.coef = math.sqrt(2 / self.rf_size)
        self.packing_samples = packing_samples

    def forward(self, input, attention_mask = None, position_ids = None, packed_seq_lens = None, last_epoch = False):
        if self.packing_samples: 
            packed_seq_lens = torch.tensor(packed_seq_lens, device=input.device)
            eos_indices = packed_seq_lens.cumsum(dim=0) - 1
            input = input.squeeze(0)[eos_indices]   # (B, H)
        else:
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            input = input.gather(dim=1, index=eos_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)).squeeze(1)
        
        phi = torch.cos(self.rf_layer(input)) * self.coef
        logit = self.out_layer(phi)
        if last_epoch:
            # p = torch.sigmoid(logit.float()) * torch.sigmoid(-logit.float())
            self.inv_cov = self.inv_cov + torch.matmul(phi.T.float(), phi.float())
            # print(self.inv_cov.norm())
        return logit.squeeze(-1)
    
    def predict(self, input, attention_mask = None, position_ids = None, packed_seq_lens = None):
        """
        Output reward difference (pre-sigmoid)
        """
        if self.packing_samples: 
            packed_seq_lens = torch.tensor(packed_seq_lens, device=input.device)
            eos_indices = packed_seq_lens.cumsum(dim=0) - 1
            input = input.squeeze(0)[eos_indices]   # (B, H)
        
        else:
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            input = input.gather(dim=1, index=eos_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)).squeeze(1)
            
        if self.cov is None:
            self.inv_cov = self.inv_cov + torch.eye(self.rf_size, device=self.inv_cov.device) * self.ridge
            self.inv_cov = (self.inv_cov+self.inv_cov.mT)/2
            self.cov = "marked"

        phi = torch.cos(self.rf_layer(input)) * self.coef
        logit = self.out_layer(phi)
        # var = (torch.matmul(phi.double(), self.cov) * phi.double()).sum(dim=1, keepdim=True)

        var = (torch.linalg.solve(self.inv_cov, phi.t().float()).t() * phi.float()).sum(dim=1, keepdim=True) * self.ridge
        # print(var)
        # exit()
        # denom = torch.sqrt(1 + math.pi * var / 8)
        denom = torch.sqrt(1 + self.amplitude * var)

        logit = logit.float() / denom
        if (logit.isnan().any()): 
            print('>>>>>>>>>> logit', logit.isnan().any(), logit)
            print('>>>>>>>>>> phi', phi.isnan().any(), phi)
            print('>>>>>>>>>> input', input.isnan().any(), input)
            print('>>>>>>>>>> var', var.isnan().any(), denom)
            print('>>>>>>>>>> denominator', denom.isnan().any(), denom)
        # if torch.min(var) < 0:
        #     exit()

        return logit.squeeze(-1), denom.squeeze(-1)
    
    def reset_parameters(self) -> None:
        self.rf_layer.weight.data.normal_(mean=0.0, std=0.05)
        self.rf_layer.bias.data.uniform_(0, 2*math.pi)
        # self.out_layer.weight.data.normal_(mean=0.0, std=1 / (self.rf_size + 1))
        self.out_layer.weight.data.normal_(mean=0.0, std=0.05)

# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    packing_samples=False,
    use_sn=False,
    sn_range=10.,
    use_mcd=False,
    mcd_p=0.2,
    use_gp=False,
    gp_amplitude=0.1,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
        model_type in ["critic", "reward", "preference"]
    ), f"invalid model_type: {model_type}, should be critic or reward or preference."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    # print(config)

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples, use_mcd, mcd_p)
        elif model_type == "preference":
            cls_class = _get_preference_model(base_pretrained_class, base_class, value_head_prefix, packing_samples, use_sn, sn_range, use_mcd, mcd_p, use_gp, gp_amplitude)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples, use_mcd, mcd_p)
        elif model_type == "preference":
            cls_class = _get_preference_model(base_pretrained_class, base_class, value_head_prefix, packing_samples, use_sn, sn_range, use_mcd, mcd_p, use_gp, gp_amplitude)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )
    print(model)

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        if model_type == "preference":
            if dschf is not None:
                logger.info("initialize value_head for ZeRO-3 reward model training.")
                with deepspeed.zero.GatheredParameters([getattr(model, value_head_prefix).weight], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        getattr(model, value_head_prefix).weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            elif use_gp:
                getattr(model, value_head_prefix).reset_parameters()
            else:
                getattr(model, value_head_prefix).weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            # print(f'>>>>>>>>>>init weight avg: {value_head_prefix}, === ', getattr(model, value_head_prefix).weight.data)
            # exit()
        else: 
            if dschf is not None:
                logger.info("initialize value_head for ZeRO-3 reward model training.")
                with deepspeed.zero.GatheredParameters([getattr(model, value_head_prefix).weight], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        getattr(model, value_head_prefix).weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            else:
                getattr(model, value_head_prefix).weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False, use_mcd=False, mcd_p=0.2):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            logger.info(f'Use MC Dropout: {use_mcd}')
            if use_mcd:
                # print(type(getattr(self, self.base_model_prefix)))
                for i in range(len(getattr(self, self.base_model_prefix).layers)): 
                    getattr(self, self.base_model_prefix).layers[i] = mcd_wrapper(getattr(self, self.base_model_prefix).layers[i], mcd_p)

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    # reset the positions for packed samples
                    position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                if ring_attn_group is not None:
                    reward = all_gather(values, ring_attn_group).reshape(1, -1)
                else:
                    reward = values
                # TODO: convert packed_seq_lens into torch tensor in advance
                packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                reward = reward.squeeze(0).gather(dim=0, index=eos_indices)
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel

def _get_preference_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False, use_sn=False, sn_range=10., use_mcd=False, mcd_p=0.2, use_gp=False, gp_amplitude=0.1):
    class PreferenceModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples
            self.value_head_prefix = value_head_prefix
            self.use_gp = use_gp
            self.use_sn = use_sn
            self.use_mcd = use_mcd
            self.mcd_p = mcd_p

            logger.info(f'Use Spectral Normalization: {use_sn}, {sn_range}')

            logger.info(f'Use MC Dropout: {use_mcd}, {mcd_p}')
            if use_mcd:
                # print(type(getattr(self, self.base_model_prefix)))
                for i in range(len(getattr(self, self.base_model_prefix).layers)): 
                    getattr(self, self.base_model_prefix).layers[i] = mcd_wrapper(getattr(self, self.base_model_prefix).layers[i], mcd_p)


            logger.info(f'Use Gaussian Process: {use_gp}, {gp_amplitude}')
            if self.use_gp:
                setattr(self, value_head_prefix, GPLayer(
                    config.hidden_size, 
                    amplitude=gp_amplitude, 
                    packing_samples=self.packing_samples, 
                    use_sn=use_sn, 
                    sn_range=sn_range
                ))
                # setattr(self, value_head_prefix).weight.requires_grad=False
                # setattr(self, value_head_prefix).bias.requires_grad=False
            else:
                setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=None,
            last_epoch=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    # reset the positions for packed samples
                    position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            if not self.use_gp: 
                values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
                if self.packing_samples:
                    if ring_attn_group is not None:
                        reward_diff = all_gather(values, ring_attn_group).reshape(1, -1)
                    else:
                        reward_diff = values
                    # TODO: convert packed_seq_lens into torch tensor in advance
                    packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                    eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                    reward_diff = reward_diff.squeeze(0).gather(dim=0, index=eos_indices)
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward_diff = values.gather(dim=1, index=eos_indices).squeeze(1)
            else:
                reward_diff = getattr(self, self.value_head_prefix)(last_hidden_states, attention_mask, position_ids, packed_seq_lens, last_epoch)

            if not self.training and self.normalize_reward:
                reward_diff = (reward_diff - self.mean) / self.std
            # print('>>>>>> Reward Difference', reward_diff)

            return (reward_diff, outputs) if return_output else reward_diff
        
        def predict(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            if not self.use_gp: 
                reward_diff = self.forward(input_ids, attention_mask, return_output, ring_attn_group, packed_seq_lens)
                return reward_diff, torch.zeros_like(reward_diff)
            
            else: 
                if not self.packing_samples:
                    # https://github.com/OpenRLHF/OpenRLHF/issues/217
                    position_ids = attention_mask.long().cumsum(-1) - 1
                else:
                    if ring_attn_group is not None:
                        input_ids, attention_mask, position_ids = convert_ring_attn_params(
                            input_ids, attention_mask, packed_seq_lens, ring_attn_group
                        )
                    else:
                        # reset the positions for packed samples
                        position_ids = reset_position_ids(attention_mask)
                position_ids.masked_fill_(attention_mask == 0, 1)

                outputs = getattr(self, self.base_model_prefix)(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                )
                last_hidden_states = outputs["last_hidden_state"]
                reward_diff, uncertainty = getattr(self, self.value_head_prefix).predict(last_hidden_states, attention_mask, position_ids, packed_seq_lens)

                # if self.packing_samples:
                #     if ring_attn_group is not None:
                #         reward_diff = all_gather(values, ring_attn_group).reshape(1, -1)
                #     else:
                #         reward_diff = values
                #     # TODO: convert packed_seq_lens into torch tensor in advance
                #     packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                #     eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                #     reward_diff = reward_diff.squeeze(0).gather(dim=0, index=eos_indices)
                # else:
                #     eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                #     reward_diff = values.gather(dim=1, index=eos_indices).squeeze(1)

                if not self.training and self.normalize_reward:
                    reward_diff = (reward_diff - self.mean) / self.std
                # print('>>>>>> Reward Difference', reward_diff)

            return (reward_diff, outputs) if return_output else (reward_diff, uncertainty)

    return PreferenceModel

def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)[:, :-1]

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if num_actions is None:
                assert return_output
                return outputs

            if not self.packing_samples:
                action_values = values[:, -num_actions:]
            else:
                assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
                action_values = []
                offset = 0
                for num_action, seq_len in zip(num_actions, packed_seq_lens):
                    start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                    action_values.append(values[:, start:end])
                    offset += seq_len
                action_values = torch.cat(action_values, dim=1)

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
