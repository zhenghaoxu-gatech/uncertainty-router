from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, enable_ratio: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.enable_ratio = enable_ratio

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.enable_ratio: 
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages.clamp(-10., 10.)
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages.clamp(-10., 10.)
            loss = -torch.min(surr1, surr2)
            loss = masked_mean(loss, action_mask, dim=-1).mean()
        else: 
            loss = -log_probs * advantages.clamp(-10., 10.)
            loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss
    
class PMDLoss(nn.Module):
    """
    One-Step Policy Mirror Descent Loss in Logit Space
    """

    def __init__(self, lr) -> None:
        super().__init__()
        self.lr = lr

    def forward(
        self,
        logits: torch.Tensor,
        old_logits: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = (logits - old_logits - self.lr * advantages) ** 2
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss

class PRMLoss(nn.Module):
    pass

class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss

class PreferenceLoss(nn.Module):
    """
    Preference Loss for Reward Model
    """

    def forward(
        self, reward_diff: torch.Tensor, label: torch.Tensor, strength: torch.Tensor
    ) -> torch.Tensor:
        loss = -(F.logsigmoid(reward_diff) * label + F.logsigmoid(-reward_diff) * (1-label)) * strength
        # print('>>>>>>>>>label', label)
        # print('>>>>>>>>>reward_diff', reward_diff)
        # print('>>>>>>>>>lost_first', F.logsigmoid(reward_diff) * label )
        # print('>>>>>>>>>lost_second', F.logsigmoid(-reward_diff) * (1-label))
        
        
        # Create tensor on the same device as input
        half = torch.tensor(0.5, device=reward_diff.device)
        
        # Create mask for valid comparisons (excluding labels close to 0.5)
        valid_mask = ~torch.isclose(label, half, atol=1e-6)
        
        # Calculate accuracy only for valid comparisons
        pred = (reward_diff > 0).float()
        acc = torch.isclose(pred, label, atol=1e-6)[valid_mask].float()
        
        return loss.mean(), acc.mean() if torch.any(valid_mask) else torch.tensor(0.0, device=reward_diff.device)

class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()
    
class ScaledPairWiseLoss(nn.Module):
    """
    Scaled Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            zero_margin_mask = (margin == 0)
            
            regular_loss = -F.logsigmoid(chosen_reward - reject_reward)
            
            smooth_loss_forward = -F.logsigmoid(chosen_reward - reject_reward)
            smooth_loss_reverse = -F.logsigmoid(reject_reward - chosen_reward)
            smooth_loss = 0.5 * (smooth_loss_forward + smooth_loss_reverse)
            
            loss = torch.where(
                zero_margin_mask,
                smooth_loss,
                regular_loss * margin
            )
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()
    
class CenteredPairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def __init__(self, center_coef: float = 0.0) -> None:
        super().__init__()
        self.center_coef = center_coef

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin) + self.center_coef * (chosen_reward + reject_reward) ** 2
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward) + self.center_coef * (chosen_reward + reject_reward) ** 2
        return loss.mean()

class AdaptivePairWiseLoss(nn.Module):
    """
    Adaptive Pairwise Loss for Reward Model https://arxiv.org/pdf/2406.02764
    """
    def __init__(self, 
                 n_tau_iters: int = 3, 
                 rho: float = 0.5,
                 tau_init: float = 1.0, 
                 tau_min: float = 0.1, 
                 tau_max: float = 4.0, 
                 mode: str = "linear", 
    ) -> None:
        super().__init__()
        self.n_tau_iters = n_tau_iters
        self.rho = rho
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.log_two = np.log(2)
        self.mode = mode

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            tau = self.get_optimal_tau((chosen_reward - reject_reward - margin).detach(), self.mode)
            loss = -F.logsigmoid((chosen_reward - reject_reward - margin) / tau)
        else:
            tau = self.get_optimal_tau((chosen_reward - reject_reward).detach(), self.mode)
            loss = -F.logsigmoid((chosen_reward - reject_reward) / tau)
        return loss.mean(), tau
    
    def get_optimal_tau(
        self,
        logits_detach: torch.FloatTensor,
        mode: str
    ) -> torch.FloatTensor:
        tau = torch.ones_like(logits_detach) * self.tau_init
        rho = torch.ones_like(logits_detach) * self.rho
        log_two = torch.ones_like(logits_detach) * self.log_two
        if mode == "linear":
            for i in range(self.n_tau_iters):
                grad_tau = - F.logsigmoid(logits_detach/tau) + (1 - torch.sigmoid(logits_detach/tau)) * (logits_detach/tau) + rho - log_two
                hess_tau = ((logits_detach**2/tau**3) * (1 - torch.sigmoid(logits_detach/tau)) * torch.sigmoid(logits_detach/tau)).clamp(min=1e-6)
                newton_dir = - grad_tau/hess_tau
                tau = (tau + newton_dir).clamp(min=self.tau_min, max=self.tau_max)
        elif mode == "quadratic":
            for i in range(self.n_tau_iters):
                grad_tau = - F.logsigmoid(logits_detach/tau) + (1 - torch.sigmoid(logits_detach/tau)) * (logits_detach/tau) + 2 * rho * tau - log_two
                hess_tau = ((logits_detach**2/tau**3) * (1 - torch.sigmoid(logits_detach/tau)) * torch.sigmoid(logits_detach/tau) + 2 * rho).clamp(min=1e-6)
                newton_dir = - grad_tau/hess_tau
                tau = (tau + newton_dir).clamp(min=self.tau_min, max=1e6)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Should be one of ['linear', 'quadratic']"
            )
        return tau.detach()

class TemperaturePairWiseLoss(nn.Module):
    """
    Pairwise Loss with Temperature for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, temperature: torch.Tensor = None
    ) -> torch.Tensor:
        assert temperature is not None
        loss = -F.logsigmoid((chosen_reward - reject_reward) / temperature)
        return loss.mean()

class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

class CenteredDPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False, center_coef: float = 0.0) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.center_coef = center_coef

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) + self.center_coef * (policy_chosen_logps + policy_rejected_logps - reference_chosen_logps - reference_rejected_logps) ** 2

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss
