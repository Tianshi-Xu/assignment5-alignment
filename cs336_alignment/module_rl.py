from einops import rearrange
import torch
from typing import Literal
from transformers import PreTrainedTokenizer
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size,
    advantage_eps,
    normalize_by_std,
):
    rewards = []
    total_num = 0
    total_reward = 0
    total_format_reward = 0
    total_answer_reward = 0
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(response, ground_truth)
        rewards.append(reward["reward"])
        total_num += 1
        total_reward += reward["reward"]
        total_format_reward += reward["format_reward"]
        total_answer_reward += reward["answer_reward"]
    rewards = torch.tensor(rewards)
    raw_rewards = rewards.clone()
    metadata = {"total_num": total_num, "total_reward": total_reward, "total_format_reward": total_format_reward, "total_answer_reward": total_answer_reward}
    rewards = rearrange(rewards, '(b g) -> b g', g=group_size)
    rewards = rewards - rewards.mean(dim=-1, keepdim=True)
    if normalize_by_std:
        rewards = rewards / (rewards.std(dim=-1, keepdim=True)+advantage_eps)
    advantages = rewards.flatten()
    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    # print(policy_log_probs.shape, raw_rewards_or_advantages.shape)
    return -(policy_log_probs * raw_rewards_or_advantages.unsqueeze(-1))

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    ratio_clipped  = torch.clamp(ratio, 1-cliprange, 1+cliprange)
    clamped_at_min_count = torch.sum(ratio < 1-cliprange)
    clamped_at_max_count = torch.sum(ratio > 1+cliprange)
    # count the number of clamped at min and max
    meta_data = {
        "clamped_at_min_count": clamped_at_min_count/torch.numel(ratio),
        "clamped_at_max_count": clamped_at_max_count/torch.numel(ratio),
    }
    grpo_loss = -torch.minimum(ratio*advantages, ratio_clipped*advantages)
    return grpo_loss, meta_data

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    meta_data = {}
    if loss_type == "no_baseline":
        rl_loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        rl_loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        rl_loss, meta_data = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    return rl_loss, meta_data

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    tensor = torch.sum(tensor * mask, dim=dim)
    return tensor/torch.sum(mask, dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rl_loss, meta_data = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    rl_loss = masked_mean(rl_loss, response_mask)
    rl_loss = rl_loss / gradient_accumulation_steps
    rl_loss.backward()
    return rl_loss, meta_data