"""DQN network, optimizer, loss, target, and exploration schedule primitives."""

from __future__ import annotations

from typing import Callable, Iterable

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, env: gym.Env, network: str = "nature"):
        super().__init__()
        self.network_name = network
        self.env = env
        if network == "nature":
            conv_layers = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            flat_dim = int(conv_layers(torch.zeros(1, 4, 84, 84)).shape[1])
            self.network = nn.Sequential(
                conv_layers,
                nn.Linear(flat_dim, 512),
                nn.ReLU(),
                nn.Linear(512, env.action_space.n),
            )
        elif network == "arxiv2013":
            conv_layers = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            flat_dim = int(conv_layers(torch.zeros(1, 4, 84, 84)).shape[1])
            assert flat_dim == 2592, f"Expected flattened size 2592, got {flat_dim}"
            self.network = nn.Sequential(
                conv_layers,
                nn.Linear(flat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.n),
            )
        else:
            raise ValueError(f"Unknown network type '{network}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)

class DM_RMSProp(torch.optim.Optimizer):
    """Custom RMSProp variant matching the preprint/Lua implementation."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 2.5e-4):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                m = state["m"]
                v = state["v"]
                m.mul_(0.95).add_(g, alpha=0.05)
                v.mul_(0.95).addcmul_(g, g, value=0.05)
                denom = torch.sqrt(v - m * m + 0.01)
                p.addcdiv_(g, denom, value=-group["lr"])
        return loss

def _resolve_loss(name: str):
    if name == "cleanrl":
        return lambda td_target, old_val: F.mse_loss(td_target, old_val)
    if name == "dm_sq":
        return lambda td_target, old_val: 0.5 * ((td_target - old_val) ** 2).sum()
    if name == "dm_huber":
        return lambda td_target, old_val: F.smooth_l1_loss(td_target, old_val, beta=1.0, reduction="none").sum()
    raise ValueError(f"Unknown loss '{name}'")

def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def linear_schedule_with_anchor(start_e: float, end_e: float, duration: int, t: int, learning_starts: int, anneal_start: str) -> float:
    if anneal_start == "learning_starts":
        if t < learning_starts:
            return start_e
        t = max(t - learning_starts, 0)
        if duration <= 0:
            return end_e
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
    return linear_schedule(start_e, end_e, duration, t)

def _make_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float):
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "dm_rmsprop":
        return DM_RMSProp(params, lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'")

def _compute_td_target(q_network: nn.Module, target_network: nn.Module | None, next_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float, use_target_network: bool):
    if use_target_network and target_network is not None:
        with torch.no_grad():
            next_q = target_network(next_obs)
            target_max, _ = next_q.max(dim=1)
    else:
        with torch.no_grad():
            next_q = q_network(next_obs)
            target_max, _ = next_q.max(dim=1)
    return rewards.flatten() + gamma * target_max * (1.0 - dones.flatten())
