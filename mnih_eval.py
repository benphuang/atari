"""Standalone policy evaluation and held-out state sampling."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from mnih_env import make_env

def _run_eval(
    env_id: str,
    seed: int,
    q_network: nn.Module,
    device: torch.device,
    *,
    frame_skip: int,
    frame_pool: str,
    preprocess: str,
    noop_max: int,
    episodic_life: bool,
    fire_reset: bool,
    crop_top: int,
    frame_stack_pad: str,
    eval_steps: int,
    eval_eps: float,
    dump_frames: int = 0,
):
    eval_env = make_env(
        env_id,
        seed + 12345,
        0,
        False,
        "eval",
        network="nature",
        preprocess=preprocess,
        noop_max=0,
        episodic_life=False,
        fire_reset=False,
        frame_skip=frame_skip,
        frame_pool=frame_pool,
        frame_stack_pad=frame_stack_pad,
        crop_top=crop_top,
        dump_frames=dump_frames,
        eval_mode=True,
    )()
    eval_rng = random.Random(seed)
    obs, _ = eval_env.reset(seed=seed + 4321)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    episode_returns: list[float] = []
    current_return = 0.0
    episodes = 0
    partial_episode = False
    n_steps = 0
    while n_steps < eval_steps:
        if eval_rng.random() < eval_eps:
            action = eval_rng.randrange(eval_env.action_space.n)
        else:
            with torch.no_grad():
                q_values = q_network(obs.unsqueeze(0))
                action = int(torch.argmax(q_values).item())
        next_obs, reward, terminated, truncated, info = eval_env.step(action)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        current_return += float(reward)
        n_steps += 1
        if terminated or truncated:
            episode_returns.append(current_return)
            current_return = 0.0
            episodes += 1
            partial_episode = False
            obs, _ = eval_env.reset(seed=seed + 4321 + episodes)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            partial_episode = True
            obs = next_obs

    if episode_returns:
        mean_return = float(np.mean(episode_returns))
        max_return = float(np.max(episode_returns))
        num_episodes = len(episode_returns)
    else:
        mean_return = float(current_return)
        max_return = float(current_return)
        num_episodes = 0
        partial_episode = True
    eval_env.close()
    return {"mean_return": mean_return, "max_return": max_return, "num_episodes": num_episodes, "partial": partial_episode}

def _sample_heldout_states(
    env_id: str,
    seed: int,
    *,
    frame_skip: int,
    frame_pool: str,
    preprocess: str,
    noop_max: int,
    episodic_life: bool,
    fire_reset: bool,
    crop_top: int,
    frame_stack_pad: str,
    steps: int = 2000,
    states: int = 500,
):
    env = make_env(
        env_id,
        seed + 77,
        0,
        False,
        "heldout",
        network="nature",
        preprocess=preprocess,
        noop_max=noop_max,
        episodic_life=episodic_life,
        fire_reset=fire_reset,
        frame_skip=frame_skip,
        frame_pool=frame_pool,
        frame_stack_pad=frame_stack_pad,
        crop_top=crop_top,
        dump_frames=0,
        eval_mode=True,
    )()
    obs, _ = env.reset(seed=seed + 99)
    rng = np.random.default_rng(seed)
    states_over_rollout: list[np.ndarray] = []
    for _ in range(steps):
        states_over_rollout.append(np.asarray(obs, dtype=np.uint8))
        action = int(rng.integers(0, env.action_space.n))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 101)
    sample_indices = np.linspace(0, steps - 1, num=min(states, steps), dtype=np.int64)
    sampled = [states_over_rollout[index] for index in sample_indices]
    env.close()
    return np.stack(sampled, axis=0)

def _log_scalar(writer: SummaryWriter, name: str, value: float, step: int) -> None:
    if np.isfinite(value):
        writer.add_scalar(name, float(value), step)
    else:
        writer.add_scalar(name, 0.0, step)
