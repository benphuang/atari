from __future__ import annotations

import copy
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, SupportsFloat

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import ale_py
from gymnasium import spaces
from torch.utils.tensorboard.writer import SummaryWriter

from atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from buffers import ReplayBuffer, ReplayBufferSamples

if hasattr(gym, "register_envs"):
    gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Utilities and defaults
# ---------------------------------------------------------------------------

def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


CLEANRL_DEFAULTS = {
    "env_id": "BreakoutNoFrameskip-v4",
    "total_timesteps": 10_000_000,
    "learning_rate": 1e-4,
    "num_envs": 1,
    "buffer_size": 1_000_000,
    "gamma": 0.99,
    "tau": 1.0,
    "target_network_frequency": 1000,
    "batch_size": 32,
    "start_e": 1.0,
    "end_e": 0.01,
    "exploration_steps": 1_000_000,
    "learning_starts": 80_000,
    "train_frequency": 4,
    "network": "nature",
    "loss": "cleanrl",
    "optimizer": "adam",
    "use_target_network": True,
    "frame_skip": 4,
    "frame_pool": "max",
    "preprocess": "cleanrl",
    "noop_max": 30,
    "episodic_life": True,
    "fire_reset": True,
    "frame_stack_pad": "repeat",
    "crop_top": 18,
    "dump_frames": 0,
    "eval_frequency": 50_000,
    "eval_steps": 10_000,
    "eval_eps": 0.05,
    "heldout_frequency": 10_000,
    "replay_storage": "stacked",
    "anneal_start": "zero",
    "preset": "cleanrl",
}

ARXIV2013_DEFAULTS = {
    "env_id": "BreakoutNoFrameskip-v4",
    "total_timesteps": 10_000_000,
    "learning_rate": 2.5e-4,
    "num_envs": 1,
    "buffer_size": 1_000_000,
    "gamma": 0.99,
    "tau": 1.0,
    "target_network_frequency": 1000,
    "batch_size": 32,
    "start_e": 1.0,
    "end_e": 0.1,
    "exploration_steps": 1_000_000,
    "learning_starts": 50_000,
    "train_frequency": 1,
    "network": "arxiv2013",
    "loss": "dm_sq",
    "optimizer": "dm_rmsprop",
    "use_target_network": False,
    "frame_skip": -1,
    "frame_pool": "last",
    "preprocess": "arxiv2013",
    "noop_max": 0,
    "episodic_life": False,
    "fire_reset": False,
    "frame_stack_pad": "zeros",
    "crop_top": 18,
    "dump_frames": 0,
    "eval_frequency": 50_000,
    "eval_steps": 10_000,
    "eval_eps": 0.05,
    "heldout_frequency": 10_000,
    "replay_storage": "stacked",
    "anneal_start": "zero",
    "preset": "arxiv2013",
}

NATURE2015_DEFAULTS = {
    "env_id": "BreakoutNoFrameskip-v4",
    "total_timesteps": 50_000_000,
    "learning_rate": 2.5e-4,
    "num_envs": 1,
    "buffer_size": 1_000_000,
    "gamma": 0.99,
    "tau": 1.0,
    "target_network_frequency": 10_000,
    "batch_size": 32,
    "start_e": 1.0,
    "end_e": 0.1,
    "exploration_steps": 1_000_000,
    "learning_starts": 50_000,
    "train_frequency": 4,
    "network": "nature",
    "loss": "dm_huber",
    "optimizer": "dm_rmsprop",
    "use_target_network": True,
    "frame_skip": 4,
    "frame_pool": "max",
    "preprocess": "cleanrl",
    "noop_max": 30,
    "episodic_life": True,
    "fire_reset": False,
    "frame_stack_pad": "repeat",
    "crop_top": 18,
    "dump_frames": 0,
    "eval_frequency": 50_000,
    "eval_steps": 10_000,
    "eval_eps": 0.05,
    "heldout_frequency": 10_000,
    "replay_storage": "stacked",
    "anneal_start": "learning_starts",
    "preset": "nature2015",
}


PRESETS = {"cleanrl": CLEANRL_DEFAULTS, "arxiv2013": ARXIV2013_DEFAULTS, "nature2015": NATURE2015_DEFAULTS}


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "on"}
    return bool(v)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = CLEANRL_DEFAULTS["env_id"]
    total_timesteps: int = CLEANRL_DEFAULTS["total_timesteps"]
    learning_rate: float = CLEANRL_DEFAULTS["learning_rate"]
    num_envs: int = CLEANRL_DEFAULTS["num_envs"]
    buffer_size: int = CLEANRL_DEFAULTS["buffer_size"]
    gamma: float = CLEANRL_DEFAULTS["gamma"]
    tau: float = CLEANRL_DEFAULTS["tau"]
    target_network_frequency: int = CLEANRL_DEFAULTS["target_network_frequency"]
    batch_size: int = CLEANRL_DEFAULTS["batch_size"]
    start_e: float = CLEANRL_DEFAULTS["start_e"]
    end_e: float = CLEANRL_DEFAULTS["end_e"]
    exploration_steps: int = CLEANRL_DEFAULTS["exploration_steps"]
    learning_starts: int = CLEANRL_DEFAULTS["learning_starts"]
    train_frequency: int = CLEANRL_DEFAULTS["train_frequency"]
    network: str = CLEANRL_DEFAULTS["network"]
    loss: str = CLEANRL_DEFAULTS["loss"]
    optimizer: str = CLEANRL_DEFAULTS["optimizer"]
    use_target_network: bool = CLEANRL_DEFAULTS["use_target_network"]
    frame_skip: int = CLEANRL_DEFAULTS["frame_skip"]
    frame_pool: str = CLEANRL_DEFAULTS["frame_pool"]
    preprocess: str = CLEANRL_DEFAULTS["preprocess"]
    noop_max: int = CLEANRL_DEFAULTS["noop_max"]
    episodic_life: bool = CLEANRL_DEFAULTS["episodic_life"]
    fire_reset: bool = CLEANRL_DEFAULTS["fire_reset"]
    frame_stack_pad: str = CLEANRL_DEFAULTS["frame_stack_pad"]
    crop_top: int = CLEANRL_DEFAULTS["crop_top"]
    dump_frames: int = CLEANRL_DEFAULTS["dump_frames"]
    eval_frequency: int = CLEANRL_DEFAULTS["eval_frequency"]
    eval_steps: int = CLEANRL_DEFAULTS["eval_steps"]
    eval_eps: float = CLEANRL_DEFAULTS["eval_eps"]
    heldout_frequency: int = CLEANRL_DEFAULTS["heldout_frequency"]
    """agent steps between held-out Q estimates; 0 disables held-out evaluation"""
    replay_storage: Literal["stacked", "frames"] = CLEANRL_DEFAULTS["replay_storage"]
    """replay storage layout: full stacked observations or individual frames"""
    anneal_start: str = CLEANRL_DEFAULTS["anneal_start"]
    preset: str = CLEANRL_DEFAULTS["preset"]

    def resolved(self) -> "Args":
        return apply_preset(copy.deepcopy(self))


def _explicit_fields(argv: list[str] | None = None) -> set[str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    seen: set[str] = set()
    i = 0
    while i < len(argv):
        item = argv[i]
        if item.startswith("--"):
            name = item[2:]
            if name.startswith("no-"):
                name = name[3:]
            if "=" in name:
                name = name.split("=", 1)[0]
            seen.add(name.replace("-", "_"))
            if name in {"preset", "exp_name", "env_id"} and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
        i += 1
    return seen


def apply_preset(args: Args, explicit_fields: set[str] | None = None) -> Args:
    """Apply a preset while preserving any parameters explicitly passed on the CLI."""
    if args.preset not in PRESETS:
        raise ValueError(f"Unknown preset '{args.preset}'. Valid presets: {sorted(PRESETS)}")

    explicit = _explicit_fields() if explicit_fields is None else explicit_fields
    defaults = PRESETS[args.preset]
    original_defaults = PRESETS["cleanrl"]
    live = copy.deepcopy(args)

    for field_name, value in defaults.items():
        if not hasattr(live, field_name):
            continue
        current = getattr(live, field_name)
        if field_name == "preset":
            setattr(live, field_name, args.preset)
            continue
        if field_name in explicit:
            continue
        if current == original_defaults.get(field_name, current):
            setattr(live, field_name, value)
    return live


def _auto_frame_skip(env_id: str) -> int:
    env_name = str(env_id).lower()
    if "spaceinvaders" in env_name:
        return 3
    return 4


class FrameStackPad(gym.Wrapper):
    """Frame stacking wrapper that preserves episode-start padding semantics.

    The Lua implementation pads the first stacked state with repeated observations or zeros.
    """

    def __init__(self, env: gym.Env, stack_size: int = 4, mode: str = "repeat") -> None:
        super().__init__(env)
        if mode not in {"repeat", "zeros"}:
            raise ValueError(f"Unknown frame-stack padding mode '{mode}'.")
        self.stack_size = stack_size
        self.mode = mode
        self.frames: list[np.ndarray] = []
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Box):
            shape = (stack_size, *obs_space.shape)
            self.observation_space = spaces.Box(low=obs_space.low.min(), high=obs_space.high.max(), shape=shape, dtype=obs_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs)
        if self.mode == "zeros":
            self.frames = [np.zeros_like(obs) for _ in range(self.stack_size - 1)] + [obs]
        else:
            self.frames = [obs for _ in range(self.stack_size)]
        return self._stacked(), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs)
        self.frames.append(obs)
        if len(self.frames) > self.stack_size:
            self.frames = self.frames[-self.stack_size :]
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        stack = np.stack(self.frames, axis=0)
        return stack


class FrameReplayBuffer:
    """Replay buffer storing one image per transition and reconstructing frame stacks."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        device: torch.device,
        frame_stack_pad: str,
    ) -> None:
        if len(observation_space.shape) != 3 or observation_space.shape[0] != 4:
            raise ValueError(f"Frame storage requires observations shaped (4, H, W), got {observation_space.shape}.")
        if buffer_size < 5:
            raise ValueError("Frame replay buffer size must be at least 5.")
        if frame_stack_pad not in {"repeat", "zeros"}:
            raise ValueError(f"Unknown frame-stack padding mode '{frame_stack_pad}'.")
        self.buffer_size = buffer_size
        self.device = device
        self.frame_stack_pad = frame_stack_pad
        self.frames = np.zeros((buffer_size, *observation_space.shape[1:]), dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size, 1), dtype=action_space.dtype)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.episode_starts = np.zeros(buffer_size, dtype=np.bool_)
        self.truncated_next_frames: dict[int, np.ndarray] = {}
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        *,
        episode_start: bool = False,
    ) -> None:
        index = self.pos
        next_index = (index + 1) % self.buffer_size
        self.truncated_next_frames.pop(index, None)
        self.frames[index] = np.asarray(obs)[0, -1]
        self.episode_starts[index] = episode_start
        self.actions[index] = np.asarray(action).reshape(1)
        self.rewards[index] = float(np.asarray(reward).reshape(-1)[0])
        self.dones[index] = float(np.asarray(done).reshape(-1)[0])
        self.frames[next_index] = np.asarray(next_obs)[0, -1]
        if infos and infos[0].get("_truncated", False):
            self.truncated_next_frames[index] = np.asarray(next_obs)[0, -1].copy()

        self.pos = next_index
        if self.pos == 0:
            self.full = True

    def _valid_indices(self) -> np.ndarray:
        if not self.full:
            candidates = range(self.pos)
        else:
            excluded = {(self.pos + offset) % self.buffer_size for offset in range(4)}
            candidates = (index for index in range(self.buffer_size) if index not in excluded)
        valid = []
        for index in candidates:
            try:
                self._get_stack(index)
            except ValueError:
                continue
            valid.append(index)
        return np.asarray(valid, dtype=np.int64)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        valid = self._valid_indices()
        if valid.size == 0:
            raise ValueError("Not enough valid transitions to sample from frame replay buffer.")
        return self._get_samples(np.random.choice(valid, size=batch_size, replace=True))

    def _get_samples(self, indices: np.ndarray) -> ReplayBufferSamples:
        observations = np.stack([self._get_stack(index) for index in indices])
        next_observations = []
        for index, obs_stack in zip(indices, observations):
            next_index = (int(index) + 1) % self.buffer_size
            if self.dones[index] and self._is_valid_index(next_index) and self.episode_starts[next_index]:
                next_stack = self._get_stack(next_index)
            else:
                next_frame = self.truncated_next_frames.get(int(index), self.frames[next_index])
                next_stack = np.concatenate((obs_stack[1:], next_frame[None]), axis=0)
            next_observations.append(next_stack)

        return ReplayBufferSamples(
            torch.as_tensor(observations, device=self.device),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(np.stack(next_observations), device=self.device),
            torch.as_tensor(self.dones[indices].reshape(-1, 1), device=self.device),
            torch.as_tensor(self.rewards[indices].reshape(-1, 1), device=self.device),
        )

    def _is_valid_index(self, index: int) -> bool:
        if self.full:
            return index != self.pos and index not in {
                (self.pos + offset) % self.buffer_size for offset in range(1, 4)
            }
        return index < self.pos

    def _get_stack(self, index: int) -> np.ndarray:
        history = [self.frames[index]]
        cursor = index
        while len(history) < 4:
            if self.episode_starts[cursor]:
                break
            previous = (cursor - 1) % self.buffer_size
            if not self._is_valid_index(previous):
                raise ValueError(f"Frame history for replay index {index} has been overwritten.")
            history.append(self.frames[previous])
            cursor = previous
        history.reverse()
        missing = 4 - len(history)
        if missing:
            if self.frame_stack_pad == "zeros":
                padding = [np.zeros_like(history[0]) for _ in range(missing)]
            else:
                padding = [history[0] for _ in range(missing)]
            history = padding + history
        return np.stack(history, axis=0)


class TruncationAwareReplayBuffer(ReplayBuffer):
    """Keep true final observations for truncated transitions in optimized storage."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.truncated_next_observations: dict[int, np.ndarray] = {}

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        index = self.pos
        self.truncated_next_observations.pop(index, None)
        if infos and infos[0].get("_truncated", False):
            self.truncated_next_observations[index] = np.asarray(next_obs)[0].copy()
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        samples = super()._get_samples(batch_inds)
        overrides = [self.truncated_next_observations.get(int(index)) for index in batch_inds]
        if any(observation is not None for observation in overrides):
            next_observations = samples.next_observations.clone()
            for batch_index, observation in enumerate(overrides):
                if observation is not None:
                    next_observations[batch_index] = torch.as_tensor(observation, device=self.device)
            samples = samples._replace(next_observations=next_observations)
        return samples


class SkipEnv(gym.Wrapper):
    """Frame-skip wrapper with reward accumulation and configurable pooling.

    This matches the DQN frame-skip semantics from the preprint: repeat the action a fixed
    number of times, sum rewards, stop early on done, and either keep the last frame or the
    max of the last two observations.
    """

    def __init__(self, env: gym.Env, skip: int = 4, pool: str = "max") -> None:
        super().__init__(env)
        self.skip = max(1, int(skip))
        self.pool = pool
        assert env.observation_space.dtype is not None
        assert env.observation_space.shape is not None
        if pool not in {"max", "last"}:
            raise ValueError(f"Unknown frame pool '{pool}'. Expected 'max' or 'last'.")

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        executed_frames: list[np.ndarray] = []
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            executed_frames.append(np.asarray(obs))
            if terminated or truncated:
                break

        if self.pool == "max":
            obs = np.maximum.reduce(executed_frames[-2:])
        else:
            obs = executed_frames[-1]
        return obs, total_reward, terminated, truncated, info


class ArxivPreprocessEnv(gym.ObservationWrapper):
    """Mnih-style RGB -> grayscale -> resize -> crop preprocessing."""

    def __init__(self, env: gym.Env, crop_top: int = 18) -> None:
        super().__init__(env)
        self.crop_top = crop_top
        self.height = 84
        self.width = 84
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box observation space, got {env.observation_space}"
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)

    def observation(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV is required for the Mnih preprocessing pipeline. Install opencv-python-headless.")
        frame = np.asarray(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_LINEAR)
        cropped = resized[self.crop_top : self.crop_top + 84, :]
        return cropped.astype(np.uint8)


class CleanRLPreprocessEnv(gym.ObservationWrapper):
    """CleanRL's explicit RGB resize-then-grayscale preprocessing pipeline."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box observation space, got {env.observation_space}"
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV is required for Atari preprocessing. Install opencv-python-headless.")
        resized = cv2.resize(np.asarray(frame), (84, 84), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.uint8)


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


def _print_memory_warning(buffer_size: int, replay_storage: str = "stacked") -> None:
    channels = 4 if replay_storage == "stacked" else 1
    approx_bytes = buffer_size * 84 * 84 * channels
    try:
        import psutil
    except ImportError:
        available_bytes = None
    else:
        available_bytes = psutil.virtual_memory().available
    if available_bytes is None or approx_bytes > available_bytes:
        available = "unknown" if available_bytes is None else f"{available_bytes / (1024 ** 3):.1f} GiB"
        print(
            f"Warning: replay storage '{replay_storage}' needs approximately "
            f"{approx_bytes / (1024 ** 3):.1f} GiB; currently available memory is {available}. "
            "The requested buffer size will not be reduced."
        )


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    *,
    network: str = "nature",
    preprocess: str = "cleanrl",
    noop_max: int = 30,
    episodic_life: bool = True,
    fire_reset: bool = True,
    frame_skip: int = 4,
    frame_pool: str = "max",
    frame_stack_pad: str = "repeat",
    crop_top: int = 18,
    dump_frames: int = 0,
    eval_mode: bool = False,
):
    """Construct an Atari environment using the requested preprocessing recipe."""

    def thunk():
        if capture_video and idx == 0 and not eval_mode:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if frame_skip < 0:
            frame_skip_eff = _auto_frame_skip(env_id)
        else:
            frame_skip_eff = frame_skip

        if noop_max > 0 and not eval_mode:
            env = NoopResetEnv(env, noop_max=noop_max)
        if frame_skip_eff > 0:
            env = SkipEnv(env, skip=frame_skip_eff, pool=frame_pool)
        if episodic_life and not eval_mode:
            env = EpisodicLifeEnv(env)
        if fire_reset and "FIRE" in env.unwrapped.get_action_meanings() and not eval_mode:
            env = FireResetEnv(env)

        if not eval_mode:
            env = ClipRewardEnv(env)

        if preprocess == "arxiv2013":
            env = ArxivPreprocessEnv(env, crop_top=crop_top)
        else:
            env = CleanRLPreprocessEnv(env)

        if dump_frames > 0:
            env = DumpFrameEnv(env, dump_limit=dump_frames)

        env = FrameStackPad(env, stack_size=4, mode=frame_stack_pad)

        env.action_space.seed(seed)
        return env

    return thunk


class DumpFrameEnv(gym.Wrapper):
    """Utility wrapper to save a small number of processed frames for inspection."""

    def __init__(self, env: gym.Env, dump_limit: int = 1, out_dir: str = "dump_frames") -> None:
        super().__init__(env)
        self.dump_limit = dump_limit
        self.out_dir = out_dir
        self.count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def step(self, action: int):
        if cv2 is None:
            raise ImportError("OpenCV is required for frame dumping. Install opencv-python-headless.")
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.count < self.dump_limit:
            path = os.path.join(self.out_dir, f"frame_{self.count:04d}.png")
            cv2.imwrite(path, np.asarray(obs)[..., 0] if obs.ndim == 3 and obs.shape[-1] == 1 else np.asarray(obs))
            self.count += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if cv2 is None:
            raise ImportError("OpenCV is required for frame dumping. Install opencv-python-headless.")
        obs, info = self.env.reset(**kwargs)
        if self.count < self.dump_limit:
            path = os.path.join(self.out_dir, f"frame_{self.count:04d}.png")
            cv2.imwrite(path, np.asarray(obs)[..., 0] if obs.ndim == 3 and obs.shape[-1] == 1 else np.asarray(obs))
            self.count += 1
        return obs, info


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args = apply_preset(args)
    assert args.num_envs == 1, "vectorized envs are not supported"
    if args.replay_storage not in {"stacked", "frames"}:
        raise ValueError(f"Unknown replay storage '{args.replay_storage}'. Expected 'stacked' or 'frames'.")
    print(f"Resolved config: {vars(args)}")

    if args.loss == "cleanrl" and args.optimizer == "dm_rmsprop":
        print("Warning: dm_rmsprop with the cleanrl mean-squared loss is not the original DeepMind setup.")

    _print_memory_warning(args.buffer_size, args.replay_storage)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s\n|gymnasium|%s|\n|ale-py|%s|\n|torch|%s|\n|numpy|%s|"
        % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
            gym.__version__,
            getattr(ale_py, "__version__", "unknown"),
            torch.__version__,
            np.__version__,
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = make_env(
        args.env_id,
        args.seed,
        0,
        args.capture_video,
        run_name,
        network=args.network,
        preprocess=args.preprocess,
        noop_max=args.noop_max,
        episodic_life=args.episodic_life,
        fire_reset=args.fire_reset,
        frame_skip=args.frame_skip,
        frame_pool=args.frame_pool,
        frame_stack_pad=args.frame_stack_pad,
        crop_top=args.crop_top,
        dump_frames=args.dump_frames,
    )()

    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    env.action_space.seed(args.seed)

    q_network = QNetwork(env, network=args.network).to(device)
    optimizer = _make_optimizer(args.optimizer, q_network.parameters(), args.learning_rate)
    target_network = QNetwork(env, network=args.network).to(device) if args.use_target_network else None
    if target_network is not None:
        target_network.load_state_dict(q_network.state_dict())

    if args.replay_storage == "frames":
        rb = FrameReplayBuffer(
            args.buffer_size,
            env.observation_space,
            env.action_space,
            device,
            args.frame_stack_pad,
        )
    else:
        rb = TruncationAwareReplayBuffer(
            args.buffer_size,
            env.observation_space,
            env.action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
    start_time = time.time()
    obs, _ = env.reset(seed=args.seed)
    episode_start = True

    heldout_states_tensor = None
    if args.heldout_frequency > 0:
        heldout_states = _sample_heldout_states(
            args.env_id,
            args.seed,
            frame_skip=args.frame_skip,
            frame_pool=args.frame_pool,
            preprocess=args.preprocess,
            noop_max=args.noop_max,
            episodic_life=args.episodic_life,
            fire_reset=args.fire_reset,
            crop_top=args.crop_top,
            frame_stack_pad=args.frame_stack_pad,
            steps=2000,
            states=500,
        )
        assert heldout_states.shape[1:] == (4, 84, 84), f"Expected held-out states shaped (N, 4, 84, 84), got {heldout_states.shape}"
        heldout_states_tensor = torch.from_numpy(heldout_states).float().to(device)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule_with_anchor(
            args.start_e,
            args.end_e,
            args.exploration_steps,
            global_step,
            args.learning_starts,
            args.anneal_start,
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        if "episode" in info:
            episode_return = float(np.asarray(info["episode"]["r"]).item())
            episode_length = int(np.asarray(info["episode"]["l"]).item())
            print(f"global_step={global_step}, episodic_return={episode_return}")
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)

        buffer_info = dict(info)
        buffer_info["_truncated"] = bool(truncated)
        buffer_args = (
            obs[None],
            next_obs[None],
            np.array([action]),
            np.array([reward]),
            np.array([terminated]),
            [buffer_info],
        )
        if args.replay_storage == "frames":
            rb.add(*buffer_args, episode_start=episode_start)
        else:
            rb.add(*buffer_args)

        if terminated or truncated:
            obs, _ = env.reset()
            episode_start = True
        else:
            obs = next_obs
            episode_start = False

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                q_values = q_network(data.observations)
                old_val = q_values.gather(1, data.actions).squeeze(-1)
                target_values = _compute_td_target(q_network, target_network, data.next_observations, data.rewards, data.dones, args.gamma, args.use_target_network)
                loss_fn = _resolve_loss(args.loss)
                loss = loss_fn(target_values, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", float(loss.item()), global_step)
                    writer.add_scalar("losses/q_values", float(old_val.mean().item()), global_step)
                    _log_scalar(writer, "charts/SPS", int(global_step / max(time.time() - start_time, 1e-8)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.use_target_network and global_step % args.target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * q_param.data + (1.0 - args.tau) * target_param.data)

        if args.heldout_frequency > 0 and global_step > 0 and global_step % args.heldout_frequency == 0:
            assert heldout_states_tensor is not None
            with torch.no_grad():
                heldout_q = q_network(heldout_states_tensor)
                heldout_max = heldout_q.max(dim=1).values.mean().item()
            writer.add_scalar("charts/heldout_avg_max_q", heldout_max, global_step)

        if args.eval_frequency > 0 and global_step > 0 and global_step % args.eval_frequency == 0:
            metrics = _run_eval(
                args.env_id,
                args.seed + 1000,
                q_network,
                device,
                frame_skip=args.frame_skip,
                frame_pool=args.frame_pool,
                preprocess=args.preprocess,
                noop_max=args.noop_max,
                episodic_life=args.episodic_life,
                fire_reset=args.fire_reset,
                crop_top=args.crop_top,
                frame_stack_pad=args.frame_stack_pad,
                eval_steps=args.eval_steps,
                eval_eps=args.eval_eps,
                dump_frames=args.dump_frames,
            )
            writer.add_scalar("eval/mean_return", metrics["mean_return"], global_step)
            writer.add_scalar("eval/max_return", metrics["max_return"], global_step)
            writer.add_scalar("eval/num_episodes", metrics["num_episodes"], global_step)
            writer.add_scalar("eval/partial", metrics["partial"], global_step)

    final_eval = _run_eval(
        args.env_id,
        args.seed + 2000,
        q_network,
        device,
        frame_skip=args.frame_skip,
        frame_pool=args.frame_pool,
        preprocess=args.preprocess,
        noop_max=args.noop_max,
        episodic_life=args.episodic_life,
        fire_reset=args.fire_reset,
        crop_top=args.crop_top,
        frame_stack_pad=args.frame_stack_pad,
        eval_steps=args.eval_steps,
        eval_eps=args.eval_eps,
        dump_frames=args.dump_frames,
    )
    writer.add_scalar("eval/mean_return", final_eval["mean_return"], args.total_timesteps)
    writer.add_scalar("eval/max_return", final_eval["max_return"], args.total_timesteps)
    writer.add_scalar("eval/num_episodes", final_eval["num_episodes"], args.total_timesteps)
    writer.add_scalar("eval/partial", final_eval["partial"], args.total_timesteps)

    if args.save_model:
        save_path = os.path.join("runs", run_name, f"{args.exp_name}.cleanrl_model")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(q_network.state_dict(), save_path)
        print(f"model saved to {save_path}")

    env.close()
    writer.close()
