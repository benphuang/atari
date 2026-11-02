"""Frame stacking and memory-aware replay storage implementations."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from buffers import ReplayBuffer, ReplayBufferSamples

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
