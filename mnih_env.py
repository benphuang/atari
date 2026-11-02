"""Atari environment construction and image preprocessing wrappers."""

from __future__ import annotations

import os
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, NoopResetEnv
from mnih_config import _auto_frame_skip
from mnih_replay import FrameStackPad

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
