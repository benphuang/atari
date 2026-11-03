"""Atomic full-state checkpoints for resuming Atari DQN training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_VERSION = 1


def training_config(args: Any) -> dict[str, Any]:
    ignored = {"checkpoint_path", "checkpoint_frequency", "resume_from"}
    return {name: value for name, value in vars(args).items() if name not in ignored}


def capture_rng_state(action_space: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "action_space": action_space.np_random.bit_generator.state,
    }


def restore_rng_state(state: dict[str, Any], action_space: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    action_space.np_random.bit_generator.state = state["action_space"]


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        with temporary_path.open("wb") as checkpoint_file:
            torch.save(state, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported or invalid checkpoint: {checkpoint_path}")
    return state


def validate_checkpoint_config(state: dict[str, Any], args: Any, device: torch.device) -> None:
    saved_config = state["config"]
    current_config = training_config(args)
    differences = [
        name
        for name, value in saved_config.items()
        if current_config.get(name) != value
    ]
    differences.extend(name for name in current_config if name not in saved_config)
    if differences:
        raise ValueError(
            "Checkpoint configuration differs from the current run for: "
            + ", ".join(sorted(set(differences)))
        )
    if state["device_type"] != device.type:
        raise ValueError(
            f"Checkpoint was created on {state['device_type']}, but the current device is {device.type}."
        )
    if not 0 <= int(state["next_step"]) <= args.total_timesteps:
        raise ValueError(
            f"Checkpoint next step {state['next_step']} is outside the configured "
            f"training range 0..{args.total_timesteps}."
        )
