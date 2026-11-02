"""CLI arguments, presets, and configuration resolution for the Mnih DQN."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Literal

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
    exp_name: str = "atari_mnih"
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
