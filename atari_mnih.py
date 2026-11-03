from __future__ import annotations

import os
import random
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

from mnih_config import (
    ARXIV2013_DEFAULTS,
    CLEANRL_DEFAULTS,
    NATURE2015_DEFAULTS,
    PRESETS,
    Args,
    _as_bool,
    _auto_frame_skip,
    _explicit_fields,
    _float_or_none,
    apply_preset,
)
from mnih_checkpoint import (
    CHECKPOINT_VERSION,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    training_config,
    validate_checkpoint_config,
)
from mnih_dqn import (
    DM_RMSProp,
    QNetwork,
    _compute_td_target,
    _make_optimizer,
    _resolve_loss,
    linear_schedule,
    linear_schedule_with_anchor,
)
from mnih_env import (
    ArxivPreprocessEnv,
    CleanRLPreprocessEnv,
    DumpFrameEnv,
    SkipEnv,
    make_env,
)
from mnih_eval import _log_scalar, _run_eval, _sample_heldout_states
from mnih_replay import (
    FrameReplayBuffer,
    FrameStackPad,
    TruncationAwareReplayBuffer,
)

if hasattr(gym, "register_envs"):
    gym.register_envs(ale_py)


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


def main() -> None:
    args = tyro.cli(Args)
    args = apply_preset(args)
    assert args.num_envs == 1, "vectorized envs are not supported"
    if args.replay_storage not in {"stacked", "frames"}:
        raise ValueError(f"Unknown replay storage '{args.replay_storage}'. Expected 'stacked' or 'frames'.")
    if args.checkpoint_frequency < 0:
        raise ValueError("checkpoint_frequency must be non-negative.")
    if args.checkpoint_path is not None and args.checkpoint_frequency == 0:
        raise ValueError("Set --checkpoint-frequency to a positive value when using --checkpoint-path.")
    if args.resume_from is not None and args.checkpoint_frequency == 0:
        raise ValueError("Set --checkpoint-frequency to a positive value when resuming.")
    if args.checkpoint_frequency > 0 and args.checkpoint_path is None and args.resume_from is None:
        raise ValueError("--checkpoint-path is required when checkpointing a new run.")
    print(f"Resolved config: {vars(args)}")

    if args.loss == "cleanrl" and args.optimizer == "dm_rmsprop":
        print("Warning: dm_rmsprop with the cleanrl mean-squared loss is not the original DeepMind setup.")

    _print_memory_warning(args.buffer_size, args.replay_storage)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    checkpoint = load_checkpoint(args.resume_from) if args.resume_from is not None else None
    if checkpoint is not None:
        validate_checkpoint_config(checkpoint, args, device)
        run_name = checkpoint["run_name"]
        checkpoint_path = args.checkpoint_path or args.resume_from
        start_step = int(checkpoint["next_step"])
    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        checkpoint_path = args.checkpoint_path
        start_step = 0
    run_dir = checkpoint["run_dir"] if checkpoint is not None else os.path.abspath(f"runs/{run_name}")
    writer = SummaryWriter(run_dir)
    if checkpoint is None:
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
    else:
        print(f"Resuming {run_name} at global_step={start_step} from {args.resume_from}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

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
    obs, _ = env.reset(seed=args.seed if checkpoint is None else args.seed + start_step)
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

    if checkpoint is not None:
        q_network.load_state_dict(checkpoint["q_network"])
        if (target_network is None) != (checkpoint["target_network"] is None):
            raise ValueError("Checkpoint target-network configuration does not match this run.")
        if target_network is not None:
            target_network.load_state_dict(checkpoint["target_network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rb.load_state_dict(checkpoint["replay_buffer"])
        rb.mark_restart_boundary(checkpoint["last_next_observation"])
        restore_rng_state(checkpoint["rng_state"], env.action_space)

    last_next_observation: np.ndarray | None = None
    for global_step in range(start_step, args.total_timesteps):
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
        last_next_observation = np.asarray(next_obs).copy()
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
                    _log_scalar(
                        writer,
                        "charts/SPS",
                        int((global_step - start_step) / max(time.time() - start_time, 1e-8)),
                        global_step,
                    )

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

        next_step = global_step + 1
        if (
            checkpoint_path is not None
            and args.checkpoint_frequency > 0
            and (next_step % args.checkpoint_frequency == 0 or next_step == args.total_timesteps)
        ):
            assert last_next_observation is not None
            writer.flush()
            save_checkpoint(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "next_step": next_step,
                    "run_name": run_name,
                    "run_dir": run_dir,
                    "device_type": device.type,
                    "config": training_config(args),
                    "q_network": q_network.state_dict(),
                    "target_network": target_network.state_dict() if target_network is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "replay_buffer": rb.state_dict(),
                    "last_next_observation": last_next_observation,
                    "rng_state": capture_rng_state(env.action_space),
                },
            )
            print(f"Checkpoint saved at global_step={next_step}: {checkpoint_path}")

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
        save_path = os.path.join(run_dir, f"{args.exp_name}.cleanrl_model")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(q_network.state_dict(), save_path)
        print(f"model saved to {save_path}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
