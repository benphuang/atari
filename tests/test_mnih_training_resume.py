import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

import atari_mnih
from mnih_checkpoint import load_checkpoint


class FakeEnv:
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.step_number = 0

    def reset(self, seed: int | None = None):
        self.step_number = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action: int):
        self.step_number += 1
        observation = np.full((4, 84, 84), self.step_number, dtype=np.uint8)
        return observation, 0.0, False, False, {}

    def close(self) -> None:
        pass


class FakeWriter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def add_text(self, *args, **kwargs) -> None:
        pass

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TrainingResumeTests(unittest.TestCase):
    def test_training_resumes_from_last_atomic_checkpoint(self) -> None:
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = Path(directory) / "latest.pt"
                arguments = [
                    "--preset",
                    "arxiv2013",
                    "--exp-name",
                    "checkpoint-integration-test",
                    "--total-timesteps",
                    "8",
                    "--buffer-size",
                    "16",
                    "--batch-size",
                    "2",
                    "--learning-starts",
                    "1",
                    "--exploration-steps",
                    "8",
                    "--eval-frequency",
                    "0",
                    "--eval-steps",
                    "4",
                    "--heldout-frequency",
                    "0",
                    "--replay-storage",
                    "frames",
                    "--checkpoint-frequency",
                    "4",
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--no-cuda",
                ]
                save_checkpoint = atari_mnih.save_checkpoint

                def interrupt_after_first_checkpoint(path, state) -> None:
                    save_checkpoint(path, state)
                    if state["next_step"] == 4:
                        raise RuntimeError("simulated runtime interruption")

                with (
                    patch.object(atari_mnih, "make_env", side_effect=lambda *args, **kwargs: FakeEnv),
                    patch.object(atari_mnih, "SummaryWriter", FakeWriter),
                    patch.object(
                        atari_mnih,
                        "_run_eval",
                        return_value={"mean_return": 0.0, "max_return": 0.0, "num_episodes": 0, "partial": True},
                    ),
                    patch.object(atari_mnih, "save_checkpoint", side_effect=interrupt_after_first_checkpoint),
                    patch.object(sys, "argv", ["atari_mnih.py", *arguments]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated runtime interruption"):
                        atari_mnih.main()

                self.assertEqual(load_checkpoint(checkpoint_path)["next_step"], 4)

                with (
                    patch.object(atari_mnih, "make_env", side_effect=lambda *args, **kwargs: FakeEnv),
                    patch.object(atari_mnih, "SummaryWriter", FakeWriter),
                    patch.object(
                        atari_mnih,
                        "_run_eval",
                        return_value={"mean_return": 0.0, "max_return": 0.0, "num_episodes": 0, "partial": True},
                    ),
                    patch.object(sys, "argv", ["atari_mnih.py", *arguments, "--resume-from", str(checkpoint_path)]),
                ):
                    atari_mnih.main()

                final_checkpoint = load_checkpoint(checkpoint_path)
                self.assertEqual(final_checkpoint["next_step"], 8)
                self.assertEqual(final_checkpoint["replay_buffer"]["pos"], 8)
        finally:
            torch.set_num_threads(thread_count)


if __name__ == "__main__":
    unittest.main()
