import random
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mnih_checkpoint
from mnih_checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from mnih_replay import FrameReplayBuffer, TruncationAwareReplayBuffer


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation_space = gym.spaces.Box(0, 255, shape=(4, 8, 8), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)

    def _fill_frame_replay(self, replay: FrameReplayBuffer) -> None:
        for step in range(8):
            observation = np.full((1, 4, 8, 8), step, dtype=np.uint8)
            next_observation = np.full((1, 4, 8, 8), step + 1, dtype=np.uint8)
            replay.add(
                observation,
                next_observation,
                np.array([step % 3]),
                np.array([1.0]),
                np.array([False]),
                [{}],
                episode_start=step == 0,
            )

    def test_atomic_checkpoint_round_trip_and_rng_restore(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()

        self.action_space.seed(123)
        random.seed(41)
        np.random.seed(42)
        torch.manual_seed(43)
        rng_state = capture_rng_state(self.action_space)
        expected = (random.random(), np.random.random(), torch.rand(1), self.action_space.sample())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(
                path,
                {
                    "version": 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": rng_state,
                },
            )
            state = load_checkpoint(path)
            self.assertTrue(path.is_file())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

            restored_model = torch.nn.Linear(3, 2)
            restored_model.load_state_dict(state["model"])
            self.assertTrue(torch.equal(model.weight, restored_model.weight))
            restored_optimizer = torch.optim.Adam(restored_model.parameters())
            restored_optimizer.load_state_dict(state["optimizer"])
            self.assertTrue(restored_optimizer.state)

            restore_rng_state(state["rng_state"], self.action_space)
            actual = (random.random(), np.random.random(), torch.rand(1), self.action_space.sample())
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            self.assertEqual(actual[3], expected[3])

    def test_failed_write_keeps_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, {"version": 1, "value": "previous"})
            torch_save = mnih_checkpoint.torch.save

            def partial_write(state, file_object):
                file_object.write(b"partial")
                raise OSError("simulated storage failure")

            mnih_checkpoint.torch.save = partial_write
            try:
                with self.assertRaisesRegex(OSError, "simulated storage failure"):
                    save_checkpoint(path, {"version": 1, "value": "replacement"})
            finally:
                mnih_checkpoint.torch.save = torch_save

            self.assertEqual(load_checkpoint(path)["value"], "previous")
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_frame_replay_preserves_next_observation_across_restart(self) -> None:
        replay = FrameReplayBuffer(16, self.observation_space, self.action_space, torch.device("cpu"), "zeros")
        self._fill_frame_replay(replay)
        expected_next = np.full((4, 8, 8), 90, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, {"version": 1, "replay": replay.state_dict()})
            state = load_checkpoint(path)
            restored = FrameReplayBuffer(16, self.observation_space, self.action_space, torch.device("cpu"), "zeros")
            restored.load_state_dict(state["replay"])

        restored.mark_restart_boundary(expected_next)
        initial_observation = np.full((1, 4, 8, 8), 100, dtype=np.uint8)
        next_observation = np.full((1, 4, 8, 8), 101, dtype=np.uint8)
        restored.add(
            initial_observation,
            next_observation,
            np.array([0]),
            np.array([0.0]),
            np.array([False]),
            [{}],
            episode_start=True,
        )
        sample = restored._get_samples(np.array([7]))
        self.assertTrue(np.array_equal(sample.next_observations.numpy()[0], expected_next))

    def test_stacked_replay_preserves_next_observation_across_restart(self) -> None:
        replay = TruncationAwareReplayBuffer(
            16,
            self.observation_space,
            self.action_space,
            torch.device("cpu"),
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        for step in range(8):
            observation = np.full((1, 4, 8, 8), step, dtype=np.uint8)
            next_observation = np.full((1, 4, 8, 8), step + 1, dtype=np.uint8)
            replay.add(
                observation,
                next_observation,
                np.array([step % 3]),
                np.array([1.0]),
                np.array([False]),
                [{}],
            )

        expected_next = np.full((4, 8, 8), 90, dtype=np.uint8)
        restored = TruncationAwareReplayBuffer(
            16,
            self.observation_space,
            self.action_space,
            torch.device("cpu"),
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        restored.load_state_dict(replay.state_dict())
        restored.mark_restart_boundary(expected_next)
        sample = restored._get_samples(np.array([7]))
        self.assertTrue(np.array_equal(sample.next_observations.numpy()[0], expected_next))


if __name__ == "__main__":
    unittest.main()
