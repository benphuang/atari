# Resuming Atari DQN training

The training CLI can save and restore the model, target model, optimizer, random-number-generator states, step counter, and complete replay buffer. Replay is saved so the resumed run continues with the experience it has already collected. The Atari environment itself is reset on resume; replay preserves the last transition's next observation and starts a fresh episode.

## iLab storage and memory

Use `/common/users/bh567` for checkpoints and run outputs. It is the permanent storage area with a 250 GB per-user allocation; about 16 GB was in use when this workflow was checked. On the iLab machine at that time:

- Host memory was 30 GiB total, with about 25 GiB available.
- The NVIDIA GeForce RTX 4060 had 8 GiB VRAM, with about 7.6 GiB free.
- `/freespace/local` had 277 GB free on local SSD, but it is temporary and not backed up.
- `/filer/tmp1` was at 100% utilization and is also temporary and not backed up.

For a one-million-entry replay buffer, `--replay-storage frames` uses about 6.6 GiB of host RAM and stores replay as one 84×84 `uint8` frame per entry. Replay is not in VRAM. Do not use the default `stacked` replay layout for this full run: it needs about 26.3 GiB of host RAM, more than was available alongside the training process.

A full checkpoint is about the size of the replay buffer (roughly 6.6 GiB). The atomic save writes a temporary sibling file and renames it over the previous checkpoint only after the new file is complete. Thus, allow roughly 13.2 GiB of free space during a save. Only one completed checkpoint is retained; choose a checkpoint interval that balances write time against work you are willing to repeat after an interruption. Use only one training process per checkpoint path. The examples below checkpoint every one million agent steps.

## Start a full run

Run from the permanent output directory so TensorBoard logs and the final model persist there too:

```bash
mkdir -p /common/users/bh567/atari_dqn/arxiv2013
cd /common/users/bh567/atari_dqn/arxiv2013
/path/to/atari-venv/bin/python /common/home/bh567/cs520/atari/atari_mnih.py \
  --preset arxiv2013 \
  --exp-name ilab_breakout_arxiv2013 \
  --save-model \
  --replay-storage frames \
  --checkpoint-frequency 1000000 \
  --checkpoint-path /common/users/bh567/atari_dqn/arxiv2013/latest.pt
```

For Nature 2015, use a separate output directory and checkpoint:

```bash
mkdir -p /common/users/bh567/atari_dqn/nature2015
cd /common/users/bh567/atari_dqn/nature2015
/path/to/atari-venv/bin/python /common/home/bh567/cs520/atari/atari_mnih.py \
  --preset nature2015 \
  --exp-name ilab_breakout_nature2015 \
  --save-model \
  --replay-storage frames \
  --checkpoint-frequency 1000000 \
  --checkpoint-path /common/users/bh567/atari_dqn/nature2015/latest.pt
```

Replace `/path/to/atari-venv/bin/python` with the Python environment that satisfies [`requirements-atari.txt`](./requirements-atari.txt). Do not use the previously inspected `.venv` without upgrading it: it had Gymnasium 0.29.1 and ale-py 0.8.1, below the versions this code requires.

## Resume after interruption

Rerun the same command with `--resume-from` pointing at `latest.pt`. Keep the preset, experiment name, total training budget, seed, and algorithm settings unchanged; the trainer rejects incompatible checkpoints. The checkpoint interval and checkpoint destination can be changed.

```bash
cd /common/users/bh567/atari_dqn/arxiv2013
/path/to/atari-venv/bin/python /common/home/bh567/cs520/atari/atari_mnih.py \
  --preset arxiv2013 \
  --exp-name ilab_breakout_arxiv2013 \
  --save-model \
  --replay-storage frames \
  --checkpoint-frequency 1000000 \
  --checkpoint-path /common/users/bh567/atari_dqn/arxiv2013/latest.pt \
  --resume-from /common/users/bh567/atari_dqn/arxiv2013/latest.pt
```

Only load checkpoint files you trust. Checkpoints use Python/PyTorch serialization and should be treated as executable input.
