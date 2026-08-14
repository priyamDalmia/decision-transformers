# decision-transformers

Research code for studying **reinforcement learning as conditional sequence modeling**.

## Goal

Decision Transformers reframe RL as sequence prediction. Instead of learning a value
function or computing policy gradients, you train a causal transformer on offline
trajectories and generate actions autoregressively, conditioned on a *desired* return.

This repo builds up to that: starting from simple models (MLP, LSTM) and working
towards transformers and Decision Transformer variants, on a small set of environments.
A large part of the work is reproducing published baselines so comparisons stay honest.

We compare against offline RL, online RL, and simpler sequence-modeling baselines.
Standard RL algorithms are **not** reimplemented here — those come from Ray RLlib.

## Layout

```
src/dt/
  dataset/   trajectory collection and torch Datasets
  models/    model implementations
  train/     training loops
  utils/     logging, seeding
  config.py  experiment configs
  paths.py   project paths
configs/     YAML config per workflow
scripts/     entry points
data/        datasets and checkpoints (gitignored)
```

## Getting started

```bash
uv sync                                    # create the environment
uv run python -m dt.dataset.data_collection  # collect trajectories
uv run python -m dt.train.train              # train
uv run python scripts/evaluate.py            # evaluate
```

Settings live in `configs/`. Paths resolve against the project root, so commands work
from any directory.

## Status

Early. The working end-to-end path is a reward-conditioned behaviour cloning MLP on
Pendulum — this is a baseline, not the destination. The Decision Transformer itself is
still being written (`scripts/minimal_dt.py` is incomplete).

## More

- `docs/DESIGN.md` — design principles and scope
- `docs/PLAN.md` — current plan

Contributions welcome — the repo is deliberately small, so please keep new abstractions
minimal and raise anything structural before building it.
