"""Experiment configuration for the three initial workflows.

Each workflow has a dataclass defining its schema and defaults. YAML files in
configs/ override those defaults, and OmegaConf type-checks the result. Unknown
keys are rejected, so a typo in a config file fails loudly instead of being
silently ignored.

Paths are written relative to the project root and resolved with
paths.resolve_path() at the point of use.

    cfg = load_config(TrainConfig, CONFIG_DIR / "train.yaml")
    cfg = load_config(TrainConfig, CONFIG_DIR / "train.yaml", {"epochs": 5})
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class DataCollectionConfig:
    """Roll out a policy and save trajectories."""

    env_name: str = "Pendulum-v1"
    num_episodes: int = 2000
    seed: int = 0
    # 1.0 -> undiscounted returns-to-go (Decision Transformers).
    gamma: float = 1.0
    output_path: str = "data/pendulum_random.pkl"


@dataclass
class TrainConfig:
    """Train a model on a collected dataset."""

    data_path: str = "data/pendulum_random.pkl"
    checkpoint_path: str = "data/checkpoints/rcbc_mlp.pt"

    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 2

    # Model. Pendulum actions live in [-2, 2].
    hidden_dim: int = 128
    n_layers: int = 3
    action_low: float = -2.0
    action_high: float = 2.0

    seed: int = 0
    # null in YAML -> pick cuda when available.
    device: str | None = None


@dataclass
class EvalConfig:
    """Evaluate a trained checkpoint in the environment."""

    checkpoint_path: str = "data/checkpoints/rcbc_mlp.pt"
    env_name: str = "Pendulum-v1"

    # Sweeping targets shows whether the model conditions on return at all.
    target_returns: list[float] = field(
        default_factory=lambda: [-1500.0, -1000.0, -500.0, -200.0, -100.0]
    )
    num_episodes: int = 100
    results_path: str = "output/results/eval.json"

    seed: int = 0
    device: str | None = None


def load_config[T](
    schema: type[T],
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> T:
    """Build a config from dataclass defaults, an optional YAML file, and overrides.

    Later sources win. Returns an instance of `schema`, so attribute access is
    typed and misspelled fields raise instead of returning None.
    """
    cfg = OmegaConf.structured(schema)

    if config_path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    return OmegaConf.to_object(cfg)


def save_config(cfg: Any, path: str | Path) -> None:
    """Write a resolved config to YAML, next to whatever it produced.

    Snapshotting the config alongside the checkpoint or results is what makes an
    experiment reproducible after the fact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), path)
