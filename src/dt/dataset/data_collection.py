import logging
import pickle

import gymnasium as gym
import numpy as np

from dt.paths import DATA_DIR
from dt.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def collect_trajectories(env_name="Pendulum-v1", num_episodes=2000, seed=0, gamma=1.0):
    """Roll out a random policy and save trajectories.

    Returns-to-go are attached here so every trajectory is complete on return.
    gamma=1.0 gives the undiscounted returns-to-go used by Decision Transformers;
    pass gamma<1.0 for the discounted variant used by Trajectory Transformers."""
    env = gym.make(env_name)
    # The action space has its own RNG; without this, env.action_space.sample()
    # below ignores `seed` and the collected dataset is not reproducible.
    env.action_space.seed(seed)
    trajectories = []
    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        states, actions, rewards = [], [], []
        done = False

        while not done:
            # Random Policy. For richer data, mix in some scripted/noisy-expert epsidoes
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated

        rewards_arr = np.array(rewards, dtype=np.float32)
        trajectories.append(
            {
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.float32),
                "rewards": rewards_arr,
                "rtg": compute_discounted_returns_to_go(rewards_arr, gamma),
            }
        )
    env.close()
    return trajectories


def compute_discounted_returns_to_go(
    rewards: np.ndarray, gamma: float = 0.99
) -> np.ndarray:
    """Compute discounted Return-to-Go (RTG) for a trajectory.

    R_hat_t = sum_{t'=t}^{T} gamma^{t'-t} r_{t'}. Computed in O(T) via reverse discounted cumsum."""
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


def compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """Compute Return-to-Go (RTG) for a trajectory.

    R_hat_t = sum_{t'=t}^{T} r_{t'}. Computed in O(T) via reverse cumsum."""
    return np.flip(np.cumsum(np.flip(rewards))).copy()


if __name__ == "__main__":
    setup_logging()

    # gamma=1.0 -> undiscounted RTG (Decision Transformers).
    # Pass gamma=0.99 for the discounted RTG used by Trajectory Transformers.
    trajs = collect_trajectories(gamma=1.0)

    returns = np.array([t["rewards"].sum() for t in trajs])
    logger.info(
        "Collected %d trajectories | avg return %.2f +/- %.2f",
        len(trajs),
        returns.mean(),
        returns.std(),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "pendulum_random.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(trajs, f)
    logger.info("Saved dataset -> %s", out_path)
