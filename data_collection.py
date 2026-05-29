import gymnasium as gym
import numpy as np
import pickle
from pathlib import Path


def collect_trajectories(env_name="Pendulum-v1", num_episodes=2000, seed=0):
    """Roll out a random policy and save trajectories."""
    env = gym.make(env_name)
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

        trajectories.append(
            {
                "state": np.array(states, dtype=np.float32),
                "action": np.array(actions, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
            }
        )
    env.close()
    return trajectories


def discounted_rtg(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute discounted Return-to-Go (RTG) for a trajectory.

    R_hat_t = sum_{t'=t}^{T} gamma^{t'-t} r_{t'}. Computed in O(T) via reverse discounted cumsum."""
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


def compute_return_to_go(rewards: np.ndarray) -> np.ndarray:
    """Compute Return-to-Go (RTG) for a trajectory.

    R_hat_t = sum_{t'=t}^{T} r_{t'}. Computed in O(T) via reverse cumsum."""
    return np.flip(np.cumsum(np.flip(rewards))).copy()


if __name__ == "__main__":
    trajs = collect_trajectories()

    # For Decision Transformers: Attach Return-to-Go (RTG) to each trajectory so downstream code never recomputes
    # for t in trajs:
    #     t["rtg"] = compute_return_to_go(t["reward"])

    # For Trajectory Transformers: Attach discounted Return-to-Go (RTG) to each trajectory so downstream code never recomputes
    for t in trajs:
        t["rtg"] = discounted_rtg(t["reward"])

    returns = np.array([t["reward"].sum() for t in trajs])
    print(f"Collected {len(trajs)} trajectories.")
    print(f"Average Return: {returns.mean():.2f} +/- {returns.std():.2f}")
    Path("data").mkdir(exist_ok=True)
    with open("data/pendulum_random.pkl", "wb") as f:
        pickle.dump(trajs, f)
