from pathlib import Path
import gymnasium as gym
import numpy as np
import pickle
import torch

from ray.rllib.core.rl_module import RLModule
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)

# resuing old utility
from data_collection import compute_returns_to_go


def load_rl_module(checkpoint_path):
    module_path = (
        Path(checkpoint_path)
        / COMPONENT_LEARNER_GROUP
        / COMPONENT_LEARNER
        / COMPONENT_RL_MODULE
        / DEFAULT_MODULE_ID
    )
    return RLModule.from_checkpoint(module_path)


def collect_expert_trajectories(
    checkpoint_path: str,
    env_name: str = "Pendulum-v1",
    num_episodes: int = 1000,
    action_noise_std: float = 0.0,
    seed: int = 42,
):
    rl_module = load_rl_module(checkpoint_path)
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    trajectories = []

    for ep in range(num_episodes):
        obs, _ = env.reset(
            seed=int(rng.integers(0, 2**31 - 1))
        )  # different seed for each episode
        states, actions, rewards = [], [], []
        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                # forward interfernce returns a dict; SAC's continous policy puts the mean action under "action_dist_inputs"; structure may differ for other algos
                out = rl_module.forward_inference({"obs": obs_tensor})
            # For SAC's default GaussianMixin, take the determinisitc Mean
            if "actions" in out:
                action = out["actions"].sequeze(0).numpy()
            else:
                # Squashed Gaussain: take tanh(mean)
                action_dist_inputs = out["action_dist_inputs"].squeeze(0).numpy()
                mean = action_dist_inputs[: len(action_dist_inputs) // 2]
                action = np.tanh(mean) * env.action_space.high

            if action_noise_std > 0.0:
                action = action + rng.normal(0, action_noise_std, size=action.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, _ = env.step(
                action.astype(np.float32)
            )
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated
        trajectories.append(
            {
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.float32),
                "rewards": np.array(rewards, dtype=np.float32),
                "rtg": compute_returns_to_go(np.array(rewards, dtype=np.float32)),
            }
        )

    env.close()
    return trajectories


if __name__ == "__main__":
    trajs = collect_expert_trajectories(
        "./checkpoints/sac_pendulum_rllib",
        num_episodes=1000,
        action_noise_std=0.1,  # mild noise = "medium-expert" quality
    )
    returns = np.array([t["rewards"].sum() for t in trajs])
    print(
        f"Expert: mean={returns.mean():.2f}, min={returns.min():.2f}, max={returns.max():.2f}"
    )
    with open("data/pendulum_expert.pkl", "wb") as f:
        pickle.dump(trajs, f)
