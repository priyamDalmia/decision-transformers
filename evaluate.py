import gymnasium as gym
import numpy as np
import torch

from models.mlp import RCBCModel


@torch.no_grad()
def evaluate(
    checkpoint_path,
    env_name="Pendulum-v1",
    target_return=-200.0,
    num_episodes=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = RCBCModel(state_dim, action_dim, action_low=-2.0, action_high=2.0).to(
        device
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    state_mean = torch.tensor(ckpt["state_mean"], dtype=torch.float32).to(device)
    state_std = torch.tensor(ckpt["state_std"], dtype=torch.float32).to(device)
    rtg_scale = ckpt["rtg_scale"]

    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        target_rtg = target_return  # the desired sum of future rewards
        done = False
        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            s_norm = (s - state_mean) / state_std
            r = torch.tensor(
                [[target_rtg / rtg_scale]], dtype=torch.float32, device=device
            )
            action = model(s_norm, r).cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            target_rtg -= reward
            done = terminated or truncated
        returns.append(ep_return)

    env.close()
    returns = np.array(returns)
    print(
        f"Target RTG = {target_return} | "
        f"Mean return = {returns.mean():.2f} +/- {returns.std():.2f}"
    )
    return returns


if __name__ == "__main__":
    # Sweep target returns to see if the model actually conditions on RTG
    for target in [-1500, -1000, -500, -200, -100]:
        evaluate("checkpoints/rcbc_mlp.pt", target_return=target)
