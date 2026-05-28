import numpy as np
import torch
from torch.utils.data import Dataset


class RCBCDataset(Dataset):
    """Reward-Conditioned Behavior Cloning (RCBC) Dataset.

    Each sample is (state, rtg) -> action. No sequences yet."""

    def __init__(self, trajectories):
        states = np.concatenate([t["state"] for t in trajectories], axis=0)
        actions = np.concatenate([t["action"] for t in trajectories], axis=0)
        rtgs = np.concatenate([t["rtg"] for t in trajectories], axis=0)

        # Normalize states (fit on training data, save stats for eval-time use)
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-6
        self.states = ((states - self.state_mean) / self.states_std).astype(np.float32)
        self.actions = actions.astype(np.float32)

        # Scale RTG so it's not orders of magnitude larger than states.
        # Use a fixed scale (max abs in dataset) - This matters at eval time too.
        self.rtg_scale = float(np.abs(rtgs).max()) + 1e-6
        self.rtgs = (rtgs / self.rtg_scale).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.rtgs[idx]),
            torch.from_numpy(self.actions[idx]),
        )
