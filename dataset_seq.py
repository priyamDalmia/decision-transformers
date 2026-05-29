import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectorySequenceDataset(Dataset):
    """Samples length-k windows from trajectories.

    Each item is a window of (Rtg, state, action) triples, length K.
    The target is the action sequence (we predict actions at every position, which gives K losses per window -- more efficient than predicting only the last.)

    Shorter trajectories or windows near the start are left-padded with zeros and masked out of the loss."""

    def __init__(self, trajectories, context_length=20):
        self.K = context_length
        self.trajectories = trajectories

        # sample windows proportional to trajectory length (so longer trajs aren't underweighted). We build an index of (traj_idx, end_timestep) pairs.
        self.index = []
        for ti, t in enumerate(trajectories):
            T = len(t["states"])
            for end in range(1, T + 1):  # window ends at any timestep 1..T
                self.index.append((ti, end))

        # Fit normalization stats on states and RTGs across the dataset (actions are already in [-1, 1] so no need to normalize)
        states = np.concatenate([t["state"] for t in trajectories], axis=0)
        rtgs = np.concatenate([t["rtg"] for t in trajectories], axis=0)
        self.state_mean = states.mean(axis=0).astype(np.float32)
        self.state_std = (states.std(axis=0) + 1e-6).astype(np.float32)
        self.rtg_scale = float(np.abs(rtgs).max() + 1e-6)

        self.state_dim = trajectories[0]["state"].shape[1]
        self.action_dim = trajectories[0]["action"].shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ti, end = self.index[idx]
        t = self.trajectories[ti]
        start = max(0, end - self.K)
        length = end - start

        # slice the trajectory
        s = (t["states"][start:end] - self.state_mean) / self.state_std
        a = t["actions"][start:end]
        r = (t["rtg"][start:end] / self.rtg_scale).reshape(-1, 1)

        # Left-pad to length K with zeros, mask records which positions are real
        #
        s_pad = np.zeros((self.K, self.state_dim), dtype=np.float32)
        a_pad = np.zeros((self.K, self.action_dim), dtype=np.float32)
        r_pad = np.zeros((self.K, 1), dtype=np.float32)
        mask = np.zeros((self.K,), dtype=np.float32)

        s_pad[-length:] = s
        a_pad[-length:] = a
        r_pad[-length:] = r
        mask[-length:] = 1.0

        return {
            "states": torch.from_numpy(s_pad),
            "actions": torch.from_numpy(a_pad),
            "rtgs": torch.from_numpy(r_pad),
            "mask": torch.from_numpy(mask),
        }
