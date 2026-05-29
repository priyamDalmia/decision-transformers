import pickle
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset_seq import TrajectorySequenceDataset
from models.lstm import LSTMDecisionModel


def masked_mse(pred, target, mask):
    raise NotImplementedError("Implement this for the LSTM-based decision model")


def train(
    data_paths,
    epochs=50,
    batch_size=64,
    lr=1e-4,
    context_len=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    # Pool data from multiple file (e.g. random + expert)
    trajectories = []
    for p in data_paths:
        with open(p, "rb") as f:
            trajectories.extend(pickle.load(f))
    print(f"Loaded {len(trajectories)} trajectories from {len(data_paths)} files")

    dataset = TrajectorySequenceDataset(trajectories, context_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMDecisionModel(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        action_low=-2.0,
        action_high=2.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            s = batch["states"].to(device)
            a = batch["actions"].to(device)
            r = batch["rtg"].to(device)
            m = batch["mask"].to(device)

            pred = model(s, a, r)
            loss = masked_mse(pred, a, m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            opt.step()

            total += loss.item() * s.size(0)
            count += s.size(0)
        print(f"Epoch {epoch + 1:3d} | loss = {total / count:.5f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.states_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "rtg_scale": dataset.rtg_scale,
            "context_length": context_len,
        },
        "checkpoints/lstm_deicison.pt",
    )
