import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dt.dataset.dataset import RCBCDataset
from dt.models.mlp import RCBCModel
from dt.paths import CHECKPOINT_DIR, DATA_DIR
from dt.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def train(
    data_path=DATA_DIR / "pendulum_random.pkl",
    checkpoint_path=CHECKPOINT_DIR / "rcbc_mlp.pt",
    epochs=50,
    batch_size=256,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Load trajectories and create dataset
    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)

    dataset = RCBCDataset(trajectories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # infer dims from data
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    # modeling
    model = RCBCModel(state_dim, action_dim, action_low=-2.0, action_high=2.0).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()  # continous actions -> MSE, as in the paper

    logger.info(
        "Training %s on %d transitions for %d epochs (device=%s)",
        type(model).__name__,
        len(dataset),
        epochs,
        device,
    )

    epoch_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for s, r, a in loader:
            s, r, a = s.to(device), r.to(device), a.to(device)
            pred = model(s, r)
            loss = loss_fn(pred, a)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * s.size(0)
        epoch_loss = total_loss / len(dataset)
        logger.debug("Epoch %3d/%d | loss %.4f", epoch + 1, epochs, epoch_loss)

    logger.info("Training complete | final loss %.4f", epoch_loss)

    # Save model + normalization stats together - eval needs both
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "state_mean": torch.from_numpy(dataset.state_mean).float(),
            "state_std": torch.from_numpy(dataset.state_std).float(),
            "rtg_scale": float(dataset.rtg_scale),
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint -> %s", checkpoint_path)

    return model, dataset


if __name__ == "__main__":
    setup_logging()
    train()
