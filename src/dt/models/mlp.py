import torch
import torch.nn as nn


class RCBCModel(nn.Module):
    """MLP: (state, rtg) -> action,"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        n_layers=3,
        action_low=-1.0,
        action_high=1.0,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim + 1, hidden_dim), nn.ReLU()]

        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

        # Squash to action bounds - Pendulum actions are in [-2, 2]
        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )

    def forward(self, state, rtg):
        # state: (B, state_dim), rtg: (B, 1)
        x = torch.cat([state, rtg], dim=-1)
        raw = self.net(x)
        # tanh-squash to action bounds
        return self.action_low + (self.action_high - self.action_low) * 0.5 * (
            torch.tanh(raw) + 1.0
        )
