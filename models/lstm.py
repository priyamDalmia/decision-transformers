import torch
import torch.nn as nn


class LSTMDecisionModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        n_layers=2,
        action_low=-2.0,
        action_high=2.0,
        dropout=0.1,
    ):
        super().__init__()

        # Per-modality input embeddings (miics what the transformer will do)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # the LSTM consumes one combined token per timestep
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("a_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("a_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, states, actions, rtgs):
        # states: (B, K, state_dim); actions: (B, K, action_dim); rtgs: (B, K, 1)
        B, K, _ = states.shape

        # Shift actions right by one position so we predict a_t givem
        prev_actions = torch.zeros_like(actions)
        prev_actions[:, 1:] = actions[:, :-1]

        s_emb = self.embed_ln(self.embed_state(states))
        a_emb = self.embed_ln(self.embed_action(prev_actions))
        r_emb = self.embed_ln(self.embed_rtg(rtgs))

        # Combine - using concatenation here, but you could also sum (paper-style)
        x = torch.cat([s_emb, a_emb, r_emb], dim=-1)  # (B, K, hidden_dim*3)
