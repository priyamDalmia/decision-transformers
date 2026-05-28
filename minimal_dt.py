"""
Decision Transoformer - Minimal Implementation from Scratch 
---------------------
Dependencies: Torch, numpy (nothing else)

Implements the core loop from Chen et al.
    1. Collect Offline data (random policy on a toy env) 
    2. Compute returns-to-go
    3. Train a causal Transformer conditioned on (R, s, a)
    4. At test time, prompt with a desired return -> get actions
"""
import math 
import random 
import numpy as np
from typing import List, Dict 

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader


# TOY ENVIRONMENT 
# 1-D point navigation on a discrete grid. 
# State     [position / grid_size, goal / grid_size] (2-dim, normalized)
# Action    0 = Left; 1 = Stay; 2 = Right
# Reward    +1.0 on reaching the goal, -0.1 per step 
# Episode   terminates on success or after max_steps 

class PointNav1D:

    """Minimal 1-D navigation environment."""

    def __init__(self, grid_size:int = 10, max_steps: int = 20):
        self.grid_size = grid_size 
        self.max_steps = max_steps
        self.pos = 0 
        self.goal = 0
        self.steps = 0

    def reset(self)  -> np.ndarray: 
        self.pos = np.random.randint(0, self.grid_size)
        self.goal = np.random.randint(0, self.grid_size)
        # avoid trivial episodes where start == goal 
        while self.goal == self.pos:
            self.goal = np.random.randint(0, self.grid_size)
        self.steps = 0
        return self._obs()
    
    def step(self, action:int):
        if action == 0:
            self.pos = max(0, self.pos - 1)
        elif action == 2:
            self.pos = min(self.grid_size - 1, self.pos + 1) 
        # action == 1 -> stay

        self.steps += 1
        reached = self.pos == self.goal 
        reward = 1.0 if reached else -0.1
        done = reached or self.steps >= self.max_steps
        return self._obs(), reward, done
    
    def _obs(self) -> np.ndarray:
        return np.array(
            [self.pos / self.grid_size, self.goal / self.grid_size], 
            dtype = np.float32
        )

    @property
    def state_dim(self):
        return 2 
    
    @property
    def act_dim(self):
        return 3 # discrete
    

# 2. OFFLINE DATA COLLECTION 
# We run a random policy and store trajectories 
# Key insight from the paper; DT can learn from suboptimal
# (even random) data - it just needs coverage.  

def collect_random_trajectories(
        env:PointNav1D, 
        n_episodes: int = 10000,
) -> List[Dict[str, np.ndarray]]:
    """Collect trajectories using a random uniform policy"""
    trajectories = []
    for _ in range(n_episodes):
        states, actions, rewards = [], [], [] 
        s = env.reset()
        done = False 
        while not done:
            a = np.random.randint(0, env.act_dim)
            states.append(s)
            actions.append(a)
            s_next, r, done = env.step(a)
            rewards.append(r)
            s = s_next
        trajectories.append(
            {
                "states": np.stack(states),
                "actions": np.stack(actions),
                "rewards": np.stack(rewards),
            }
        )
    return trajectories

def compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """
    R_T = sum()
    
    This is the key conditioning signal that lets us 
    request high-return behaviour at test time.
    """
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0 
    for t in reversed(range(len(rewards))):
        running += rewards[t] 
        rtg[t] = running 
    return rtg


# 3. DATASET 
# Following the paper: sample random windows of length K from each trajectory. 
# Each sample is a tuple of (returns-to-go, states, actions, timesteps)

class TrajectoryDataset(Dataset):
    """
    Stores preprocessed trajectories and yields windows of length K. 
    Mirrors the paper's trainigng setup (Algorithm 1, training loop).
    """
    
    def __init__(self, trajectories: List[Dict], context_len: int):
        self.context_len = context_len

        # Preprocess: compute returns-to-go and flatten 
        self.states, self.actions, self.rtgs, self.timesteps = [], [], [], [] 
        self.traj_starts = [] # index boundaries per trajectory

        for traj in trajectories:
            T = len(traj["rewards"]) 
            rtg = compute_returns_to_go(traj["rewards"])
            self.states.append(traj["states"])
            self.actions.append(traj["actions"])
            self.rtgs.append(rtg)
            self.timesteps.append(np.arange(T))

        # Flatten into one big array with trajectory boundaries 
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rtgs = np.concatenate(self.rtgs, axis=0)
        self.timesteps = np.concatenate(self.timesteps, axis=0)

        # Build an index: (traj_idx, start_within_traj) for sampling
        self._indices = []
        offset = 0
        for traj in trajectories:
            T = len(traj["rewards"])
            for start in range(T):
                self._indices.append((offset, start, T))
            offset += T
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx):
        offset, start, traj_len = self._indices[idx]
        K = self.context_len

        # Grab a window of up to K steps
        end = min(start + K, traj_len) 
        actual_len = end - start 

        # Slice 
        s = self.states[offset + start : offset + end]
        a = self.actions[offset + start : offset + end]
        r = self.rtgs[offset + start : offset + end]
        t = self.timesteps[offset + start : offset + end]

        # Pad to length K if needed ( left-pad with zeros, mask later)
        pad = K - actual_len
        s = np.pad(s, ((pad, 0), (0, 0)), mode="constant")
        a = np.pad(a, (pad, 0), mode="constant")
        r = np.pad(r, (pad, 0), mode="constant")
        t = np.pad(t, (pad, 0), mode="constant")
        mask = np.concatenate([np.zeros(pad), np.ones(actual_len)])

        return (
            torch.tensor(r, dtype=torch.float32),   # returns-to-go
            torch.tensor(s, dtype=torch.float32),   # states  
            torch.tensor(a, dtype=torch.float32),   # actions 
            torch.tensor(t, dtype=torch.float32),   # timesteps 
            torch.tensor(mask, dtype=torch.float32) # attention mask
        )


# 4. CAUSAL SELF-ATTENTION (from scratch)
# This is the core of GPT. We implement it manually 
# so you can see every dot product and mask. 

class CasualSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention. 

    Key idea from the paper: the causal mask ensures that each 
    token can only atten to previous tokens, whihc is what makes 
    autoregressive generation possible.
    """

    def __init__(self, embed_dim:int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dims % num_heads == 0 
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads 

        # Single Linear Projection for Q, K, V (efficient)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Causal Mask: lower triangular matrix 
        # Token i can atten to tokens j <= i 
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Projects to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


# 5. Transformer Block 
#

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual"""

    def __init__(self, embed_dim: int, num_heads:int, max_seq_len:int, dropout:float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CasualSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# 6. Decision Transformer  
# The central model. Maps (R, s, a) -> Predicted a
# 
# Architecture (from Figure 1 of the paper) 
#   - Three modality-specific linear embeddings 
#   - One shared timestep embedding (NOT per-token positional encoding)
#   - Tokens interleaved as 
#   - GPT (stack of causal transformer blocks)
#   - Linear head to predict actions from s-token hidden states

class DecisionTransformer(nn.Module):
    def __init__(
            self, 
            state_dim: int,
            act_dim: int, 
            embed_dim: int = 64,
            num_heads: int = 4, 
            num_layers: int = 3,
            context_len: int = 20,
            max_timestep: int = 25,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim 
        self.act_dim = act_dim 


        # --- Modality Specific Embeddings --- 
        # Each of (R, s, a) gets is own linear projection
        self.embed_rtg = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Embedding(act_dim, embed_dim)

        # --- Timestep Embedding ---
        self.embed_timestep = nn.Embedding(max_timestep + 1, embed_dim)

        # --- Layer norm after embeddings ---
        self.embed_ln = nn.LayerNorm(embed_dim)

        # --- Action prediction head --- 
        # Predicts actions from the hidden state at the s-token position
        self.pred_action = nn.Linear(embed_dim, act_dim)

        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
            self, 
            rtg: torch.Tensor,          # (B, K)
            states: torch.Tensor,       # (B, K)
            actions: torch.Tensor,      # (B, K, state_dim)
            timesteps: torch.Tensor,    # (B, K)
    )-> torch.Tensor:
        B, K = rtg.shape

        # --- Embed each modality ---
        rtg_emb = self.embeb(rtg.unsqueeze(-1))      # (B, K, D)
        state_emb = self.embed_state(states)      # (B, K, D)
        action_emb = self.embed_action(actions)      # (B, K, D)

        # --- Add timestep embedding (shared across modalities) ---
        time_emb = self.embed_timesteps(timesteps)      # (B, K, D)
        rtg_emb = rtg_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb 

        # --- Interleave: []
        # This is the sequence that goes into the transformer 
        # Shape: (B, 3K, D)
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        sequence = stacked.reshape(B, 3 * K, self.embed_dim)
        sequence = self.embed_ln(sequence)

# 7. Training 

@dataclass 
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64 
    epochs: int = 15 
    context_len: int = 20 
    embed_dim: int = 64 
    num_heads: int = 4 
    num_layers: int = 3 
    dropout: float = 0.3 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train(model: DecisionTransformer,
           dataset: TrajectoryDataset, config:TrainConfig):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.lr,
        weight_decay=config.weight_decay
    )
    device = config.device 
    model  = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    losses = [] 
    for epoch in range(config.epochs):
        epoch_loss= 0.0
        n_batches = 0
        for rtg, states, actions, timesteps, mask in loader:
            rtg = rtg.to(device) 
            states = states.to(device)
            actions = actions.to(device)
            timesteps = timesteps.to(device)
            mask = mask.to(device)

            # Forwards 
            action_logits = model(rtg, states, actions, timesteps) # (B, K, act_dim)

            # Cross-entropy loss on action prediction (masked)
            logits_flat = action_logits.reshape(-1, model.act_dim)
            targets_flat = actions.reshape(-1)
            mask_flat = mask.reshape(-1)

            loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss_per_token * mask_flat).sum() / mask_flat.sum()



def main():
    seed = 42 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = TrainConfig()
    env = PointNav1D(grid_size=10, max_steps=20)

    # --- Collect data --- 
    # Start with 5,000 episodes (runs in ~2min on CPU)
    # For better results, increase to 10,000-50,000.
    n_episodes = 5000
    print(f"Collecting {n_episodes} random trajectories...")
    trajectories = collect_random_trajectories(env, n_episodes=n_episodes)

    # Quick Start 
    all_returns = [t["rewards"].sum() for t in trajectories]
    print(f"    Return stats: min={min(all_returns)}:.1f "
          f"mean={np.mean(all_returns):.1f} max={max(all_returns):.1f}")
    print(f"    Trajectoreis with positive return: "
          f"{sum(1 for r in all_returns if r > 0)}")

    # Build the dataset 
    dataset = TrajectoryDataset(trajectories, context_len=config.context_len)
    print(f"    Dataset size: {len(dataset)} windows\n")

    # Create a model 
    model = DecisionTransformer(
        state_dim=env.state_dim,
        act_dim=env.act_dim,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_len=config.context_len,
    )
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_param:,}\n")

    # Train 
    print("Training the model....")
    train(model, dataset, config)

if __name__ == "__main__":
    main()