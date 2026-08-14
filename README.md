# decision-transformers

This repository contains implementations of Decisions Tranformers models for Sequential Decision-Making.

## Core Idea

Decision Transformers reframe reinforcement learning as a **conditional sequence modeling** problem. Instead of learning a value functions or computing policy gradients, it trains a Transformer (GPT-style) on offline trajectories and generates actions autoregressively - conditioned on a *desired* return.

For a list of papers on see -
