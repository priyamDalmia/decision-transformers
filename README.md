# decision-transformers

This repository contains implementations of Decisions Tranformers models for Sequential Decision-Making. 

## Core Idea 

Decision Transformers reframe reinforcement learning as a **conditional sequence modeling** problem. Instead of learning a value functions or computing policy gradients, it trains a Transformer (GPT-style) on offline trajectories and generates actions autoregressively - conditioned on a *desired* return. 

**Sequence** 
```
R_1, s_1, a_1, R_2, s_2, a_2, ...
```

where R_1 is ther *return-to-go* (sum of the future rewards from timestep t onward). At test time, you set R_1 to a high value - essentially telling to model "generate a trajectory that achieves this total reward" - and it produces the corresponding actions. 

## Why this works 

Even in a dataset of random walks on a graph, some walks happen to be short. The return-to-go for nodes along those short walks is higher than for nodes along long walks. By conditioning on *high* returns-to-go, the model learns to reproduce the patterns that led to efficient paths - without ever doing dynamic programming. 