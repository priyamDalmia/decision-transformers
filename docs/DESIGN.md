# Decision Transformer Research Project

## 1. Project Goal

The goal of this project is to build and evaluate **Decision Transformers (DTs)** and, more broadly, study **reinforcement learning as conditional sequence modeling**.

The project will implement sequence models—starting with simple neural architectures and progressing to transformers and Decision Transformer variants—and test their ability to solve Markov Decision Processes (MDPs).

The work will progress from simple models and environments toward more complete Decision Transformer experiments. A major focus will be reproducing established baselines on selected environments so that comparisons remain grounded in existing research.

The primary comparisons will be against:

* **Offline RL methods**
* **Online RL methods**
* Simpler sequence-modeling baselines

Existing RL algorithms will **not** be implemented from scratch. Instead:

* RL baselines will use **Ray RLlib** implementations where available.
* Any future custom RL algorithms should be built on top of RLlib where practical.
* Support for other learning algorithms may be added when required by the research.

---

## 2. Design Principles

This is a research codebase intended to be maintained primarily by a single PhD student. Its purpose is to support experiments, baseline reproduction, and rapid investigation of small research questions.

The repository should therefore remain deliberately simple.

### Three core design principles

1. **Easy experimentation**

   * New models and techniques should be quick to implement and test.
   * Experiment configuration and execution should remain simple.
   * The codebase should favour research iteration over complex software abstractions.

2. **Reproducibility and data consistency**

   * Experiments should be reproducible from their configuration and associated data.
   * Structured configuration and metadata should use tools such as Pydantic models and dataclasses where useful.
   * These abstractions should only be introduced where they improve consistency or reduce mistakes.

3. **Easy execution and deployment**

   * Experiments should be straightforward to run locally or on available compute.
   * Training, evaluation, data generation, and result collection should follow predictable interfaces.
   * Infrastructure should remain lightweight and proportional to the needs of the research.

The repository should avoid unnecessary abstractions, frameworks, and architectural complexity.

---

## 3. Decision Transformer

Decision Transformer reframes offline reinforcement learning as a **conditional sequence-modeling problem**.

Instead of learning a value function through temporal-difference methods or directly optimizing a policy using policy gradients, DT trains an autoregressive sequence model over trajectories.

A trajectory can contain information such as:

* states,
* actions,
* rewards,
* returns-to-go.

During inference, the model is conditioned on the desired return and previous trajectory information to predict subsequent actions.

This project will use this formulation as the starting point for studying RL problems through sequence modeling.

---

## 4. Repository Structure

The initial repository structure is **speculative**. It reflects what was useful while building the first demonstrations and should be allowed to evolve as the research develops.

The structure should only contain abstractions that continue to provide practical value.

Some scripts may currently be incomplete or broken; the important part at this stage is the overall separation of responsibilities.

### `dataset/` — Data Collection and Generation

Sequence-modeling experiments require trajectories that can be generated, stored, loaded, and reused.

This module contains the tools required to create datasets.

Examples include:

* generating random trajectories,
* loading an existing policy and using it to generate trajectories,
* generating trajectories using external RL libraries or tools,
* converting generated trajectories into the format expected by the sequence models.

Dataset generation should be kept separate from model training so that the same data can be reused across experiments.

---

### `models/` — Models and Algorithms

This module contains implementations of the methods being studied.

Examples include:

* MLPs,
* LSTMs and other recurrent models,
* transformers,
* Decision Transformers,
* variants of Decision Transformers,
* alternative ways of representing or modeling trajectories as sequences.

Where practical, models should expose consistent interfaces so that different architectures can use the same training and evaluation infrastructure.

---

### `train/` — Training and Losses

This module contains the reusable training infrastructure.

It may include:

* training loops,
* validation logic,
* optimization,
* loss functions,
* checkpoint handling,
* common training utilities.

Loss functions may either live here or alongside the model when they are tightly coupled to a particular architecture.

After the basic infrastructure stabilizes, this part of the repository should change relatively little. The goal is for most research iterations to involve changing models, data, or experiment configurations rather than repeatedly modifying the training system.

---

### `envs/` — Environments

This module contains the environments supported by the project and any environment-specific utilities.

The set of supported environments should remain intentionally limited.

Environment-specific concerns may include:

* environment creation,
* wrappers,
* observation or action preprocessing,
* evaluation settings,
* environment metadata.

Restricting experiments to explicitly supported environments keeps the problem space manageable and makes comparisons easier to reproduce.

---

### `data/` — Datasets and Experiment Data

Generated data should be stored and versioned appropriately.

This includes:

* trajectories,
* expert demonstrations,
* trajectories generated from trained policies,
* offline RL datasets,
* processed sequence-modeling datasets.

Large datasets should be managed using an appropriate data-versioning system such as **DVC**, rather than committed directly to Git.

The relationship between a dataset and the experiment that generated or consumed it should remain traceable.

---

### Outputs, Reports, and Results

Experiment outputs should be organized so that results can be reproduced and compared without introducing a large experiment-management system.

Outputs may include:

* experiment metrics,
* evaluation results,
* plots,
* summary tables,
* written reports,
* experiment configurations.

Small reports, summaries, and metadata can be tracked with Git.

Large generated artifacts should be stored using the appropriate external or versioned storage mechanism.

---

### Model Checkpoints

Trained model checkpoints and weights should be stored consistently and associated with the experiment configuration that produced them.

Large checkpoints should not be stored directly in Git. They should use the same lightweight artifact or data-versioning approach used for other large research outputs.

---

## 5. Experiment Workflow

A typical experiment should follow a simple flow:

1. Select or generate a dataset.
2. Select a supported environment.
3. Configure a model and training setup.
4. Train the model.
5. Evaluate it using a consistent evaluation procedure.
6. Store the configuration, metrics, and relevant artifacts.
7. Compare the result against appropriate sequence-modeling, offline RL, or online RL baselines.

The same dataset and evaluation setup should be reusable across multiple models wherever possible. This is important for making comparisons meaningful.

---

## 6. Baselines

Baseline reproduction is a central part of the project.

For selected environments, experiments should reproduce relevant results from existing work before introducing new methods.

RL baselines should rely on established implementations, primarily through **Ray RLlib**, rather than maintaining independent implementations of standard RL algorithms.

This keeps the research focused on:

* sequence models,
* Decision Transformers,
* experimental comparisons,
* modifications relevant to the research questions.

The objective is not to build a general-purpose RL framework.

---

## 7. Scope

The repository should remain a **small research system**, not evolve into a general reinforcement-learning platform.

The design should optimise for:

* rapid experimentation,
* clear comparisons,
* reproducible results,
* consistent data,
* understandable code,
* low maintenance overhead.

New abstractions should be introduced only when repeated experiments demonstrate that they are necessary.

The repository structure itself should remain flexible. As the research progresses, modules may be reorganized, combined, or removed if doing so makes experimentation simpler.
