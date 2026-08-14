# Refined Project Plan

This plan follows the design document’s core constraints: keep the repository small, easy to experiment with, reproducible, and lightweight; prefer reusable package code over script-specific implementations; and add abstractions only when they provide clear practical value.

The repository structure should also be treated as flexible rather than something that must be fully designed upfront.

## Working Constraints

* Read and follow `DESIGN.md` as the primary design guide.
* Do not inspect or modify:

  * `temp/`
  * `docs/notes/`
  * `notebooks/`
* Do not commit any changes.
* Keep the implementation deliberately simple and research-oriented.
* Avoid introducing frameworks or abstractions unless they are directly useful for the current work.
* Scripts should primarily orchestrate reusable package functionality rather than contain substantive implementation logic.
* Use the existing logger in `utils/...logger.py` consistently throughout project code and scripts.
* Preserve a clear separation between data generation, training, and evaluation, consistent with the intended experiment workflow.

---

# Phase 1 — Basic Experiment Workflow

The goal of this phase is to establish one simple end-to-end workflow:

**collect data → train model → evaluate model**

The reusable logic should live inside the package. The scripts should provide thin executable entry points.

This matches the intended separation between datasets, models, reusable training infrastructure, and evaluation workflows.

## Script 1 — Data Collection

Add a data-collection script under `scripts/`.

The script should:

1. load the data-collection configuration;
2. initialise whatever environment or data source is required;
3. call the existing reusable functions from `data_collection.py`;
4. save the resulting dataset through the project's existing/package-level data handling;
5. log the important execution information.

### Boundary

The script should **not** reimplement data-collection logic.

Any functionality that could reasonably be reused by another experiment should remain in `data_collection.py` or the appropriate package module.

Data generation must remain separate from training so generated trajectories can be reused across experiments.

---

## Script 2 — Training

Add a training script under `scripts/`.

The script should provide the executable orchestration for training a model while keeping reusable training behaviour inside the package.

The script should:

1. load the training configuration;
2. load the required dataset;
3. initialise the selected model from `models/`;
4. initialise the required optimiser/training components;
5. select or construct the appropriate package-level loss;
6. call the reusable training functionality;
7. save the relevant training outputs/checkpoint;
8. log the important execution information.

### Training Module Responsibilities

Move or keep reusable training functionality inside the training package/module, including:

* the general training loop;
* loss functions;
* reusable optimisation logic;
* validation logic where currently required;
* common training utilities.

The design already identifies these as responsibilities of the reusable training layer rather than individual scripts.

### Script Responsibilities

Keep only model/experiment-specific orchestration in the script when that behaviour cannot reasonably be made generic.

Do **not** implement loss functions directly in the script.

Do **not** duplicate the general training loop in individual experiment scripts.

The objective is not to make every model identical, but to abstract the genuinely common training behaviour while leaving model-specific behaviour explicit.

---

## Script 3 — Evaluation

Add an evaluation script under `scripts/`.

The script should:

1. load the evaluation configuration;
2. initialise the required environment;
3. initialise the selected model;
4. load the relevant trained checkpoint;
5. call the reusable package-level evaluation functionality;
6. collect and save the evaluation results;
7. log the important execution information.

### Evaluation Module Responsibilities

Keep common evaluation behaviour in the package rather than the script.

Abstract functionality that can be shared across models and experiments while leaving genuinely model- or environment-specific behaviour explicit.

The goal is to provide a consistent evaluation procedure so results can later be meaningfully compared across models and baselines. This follows the experiment workflow defined in the design document.

---

# Phase 1 Completion State

At the end of Phase 1, the repository should support a simple workflow where a researcher can:

1. install/synchronise the project with `uv`;
2. run data collection;
3. use the collected data to train a model;
4. evaluate the resulting model;
5. repeat the process through configuration changes rather than rewriting infrastructure.

The result should remain a **small research system**, not the beginning of a general-purpose RL framework.

---

# Later Work — Document Only, Do Not Implement

These items should only be recorded as future work during the current phase.

## Revisit `data_collection.py`

Inspect the existing `data_collection.py` implementation in greater depth.

Develop a more standardised data-collection interface that better reflects the choices established in `DESIGN.md`, particularly around:

* reusable trajectory generation;
* consistent dataset representation;
* separation between collection and training;
* reproducibility and traceability of generated datasets.

The exact abstraction should **not** be designed or implemented yet. It should be informed by experience from the initial Phase 1 workflow rather than anticipated prematurely.

This is consistent with the project's principle that new abstractions should be introduced only after repeated experiments demonstrate their usefulness.

---

## Suggested Git Commit Messages

No commits should be made during the work. If the eventual changes are split into logical commits, appropriate messages would be:

* `chore: establish uv project foundation`
* `refactor: standardise project logging and configuration`
* `feat: add data collection entrypoint`
* `feat: add reusable training workflow`
* `feat: add model evaluation workflow`
