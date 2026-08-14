"""Seeding helper for reproducible experiments.

Every source of randomness that the current workflows touch gets seeded from one
call, so an experiment is reproducible from the `seed` field in its config.

Note that gymnasium environments carry two separate RNGs: one for reset() and
one on the action space. Both need seeding - missing the second is why random
data collection silently ignored its seed.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, env=None) -> None:
    """Seed Python, NumPy and torch, and optionally a gymnasium environment.

    Pass `env` to also seed its reset() stream and action space. Call this
    before building models or collecting data, not partway through.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    logger.debug("Seeded python/numpy/torch%s with %d", " and env" if env else "", seed)
