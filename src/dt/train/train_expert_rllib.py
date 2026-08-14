import logging

import ray
from ray.rllib.algorithms.sac import SACConfig

from dt.paths import CHECKPOINT_DIR
from dt.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# This module is a top-to-bottom training script, so it configures logging on
# import rather than under a __main__ guard.
setup_logging()

ray.init(ignore_reinit_error=True)


config = (
    SACConfig()
    .environment(env="Pendulum-v1")
    .env_runners(num_env_runners=1)
    .training(
        train_batch_size_per_learner=256,
        gamma=0.99,
        tau=0.005,
        # SAC defaults are reasonable; tune if needed
    )
    .learners(num_learners=0)  # single-process learner; bump up if you have GPUs
)

algo = config.build_algo()

target_return = -150.0  # near-expert for Pendulum
result = algo.train()
for i in range(200):
    result = algo.train()
    mean_ret = result["env_runners"]["episode_return_mean"]
    logger.debug("iter %3d | mean_return %.1f", i, mean_ret)
    if mean_ret >= target_return:
        logger.info("Target return %.1f reached at iter %d", target_return, i)
        break

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = algo.save_to_path(str(CHECKPOINT_DIR / "sac_pendulum_rllib"))
logger.info("Saved checkpoint -> %s", checkpoint_path)
algo.stop()
ray.shutdown()
