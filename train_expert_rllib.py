import ray
from ray.rllib.algorithms.sac import SACConfig

ray.init(ignore_reinit_error=True)


config = (
    SACConfig()
    .environment(env="Pendulum-v1")
    .env_runners(num_env_runners=2)
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
    print(f"iter {i:3d} | mean_return = {mean_ret:.1f}")
    if mean_ret >= target_return:
        print(f"Target return {target_return} reached at iter {i}!")
        break

checkpoint_path = algo.save_to_path("./checkpoints/sac_pendulum_rllib")
print(f"Saved checkpoint to {checkpoint_path}")
algo.stop()
ray.shutdown()
