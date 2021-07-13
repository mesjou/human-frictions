import os

import ray
from human_friction.rllib.rllib_env import RllibEnv
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

seeds = list(range(1))
env_config = {
    "episode_length": 200,
    "n_agents": 2,
}

rllib_config = {
    "env": RllibEnv,
    "env_config": env_config,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 128,
    # Number of timesteps collected for each SGD round.
    # This defines the size of each SGD epoch.
    "train_batch_size": 256,
    "model": {"fcnet_hiddens": [50, 50]},
    "lr": 5e-3,
    "seed": tune.grid_search(seeds),
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "framework": "tf",
}


def run(debug=True, iteration=200):
    stop = {"training_iteration": 2 if debug else iteration}
    tune_analysis = tune.run(
        PPOTrainer, config=rllib_config, stop=stop, checkpoint_freq=0, checkpoint_at_end=True, name="PPO_New_Keynes"
    )
    return tune_analysis


if __name__ == "__main__":
    ray.init()
    run()
    ray.shutdown()
