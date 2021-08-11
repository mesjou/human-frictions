import os

import ray
from human_friction.rllib.helper import SimpleRllibEnv
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# seeds = list(range(1))
env_config = {
    "episode_length": 5,
    "n_agents": 1,
}

rllib_config = {
    # === Settings for Environment ===
    "env": SimpleRllibEnv,
    "env_config": env_config
    }

def run(debug=True, iteration=200):
    stop = {"training_iteration": 1 if debug else iteration}
    tune_analysis = tune.run(
        PPOTrainer, config=rllib_config, stop=stop, max_failures = 1, resume = False,
        checkpoint_freq=100, checkpoint_at_end=True, name="SimpleEnv",
        local_dir = os.path.join(os.getcwd(),"human_friction/checkpoints")
    )

    with open("human_friction/checkpoints/SimpleEnv/last_checkpoint_path","w") as f:
        f.write(tune_analysis.get_last_checkpoint())

    return tune_analysis


if __name__ == "__main__":
    ray.init(num_gpus=2)
    run(debug=False)
    ray.shutdown()
