import os

import ray
from human_friction.rllib.rllib_env import RllibEnv
from ray import tune
from ray.rllib.agents.a3c import A3CTrainer

# seeds = list(range(1))
env_config = {
    "episode_length": 800,
    "n_agents": 1,
}

rllib_config = {
    "env": RllibEnv,
    "env_config": env_config
}


def run(debug=True, iteration=20000):
    stop = {"training_iteration": 2 if debug else iteration}
    tune_analysis = tune.run(
        A3CTrainer, config=rllib_config, stop=stop, max_failures = 3, resume = True,
        checkpoint_freq=100, checkpoint_at_end=True, name="PPO_New_Keynes",
        local_dir = os.path.join(os.getcwd(),"human_friction/checkpoints")
    )

    with open("human_friction/results/last_checkpoint_path","w") as f:
        f.write(tune_analysis.get_last_checkpoint())

    return tune_analysis


if __name__ == "__main__":
    ray.init()
    run(debug=False)
    ray.shutdown()
