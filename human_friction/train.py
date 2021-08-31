import os

import ray
from human_friction.rllib.models import FCNet
from human_friction.run_configurations.rllib_config import rllib_config, rllib_config_nk
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_model", FCNet)


def run(debug=False, iteration=2000, level="simple"):
    stop = {"training_iteration": 2 if debug else iteration}
    tune_analysis = tune.run(
        PPOTrainer,
        config=rllib_config if level == "simple" else rllib_config_nk,
        stop=stop,
        max_failures=5,
        resume=False,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        name="PPO_New_Keynes",
        local_dir=os.path.join(os.getcwd(), "run_configurations/checkpoints"),
    )

    with open(os.path.join(os.getcwd(), "run_configurations/checkpoints/last_checkpoint_path"), "w") as f:
        f.write(tune_analysis.get_last_checkpoint())

    return tune_analysis


if __name__ == "__main__":
    ray.init()
    run(debug=False)
    ray.shutdown()
