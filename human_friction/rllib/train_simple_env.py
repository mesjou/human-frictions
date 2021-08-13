import os

import ray
from human_friction.rllib.helper import SimpleRllibEnv
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from human_friction.rllib.SimpleModelClass import SimpleModelMasked


ModelCatalog.register_custom_model("SimpleModelMasked", SimpleModelMasked)

# seeds = list(range(1))
env_config = {
    "episode_length": 5,
    "n_agents": 1,
}

rllib_config = {
    # === Settings for Environment ===
    "env": SimpleRllibEnv,
    "env_config": env_config,
    "vf_loss_coeff": 1.0,   #need to tune this!
    "model": {
        "custom_model": "SimpleModelMasked",
        "vf_share_layers": True,
        "custom_model_config": {'nbr_choices':10, 'true_obs_nbr':3}
    }
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
