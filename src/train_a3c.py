import argparse
import os

import ray
from environment.single_adv_env import LifeCycle
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--tf", action="store_false")
parser.add_argument("--stop-iters", type=int, default=2000)


def main(debug, stop_iters=2000, tf=False):
    train_n_replicates = 1 if debug else 1
    seeds = list(range(train_n_replicates))

    ray.init()

    stop = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "episode_length": 20,
        "retirement_date": 17,
    }

    rllib_config = {
        "env": LifeCycle,
        "env_config": env_config,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 50,
        "num_workers":5,
        #"num_events_per_worker":
        # Number of timesteps collected for each SGD round.
        # This defines the size of each SGD epoch.
        "train_batch_size": 512,
        "model": {"fcnet_hiddens": [50, 50]},
        "lr": 5e-3,
        "seed": tune.grid_search(seeds),
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "tf" if tf else "torch",
    }

    tune_analysis = tune.run(
        PPOTrainer, config=rllib_config, stop=stop, checkpoint_freq=100,
        checkpoint_at_end=True, name="PPO_Life_Cycle",
        local_dir = os.path.join(os.getcwd(),"checkpoints")
    )

    with open("last_checkpoint_path","w") as f:
        f.write(tune_analysis.get_last_checkpoint())

    ray.shutdown()
    return tune_analysis


if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = False
    main(debug_mode, args.stop_iters, args.tf)