import argparse
import os
import yaml

import ray
from human_friction.rllib.rllib_env import RllibEnv
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer


import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path_stored", default = "human_friction/last_checkpoint_path")
parser.add_argument("--var_names", default = None)
parser.add_argument("--config_file", default = None)
parser.add_argument("--env_config_file", default = None)


def main(debug, checkpoint_path_stored, var_names=None, config_file=None, env_config_file=None):
    """ Compute and plot actions from the trained model

        Parameters
        ----------
        debug: bool
            whether to enter a debugging mode
        checkpoint_path_stored: str
            name of the file with the path to the last checkout
        var_names: List[str]
            names of the choice variables
        config_file : str
            path to the configuration file
        env_config_file : str
            path to the environmen configuration file

        Outputs
        ------
        config_dict: dict

    """
    train_n_replicates = 1 if debug else 1
    #seeds = list(range(train_n_replicates))

    with open(checkpoint_path_stored, "r") as f:
        checkpoint = f.read()

    if env_config_file:
        env_config = read_config(env_config_file)
    else:
        env_config = {
            "episode_length": 200,
            "n_agents": 2,
        }

    if config_file:
        rllib_config = read_config(config_file)
    else:
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
            #"seed": tune.grid_search(seeds),
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": "tf",
        }

    ray.init()
    env = rllib_config["env"]
    agent = PPOTrainer(config = rllib_config, env = env)
    agent.restore(checkpoint)
    actions, rewards = compute_actions(agent, env(env_config))
    print(actions)
    plot_results(actions, rewards, var_names)
    ray.shutdown()

    return actions, rewards


def read_config(config_file):
    """Reads the config_file into dictionary

    Parameters
    ----------
    config_file : str
        Path to the configuration file
    Outputs
    ------
    config_dict: dict
    """
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def compute_actions(agent, env):
    done = False
    actions = []
    obs = env.reset()
    rewards = []
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
    return actions, rewards


def plot_results(actions, rewards, var_names, fig_results_fname="human_friction/results/Results.png"):
    actions_time_series = [[s[i] for s in actions] for i in range(len(actions[0]))]
    if not var_names:
        var_names = ["Action_{}".format(i) for i in range(len(actions_time_series))]
    vars = {name: var for name,var in zip(var_names,actions_time_series)}

    fig, axes = plt.subplots(1,len(var_names), figsize=(12,5))
    for ax, var in zip(axes, vars.keys()):
        ax.plot(vars[var])
        ax.set_title(var)
        ax.set_xlabel("Time")

    fig.suptitle('Simulation Results')
    plt.savefig(fig_results_fname)
    plt.close(fig)


if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = False
    main(debug_mode, args.checkpoint_path_stored, args.var_names, args.config_file, args.env_config_file)
