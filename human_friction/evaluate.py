import argparse

import matplotlib.pyplot as plt
import numpy as np
import ray
import yaml
from human_friction.rllib.models import FCNet
from human_friction.run_configurations.rllib_config import rllib_config
from matplotlib.ticker import MaxNLocator
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from scipy.interpolate import make_interp_spline

ModelCatalog.register_custom_model("my_model", FCNet)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path_stored", default="human_friction/run_configurations/checkpoints/last_checkpoint_path"
)
parser.add_argument("--var_names", default=None)
parser.add_argument("--config_file", default=None)


def main(checkpoint_path_stored, var_names=None, config_file=None):
    """ Computes and plots actions from the trained model

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
    # seeds = list(range(1))

    with open(checkpoint_path_stored, "r") as f:
        checkpoint = f.read()

    if config_file:
        config = read_config(config_file)
    else:
        config = rllib_config

    ray.init()
    env = config["env"]
    agent = PPOTrainer(config=config, env=env)
    agent.restore(checkpoint)
    actions, rewards = compute_actions(agent, env(config["env_config"]))
    plot_results(actions, rewards, var_names, config)
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
    with open(config_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def compute_actions(agent, env):
    done = {"__all__": False}
    actions_path = []
    obs = env.reset()
    rewards = []
    while not done["__all__"]:
        actions = {}
        for agent_id in obs:
            actions[agent_id] = agent.compute_action(obs[agent_id])
        # actions = agent.compute_actions(obs)  #- should but does not work
        obs, reward, done, info = env.step(actions)
        actions_path.append(actions)
        rewards.append(reward)
    return actions_path, rewards


def plot_results(actions, var_names, config, fig_results_fname="human_friction/results/Results.png"):

    n_agents = config.get("n_agents", 1)
    episode_len = config.get("episode_length", 1)
    time = range(1, episode_len + 1)
    agent_ids = list(actions[0].keys())
    nbr_actions = len(actions[0][agent_ids[0]])

    actions_time_series = [[[s[id][i] for s in actions] for id in agent_ids] for i in range(nbr_actions)]

    if not var_names:
        var_names = ["Action_{}".format(i) for i in range(nbr_actions)]
    vars = {name: var for name, var in zip(var_names, actions_time_series)}

    fig, axes = plt.subplots(1, len(var_names), figsize=(12, 5))
    for ax, var in zip(axes, vars.keys()):
        for i in range(n_agents):
            print(vars[var][i])
            ax.plot(*smoothed(time, vars[var][i]))
        ax.set_title(var)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Time")

    fig.suptitle("Simulation Results")
    plt.savefig(fig_results_fname)
    plt.close(fig)


def smoothed(x, y):
    xnew = np.linspace(min(x), max(x), 10 * len(x))
    spl = make_interp_spline(x, y, k=3)
    ynew = spl(xnew)
    return xnew, ynew


if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = False
    main(args.checkpoint_path_stored, args.var_names, args.config_file)
