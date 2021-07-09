import argparse
import os

import ray
from environment.single_adv_env import LifeCycle
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path_stored", default = "last_checkpoint_path")


def main(debug, checkpoint_path_stored):
    train_n_replicates = 1 if debug else 1
    seeds = list(range(train_n_replicates))
    with open(checkpoint_path_stored, "r") as f:
        checkpoint = f.read()

    ray.init()


    env_config = {
        "episode_length": 10,
        "retirement_date": 7,
    }

    rllib_config = {
        "env": LifeCycle,
        "env_config": env_config,
        "rollout_fragment_length": 50,
        "num_workers":5,
        "train_batch_size": 512,
        "model": {"fcnet_hiddens": [50, 50]},
    }

    agent = PPOTrainer(config = rllib_config, env = rllib_config["env"])
    agent.restore(checkpoint)
    actions, rewards = compute_actions(agent, LifeCycle(env_config))
    plot_results(actions, rewards)
    ray.shutdown()

    return actions, rewards



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

def plot_results(actions, rewards,fig_results_fname="Results/Results.png"):
    con = [s[0] for s in actions]
    inv = [s[1] for s in actions]
    vars = {"C":con,"I": inv,"Rewards":rewards}
    fig, axes = plt.subplots(1,3, figsize=(12,5))

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
    main(debug_mode, args.checkpoint_path_stored)
