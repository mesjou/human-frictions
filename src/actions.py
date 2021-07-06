import argparse
import os

import ray
from environment.single_adv_env import LifeCycle
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default = "last_checkpoint")


def main(debug, checkpoint):
    train_n_replicates = 1 if debug else 1
    seeds = list(range(train_n_replicates))

    ray.init()

    stop = {
        "training_iteration": 2 if debug else stop_iters,
    }

    env_config = {
        "episode_length": 10,
        "retirement_date": 7,
    }

    rllib_config = {
        "env": LifeCycle,
        "env_config": env_config,
    }

    agent = PPOTrainer(config = rllib_config, env = rllib_config["env"])
    agent.restore(checkpoint)
    actions, rewards = compute_actions(agent, LifeCycle(env_config))
    ray.shutdown()
    print(actions)
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

if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = True
    main(debug_mode, args.checkpoint)
