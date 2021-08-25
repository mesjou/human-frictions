import json
import os

import matplotlib.pyplot as plt


def plot_results(json_path, figure_path):

    mean_rewards = []
    min_rewards = []
    max_rewards = []
    for line in open(json_path, "r"):
        mean_rewards.append(json.loads(line)["episode_reward_mean"])
        min_rewards.append(json.loads(line)["episode_reward_min"])
        max_rewards.append(json.loads(line)["episode_reward_max"])

    plt.plot(mean_rewards)
    plt.plot(min_rewards)
    plt.plot(max_rewards)
    plt.savefig(figure_path)
    plt.close()


if __name__ == "__main__":
    plot_results(
        os.path.join(
            os.getcwd(),
            "/Users/matthias/Desktop/human-frictions/human_friction/rllib/checkpoints/"
            + "PPO_New_Keynes/PPO_RllibDiscrete_57aeb_00000_0_2021-08-13_12-36-39/"  # noqa W503
            + "result.json",  # noqa W503
        ),
        "/Users/matthias/Desktop/human-frictions/human_friction/rllib/checkpoints/"
        + "PPO_New_Keynes/PPO_RllibDiscrete_57aeb_00000_0_2021-08-13_12-36-39/"  # noqa W503
        + "results.png",  # noqa W503
    )
