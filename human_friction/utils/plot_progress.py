import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("path", default=".")


def plot_rewards(path):
    """Plots the min, max and mean rewards every 50th episode
        Inputs:
            path: str, path to the events file
        Outputs:
            plt.fig

    Alternatively run tensorborad --logdir path
    """
    rewards = pd.read_csv(os.path.join(path, "progress.csv"), usecols=[0, 1, 2])
    rewards.iloc[500::50, :].plot(xlabel="Episode", ylabel="Reward")
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    plot_rewards(args.path)
