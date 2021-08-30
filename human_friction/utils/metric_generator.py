import collections
from abc import ABC

import matplotlib.pyplot as plt
import pandas as pd
from human_friction.rllib.rllib_env import RllibDiscrete


class MetricGenerator(ABC):
    def __init__(self, env: RllibDiscrete):
        agent_metrics = ["wage", "budget", "consumption", "labor", "reward"]
        environment_metrics = ["interest", "inflation", "unemployment", "production"]

        self.agent_data = collections.defaultdict(dict)
        for metric in agent_metrics:
            for agent in env.wrapped_env.agents.values():
                self.agent_data[metric][agent.agent_id] = []

        self.env_data = {}
        for metric in environment_metrics:
            self.env_data[metric] = []

    def analyze(self, env, rews):
        for metric, agent_ids in self.agent_data.items():
            for agent_id, data in agent_ids.items():
                agent = env.wrapped_env.agents[agent_id]
                value = getattr(agent, metric, None)
                if value is None:
                    value = rews[agent_id]
                data.append(value)

        for metric, data in self.env_data.items():
            value = getattr(env.wrapped_env, metric, None)
            if value is None:
                value = getattr(env.wrapped_env.firm, metric, None)
            data.append(value)

    def plot(self):
        df = pd.DataFrame.from_dict(self.env_data, orient="index").transpose()
        df = df.rolling(5, axis=0).mean()
        df.plot()
        plt.show()

        for metric, data in self.agent_data.items():
            df = pd.DataFrame.from_dict(data, orient="index").transpose()
            df = df.rolling(5, axis=0).mean()
            df.plot()
            plt.title(metric)
            plt.show()
