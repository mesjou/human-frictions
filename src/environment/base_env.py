import random
from abc import abstractmethod
from typing import Tuple

import numpy as np
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class BaseEnv(MultiAgentEnv):
    def __init__(self, config):
        """
        :param config: environment configuration that specifies all adjustable parameters of the environment
        """
        n_agents = config["n_agents"]
        assert isinstance(n_agents, int)
        assert n_agents >= 1
        self.n_agents = n_agents

        episode_length = config["episode_length"]
        assert isinstance(episode_length, int)
        assert episode_length >= 1
        self._episode_length = episode_length

        self.timestep = 0

    @property
    def episode_length(self):
        """Length of an episode, in timesteps."""
        return int(self._episode_length)

    @staticmethod
    def seed(seed):
        """Sets the numpy and built-in random number generator seed.

        :param seed: Seed value to use. Must be > 0. Converted to int
                internally if provided value is a float.
        """
        assert isinstance(seed, (int, float))
        seed = int(seed)
        assert seed > 0

        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        self.timestep = 0
        obs = self._generate_observations()
        return obs

    def step(self, action: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        self.timestep += 1

        obs = self._generate_observations()
        rewards = self._generate_rewards()
        done = {"__all__": self.timestep >= self._episode_length}
        info = {}

        return obs, rewards, done, info

    @abstractmethod
    def _generate_observations(self):
        pass

    @abstractmethod
    def _generate_rewards(self):
        pass
